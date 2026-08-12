"""TRIBE inference: run once per stimulus, cache forever.

All HBN participants watched the same clip, so the expected cortical response is
a property of the *stimulus*, not of the subject. Running TRIBE per subject
would waste enormous compute for identical output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sqlalchemy import select
from sqlalchemy.orm import Session

from neurotribe.config import Settings
from neurotribe.database.enums import BlockerKind, BlockerSeverity, TribeBackend
from neurotribe.database.models import Stimulus, TribeRun
from neurotribe.database.repository import clear_blocker, raise_blocker, record_audit
from neurotribe.hashing import cache_key, hash_file
from neurotribe.logging_setup import get_logger
from neurotribe.tribe import geometry, model as model_mod

log = get_logger(__name__)


class InferenceError(RuntimeError):
    pass


@dataclass
class TribePrediction:
    """A validated TRIBE prediction with its own time base."""

    predictions: np.ndarray          # (n_timepoints, n_vertices)
    times: np.ndarray                # segment start times in seconds
    ends: np.ndarray                 # segment end times in seconds
    hemi_order: list[str]
    backend: TribeBackend
    manifest: dict
    geometry_report: dict

    @property
    def n_timepoints(self) -> int:
        return int(self.predictions.shape[0])

    @property
    def n_vertices(self) -> int:
        return int(self.predictions.shape[1])

    @property
    def is_mock(self) -> bool:
        return self.backend is TribeBackend.MOCK

    @property
    def support(self) -> tuple[float, float]:
        """Time range over which predictions exist. Never extrapolate beyond it."""
        return float(self.times[0]), float(self.ends[-1])


def _paths_for(settings: Settings, movie: str) -> dict[str, Path]:
    base = settings.paths.tribe / movie
    base.mkdir(parents=True, exist_ok=True)
    return {
        "base": base,
        "predictions": base / "predictions.npy",
        "segments": base / "segments.parquet",
        "events": base / "events.parquet",
        "model_manifest": base / "model_manifest.json",
        "stimulus_manifest": base / "stimulus_manifest.json",
        "geometry": base / "geometry_report.json",
    }


def build_cache_key(settings: Settings, stimulus: Stimulus, version: model_mod.TribeVersion,
                    backend: TribeBackend) -> str:
    """Cache identity = stimulus content + model identity + relevant config."""
    return cache_key(
        "tribe",
        stimulus_sha256=stimulus.sha256,
        model_id=version.model_id,
        model_revision=version.model_revision,
        tribe_commit=version.commit,
        tribe_version=version.package_version,
        backend=backend.value,
        surface=settings.get("surface"),
        tribe_config={k: settings.get(f"tribe.{k}")
                      for k in ("applies_own_hrf_offset", "additional_hrf_shift_sec",
                                "batch_size")},
    )


def _write_parquet(frame: pd.DataFrame, path: Path) -> Path:
    """Write parquet, falling back to CSV when no parquet engine is installed."""
    try:
        frame.to_parquet(path, index=False)
        return path
    except (ImportError, ValueError) as exc:
        fallback = path.with_suffix(".csv")
        frame.to_csv(fallback, index=False)
        log.info("Parquet engine unavailable; wrote CSV instead",
                 extra={"path": str(fallback), "error": str(exc)})
        return fallback


def _read_table(path: Path) -> pd.DataFrame:
    candidate = Path(path)
    if candidate.exists():
        if candidate.suffix == ".parquet":
            return pd.read_parquet(candidate)
        return pd.read_csv(candidate)
    fallback = candidate.with_suffix(".csv")
    if fallback.exists():
        return pd.read_csv(fallback)
    raise FileNotFoundError(f"No table found at {path} or {fallback}")


def load_cached(session: Session, settings: Settings, movie: str) -> TribePrediction | None:
    """Load a completed TRIBE run for this movie, if one exists and is valid."""
    run = session.execute(
        select(TribeRun).where(TribeRun.movie == movie, TribeRun.status == "DONE")
        .order_by(TribeRun.created_at.desc())
    ).scalars().first()
    if run is None or not run.predictions_path:
        return None

    predictions_path = Path(run.predictions_path)
    if not predictions_path.exists():
        log.warning("Cached TRIBE prediction is missing from disk",
                    extra={"path": str(predictions_path)})
        return None

    predictions = np.load(predictions_path)
    segments = _read_table(Path(run.segments_path)) if run.segments_path else None
    if segments is None:
        return None

    manifest = {}
    if run.manifest_path and Path(run.manifest_path).exists():
        manifest = json.loads(Path(run.manifest_path).read_text(encoding="utf-8"))

    return TribePrediction(
        predictions=predictions,
        times=np.asarray(segments["start"], dtype=float),
        ends=np.asarray(segments["end"], dtype=float),
        hemi_order=list(run.hemi_order or ["L", "R"]),
        backend=TribeBackend(run.backend),
        manifest=manifest,
        geometry_report=run.geometry_report or {},
    )


def run(session: Session, settings: Settings, stimulus: Stimulus,
        *, force: bool = False) -> TribePrediction:
    """Run TRIBE on a stimulus (or return the cached result)."""
    if not stimulus.path or not Path(stimulus.path).exists():
        raise InferenceError(f"Stimulus file is missing: {stimulus.path}")

    loaded = model_mod.load(settings)
    key = build_cache_key(settings, stimulus, loaded.version, loaded.backend)

    existing = session.execute(
        select(TribeRun).where(TribeRun.cache_key == key)
    ).scalar_one_or_none()

    if existing is not None and existing.status == "DONE" and not force:
        cached = load_cached(session, settings, stimulus.key)
        if cached is not None:
            log.info("TRIBE prediction served from cache",
                     extra={"movie": stimulus.key, "cache_key": key})
            record_audit(session, "tribe.cache_hit", entity_type="tribe_run",
                         entity_id=existing.id, summary=stimulus.key)
            return cached

    paths = _paths_for(settings, stimulus.key)
    tribe_run = existing or TribeRun(cache_key=key, movie=stimulus.key)
    tribe_run.stimulus_id = stimulus.id
    tribe_run.backend = loaded.backend.value
    tribe_run.model_id = loaded.version.model_id
    tribe_run.model_revision = loaded.version.model_revision
    tribe_run.tribe_commit = loaded.version.commit
    tribe_run.tribe_version = loaded.version.package_version
    tribe_run.device = loaded.device.device
    tribe_run.stimulus_sha256 = stimulus.sha256
    tribe_run.status = "RUNNING"
    tribe_run.applies_own_hrf_offset = bool(settings.get("tribe.applies_own_hrf_offset", True))
    if existing is None:
        session.add(tribe_run)
    session.flush()

    try:
        handle = loaded.handle
        if handle is None:
            raise InferenceError("No TRIBE model handle available")

        log.info("Running TRIBE inference", extra={
            "movie": stimulus.key, "backend": loaded.backend.value,
            "device": loaded.device.device, "duration_sec": stimulus.duration_sec,
        })
        events = handle.get_events_dataframe(video_path=str(stimulus.path))  # type: ignore[attr-defined]
        predictions, segments = handle.predict(events=events)  # type: ignore[attr-defined]

        predictions = np.asarray(predictions, dtype=np.float32)
        if not isinstance(segments, pd.DataFrame):
            segments = pd.DataFrame(segments)
        segments = _normalize_segments(segments, predictions.shape[0], settings)

    except Exception as exc:  # noqa: BLE001
        tribe_run.status = "FAILED"
        tribe_run.error_message = str(exc)[:2000]
        record_audit(session, "tribe.failed", entity_type="tribe_run", entity_id=tribe_run.id,
                     summary=str(exc)[:200])
        raise InferenceError(f"TRIBE inference failed: {exc}") from exc

    report = geometry.validate(predictions, segments, settings, backend=loaded.backend.value)
    tribe_run.geometry_report = report.to_dict()
    tribe_run.geometry_validated = report.ok
    if not report.ok:
        tribe_run.status = "FAILED"
        tribe_run.error_message = "; ".join(report.errors)[:2000]
        raise geometry.GeometryError(
            "TRIBE output failed geometry validation: " + "; ".join(report.errors)
        )

    np.save(paths["predictions"], predictions)
    segments_path = _write_parquet(segments, paths["segments"])
    events_path = _write_parquet(pd.DataFrame(events), paths["events"])

    stimulus_manifest = {
        "movie": stimulus.key, "label": stimulus.label,
        "path": stimulus.path, "sha256": stimulus.sha256,
        "duration_sec": stimulus.duration_sec, "fps": stimulus.fps,
        "width": stimulus.width, "height": stimulus.height,
        "source_interval_start": stimulus.source_interval_start,
        "source_interval_end": stimulus.source_interval_end,
        "provenance_note": stimulus.provenance_note,
    }
    model_manifest = {
        **loaded.manifest(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cache_key": key,
        "analysis_config_hash": settings.analysis_config_hash,
        "profile": settings.profile,
        "n_timepoints": int(predictions.shape[0]),
        "n_vertices": int(predictions.shape[1]),
        "hrf_note": (
            "TRIBE v2 states its prediction timing already incorporates a 5 s "
            "hemodynamic-lag offset. No additional shift is applied here; the "
            "timestamps returned by TRIBE are used verbatim."
        ),
        "additional_hrf_shift_sec": float(settings.get("tribe.additional_hrf_shift_sec", 0.0)),
    }
    paths["model_manifest"].write_text(json.dumps(model_manifest, indent=2), encoding="utf-8")
    paths["stimulus_manifest"].write_text(json.dumps(stimulus_manifest, indent=2), encoding="utf-8")
    paths["geometry"].write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    tribe_run.predictions_path = str(paths["predictions"])
    tribe_run.segments_path = str(segments_path)
    tribe_run.events_path = str(events_path)
    tribe_run.manifest_path = str(paths["model_manifest"])
    tribe_run.n_timepoints = int(predictions.shape[0])
    tribe_run.n_vertices = int(predictions.shape[1])
    tribe_run.surface_space = report.surface_space
    tribe_run.hemi_order = report.hemi_order
    tribe_run.time_start_sec = float(segments["start"].iloc[0])
    tribe_run.time_end_sec = float(segments["end"].iloc[-1])
    tribe_run.status = "DONE"
    tribe_run.error_message = None

    clear_blocker(session, BlockerKind.TRIBE_MODEL)
    record_audit(session, "tribe.completed", entity_type="tribe_run", entity_id=tribe_run.id,
                 summary=f"{stimulus.key}: {predictions.shape[0]} x {predictions.shape[1]}",
                 payload={"backend": loaded.backend.value, "cache_key": key})

    return TribePrediction(
        predictions=predictions,
        times=np.asarray(segments["start"], dtype=float),
        ends=np.asarray(segments["end"], dtype=float),
        hemi_order=report.hemi_order,
        backend=loaded.backend,
        manifest=model_manifest,
        geometry_report=report.to_dict(),
    )


def _normalize_segments(segments: pd.DataFrame, n_timepoints: int,
                        settings: Settings) -> pd.DataFrame:
    """Coerce TRIBE's segment table into ``start``/``end`` seconds.

    Different releases name these columns differently. We never invent
    timestamps: if none can be found, we fail loudly.
    """
    frame = segments.copy()
    columns = {c.lower(): c for c in frame.columns}

    start_col = next((columns[c] for c in ("start", "onset", "start_time", "t_start",
                                           "start_sec", "time") if c in columns), None)
    end_col = next((columns[c] for c in ("end", "offset", "end_time", "t_end",
                                         "end_sec", "stop") if c in columns), None)

    if start_col is None:
        raise InferenceError(
            "TRIBE segments contain no recognisable start-time column "
            f"(columns: {list(frame.columns)}). Refusing to synthesise timestamps."
        )

    out = pd.DataFrame({"start": pd.to_numeric(frame[start_col], errors="coerce")})
    if end_col is not None:
        out["end"] = pd.to_numeric(frame[end_col], errors="coerce")
    else:
        # Derive end from the sampling interval - the only safe inference.
        deltas = np.diff(out["start"].to_numpy(dtype=float))
        step = float(np.median(deltas)) if deltas.size else 1.0
        out["end"] = out["start"] + step

    if out.shape[0] != n_timepoints:
        raise InferenceError(
            f"TRIBE returned {n_timepoints} predictions but {out.shape[0]} segments."
        )
    if out["start"].isna().any() or out["end"].isna().any():
        raise InferenceError("TRIBE segment timestamps contain non-numeric values.")

    shift = float(settings.get("tribe.additional_hrf_shift_sec", 0.0))
    if shift:
        # Only applied if an operator deliberately configured it; the default is 0
        # because TRIBE already accounts for hemodynamic lag.
        log.warning("Applying a configured ADDITIONAL hemodynamic shift to TRIBE timing",
                    extra={"shift_sec": shift})
        out["start"] += shift
        out["end"] += shift

    for extra in ("index", "scene"):
        if extra in frame.columns:
            out[extra] = frame[extra].to_numpy()
    return out.reset_index(drop=True)


def ensure_available(session: Session, settings: Settings) -> dict:
    """Report TRIBE readiness and raise a blocker when it is unusable."""
    available, version, error = model_mod.probe_tribe()
    backend_policy = str(settings.get("tribe.backend", "auto"))

    if available:
        clear_blocker(session, BlockerKind.TRIBE_MODEL)
        return {"available": True, "backend": "real", **version.to_dict()}

    if backend_policy == "real" or settings.is_production:
        raise_blocker(
            session, BlockerKind.TRIBE_MODEL, "TRIBE v2 not installed",
            f"The real TRIBE v2 backend is required by the '{settings.profile}' profile "
            f"but is not importable: {error}",
            severity=BlockerSeverity.ACTIONABLE,
            required_action=model_mod.install_hint(settings)["command"],
            reference_url=str(settings.get("tribe.repo_url")),
            blocks_stages=["tribe_inference", "subject_analysis", "group_analysis"],
        )
        return {"available": False, "backend": "none", "error": error}

    return {
        "available": False, "backend": "mock", "error": error,
        "note": ("Development profile: the mock backend keeps the pipeline testable. "
                 "No mock-derived output is reportable."),
        "install": model_mod.install_hint(settings),
    }


def smoke_test(session: Session, settings: Settings) -> dict:
    """Prove the TRIBE inference path works end to end on a synthetic clip.

    The synthetic video deliberately matches no catalog duration, so it can never
    be mistaken for an HBN stimulus.
    """
    from neurotribe.tribe.mock import synthetic_video

    duration = float(settings.get("autopilot.smoke_test.synthetic_video_sec", 20))
    target = settings.paths.cache / "smoke" / "synthetic.mp4"
    video = synthetic_video(target, duration_sec=duration)
    if video is None:
        return {"ok": False, "reason": "ffmpeg unavailable; cannot render a synthetic clip."}

    loaded = model_mod.load(settings)
    if loaded.handle is None:
        return {"ok": False, "reason": "No TRIBE handle available."}

    try:
        events = loaded.handle.get_events_dataframe(video_path=str(video))  # type: ignore[attr-defined]
        predictions, segments = loaded.handle.predict(events=events)  # type: ignore[attr-defined]
        predictions = np.asarray(predictions, dtype=np.float32)
        segments = _normalize_segments(pd.DataFrame(segments), predictions.shape[0], settings)
        report = geometry.validate(predictions, segments, settings, backend=loaded.backend.value)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(exc).__name__}: {exc}",
                "backend": loaded.backend.value}

    digest = hash_file(video)
    result = {
        "ok": report.ok,
        "backend": loaded.backend.value,
        "device": loaded.device.device,
        "n_timepoints": report.n_timepoints,
        "n_vertices": report.n_vertices,
        "hemi_order": report.hemi_order,
        "hemi_order_source": report.hemi_order_source,
        "medial_wall_vertices": report.medial_wall_vertices,
        "geometry_errors": report.errors,
        "geometry_warnings": report.warnings,
        "synthetic_video_sha256": digest.sha256,
        "synthetic_video_sec": duration,
    }
    record_audit(session, "tribe.smoke_test", entity_type="tribe",
                 summary=f"ok={report.ok} backend={loaded.backend.value}", payload=result)
    return result
