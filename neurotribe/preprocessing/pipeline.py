"""From fMRIPrep derivatives to an analysis-ready cortical time series.

    GIFTI surfaces (L/R)
        -> drop non-steady-state volumes
        -> build censor mask
        -> confound regression
        -> detrend / high-pass
        -> standardise
        -> save denoised (time x 20484) + censor mask

The denoised array and its mask are what the alignment engine consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sqlalchemy.orm import Session

from neurotribe.config import Settings
from neurotribe.database.enums import PreprocStatus
from neurotribe.database.models import PreprocessingRun, Scan, Subject
from neurotribe.database.repository import record_audit
from neurotribe.hashing import cache_key
from neurotribe.logging_setup import get_logger
from neurotribe.preprocessing import censoring, confounds as confounds_mod, surfaces

log = get_logger(__name__)


@dataclass
class PreparedTimeSeries:
    """Analysis-ready cortical data for one run."""

    data: np.ndarray                     # (n_timepoints, n_vertices), z-scored
    mask: censoring.CensorMask
    tr: float
    n_dropped: int
    time_sec: np.ndarray                 # acquisition time of each retained frame
    denoise_report: dict = field(default_factory=dict)
    is_approximate: bool = False

    @property
    def n_timepoints(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_vertices(self) -> int:
        return int(self.data.shape[1])


def acquisition_time_axis(n_timepoints: int, tr: float, n_dropped: int = 0) -> np.ndarray:
    """Shared time axis in seconds from stimulus onset.

    Frame ``i`` of the retained series was acquired at ``(i + n_dropped) * TR``,
    because dropping non-steady-state volumes removes time from the *start* of
    the run, not from the stimulus.
    """
    if tr <= 0:
        raise ValueError(f"RepetitionTime must be positive, got {tr}")
    return (np.arange(n_timepoints, dtype=float) + float(n_dropped)) * float(tr)


def prepare(settings: Settings, run: PreprocessingRun, scan: Scan) -> PreparedTimeSeries:
    """Build the analysis-ready time series for a completed preprocessing run."""
    if not run.surface_lh_path or not run.surface_rh_path:
        raise surfaces.SurfaceError(
            "Preprocessing run has no fsaverage5 surfaces; cannot prepare time series."
        )
    if not run.confounds_path:
        raise confounds_mod.ConfoundError(
            "Preprocessing run has no confounds file; refusing to skip denoising."
        )

    tr = scan.repetition_time
    if not tr:
        raise ValueError("RepetitionTime is required to build the shared time axis.")

    left = surfaces.load_gifti_timeseries(Path(run.surface_lh_path))
    right = surfaces.load_gifti_timeseries(Path(run.surface_rh_path))
    order = list(settings.get("surface.hemi_order", ["L", "R"]))
    combined = surfaces.concatenate_hemispheres(left, right, order)
    surfaces.validate_shape(combined, settings, label=f"observed BOLD ({scan.id})")

    frame = confounds_mod.load_confounds(Path(run.confounds_path))
    n_dropped = confounds_mod.detect_nonsteady_state(frame, settings)

    if frame.shape[0] != combined.shape[0]:
        raise confounds_mod.ConfoundError(
            f"Confounds rows ({frame.shape[0]}) do not match BOLD timepoints "
            f"({combined.shape[0]})."
        )

    if n_dropped:
        combined = combined[n_dropped:]
        frame = frame.iloc[n_dropped:].reset_index(drop=True)

    mask = censoring.build_mask(frame, settings, n_nonsteady=0,
                                n_timepoints=combined.shape[0])

    cleaned, _selection, report = confounds_mod.denoise(
        combined, Path(run.confounds_path), settings, float(tr),
        sample_mask=mask.usable, drop=n_dropped,
    )

    time_sec = acquisition_time_axis(cleaned.shape[0], float(tr), n_dropped)

    return PreparedTimeSeries(
        data=cleaned, mask=mask, tr=float(tr), n_dropped=n_dropped,
        time_sec=time_sec, denoise_report=report,
        is_approximate=bool(run.is_approximate),
    )


def prepare_and_cache(session: Session, settings: Settings, run: PreprocessingRun,
                      scan: Scan, subject: Subject) -> PreparedTimeSeries:
    """Prepare the time series, caching the result on disk keyed by inputs."""
    key = cache_key(
        "prepared",
        run=run.id, surfaces=[run.surface_lh_path, run.surface_rh_path],
        confounds=run.confounds_path,
        denoise=settings.get("preprocessing.denoise"),
        motion=settings.get("qc.motion"),
        surface=settings.get("surface"),
    )
    cache_dir = settings.paths.derivatives / "prepared" / subject.external_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path = cache_dir / f"{key.split(':')[1]}.npz"
    mask_path = cache_dir / f"{key.split(':')[1]}_mask.npy"

    if data_path.exists() and mask_path.exists():
        with np.load(data_path, allow_pickle=False) as payload:
            prepared = PreparedTimeSeries(
                data=payload["data"], mask=censoring.CensorMask.load(mask_path),
                tr=float(payload["tr"]), n_dropped=int(payload["n_dropped"]),
                time_sec=payload["time_sec"],
                denoise_report={"cached": True},
                is_approximate=bool(payload["is_approximate"]),
            )
        log.info("Prepared time series loaded from cache",
                 extra={"subject": subject.external_id, "cache_key": key})
        return prepared

    prepared = prepare(settings, run, scan)
    np.savez_compressed(
        data_path, data=prepared.data.astype(np.float32), tr=prepared.tr,
        n_dropped=prepared.n_dropped, time_sec=prepared.time_sec,
        is_approximate=prepared.is_approximate,
    )
    prepared.mask.save(mask_path)

    run.denoise_strategy = str(settings.get("preprocessing.denoise.strategy"))
    run.n_volumes = int(prepared.data.shape[0] + prepared.n_dropped)
    run.n_usable_frames = prepared.mask.n_usable
    run.usable_frame_fraction = prepared.mask.fraction
    run.mean_fd = prepared.mask.mean_fd
    run.n_nonsteady_state = prepared.n_dropped
    run.denoised_path = str(data_path)
    run.censor_mask_path = str(mask_path)
    if run.status == PreprocStatus.RUNNING.value:
        run.status = PreprocStatus.SUCCEEDED.value

    record_audit(session, "preprocessing.prepared", entity_type="preprocessing_run",
                 entity_id=run.id, summary=subject.external_id,
                 payload={"usable_fraction": prepared.mask.fraction,
                          "n_timepoints": prepared.n_timepoints,
                          "denoise": prepared.denoise_report})
    log.info("Prepared time series", extra={
        "subject": subject.external_id, "n_timepoints": prepared.n_timepoints,
        "usable_fraction": round(prepared.mask.fraction, 3),
    })
    return prepared


def load_prepared(run: PreprocessingRun) -> PreparedTimeSeries:
    """Reload a previously cached prepared series."""
    if not run.denoised_path or not run.censor_mask_path:
        raise FileNotFoundError("Preprocessing run has no cached prepared time series.")
    with np.load(run.denoised_path, allow_pickle=False) as payload:
        return PreparedTimeSeries(
            data=payload["data"], mask=censoring.CensorMask.load(Path(run.censor_mask_path)),
            tr=float(payload["tr"]), n_dropped=int(payload["n_dropped"]),
            time_sec=payload["time_sec"], denoise_report={"cached": True},
            is_approximate=bool(payload["is_approximate"]),
        )
