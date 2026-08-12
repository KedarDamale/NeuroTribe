"""Stimulus intake and validation.

The exact movie clip used during HBN scanning is copyrighted. This module
**never downloads video from anywhere** - no torrents, no piracy sites, no
scraping of arbitrary uploads. It watches ``data/stimuli/incoming/`` and
validates whatever the operator legally supplies, per HBN's guidance to contact
the Child Mind Institute for exact clip information.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from neurotribe.config import Settings
from neurotribe.database.enums import AssetKind, AssetStatus, MovieKey
from neurotribe.database.models import DataAsset, Stimulus
from neurotribe.database.repository import record_audit
from neurotribe.hashing import hash_file
from neurotribe.logging_setup import get_logger

log = get_logger(__name__)

FFPROBE = shutil.which("ffprobe") or "ffprobe"
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"


class FFmpegUnavailable(RuntimeError):
    """Raised when ffprobe/ffmpeg are not installed."""


@dataclass
class MediaInfo:
    duration_sec: float | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    has_audio: bool = False
    container: str | None = None
    video_codec: str | None = None
    audio_codec: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "duration_sec": self.duration_sec, "fps": self.fps, "width": self.width,
            "height": self.height, "has_audio": self.has_audio,
            "container": self.container, "video_codec": self.video_codec,
            "audio_codec": self.audio_codec,
        }


def _parse_fraction(value: str | None) -> float | None:
    if not value or "/" not in str(value):
        try:
            return float(value) if value else None
        except (TypeError, ValueError):
            return None
    numerator, _, denominator = str(value).partition("/")
    try:
        den = float(denominator)
        return float(numerator) / den if den else None
    except ValueError:
        return None


def probe(path: Path, timeout: int = 120) -> MediaInfo:
    """Read container/stream properties with ffprobe."""
    if shutil.which(FFPROBE) is None and not Path(FFPROBE).exists():
        raise FFmpegUnavailable("ffprobe not found on PATH")

    command = [
        FFPROBE, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed ({result.returncode}): {result.stderr.strip()[:400]}")

    payload = json.loads(result.stdout or "{}")
    info = MediaInfo(raw=payload)
    fmt = payload.get("format", {})
    try:
        info.duration_sec = float(fmt.get("duration"))
    except (TypeError, ValueError):
        info.duration_sec = None
    info.container = (fmt.get("format_name") or "").split(",")[0] or None

    for stream in payload.get("streams", []):
        if stream.get("codec_type") == "video" and info.width is None:
            info.width = stream.get("width")
            info.height = stream.get("height")
            info.video_codec = stream.get("codec_name")
            info.fps = _parse_fraction(stream.get("avg_frame_rate")) or _parse_fraction(
                stream.get("r_frame_rate")
            )
            if info.duration_sec is None:
                try:
                    info.duration_sec = float(stream.get("duration"))
                except (TypeError, ValueError):
                    pass
        elif stream.get("codec_type") == "audio":
            info.has_audio = True
            info.audio_codec = info.audio_codec or stream.get("codec_name")

    return info


def extract_frame(video: Path, out_path: Path, at_sec: float) -> Path | None:
    """Extract a single frame for visual verification of the supplied clip."""
    if shutil.which(FFMPEG) is None and not Path(FFMPEG).exists():
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        FFMPEG, "-y", "-loglevel", "error", "-ss", f"{max(0.0, at_sec):.3f}",
        "-i", str(video), "-frames:v", "1", "-q:v", "3", str(out_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180, check=False)
    if result.returncode != 0 or not out_path.exists():
        log.warning("Frame extraction failed",
                    extra={"video": str(video), "at_sec": at_sec, "stderr": result.stderr[:300]})
        return None
    return out_path


@dataclass
class ValidationResult:
    ok: bool
    movie: MovieKey
    reasons: list[str] = field(default_factory=list)
    info: MediaInfo = field(default_factory=MediaInfo)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "movie": self.movie.value, "reasons": self.reasons,
                "media": self.info.to_dict()}


def match_catalog(info: MediaInfo, settings: Settings) -> tuple[MovieKey, list[str]]:
    """Identify which documented HBN interval a supplied clip corresponds to."""
    catalog: dict[str, dict] = settings.get("stimulus.catalog", {})
    notes: list[str] = []
    if info.duration_sec is None:
        return MovieKey.UNKNOWN, ["Clip duration could not be determined."]

    best: tuple[str, float] | None = None
    for key, spec in catalog.items():
        expected = float(spec.get("expected_duration_sec") or 0.0)
        tolerance = float(spec.get("duration_tolerance_sec") or 3.0)
        delta = abs(info.duration_sec - expected)
        notes.append(
            f"{key}: expected {expected:.1f}s, supplied {info.duration_sec:.1f}s "
            f"(delta {delta:.1f}s, tolerance {tolerance:.1f}s)"
        )
        if delta <= tolerance and (best is None or delta < best[1]):
            best = (key, delta)

    if best is None:
        return MovieKey.UNKNOWN, notes + [
            "Supplied clip duration matches no documented HBN movie interval. "
            "Verify you trimmed the exact interval documented in the HBN MRI protocol."
        ]
    return MovieKey(best[0]), notes


def validate(path: Path, settings: Settings) -> ValidationResult:
    """Validate a candidate stimulus file against the configured constraints."""
    reasons: list[str] = []
    try:
        info = probe(path)
    except FFmpegUnavailable as exc:
        return ValidationResult(False, MovieKey.UNKNOWN, [str(exc)], MediaInfo())
    except Exception as exc:  # noqa: BLE001
        return ValidationResult(False, MovieKey.UNKNOWN, [f"Probe failed: {exc}"], MediaInfo())

    rules = settings.get("stimulus.validation", {})
    if info.width is None or info.height is None:
        reasons.append("No video stream detected.")
    else:
        if info.width < int(rules.get("min_width", 0)):
            reasons.append(f"Width {info.width} below minimum {rules.get('min_width')}.")
        if info.height < int(rules.get("min_height", 0)):
            reasons.append(f"Height {info.height} below minimum {rules.get('min_height')}.")
    if info.fps is not None:
        if info.fps < float(rules.get("min_fps", 0)):
            reasons.append(f"Frame rate {info.fps:.2f} below minimum {rules.get('min_fps')}.")
        if info.fps > float(rules.get("max_fps", 1e6)):
            reasons.append(f"Frame rate {info.fps:.2f} above maximum {rules.get('max_fps')}.")
    if not info.has_audio:
        # TRIBE v2 is a vision+audition+language model; audio is required.
        reasons.append("No audio stream: TRIBE v2 requires audio for its auditory pathway.")

    movie, notes = match_catalog(info, settings)
    if movie is MovieKey.UNKNOWN:
        reasons.extend(notes[-1:])

    return ValidationResult(ok=not reasons and movie is not MovieKey.UNKNOWN,
                            movie=movie, reasons=reasons + notes, info=info)


def register_stimulus(session: Session, settings: Settings, asset: DataAsset) -> Stimulus | None:
    """Validate a discovered video and register it if it matches the catalog."""
    path = Path(asset.absolute_path)
    if not path.exists():
        asset.status = AssetStatus.MISSING.value
        return None

    result = validate(path, settings)
    asset.validation_report = {**(asset.validation_report or {}), "stimulus": result.to_dict()}

    if not result.ok:
        # Not an error: an unrelated video may simply be sitting in the tree.
        asset.status = AssetStatus.QUARANTINED.value
        log.info("Video did not validate as an HBN stimulus",
                 extra={"path": str(path), "reasons": result.reasons[:3]})
        return None

    digest = hash_file(path, full_max_bytes=int(settings.get("discovery.full_hash_max_bytes", 2**28)))
    spec = settings.get(f"stimulus.catalog.{result.movie.value}", {})

    stimulus = session.execute(
        select(Stimulus).where(Stimulus.key == result.movie.value)
    ).scalar_one_or_none()
    if stimulus is None:
        stimulus = Stimulus(key=result.movie.value)
        session.add(stimulus)

    stimulus.label = spec.get("label", result.movie.value)
    stimulus.path = str(path)
    stimulus.sha256 = digest.sha256
    stimulus.size_bytes = digest.size_bytes
    stimulus.duration_sec = result.info.duration_sec
    stimulus.fps = result.info.fps
    stimulus.width = result.info.width
    stimulus.height = result.info.height
    stimulus.has_audio = result.info.has_audio
    stimulus.container = result.info.container
    stimulus.expected_duration_sec = spec.get("expected_duration_sec")
    stimulus.source_interval_start = spec.get("source_interval_start")
    stimulus.source_interval_end = spec.get("source_interval_end")
    stimulus.validated = True
    stimulus.validation_notes = result.to_dict()
    stimulus.provenance_note = (
        "Supplied by operator into data/stimuli/incoming. Never auto-downloaded."
    )

    # Visual verification frames.
    frames_dir = settings.paths.stimuli / "frames" / result.movie.value
    first = extract_frame(path, frames_dir / "first.jpg", 0.5)
    duration = result.info.duration_sec or 1.0
    last = extract_frame(path, frames_dir / "last.jpg", max(0.0, duration - 0.5))
    stimulus.first_frame_path = str(first) if first else None
    stimulus.last_frame_path = str(last) if last else None

    asset.status = AssetStatus.VALIDATED.value
    session.flush()
    record_audit(session, "stimulus.registered", entity_type="stimulus", entity_id=stimulus.id,
                 summary=f"{stimulus.label} ({result.info.duration_sec:.1f}s)",
                 payload={"sha256": digest.sha256})
    log.info("Stimulus registered", extra={"movie": result.movie.value,
                                           "duration_sec": result.info.duration_sec})
    return stimulus


def scan_incoming(session: Session, settings: Settings) -> dict:
    """Validate every discovered video asset; register any that match."""
    settings.paths.stimuli_incoming.mkdir(parents=True, exist_ok=True)
    assets = list(session.execute(
        select(DataAsset).where(DataAsset.kind == AssetKind.STIMULUS_VIDEO.value)
    ).scalars())

    registered: list[str] = []
    for asset in assets:
        if asset.status == AssetStatus.VALIDATED.value:
            continue
        stimulus = register_stimulus(session, settings, asset)
        if stimulus is not None:
            registered.append(stimulus.key)

    available = [
        s.key for s in session.execute(
            select(Stimulus).where(Stimulus.validated.is_(True))
        ).scalars()
    ]
    return {
        "incoming_dir": str(settings.paths.stimuli_incoming),
        "n_candidates": len(assets),
        "registered": registered,
        "available": available,
    }


def select_primary(session: Session, settings: Settings) -> Stimulus | None:
    """Apply the configured primary-movie policy.

    ``auto`` prefers Despicable Me (~10 min) over The Present (~3:21) because a
    longer stimulus yields far more timepoints for the deviation analysis.
    """
    configured = settings.get("stimulus.primary", "auto")
    available = {
        s.key: s for s in session.execute(
            select(Stimulus).where(Stimulus.validated.is_(True))
        ).scalars()
    }
    if not available:
        return None
    if configured != "auto":
        return available.get(configured)
    for key in settings.get("stimulus.preference_order", []):
        if key in available:
            return available[key]
    return next(iter(available.values()))


def write_intake_readme(settings: Settings) -> Path:
    """Explain to the operator exactly what to drop where."""
    catalog = settings.get("stimulus.catalog", {})
    lines = [
        "# Stimulus intake",
        "",
        "Place the **legally obtained** movie clip used during HBN scanning in this",
        "directory. NeuroTRIBE never downloads video and never bypasses access",
        "controls; it only validates what you supply here.",
        "",
        "HBN documents the exact source intervals but asks researchers to contact the",
        "Child Mind Institute for exact-clip information:",
        "https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/MRI_Protocol.html",
        "",
        "## Expected clips",
        "",
    ]
    for key, spec in catalog.items():
        lines += [
            f"### {spec.get('label', key)}  (`{key}`)",
            f"- Source interval: `{spec.get('source_interval_start')}` -> "
            f"`{spec.get('source_interval_end')}`",
            f"- Expected duration: **{spec.get('expected_duration_sec')} s** "
            f"(+/- {spec.get('duration_tolerance_sec')} s)",
            "- Must contain an **audio** stream (TRIBE v2 uses the auditory pathway).",
            "",
        ]
    lines += [
        "A dropped file is auto-detected, probed with ffprobe, matched to the catalog",
        "by duration, hashed, and its first/last frames extracted for visual",
        "verification. Files that match nothing are quarantined, not deleted.",
        "",
    ]
    target = settings.paths.stimuli_incoming / "README.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
    return target
