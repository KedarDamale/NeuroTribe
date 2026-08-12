"""Frame censoring: which fMRI volumes may enter the TRIBE comparison.

For every frame we record ``usable = true/false`` plus the reason. Censored
frames are excluded from every correlation, residual and ROI aggregate - never
silently interpolated over.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from neurotribe.config import Settings
from neurotribe.logging_setup import get_logger

log = get_logger(__name__)

REASON_OK = "ok"
REASON_NONSTEADY = "nonsteady_state"
REASON_FD = "high_fd"
REASON_DVARS = "dvars_outlier"
REASON_PAD = "censor_pad"
REASON_MISSING = "missing_data"


@dataclass
class CensorMask:
    usable: np.ndarray                      # bool, shape (n_timepoints,)
    reasons: list[str] = field(default_factory=list)
    n_total: int = 0
    n_usable: int = 0
    n_nonsteady: int = 0
    n_high_fd: int = 0
    n_dvars: int = 0
    n_padded: int = 0
    n_missing: int = 0
    mean_fd: float | None = None
    max_fd: float | None = None
    fd_threshold: float | None = None
    dvars_threshold: float | None = None

    @property
    def fraction(self) -> float:
        return float(self.n_usable / self.n_total) if self.n_total else 0.0

    def to_dict(self) -> dict:
        return {
            "n_total": self.n_total, "n_usable": self.n_usable,
            "usable_fraction": round(self.fraction, 4),
            "n_nonsteady": self.n_nonsteady, "n_high_fd": self.n_high_fd,
            "n_dvars": self.n_dvars, "n_padded": self.n_padded,
            "n_missing": self.n_missing, "mean_fd": self.mean_fd,
            "max_fd": self.max_fd, "fd_threshold": self.fd_threshold,
            "dvars_threshold": self.dvars_threshold,
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.usable)
        path.with_suffix(".json").write_text(
            json.dumps({**self.to_dict(), "reasons": self.reasons}, indent=2),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "CensorMask":
        usable = np.load(path)
        meta_path = Path(path).with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        mask = cls(usable=usable.astype(bool))
        mask.reasons = meta.pop("reasons", [])
        for key, value in meta.items():
            if hasattr(mask, key):
                setattr(mask, key, value)
        mask.n_total = int(usable.size)
        mask.n_usable = int(usable.sum())
        return mask


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray | None:
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    return values


def _robust_dvars_threshold(dvars: np.ndarray, n_sd: float) -> float:
    """Median + n_sd * (1.4826 * MAD): robust to the outliers we are detecting."""
    finite = dvars[np.isfinite(dvars)]
    if finite.size < 3:
        return float("inf")
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    scale = 1.4826 * mad
    if scale <= 0:
        scale = float(finite.std()) or 1.0
    return median + n_sd * scale


def build_mask(confounds: pd.DataFrame, settings: Settings,
               n_nonsteady: int = 0, n_timepoints: int | None = None) -> CensorMask:
    """Compute the per-frame usability mask from fMRIPrep confounds."""
    total = int(n_timepoints if n_timepoints is not None else confounds.shape[0])
    usable = np.ones(total, dtype=bool)
    reasons = [REASON_OK] * total

    fd_threshold = float(settings.get("qc.motion.fd_threshold_mm", 0.5))
    dvars_sd = float(settings.get("qc.motion.dvars_threshold_sd", 3.0))
    pad_before = int(settings.get("qc.motion.censor_pad_before", 0))
    pad_after = int(settings.get("qc.motion.censor_pad_after", 1))

    mask = CensorMask(usable=usable, n_total=total, fd_threshold=fd_threshold)

    # Non-steady-state volumes at the start of the run.
    for index in range(min(n_nonsteady, total)):
        usable[index] = False
        reasons[index] = REASON_NONSTEADY
    mask.n_nonsteady = min(n_nonsteady, total)

    fd = _numeric(confounds, "framewise_displacement")
    if fd is not None:
        fd = fd[:total]
        finite = fd[np.isfinite(fd)]
        if finite.size:
            mask.mean_fd = float(finite.mean())
            mask.max_fd = float(finite.max())
        high = np.zeros(total, dtype=bool)
        high[: fd.size] = np.isfinite(fd) & (fd > fd_threshold)
        for index in np.flatnonzero(high):
            if usable[index]:
                usable[index] = False
                reasons[index] = REASON_FD
                mask.n_high_fd += 1
        # Missing FD (first volume) is not itself a censor reason.
        missing = np.zeros(total, dtype=bool)
        missing[: fd.size] = ~np.isfinite(fd)
        missing[:mask.n_nonsteady] = False
        if missing.any() and mask.n_nonsteady == 0:
            # Only the very first frame legitimately lacks FD.
            for index in np.flatnonzero(missing):
                if index > 0 and usable[index]:
                    usable[index] = False
                    reasons[index] = REASON_MISSING
                    mask.n_missing += 1

    dvars = _numeric(confounds, "std_dvars")
    if dvars is None:
        dvars = _numeric(confounds, "dvars")
    if dvars is not None and dvars_sd > 0:
        dvars = dvars[:total]
        threshold = _robust_dvars_threshold(dvars, dvars_sd)
        mask.dvars_threshold = None if np.isinf(threshold) else float(threshold)
        outliers = np.zeros(total, dtype=bool)
        outliers[: dvars.size] = np.isfinite(dvars) & (dvars > threshold)
        for index in np.flatnonzero(outliers):
            if usable[index]:
                usable[index] = False
                reasons[index] = REASON_DVARS
                mask.n_dvars += 1

    # Pad around every censored frame to remove motion spillover.
    if pad_before or pad_after:
        censored = np.flatnonzero(~usable)
        for index in censored:
            if reasons[index] == REASON_NONSTEADY:
                continue
            low = max(0, index - pad_before)
            high_bound = min(total, index + pad_after + 1)
            for neighbour in range(low, high_bound):
                if usable[neighbour]:
                    usable[neighbour] = False
                    reasons[neighbour] = REASON_PAD
                    mask.n_padded += 1

    mask.usable = usable
    mask.reasons = reasons
    mask.n_usable = int(usable.sum())
    log.debug("Censor mask built", extra=mask.to_dict())
    return mask


def apply_mask(signals: np.ndarray, mask: CensorMask) -> np.ndarray:
    """Return only usable timepoints."""
    if signals.shape[0] != mask.usable.size:
        raise ValueError(
            f"Signal has {signals.shape[0]} timepoints, mask has {mask.usable.size}"
        )
    return signals[mask.usable]


def intersect(masks: list[CensorMask]) -> np.ndarray:
    """Frames usable in every supplied mask."""
    if not masks:
        raise ValueError("No masks supplied")
    lengths = {m.usable.size for m in masks}
    if len(lengths) != 1:
        raise ValueError(f"Masks have differing lengths: {sorted(lengths)}")
    combined = masks[0].usable.copy()
    for mask in masks[1:]:
        combined &= mask.usable
    return combined


def summarize_windows(mask: CensorMask, tr: float) -> list[dict]:
    """Contiguous censored blocks, for the subject timeline visualisation."""
    blocks: list[dict] = []
    index = 0
    total = mask.usable.size
    while index < total:
        if mask.usable[index]:
            index += 1
            continue
        start = index
        reason = mask.reasons[index] if index < len(mask.reasons) else REASON_FD
        while index < total and not mask.usable[index]:
            index += 1
        blocks.append({
            "start_frame": int(start), "end_frame": int(index - 1),
            "start_sec": round(start * tr, 3), "end_sec": round((index - 1) * tr + tr, 3),
            "n_frames": int(index - start), "reason": reason,
        })
    return blocks
