"""Confound selection and nuisance regression.

fMRIPrep emits a ``*_desc-confounds_timeseries.tsv`` per BOLD run. Column names
vary across versions, so selection is done by *pattern*, and the exact resolved
column list is recorded with every run.

Global signal regression is deliberately **not** enabled by default: HBN itself
notes there is no single consensus approach to motion correction, so the applied
strategy must always be recorded rather than assumed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from neurotribe.config import Settings
from neurotribe.logging_setup import get_logger
from neurotribe.numerics import safe_zscore

log = get_logger(__name__)

MOTION_6 = ("trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z")
# 24-parameter Friston expansion: 6 params + derivatives + both squared.
MOTION_24_SUFFIXES = ("", "_derivative1", "_power2", "_derivative1_power2")

_NONSTEADY_RE = re.compile(r"^non_steady_state_outlier\d+$")
_MOTION_OUTLIER_RE = re.compile(r"^motion_outlier\d+$")
_ACOMPCOR_RE = re.compile(r"^a_comp_cor_\d+$")


class ConfoundError(RuntimeError):
    """Raised when the confounds file cannot support the requested strategy."""


@dataclass
class ConfoundSelection:
    columns: list[str] = field(default_factory=list)
    strategy: str = ""
    missing: list[str] = field(default_factory=list)
    n_acompcor: int = 0
    includes_gsr: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "columns": self.columns, "strategy": self.strategy,
            "missing": self.missing, "n_acompcor": self.n_acompcor,
            "includes_gsr": self.includes_gsr, "notes": self.notes,
        }


def load_confounds(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", na_values=["n/a", "N/A", ""], keep_default_na=True)
    if frame.empty:
        raise ConfoundError(f"Confounds file is empty: {path}")
    return frame


def select_confounds(frame: pd.DataFrame, settings: Settings) -> ConfoundSelection:
    """Resolve the configured denoising strategy against actual columns."""
    strategy = str(settings.get("preprocessing.denoise.strategy", "motion24_acompcor"))
    gsr = bool(settings.get("preprocessing.denoise.global_signal_regression", False))
    n_acompcor = int(settings.get("preprocessing.denoise.n_acompcor_components", 5))

    available = set(frame.columns)
    selection = ConfoundSelection(strategy=strategy, includes_gsr=gsr)
    wanted: list[str] = []

    if strategy in {"motion6", "motion24", "motion24_acompcor", "custom"}:
        if strategy == "motion6":
            wanted.extend(MOTION_6)
        elif strategy in {"motion24", "motion24_acompcor"}:
            for base in MOTION_6:
                wanted.extend(f"{base}{suffix}" for suffix in MOTION_24_SUFFIXES)
        if strategy == "motion24_acompcor":
            acompcor = sorted(c for c in available if _ACOMPCOR_RE.match(c))
            chosen = acompcor[:n_acompcor]
            selection.n_acompcor = len(chosen)
            if len(chosen) < n_acompcor:
                selection.notes.append(
                    f"Requested {n_acompcor} aCompCor components; {len(chosen)} available."
                )
            wanted.extend(chosen)
    else:
        raise ConfoundError(f"Unknown denoise strategy: {strategy}")

    if gsr:
        wanted.append("global_signal")
        selection.notes.append(
            "Global signal regression is ENABLED by configuration. This is a "
            "non-default choice and is recorded in the provenance manifest."
        )

    # Spike / outlier regressors are always included when present.
    outliers = sorted(c for c in available if _MOTION_OUTLIER_RE.match(c))
    wanted.extend(outliers)
    if outliers:
        selection.notes.append(f"Included {len(outliers)} motion_outlier spike regressors.")

    for column in wanted:
        if column in available:
            if column not in selection.columns:
                selection.columns.append(column)
        else:
            selection.missing.append(column)

    if selection.missing:
        selection.notes.append(
            f"{len(selection.missing)} requested confound column(s) absent from the "
            "fMRIPrep output; the strategy was applied with the available subset."
        )
    if not selection.columns:
        raise ConfoundError(
            "No usable confound columns were found. Refusing to proceed with an "
            "unspecified denoising strategy."
        )
    return selection


def detect_nonsteady_state(frame: pd.DataFrame, settings: Settings) -> int:
    """Count leading non-steady-state volumes flagged by fMRIPrep."""
    if not bool(settings.get("preprocessing.denoise.drop_nonsteady_state", True)):
        return 0
    maximum = int(settings.get("preprocessing.denoise.nonsteady_state_max", 8))
    columns = [c for c in frame.columns if _NONSTEADY_RE.match(c)]
    if not columns:
        return 0
    flagged = frame[columns].to_numpy(dtype=float)
    flagged = np.nan_to_num(flagged, nan=0.0)
    per_volume = flagged.sum(axis=1) > 0
    count = 0
    for is_flagged in per_volume:
        if not is_flagged:
            break
        count += 1
    if count > maximum:
        log.warning("Non-steady-state count clipped",
                    extra={"detected": count, "max": maximum})
        count = maximum
    return int(count)


def build_design_matrix(frame: pd.DataFrame, selection: ConfoundSelection,
                        settings: Settings, tr: float) -> np.ndarray:
    """Assemble the nuisance design matrix (n_timepoints x n_regressors).

    Includes an intercept and, when detrending is enabled, a linear trend plus a
    discrete-cosine high-pass basis.
    """
    matrix = frame[selection.columns].to_numpy(dtype=float)
    # fMRIPrep writes n/a for the first row of derivative columns.
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    n_timepoints = matrix.shape[0]
    extras = [np.ones((n_timepoints, 1))]

    if bool(settings.get("preprocessing.denoise.detrend", True)):
        trend = np.linspace(-1.0, 1.0, n_timepoints).reshape(-1, 1)
        extras.append(trend)

    high_pass = settings.get("preprocessing.denoise.high_pass_hz")
    if high_pass:
        extras.append(_cosine_basis(n_timepoints, tr, float(high_pass)))

    design = np.hstack([matrix, *extras])
    # Drop zero-variance columns which would make the pseudo-inverse unstable.
    keep = np.ones(design.shape[1], dtype=bool)
    keep[:matrix.shape[1]] = design[:, :matrix.shape[1]].std(axis=0) > 1e-12
    return design[:, keep]


def _cosine_basis(n_timepoints: int, tr: float, high_pass_hz: float) -> np.ndarray:
    """Discrete cosine transform basis implementing a high-pass filter."""
    duration = n_timepoints * tr
    if duration <= 0 or high_pass_hz <= 0:
        return np.zeros((n_timepoints, 0))
    n_basis = int(np.floor(2.0 * duration * high_pass_hz))
    n_basis = max(0, min(n_basis, n_timepoints - 2))
    if n_basis <= 0:
        return np.zeros((n_timepoints, 0))
    time_index = np.arange(n_timepoints)
    basis = np.zeros((n_timepoints, n_basis))
    for k in range(1, n_basis + 1):
        basis[:, k - 1] = np.sqrt(2.0 / n_timepoints) * np.cos(
            np.pi * (2 * time_index + 1) * k / (2.0 * n_timepoints)
        )
    return basis


def regress_out(signals: np.ndarray, design: np.ndarray,
                sample_mask: np.ndarray | None = None) -> np.ndarray:
    """Remove nuisance variance from ``signals`` (n_timepoints x n_vertices).

    When ``sample_mask`` is provided the regression *weights* are estimated from
    usable frames only, but the fit is removed from all frames so the output
    keeps its original time base.
    """
    if signals.ndim != 2:
        raise ValueError("signals must be 2-D (timepoints x vertices)")
    if design.shape[0] != signals.shape[0]:
        raise ValueError(
            f"design has {design.shape[0]} timepoints, signals have {signals.shape[0]}"
        )

    if sample_mask is None:
        fit_design, fit_signals = design, signals
    else:
        mask = np.asarray(sample_mask, dtype=bool)
        if mask.sum() < design.shape[1] + 2:
            raise ConfoundError(
                f"Only {int(mask.sum())} usable frames for {design.shape[1]} regressors; "
                "the nuisance model is not identifiable."
            )
        fit_design, fit_signals = design[mask], signals[mask]

    beta, *_ = np.linalg.lstsq(fit_design, fit_signals, rcond=None)
    return signals - design @ beta


def standardize(signals: np.ndarray, sample_mask: np.ndarray | None = None) -> np.ndarray:
    """Z-score each vertex over usable timepoints.

    Vertices with no variance (e.g. medial wall) become exact zeros rather than
    NaN, so downstream correlations stay well-defined.
    """
    reference = signals if sample_mask is None else signals[np.asarray(sample_mask, dtype=bool)]
    out, _flat = safe_zscore(signals, reference=reference)
    return out


def denoise(signals: np.ndarray, confounds_path: Path, settings: Settings, tr: float,
            sample_mask: np.ndarray | None = None,
            drop: int = 0) -> tuple[np.ndarray, ConfoundSelection, dict]:
    """Full denoising: confound regression, filtering and standardisation."""
    frame = load_confounds(confounds_path)
    if drop:
        frame = frame.iloc[drop:].reset_index(drop=True)
    if frame.shape[0] != signals.shape[0]:
        raise ConfoundError(
            f"Confounds have {frame.shape[0]} rows but BOLD has {signals.shape[0]} "
            "timepoints after non-steady-state removal."
        )

    selection = select_confounds(frame, settings)
    design = build_design_matrix(frame, selection, settings, tr)
    cleaned = regress_out(signals, design, sample_mask)

    report = {
        "n_regressors": int(design.shape[1]),
        "n_timepoints": int(signals.shape[0]),
        "n_vertices": int(signals.shape[1]),
        "dropped_nonsteady_state": int(drop),
        **selection.to_dict(),
    }

    if str(settings.get("preprocessing.denoise.standardize", "zscore")) == "zscore":
        cleaned = standardize(cleaned, sample_mask)
        report["standardized"] = "zscore"
    else:
        report["standardized"] = "none"

    return cleaned, selection, report


def mean_framewise_displacement(frame: pd.DataFrame) -> float | None:
    if "framewise_displacement" not in frame.columns:
        return None
    values = pd.to_numeric(frame["framewise_displacement"], errors="coerce").to_numpy()
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else None
