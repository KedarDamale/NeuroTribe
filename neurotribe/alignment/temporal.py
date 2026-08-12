"""Temporal alignment.

The single most important rule in this module:

    NEVER compute ``observed - predicted`` without first aligning timestamps.

Inputs to the alignment are the stimulus time base, TRIBE's own segment
timestamps, the acquisition TR, the volume count, the number of removed initial
volumes, and the censor mask.

TRIBE v2 documents that its prediction timing **already** incorporates a 5 s
hemodynamic-lag offset. We therefore apply no additional shift and interpolate
onto the acquisition grid using the timestamps TRIBE returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from neurotribe.config import Settings
from neurotribe.logging_setup import get_logger

log = get_logger(__name__)


class AlignmentError(RuntimeError):
    """Alignment could not be performed safely."""


@dataclass
class AlignedPair:
    """Prediction and observation on one common, censored time base."""

    predicted: np.ndarray            # (n_shared, n_vertices)
    observed: np.ndarray             # (n_shared, n_vertices)
    time_sec: np.ndarray             # acquisition time of each shared frame
    usable: np.ndarray               # bool mask over the shared frames
    report: dict = field(default_factory=dict)

    @property
    def n_shared(self) -> int:
        return int(self.time_sec.size)

    @property
    def n_usable(self) -> int:
        return int(self.usable.sum())

    def usable_only(self) -> tuple[np.ndarray, np.ndarray]:
        return self.predicted[self.usable], self.observed[self.usable]


def interpolate_to_grid(values: np.ndarray, source_times: np.ndarray,
                        target_times: np.ndarray, *, method: str = "linear",
                        allow_extrapolation: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Resample ``values`` from ``source_times`` onto ``target_times``.

    Returns ``(resampled, in_support)`` where ``in_support`` marks the target
    samples that lie inside the source support. Extrapolation is refused by
    default: inventing predictions outside TRIBE's support would fabricate data.
    """
    if values.ndim != 2:
        raise AlignmentError(f"values must be 2-D (time x vertices); got {values.shape}")
    if source_times.size != values.shape[0]:
        raise AlignmentError(
            f"source_times has {source_times.size} entries for {values.shape[0]} timepoints"
        )
    if source_times.size < 2:
        raise AlignmentError("At least two source timepoints are required to interpolate.")
    if not np.all(np.diff(source_times) > 0):
        raise AlignmentError("source_times must be strictly increasing.")

    low, high = float(source_times[0]), float(source_times[-1])
    in_support = (target_times >= low) & (target_times <= high)

    if not allow_extrapolation and not in_support.any():
        raise AlignmentError(
            f"No acquisition timepoint falls inside the prediction support "
            f"[{low:.2f}, {high:.2f}] s. The stimulus and the scan do not overlap."
        )

    if method == "cubic":
        try:
            from scipy.interpolate import CubicSpline

            spline = CubicSpline(source_times, values, axis=0, extrapolate=allow_extrapolation)
            resampled = np.asarray(spline(target_times), dtype=np.float32)
        except ImportError:
            log.warning("SciPy unavailable; falling back to linear interpolation")
            method = "linear"
        else:
            if not allow_extrapolation:
                resampled[~in_support] = np.nan
            return resampled, in_support

    # Linear path: vectorised over vertices.
    resampled = np.empty((target_times.size, values.shape[1]), dtype=np.float32)
    for vertex in range(values.shape[1]):
        resampled[:, vertex] = np.interp(target_times, source_times, values[:, vertex])
    if not allow_extrapolation:
        resampled[~in_support] = np.nan
    return resampled, in_support


def build_shared_axis(observed_times: np.ndarray, prediction_start: float,
                      prediction_end: float, *, edge_trim_sec: float = 0.0
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Restrict the acquisition grid to the interval where predictions exist.

    The acquisition grid is authoritative - we resample the model onto the
    scanner's clock, not the other way round, so no observed data is invented.
    """
    low = prediction_start + edge_trim_sec
    high = prediction_end - edge_trim_sec
    if high <= low:
        raise AlignmentError(
            f"Prediction support [{prediction_start:.2f}, {prediction_end:.2f}] s is "
            f"shorter than twice the {edge_trim_sec:.2f} s edge trim."
        )
    keep = (observed_times >= low) & (observed_times <= high)
    if not keep.any():
        raise AlignmentError(
            f"No acquisition frame lies within the trimmed prediction support "
            f"[{low:.2f}, {high:.2f}] s."
        )
    return observed_times[keep], keep


def align(predicted: np.ndarray, prediction_times: np.ndarray, prediction_ends: np.ndarray,
          observed: np.ndarray, observed_times: np.ndarray, usable: np.ndarray,
          settings: Settings) -> AlignedPair:
    """Put TRIBE predictions and observed BOLD on one censored time base."""
    if predicted.shape[1] != observed.shape[1]:
        raise AlignmentError(
            f"Vertex counts differ: predicted {predicted.shape[1]}, observed "
            f"{observed.shape[1]}. Spatial alignment must run first."
        )
    if observed.shape[0] != observed_times.size:
        raise AlignmentError(
            f"observed has {observed.shape[0]} timepoints but {observed_times.size} times"
        )
    if usable.size != observed_times.size:
        raise AlignmentError(
            f"Censor mask length {usable.size} does not match {observed_times.size} frames"
        )

    method = str(settings.get("alignment.interpolation", "linear"))
    allow_extrapolation = bool(settings.get("alignment.allow_extrapolation", False))
    edge_trim = float(settings.get("alignment.edge_trim_sec", 0.0))

    prediction_start = float(prediction_times[0])
    prediction_end = float(prediction_ends[-1])

    shared_times, keep = build_shared_axis(
        observed_times, prediction_start, prediction_end, edge_trim_sec=edge_trim,
    )
    observed_shared = observed[keep]
    usable_shared = usable[keep].copy()

    resampled, in_support = interpolate_to_grid(
        predicted, prediction_times, shared_times, method=method,
        allow_extrapolation=allow_extrapolation,
    )

    # Frames outside the prediction support, or containing non-finite values in
    # either series, are censored rather than imputed.
    non_finite = ~np.isfinite(resampled).all(axis=1) | ~np.isfinite(observed_shared).all(axis=1)
    usable_shared &= in_support
    usable_shared &= ~non_finite

    report = {
        "n_observed_frames": int(observed_times.size),
        "n_shared_frames": int(shared_times.size),
        "n_usable_frames": int(usable_shared.sum()),
        "usable_fraction": round(float(usable_shared.mean()) if usable_shared.size else 0.0, 4),
        "prediction_support_sec": [prediction_start, prediction_end],
        "observed_span_sec": [float(observed_times[0]), float(observed_times[-1])],
        "shared_span_sec": [float(shared_times[0]), float(shared_times[-1])],
        "interpolation": method,
        "edge_trim_sec": edge_trim,
        "extrapolation_allowed": allow_extrapolation,
        "n_frames_outside_support": int((~in_support).sum()),
        "n_frames_non_finite": int(non_finite.sum()),
        "additional_hrf_shift_sec": float(settings.get("tribe.additional_hrf_shift_sec", 0.0)),
        "hrf_note": (
            "TRIBE's own timestamps are used verbatim; its published timing already "
            "includes a 5 s hemodynamic offset, so no second shift is applied."
        ),
    }

    if usable_shared.sum() < 2:
        raise AlignmentError(
            f"Only {int(usable_shared.sum())} usable frame(s) remain after alignment and "
            "censoring; no meaningful comparison is possible."
        )

    log.debug("Temporal alignment complete", extra=report)
    return AlignedPair(predicted=resampled, observed=observed_shared,
                       time_sec=shared_times, usable=usable_shared, report=report)


def estimate_lag(predicted: np.ndarray, observed: np.ndarray, tr: float,
                 max_lag_sec: float = 8.0) -> dict:
    """Cross-correlation lag diagnostic.

    This is a **validation diagnostic only**. It never silently re-aligns the
    data: a large residual lag means something is wrong with the timing
    assumptions and must be investigated, not patched.
    """
    if tr <= 0:
        raise AlignmentError("TR must be positive")

    # Work on the global mean signal - robust and cheap.
    a = np.nanmean(predicted, axis=1)
    b = np.nanmean(observed, axis=1)
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    if a.size < 8:
        return {"ok": False, "reason": "Too few finite frames for a lag estimate."}

    a = (a - a.mean()) / (a.std() or 1.0)
    b = (b - b.mean()) / (b.std() or 1.0)

    max_lag_frames = int(round(max_lag_sec / tr))
    max_lag_frames = max(1, min(max_lag_frames, a.size // 3))

    lags = np.arange(-max_lag_frames, max_lag_frames + 1)
    correlations = []
    for lag in lags:
        if lag < 0:
            x, y = a[-lag:], b[:lag]
        elif lag > 0:
            x, y = a[:-lag], b[lag:]
        else:
            x, y = a, b
        correlations.append(float(np.dot(x, y) / max(len(x), 1)))

    correlations = np.asarray(correlations)
    best = int(np.argmax(correlations))
    return {
        "ok": True,
        "best_lag_frames": int(lags[best]),
        "best_lag_sec": float(lags[best] * tr),
        "best_correlation": float(correlations[best]),
        "zero_lag_correlation": float(correlations[max_lag_frames]),
        "searched_sec": max_lag_sec,
    }
