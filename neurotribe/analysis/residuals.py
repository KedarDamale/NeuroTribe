"""Core subject-level deviation metrics.

Raw BOLD amplitude and arbitrary model output are not on a comparable scale, so
both series are z-scored **over the usable timepoints only** before any
comparison (specification section 29). Every metric below is computed on
censored data; censored frames never contribute.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from neurotribe.logging_setup import get_logger
from neurotribe.numerics import safe_zscore

log = get_logger(__name__)


def zscore_usable(data: np.ndarray, usable: np.ndarray) -> np.ndarray:
    """Z-score each vertex using statistics from usable frames only.

    Flat vertices become exact zeros so downstream correlations stay defined
    rather than producing NaN that silently propagates.
    """
    if data.shape[0] != usable.size:
        raise ValueError(f"data has {data.shape[0]} frames, mask has {usable.size}")
    if usable.sum() < 2:
        raise ValueError("At least two usable frames are required to standardise.")

    out, _flat = safe_zscore(data.astype(np.float32, copy=False),
                             reference=data[usable])
    return out


def pearson_per_vertex(predicted: np.ndarray, observed: np.ndarray,
                       usable: np.ndarray) -> np.ndarray:
    """Vertex-wise Pearson r between prediction and observation.

    High r: TRIBE explains the response pattern well.
    Low  r: the response differs from the normative prediction.
    """
    p = predicted[usable]
    o = observed[usable]
    if p.shape[0] < 3:
        raise ValueError("At least three usable frames are required for a correlation.")

    p_centered = p - p.mean(axis=0, keepdims=True)
    o_centered = o - o.mean(axis=0, keepdims=True)
    numerator = (p_centered * o_centered).sum(axis=0)
    denominator = np.sqrt((p_centered ** 2).sum(axis=0) * (o_centered ** 2).sum(axis=0))

    # A vanishing denominator means one side is constant: the correlation is
    # undefined, so it must be NaN rather than an arbitrary finite number.
    scale = np.maximum(np.abs(p_centered).max(axis=0), 1e-12) * \
        np.maximum(np.abs(o_centered).max(axis=0), 1e-12) * p.shape[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(denominator > 1e-6 * scale, numerator / denominator, np.nan)
    # Guard against floating point drift outside [-1, 1].
    return np.clip(r, -1.0, 1.0).astype(np.float32)


def standardized_residual(predicted_z: np.ndarray, observed_z: np.ndarray) -> np.ndarray:
    """residual(t, v) = observed_z(t, v) - predicted_z(t, v)."""
    if predicted_z.shape != observed_z.shape:
        raise ValueError(
            f"Shape mismatch: predicted {predicted_z.shape}, observed {observed_z.shape}"
        )
    return (observed_z - predicted_z).astype(np.float32)


def mean_absolute_deviation(residual: np.ndarray, usable: np.ndarray) -> np.ndarray:
    """MAD(v) = mean over usable t of |residual(t, v)|."""
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.abs(residual[usable]), axis=0).astype(np.float32)


def residual_variance(residual: np.ndarray, usable: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanvar(residual[usable], axis=0).astype(np.float32)


@dataclass
class SubjectMetrics:
    """Per-vertex deviation maps plus whole-cortex summaries."""

    vertex_r: np.ndarray
    vertex_mad: np.ndarray
    vertex_residual_variance: np.ndarray
    residual: np.ndarray
    valid_vertices: np.ndarray
    global_r: float
    global_mad: float
    global_residual_variance: float
    n_usable_frames: int
    n_shared_frames: int
    details: dict = field(default_factory=dict)

    def to_summary(self) -> dict:
        return {
            "global_agreement_r": self.global_r,
            "global_mad": self.global_mad,
            "global_residual_variance": self.global_residual_variance,
            "n_usable_frames": self.n_usable_frames,
            "n_shared_frames": self.n_shared_frames,
            "n_valid_vertices": int(self.valid_vertices.sum()),
            **self.details,
        }


def compute(predicted: np.ndarray, observed: np.ndarray, usable: np.ndarray,
            valid_vertices: np.ndarray) -> SubjectMetrics:
    """Full per-subject metric computation on aligned, censored data."""
    predicted_z = zscore_usable(predicted, usable)
    observed_z = zscore_usable(observed, usable)

    r = pearson_per_vertex(predicted_z, observed_z, usable)
    residual = standardized_residual(predicted_z, observed_z)
    mad = mean_absolute_deviation(residual, usable)
    variance = residual_variance(residual, usable)

    # Invalid vertices (medial wall, flat signal) are excluded from every map.
    invalid = ~valid_vertices
    r[invalid] = np.nan
    mad[invalid] = np.nan
    variance[invalid] = np.nan
    residual[:, invalid] = np.nan

    with np.errstate(invalid="ignore"):
        global_r = float(np.nanmean(r))
        global_mad = float(np.nanmean(mad))
        global_variance = float(np.nanmean(variance))

    return SubjectMetrics(
        vertex_r=r, vertex_mad=mad, vertex_residual_variance=variance,
        residual=residual, valid_vertices=valid_vertices,
        global_r=global_r if np.isfinite(global_r) else float("nan"),
        global_mad=global_mad if np.isfinite(global_mad) else float("nan"),
        global_residual_variance=(global_variance if np.isfinite(global_variance)
                                  else float("nan")),
        n_usable_frames=int(usable.sum()), n_shared_frames=int(usable.size),
        details={
            "median_vertex_r": float(np.nanmedian(r)) if np.isfinite(r).any() else None,
            "r_percentiles": (
                [float(v) for v in np.nanpercentile(r[np.isfinite(r)], [5, 25, 50, 75, 95])]
                if np.isfinite(r).any() else None
            ),
        },
    )


# --------------------------------------------------------------------------
# Movie-moment analysis
# --------------------------------------------------------------------------

@dataclass
class RollingDeviation:
    window_starts: np.ndarray        # seconds
    window_ends: np.ndarray
    deviation: np.ndarray            # mean |residual| within each window
    coverage: np.ndarray             # fraction of usable frames per window

    def to_dict(self) -> dict:
        return {
            "window_starts": [round(float(v), 3) for v in self.window_starts],
            "window_ends": [round(float(v), 3) for v in self.window_ends],
            "deviation": [None if not np.isfinite(v) else round(float(v), 6)
                          for v in self.deviation],
            "coverage": [round(float(v), 3) for v in self.coverage],
        }


def rolling_deviation(residual: np.ndarray, time_sec: np.ndarray, usable: np.ndarray,
                      window_sec: float = 10.0, step_sec: float = 1.0,
                      min_coverage: float = 0.5) -> RollingDeviation:
    """Sliding-window global deviation, answering *when* the brain diverged.

    Windows whose usable-frame coverage falls below ``min_coverage`` yield NaN
    rather than a deviation computed from a handful of surviving frames.
    """
    if time_sec.size != residual.shape[0]:
        raise ValueError("time_sec length must match the residual timepoints")
    if window_sec <= 0 or step_sec <= 0:
        raise ValueError("window_sec and step_sec must be positive")

    start_time = float(time_sec[0])
    end_time = float(time_sec[-1])
    if end_time - start_time < window_sec:
        window_sec = max(1.0, end_time - start_time)

    starts = np.arange(start_time, end_time - window_sec + 1e-9, step_sec)
    if starts.size == 0:
        starts = np.array([start_time])
    ends = starts + window_sec

    deviation = np.full(starts.size, np.nan, dtype=np.float32)
    coverage = np.zeros(starts.size, dtype=np.float32)

    for index, (low, high) in enumerate(zip(starts, ends)):
        in_window = (time_sec >= low) & (time_sec < high)
        n_frames = int(in_window.sum())
        if n_frames == 0:
            continue
        selected = in_window & usable
        coverage[index] = selected.sum() / n_frames
        if coverage[index] < min_coverage or selected.sum() == 0:
            continue
        with np.errstate(invalid="ignore"):
            deviation[index] = float(np.nanmean(np.abs(residual[selected])))

    return RollingDeviation(window_starts=starts, window_ends=ends,
                            deviation=deviation, coverage=coverage)


def peak_windows(rolling: RollingDeviation, top_n: int = 10,
                 min_separation_sec: float = 5.0) -> list[dict]:
    """Rank the highest-deviation moments, suppressing overlapping duplicates."""
    finite = np.isfinite(rolling.deviation)
    if not finite.any():
        return []

    order = np.argsort(-np.where(finite, rolling.deviation, -np.inf))
    chosen: list[int] = []
    for index in order:
        if not finite[index]:
            break
        start = rolling.window_starts[index]
        if any(abs(start - rolling.window_starts[picked]) < min_separation_sec
               for picked in chosen):
            continue
        chosen.append(int(index))
        if len(chosen) >= top_n:
            break

    return [
        {
            "rank": rank + 1,
            "start_sec": round(float(rolling.window_starts[i]), 2),
            "end_sec": round(float(rolling.window_ends[i]), 2),
            "start_label": _format_timecode(float(rolling.window_starts[i])),
            "end_label": _format_timecode(float(rolling.window_ends[i])),
            "deviation": round(float(rolling.deviation[i]), 6),
            "coverage": round(float(rolling.coverage[i]), 3),
        }
        for rank, i in enumerate(chosen)
    ]


def _format_timecode(seconds: float) -> str:
    minutes, secs = divmod(max(0.0, seconds), 60.0)
    return f"{int(minutes):02d}:{int(secs):02d}"
