"""Numerical helpers shared by every module that standardises or masks signals.

Why this exists: a fixed absolute threshold such as ``sd < 1e-12`` is wrong for
float32 data. A genuinely constant float32 vertex holding, say, 3.14 has a
computed standard deviation around 5e-7 - far above 1e-12 - so it would be
treated as varying, and ``(x - mean) / sd`` would amplify pure rounding noise
into a z-score of +/-1. On the medial wall, where values are constant, that
turns nothing into a confident-looking signal.

Every flatness test therefore uses a *scale-aware* tolerance.
"""

from __future__ import annotations

import numpy as np

# Relative tolerance: float32 carries ~7 significant digits, so anything within
# ~1e-5 of the value's own magnitude is indistinguishable from constant.
RELATIVE_TOLERANCE = 1e-5
# Absolute floor for values centred near zero.
ABSOLUTE_TOLERANCE = 1e-8


def flatness_tolerance(reference: np.ndarray) -> np.ndarray:
    """Per-element tolerance below which a standard deviation counts as zero."""
    return np.maximum(ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE * np.abs(reference))


def is_flat_sd(sd: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """True where a standard deviation is indistinguishable from zero."""
    sd = np.asarray(sd)
    return ~np.isfinite(sd) | (sd <= flatness_tolerance(np.asarray(mean)))


def is_flat_variance(variance: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Variance-domain equivalent of :func:`is_flat_sd`."""
    variance = np.asarray(variance)
    tolerance = flatness_tolerance(np.asarray(mean))
    return ~np.isfinite(variance) | (variance <= tolerance ** 2)


def safe_zscore(data: np.ndarray, *, axis: int = 0,
                reference: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Z-score along ``axis``, returning ``(z, flat_mask)``.

    Flat elements become exact zeros rather than amplified rounding noise, so
    downstream correlations stay well-defined instead of producing NaN or
    spurious +/-1 values.

    ``reference`` lets the statistics be estimated from a subset (e.g. usable
    frames only) while the transform is applied to the whole array.
    """
    source = data if reference is None else reference
    mean = np.nanmean(source, axis=axis, keepdims=True)
    sd = np.nanstd(source, axis=axis, keepdims=True)

    flat = is_flat_sd(sd, mean)
    safe_sd = np.where(flat, 1.0, sd)
    out = (data - mean) / safe_sd

    flat_1d = flat.reshape(-1)
    if axis == 0:
        out[:, flat_1d] = 0.0
    else:
        out[flat_1d] = 0.0
    return out.astype(data.dtype, copy=False), flat_1d


def variance_along_time(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column variance and mean, NaN-safe."""
    with np.errstate(invalid="ignore"):
        return np.nanvar(data, axis=0), np.nanmean(data, axis=0)
