"""Spatial alignment between TRIBE's cortical output and observed BOLD surfaces.

Both sides live on ``fsaverage5``, so no resampling is required - but the
*hemisphere ordering* and *medial-wall handling* must match exactly. This module
enforces that, and it is the only place permitted to reorder vertices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from neurotribe.config import Settings
from neurotribe.logging_setup import get_logger
from neurotribe.numerics import is_flat_variance, variance_along_time
from neurotribe.preprocessing.surfaces import (
    SurfaceError, concatenate_hemispheres, split_hemispheres,
)
from neurotribe.tribe.geometry import assert_compatible

log = get_logger(__name__)


@dataclass
class SpatialAlignment:
    predicted: np.ndarray
    observed: np.ndarray
    valid_vertices: np.ndarray       # bool mask of vertices usable in both
    report: dict = field(default_factory=dict)

    @property
    def n_valid(self) -> int:
        return int(self.valid_vertices.sum())


def reorder_to(data: np.ndarray, source_order: list[str], target_order: list[str],
               per_hemi: int) -> np.ndarray:
    """Rewrite the flattened vertex axis from one hemisphere order to another."""
    if source_order == target_order:
        return data
    hemispheres = split_hemispheres(data, source_order, per_hemi)
    reordered = concatenate_hemispheres(
        hemispheres[target_order[0]], hemispheres[target_order[1]], ["L", "R"],
    )
    log.warning("Reordered hemispheres to match the observed surface convention",
                extra={"from": source_order, "to": target_order})
    return reordered


def align(predicted: np.ndarray, observed: np.ndarray, settings: Settings,
          *, tribe_hemi_order: list[str] | None = None) -> SpatialAlignment:
    """Put both arrays on the canonical vertex axis and mask unusable vertices."""
    assert_compatible(predicted.shape[-1], observed.shape[-1])

    target_order = list(settings.get("surface.hemi_order", ["L", "R"]))
    per_hemi = int(settings.get("surface.vertices_per_hemi", predicted.shape[-1] // 2))
    source_order = list(tribe_hemi_order or target_order)

    if source_order != target_order:
        predicted = reorder_to(predicted, source_order, target_order, per_hemi)

    # A vertex is usable only where BOTH series carry signal across time.
    predicted_var, predicted_mean = variance_along_time(predicted)
    observed_var, observed_mean = variance_along_time(observed)
    with np.errstate(invalid="ignore"):
        predicted_finite = np.isfinite(predicted).all(axis=0)
        observed_finite = np.isfinite(observed).all(axis=0)

    predicted_ok = predicted_finite & ~is_flat_variance(predicted_var, predicted_mean)
    observed_ok = observed_finite & ~is_flat_variance(observed_var, observed_mean)
    valid = predicted_ok & observed_ok

    policy = str(settings.get("surface.medial_wall_policy", "mask"))
    if policy != "mask":
        valid = np.ones_like(valid)

    report = {
        "n_vertices": int(predicted.shape[-1]),
        "per_hemi_vertices": per_hemi,
        "tribe_hemi_order": source_order,
        "target_hemi_order": target_order,
        "reordered": source_order != target_order,
        "n_valid_vertices": int(valid.sum()),
        "n_masked_predicted": int((~predicted_ok).sum()),
        "n_masked_observed": int((~observed_ok).sum()),
        "medial_wall_policy": policy,
        "valid_fraction": round(float(valid.mean()), 4),
    }

    if valid.sum() == 0:
        raise SurfaceError(
            "No vertex carries usable signal in both the prediction and the "
            "observation. Check surface space and denoising."
        )
    if valid.mean() < 0.5:
        log.warning("Fewer than half of cortical vertices are usable", extra=report)

    log.debug("Spatial alignment complete", extra=report)
    return SpatialAlignment(predicted=predicted, observed=observed,
                            valid_vertices=valid, report=report)


def hemisphere_of(vertex_indices: np.ndarray, settings: Settings) -> np.ndarray:
    """Map flattened vertex indices to 'L'/'R' using the configured order."""
    order = list(settings.get("surface.hemi_order", ["L", "R"]))
    per_hemi = int(settings.get("surface.vertices_per_hemi", 10242))
    return np.where(np.asarray(vertex_indices) < per_hemi, order[0], order[1])
