"""TRIBE output geometry validation.

Specification section 26: *do not assume hemisphere concatenation order.* This
module inspects TRIBE's own implementation and prediction array to establish:

    vertex indexing, hemisphere ordering, medial-wall treatment, surface mask

and refuses to let an analysis proceed on an unverified convention. A silent
left/right swap would invert every spatial conclusion while leaving all the
summary statistics looking perfectly plausible - which is exactly why this check
is mandatory rather than advisory.
"""

from __future__ import annotations

import importlib
import inspect
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from neurotribe.config import Settings
from neurotribe.logging_setup import get_logger
from neurotribe.numerics import is_flat_variance, variance_along_time
from neurotribe.preprocessing.surfaces import (
    FSAVERAGE5_TOTAL_VERTICES, FSAVERAGE5_VERTICES_PER_HEMI,
)

log = get_logger(__name__)


class GeometryError(RuntimeError):
    """Geometry could not be verified. Never downgraded to a warning."""


@dataclass
class GeometryReport:
    ok: bool = False
    n_vertices: int | None = None
    n_timepoints: int | None = None
    surface_space: str | None = None
    hemi_order: list[str] = field(default_factory=list)
    hemi_order_source: str = "configuration"
    per_hemi_vertices: int | None = None
    medial_wall_vertices: int = 0
    medial_wall_policy: str = "mask"
    evidence: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok, "n_vertices": self.n_vertices,
            "n_timepoints": self.n_timepoints, "surface_space": self.surface_space,
            "hemi_order": self.hemi_order, "hemi_order_source": self.hemi_order_source,
            "per_hemi_vertices": self.per_hemi_vertices,
            "medial_wall_vertices": self.medial_wall_vertices,
            "medial_wall_policy": self.medial_wall_policy,
            "evidence": self.evidence, "errors": self.errors,
            "warnings": self.warnings,
        }


# --------------------------------------------------------------------------
# Introspection of the TRIBE package
# --------------------------------------------------------------------------

_HEMI_ORDER_PATTERNS = (
    (re.compile(r"\[\s*['\"]lh['\"]\s*,\s*['\"]rh['\"]\s*\]", re.I), ["L", "R"]),
    (re.compile(r"\[\s*['\"]rh['\"]\s*,\s*['\"]lh['\"]\s*\]", re.I), ["R", "L"]),
    (re.compile(r"\[\s*['\"]left['\"]\s*,\s*['\"]right['\"]\s*\]", re.I), ["L", "R"]),
    (re.compile(r"\[\s*['\"]right['\"]\s*,\s*['\"]left['\"]\s*\]", re.I), ["R", "L"]),
    (re.compile(r"concat.*\blh\b.*\brh\b", re.I | re.S), ["L", "R"]),
)

_SPACE_PATTERN = re.compile(r"fsaverage\d?", re.I)


def inspect_tribe_package() -> dict:
    """Read TRIBE's source to discover its declared surface conventions.

    Best-effort and non-fatal: when the package is absent (or its internals
    change), we fall back to the configured convention and record that the
    evidence was unavailable, rather than silently assuming correctness.
    """
    evidence: dict = {"available": False, "hemi_order_hits": [], "spaces": [],
                      "modules_scanned": []}
    try:
        module = importlib.import_module("tribev2")
    except ImportError as exc:
        evidence["error"] = str(exc)
        return evidence

    evidence["available"] = True
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return evidence

    package_root = Path(module_file).parent
    sources: list[tuple[str, str]] = []
    for path in sorted(package_root.rglob("*.py"))[:200]:
        try:
            sources.append((str(path.relative_to(package_root)),
                            path.read_text(encoding="utf-8", errors="replace")))
        except OSError:
            continue

    evidence["modules_scanned"] = [name for name, _ in sources]
    for name, text in sources:
        for pattern, order in _HEMI_ORDER_PATTERNS:
            if pattern.search(text):
                evidence["hemi_order_hits"].append({"module": name, "order": order})
        for match in _SPACE_PATTERN.findall(text):
            space = match.lower()
            if space not in evidence["spaces"]:
                evidence["spaces"].append(space)

    # Public attributes are the most reliable signal when they exist.
    for attribute in ("SURFACE_SPACE", "HEMI_ORDER", "N_VERTICES", "FSAVERAGE"):
        if hasattr(module, attribute):
            evidence[f"attr_{attribute}"] = repr(getattr(module, attribute))

    model_cls = getattr(module, "TribeModel", None)
    if model_cls is not None:
        try:
            evidence["predict_signature"] = str(inspect.signature(model_cls.predict))
        except (TypeError, ValueError):
            pass

    return evidence


def infer_hemi_order(evidence: dict, settings: Settings) -> tuple[list[str], str, list[str]]:
    """Decide the hemisphere order, preferring TRIBE's own declaration."""
    configured = list(settings.get("surface.hemi_order", ["L", "R"]))
    warnings: list[str] = []

    hits = evidence.get("hemi_order_hits") or []
    if not hits:
        if evidence.get("available"):
            warnings.append(
                "TRIBE's source did not declare a hemisphere order explicitly; using the "
                "configured order. Confirm against the model card before publishing."
            )
        return configured, "configuration", warnings

    orders = {tuple(hit["order"]) for hit in hits}
    if len(orders) > 1:
        warnings.append(
            f"TRIBE's source contains conflicting hemisphere-order evidence: "
            f"{sorted(tuple(o) for o in orders)}. Using the configured order."
        )
        return configured, "configuration (conflicting evidence)", warnings

    discovered = list(next(iter(orders)))
    if discovered != configured:
        warnings.append(
            f"TRIBE declares hemisphere order {discovered} but configuration says "
            f"{configured}. Adopting TRIBE's order - the configuration should be updated."
        )
        return discovered, "tribe source", warnings

    return discovered, "tribe source (matches configuration)", warnings


# --------------------------------------------------------------------------
# Validation of an actual prediction array
# --------------------------------------------------------------------------

def detect_medial_wall(predictions: np.ndarray) -> np.ndarray:
    """Vertices with no signal at all - TRIBE's medial-wall / masked vertices."""
    variance, mean = variance_along_time(predictions)
    with np.errstate(invalid="ignore"):
        all_nan = np.all(~np.isfinite(predictions), axis=0)
    return is_flat_variance(variance, mean) | all_nan


def validate(predictions: np.ndarray, segments, settings: Settings,
             *, backend: str = "real") -> GeometryReport:
    """Full geometry validation. Any failure means the analysis is invalid."""
    report = GeometryReport(medial_wall_policy=str(settings.get("surface.medial_wall_policy", "mask")))

    if predictions.ndim != 2:
        report.errors.append(
            f"TRIBE predictions must be 2-D (time x vertices); got shape {predictions.shape}."
        )
        return report

    report.n_timepoints, report.n_vertices = (int(predictions.shape[0]),
                                              int(predictions.shape[1]))
    expected_total = int(settings.get("surface.total_vertices", FSAVERAGE5_TOTAL_VERTICES))
    expected_per_hemi = int(settings.get("surface.vertices_per_hemi",
                                         FSAVERAGE5_VERTICES_PER_HEMI))
    report.surface_space = str(settings.get("surface.space", "fsaverage5"))

    evidence = inspect_tribe_package()
    report.evidence = evidence
    order, source, warnings = infer_hemi_order(evidence, settings)
    report.hemi_order = order
    report.hemi_order_source = source
    report.warnings.extend(warnings)

    if report.n_vertices != expected_total:
        report.errors.append(
            f"TRIBE returned {report.n_vertices} vertices but the configured "
            f"{report.surface_space} surface has {expected_total}. The two data "
            "sources are not in the same space; comparison is impossible."
        )
        return report

    if report.n_vertices % 2 != 0:
        report.errors.append("Vertex count is odd; hemispheres cannot be split evenly.")
        return report
    report.per_hemi_vertices = report.n_vertices // 2
    if report.per_hemi_vertices != expected_per_hemi:
        report.errors.append(
            f"Per-hemisphere vertex count {report.per_hemi_vertices} does not match "
            f"the configured {expected_per_hemi}."
        )
        return report

    if report.n_timepoints < 2:
        report.errors.append(f"Only {report.n_timepoints} predicted timepoint(s).")
        return report

    medial = detect_medial_wall(predictions)
    report.medial_wall_vertices = int(medial.sum())
    left_masked = int(medial[:report.per_hemi_vertices].sum())
    right_masked = int(medial[report.per_hemi_vertices:].sum())
    report.evidence["medial_wall_per_hemi"] = {"first_half": left_masked,
                                               "second_half": right_masked}

    # A real fsaverage5 medial wall is a few hundred vertices per hemisphere and
    # roughly symmetric. Gross asymmetry means the halves are not hemispheres.
    if report.medial_wall_vertices > 0:
        larger = max(left_masked, right_masked)
        smaller = min(left_masked, right_masked)
        if smaller == 0 and larger > report.per_hemi_vertices * 0.05:
            report.warnings.append(
                f"Masked vertices are entirely in one half ({larger} vs 0). Verify the "
                "hemisphere split - this is not a normal medial-wall pattern."
            )
        elif smaller > 0 and larger / smaller > 3.0:
            report.warnings.append(
                f"Medial-wall masks are strongly asymmetric ({left_masked} vs "
                f"{right_masked}); verify hemisphere ordering."
            )
    if report.medial_wall_vertices > report.n_vertices * 0.4:
        report.errors.append(
            f"{report.medial_wall_vertices} of {report.n_vertices} vertices carry no "
            "signal. The prediction is largely empty."
        )
        return report

    finite_fraction = float(np.isfinite(predictions).mean())
    report.evidence["finite_fraction"] = round(finite_fraction, 6)
    max_nan = float(settings.get("analysis.sanity.max_nan_fraction", 0.02))
    if 1.0 - finite_fraction > max(max_nan, report.medial_wall_vertices / report.n_vertices):
        report.errors.append(
            f"{(1 - finite_fraction):.1%} of prediction values are non-finite, exceeding "
            f"the {max_nan:.1%} tolerance."
        )
        return report

    # Timestamp support.
    try:
        starts = np.asarray(segments["start"], dtype=float)
        ends = np.asarray(segments["end"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        report.errors.append(f"TRIBE segments lack usable start/end timestamps: {exc}")
        return report

    if starts.size != report.n_timepoints:
        report.errors.append(
            f"Segment count ({starts.size}) does not match prediction timepoints "
            f"({report.n_timepoints})."
        )
        return report
    if not np.all(np.diff(starts) > 0):
        report.errors.append("TRIBE segment start times are not strictly increasing.")
        return report

    report.evidence["time_start_sec"] = float(starts[0])
    report.evidence["time_end_sec"] = float(ends[-1])
    report.evidence["median_segment_sec"] = float(np.median(np.diff(starts)))

    if backend == "mock":
        report.warnings.append(
            "Geometry validated against the MOCK backend. This proves the interface "
            "contract only; it is not evidence about the real model."
        )

    report.ok = not report.errors
    log.info("TRIBE geometry validated", extra=report.to_dict())
    return report


def assert_compatible(tribe_vertices: int, observed_vertices: int) -> None:
    """Hard gate used before any subtraction between prediction and observation."""
    if tribe_vertices != observed_vertices:
        raise GeometryError(
            f"Vertex-count mismatch: TRIBE has {tribe_vertices}, observed BOLD has "
            f"{observed_vertices}. Refusing to compare arrays in different spaces."
        )
