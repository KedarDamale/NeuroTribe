"""Scientific sanity gate (specification section 71).

Every check below runs before a result may be declared. A failure produces

    ANALYSIS INVALID

not a silent warning, because each of these failure modes produces output that
*looks* perfectly reasonable while being scientifically meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from neurotribe.config import Settings
from neurotribe.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class SanityReport:
    checks: dict[str, bool] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return not self.failures

    @property
    def verdict(self) -> str:
        return "VALID" if self.valid else "ANALYSIS INVALID"

    def record(self, name: str, passed: bool, message: str = "", *,
               warn_only: bool = False) -> bool:
        self.checks[name] = passed
        if not passed:
            (self.warnings if warn_only else self.failures).append(message or name)
        return passed

    def to_dict(self) -> dict:
        return {
            "valid": self.valid, "verdict": self.verdict, "checks": self.checks,
            "failures": self.failures, "warnings": self.warnings,
            "details": self.details,
        }


def check_subject_comparison(
    *, predicted: np.ndarray, observed: np.ndarray, tribe_vertices: int,
    observed_vertices: int, hemi_order_verified: bool, movie_duration_sec: float | None,
    stimulus_expected_sec: float | None, tr: float | None, n_volumes: int | None,
    shared_frames: int, usable_frames: int, correlations: np.ndarray | None,
    censoring_applied: bool, settings: Settings, subject_seen_before: bool = False,
) -> SanityReport:
    """Run the full per-subject sanity battery."""
    report = SanityReport()

    report.record(
        "vertex_counts_match", tribe_vertices == observed_vertices,
        f"TRIBE vertex count ({tribe_vertices}) != observed ({observed_vertices}).",
    )
    report.record(
        "hemisphere_order_confirmed", hemi_order_verified,
        "Hemisphere ordering was not confirmed against TRIBE's implementation.",
        warn_only=True,
    )

    if movie_duration_sec is not None and stimulus_expected_sec:
        tolerance = max(5.0, stimulus_expected_sec * 0.02)
        delta = abs(movie_duration_sec - stimulus_expected_sec)
        report.details["movie_duration_delta_sec"] = round(delta, 3)
        report.record(
            "movie_duration_matches", delta <= tolerance,
            f"Supplied stimulus is {movie_duration_sec:.1f}s but the documented HBN "
            f"interval is {stimulus_expected_sec:.1f}s (delta {delta:.1f}s).",
        )
    else:
        report.record("movie_duration_matches", True, warn_only=True)

    report.record(
        "tr_valid", bool(tr and 0.1 < tr < 5.0),
        f"RepetitionTime {tr} is outside the plausible 0.1-5.0 s range.",
    )
    report.record(
        "volume_count_present", bool(n_volumes and n_volumes > 0),
        "Volume count is missing or zero.",
    )

    report.record(
        "temporal_overlap_valid", shared_frames >= 2,
        f"Only {shared_frames} frame(s) overlap between the stimulus and the scan.",
    )
    min_frames = int(settings.get("qc.motion.min_usable_frames", 60))
    report.record(
        "sufficient_usable_frames", usable_frames >= min_frames,
        f"{usable_frames} usable frames after censoring (minimum {min_frames}).",
    )
    report.record(
        "censoring_applied", censoring_applied,
        "Motion censoring was not applied before comparison.",
    )

    for name, array in (("predicted", predicted), ("observed", observed)):
        if array.size == 0:
            report.record(f"{name}_non_empty", False, f"{name} array is empty.")
            continue
        nan_fraction = float(np.mean(~np.isfinite(array)))
        report.details[f"{name}_nan_fraction"] = round(nan_fraction, 6)
        limit = float(settings.get("analysis.sanity.max_nan_fraction", 0.02))
        report.record(
            f"{name}_no_nan_explosion", nan_fraction <= limit,
            f"{nan_fraction:.1%} of {name} values are non-finite (limit {limit:.1%}).",
        )

    if correlations is not None and correlations.size:
        finite = correlations[np.isfinite(correlations)]
        low, high = settings.get("analysis.sanity.correlation_bounds", [-1.0, 1.0])
        in_bounds = bool(finite.size == 0 or (finite.min() >= low - 1e-6
                                              and finite.max() <= high + 1e-6))
        report.details["correlation_range"] = (
            [float(finite.min()), float(finite.max())] if finite.size else None
        )
        report.record(
            "correlations_in_bounds", in_bounds,
            f"Correlations fall outside [{low}, {high}] - the computation is wrong.",
        )
    else:
        report.record("correlations_in_bounds", True, warn_only=True)

    report.record(
        "subject_not_duplicated", not subject_seen_before,
        "This participant already appears in the analysis.",
    )

    if not report.valid:
        log.error("Subject comparison failed the sanity gate",
                  extra={"failures": report.failures})
    return report


def check_group_analysis(*, n_case: int, n_control: int, n_units_tested: int,
                         p_values: np.ndarray | None, settings: Settings,
                         duplicate_subjects: list[str] | None = None,
                         covariates_present: dict[str, bool] | None = None) -> SanityReport:
    """Sanity battery for a group-level contrast."""
    report = SanityReport()
    minimum = int(settings.get("cohort.min_group_size", 10))

    report.record(
        "case_group_size", n_case >= minimum,
        f"Confirmed ADHD group has {n_case} participants (minimum {minimum}).",
    )
    report.record(
        "control_group_size", n_control >= minimum,
        f"Comparison group has {n_control} participants (minimum {minimum}).",
    )
    report.record(
        "units_tested", n_units_tested > 0, "No ROI or network was tested.",
    )

    duplicates = duplicate_subjects or []
    report.record(
        "no_duplicate_subjects", not duplicates,
        f"Duplicate participants in the analysis: {duplicates[:5]}",
    )

    if p_values is not None and p_values.size:
        finite = p_values[np.isfinite(p_values)]
        in_range = bool(finite.size == 0 or (finite.min() >= 0.0 and finite.max() <= 1.0))
        report.record("p_values_in_range", in_range,
                      "p-values fall outside [0, 1].")
        report.details["n_p_values"] = int(finite.size)
    else:
        report.record("p_values_in_range", True, warn_only=True)

    for name, present in (covariates_present or {}).items():
        report.record(
            f"covariate_{name}", present,
            f"Covariate '{name}' has no usable values; the adjusted model is "
            "not the one specified.",
            warn_only=True,
        )

    if not report.valid:
        log.error("Group analysis failed the sanity gate",
                  extra={"failures": report.failures})
    return report


def check_alignment_lag(lag_report: dict, settings: Settings) -> SanityReport:
    """Flag a residual timing offset that the pipeline must not silently absorb."""
    report = SanityReport()
    if not lag_report.get("ok"):
        report.record("lag_estimable", True,
                      lag_report.get("reason", "Lag not estimable"), warn_only=True)
        return report

    limit = float(settings.get("alignment.validation_max_acceptable_lag_sec", 2.0))
    lag = abs(float(lag_report.get("best_lag_sec", 0.0)))
    report.details["best_lag_sec"] = lag
    report.record(
        "residual_lag_acceptable", lag <= limit,
        f"Peak cross-correlation occurs at {lag:.2f}s lag (limit {limit:.2f}s). "
        "Investigate the timing assumptions - do NOT apply a corrective shift "
        "silently, as TRIBE already accounts for hemodynamic delay.",
        warn_only=True,
    )
    return report
