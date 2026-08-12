"""Cohort balance diagnostics and optional covariate matching.

Matching is a *sensitivity analysis*, never the primary approach: the primary
model adjusts for confounds statistically (section 35). This module reports
imbalance and can propose a matched subset for exploratory comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from neurotribe.database.enums import CohortGroup
from neurotribe.database.models import CohortMember


@dataclass
class BalanceReport:
    variable: str
    case_mean: float | None = None
    control_mean: float | None = None
    case_sd: float | None = None
    control_sd: float | None = None
    std_mean_difference: float | None = None
    n_case: int = 0
    n_control: int = 0
    note: str | None = None

    def to_dict(self) -> dict:
        return {
            "variable": self.variable, "case_mean": self.case_mean,
            "control_mean": self.control_mean, "case_sd": self.case_sd,
            "control_sd": self.control_sd,
            "std_mean_difference": self.std_mean_difference,
            "n_case": self.n_case, "n_control": self.n_control, "note": self.note,
        }


def _split(members: Sequence[CohortMember]) -> tuple[list[CohortMember], list[CohortMember]]:
    included = [m for m in members if m.included]
    case = [m for m in included if m.group == CohortGroup.CONFIRMED_ADHD.value]
    control = [m for m in included if m.group != CohortGroup.CONFIRMED_ADHD.value]
    return case, control


def _values(members: Sequence[CohortMember], attribute: str) -> np.ndarray:
    raw = [getattr(m, attribute, None) for m in members]
    return np.array([v for v in raw if v is not None], dtype=float)


def standardized_mean_difference(case: np.ndarray, control: np.ndarray) -> float | None:
    """Cohen's d style SMD used as the standard covariate-balance diagnostic."""
    if case.size < 2 or control.size < 2:
        return None
    pooled_var = (case.var(ddof=1) + control.var(ddof=1)) / 2.0
    if pooled_var <= 0:
        return 0.0
    return float((case.mean() - control.mean()) / np.sqrt(pooled_var))


def continuous_balance(members: Sequence[CohortMember], variable: str) -> BalanceReport:
    case_members, control_members = _split(members)
    case = _values(case_members, variable)
    control = _values(control_members, variable)
    report = BalanceReport(variable=variable, n_case=case.size, n_control=control.size)
    if case.size:
        report.case_mean = float(case.mean())
        report.case_sd = float(case.std(ddof=1)) if case.size > 1 else 0.0
    if control.size:
        report.control_mean = float(control.mean())
        report.control_sd = float(control.std(ddof=1)) if control.size > 1 else 0.0
    report.std_mean_difference = standardized_mean_difference(case, control)
    if report.std_mean_difference is not None and abs(report.std_mean_difference) > 0.25:
        report.note = (
            f"Imbalanced (|SMD| = {abs(report.std_mean_difference):.2f} > 0.25); "
            "the covariate-adjusted model is essential here."
        )
    return report


def categorical_balance(members: Sequence[CohortMember], attribute: str) -> dict:
    case_members, control_members = _split(members)

    def counts(group: Sequence[CohortMember]) -> dict[str, int]:
        out: dict[str, int] = {}
        for member in group:
            key = str(getattr(member, attribute, None) or "unknown")
            out[key] = out.get(key, 0) + 1
        return out

    case_counts = counts(case_members)
    control_counts = counts(control_members)
    levels = sorted(set(case_counts) | set(control_counts))

    notes: list[str] = []
    for level in levels:
        in_case = case_counts.get(level, 0)
        in_control = control_counts.get(level, 0)
        if in_case == 0 or in_control == 0:
            notes.append(
                f"Level '{level}' appears in only one group "
                f"(case={in_case}, control={in_control}); it cannot be estimated "
                f"as a {attribute} effect."
            )
    return {
        "variable": attribute, "levels": levels,
        "case": case_counts, "control": control_counts, "notes": notes,
    }


@dataclass
class CohortDiagnostics:
    continuous: list[BalanceReport] = field(default_factory=list)
    categorical: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "continuous": [r.to_dict() for r in self.continuous],
            "categorical": self.categorical,
            "warnings": self.warnings,
        }


def diagnose(members: Sequence[CohortMember]) -> CohortDiagnostics:
    """Full balance report used by the Cohort page and the research report."""
    diagnostics = CohortDiagnostics()
    for variable in ("age", "mean_fd", "usable_frame_fraction"):
        report = continuous_balance(members, variable)
        diagnostics.continuous.append(report)
        if report.note:
            diagnostics.warnings.append(f"{variable}: {report.note}")

    for attribute in ("sex", "site", "scanner"):
        summary = categorical_balance(members, attribute)
        diagnostics.categorical.append(summary)
        diagnostics.warnings.extend(summary["notes"])

    case, control = _split(members)
    if case and control:
        ratio = max(len(case), len(control)) / min(len(case), len(control))
        if ratio > 3.0:
            diagnostics.warnings.append(
                f"Group sizes differ by {ratio:.1f}x ({len(case)} vs {len(control)})."
            )
    return diagnostics


def propose_matched_subset(members: Sequence[CohortMember], *,
                           caliper_age_years: float = 2.0,
                           caliper_fd_mm: float = 0.1,
                           require_same_site: bool = True) -> dict:
    """Greedy 1:1 nearest-neighbour matching on age / motion / site.

    Exploratory only: the returned subset is labelled as such and never becomes
    the primary analysis.
    """
    case_members, control_members = _split(members)
    available = list(control_members)
    pairs: list[tuple[str, str]] = []
    unmatched: list[str] = []

    def distance(a: CohortMember, b: CohortMember) -> float | None:
        if require_same_site and (a.site or "") != (b.site or ""):
            return None
        if a.age is None or b.age is None:
            return None
        age_gap = abs(a.age - b.age)
        if age_gap > caliper_age_years:
            return None
        fd_gap = 0.0
        if a.mean_fd is not None and b.mean_fd is not None:
            fd_gap = abs(a.mean_fd - b.mean_fd)
            if fd_gap > caliper_fd_mm:
                return None
        return age_gap / max(caliper_age_years, 1e-6) + fd_gap / max(caliper_fd_mm, 1e-6)

    for case_member in sorted(case_members, key=lambda m: (m.age is None, m.age or 0.0)):
        scored = [(distance(case_member, c), c) for c in available]
        scored = [(d, c) for d, c in scored if d is not None]
        if not scored:
            unmatched.append(case_member.subject_id)
            continue
        _, best = min(scored, key=lambda item: item[0])
        available.remove(best)
        pairs.append((case_member.subject_id, best.subject_id))

    return {
        "tier": "EXPLORATORY",
        "n_pairs": len(pairs),
        "n_unmatched_cases": len(unmatched),
        "pairs": pairs,
        "unmatched_case_subject_ids": unmatched,
        "calipers": {
            "age_years": caliper_age_years, "fd_mm": caliper_fd_mm,
            "same_site": require_same_site,
        },
        "note": (
            "Matching is a sensitivity analysis. The primary result remains the "
            "covariate-adjusted model on the full eligible cohort."
        ),
    }
