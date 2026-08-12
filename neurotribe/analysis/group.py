"""Group analysis: Confirmed ADHD vs the comparison cohort.

Primary outcomes are the pre-specified ROI and network measures. Vertex-level
maps and alternative comparison groups are EXPLORATORY and are labelled as such
everywhere they appear.

The model is always covariate-adjusted:

    ROI_metric ~ ADHD + age + sex + site + mean_FD
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sqlalchemy import select
from sqlalchemy.orm import Session

from neurotribe.alignment import validate as sanity
from neurotribe.analysis import statistics as stats_mod
from neurotribe.config import Settings
from neurotribe.database.enums import AnalysisTier, CohortGroup
from neurotribe.database.models import (
    Cohort, CohortMember, GroupAnalysisRun, GroupResult, NetworkMetric, RoiMetric,
    Subject, SubjectComparison,
)
from neurotribe.database.repository import record_audit
from neurotribe.logging_setup import get_logger

log = get_logger(__name__)

METRICS = ("agreement_r", "mad", "residual_variance")


@dataclass
class UnitObservations:
    """Per-subject values for one testable unit (ROI or network)."""

    unit_type: str
    unit_name: str
    unit_index: int | None
    network: str | None
    values: dict[str, list[float | None]] = field(default_factory=dict)
    subject_ids: list[str] = field(default_factory=list)


@dataclass
class GroupAnalysisResult:
    run_id: str
    tier: str
    n_case: int
    n_control: int
    n_units: int
    n_significant: int
    sanity_passed: bool
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    top_results: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id, "tier": self.tier, "n_case": self.n_case,
            "n_control": self.n_control, "n_units": self.n_units,
            "n_significant": self.n_significant, "sanity_passed": self.sanity_passed,
            "warnings": self.warnings, "failures": self.failures,
            "top_results": self.top_results,
        }


def _collect(session: Session, cohort: Cohort, control_group: CohortGroup
             ) -> tuple[list[dict], list[str]]:
    """Gather one valid comparison per included cohort member."""
    rows: list[dict] = []
    warnings: list[str] = []
    seen: set[str] = set()

    for member in cohort.members:
        if not member.included:
            continue
        if member.group not in (CohortGroup.CONFIRMED_ADHD.value, control_group.value):
            continue

        subject = session.get(Subject, member.subject_id)
        if subject is None:
            continue
        if subject.external_id in seen:
            warnings.append(f"Duplicate participant {subject.external_id} skipped.")
            continue

        comparison = session.execute(
            select(SubjectComparison)
            .where(SubjectComparison.subject_id == member.subject_id,
                   SubjectComparison.movie == cohort.movie,
                   SubjectComparison.valid.is_(True))
            .order_by(SubjectComparison.created_at.desc())
        ).scalars().first()
        if comparison is None:
            warnings.append(
                f"{subject.external_id}: no valid subject comparison; excluded from the model."
            )
            continue
        if comparison.is_approximate:
            warnings.append(
                f"{subject.external_id}: approximate surfaces - excluded from the final analysis."
            )
            continue

        seen.add(subject.external_id)
        rows.append({
            "subject_id": subject.id,
            "external_id": subject.external_id,
            "comparison_id": comparison.id,
            "adhd": 1 if member.group == CohortGroup.CONFIRMED_ADHD.value else 0,
            "age": member.age, "sex": member.sex, "site": member.site,
            "mean_fd": member.mean_fd,
        })
    return rows, warnings


def _units(session: Session, rows: list[dict]) -> list[UnitObservations]:
    """Pivot per-subject ROI/network metrics into per-unit observation vectors."""
    comparison_ids = [r["comparison_id"] for r in rows]
    if not comparison_ids:
        return []

    index_of = {r["comparison_id"]: i for i, r in enumerate(rows)}
    n = len(rows)
    units: dict[tuple[str, str], UnitObservations] = {}

    roi_rows = session.execute(
        select(RoiMetric).where(RoiMetric.comparison_id.in_(comparison_ids))
    ).scalars()
    for metric in roi_rows:
        key = ("roi", metric.roi_name)
        unit = units.get(key)
        if unit is None:
            unit = UnitObservations("roi", metric.roi_name, metric.roi_index, metric.network,
                                    {m: [None] * n for m in METRICS},
                                    [r["external_id"] for r in rows])
            units[key] = unit
        position = index_of[metric.comparison_id]
        unit.values["agreement_r"][position] = metric.agreement_r
        unit.values["mad"][position] = metric.mad
        unit.values["residual_variance"][position] = metric.residual_variance

    network_rows = session.execute(
        select(NetworkMetric).where(NetworkMetric.comparison_id.in_(comparison_ids))
    ).scalars()
    for metric in network_rows:
        key = ("network", metric.network)
        unit = units.get(key)
        if unit is None:
            unit = UnitObservations("network", metric.network, None, metric.network,
                                    {m: [None] * n for m in METRICS},
                                    [r["external_id"] for r in rows])
            units[key] = unit
        position = index_of[metric.comparison_id]
        unit.values["agreement_r"][position] = metric.agreement_r
        unit.values["mad"][position] = metric.mad
        unit.values["residual_variance"][position] = metric.residual_variance

    globals_unit = UnitObservations("global", "whole_cortex", None, None,
                                    {m: [None] * n for m in METRICS},
                                    [r["external_id"] for r in rows])
    for row in rows:
        comparison = session.get(SubjectComparison, row["comparison_id"])
        if comparison is None:
            continue
        position = index_of[row["comparison_id"]]
        globals_unit.values["agreement_r"][position] = comparison.global_agreement_r
        globals_unit.values["mad"][position] = comparison.global_mad
        globals_unit.values["residual_variance"][position] = comparison.global_residual_variance

    ordered = sorted(units.values(), key=lambda u: (u.unit_type, u.unit_name))
    return [globals_unit, *ordered]


def run(session: Session, settings: Settings, cohort: Cohort,
        *, tier: AnalysisTier = AnalysisTier.PRIMARY) -> GroupAnalysisResult:
    """Execute the covariate-adjusted group contrast for a cohort."""
    control_group = (CohortGroup.NO_DIAGNOSIS_GIVEN if tier is AnalysisTier.PRIMARY
                     else CohortGroup.NON_ADHD_COMPARISON)

    analysis_run = GroupAnalysisRun(
        cohort_id=cohort.id, tier=tier.value,
        name=f"{'Primary' if tier is AnalysisTier.PRIMARY else 'Exploratory'}: "
             f"Confirmed ADHD vs {control_group.display} ({cohort.movie})",
        case_group=CohortGroup.CONFIRMED_ADHD.value, control_group=control_group.value,
        model_formula=str(settings.get("analysis.group.model")),
        covariates=list(settings.get("analysis.group.covariates", [])),
        correction=str(settings.get("analysis.group.multiple_comparisons", "fdr_bh")),
        alpha=float(settings.get("analysis.group.alpha", 0.05)),
        status="RUNNING",
    )
    session.add(analysis_run)
    session.flush()

    rows, warnings = _collect(session, cohort, control_group)
    n_case = sum(r["adhd"] for r in rows)
    n_control = len(rows) - n_case
    analysis_run.n_case = n_case
    analysis_run.n_control = n_control

    units = _units(session, rows)
    alpha = float(settings.get("analysis.group.alpha", 0.05))
    n_boot = int(settings.get("analysis.group.bootstrap_ci", 2000))

    covariate_names = list(settings.get("analysis.group.covariates", []))
    covariates = {name: [r.get(name) for r in rows] for name in covariate_names}
    covariates_present = {
        name: any(v is not None for v in values) for name, values in covariates.items()
    }

    computed: list[dict] = []
    design: stats_mod.DesignMatrix | None = None

    if rows and units:
        try:
            design = stats_mod.build_design([r["adhd"] for r in rows], covariates)
            warnings.extend(design.notes)
            if design.dropped:
                warnings.append(f"Dropped model terms: {', '.join(design.dropped)}")
        except ValueError as exc:
            warnings.append(f"Design matrix could not be built: {exc}")

    if design is not None:
        adhd_flags = np.array([r["adhd"] for r in rows], dtype=bool)
        for unit in units:
            for metric in METRICS:
                raw = unit.values[metric]
                y = np.array([np.nan if v is None else float(v) for v in raw], dtype=float)
                finite = np.isfinite(y)
                if finite.sum() < design.n_terms + 2:
                    continue
                if adhd_flags[finite].sum() < 2 or (~adhd_flags[finite]).sum() < 2:
                    continue

                sub_design = stats_mod.DesignMatrix(
                    matrix=design.matrix[finite], names=list(design.names),
                )
                try:
                    fit = stats_mod.ols(y[finite], sub_design)
                except (ValueError, np.linalg.LinAlgError) as exc:
                    warnings.append(f"{unit.unit_name}/{metric}: model failed ({exc})")
                    continue

                term = fit.term("adhd")
                case_values = y[finite & adhd_flags]
                control_values = y[finite & ~adhd_flags]
                ci_low, ci_high = stats_mod.coefficient_ci(
                    term["beta"], term["se"], fit.residual_df, alpha,
                )
                computed.append({
                    "unit_type": unit.unit_type, "unit_name": unit.unit_name,
                    "unit_index": unit.unit_index, "network": unit.network,
                    "metric": metric,
                    "mean_case": float(case_values.mean()) if case_values.size else None,
                    "mean_control": float(control_values.mean()) if control_values.size else None,
                    "sd_case": float(case_values.std(ddof=1)) if case_values.size > 1 else None,
                    "sd_control": (float(control_values.std(ddof=1))
                                   if control_values.size > 1 else None),
                    "beta_adhd": term["beta"], "se_adhd": term["se"],
                    "t_stat": term["t"], "p_value": term["p"],
                    "effect_size": stats_mod.cohens_d(case_values, control_values),
                    "ci_low": ci_low, "ci_high": ci_high,
                    "n_case": int(case_values.size), "n_control": int(control_values.size),
                })

    # FDR is applied within each (unit_type, metric) family, not across everything.
    families: dict[tuple[str, str], list[int]] = {}
    for index, record in enumerate(computed):
        families.setdefault((record["unit_type"], record["metric"]), []).append(index)
    for indices in families.values():
        p_values = np.array([computed[i]["p_value"] for i in indices], dtype=float)
        q_values = stats_mod.fdr_bh(p_values)
        for position, index in enumerate(indices):
            computed[index]["q_value"] = (float(q_values[position])
                                          if np.isfinite(q_values[position]) else None)

    duplicates = [
        external for external in {r["external_id"] for r in rows}
        if [r["external_id"] for r in rows].count(external) > 1
    ]
    sanity_report = sanity.check_group_analysis(
        n_case=n_case, n_control=n_control, n_units_tested=len(computed),
        p_values=np.array([r["p_value"] for r in computed], dtype=float) if computed else None,
        settings=settings, duplicate_subjects=duplicates,
        covariates_present=covariates_present,
    )
    sanity_report.warnings.extend(warnings)

    for record in computed:
        session.add(GroupResult(run_id=analysis_run.id, **{
            k: record.get(k) for k in (
                "unit_type", "unit_name", "unit_index", "network", "metric",
                "mean_case", "mean_control", "sd_case", "sd_control", "beta_adhd",
                "se_adhd", "t_stat", "p_value", "q_value", "effect_size",
                "ci_low", "ci_high", "n_case", "n_control",
            )
        }))

    significant = [
        r for r in computed
        if r.get("q_value") is not None and r["q_value"] < alpha
        and r["unit_type"] in ("roi", "network")
    ]
    top = sorted(
        [r for r in computed if r["unit_type"] in ("network", "global")],
        key=lambda r: (r.get("q_value") if r.get("q_value") is not None else 1.0),
    )[:10]

    analysis_run.status = "DONE" if sanity_report.valid else "INVALID"
    analysis_run.sanity_passed = sanity_report.valid
    analysis_run.sanity_report = sanity_report.to_dict()
    analysis_run.results_summary = {
        "n_units_tested": len(computed),
        "n_significant_fdr": len(significant),
        "alpha": alpha,
        "bootstrap_ci": n_boot,
        "model_terms": design.names if design else [],
        "dropped_terms": design.dropped if design else [],
        "tier": tier.value,
        "warnings": warnings[:50],
    }
    analysis_run.provenance = build_provenance(session, settings, cohort, analysis_run)

    results_dir = settings.paths.analysis / "group" / analysis_run.id
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "results.json").write_text(
        json.dumps({"results": computed, "summary": analysis_run.results_summary,
                    "provenance": analysis_run.provenance}, indent=2, default=str),
        encoding="utf-8",
    )
    analysis_run.results_path = str(results_dir / "results.json")

    record_audit(session, "analysis.group_completed", entity_type="group_analysis_run",
                 entity_id=analysis_run.id,
                 summary=f"{n_case} vs {n_control}, {len(significant)} significant",
                 payload=analysis_run.results_summary)
    log.info("Group analysis complete", extra={
        "tier": tier.value, "n_case": n_case, "n_control": n_control,
        "n_units": len(computed), "n_significant": len(significant),
        "valid": sanity_report.valid,
    })

    return GroupAnalysisResult(
        run_id=analysis_run.id, tier=tier.value, n_case=n_case, n_control=n_control,
        n_units=len(computed), n_significant=len(significant),
        sanity_passed=sanity_report.valid, warnings=sanity_report.warnings,
        failures=sanity_report.failures, top_results=top,
    )


def build_provenance(session: Session, settings: Settings, cohort: Cohort,
                     analysis_run: GroupAnalysisRun) -> dict:
    """Reproducibility manifest (specification section 58). Mandatory."""
    from neurotribe.database.models import TribeRun

    tribe_run = session.execute(
        select(TribeRun).where(TribeRun.movie == cohort.movie, TribeRun.status == "DONE")
        .order_by(TribeRun.created_at.desc())
    ).scalars().first()

    from neurotribe.database.models import Stimulus

    stimulus = session.execute(
        select(Stimulus).where(Stimulus.key == cohort.movie)
    ).scalar_one_or_none()

    from neurotribe import __version__

    return {
        "neurotribe_version": __version__,
        "profile": settings.profile,
        "tribe_commit": tribe_run.tribe_commit if tribe_run else None,
        "tribe_model": tribe_run.model_id if tribe_run else None,
        "tribe_model_revision": tribe_run.model_revision if tribe_run else None,
        "tribe_backend": tribe_run.backend if tribe_run else None,
        "fmriprep_version": str(settings.get("preprocessing.fmriprep.version_pin")),
        "fmriprep_image": str(settings.get("preprocessing.fmriprep.image")),
        "dataset_release": _dataset_release(session),
        "stimulus_sha256": stimulus.sha256 if stimulus else None,
        "stimulus_duration_sec": stimulus.duration_sec if stimulus else None,
        "cohort_hash": cohort.cohort_hash,
        "analysis_config_hash": settings.analysis_config_hash,
        "denoise_strategy": str(settings.get("preprocessing.denoise.strategy")),
        "global_signal_regression": bool(
            settings.get("preprocessing.denoise.global_signal_regression", False)
        ),
        "atlas": settings.get("surface.atlas"),
        "surface_space": str(settings.get("surface.space")),
        "hemi_order": list(tribe_run.hemi_order) if tribe_run and tribe_run.hemi_order
        else list(settings.get("surface.hemi_order", [])),
        "model_formula": analysis_run.model_formula,
        "multiple_comparisons": analysis_run.correction,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "research_use_only": True,
    }


def _dataset_release(session: Session) -> str | None:
    row = session.execute(
        select(Subject.release).where(Subject.release.is_not(None)).limit(1)
    ).scalar_one_or_none()
    return row


def load_results(analysis_run: GroupAnalysisRun) -> dict:
    if not analysis_run.results_path or not Path(analysis_run.results_path).exists():
        return {"results": [], "summary": analysis_run.results_summary or {}}
    return json.loads(Path(analysis_run.results_path).read_text(encoding="utf-8"))
