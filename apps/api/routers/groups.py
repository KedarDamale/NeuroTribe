"""Group analysis results: networks, ROIs and cortical effect maps."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from neurotribe.config import Settings
from neurotribe.database.enums import AnalysisTier, CohortGroup
from neurotribe.database.models import Cohort, GroupAnalysisRun, GroupResult

from apps.api.deps import get_db, get_settings

router = APIRouter(prefix="/groups", tags=["groups"])


@router.get("/runs")
def runs(session: Session = Depends(get_db)) -> dict:
    items = list(session.execute(
        select(GroupAnalysisRun).order_by(GroupAnalysisRun.created_at.desc()).limit(50)
    ).scalars())
    return {
        "runs": [
            {
                "id": r.id, "name": r.name, "tier": r.tier, "status": r.status,
                "case_group": r.case_group, "control_group": r.control_group,
                "n_case": r.n_case, "n_control": r.n_control,
                "model_formula": r.model_formula, "correction": r.correction,
                "alpha": r.alpha, "sanity_passed": r.sanity_passed,
                "summary": r.results_summary,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in items
        ]
    }


def _latest_run(session: Session, tier: str) -> GroupAnalysisRun | None:
    return session.execute(
        select(GroupAnalysisRun).where(GroupAnalysisRun.tier == tier)
        .order_by(GroupAnalysisRun.created_at.desc())
    ).scalars().first()


@router.get("/results")
def results(tier: str = "PRIMARY", unit_type: str | None = None,
            metric: str = "mad", run_id: str | None = None,
            session: Session = Depends(get_db)) -> dict:
    run = session.get(GroupAnalysisRun, run_id) if run_id else _latest_run(session, tier)
    if run is None:
        return {"available": False, "tier": tier,
                "reason": "No group analysis has completed yet."}

    stmt = select(GroupResult).where(GroupResult.run_id == run.id,
                                     GroupResult.metric == metric)
    if unit_type:
        stmt = stmt.where(GroupResult.unit_type == unit_type)
    rows = list(session.execute(stmt).scalars())

    def payload(r: GroupResult) -> dict:
        return {
            "unit_type": r.unit_type, "unit_name": r.unit_name,
            "unit_index": r.unit_index, "network": r.network, "metric": r.metric,
            "mean_case": r.mean_case, "mean_control": r.mean_control,
            "sd_case": r.sd_case, "sd_control": r.sd_control,
            "beta_adhd": r.beta_adhd, "se_adhd": r.se_adhd, "t_stat": r.t_stat,
            "p_value": r.p_value, "q_value": r.q_value,
            "effect_size": r.effect_size, "ci_low": r.ci_low, "ci_high": r.ci_high,
            "n_case": r.n_case, "n_control": r.n_control,
        }

    ordered = sorted(rows, key=lambda r: (r.q_value if r.q_value is not None else 1.0))
    alpha = run.alpha or 0.05
    control = (CohortGroup.NO_DIAGNOSIS_GIVEN if run.tier == "PRIMARY"
               else CohortGroup.NON_ADHD_COMPARISON)

    return {
        "available": True,
        "run": {
            "id": run.id, "name": run.name, "tier": run.tier, "status": run.status,
            "n_case": run.n_case, "n_control": run.n_control,
            "model_formula": run.model_formula, "correction": run.correction,
            "alpha": alpha, "sanity_passed": run.sanity_passed,
            "sanity_report": run.sanity_report, "summary": run.results_summary,
            "provenance": run.provenance,
            "case_label": CohortGroup.CONFIRMED_ADHD.display,
            "control_label": control.display,
        },
        "results": [payload(r) for r in ordered],
        "n_significant": sum(1 for r in rows if r.q_value is not None and r.q_value < alpha),
        "note": (
            "Effect sizes with confidence intervals and sample sizes are reported "
            "alongside FDR-adjusted q-values. Significance markers alone are never "
            "the whole result."
        ),
    }


@router.get("/effect-map")
def effect_map(tier: str = "PRIMARY", metric: str = "mad", format: str = "json",
               session: Session = Depends(get_db),
               settings: Settings = Depends(get_settings)):
    """Project ROI-level ADHD-minus-comparison effect sizes onto the surface.

    The map shows **effect size**, not p-values, exactly as specified.
    """
    run = _latest_run(session, tier)
    if run is None:
        raise HTTPException(404, "No group analysis has completed yet")

    from neurotribe.preprocessing.surfaces import load_parcellation

    parcellation = load_parcellation(settings)
    values = np.full(parcellation.labels.size, np.nan, dtype=np.float32)

    rows = list(session.execute(
        select(GroupResult).where(GroupResult.run_id == run.id,
                                  GroupResult.metric == metric,
                                  GroupResult.unit_type == "roi")
    ).scalars())

    by_name = {parcellation.names[i]: i for i in parcellation.parcel_indices()}
    painted = 0
    for row in rows:
        parcel_id = row.unit_index if row.unit_index is not None else by_name.get(row.unit_name)
        if parcel_id is None or row.effect_size is None:
            continue
        values[parcellation.labels == parcel_id] = np.float32(row.effect_size)
        painted += 1

    if format == "binary":
        return Response(
            content=values.tobytes(order="C"),
            media_type="application/octet-stream",
            headers={"X-Vertex-Count": str(values.size),
                     "X-Painted-Parcels": str(painted)},
        )

    finite = values[np.isfinite(values)]
    return {
        "tier": tier, "metric": metric, "run_id": run.id,
        "n_vertices": int(values.size), "n_painted_parcels": painted,
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "atlas": parcellation.to_dict(),
        "values": [None if not np.isfinite(v) else round(float(v), 5) for v in values],
        "legend": "ADHD minus comparison, Cohen's d",
    }


@router.get("/roi/{unit_name}")
def roi_detail(unit_name: str, tier: str = "PRIMARY", metric: str = "mad",
               session: Session = Depends(get_db)) -> dict:
    run = _latest_run(session, tier)
    if run is None:
        raise HTTPException(404, "No group analysis has completed yet")
    row = session.execute(
        select(GroupResult).where(GroupResult.run_id == run.id,
                                  GroupResult.unit_name == unit_name,
                                  GroupResult.metric == metric)
    ).scalars().first()
    if row is None:
        raise HTTPException(404, f"No result for '{unit_name}' / {metric}")
    return {
        "unit_name": row.unit_name, "unit_type": row.unit_type,
        "network": row.network, "metric": row.metric,
        "mean_case": row.mean_case, "mean_control": row.mean_control,
        "effect_size": row.effect_size, "ci_low": row.ci_low, "ci_high": row.ci_high,
        "p_value": row.p_value, "q_value": row.q_value,
        "n_case": row.n_case, "n_control": row.n_control,
        "beta_adhd": row.beta_adhd, "se_adhd": row.se_adhd,
        "model_formula": run.model_formula,
    }


@router.post("/run")
def run_group_analysis(tier: str = "PRIMARY", session: Session = Depends(get_db),
                       settings: Settings = Depends(get_settings)) -> dict:
    from neurotribe.analysis.group import run as execute

    try:
        analysis_tier = AnalysisTier(tier)
    except ValueError:
        raise HTTPException(400, f"Unknown tier: {tier}")

    cohort = session.execute(
        select(Cohort).where(Cohort.tier == tier).order_by(Cohort.updated_at.desc())
    ).scalars().first()
    if cohort is None:
        raise HTTPException(409, "No cohort available; build one first")

    return execute(session, settings, cohort, tier=analysis_tier).to_dict()
