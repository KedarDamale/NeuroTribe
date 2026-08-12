"""Dashboard: pipeline state, blockers and headline counts."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from neurotribe import RESEARCH_DISCLAIMER
from neurotribe.config import Settings
from neurotribe.database.enums import MovieKey, StageState
from neurotribe.database.models import (
    Cohort, DataAsset, GroupAnalysisRun, Scan, Stimulus, Subject, SubjectComparison,
    SystemProbe, TribeRun,
)
from neurotribe.database.repository import active_blockers, all_stages
from neurotribe.jobs.stages import STAGE_BY_KEY, groups as stage_groups

from apps.api.deps import get_db, get_settings

router = APIRouter(tags=["dashboard"])


def _count(session: Session, model, *where) -> int:
    stmt = select(func.count()).select_from(model)
    for clause in where:
        stmt = stmt.where(clause)
    return int(session.execute(stmt).scalar_one())


@router.get("/dashboard")
def dashboard(session: Session = Depends(get_db),
              settings: Settings = Depends(get_settings)) -> dict:
    """Everything the home screen needs in one round trip."""
    n_subjects = _count(session, Subject)
    n_movie_subjects = _count(session, Subject, Subject.has_movie_bold.is_(True))
    n_phenotype = _count(session, Subject, Subject.has_phenotype.is_(True))
    n_movie_scans = _count(session, Scan, Scan.movie != MovieKey.UNKNOWN.value)

    stimulus = session.execute(
        select(Stimulus).where(Stimulus.validated.is_(True))
    ).scalars().first()
    tribe_run = session.execute(
        select(TribeRun).where(TribeRun.status == "DONE")
        .order_by(TribeRun.created_at.desc())
    ).scalars().first()
    probe = session.execute(
        select(SystemProbe).order_by(SystemProbe.created_at.desc())
    ).scalars().first()

    stages = all_stages(session)
    by_state: dict[str, int] = {}
    for stage in stages:
        by_state[stage.state] = by_state.get(stage.state, 0) + 1

    blockers = active_blockers(session)
    external = [b for b in blockers if b.severity == "EXTERNAL"]

    done = by_state.get(StageState.DONE.value, 0)
    overall = done / len(stages) if stages else 0.0

    if any(s.state == StageState.RUNNING.value for s in stages):
        pipeline_state = "RUNNING"
    elif external:
        pipeline_state = f"BLOCKED BY {len(external)} EXTERNAL ITEM" + ("S" if len(external) != 1 else "")
    elif done == len(stages) and stages:
        pipeline_state = "COMPLETE"
    else:
        pipeline_state = "PENDING"

    cohort = session.execute(
        select(Cohort).where(Cohort.tier == "PRIMARY").order_by(Cohort.updated_at.desc())
    ).scalars().first()
    group_run = session.execute(
        select(GroupAnalysisRun).order_by(GroupAnalysisRun.created_at.desc())
    ).scalars().first()

    return {
        "disclaimer": RESEARCH_DISCLAIMER,
        "profile": settings.profile,
        "analysis_config_hash": settings.analysis_config_hash,
        "cards": {
            "dataset": "HBN",
            "subjects_indexed": n_subjects,
            "movie_fmri_subjects": n_movie_subjects,
            "movie_scans": n_movie_scans,
            "adhd_labels": "READY" if n_phenotype else "WAITING",
            "n_phenotype_subjects": n_phenotype,
            "tribe_model": (tribe_run.backend.upper() if tribe_run else
                            ("READY" if _tribe_installed() else "MISSING")),
            "stimulus": stimulus.label if stimulus else "MISSING",
            "pipeline": pipeline_state,
            "overall_progress": round(overall, 4),
            "n_valid_comparisons": _count(session, SubjectComparison,
                                          SubjectComparison.valid.is_(True)),
            "n_assets": _count(session, DataAsset),
        },
        "pipeline": {
            "groups": [
                {
                    "name": group.name,
                    "stages": [
                        _stage_payload(next(s for s in stages if s.key == key))
                        for key in group.stages
                        if any(s.key == key for s in stages)
                    ],
                }
                for group in stage_groups()
            ],
            "by_state": by_state,
            "n_stages": len(stages),
        },
        "blockers": [
            {
                "id": b.id, "kind": b.kind, "severity": b.severity, "title": b.title,
                "description": b.description, "required_action": b.required_action,
                "reference_url": b.reference_url, "blocks_stages": b.blocks_stages,
                "context": b.context,
            }
            for b in blockers
        ],
        "system": probe.payload if probe else None,
        "cohort": {
            "n_case": cohort.n_case, "n_control": cohort.n_control,
            "n_excluded": cohort.n_excluded, "warnings": cohort.warnings,
            "movie": cohort.movie,
        } if cohort else None,
        "latest_group_run": {
            "id": group_run.id, "name": group_run.name, "tier": group_run.tier,
            "status": group_run.status, "sanity_passed": group_run.sanity_passed,
            "summary": group_run.results_summary,
        } if group_run else None,
    }


def _stage_payload(stage) -> dict:
    spec = STAGE_BY_KEY.get(stage.key)
    return {
        "key": stage.key, "label": stage.label, "phase": stage.phase,
        "state": stage.state, "detail": stage.detail, "progress": stage.progress,
        "attempts": stage.attempts, "max_attempts": stage.max_attempts,
        "last_error": stage.last_error, "depends_on": stage.depends_on,
        "description": spec.description if spec else "",
        "result": stage.result,
    }


def _tribe_installed() -> bool:
    from neurotribe.tribe.model import probe_tribe

    return probe_tribe()[0]


@router.get("/pipeline")
def pipeline(session: Session = Depends(get_db)) -> dict:
    return {"stages": [_stage_payload(s) for s in all_stages(session)]}


@router.post("/pipeline/tick")
def trigger_tick(settings: Settings = Depends(get_settings)) -> dict:
    """Run one Autopilot iteration immediately (inline, for the UI button)."""
    from neurotribe.jobs.autopilot import tick

    return tick(settings).to_dict()


@router.post("/pipeline/stages/{key}/retry")
def retry_stage(key: str, settings: Settings = Depends(get_settings)) -> dict:
    from neurotribe.database.base import session_scope
    from neurotribe.jobs.autopilot import ensure_stages, run_stage

    with session_scope() as session:
        ensure_stages(session)
        outcome = run_stage(session, settings, key)
    return {"stage": key, "state": outcome.state.value, "detail": outcome.detail,
            "error": outcome.error}


@router.get("/blockers")
def blockers(session: Session = Depends(get_db)) -> dict:
    items = active_blockers(session)
    return {
        "external": [_blocker(b) for b in items if b.severity == "EXTERNAL"],
        "actionable": [_blocker(b) for b in items if b.severity == "ACTIONABLE"],
        "info": [_blocker(b) for b in items if b.severity == "INFO"],
    }


def _blocker(b) -> dict:
    return {
        "id": b.id, "kind": b.kind, "severity": b.severity, "title": b.title,
        "description": b.description, "required_action": b.required_action,
        "reference_url": b.reference_url, "blocks_stages": b.blocks_stages,
        "context": b.context, "raised_at": b.created_at.isoformat() if b.created_at else None,
    }
