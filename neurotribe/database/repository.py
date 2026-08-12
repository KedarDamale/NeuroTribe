"""Small helper layer over the ORM used by services, jobs and the API."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from neurotribe.database.enums import (
    AssetStatus, BlockerKind, BlockerSeverity, JobState, StageState,
)
from neurotribe.database.models import (
    Artifact, AuditEvent, Blocker, DataAsset, Job, PipelineStage, Subject,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------

def record_audit(session: Session, action: str, *, actor: str = "autopilot",
                 entity_type: str | None = None, entity_id: str | None = None,
                 summary: str | None = None, payload: dict[str, Any] | None = None) -> AuditEvent:
    event = AuditEvent(
        action=action, actor=actor, entity_type=entity_type, entity_id=entity_id,
        summary=summary, payload=payload or {},
    )
    session.add(event)
    return event


# --------------------------------------------------------------------------
# Blockers
# --------------------------------------------------------------------------

def raise_blocker(session: Session, kind: BlockerKind | str, title: str, description: str,
                  *, severity: BlockerSeverity | str = BlockerSeverity.EXTERNAL,
                  required_action: str | None = None, reference_url: str | None = None,
                  blocks_stages: Sequence[str] | None = None,
                  context: dict[str, Any] | None = None) -> Blocker:
    """Create or refresh an active blocker (idempotent on kind+title)."""
    kind_value = kind.value if isinstance(kind, BlockerKind) else kind
    severity_value = severity.value if isinstance(severity, BlockerSeverity) else severity

    existing = session.execute(
        select(Blocker).where(Blocker.kind == kind_value, Blocker.title == title)
    ).scalar_one_or_none()

    if existing is None:
        existing = Blocker(kind=kind_value, title=title)
        session.add(existing)
        record_audit(session, "blocker.raised", entity_type="blocker", summary=title,
                     payload={"kind": kind_value})
    elif not existing.active:
        record_audit(session, "blocker.reraised", entity_type="blocker",
                     entity_id=existing.id, summary=title)

    existing.description = description
    existing.severity = severity_value
    existing.required_action = required_action
    existing.reference_url = reference_url
    existing.blocks_stages = list(blocks_stages or [])
    existing.context = context or {}
    existing.active = True
    existing.resolved_at = None
    return existing


def clear_blocker(session: Session, kind: BlockerKind | str, title: str | None = None) -> int:
    """Deactivate matching blockers. Returns how many were cleared."""
    kind_value = kind.value if isinstance(kind, BlockerKind) else kind
    stmt = select(Blocker).where(Blocker.kind == kind_value, Blocker.active.is_(True))
    if title is not None:
        stmt = stmt.where(Blocker.title == title)
    cleared = 0
    for blocker in session.execute(stmt).scalars():
        blocker.active = False
        blocker.resolved_at = _now()
        cleared += 1
        record_audit(session, "blocker.cleared", entity_type="blocker",
                     entity_id=blocker.id, summary=blocker.title)
    return cleared


def active_blockers(session: Session) -> list[Blocker]:
    return list(
        session.execute(
            select(Blocker).where(Blocker.active.is_(True)).order_by(Blocker.severity, Blocker.created_at)
        ).scalars()
    )


# --------------------------------------------------------------------------
# Pipeline stages
# --------------------------------------------------------------------------

def get_stage(session: Session, key: str) -> PipelineStage | None:
    return session.execute(select(PipelineStage).where(PipelineStage.key == key)).scalar_one_or_none()


def upsert_stage(session: Session, key: str, label: str, phase: int, order: int,
                 depends_on: Sequence[str] = (), max_attempts: int = 3) -> PipelineStage:
    stage = get_stage(session, key)
    if stage is None:
        stage = PipelineStage(key=key, label=label, phase=phase, order=order,
                              depends_on=list(depends_on), max_attempts=max_attempts)
        session.add(stage)
    else:
        stage.label = label
        stage.phase = phase
        stage.order = order
        stage.depends_on = list(depends_on)
        stage.max_attempts = max_attempts
    return stage


def set_stage_state(session: Session, key: str, state: StageState, *, detail: str | None = None,
                    progress: float | None = None, error: str | None = None,
                    result: dict[str, Any] | None = None) -> PipelineStage:
    stage = get_stage(session, key)
    if stage is None:
        raise KeyError(f"Unknown pipeline stage: {key}")

    previous = stage.state
    stage.state = state.value
    if detail is not None:
        stage.detail = detail
    if progress is not None:
        stage.progress = max(0.0, min(1.0, progress))
    if result is not None:
        stage.result = result

    if state is StageState.RUNNING and previous != StageState.RUNNING.value:
        stage.started_at = _now()
        stage.attempts += 1
    if state in (StageState.DONE, StageState.SKIPPED):
        stage.finished_at = _now()
        stage.progress = 1.0
        stage.last_error = None
        stage.next_attempt_at = None
    if state in (StageState.FAILED_RETRYABLE, StageState.FAILED_FINAL):
        stage.finished_at = _now()
        stage.last_error = error

    if previous != stage.state:
        record_audit(session, "stage.transition", entity_type="pipeline_stage", entity_id=stage.key,
                     summary=f"{previous} -> {stage.state}",
                     payload={"detail": detail, "error": error})
    return stage


def schedule_retry(session: Session, key: str, delay_sec: float, error: str) -> PipelineStage:
    stage = get_stage(session, key)
    if stage is None:
        raise KeyError(f"Unknown pipeline stage: {key}")
    if stage.attempts >= stage.max_attempts:
        return set_stage_state(session, key, StageState.FAILED_FINAL, error=error,
                               detail=f"Exhausted {stage.max_attempts} attempts")
    stage.next_attempt_at = _now() + timedelta(seconds=delay_sec)
    return set_stage_state(session, key, StageState.FAILED_RETRYABLE, error=error,
                           detail=f"Retry in {int(delay_sec)}s (attempt {stage.attempts}/{stage.max_attempts})")


def all_stages(session: Session) -> list[PipelineStage]:
    return list(
        session.execute(select(PipelineStage).order_by(PipelineStage.phase, PipelineStage.order)).scalars()
    )


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------

def create_job(session: Session, name: str, kind: str, *, stage_key: str | None = None,
               subject_external_id: str | None = None, payload: dict[str, Any] | None = None) -> Job:
    job = Job(name=name, kind=kind, stage_key=stage_key,
              subject_external_id=subject_external_id, payload=payload or {})
    session.add(job)
    return job


def finish_job(session: Session, job: Job, state: JobState, *, message: str | None = None,
               error: str | None = None) -> Job:
    job.state = state.value
    job.finished_at = _now()
    if job.started_at is not None:
        started = job.started_at
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        job.elapsed_sec = (job.finished_at - started).total_seconds()
    job.message = message
    job.error_message = error
    job.progress = 1.0 if state in (JobState.SUCCEEDED, JobState.CACHED) else job.progress
    return job


def recent_jobs(session: Session, limit: int = 100) -> list[Job]:
    return list(session.execute(select(Job).order_by(Job.created_at.desc()).limit(limit)).scalars())


# --------------------------------------------------------------------------
# Assets & artifacts
# --------------------------------------------------------------------------

def find_assets(session: Session, kind: str | None = None,
                status: AssetStatus | str | None = None) -> list[DataAsset]:
    stmt = select(DataAsset)
    if kind:
        stmt = stmt.where(DataAsset.kind == kind)
    if status:
        stmt = stmt.where(DataAsset.status == (status.value if isinstance(status, AssetStatus) else status))
    return list(session.execute(stmt.order_by(DataAsset.kind, DataAsset.path)).scalars())


def register_artifact(session: Session, kind: str, label: str, path: str, *,
                      media_type: str | None = None, sha256: str | None = None,
                      size_bytes: int | None = None, subject_external_id: str | None = None,
                      group_run_id: str | None = None, tier: str | None = None,
                      provenance: dict[str, Any] | None = None) -> Artifact:
    artifact = Artifact(
        kind=kind, label=label, path=path, media_type=media_type, sha256=sha256,
        size_bytes=size_bytes, subject_external_id=subject_external_id,
        group_run_id=group_run_id, tier=tier, provenance=provenance or {},
    )
    session.add(artifact)
    return artifact


def get_subject(session: Session, external_id: str) -> Subject | None:
    return session.execute(
        select(Subject).where(Subject.external_id == external_id)
    ).scalar_one_or_none()


def get_or_create_subject(session: Session, external_id: str, **fields: Any) -> Subject:
    subject = get_subject(session, external_id)
    if subject is None:
        subject = Subject(external_id=external_id, **fields)
        session.add(subject)
        session.flush()
    else:
        for key, value in fields.items():
            if value is not None:
                setattr(subject, key, value)
    return subject
