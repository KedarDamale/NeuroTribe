"""Jobs page: progress, resource usage and per-job logs."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from neurotribe.database.enums import JobState
from neurotribe.database.models import Job

from apps.api.deps import get_db

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("")
def list_jobs(state: str | None = None, kind: str | None = None,
              limit: int = 100, session: Session = Depends(get_db)) -> dict:
    stmt = select(Job).order_by(Job.created_at.desc())
    if state:
        stmt = stmt.where(Job.state == state)
    if kind:
        stmt = stmt.where(Job.kind == kind)
    items = list(session.execute(stmt.limit(limit)).scalars())

    counts: dict[str, int] = {}
    for job in session.execute(select(Job)).scalars():
        counts[job.state] = counts.get(job.state, 0) + 1

    return {
        "jobs": [_job(j) for j in items],
        "count": len(items),
        "by_state": counts,
        "active": counts.get(JobState.RUNNING.value, 0),
    }


def _job(j: Job) -> dict:
    return {
        "id": j.id, "name": j.name, "kind": j.kind, "stage_key": j.stage_key,
        "subject_external_id": j.subject_external_id, "state": j.state,
        "progress": j.progress, "message": j.message,
        "started_at": j.started_at.isoformat() if j.started_at else None,
        "finished_at": j.finished_at.isoformat() if j.finished_at else None,
        "elapsed_sec": j.elapsed_sec, "eta_sec": j.eta_sec,
        "retry_count": j.retry_count, "cache_hit": j.cache_hit,
        "cpu_percent": j.cpu_percent, "mem_mb": j.mem_mb,
        "gpu_name": j.gpu_name, "disk_mb": j.disk_mb,
        "error_message": j.error_message,
        "has_log": bool(j.log_path and Path(j.log_path).exists()),
        "created_at": j.created_at.isoformat() if j.created_at else None,
    }


@router.get("/{job_id}")
def job_detail(job_id: str, session: Session = Depends(get_db)) -> dict:
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return {**_job(job), "payload": job.payload}


@router.get("/{job_id}/log", response_class=PlainTextResponse)
def job_log(job_id: str, tail: int = 2000, session: Session = Depends(get_db)) -> str:
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if not job.log_path or not Path(job.log_path).exists():
        raise HTTPException(404, "No log file for this job")
    lines = Path(job.log_path).read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-tail:])
