"""Logs and the append-only audit trail.

The dashboard shows INFO / WARNING / ERROR. Full DEBUG output stays in files and
is downloadable rather than streamed into the UI.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from neurotribe.config import Settings
from neurotribe.database.models import AuditEvent

from apps.api.deps import get_db, get_settings

router = APIRouter(prefix="/logs", tags=["logs"])

LEVELS = ("INFO", "WARNING", "ERROR")


@router.get("")
def tail_logs(level: str | None = None, limit: int = 200,
              settings: Settings = Depends(get_settings)) -> dict:
    """Recent structured log records at INFO and above."""
    log_path = settings.paths.data / "logs" / "neurotribe.log"
    if not log_path.exists():
        return {"entries": [], "note": "No log file yet.", "path": str(log_path)}

    wanted = {level.upper()} if level else set(LEVELS)
    entries: list[dict] = []
    # Read the tail without loading a large file into memory.
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()[-5000:]

    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("level") not in wanted:
            continue
        entries.append({
            "ts": record.get("ts"), "level": record.get("level"),
            "logger": record.get("logger"), "message": record.get("message"),
            "context": {k: v for k, v in record.items()
                        if k not in {"ts", "level", "logger", "message"}},
        })
        if len(entries) >= limit:
            break

    return {"entries": entries, "count": len(entries), "levels": list(LEVELS)}


@router.get("/download")
def download(settings: Settings = Depends(get_settings)) -> FileResponse:
    log_path = settings.paths.data / "logs" / "neurotribe.log"
    if not log_path.exists():
        raise HTTPException(404, "No log file available")
    return FileResponse(log_path, media_type="text/plain", filename="neurotribe.log")


@router.get("/audit")
def audit(action: str | None = None, entity_type: str | None = None,
          limit: int = 200, session: Session = Depends(get_db)) -> dict:
    """Append-only audit trail: every state transition and exclusion decision."""
    stmt = select(AuditEvent).order_by(AuditEvent.at.desc())
    if action:
        stmt = stmt.where(AuditEvent.action == action)
    if entity_type:
        stmt = stmt.where(AuditEvent.entity_type == entity_type)
    items = list(session.execute(stmt.limit(limit)).scalars())
    return {
        "events": [
            {
                "id": e.id, "at": e.at.isoformat() if e.at else None,
                "actor": e.actor, "action": e.action,
                "entity_type": e.entity_type, "entity_id": e.entity_id,
                "summary": e.summary, "payload": e.payload,
            }
            for e in items
        ],
        "count": len(items),
    }
