"""Report artefacts: listing, download and on-demand generation."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from neurotribe.config import Settings
from neurotribe.database.models import Artifact

from apps.api.deps import get_db, get_settings

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("")
def list_reports(kind: str | None = None, tier: str | None = None,
                 limit: int = 100, session: Session = Depends(get_db)) -> dict:
    stmt = select(Artifact).order_by(Artifact.created_at.desc())
    if kind:
        stmt = stmt.where(Artifact.kind == kind)
    if tier:
        stmt = stmt.where(Artifact.tier == tier)
    items = list(session.execute(stmt.limit(limit)).scalars())
    return {
        "artifacts": [
            {
                "id": a.id, "kind": a.kind, "label": a.label, "tier": a.tier,
                "media_type": a.media_type, "size_bytes": a.size_bytes,
                "sha256": a.sha256,
                "created_at": a.created_at.isoformat() if a.created_at else None,
                "exists": bool(a.path and Path(a.path).exists()),
                "download_url": f"/api/reports/{a.id}/download",
                "provenance": a.provenance,
            }
            for a in items
        ],
        "count": len(items),
    }


@router.get("/{artifact_id}/download")
def download(artifact_id: str, session: Session = Depends(get_db)) -> FileResponse:
    artifact = session.get(Artifact, artifact_id)
    if artifact is None:
        raise HTTPException(404, "Artifact not found")
    path = Path(artifact.path)
    if not path.exists():
        raise HTTPException(404, "Artifact file is missing from disk")
    return FileResponse(path, media_type=artifact.media_type or "application/octet-stream",
                        filename=path.name)


@router.post("/generate")
def generate(session: Session = Depends(get_db),
             settings: Settings = Depends(get_settings)) -> dict:
    from neurotribe.reporting.report import generate_all

    return generate_all(session, settings)


@router.get("/provenance")
def provenance(tier: str = "PRIMARY", session: Session = Depends(get_db),
               settings: Settings = Depends(get_settings)) -> dict:
    """Current reproducibility manifest, even before a report is rendered."""
    from neurotribe.database.enums import AnalysisTier
    from neurotribe.reporting.report import build_context

    try:
        analysis_tier = AnalysisTier(tier)
    except ValueError:
        raise HTTPException(400, f"Unknown tier: {tier}")

    context = build_context(session, settings, analysis_tier)
    return {"provenance": context.provenance, "limitations": context.limitations,
            "generated_at": context.generated_at}
