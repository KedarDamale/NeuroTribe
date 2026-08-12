"""Data Explorer: registered assets and their validation reports."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from neurotribe.config import Settings
from neurotribe.database.enums import AssetKind, AssetStatus, MovieKey
from neurotribe.database.models import DataAsset, Scan, Subject

from apps.api.deps import get_db, get_settings

router = APIRouter(prefix="/data", tags=["data"])

# Human-facing description of each source row in the Data Explorer table.
SOURCE_LABELS = {
    AssetKind.HBN_METADATA.value: "HBN release metadata",
    AssetKind.MRIQC_FUNCTIONAL.value: "MRIQC (functional)",
    AssetKind.MRIQC_ANATOMICAL.value: "MRIQC (anatomical)",
    AssetKind.BIDS_ROOT.value: "HBN BIDS repository",
    AssetKind.PHENOTYPE_CSV.value: "Phenotype (DUA-controlled)",
    AssetKind.STIMULUS_VIDEO.value: "Movie stimulus",
}


@router.get("/sources")
def sources(session: Session = Depends(get_db)) -> dict:
    """One row per expected data source, present or missing."""
    rows: list[dict] = []
    for kind, label in SOURCE_LABELS.items():
        assets = list(session.execute(
            select(DataAsset).where(DataAsset.kind == kind)
        ).scalars())
        if not assets:
            rows.append({
                "kind": kind, "label": label, "status": AssetStatus.MISSING.value,
                "n_files": 0, "records": None, "size_bytes": 0, "protected": False,
            })
            continue
        status = AssetStatus.VALIDATED.value if any(
            a.status == AssetStatus.VALIDATED.value for a in assets
        ) else assets[0].status
        rows.append({
            "kind": kind, "label": label, "status": status,
            "n_files": len(assets),
            "records": sum(a.n_records or 0 for a in assets) or None,
            "size_bytes": sum(a.size_bytes or 0 for a in assets),
            "protected": any(a.protected for a in assets),
            "paths": [a.path for a in assets[:10]],
        })
    return {"sources": rows}


@router.get("/assets")
def assets(kind: str | None = None, status: str | None = None,
           limit: int = 200, session: Session = Depends(get_db)) -> dict:
    stmt = select(DataAsset)
    if kind:
        stmt = stmt.where(DataAsset.kind == kind)
    if status:
        stmt = stmt.where(DataAsset.status == status)
    items = list(session.execute(stmt.order_by(DataAsset.kind, DataAsset.path)
                                 .limit(limit)).scalars())
    return {"assets": [_asset(a) for a in items], "count": len(items)}


@router.get("/assets/{asset_id}")
def asset_detail(asset_id: str, session: Session = Depends(get_db)) -> dict:
    asset = session.get(DataAsset, asset_id)
    if asset is None:
        raise HTTPException(404, "Asset not found")
    payload = _asset(asset)
    # Protected phenotype exports never leak row-level content through the API.
    report = dict(asset.validation_report or {})
    if asset.protected:
        report.pop("records", None)
        report["note"] = "Row-level phenotype content is withheld from the API."
    payload["validation_report"] = report
    return payload


def _asset(a: DataAsset) -> dict:
    return {
        "id": a.id, "kind": a.kind, "path": a.path, "status": a.status,
        "size_bytes": a.size_bytes, "sha256": a.sha256,
        "hash_is_partial": a.hash_is_partial, "n_records": a.n_records,
        "is_directory": a.is_directory, "protected": a.protected,
        "modified_at": a.modified_at.isoformat() if a.modified_at else None,
    }


@router.post("/rescan")
def rescan(session: Session = Depends(get_db),
           settings: Settings = Depends(get_settings)) -> dict:
    from neurotribe.acquisition.discover import run_discovery

    return run_discovery(session, settings)


@router.get("/scans")
def scans(movie: str | None = None, limit: int = 500,
          session: Session = Depends(get_db)) -> dict:
    stmt = select(Scan)
    if movie:
        stmt = stmt.where(Scan.movie == movie)
    items = list(session.execute(stmt.limit(limit)).scalars())
    subjects = {s.id: s for s in session.execute(select(Subject)).scalars()}
    return {
        "scans": [
            {
                "id": s.id,
                "subject": subjects[s.subject_id].external_id if s.subject_id in subjects else None,
                "task": s.task, "run": s.run, "session": s.session, "movie": s.movie,
                "movie_confidence": s.movie_confidence,
                "repetition_time": s.repetition_time, "n_volumes": s.n_volumes,
                "duration_sec": s.duration_sec, "site": s.site, "scanner": s.scanner,
                "content_present": s.content_present,
                "qc_status": s.qc.qc_status if s.qc else None,
                "mean_fd": s.qc.mean_fd if s.qc else None,
            }
            for s in items
        ],
        "count": len(items),
    }


@router.get("/scans/{scan_id}/evidence")
def scan_evidence(scan_id: str, session: Session = Depends(get_db)) -> dict:
    """Why this run was (or was not) bound to a movie - full evidence trail."""
    scan = session.get(Scan, scan_id)
    if scan is None:
        raise HTTPException(404, "Scan not found")
    return {
        "scan_id": scan.id, "movie": scan.movie,
        "confidence": scan.movie_confidence, "evidence": scan.movie_evidence,
        "sidecar": scan.sidecar_json,
    }


@router.get("/summary")
def summary(session: Session = Depends(get_db)) -> dict:
    movie_counts = dict(
        session.execute(select(Scan.movie, func.count()).group_by(Scan.movie)).all()
    )
    site_counts = dict(
        session.execute(
            select(Subject.site, func.count()).group_by(Subject.site)
        ).all()
    )
    return {
        "n_subjects": int(session.execute(select(func.count()).select_from(Subject)).scalar_one()),
        "n_scans": int(session.execute(select(func.count()).select_from(Scan)).scalar_one()),
        "by_movie": movie_counts,
        "by_site": {str(k): v for k, v in site_counts.items()},
        "n_with_movie": int(session.execute(
            select(func.count()).select_from(Scan).where(Scan.movie != MovieKey.UNKNOWN.value)
        ).scalar_one()),
    }
