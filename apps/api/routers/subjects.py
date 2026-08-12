"""Subject Explorer: per-participant detail, vertex maps and timelines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from neurotribe.config import Settings
from neurotribe.database.enums import MovieKey
from neurotribe.database.models import (
    NetworkMetric, PreprocessingRun, RoiMetric, Scan, Subject, SubjectComparison,
)

from apps.api.deps import get_db, get_settings

router = APIRouter(prefix="/subjects", tags=["subjects"])

MAP_KINDS = {
    "agreement": "vertex_r_path",
    "deviation": "vertex_mad_path",
}


@router.get("")
def list_subjects(has_comparison: bool | None = None, movie: str | None = None,
                  limit: int = 500, session: Session = Depends(get_db)) -> dict:
    subjects = list(session.execute(
        select(Subject).order_by(Subject.external_id).limit(limit)
    ).scalars())

    comparisons: dict[str, SubjectComparison] = {}
    for comparison in session.execute(
        select(SubjectComparison).order_by(SubjectComparison.created_at.desc())
    ).scalars():
        comparisons.setdefault(comparison.subject_id, comparison)

    rows = []
    for subject in subjects:
        comparison = comparisons.get(subject.id)
        if has_comparison is True and comparison is None:
            continue
        if has_comparison is False and comparison is not None:
            continue
        movie_scans = [s for s in subject.scans if s.movie != MovieKey.UNKNOWN.value]
        if movie and not any(s.movie == movie for s in movie_scans):
            continue
        rows.append({
            "external_id": subject.external_id, "site": subject.site,
            "age": subject.age, "sex": subject.sex,
            "has_phenotype": subject.has_phenotype,
            "has_movie_bold": subject.has_movie_bold,
            "movies": sorted({s.movie for s in movie_scans}),
            "diagnoses": [
                {"label": d.normalized_label or d.raw_label, "certainty": d.certainty,
                 "is_adhd": d.is_adhd}
                for d in subject.diagnoses
            ],
            "comparison": {
                "id": comparison.id, "valid": comparison.valid,
                "global_agreement_r": comparison.global_agreement_r,
                "global_mad": comparison.global_mad,
                "usable_frame_fraction": comparison.usable_frame_fraction,
                "is_approximate": comparison.is_approximate,
            } if comparison else None,
        })
    return {"subjects": rows, "count": len(rows)}


def _resolve(session: Session, external_id: str) -> Subject:
    subject = session.execute(
        select(Subject).where(Subject.external_id == external_id)
    ).scalar_one_or_none()
    if subject is None:
        raise HTTPException(404, f"Unknown participant: {external_id}")
    return subject


@router.get("/{external_id}")
def subject_detail(external_id: str, session: Session = Depends(get_db)) -> dict:
    subject = _resolve(session, external_id)
    comparison = session.execute(
        select(SubjectComparison).where(SubjectComparison.subject_id == subject.id)
        .order_by(SubjectComparison.created_at.desc())
    ).scalars().first()

    run = None
    if comparison and comparison.preprocessing_run_id:
        run = session.get(PreprocessingRun, comparison.preprocessing_run_id)

    scans = [
        {
            "id": s.id, "task": s.task, "run": s.run, "movie": s.movie,
            "movie_confidence": s.movie_confidence,
            "repetition_time": s.repetition_time, "n_volumes": s.n_volumes,
            "duration_sec": s.duration_sec, "site": s.site, "scanner": s.scanner,
            "qc_status": s.qc.qc_status if s.qc else None,
            "mean_fd": s.qc.mean_fd if s.qc else None,
        }
        for s in subject.scans
    ]

    payload: dict = {
        "external_id": subject.external_id,
        "bids_participant_id": subject.bids_participant_id,
        "site": subject.site, "age": subject.age, "sex": subject.sex,
        "release": subject.release,
        "commercial_use_allowed": subject.commercial_use_allowed,
        "has_phenotype": subject.has_phenotype,
        "diagnoses": [
            {"ordinal": d.ordinal, "label": d.normalized_label or d.raw_label,
             "certainty": d.certainty, "category": d.category, "is_adhd": d.is_adhd,
             "is_no_diagnosis": d.is_no_diagnosis}
            for d in sorted(subject.diagnoses, key=lambda d: d.ordinal)
        ],
        "scans": scans,
        "preprocessing": {
            "status": run.status, "engine": run.engine, "version": run.engine_version,
            "denoise_strategy": run.denoise_strategy,
            "n_volumes": run.n_volumes, "n_usable_frames": run.n_usable_frames,
            "usable_frame_fraction": run.usable_frame_fraction,
            "mean_fd": run.mean_fd, "n_nonsteady_state": run.n_nonsteady_state,
            "is_approximate": run.is_approximate, "error": run.error_message,
        } if run else None,
        "comparison": None,
    }

    if comparison is not None:
        rois = list(session.execute(
            select(RoiMetric).where(RoiMetric.comparison_id == comparison.id)
        ).scalars())
        networks = list(session.execute(
            select(NetworkMetric).where(NetworkMetric.comparison_id == comparison.id)
        ).scalars())

        deviating = sorted(
            [r for r in rois if r.mad is not None],
            key=lambda r: -r.mad,
        )[:5]
        deviating_networks = sorted(
            [n for n in networks if n.mad is not None], key=lambda n: -n.mad,
        )[:5]

        payload["comparison"] = {
            "id": comparison.id, "valid": comparison.valid,
            "invalid_reason": comparison.invalid_reason,
            "movie": comparison.movie, "tr": comparison.tr,
            "global_agreement_r": comparison.global_agreement_r,
            "global_mad": comparison.global_mad,
            "global_residual_variance": comparison.global_residual_variance,
            "n_shared_timepoints": comparison.n_shared_timepoints,
            "n_usable_timepoints": comparison.n_usable_timepoints,
            "usable_frame_fraction": comparison.usable_frame_fraction,
            "is_approximate": comparison.is_approximate,
            "peak_windows": comparison.peak_windows,
            "alignment_report": comparison.alignment_report,
            "sanity_report": comparison.sanity_report,
            "top_deviation_rois": [
                {"roi_name": r.roi_name, "network": r.network, "hemisphere": r.hemisphere,
                 "mad": r.mad, "agreement_r": r.agreement_r}
                for r in deviating
            ],
            "top_deviation_networks": [
                {"network": n.network, "mad": n.mad, "agreement_r": n.agreement_r}
                for n in deviating_networks
            ],
            "networks": [
                {"network": n.network, "agreement_r": n.agreement_r, "mad": n.mad,
                 "residual_variance": n.residual_variance, "n_vertices": n.n_vertices}
                for n in sorted(networks, key=lambda n: n.network)
            ],
        }
    return payload


@router.get("/{external_id}/map/{kind}", response_model=None)
def vertex_map(external_id: str, kind: str, format: str = Query("json"),
               session: Session = Depends(get_db)) -> Response | dict:
    """Per-vertex map for the 3D cortical viewer.

    ``format=binary`` returns raw float32 little-endian, which the browser reads
    directly into a Float32Array - roughly 8x smaller than JSON for 20 484
    values and far faster to parse.
    """
    if kind not in MAP_KINDS:
        raise HTTPException(400, f"kind must be one of {sorted(MAP_KINDS)}")

    subject = _resolve(session, external_id)
    comparison = session.execute(
        select(SubjectComparison).where(SubjectComparison.subject_id == subject.id)
        .order_by(SubjectComparison.created_at.desc())
    ).scalars().first()
    if comparison is None:
        raise HTTPException(404, "No comparison computed for this participant")

    path = getattr(comparison, MAP_KINDS[kind])
    if not path or not Path(path).exists():
        raise HTTPException(404, f"No '{kind}' map on disk for this participant")

    values = np.load(path).astype(np.float32)
    finite = values[np.isfinite(values)]

    if format == "binary":
        return Response(
            content=np.nan_to_num(values, nan=np.float32(np.nan)).tobytes(order="C"),
            media_type="application/octet-stream",
            headers={
                "X-Vertex-Count": str(values.size),
                "X-Value-Min": str(float(finite.min()) if finite.size else 0.0),
                "X-Value-Max": str(float(finite.max()) if finite.size else 0.0),
                "Cache-Control": "public, max-age=3600",
            },
        )

    return {
        "kind": kind,
        "n_vertices": int(values.size),
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "p05": float(np.percentile(finite, 5)) if finite.size else None,
        "p95": float(np.percentile(finite, 95)) if finite.size else None,
        "values": [None if not np.isfinite(v) else round(float(v), 5) for v in values],
    }


@router.get("/{external_id}/timeline")
def timeline(external_id: str, session: Session = Depends(get_db)) -> dict:
    """Synchronised time courses for the subject timeline strip."""
    from neurotribe.analysis.subject import load_timecourses

    subject = _resolve(session, external_id)
    comparison = session.execute(
        select(SubjectComparison).where(SubjectComparison.subject_id == subject.id)
        .order_by(SubjectComparison.created_at.desc())
    ).scalars().first()
    if comparison is None:
        raise HTTPException(404, "No comparison computed for this participant")

    courses = load_timecourses(comparison)
    if courses is None:
        raise HTTPException(404, "Residual time series not on disk")

    rolling = None
    if comparison.rolling_deviation_path and Path(comparison.rolling_deviation_path).exists():
        with np.load(comparison.rolling_deviation_path, allow_pickle=False) as payload:
            rolling = {
                "starts": payload["starts"].tolist(),
                "ends": payload["ends"].tolist(),
                "deviation": [None if not np.isfinite(v) else round(float(v), 5)
                              for v in payload["deviation"]],
                "coverage": payload["coverage"].tolist(),
            }

    return {
        "movie": comparison.movie, "tr": comparison.tr,
        "peak_windows": comparison.peak_windows,
        "timecourses": courses,
        "rolling": rolling,
    }


@router.get("/{external_id}/frame/{index}")
def frame_maps(external_id: str, index: int, session: Session = Depends(get_db)) -> Response:
    """Predicted / observed / residual vertex values at a single timepoint.

    Returns three concatenated float32 blocks so the viewer can scrub the time
    slider with one request per frame.
    """
    subject = _resolve(session, external_id)
    comparison = session.execute(
        select(SubjectComparison).where(SubjectComparison.subject_id == subject.id)
        .order_by(SubjectComparison.created_at.desc())
    ).scalars().first()
    if comparison is None or not comparison.residual_path:
        raise HTTPException(404, "No residual series for this participant")
    if not Path(comparison.residual_path).exists():
        raise HTTPException(404, "Residual file missing from disk")

    with np.load(comparison.residual_path, allow_pickle=False) as payload:
        residual = payload["residual"]
        usable = payload["usable"]
        time_sec = payload["time_sec"]

    if not 0 <= index < residual.shape[0]:
        raise HTTPException(400, f"index out of range 0..{residual.shape[0] - 1}")

    values = residual[index].astype(np.float32)
    return Response(
        content=values.tobytes(order="C"),
        media_type="application/octet-stream",
        headers={
            "X-Vertex-Count": str(values.size),
            "X-Time-Sec": str(float(time_sec[index])),
            "X-Usable": "1" if bool(usable[index]) else "0",
            "X-N-Frames": str(int(residual.shape[0])),
        },
    )


@router.post("/{external_id}/analyze")
def analyze_subject(external_id: str, force: bool = False,
                    session: Session = Depends(get_db),
                    settings: Settings = Depends(get_settings)) -> dict:
    """Recompute one participant's comparison on demand."""
    from neurotribe.analysis.subject import analyze
    from neurotribe.database.models import Stimulus, TribeRun
    from neurotribe.tribe.inference import load_cached

    subject = _resolve(session, external_id)
    run = session.execute(
        select(PreprocessingRun).where(PreprocessingRun.subject_id == subject.id,
                                       PreprocessingRun.denoised_path.is_not(None))
        .order_by(PreprocessingRun.created_at.desc())
    ).scalars().first()
    if run is None:
        raise HTTPException(409, "Participant has no prepared preprocessing output")

    scan = session.get(Scan, run.scan_id) if run.scan_id else None
    if scan is None:
        raise HTTPException(409, "Preprocessing run has no linked scan")

    prediction = load_cached(session, settings, scan.movie)
    tribe_run = session.execute(
        select(TribeRun).where(TribeRun.movie == scan.movie, TribeRun.status == "DONE")
        .order_by(TribeRun.created_at.desc())
    ).scalars().first()
    if prediction is None or tribe_run is None:
        raise HTTPException(409, "No TRIBE prediction available for this stimulus")

    stimulus = session.execute(
        select(Stimulus).where(Stimulus.key == scan.movie)
    ).scalar_one_or_none()

    result = analyze(session, settings, subject, scan, run, prediction, tribe_run,
                     stimulus, force=force)
    return result.to_dict()
