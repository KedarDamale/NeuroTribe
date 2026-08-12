"""Quality-control page."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from neurotribe.config import Settings
from neurotribe.database.enums import CohortGroup
from neurotribe.database.models import Cohort, Subject

from apps.api.deps import get_db, get_settings

router = APIRouter(prefix="/qc", tags=["qc"])


@router.get("")
def qc_table(status: str | None = None, site: str | None = None,
             group: str | None = None, only_failures: bool = False,
             session: Session = Depends(get_db),
             settings: Settings = Depends(get_settings)) -> dict:
    from neurotribe.preprocessing.qc import build_rows, summarize

    rows = build_rows(session, settings)

    # Attach cohort group so the QC page can filter to ADHD participants.
    cohort = session.execute(
        select(Cohort).where(Cohort.tier == "PRIMARY").order_by(Cohort.updated_at.desc())
    ).scalars().first()
    if cohort is not None:
        subjects = {s.id: s.external_id for s in session.execute(select(Subject)).scalars()}
        by_external = {
            subjects.get(m.subject_id): m.group
            for m in cohort.members if m.included
        }
        for row in rows:
            row.group = by_external.get(row.subject_external_id)

    filtered = rows
    if only_failures:
        filtered = [r for r in filtered if r.overall in ("FAIL", "WARNING")]
    if status:
        filtered = [r for r in filtered if r.overall == status]
    if site:
        filtered = [r for r in filtered if r.site == site]
    if group:
        filtered = [r for r in filtered if r.group == group]

    return {
        "rows": [r.to_dict() for r in filtered],
        "count": len(filtered),
        "summary": summarize(rows),
        "sites": sorted({r.site for r in rows if r.site}),
        "groups": [
            {"key": CohortGroup.CONFIRMED_ADHD.value,
             "label": CohortGroup.CONFIRMED_ADHD.display},
            {"key": CohortGroup.NO_DIAGNOSIS_GIVEN.value,
             "label": CohortGroup.NO_DIAGNOSIS_GIVEN.display},
            {"key": CohortGroup.NON_ADHD_COMPARISON.value,
             "label": CohortGroup.NON_ADHD_COMPARISON.display},
        ],
        "policy": {
            "fd_threshold_mm": settings.get("qc.motion.fd_threshold_mm"),
            "dvars_threshold_sd": settings.get("qc.motion.dvars_threshold_sd"),
            "min_usable_frame_fraction": settings.get("qc.motion.min_usable_frame_fraction"),
            "min_usable_frames": settings.get("qc.motion.min_usable_frames"),
            "max_mean_fd": settings.get("qc.mriqc.max_mean_fd"),
        },
    }
