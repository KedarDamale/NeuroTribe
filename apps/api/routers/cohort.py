"""Cohort composition, balance diagnostics and exclusion accounting."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from neurotribe.config import Settings
from neurotribe.database.enums import AnalysisTier, CohortGroup, MovieKey
from neurotribe.database.models import Cohort, CohortMember, Subject

from apps.api.deps import get_db, get_settings

router = APIRouter(prefix="/cohort", tags=["cohort"])


def _latest(session: Session, tier: str) -> Cohort | None:
    return session.execute(
        select(Cohort).where(Cohort.tier == tier).order_by(Cohort.updated_at.desc())
    ).scalars().first()


@router.get("")
def cohort_summary(tier: str = "PRIMARY", session: Session = Depends(get_db)) -> dict:
    cohort = _latest(session, tier)
    if cohort is None:
        return {
            "available": False,
            "reason": "No cohort has been built yet. It requires phenotype data.",
            "tier": tier,
            "groups": [],
        }

    from neurotribe.cohort.matching import diagnose

    members = cohort.members
    control_group = (CohortGroup.NO_DIAGNOSIS_GIVEN if tier == "PRIMARY"
                     else CohortGroup.NON_ADHD_COMPARISON)

    counts: dict[str, int] = {}
    for member in members:
        if member.included:
            counts[member.group] = counts.get(member.group, 0) + 1

    exclusions: dict[str, int] = {}
    for member in members:
        if not member.included and member.exclusion_reason:
            exclusions[member.exclusion_reason] = exclusions.get(member.exclusion_reason, 0) + 1

    return {
        "available": True,
        "id": cohort.id, "name": cohort.name, "tier": cohort.tier,
        "movie": cohort.movie, "cohort_hash": cohort.cohort_hash,
        "analysis_config_hash": cohort.analysis_config_hash,
        "definition": cohort.definition,
        "warnings": cohort.warnings,
        "groups": [
            {"key": CohortGroup.CONFIRMED_ADHD.value,
             "label": CohortGroup.CONFIRMED_ADHD.display,
             "n": counts.get(CohortGroup.CONFIRMED_ADHD.value, 0)},
            {"key": control_group.value, "label": control_group.display,
             "n": counts.get(control_group.value, 0)},
            {"key": CohortGroup.EXCLUDED_UNCERTAIN.value,
             "label": CohortGroup.EXCLUDED_UNCERTAIN.display,
             "n": cohort.n_excluded},
        ],
        "n_case": cohort.n_case, "n_control": cohort.n_control,
        "n_excluded": cohort.n_excluded,
        "exclusion_breakdown": exclusions,
        "diagnostics": diagnose(members).to_dict(),
        "distributions": _distributions(members),
        "note": (
            "The comparison group is never described as 'healthy controls'. In the "
            "exploratory tier it may include participants with other diagnoses."
        ),
    }


def _distributions(members: list[CohortMember]) -> dict:
    """Histogram-ready distributions for the cohort charts."""
    included = [m for m in members if m.included]

    def bucket(values: list[float], width: float) -> dict[str, int]:
        out: dict[str, int] = {}
        for value in values:
            if value is None:
                continue
            low = int(value // width) * width
            out[f"{low:g}"] = out.get(f"{low:g}", 0) + 1
        return dict(sorted(out.items(), key=lambda kv: float(kv[0])))

    def by_group(attribute: str) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for member in included:
            group = member.group
            key = str(getattr(member, attribute, None) or "unknown")
            out.setdefault(group, {})
            out[group][key] = out[group].get(key, 0) + 1
        return out

    return {
        "age": {
            group: bucket([m.age for m in included if m.group == group and m.age is not None], 1.0)
            for group in {m.group for m in included}
        },
        "mean_fd": {
            group: bucket([m.mean_fd for m in included
                           if m.group == group and m.mean_fd is not None], 0.05)
            for group in {m.group for m in included}
        },
        "sex": by_group("sex"),
        "site": by_group("site"),
    }


@router.get("/members")
def members(tier: str = "PRIMARY", included: bool | None = None,
            group: str | None = None, limit: int = 1000,
            session: Session = Depends(get_db)) -> dict:
    cohort = _latest(session, tier)
    if cohort is None:
        raise HTTPException(404, "No cohort built yet")

    subjects = {s.id: s for s in session.execute(select(Subject)).scalars()}
    rows = []
    for member in cohort.members:
        if included is not None and member.included != included:
            continue
        if group and member.group != group:
            continue
        subject = subjects.get(member.subject_id)
        rows.append({
            "subject_external_id": subject.external_id if subject else None,
            "group": member.group, "included": member.included,
            "exclusion_reason": member.exclusion_reason,
            "exclusion_detail": member.exclusion_detail,
            "age": member.age, "sex": member.sex, "site": member.site,
            "scanner": member.scanner, "mean_fd": member.mean_fd,
            "usable_frame_fraction": member.usable_frame_fraction,
            "diagnostic_certainty": member.diagnostic_certainty,
            "comorbidities": member.comorbidities,
            "commercial_use_allowed": member.commercial_use_allowed,
        })
        if len(rows) >= limit:
            break
    return {"members": rows, "count": len(rows), "tier": tier}


@router.post("/rebuild")
def rebuild(tier: str = "PRIMARY", require_preprocessing: bool = False,
            session: Session = Depends(get_db),
            settings: Settings = Depends(get_settings)) -> dict:
    from neurotribe.acquisition.stimulus import select_primary
    from neurotribe.cohort.eligibility import build_cohort

    stimulus = select_primary(session, settings)
    movie = MovieKey(stimulus.key) if stimulus else None
    if movie is None:
        existing = _latest(session, tier)
        movie = MovieKey(existing.movie) if existing else None
    if movie is None:
        raise HTTPException(409, "No movie selected; cannot build a cohort")

    try:
        analysis_tier = AnalysisTier(tier)
    except ValueError:
        raise HTTPException(400, f"Unknown tier: {tier}")

    result = build_cohort(session, settings, movie, tier=analysis_tier,
                          require_preprocessing=require_preprocessing)
    return result.to_dict()


@router.get("/matching")
def matching(tier: str = "PRIMARY", session: Session = Depends(get_db)) -> dict:
    """Exploratory 1:1 matched subset. Never the primary analysis."""
    cohort = _latest(session, tier)
    if cohort is None:
        raise HTTPException(404, "No cohort built yet")

    from neurotribe.cohort.matching import propose_matched_subset

    return propose_matched_subset(cohort.members)
