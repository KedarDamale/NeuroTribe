"""Eligibility evaluation and cohort building.

Eligibility chain (specification section 15):

    has phenotype AND has supported movie BOLD AND has usable anatomical data
    AND has scan metadata AND preprocessing succeeds AND QC passes policy

Every failure produces a machine-readable :class:`ExclusionReason`. Nothing is
ever dropped silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from neurotribe.config import Settings
from neurotribe.database.enums import (
    AnalysisTier, CohortGroup, ExclusionReason, MovieKey, PreprocStatus, QCStatus,
)
from neurotribe.database.models import (
    Cohort, CohortMember, PreprocessingRun, Scan, Subject,
)
from neurotribe.database.repository import record_audit
from neurotribe.hashing import hash_json
from neurotribe.logging_setup import get_logger

from neurotribe.cohort.diagnoses import (
    assign_exploratory_group, assign_primary_group, build_profile,
)

log = get_logger(__name__)


@dataclass
class Eligibility:
    subject: Subject
    scan: Scan | None
    preprocessing: PreprocessingRun | None
    eligible: bool
    reason: ExclusionReason | None = None
    detail: str | None = None
    checks: dict[str, bool] = field(default_factory=dict)

    # Confound snapshot
    mean_fd: float | None = None
    usable_frame_fraction: float | None = None
    is_approximate: bool = False


def _select_scan(scans: Iterable[Scan], movie: MovieKey) -> Scan | None:
    """Choose the best run for the target movie: highest confidence, then longest."""
    candidates = [s for s in scans if s.movie == movie.value]
    if not candidates:
        return None
    return max(candidates, key=lambda s: ((s.movie_confidence or 0.0), (s.n_volumes or 0)))


def evaluate(session: Session, settings: Settings, subject: Subject, movie: MovieKey,
             *, require_preprocessing: bool = True) -> Eligibility:
    """Run the full eligibility chain for one participant."""
    checks: dict[str, bool] = {}

    checks["has_phenotype"] = bool(subject.has_phenotype)
    if not checks["has_phenotype"]:
        return Eligibility(subject, None, None, False, ExclusionReason.NO_PHENOTYPE,
                           "No clinician-consensus record for this participant.", checks)

    scan = _select_scan(subject.scans, movie)
    checks["has_movie_bold"] = scan is not None
    if scan is None:
        return Eligibility(subject, None, None, False, ExclusionReason.NO_MOVIE_BOLD,
                           f"No BOLD run classified as '{movie.value}'.", checks)

    checks["has_anatomical"] = bool(scan.t1w_path)
    if settings.get("qc.anat.require_t1w", True) and not checks["has_anatomical"]:
        return Eligibility(subject, scan, None, False, ExclusionReason.NO_ANATOMICAL,
                           "No T1w anatomical image available for surface reconstruction.", checks)

    checks["has_scan_metadata"] = bool(scan.repetition_time and scan.n_volumes)
    if not checks["has_scan_metadata"]:
        return Eligibility(subject, scan, None, False, ExclusionReason.NO_SCAN_METADATA,
                           "RepetitionTime or volume count missing from BIDS metadata.", checks)

    # MRIQC policy.
    qc = scan.qc
    if qc is not None and qc.qc_status == QCStatus.FAIL.value:
        checks["mriqc_pass"] = False
        return Eligibility(subject, scan, None, False, ExclusionReason.QC_FAILED_MRIQC,
                           qc.qc_reason or "MRIQC policy failure.", checks,
                           mean_fd=qc.mean_fd)
    checks["mriqc_pass"] = True

    run = session.execute(
        select(PreprocessingRun)
        .where(PreprocessingRun.scan_id == scan.id)
        .order_by(PreprocessingRun.created_at.desc())
    ).scalars().first()

    if run is None or run.status not in (PreprocStatus.SUCCEEDED.value, PreprocStatus.APPROXIMATE.value):
        checks["preprocessed"] = False
        if require_preprocessing:
            detail = ("Preprocessing has not completed for this run."
                      if run is None else f"Preprocessing status: {run.status}.")
            reason = (ExclusionReason.PREPROCESSING_FAILED
                      if run is not None and run.status == PreprocStatus.FAILED.value
                      else ExclusionReason.PREPROCESSING_FAILED)
            return Eligibility(subject, scan, run, False, reason, detail, checks,
                               mean_fd=qc.mean_fd if qc else None)
        # Planning mode: report as provisionally eligible so the target list can
        # be built before preprocessing runs.
        return Eligibility(subject, scan, run, True, None, "Pending preprocessing.", checks,
                           mean_fd=qc.mean_fd if qc else None)

    checks["preprocessed"] = True
    is_approximate = bool(run.is_approximate) or run.status == PreprocStatus.APPROXIMATE.value

    # Approximate surfaces are excluded from any final analysis by policy.
    if is_approximate and settings.get(
        "preprocessing.approximate_volume_projection.excluded_from_final_analysis", True
    ):
        if settings.is_production:
            return Eligibility(subject, scan, run, False, ExclusionReason.APPROXIMATE_SURFACE,
                               "Approximate development projection is not permitted in the "
                               "production profile.", checks, mean_fd=run.mean_fd,
                               usable_frame_fraction=run.usable_frame_fraction,
                               is_approximate=True)

    min_fraction = float(settings.get("qc.motion.min_usable_frame_fraction", 0.5))
    min_frames = int(settings.get("qc.motion.min_usable_frames", 60))
    fraction = run.usable_frame_fraction
    frames = run.n_usable_frames

    checks["usable_frames"] = True
    if fraction is not None and fraction < min_fraction:
        checks["usable_frames"] = False
        return Eligibility(subject, scan, run, False, ExclusionReason.INSUFFICIENT_USABLE_FRAMES,
                           f"Usable frame fraction {fraction:.2f} below {min_fraction:.2f}.",
                           checks, mean_fd=run.mean_fd, usable_frame_fraction=fraction,
                           is_approximate=is_approximate)
    if frames is not None and frames < min_frames:
        checks["usable_frames"] = False
        return Eligibility(subject, scan, run, False, ExclusionReason.INSUFFICIENT_USABLE_FRAMES,
                           f"Only {frames} usable frames (minimum {min_frames}).",
                           checks, mean_fd=run.mean_fd, usable_frame_fraction=fraction,
                           is_approximate=is_approximate)

    fd_limit = settings.get("qc.mriqc.max_mean_fd")
    if fd_limit is not None and run.mean_fd is not None and run.mean_fd > float(fd_limit):
        checks["motion_pass"] = False
        return Eligibility(subject, scan, run, False, ExclusionReason.QC_FAILED_MOTION,
                           f"Mean FD {run.mean_fd:.3f} mm exceeds {float(fd_limit):.3f} mm.",
                           checks, mean_fd=run.mean_fd, usable_frame_fraction=fraction,
                           is_approximate=is_approximate)
    checks["motion_pass"] = True

    return Eligibility(subject, scan, run, True, None, None, checks,
                       mean_fd=run.mean_fd, usable_frame_fraction=fraction,
                       is_approximate=is_approximate)


@dataclass
class CohortBuildResult:
    cohort_id: str
    name: str
    tier: str
    n_case: int
    n_control: int
    n_excluded: int
    warnings: list[str] = field(default_factory=list)
    exclusion_breakdown: dict[str, int] = field(default_factory=dict)
    cohort_hash: str = ""

    def to_dict(self) -> dict:
        return {
            "cohort_id": self.cohort_id, "name": self.name, "tier": self.tier,
            "n_case": self.n_case, "n_control": self.n_control,
            "n_excluded": self.n_excluded, "warnings": self.warnings,
            "exclusion_breakdown": self.exclusion_breakdown,
            "cohort_hash": self.cohort_hash,
        }


def build_cohort(session: Session, settings: Settings, movie: MovieKey,
                 *, tier: AnalysisTier = AnalysisTier.PRIMARY,
                 require_preprocessing: bool = True) -> CohortBuildResult:
    """Construct (or refresh) a cohort for the given movie and analysis tier."""
    cohort_config = settings.get("cohort", {})
    name = (f"{'Primary' if tier is AnalysisTier.PRIMARY else 'Exploratory'} "
            f"- {movie.value}")

    subjects = list(session.execute(select(Subject)).scalars())
    assign = assign_primary_group if tier is AnalysisTier.PRIMARY else assign_exploratory_group
    control_group = (CohortGroup.NO_DIAGNOSIS_GIVEN if tier is AnalysisTier.PRIMARY
                     else CohortGroup.NON_ADHD_COMPARISON)

    members: list[dict] = []
    exclusion_breakdown: dict[str, int] = {}
    seen_external_ids: set[str] = set()

    for subject in subjects:
        if subject.external_id in seen_external_ids:
            # Guard against duplicated participants entering an analysis.
            members.append({
                "subject_id": subject.id, "scan_id": None,
                "group": CohortGroup.EXCLUDED_UNCERTAIN.value, "included": False,
                "exclusion_reason": ExclusionReason.DUPLICATE_SUBJECT.value,
                "exclusion_detail": "Duplicate participant identifier.",
            })
            continue
        seen_external_ids.add(subject.external_id)

        profile = build_profile(subject)
        group, group_reason = assign(profile, cohort_config)
        eligibility = evaluate(session, settings, subject, movie,
                               require_preprocessing=require_preprocessing)

        included = eligibility.eligible and group in (CohortGroup.CONFIRMED_ADHD, control_group)
        reason: str | None = None
        detail: str | None = None

        if not eligibility.eligible:
            reason = eligibility.reason.value if eligibility.reason else ExclusionReason.NOT_ELIGIBLE_FOR_CONTRAST.value
            detail = eligibility.detail
        elif group not in (CohortGroup.CONFIRMED_ADHD, control_group):
            reason = ExclusionReason.UNCERTAIN_DIAGNOSIS.value
            detail = group_reason or "Not assignable to either contrast group."

        # Non-commercial restriction is recorded, not silently applied.
        if included and subject.commercial_use_allowed is False:
            detail = (detail or "") + " Participant flagged non-commercial-use by HBN."

        if not included and reason:
            exclusion_breakdown[reason] = exclusion_breakdown.get(reason, 0) + 1

        members.append({
            "subject_id": subject.id,
            "scan_id": eligibility.scan.id if eligibility.scan else None,
            "group": group.value if included else CohortGroup.EXCLUDED_UNCERTAIN.value,
            "included": included,
            "exclusion_reason": reason,
            "exclusion_detail": detail,
            "age": subject.age,
            "sex": subject.sex,
            "site": subject.site or (eligibility.scan.site if eligibility.scan else None),
            "scanner": eligibility.scan.scanner if eligibility.scan else None,
            "mean_fd": eligibility.mean_fd,
            "usable_frame_fraction": eligibility.usable_frame_fraction,
            "diagnostic_certainty": profile.strongest_adhd_certainty,
            "comorbidities": profile.comorbidities[:20],
            "commercial_use_allowed": subject.commercial_use_allowed,
        })

    n_case = sum(1 for m in members if m["included"] and m["group"] == CohortGroup.CONFIRMED_ADHD.value)
    n_control = sum(1 for m in members if m["included"] and m["group"] == control_group.value)
    n_excluded = sum(1 for m in members if not m["included"])

    definition = {
        "movie": movie.value, "tier": tier.value,
        "case": cohort_config.get("primary", {}).get("case"),
        "control": (cohort_config.get("primary", {}).get("control")
                    if tier is AnalysisTier.PRIMARY
                    else cohort_config.get("exploratory", {}).get("control")),
        "require_preprocessing": require_preprocessing,
    }
    # The cohort hash pins exactly which participants entered the analysis.
    cohort_hash = hash_json({
        "definition": definition,
        "members": sorted(
            (m["subject_id"], m["group"], m["included"]) for m in members if m["included"]
        ),
    }, length=32)

    cohort = session.execute(
        select(Cohort).where(Cohort.name == name, Cohort.tier == tier.value)
    ).scalar_one_or_none()
    if cohort is None:
        # cohort_hash is NOT NULL and this row is flushed before the field
        # assignments below, so it must be populated at construction.
        cohort = Cohort(name=name, tier=tier.value, movie=movie.value,
                        cohort_hash=cohort_hash, definition=definition)
        session.add(cohort)
        session.flush()

    for existing in list(cohort.members):
        session.delete(existing)
    session.flush()

    for payload in members:
        session.add(CohortMember(cohort_id=cohort.id, **payload))

    warnings: list[str] = []
    min_size = int(settings.get("cohort.min_group_size", 10))
    if n_case < min_size:
        warnings.append(f"Confirmed ADHD group has {n_case} participants (minimum {min_size}).")
    if n_control < min_size:
        warnings.append(
            f"{control_group.display} group has {n_control} participants (minimum {min_size})."
        )
    if n_case and n_control:
        ratio = max(n_case, n_control) / min(n_case, n_control)
        limit = float(settings.get("cohort.imbalance_warn_ratio", 3.0))
        if ratio > limit:
            warnings.append(
                f"Group sizes are imbalanced ({n_case} vs {n_control}, ratio {ratio:.1f}). "
                "Interpret covariate-adjusted effects with care."
            )

    cohort.movie = movie.value
    cohort.definition = definition
    cohort.cohort_hash = cohort_hash
    cohort.analysis_config_hash = settings.analysis_config_hash
    cohort.n_case = n_case
    cohort.n_control = n_control
    cohort.n_excluded = n_excluded
    cohort.warnings = warnings

    session.flush()
    result = CohortBuildResult(
        cohort_id=cohort.id, name=name, tier=tier.value, n_case=n_case,
        n_control=n_control, n_excluded=n_excluded, warnings=warnings,
        exclusion_breakdown=exclusion_breakdown, cohort_hash=cohort_hash,
    )
    record_audit(session, "cohort.built", entity_type="cohort", entity_id=cohort.id,
                 summary=f"{n_case} case / {n_control} control", payload=result.to_dict())
    log.info("Cohort built", extra=result.to_dict())
    return result


def target_subjects(session: Session, settings: Settings, movie: MovieKey) -> list[Scan]:
    """Scans the pipeline should preprocess next (selective-fetch target list)."""
    subjects = list(session.execute(select(Subject)).scalars())
    targets: list[Scan] = []
    for subject in subjects:
        eligibility = evaluate(session, settings, subject, movie, require_preprocessing=False)
        if eligibility.eligible and eligibility.scan is not None:
            targets.append(eligibility.scan)
    return targets
