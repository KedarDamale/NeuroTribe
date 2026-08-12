"""The Autopilot controller.

Each tick answers, in order:

    What is installed?  ->  What data exists?  ->  What is missing?
    ->  What can be downloaded?  ->  What needs external authorization?
    ->  What can run now?  ->  Run it  ->  Validate  ->  Continue

Design rules:
  * A missing external dependency NEVER crashes the application. The stage goes
    ``WAITING_EXTERNAL``; only its dependents are blocked.
  * Every stage is resumable: state lives in the database, so a reboot resumes
    rather than restarting.
  * Every expensive step is cached on its inputs and configuration.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from neurotribe.config import Settings, get_settings
from neurotribe.database.base import session_scope
from neurotribe.database.enums import (
    AnalysisTier, AssetKind, AssetStatus, BlockerKind, BlockerSeverity, JobState,
    MovieKey, StageState,
)
from neurotribe.database.models import (
    Cohort, DataAsset, PreprocessingRun, Scan, Stimulus, Subject, TribeRun,
)
from neurotribe.database.repository import (
    active_blockers, all_stages, clear_blocker, create_job, finish_job, get_stage,
    raise_blocker, record_audit, schedule_retry, set_stage_state, upsert_stage,
)
from neurotribe.jobs.stages import EXTERNALLY_GATED, STAGES, descendants, order
from neurotribe.logging_setup import get_logger

log = get_logger(__name__)

StageHandler = Callable[[Session, Settings], "StageOutcome"]


@dataclass
class StageOutcome:
    """Result of running one stage."""

    state: StageState
    detail: str = ""
    progress: float = 1.0
    result: dict = field(default_factory=dict)
    error: str | None = None

    @classmethod
    def done(cls, detail: str, **result) -> "StageOutcome":
        return cls(StageState.DONE, detail, 1.0, result)

    @classmethod
    def waiting(cls, detail: str, **result) -> "StageOutcome":
        return cls(StageState.WAITING_EXTERNAL, detail, 0.0, result)

    @classmethod
    def partial(cls, detail: str, progress: float, **result) -> "StageOutcome":
        return cls(StageState.PARTIAL, detail, progress, result)

    @classmethod
    def failed(cls, error: str, *, final: bool = False) -> "StageOutcome":
        return cls(StageState.FAILED_FINAL if final else StageState.FAILED_RETRYABLE,
                   error[:500], 0.0, {}, error)

    @classmethod
    def skipped(cls, detail: str) -> "StageOutcome":
        return cls(StageState.SKIPPED, detail, 1.0)


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------

def ensure_stages(session: Session) -> None:
    """Create/refresh the stage rows. Idempotent."""
    for index, spec in enumerate(order()):
        upsert_stage(session, spec.key, spec.label, spec.phase, index,
                     spec.depends_on, spec.max_attempts)


def bootstrap(session: Session, settings: Settings) -> None:
    """One-time preparation: directories, stage graph, operator instructions."""
    settings.paths.ensure()
    ensure_stages(session)

    from neurotribe.acquisition.stimulus import write_intake_readme

    write_intake_readme(settings)
    _write_phenotype_readme(settings)
    record_audit(session, "autopilot.bootstrap", summary=f"profile={settings.profile}")


def _write_phenotype_readme(settings: Settings) -> Path:
    target = settings.paths.phenotype_incoming / "README.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "# Phenotype intake\n\n"
        "Drop the **authorised** HBN clinician-consensus export here.\n\n"
        "NeuroTRIBE never authenticates to LORIS and never bypasses the HBN Data "
        "Usage Agreement. It only parses what you place in this directory.\n\n"
        "## How to produce the file\n\n"
        "1. Complete the HBN DUA / institutional approval process.\n"
        "2. In the LORIS Data Query Tool, export the "
        "`Diagnosis_ClinicianConsensus` instrument as CSV.\n"
        "3. Save it here. It is auto-detected within one Autopilot tick.\n\n"
        "## Expected columns\n\n"
        "- A participant identifier (e.g. `Anonymized ID` / `EID`)\n"
        "- Diagnosis columns `DX_01` .. `DX_10`\n"
        "- Matching certainty columns (e.g. `DX_01_Conf`)\n\n"
        "Certainty is mapped onto: Confirmed, Presumptive, Requires Confirmation, "
        "Rule-out, By History, Past, No Diagnosis Given, Incomplete Eval.\n\n"
        "The primary ADHD cohort uses **Confirmed only**. Certainty levels are "
        "never pooled.\n\n"
        "> Files here are DUA-protected: they are git-ignored and never leave this "
        "machine.\n",
        encoding="utf-8",
    )
    return target


# --------------------------------------------------------------------------
# Stage handlers
# --------------------------------------------------------------------------

def _handle_system_probe(session: Session, settings: Settings) -> StageOutcome:
    from neurotribe.system import persist

    result = persist(session, settings)
    for message in result.blockers:
        kind = (BlockerKind.DISK_SPACE if "free" in message.lower()
                else BlockerKind.DOCKER_UNAVAILABLE if "docker" in message.lower()
                else BlockerKind.HARDWARE)
        raise_blocker(session, kind, message[:180], message,
                      severity=BlockerSeverity.ACTIONABLE)
    if result.ready:
        clear_blocker(session, BlockerKind.DISK_SPACE)
        clear_blocker(session, BlockerKind.HARDWARE)

    detail = (f"{result.cpu_count} CPU / {result.ram_gb or '?'} GB RAM / "
              f"{result.free_disk_gb} GB free"
              f"{' / ' + result.gpu_name if result.gpu_name else ' / no GPU'}")
    # Warnings do not block: a slow CPU path is still a correct path.
    return StageOutcome.done(detail, **result.to_dict())


def _handle_discover_assets(session: Session, settings: Settings) -> StageOutcome:
    from neurotribe.acquisition.discover import run_discovery

    summary = run_discovery(session, settings)
    by_kind = summary["by_kind"]
    detail = ", ".join(f"{k}: {v}" for k, v in sorted(by_kind.items())) or "no assets found"
    return StageOutcome.done(f"Discovered {summary['n_assets']} asset(s) - {detail}", **summary)


def _handle_ingest_metadata(session: Session, settings: Settings) -> StageOutcome:
    from neurotribe.acquisition.discover import run_discovery
    from neurotribe.acquisition.hbn_metadata import ingest_metadata

    run_discovery(session, settings)
    assets = list(session.execute(
        select(DataAsset).where(DataAsset.kind == AssetKind.HBN_METADATA.value)
    ).scalars())
    if not assets:
        raise_blocker(
            session, BlockerKind.METADATA_MISSING, "HBN release metadata not found",
            "No HBN metadata CSV (e.g. Metadata_R11.1.csv) was found in the workspace. "
            "It identifies which participants completed imaging and phenotypic sessions.",
            severity=BlockerSeverity.EXTERNAL,
            required_action="Place the HBN release metadata CSV in data/metadata/ "
                            "(or anywhere under the project root).",
            reference_url="https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/MRI_EEG.html",
            blocks_stages=sorted(descendants("ingest_metadata")),
        )
        return StageOutcome.waiting("Awaiting the HBN release metadata CSV.")

    clear_blocker(session, BlockerKind.METADATA_MISSING)
    total = 0
    for asset in assets:
        summary = ingest_metadata(session, settings, asset)
        total += summary.n_subjects
    return StageOutcome.done(f"Indexed {total} participant(s) from release metadata",
                             n_subjects=total)


def _handle_ingest_mriqc(session: Session, settings: Settings) -> StageOutcome:
    from neurotribe.acquisition.discover import run_discovery
    from neurotribe.acquisition.hbn_metadata import ingest_mriqc

    run_discovery(session, settings)
    assets = list(session.execute(
        select(DataAsset).where(DataAsset.kind.in_([
            AssetKind.MRIQC_FUNCTIONAL.value, AssetKind.MRIQC_ANATOMICAL.value,
        ]))
    ).scalars())
    if not assets:
        raise_blocker(
            session, BlockerKind.MRIQC_MISSING, "MRIQC image-quality metrics not found",
            "No MRIQC IQM export (e.g. IQM_functional_ExternalID.csv) was found. HBN "
            "releases imaging regardless of quality, so IQMs are needed for the QC policy.",
            severity=BlockerSeverity.EXTERNAL,
            required_action="Place the MRIQC IQM CSV in data/metadata/.",
            reference_url="https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/MRI_EEG.html",
        )
        return StageOutcome.waiting("Awaiting the MRIQC IQM export.")

    clear_blocker(session, BlockerKind.MRIQC_MISSING)
    total = 0
    for asset in assets:
        report = ingest_mriqc(session, settings, asset)
        total += int(report.get("n_records", 0))
    return StageOutcome.done(f"Parsed {total} MRIQC record(s)", n_records=total)


def _handle_index_bids(session: Session, settings: Settings) -> StageOutcome:
    from neurotribe.acquisition.bids import index_all
    from neurotribe.acquisition.discover import run_discovery

    run_discovery(session, settings)
    assets = list(session.execute(
        select(DataAsset).where(DataAsset.kind == AssetKind.BIDS_ROOT.value)
    ).scalars())
    if not assets:
        raise_blocker(
            session, BlockerKind.BIDS_MISSING, "HBN BIDS repository not found",
            "No BIDS dataset (a directory containing dataset_description.json) was "
            "found in the workspace.",
            severity=BlockerSeverity.EXTERNAL,
            required_action="Place the HBN BIDS tree at data/external/HBN_BIDS/ "
                            "(a DataLad clone is supported and enables selective fetch).",
            blocks_stages=sorted(descendants("index_bids")),
        )
        return StageOutcome.waiting("Awaiting the HBN BIDS repository.")

    clear_blocker(session, BlockerKind.BIDS_MISSING)
    summary = index_all(session, settings)
    n_subjects = sum(r.get("n_subjects", 0) for r in summary["reports"])
    n_bold = sum(r.get("n_bold", 0) for r in summary["reports"])
    return StageOutcome.done(f"Indexed {n_subjects} subject(s), {n_bold} BOLD run(s)",
                             **summary)


def _handle_identify_movie_scans(session: Session, settings: Settings) -> StageOutcome:
    from neurotribe.acquisition.bids import movie_scan_counts
    from neurotribe.acquisition.hbn_metadata import attach_qc_to_scans

    qc_summary = attach_qc_to_scans(session, settings)
    counts = movie_scan_counts(session)
    identified = sum(v for k, v in counts.items() if k != MovieKey.UNKNOWN.value)

    if identified == 0:
        return StageOutcome.waiting(
            "No BOLD run could be bound to a documented HBN movie interval. "
            "Verify RepetitionTime and volume counts in the BIDS sidecars.",
            counts=counts, qc=qc_summary,
        )
    detail = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())
                       if k != MovieKey.UNKNOWN.value)
    return StageOutcome.done(f"{identified} movie run(s) identified ({detail})",
                             counts=counts, qc=qc_summary)


def _handle_fetch_imaging(session: Session, settings: Settings) -> StageOutcome:
    from neurotribe.acquisition.fetch import estimate_disk_requirement, fetch
    from neurotribe.cohort.eligibility import target_subjects

    movie = _primary_movie(session, settings)
    if movie is None:
        return StageOutcome.waiting("No primary movie selected yet.")

    targets = target_subjects(session, settings, movie)
    if not targets:
        return StageOutcome.waiting(
            "No eligible participant yet (phenotype and/or movie runs pending)."
        )

    estimate = estimate_disk_requirement(settings, len(targets))
    if not estimate["sufficient"]:
        raise_blocker(
            session, BlockerKind.DISK_SPACE, "Insufficient disk for the target cohort",
            f"Preprocessing {estimate['n_subjects']} participant(s) needs roughly "
            f"{estimate['required_gb']} GB; only {estimate['free_gb']} GB is free.",
            severity=BlockerSeverity.ACTIONABLE,
            required_action="Free disk space or reduce the cohort size.",
            context=estimate,
        )
        return StageOutcome.waiting(
            f"Paused: {estimate['free_gb']} GB free, ~{estimate['required_gb']} GB required.",
            **estimate,
        )
    clear_blocker(session, BlockerKind.DISK_SPACE)

    bids_asset = session.execute(
        select(DataAsset).where(DataAsset.kind == AssetKind.BIDS_ROOT.value)
    ).scalars().first()
    bids_root = Path(bids_asset.absolute_path) if bids_asset else None

    result = fetch(session, settings, targets, bids_root)
    if result.method in ("already_present", "none"):
        return StageOutcome.done("All required imaging content is materialised.",
                                 **result.to_dict(), disk=estimate)
    if result.method == "unavailable":
        return StageOutcome.waiting("Imaging content cannot be retrieved automatically.",
                                    **result.to_dict())
    if result.failed:
        return StageOutcome.partial(
            f"Fetched {result.fetched}/{result.requested} file(s); "
            f"{len(result.failed)} failure(s).",
            progress=result.fetched / max(result.requested, 1), **result.to_dict(),
        )
    return StageOutcome.done(f"Fetched {result.fetched} file(s)", **result.to_dict())


def _handle_phenotype_intake(session: Session, settings: Settings) -> StageOutcome:
    from neurotribe.acquisition.discover import run_discovery
    from neurotribe.acquisition.phenotype import phenotype_available, scan_incoming

    # Re-scan first: this stage exists precisely to notice a file that appeared
    # after the initial discovery pass.
    run_discovery(session, settings)
    summary = scan_incoming(session, settings)
    if phenotype_available(session):
        clear_blocker(session, BlockerKind.PHENOTYPE_ACCESS)
        n_subjects = session.execute(
            select(Subject).where(Subject.has_phenotype.is_(True))
        ).scalars().all()
        return StageOutcome.done(
            f"Phenotype ingested for {len(n_subjects)} participant(s)",
            n_subjects=len(n_subjects), **summary,
        )

    raise_blocker(
        session, BlockerKind.PHENOTYPE_ACCESS, "HBN phenotype access",
        "Full HBN phenotypic data require an approved Data Usage Agreement and LORIS "
        "access. NeuroTRIBE cannot and will not obtain them on your behalf.",
        severity=BlockerSeverity.EXTERNAL,
        required_action=(
            "Complete the HBN DUA, then export the Diagnosis_ClinicianConsensus "
            f"instrument from the LORIS Data Query Tool as CSV into "
            f"{settings.paths.phenotype_incoming}."
        ),
        reference_url="https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/Phenotypic.html",
        blocks_stages=sorted(descendants("phenotype_intake")),
        context={"incoming_dir": str(settings.paths.phenotype_incoming),
                 "instrument": settings.get("phenotype.instrument")},
    )
    return StageOutcome.waiting(
        "Waiting for DUA-approved ADHD phenotype data. No labels are invented "
        "while this is pending.",
        **summary,
    )


def _handle_stimulus_intake(session: Session, settings: Settings) -> StageOutcome:
    from neurotribe.acquisition.discover import run_discovery
    from neurotribe.acquisition.stimulus import scan_incoming, select_primary

    run_discovery(session, settings)
    summary = scan_incoming(session, settings)
    primary = select_primary(session, settings)
    if primary is not None:
        clear_blocker(session, BlockerKind.STIMULUS_MISSING)
        return StageOutcome.done(
            f"Primary stimulus: {primary.label} ({primary.duration_sec:.1f}s)",
            movie=primary.key, sha256=primary.sha256, **summary,
        )

    catalog = settings.get("stimulus.catalog", {})
    expected = "; ".join(
        f"{spec.get('label')} = {spec.get('source_interval_start')}-"
        f"{spec.get('source_interval_end')} ({spec.get('expected_duration_sec')}s)"
        for spec in catalog.values()
    )
    raise_blocker(
        session, BlockerKind.STIMULUS_MISSING, "Exact movie stimulus required",
        "The exact clip shown during HBN scanning is copyrighted. NeuroTRIBE never "
        "downloads video and never scrapes unverified uploads. HBN documents the "
        "source intervals but asks researchers to contact the Child Mind Institute "
        "for exact-clip information.",
        severity=BlockerSeverity.EXTERNAL,
        required_action=(
            f"Place the legally obtained clip in {settings.paths.stimuli_incoming}. "
            f"Expected: {expected}"
        ),
        reference_url="https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/MRI_Protocol.html",
        blocks_stages=sorted(descendants("stimulus_intake")),
        context={"incoming_dir": str(settings.paths.stimuli_incoming),
                 "catalog": catalog, "candidates_seen": summary.get("n_candidates", 0)},
    )
    return StageOutcome.waiting("Waiting for the exact, legally obtained movie stimulus.",
                                **summary)


def _handle_tribe_install(session: Session, settings: Settings) -> StageOutcome:
    from neurotribe.tribe.inference import ensure_available

    status = ensure_available(session, settings)
    if status["available"]:
        return StageOutcome.done(
            f"TRIBE v2 available (commit {str(status.get('tribe_commit'))[:8] or 'unknown'})",
            **status,
        )
    if status["backend"] == "mock":
        # Development continues on the mock backend; this is explicitly not a result.
        return StageOutcome.done(
            "TRIBE not installed - continuing on the MOCK backend "
            "(development only; no mock output is reportable).",
            **status,
        )
    return StageOutcome.waiting("TRIBE v2 must be installed for this profile.", **status)


def _handle_tribe_smoke_test(session: Session, settings: Settings) -> StageOutcome:
    from neurotribe.tribe.inference import smoke_test

    result = smoke_test(session, settings)
    if not result.get("ok"):
        return StageOutcome.failed(
            f"TRIBE smoke test failed: {result.get('reason') or result.get('geometry_errors')}"
        )
    return StageOutcome.done(
        f"Smoke test passed on the {result['backend']} backend "
        f"({result['n_timepoints']} x {result['n_vertices']}, "
        f"hemi order {result['hemi_order']} via {result['hemi_order_source']})",
        **result,
    )


def _handle_preprocessing_preflight(session: Session, settings: Settings) -> StageOutcome:
    from neurotribe.preprocessing.fmriprep import ensure_license_or_block, preflight

    report = preflight(session, settings)
    ensure_license_or_block(session, settings)

    if not report["docker_available"]:
        return StageOutcome.waiting(f"Docker unavailable: {report['docker_detail']}", **report)
    if not report["freesurfer_license"]["found"]:
        return StageOutcome.waiting(
            "FreeSurfer license required for research-grade surface preprocessing.",
            **report,
        )
    return StageOutcome.done("Preprocessing engine ready (Docker + FreeSurfer license).",
                             **report)


def _handle_surface_geometry_check(session: Session, settings: Settings) -> StageOutcome:
    from neurotribe.preprocessing.surfaces import load_geometry, load_parcellation

    geometry = load_geometry(settings)
    parcellation = load_parcellation(settings)
    expected = int(settings.get("surface.vertices_per_hemi", 10242))

    problems = [
        f"{hemi}: {geometry.n_vertices(hemi)} vertices (expected {expected})"
        for hemi in ("L", "R") if geometry.n_vertices(hemi) != expected
    ]
    if problems:
        return StageOutcome.failed(
            "fsaverage5 geometry is inconsistent: " + "; ".join(problems), final=True,
        )

    detail = (f"fsaverage5 verified ({2 * expected} vertices, "
              f"{parcellation.n_parcels} parcels from {parcellation.source})")
    if parcellation.is_approximate:
        detail += " - approximate parcellation, ROI names are not anatomical labels"
    return StageOutcome.done(detail, geometry_source=geometry.source,
                             **parcellation.to_dict())


def _handle_tribe_inference(session: Session, settings: Settings) -> StageOutcome:
    from neurotribe.acquisition.stimulus import select_primary
    from neurotribe.tribe.inference import InferenceError, run as run_tribe
    from neurotribe.tribe.geometry import GeometryError

    stimulus = select_primary(session, settings)
    if stimulus is None:
        return StageOutcome.waiting("No validated stimulus available yet.")

    try:
        prediction = run_tribe(session, settings, stimulus)
    except (InferenceError, GeometryError) as exc:
        return StageOutcome.failed(str(exc))

    detail = (f"{stimulus.label}: {prediction.n_timepoints} timepoints x "
              f"{prediction.n_vertices} vertices on the {prediction.backend.value} backend")
    if prediction.is_mock:
        detail += " (MOCK - not a scientific result)"
    return StageOutcome.done(detail, movie=stimulus.key,
                             backend=prediction.backend.value,
                             support_sec=list(prediction.support))


def _handle_build_cohort(session: Session, settings: Settings) -> StageOutcome:
    from neurotribe.acquisition.phenotype import phenotype_available
    from neurotribe.cohort.eligibility import build_cohort

    if not phenotype_available(session):
        return StageOutcome.waiting("Phenotype data unavailable; cohort cannot be built.")

    movie = _primary_movie(session, settings)
    if movie is None:
        return StageOutcome.waiting("No primary movie selected yet.")

    primary = build_cohort(session, settings, movie, tier=AnalysisTier.PRIMARY,
                           require_preprocessing=False)
    exploratory = build_cohort(session, settings, movie, tier=AnalysisTier.EXPLORATORY,
                               require_preprocessing=False)

    minimum = int(settings.get("cohort.min_group_size", 10))
    if primary.n_case < minimum or primary.n_control < minimum:
        raise_blocker(
            session, BlockerKind.COHORT_TOO_SMALL, "Primary cohort below minimum size",
            f"Confirmed ADHD n={primary.n_case}, No Diagnosis Given n={primary.n_control}; "
            f"the configured minimum is {minimum} per group.",
            severity=BlockerSeverity.INFO,
            required_action="Supply more phenotype records, or lower cohort.min_group_size "
                            "and mark results as underpowered.",
            context=primary.to_dict(),
        )
    else:
        clear_blocker(session, BlockerKind.COHORT_TOO_SMALL)

    return StageOutcome.done(
        f"Primary: {primary.n_case} Confirmed ADHD vs {primary.n_control} No Diagnosis "
        f"Given; exploratory comparison n={exploratory.n_control}",
        primary=primary.to_dict(), exploratory=exploratory.to_dict(),
    )


def _handle_preprocess_cohort(session: Session, settings: Settings) -> StageOutcome:
    from neurotribe.cohort.eligibility import target_subjects
    from neurotribe.preprocessing.fmriprep import run_participant
    from neurotribe.preprocessing.pipeline import prepare_and_cache

    movie = _primary_movie(session, settings)
    if movie is None:
        return StageOutcome.waiting("No primary movie selected yet.")

    bids_asset = session.execute(
        select(DataAsset).where(DataAsset.kind == AssetKind.BIDS_ROOT.value)
    ).scalars().first()
    if bids_asset is None:
        return StageOutcome.waiting("No BIDS root registered.")
    bids_root = Path(bids_asset.absolute_path)

    targets = target_subjects(session, settings, movie)
    if not targets:
        return StageOutcome.waiting("No eligible participant to preprocess yet.")

    max_parallel = int(settings.get("autopilot.max_parallel_preprocessing", 1))
    pending: list[Scan] = []
    completed = 0

    for scan in targets:
        run = session.execute(
            select(PreprocessingRun)
            .where(PreprocessingRun.scan_id == scan.id)
            .order_by(PreprocessingRun.created_at.desc())
        ).scalars().first()
        if run is not None and run.denoised_path and Path(run.denoised_path).exists():
            completed += 1
        else:
            pending.append(scan)

    if not pending:
        return StageOutcome.done(f"All {completed} target participant(s) preprocessed.",
                                 n_completed=completed)

    processed_now = 0
    errors: list[str] = []
    for scan in pending[:max_parallel]:
        subject = session.get(Subject, scan.subject_id)
        if subject is None:
            continue
        job = create_job(session, f"fMRIPrep {subject.external_id}", "fmriprep",
                         stage_key="preprocess_cohort",
                         subject_external_id=subject.external_id)
        job.state = JobState.RUNNING.value
        job.started_at = datetime.now(timezone.utc)
        session.flush()
        try:
            run = run_participant(session, settings, subject, scan, bids_root)
            if run.status in ("SUCCEEDED", "APPROXIMATE"):
                prepare_and_cache(session, settings, run, scan, subject)
                processed_now += 1
                finish_job(session, job, JobState.SUCCEEDED,
                           message=f"usable frames: {run.usable_frame_fraction:.0%}"
                           if run.usable_frame_fraction is not None else None)
            else:
                errors.append(f"{subject.external_id}: {run.error_message or run.status}")
                finish_job(session, job, JobState.FAILED, error=run.error_message)
        except Exception as exc:  # noqa: BLE001 - one subject must not kill the stage
            errors.append(f"{subject.external_id}: {exc}")
            finish_job(session, job, JobState.FAILED, error=str(exc)[:1000])
            log.exception("Preprocessing raised", extra={"subject": subject.external_id})

    total = completed + processed_now
    remaining = len(pending) - processed_now
    if remaining > 0:
        return StageOutcome.partial(
            f"Preprocessed {total}/{len(targets)}; {remaining} remaining"
            + (f"; {len(errors)} failure(s)" if errors else ""),
            progress=total / max(len(targets), 1),
            n_completed=total, n_remaining=remaining, errors=errors[:10],
        )
    return StageOutcome.done(f"Preprocessed {total} participant(s)",
                             n_completed=total, errors=errors[:10])


def _handle_subject_analysis(session: Session, settings: Settings) -> StageOutcome:
    from neurotribe.analysis.subject import analyze
    from neurotribe.tribe.inference import load_cached

    movie = _primary_movie(session, settings)
    if movie is None:
        return StageOutcome.waiting("No primary movie selected yet.")

    prediction = load_cached(session, settings, movie.value)
    if prediction is None:
        return StageOutcome.waiting("TRIBE prediction unavailable for this stimulus.")

    tribe_run = session.execute(
        select(TribeRun).where(TribeRun.movie == movie.value, TribeRun.status == "DONE")
        .order_by(TribeRun.created_at.desc())
    ).scalars().first()
    if tribe_run is None:
        return StageOutcome.waiting("No completed TRIBE run recorded.")

    stimulus = session.execute(
        select(Stimulus).where(Stimulus.key == movie.value)
    ).scalar_one_or_none()

    runs = list(session.execute(
        select(PreprocessingRun).where(PreprocessingRun.denoised_path.is_not(None))
    ).scalars())
    if not runs:
        return StageOutcome.waiting("No preprocessed participant available yet.")

    analysed = 0
    invalid = 0
    errors: list[str] = []
    for run in runs:
        scan = session.get(Scan, run.scan_id) if run.scan_id else None
        subject = session.get(Subject, run.subject_id)
        if scan is None or subject is None or scan.movie != movie.value:
            continue
        try:
            result = analyze(session, settings, subject, scan, run, prediction,
                             tribe_run, stimulus)
            analysed += 1
            if not result.valid:
                invalid += 1
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{subject.external_id}: {exc}")
            log.exception("Subject analysis raised", extra={"subject": subject.external_id})

    if analysed == 0:
        return StageOutcome.waiting("No participant could be analysed yet.", errors=errors[:10])
    detail = f"Analysed {analysed} participant(s)"
    if invalid:
        detail += f"; {invalid} marked ANALYSIS INVALID"
    if errors:
        detail += f"; {len(errors)} error(s)"
    return StageOutcome.done(detail, n_analysed=analysed, n_invalid=invalid,
                             errors=errors[:10])


def _handle_group_analysis(session: Session, settings: Settings) -> StageOutcome:
    from neurotribe.analysis.group import run as run_group
    from neurotribe.cohort.eligibility import build_cohort

    movie = _primary_movie(session, settings)
    if movie is None:
        return StageOutcome.waiting("No primary movie selected yet.")

    # Rebuild with preprocessing required so only analysable participants enter.
    build_cohort(session, settings, movie, tier=AnalysisTier.PRIMARY,
                 require_preprocessing=True)
    build_cohort(session, settings, movie, tier=AnalysisTier.EXPLORATORY,
                 require_preprocessing=True)
    session.flush()

    outcomes: list[dict] = []
    for tier in (AnalysisTier.PRIMARY, AnalysisTier.EXPLORATORY):
        cohort = session.execute(
            select(Cohort).where(Cohort.tier == tier.value, Cohort.movie == movie.value)
            .order_by(Cohort.updated_at.desc())
        ).scalars().first()
        if cohort is None:
            continue
        result = run_group(session, settings, cohort, tier=tier)
        outcomes.append(result.to_dict())

    if not outcomes:
        return StageOutcome.waiting("No cohort available for group analysis.")

    primary = outcomes[0]
    detail = (f"PRIMARY: {primary['n_case']} vs {primary['n_control']}, "
              f"{primary['n_significant']}/{primary['n_units']} significant after FDR")
    if not primary["sanity_passed"]:
        detail = f"ANALYSIS INVALID - {'; '.join(primary['failures'][:2])}"
        return StageOutcome.partial(detail, progress=0.9, outcomes=outcomes)
    return StageOutcome.done(detail, outcomes=outcomes)


def _handle_generate_report(session: Session, settings: Settings) -> StageOutcome:
    from neurotribe.reporting.report import generate_all

    summary = generate_all(session, settings)
    if not summary.get("artifacts"):
        return StageOutcome.waiting("Nothing to report yet.")
    return StageOutcome.done(
        f"Generated {len(summary['artifacts'])} report artefact(s)", **summary,
    )


HANDLERS: dict[str, StageHandler] = {
    "system_probe": _handle_system_probe,
    "discover_assets": _handle_discover_assets,
    "ingest_metadata": _handle_ingest_metadata,
    "ingest_mriqc": _handle_ingest_mriqc,
    "index_bids": _handle_index_bids,
    "identify_movie_scans": _handle_identify_movie_scans,
    "fetch_imaging": _handle_fetch_imaging,
    "phenotype_intake": _handle_phenotype_intake,
    "stimulus_intake": _handle_stimulus_intake,
    "tribe_install": _handle_tribe_install,
    "tribe_smoke_test": _handle_tribe_smoke_test,
    "preprocessing_preflight": _handle_preprocessing_preflight,
    "surface_geometry_check": _handle_surface_geometry_check,
    "tribe_inference": _handle_tribe_inference,
    "build_cohort": _handle_build_cohort,
    "preprocess_cohort": _handle_preprocess_cohort,
    "subject_analysis": _handle_subject_analysis,
    "group_analysis": _handle_group_analysis,
    "generate_report": _handle_generate_report,
}


def _primary_movie(session: Session, settings: Settings) -> MovieKey | None:
    from neurotribe.acquisition.stimulus import select_primary

    stimulus = select_primary(session, settings)
    if stimulus is not None:
        return MovieKey(stimulus.key)

    # Without a stimulus we can still plan against whichever movie has scans.
    counts: dict[str, int] = {}
    for scan in session.execute(select(Scan)).scalars():
        if scan.movie != MovieKey.UNKNOWN.value:
            counts[scan.movie] = counts.get(scan.movie, 0) + 1
    if not counts:
        return None
    for key in settings.get("stimulus.preference_order", []):
        if key in counts:
            return MovieKey(key)
    return MovieKey(max(counts, key=counts.get))


# --------------------------------------------------------------------------
# Tick
# --------------------------------------------------------------------------

def _dependencies_satisfied(session: Session, spec) -> tuple[bool, str | None]:
    for dependency in spec.depends_on:
        stage = get_stage(session, dependency)
        if stage is None:
            return False, f"dependency '{dependency}' is not registered"
        state = StageState(stage.state)
        if state is StageState.DONE:
            continue
        if state is StageState.SKIPPED:
            continue
        # PARTIAL upstream still permits downstream progress on what exists.
        if state is StageState.PARTIAL and spec.key in {
            "subject_analysis", "group_analysis", "generate_report",
        }:
            continue
        return False, f"waiting on '{stage.label}' ({stage.state})"
    return True, None


def runnable_stages(session: Session) -> list[str]:
    """Stages whose dependencies are met and which are due to run."""
    now = datetime.now(timezone.utc)
    ready: list[str] = []
    for spec in order():
        stage = get_stage(session, spec.key)
        if stage is None:
            continue
        state = StageState(stage.state)
        if state in (StageState.RUNNING, StageState.DONE, StageState.SKIPPED,
                     StageState.FAILED_FINAL):
            continue
        if state is StageState.WAITING_EXTERNAL and spec.key not in EXTERNALLY_GATED:
            # Non-gated stages re-check on every tick too, but gated ones are the
            # ones we expect to sit here indefinitely.
            pass
        if stage.next_attempt_at is not None:
            due = stage.next_attempt_at
            if due.tzinfo is None:
                due = due.replace(tzinfo=timezone.utc)
            if due > now:
                continue
        satisfied, reason = _dependencies_satisfied(session, spec)
        if not satisfied:
            if state not in (StageState.BLOCKED, StageState.PENDING):
                continue
            # Always refresh the reason: the blocking dependency changes as
            # upstream stages complete, and a stale message is misleading.
            if state is not StageState.BLOCKED or stage.detail != reason:
                set_stage_state(session, spec.key, StageState.BLOCKED, detail=reason)
            continue
        if state is StageState.BLOCKED:
            set_stage_state(session, spec.key, StageState.PENDING,
                            detail="Dependencies satisfied.")
        ready.append(spec.key)

    # Stages that can genuinely advance go first; gated re-checks go last. This
    # is the second half of the anti-starvation guard (the first is the
    # WAITING_EXTERNAL backoff applied in run_stage).
    def priority(key: str) -> int:
        stage = get_stage(session, key)
        return 1 if stage is not None and stage.state == StageState.WAITING_EXTERNAL.value else 0

    return sorted(ready, key=priority)


def run_stage(session: Session, settings: Settings, key: str) -> StageOutcome:
    """Execute a single stage with full error containment."""
    handler = HANDLERS.get(key)
    if handler is None:
        return StageOutcome.skipped(f"No handler registered for '{key}'.")

    set_stage_state(session, key, StageState.RUNNING, detail="Running...", progress=0.0)
    session.flush()

    job = create_job(session, key, "stage", stage_key=key)
    job.state = JobState.RUNNING.value
    job.started_at = datetime.now(timezone.utc)
    session.flush()

    started = time.monotonic()
    try:
        outcome = handler(session, settings)
    except Exception as exc:  # noqa: BLE001 - the Autopilot must never die
        detail = f"{type(exc).__name__}: {exc}"
        log.exception("Stage raised", extra={"stage": key})
        record_audit(session, "stage.exception", entity_type="pipeline_stage",
                     entity_id=key, summary=detail[:200],
                     payload={"traceback": traceback.format_exc()[-4000:]})
        finish_job(session, job, JobState.FAILED, error=detail[:1000])
        schedule_retry(session, key, _backoff(session, settings, key), detail)
        return StageOutcome.failed(detail)

    elapsed = time.monotonic() - started

    if outcome.state in (StageState.FAILED_RETRYABLE, StageState.FAILED_FINAL):
        finish_job(session, job, JobState.FAILED, error=outcome.error)
        if outcome.state is StageState.FAILED_FINAL:
            set_stage_state(session, key, StageState.FAILED_FINAL,
                            detail=outcome.detail, error=outcome.error)
        else:
            schedule_retry(session, key, _backoff(session, settings, key),
                           outcome.error or outcome.detail)
        return outcome

    stage = set_stage_state(session, key, outcome.state, detail=outcome.detail,
                            progress=outcome.progress, result=outcome.result)

    if outcome.state is StageState.WAITING_EXTERNAL:
        # Throttle the re-check. A gated stage stays runnable forever, so without
        # this it would consume the whole per-tick budget on every tick and
        # starve stages that could actually make progress. Newly supplied files
        # are still noticed promptly because `watch_intake` polls the intake
        # directories independently.
        stage.next_attempt_at = datetime.now(timezone.utc) + timedelta(
            seconds=float(settings.get("autopilot.waiting_recheck_sec", 60))
        )
        # Waiting is not an attempt: a gate may legitimately wait for weeks.
        stage.attempts = max(0, stage.attempts - 1)

    finish_job(session, job, JobState.SUCCEEDED, message=outcome.detail)
    log.info("Stage complete", extra={"stage": key, "state": outcome.state.value,
                                      "detail": outcome.detail,
                                      "elapsed_sec": round(elapsed, 2)})
    return outcome


def _backoff(session: Session, settings: Settings, key: str) -> float:
    stage = get_stage(session, key)
    attempts = stage.attempts if stage else 1
    base = float(settings.get("autopilot.retry.backoff_base_sec", 30))
    ceiling = float(settings.get("autopilot.retry.backoff_max_sec", 3600))
    return min(ceiling, base * (2 ** max(0, attempts - 1)))


@dataclass
class TickResult:
    ran: list[str] = field(default_factory=list)
    outcomes: dict[str, str] = field(default_factory=dict)
    waiting: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)
    n_active_blockers: int = 0

    def to_dict(self) -> dict:
        return {
            "ran": self.ran, "outcomes": self.outcomes, "waiting": self.waiting,
            "blocked": self.blocked, "n_active_blockers": self.n_active_blockers,
        }


def tick(settings: Settings | None = None, *, max_stages: int = 4) -> TickResult:
    """One Autopilot iteration. Safe to call on any schedule."""
    settings = settings or get_settings()
    result = TickResult()

    with session_scope() as session:
        ensure_stages(session)

        if not bool(settings.get("autopilot.enabled", True)):
            log.info("Autopilot disabled by configuration")
            return result

        # Recompute readiness after every stage: completing one stage frequently
        # unblocks the next, so a single tick can walk several steps forward.
        attempted: set[str] = set()
        while len(result.ran) < max_stages:
            candidates = [k for k in runnable_stages(session) if k not in attempted]
            if not candidates:
                break
            key = candidates[0]
            attempted.add(key)
            outcome = run_stage(session, settings, key)
            result.ran.append(key)
            result.outcomes[key] = outcome.state.value
            session.flush()

        for stage in all_stages(session):
            if stage.state == StageState.WAITING_EXTERNAL.value:
                result.waiting.append(stage.key)
            elif stage.state == StageState.BLOCKED.value:
                result.blocked.append(stage.key)
        result.n_active_blockers = len(active_blockers(session))

    return result


def status(settings: Settings | None = None) -> dict:
    """Full pipeline status for the dashboard."""
    settings = settings or get_settings()
    with session_scope() as session:
        ensure_stages(session)
        stages = all_stages(session)
        blockers = active_blockers(session)
        return {
            "profile": settings.profile,
            "research_use_only": settings.research_use_only,
            "analysis_config_hash": settings.analysis_config_hash,
            "stages": [
                {
                    "key": s.key, "label": s.label, "phase": s.phase, "state": s.state,
                    "detail": s.detail, "progress": s.progress, "attempts": s.attempts,
                    "depends_on": s.depends_on, "last_error": s.last_error,
                    "result": s.result,
                }
                for s in stages
            ],
            "blockers": [
                {
                    "id": b.id, "kind": b.kind, "severity": b.severity, "title": b.title,
                    "description": b.description, "required_action": b.required_action,
                    "reference_url": b.reference_url, "blocks_stages": b.blocks_stages,
                    "context": b.context,
                }
                for b in blockers
            ],
        }
