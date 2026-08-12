"""Celery tasks.

Everything here is a thin wrapper: the real logic lives in the Autopilot and the
scientific packages, so each task is resumable and independently testable.
"""

from __future__ import annotations

from celery import shared_task

from neurotribe.config import get_settings
from neurotribe.database.base import session_scope
from neurotribe.jobs import autopilot
from neurotribe.logging_setup import get_logger

log = get_logger(__name__)


@shared_task(name="neurotribe.autopilot.tick", bind=True, ignore_result=False)
def autopilot_tick(self, max_stages: int = 4) -> dict:
    """Advance the pipeline by one iteration."""
    settings = get_settings()
    result = autopilot.tick(settings, max_stages=max_stages)
    if result.ran:
        log.info("Autopilot tick", extra=result.to_dict())
    return result.to_dict()


@shared_task(name="neurotribe.autopilot.bootstrap")
def bootstrap() -> dict:
    """Create directories, stage graph and operator instructions."""
    settings = get_settings()
    with session_scope() as session:
        autopilot.bootstrap(session, settings)
    return {"ok": True, "profile": settings.profile,
            "root": str(settings.root)}


@shared_task(name="neurotribe.autopilot.watch_intake")
def watch_intake() -> dict:
    """Poll the gated intake directories for newly supplied files.

    Kept separate from the main tick so a long-running preprocessing stage never
    delays detection of a phenotype export or stimulus drop.
    """
    from neurotribe.acquisition import discover, phenotype, stimulus

    settings = get_settings()
    with session_scope() as session:
        discover.run_discovery(session, settings)
        phenotype_summary = phenotype.scan_incoming(session, settings)
        stimulus_summary = stimulus.scan_incoming(session, settings)
    return {"phenotype": phenotype_summary, "stimulus": stimulus_summary}


@shared_task(name="neurotribe.stage.run", bind=True)
def run_stage(self, key: str) -> dict:
    """Force a single stage to run now (used by the UI 'retry' action)."""
    settings = get_settings()
    with session_scope() as session:
        autopilot.ensure_stages(session)
        outcome = autopilot.run_stage(session, settings, key)
    return {"stage": key, "state": outcome.state.value, "detail": outcome.detail,
            "error": outcome.error}


@shared_task(name="neurotribe.preprocess.subject", bind=True)
def preprocess_subject(self, subject_external_id: str) -> dict:
    """Preprocess one participant on demand."""
    from pathlib import Path

    from sqlalchemy import select

    from neurotribe.database.enums import AssetKind, MovieKey
    from neurotribe.database.models import DataAsset, Subject
    from neurotribe.preprocessing.fmriprep import run_participant
    from neurotribe.preprocessing.pipeline import prepare_and_cache

    settings = get_settings()
    with session_scope() as session:
        subject = session.execute(
            select(Subject).where(Subject.external_id == subject_external_id)
        ).scalar_one_or_none()
        if subject is None:
            return {"ok": False, "error": f"Unknown participant {subject_external_id}"}

        scans = [s for s in subject.scans if s.movie != MovieKey.UNKNOWN.value]
        if not scans:
            return {"ok": False, "error": "No movie BOLD run for this participant"}
        scan = max(scans, key=lambda s: (s.movie_confidence or 0.0))

        asset = session.execute(
            select(DataAsset).where(DataAsset.kind == AssetKind.BIDS_ROOT.value)
        ).scalars().first()
        if asset is None:
            return {"ok": False, "error": "No BIDS root registered"}

        run = run_participant(session, settings, subject, scan, Path(asset.absolute_path))
        if run.status in ("SUCCEEDED", "APPROXIMATE"):
            prepared = prepare_and_cache(session, settings, run, scan, subject)
            return {"ok": True, "status": run.status,
                    "usable_frame_fraction": prepared.mask.fraction,
                    "n_timepoints": prepared.n_timepoints}
        return {"ok": False, "status": run.status, "error": run.error_message}


@shared_task(name="neurotribe.system.probe")
def system_probe() -> dict:
    """Refresh the hardware probe.

    Runs on the worker because only the worker mounts the Docker socket; a probe
    executed in the API container would report Docker as unavailable.
    """
    from neurotribe.system import persist

    settings = get_settings()
    with session_scope() as session:
        return persist(session, settings).to_dict()


@shared_task(name="neurotribe.report.generate")
def generate_report() -> dict:
    from neurotribe.reporting.report import generate_all

    settings = get_settings()
    with session_scope() as session:
        return generate_all(session, settings)
