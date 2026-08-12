"""System readiness and environment endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from neurotribe.config import Settings
from neurotribe.database.models import SystemProbe  # noqa: F401  (used below)

from apps.api.deps import get_db, get_settings

router = APIRouter(prefix="/system", tags=["system"])


@router.get("")
def system_status(session: Session = Depends(get_db),
                  settings: Settings = Depends(get_settings)) -> dict:
    latest = session.execute(
        select(SystemProbe).order_by(SystemProbe.created_at.desc())
    ).scalars().first()
    return {
        "probe": latest.payload if latest else None,
        "probed_at": latest.created_at.isoformat() if latest else None,
        "profile": settings.profile,
        "paths": {name: str(getattr(settings.paths, name))
                  for name in ("root", "data", "phenotype_incoming", "stimuli_incoming",
                               "derivatives", "tribe", "analysis", "reports")},
    }


@router.post("/probe")
def run_probe(session: Session = Depends(get_db),
              settings: Settings = Depends(get_settings)) -> dict:
    """Refresh the hardware probe.

    The probe is dispatched to the **worker**, not run here: only the worker
    mounts the Docker socket, so an API-local probe would report
    ``docker_available: False`` and overwrite the accurate reading. Falling back
    to a local probe keeps this useful when no worker is running (local dev).
    """
    try:
        # Send on the CONFIGURED app. Importing the task and calling
        # `.apply_async()` would bind to Celery's default app instead, whose
        # broker is not our Redis - the message would silently go nowhere.
        from neurotribe.jobs.celery_app import app as celery_app

        celery_app.send_task("neurotribe.system.probe", expires=120)
        latest = session.execute(
            select(SystemProbe).order_by(SystemProbe.created_at.desc())
        ).scalars().first()
        return {
            "dispatched": True,
            "note": "Probe dispatched to the worker; this is the last stored result.",
            **(latest.payload if latest else {}),
        }
    except Exception as exc:  # noqa: BLE001 - broker may be unavailable
        from neurotribe.system import persist

        result = persist(session, settings).to_dict()
        return {
            "dispatched": False,
            "note": (
                "Worker unreachable, probed locally instead. Docker availability "
                f"reflects this container only. ({exc})"
            ),
            **result,
        }


@router.get("/config")
def config(settings: Settings = Depends(get_settings)) -> dict:
    """Scientific configuration actually in force, plus its hash."""
    return {
        "profile": settings.profile,
        "analysis_config_hash": settings.analysis_config_hash,
        "scientific": settings.scientific_config,
        "research_use_only": settings.research_use_only,
    }


@router.get("/tribe")
def tribe_status(settings: Settings = Depends(get_settings)) -> dict:
    from neurotribe.tribe.model import install_hint, probe_tribe, resolve_device

    available, version, error = probe_tribe()
    return {
        "available": available,
        "error": error,
        "version": version.to_dict(),
        "device": resolve_device(str(settings.get("tribe.device", "auto"))).to_dict(),
        "configured_backend": settings.get("tribe.backend"),
        "install": install_hint(settings),
    }


@router.get("/preprocessing")
def preprocessing_status(session: Session = Depends(get_db),
                         settings: Settings = Depends(get_settings)) -> dict:
    from neurotribe.preprocessing.fmriprep import preflight

    return preflight(session, settings)
