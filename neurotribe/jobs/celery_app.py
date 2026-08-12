"""Celery application and beat schedule.

The Autopilot tick is a periodic task. Long scientific work (fMRIPrep, TRIBE)
runs inside stage handlers on the worker so the API stays responsive.
"""

from __future__ import annotations

import os

from celery import Celery
from celery.schedules import schedule as celery_schedule
from celery.signals import worker_process_init

from neurotribe.config import get_settings
from neurotribe.logging_setup import configure_logging


def _broker_url() -> str:
    return os.environ.get("CELERY_BROKER_URL") or os.environ.get(
        "REDIS_URL", "redis://localhost:6379/0"
    )


def _backend_url() -> str:
    return os.environ.get("CELERY_RESULT_BACKEND") or os.environ.get(
        "REDIS_URL", "redis://localhost:6379/1"
    )


app = Celery("neurotribe", broker=_broker_url(), backend=_backend_url())

app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,       # long scientific tasks: one at a time
    worker_max_tasks_per_child=50,      # bound memory growth from numpy/nibabel
    result_expires=7 * 24 * 3600,
    broker_connection_retry_on_startup=True,
    task_soft_time_limit=int(os.environ.get("NEUROTRIBE_TASK_SOFT_LIMIT", 60 * 60 * 20)),
    task_time_limit=int(os.environ.get("NEUROTRIBE_TASK_LIMIT", 60 * 60 * 24)),
    task_default_queue="neurotribe",
    imports=("neurotribe.jobs.tasks",),
)


@worker_process_init.connect
def _init_worker(**_: object) -> None:
    settings = get_settings()
    configure_logging(
        level=str(settings.get("logging.level", "INFO")),
        json_output=bool(settings.get("logging.json", True)),
        log_dir=settings.paths.data / "logs",
    )
    # Each forked worker needs its own engine.
    from neurotribe.database.base import reset_engine

    reset_engine()


def configure_beat() -> None:
    """Install the periodic Autopilot tick using the configured interval."""
    settings = get_settings()
    interval = float(settings.get("autopilot.tick_interval_sec", 20))
    app.conf.beat_schedule = {
        "autopilot-tick": {
            "task": "neurotribe.autopilot.tick",
            "schedule": celery_schedule(run_every=interval),
            "options": {"expires": max(interval * 2, 30)},
        },
        "watch-intake": {
            "task": "neurotribe.autopilot.watch_intake",
            "schedule": celery_schedule(run_every=max(interval, 15)),
            "options": {"expires": 60},
        },
    }


configure_beat()
