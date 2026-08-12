"""FastAPI application entry point."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from neurotribe import RESEARCH_DISCLAIMER, __version__
from neurotribe.config import get_settings
from neurotribe.logging_setup import configure_logging, get_logger

from apps.api.routers import (
    cohort, dashboard, data, groups, jobs, logs, qc, reports, stimulus,
    subjects, surface, system,
)

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(
        level=str(settings.get("logging.level", "INFO")),
        json_output=bool(settings.get("logging.json", True)),
        log_dir=settings.paths.data / "logs",
    )
    settings.paths.ensure()

    # Create the schema when Alembic has not been run (dev/SQLite path).
    if os.environ.get("NEUROTRIBE_AUTO_CREATE_SCHEMA", "1") == "1":
        from neurotribe.database.base import create_all

        try:
            create_all()
        except Exception as exc:  # noqa: BLE001 - never block API startup
            log.error("Schema creation failed", extra={"error": str(exc)})

    # Register the stage graph and write operator instructions.
    try:
        from neurotribe.database.base import session_scope
        from neurotribe.jobs.autopilot import bootstrap

        with session_scope() as session:
            bootstrap(session, settings)
    except Exception as exc:  # noqa: BLE001
        log.error("Autopilot bootstrap failed", extra={"error": str(exc)})

    log.info("NeuroTRIBE API ready", extra={"version": __version__,
                                            "profile": settings.profile})
    yield
    log.info("NeuroTRIBE API shutting down")


app = FastAPI(
    title="NeuroTRIBE-HBN API",
    description=(
        "Stimulus-conditioned normative cortical response analysis for ADHD using "
        "TRIBE v2 and Healthy Brain Network movie-fMRI.\n\n"
        f"**{RESEARCH_DISCLAIMER}**"
    ),
    version=__version__,
    lifespan=lifespan,
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

_origins = os.environ.get(
    "NEUROTRIBE_CORS_ORIGINS",
    "http://localhost:4321,http://localhost:3000,http://127.0.0.1:4321",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_research_headers(request: Request, call_next):
    """Stamp every response so the research-only status cannot be lost."""
    response = await call_next(request)
    response.headers["X-NeuroTRIBE-Version"] = __version__
    response.headers["X-Research-Use-Only"] = "true"
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled API error", extra={"path": str(request.url.path)})
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}",
                 "path": str(request.url.path)},
    )


@app.get("/api/health", tags=["system"])
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "version": __version__,
        "profile": settings.profile,
        "research_use_only": settings.research_use_only,
        "disclaimer": RESEARCH_DISCLAIMER,
    }


for router in (dashboard, system, data, stimulus, cohort, subjects, groups, qc,
               jobs, logs, reports, surface):
    app.include_router(router.router, prefix="/api")
