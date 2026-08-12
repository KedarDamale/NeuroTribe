"""Stimulus intake, validation status and media streaming."""

from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from neurotribe.config import Settings
from neurotribe.database.models import Stimulus

from apps.api.deps import get_db, get_settings

router = APIRouter(prefix="/stimulus", tags=["stimulus"])

CHUNK = 1024 * 1024


@router.get("")
def list_stimuli(session: Session = Depends(get_db),
                 settings: Settings = Depends(get_settings)) -> dict:
    from neurotribe.acquisition.stimulus import select_primary

    items = list(session.execute(select(Stimulus)).scalars())
    primary = select_primary(session, settings)
    return {
        "catalog": settings.get("stimulus.catalog", {}),
        "preference_order": settings.get("stimulus.preference_order", []),
        "incoming_dir": str(settings.paths.stimuli_incoming),
        "primary": primary.key if primary else None,
        "stimuli": [
            {
                "id": s.id, "key": s.key, "label": s.label, "validated": s.validated,
                "duration_sec": s.duration_sec, "expected_duration_sec": s.expected_duration_sec,
                "fps": s.fps, "width": s.width, "height": s.height,
                "has_audio": s.has_audio, "sha256": s.sha256, "size_bytes": s.size_bytes,
                "source_interval_start": s.source_interval_start,
                "source_interval_end": s.source_interval_end,
                "has_first_frame": bool(s.first_frame_path),
                "has_last_frame": bool(s.last_frame_path),
                "validation_notes": s.validation_notes,
                "provenance_note": s.provenance_note,
            }
            for s in items
        ],
        "policy": (
            "NeuroTRIBE never downloads video. Place the legally obtained clip in the "
            "intake directory; it is probed, matched to the documented HBN interval by "
            "duration, hashed and frame-sampled for verification."
        ),
    }


@router.post("/rescan")
def rescan(session: Session = Depends(get_db),
           settings: Settings = Depends(get_settings)) -> dict:
    from neurotribe.acquisition.discover import run_discovery
    from neurotribe.acquisition.stimulus import scan_incoming

    run_discovery(session, settings)
    return scan_incoming(session, settings)


@router.get("/{key}/frame/{which}")
def frame(key: str, which: str, session: Session = Depends(get_db)) -> FileResponse:
    if which not in {"first", "last"}:
        raise HTTPException(400, "which must be 'first' or 'last'")
    stimulus = session.execute(
        select(Stimulus).where(Stimulus.key == key)
    ).scalar_one_or_none()
    if stimulus is None:
        raise HTTPException(404, "Stimulus not registered")
    path = stimulus.first_frame_path if which == "first" else stimulus.last_frame_path
    if not path or not Path(path).exists():
        raise HTTPException(404, "Frame not extracted")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/{key}/media")
def media(key: str, request: Request, session: Session = Depends(get_db)):
    """Range-enabled streaming so the player can seek to a peak-deviation moment.

    The clip is served only to this local research UI; it is never re-published.
    """
    stimulus = session.execute(
        select(Stimulus).where(Stimulus.key == key)
    ).scalar_one_or_none()
    if stimulus is None or not stimulus.path:
        raise HTTPException(404, "Stimulus not registered")

    path = Path(stimulus.path)
    if not path.exists():
        raise HTTPException(404, "Stimulus file is missing from disk")

    size = path.stat().st_size
    media_type = mimetypes.guess_type(path.name)[0] or "video/mp4"
    range_header = request.headers.get("range")

    if range_header is None:
        return FileResponse(path, media_type=media_type,
                            headers={"Accept-Ranges": "bytes"})

    try:
        units, _, span = range_header.partition("=")
        if units.strip().lower() != "bytes":
            raise ValueError(units)
        start_text, _, end_text = span.partition("-")
        start = int(start_text) if start_text else 0
        end = int(end_text) if end_text else size - 1
    except ValueError:
        raise HTTPException(416, "Malformed Range header")

    start = max(0, start)
    end = min(end, size - 1)
    if start > end:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})

    def iterator():
        remaining = end - start + 1
        with path.open("rb") as handle:
            handle.seek(start)
            while remaining > 0:
                block = handle.read(min(CHUNK, remaining))
                if not block:
                    break
                remaining -= len(block)
                yield block

    return StreamingResponse(
        iterator(), status_code=206, media_type=media_type,
        headers={
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(end - start + 1),
        },
    )
