"""Private GPU endpoint for visual-only official TRIBE v2 inference."""

from __future__ import annotations

import io
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from neuralset.events.utils import standardize_events
from tribev2 import TribeModel

# Hugging Face Inference Endpoints mount the selected model repository here,
# avoiding an extra download of TRIBE's checkpoint during endpoint startup.
MODEL_ID = os.environ.get("TRIBE_MODEL_ID", "/repository")
CACHE_DIR = os.environ.get("TRIBE_CACHE", "/data/tribe")
MAX_VIDEO_BYTES = int(os.environ.get("MAX_VIDEO_BYTES", str(100 * 1024 * 1024)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not torch.cuda.is_available():
        raise RuntimeError("This TRIBE endpoint requires a CUDA GPU.")
    app.state.model = TribeModel.from_pretrained(
        MODEL_ID,
        cache_folder=CACHE_DIR,
        device="cuda",
        config_update={
            # The local app sends deliberately silent clips made from an image.
            # Never download or evaluate the audio model for such a stimulus.
            "data.video_feature.use_audio": False,
            "data.features_to_use": ["video"],
        },
    )
    yield


app = FastAPI(title="NeuroTRIBE hosted visual inference", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_ID, "device": "cuda", "mode": "visual-only"}


@app.post("/predict")
async def predict(video: UploadFile = File(...)) -> Response:
    """Return a float32 .npy vector with TRIBE's mean prediction per vertex."""
    if Path(video.filename or "video.mp4").suffix.lower() not in {".mp4", ".mov", ".mkv", ".webm", ".avi"}:
        raise HTTPException(400, "Upload an MP4, MOV, MKV, WEBM, or AVI stimulus video.")
    payload = await video.read()
    if not payload or len(payload) > MAX_VIDEO_BYTES:
        raise HTTPException(400, f"Video must be between 1 byte and {MAX_VIDEO_BYTES} bytes.")

    with tempfile.TemporaryDirectory(prefix="tribe-request-") as temporary:
        path = Path(temporary) / "stimulus.mp4"
        path.write_bytes(payload)
        events = standardize_events(pd.DataFrame([{
            "type": "Video", "filepath": str(path), "start": 0,
            "timeline": "request", "subject": "default",
        }]))
        predicted, segments = app.state.model.predict(events=events, verbose=False)

    if predicted.ndim != 2 or predicted.shape[0] == 0:
        raise HTTPException(500, "TRIBE returned no usable prediction segments.")
    mean_prediction = np.asarray(predicted.mean(axis=0), dtype=np.float32)
    stream = io.BytesIO()
    np.save(stream, mean_prediction, allow_pickle=False)
    return Response(
        stream.getvalue(),
        media_type="application/x-npy",
        headers={
            "X-Tribe-Model": MODEL_ID,
            "X-Tribe-Device": "cuda",
            "X-Tribe-Segments": str(len(segments)),
            "X-Array-Dtype": "float32",
        },
    )
