"""Hugging Face Inference Endpoint handler for visual-only TRIBE v2.

The local NeuroTRIBE app sends a short image-derived MP4 as base64 JSON.  The
handler returns a NumPy .npy payload encoded the same way, so no observed fMRI
or user data is uploaded to the hosted GPU.
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from neuralset.events.utils import standardize_events
from tribev2 import TribeModel

MODEL_ID = os.environ.get("TRIBE_MODEL_ID", "facebook/tribev2")
MAX_VIDEO_BYTES = int(os.environ.get("MAX_VIDEO_BYTES", str(100 * 1024 * 1024)))


class EndpointHandler:
    """Managed GPU handler discovered automatically by the HF inference toolkit."""

    def __init__(self, path: str) -> None:  # path is supplied by HF; weights use MODEL_ID.
        if not torch.cuda.is_available():
            raise RuntimeError("TRIBE v2 must run on a CUDA-backed Hugging Face endpoint.")
        self.model = TribeModel.from_pretrained(
            MODEL_ID,
            device="cuda",
            config_update={
                # Image-derived clips are deliberately silent: avoid the 2.3 GB
                # audio encoder and make the stimulus modality explicit.
                "data.video_feature.use_audio": False,
                "data.features_to_use": ["video"],
            },
        )

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        request = data.get("inputs", data)
        if not isinstance(request, dict) or not isinstance(request.get("video_base64"), str):
            raise ValueError("Expected inputs.video_base64 containing an MP4 encoded as base64.")
        try:
            video = base64.b64decode(request["video_base64"], validate=True)
        except Exception as exc:  # noqa: BLE001
            raise ValueError("inputs.video_base64 is not valid base64.") from exc
        if not video or len(video) > MAX_VIDEO_BYTES:
            raise ValueError(f"Video must be between 1 byte and {MAX_VIDEO_BYTES} bytes.")

        with tempfile.TemporaryDirectory(prefix="tribe-request-") as temporary:
            video_path = Path(temporary) / "stimulus.mp4"
            video_path.write_bytes(video)
            events = standardize_events(pd.DataFrame([{
                "type": "Video", "filepath": str(video_path), "start": 0,
                "timeline": "request", "subject": "default",
            }]))
            prediction, segments = self.model.predict(events=events, verbose=False)

        prediction = np.asarray(prediction, dtype=np.float32)
        if prediction.ndim != 2 or prediction.shape[0] == 0:
            raise RuntimeError("TRIBE returned no usable prediction segments.")
        stream = io.BytesIO()
        np.save(stream, prediction.mean(axis=0).astype(np.float32), allow_pickle=False)
        return {
            "prediction_npy_base64": base64.b64encode(stream.getvalue()).decode("ascii"),
            "shape": [int(prediction.shape[1])],
            "segments": int(len(segments)),
            "dtype": "float32",
        }
