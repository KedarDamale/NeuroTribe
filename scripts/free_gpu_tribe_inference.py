"""Run one visual-only TRIBE v2 inference in a free interactive GPU notebook.

Example for a Kaggle or Colab GPU session:
  pip install git+https://github.com/facebookresearch/tribev2@main
  python free_gpu_tribe_inference.py --image stimulus.jpg --output tribe_prediction.npy

Download the resulting `.npy` file and upload it on the local NeuroTRIBE page.
No BOLD fMRI file is needed in the notebook or sent off the local machine.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from neuralset.events.utils import standardize_events
from tribev2 import TribeModel


def make_video(image: Path, target: Path) -> None:
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(image),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-t", "4", "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(target),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr[-1000:] or "ffmpeg could not create the 4-second video")


def main() -> None:
    parser = argparse.ArgumentParser(description="Free interactive-GPU visual TRIBE v2 inference")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Still image used by the local NeuroTRIBE page")
    source.add_argument("--video", type=Path, help="Existing short MP4/MOV/MKV/WEBM video")
    parser.add_argument("--output", type=Path, default=Path("tribe_prediction.npy"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Enable a GPU accelerator in the notebook before running this script.")
    if args.image and not args.image.is_file():
        raise FileNotFoundError(args.image)
    if args.video and not args.video.is_file():
        raise FileNotFoundError(args.video)

    with tempfile.TemporaryDirectory(prefix="tribe-free-") as temporary:
        video = args.video
        if args.image:
            video = Path(temporary) / "stimulus.mp4"
            make_video(args.image, video)
        model = TribeModel.from_pretrained(
            "facebook/tribev2",
            device="cuda",
            config_update={
                "data.video_feature.use_audio": False,
                "data.features_to_use": ["video"],
            },
        )
        events = standardize_events(pd.DataFrame([{
            "type": "Video", "filepath": str(video), "start": 0,
            "timeline": "notebook", "subject": "default",
        }]))
        prediction, _segments = model.predict(events=events, verbose=False)

    result = np.asarray(prediction, dtype=np.float32)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] != 20484:
        raise RuntimeError(f"Unexpected TRIBE prediction shape: {result.shape}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, result.mean(axis=0).astype(np.float32), allow_pickle=False)
    print(f"Saved {args.output.resolve()} with shape (20484,).")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
