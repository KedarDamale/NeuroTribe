"""Deterministic mock TRIBE backend.

Purpose: keep the *entire* pipeline - alignment, deviation, ROI aggregation,
statistics, UI - buildable and testable while the real model or the licensed
stimulus is unavailable.

Guarantees that keep this honest:
  * Output is a deterministic function of the stimulus bytes, so a smoke test is
    reproducible.
  * It reproduces TRIBE's *interface contract* (shape, timing, hemisphere order)
    so geometry validation exercises real code paths.
  * Every artefact is stamped ``backend: mock`` and the production profile
    refuses to load it.

It is NOT a scientific model and no result derived from it is ever reportable.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MOCK_TR = 1.49          # matches TRIBE's published prediction cadence
MOCK_HRF_OFFSET = 5.0   # TRIBE states its timing already includes this offset


@dataclass
class MockTribeModel:
    """Interface-compatible stand-in for ``tribev2.TribeModel``."""

    n_vertices: int = 20484
    hemi_order: tuple[str, ...] | list[str] = ("L", "R")
    tr: float = MOCK_TR

    # -- interface parity with the real model ---------------------------
    def get_events_dataframe(self, video_path: str | Path, **_: object) -> pd.DataFrame:
        """Produce a stimulus event table (one row per prediction window)."""
        path = Path(video_path)
        duration = _probe_duration(path)
        seed = _seed_from_file(path)
        rng = np.random.default_rng(seed)

        n_windows = max(1, int(duration / self.tr))
        onsets = np.arange(n_windows, dtype=float) * self.tr
        # Synthetic "scene" structure so peak-window analysis has something to find.
        scene_ids = np.cumsum(rng.random(n_windows) < 0.02)

        return pd.DataFrame({
            "onset": onsets,
            "duration": np.full(n_windows, self.tr),
            "modality": ["audiovisual"] * n_windows,
            "scene": scene_ids.astype(int),
            "rms_audio": rng.random(n_windows).astype(np.float32),
            "visual_energy": rng.random(n_windows).astype(np.float32),
            "n_words": rng.integers(0, 6, n_windows).astype(int),
        })

    def predict(self, events: pd.DataFrame, **_: object) -> tuple[np.ndarray, pd.DataFrame]:
        """Return ``(predictions, segments)`` matching the real API's shape contract.

        ``predictions`` is (n_timepoints, n_vertices) on the configured surface;
        ``segments`` carries the timestamps, which already include the 5 s
        hemodynamic offset - exactly as the real model documents.
        """
        n_timepoints = int(len(events))
        seed = int(pd.util.hash_pandas_object(events, index=False).sum() % (2 ** 32))
        rng = np.random.default_rng(seed)

        predictions = _structured_signal(rng, n_timepoints, self.n_vertices, events)
        onsets = events["onset"].to_numpy(dtype=float)
        segments = pd.DataFrame({
            "start": onsets + MOCK_HRF_OFFSET,
            "end": onsets + MOCK_HRF_OFFSET + self.tr,
            "index": np.arange(n_timepoints),
        })
        return predictions.astype(np.float32), segments


def _seed_from_file(path: Path) -> int:
    """Deterministic seed from file content (or name when unreadable)."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            digest.update(handle.read(4 * 1024 * 1024))
    except OSError:
        digest.update(str(path).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "big") % (2 ** 32)


def _probe_duration(path: Path) -> float:
    """Best-effort duration; falls back to a fixed length for synthetic inputs."""
    try:
        from neurotribe.acquisition.stimulus import probe

        info = probe(path)
        if info.duration_sec:
            return float(info.duration_sec)
    except Exception:  # noqa: BLE001 - ffprobe absence is expected in unit tests
        pass
    return 60.0


def _structured_signal(rng: np.random.Generator, n_timepoints: int, n_vertices: int,
                       events: pd.DataFrame) -> np.ndarray:
    """Build spatially smooth, temporally autocorrelated synthetic cortex.

    Real BOLD is neither white noise nor independent across vertices; the mock
    reproduces both properties so downstream statistics behave realistically
    (e.g. correlations are not trivially zero everywhere).
    """
    n_components = 12
    # Shared temporal components driven by the stimulus event table.
    drivers = np.zeros((n_timepoints, n_components), dtype=np.float32)
    for component in range(n_components):
        noise = rng.standard_normal(n_timepoints)
        # AR(1) smoothing approximates hemodynamic low-pass behaviour.
        smoothed = np.zeros(n_timepoints)
        alpha = 0.75
        for t in range(1, n_timepoints):
            smoothed[t] = alpha * smoothed[t - 1] + (1 - alpha) * noise[t]
        drivers[:, component] = smoothed

    if "visual_energy" in events:
        drivers[:, 0] += events["visual_energy"].to_numpy(dtype=np.float32)
    if "rms_audio" in events:
        drivers[:, 1] += events["rms_audio"].to_numpy(dtype=np.float32)

    # Spatially smooth mixing weights: neighbouring vertices load similarly.
    weights = rng.standard_normal((n_components, n_vertices)).astype(np.float32)
    kernel = np.ones(64, dtype=np.float32) / 64.0
    for component in range(n_components):
        weights[component] = np.convolve(weights[component], kernel, mode="same")

    signal = drivers @ weights
    signal += 0.25 * rng.standard_normal(signal.shape).astype(np.float32)

    # Standardise per vertex so scale is comparable to a z-scored BOLD series.
    mean = signal.mean(axis=0, keepdims=True)
    sd = signal.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return (signal - mean) / sd


def synthetic_video(path: Path, duration_sec: float = 20.0, fps: int = 24,
                    width: int = 320, height: int = 240) -> Path | None:
    """Render a short synthetic clip for the TRIBE smoke test.

    Used only to prove the inference path works end to end. It is never treated
    as an HBN stimulus: it will not match any catalog duration.
    """
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc=size={width}x{height}:rate={fps}:duration={duration_sec}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration_sec}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)
    if result.returncode != 0 or not path.exists():
        return None
    return path
