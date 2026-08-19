"""Single-subject, image-driven exploratory TRIBE viewer.

This route deliberately lives outside the HBN Autopilot.  It accepts a public
BIDS task-fMRI run and a user-selected still image, turns the image into a
silent video with the run's duration, then visualises the observed volume,
TRIBE prediction, and their difference on the fsaverage5 viewer mesh.

The volume-to-surface display is an interpolation for visual exploration only;
it is never a substitute for fMRIPrep/FreeSurfer surface reconstruction and is
stamped as approximate in every response and report.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import tempfile
import uuid
from base64 import b64decode, b64encode
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from neuralset.events.utils import standardize_events

from neurotribe.config import Settings
from neurotribe.numerics import safe_zscore
from neurotribe.tribe import model as tribe_model

from apps.api.deps import get_settings

router = APIRouter(prefix="/exploratory", tags=["exploratory"])

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
_BOLD_SUFFIX = "_bold.nii.gz"
_FREE_GPU_SCRIPT = "free_gpu_tribe_inference.py"


def _root(settings: Settings) -> Path:
    path = settings.paths.analysis / "exploratory"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _relative(path: Path, base: Path) -> str:
    return str(path.resolve().relative_to(base.resolve())).replace("\\", "/")


def _safe_path(settings: Settings, relative: str, *, images: bool = False) -> Path:
    base = settings.paths.external if not images else settings.paths.stimuli
    candidate = (base / relative).resolve()
    if base.resolve() not in candidate.parents or not candidate.is_file():
        raise HTTPException(400, "Selected file is outside the local exploratory data directory.")
    return candidate


def _bold_metadata(path: Path) -> dict:
    import nibabel as nib

    image = nib.load(str(path))
    shape = image.shape
    if len(shape) != 4:
        raise ValueError("Expected a 4-D BOLD NIfTI file.")
    return {"shape": list(shape), "n_volumes": int(shape[3]), "tr_sec": float(image.header.get_zooms()[3])}


@router.get("/catalog")
def catalog(settings: Settings = Depends(get_settings)) -> dict:
    """List only local BIDS BOLD runs and locally supplied still images."""
    bold = []
    for path in sorted(settings.paths.external.rglob(f"*{_BOLD_SUFFIX}")):
        try:
            meta = _bold_metadata(path)
        except Exception:
            continue
        bold.append({"id": _relative(path, settings.paths.external), "label": str(path.relative_to(settings.paths.external)), **meta})

    image_root = settings.paths.stimuli / "exploratory"
    image_root.mkdir(parents=True, exist_ok=True)
    images = []
    for path in sorted(image_root.rglob("*")):
        if path.suffix.lower() in _IMAGE_EXTENSIONS:
            images.append({"id": _relative(path, settings.paths.stimuli), "label": path.name})
    return {"bold": bold, "images": images, "approximate": True}


@router.post("/images")
async def upload_image(file: UploadFile, settings: Settings = Depends(get_settings)) -> dict:
    suffix = Path(file.filename or "image.png").suffix.lower()
    if suffix not in _IMAGE_EXTENSIONS:
        raise HTTPException(400, "Upload a PNG, JPG, BMP, WEBP, or JPEG image.")
    payload = await file.read()
    if not payload or len(payload) > 20 * 1024 * 1024:
        raise HTTPException(400, "Image must be between 1 byte and 20 MB.")
    target = settings.paths.stimuli / "exploratory" / f"{uuid.uuid4().hex}{suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return {"id": _relative(target, settings.paths.stimuli), "label": file.filename or target.name}


@router.get("/image")
def image_file(id: str, settings: Settings = Depends(get_settings)):
    path = _safe_path(settings, id, images=True)
    return FileResponse(path)


@router.get("/free-gpu-script")
def free_gpu_script(settings: Settings = Depends(get_settings)):
    """Download the one-shot script for a free interactive GPU notebook."""
    path = settings.root / "scripts" / _FREE_GPU_SCRIPT
    if not path.is_file():
        raise HTTPException(404, "Free GPU helper script is unavailable.")
    return FileResponse(path, media_type="text/x-python", filename=_FREE_GPU_SCRIPT)


def _state_path(settings: Settings, run_id: str) -> Path:
    return _root(settings) / run_id / "state.json"


def _write_state(settings: Settings, run_id: str, **state) -> None:
    path = _state_path(settings, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Status polling can happen while a background task advances a run.  Write
    # beside the current state and atomically replace it so a reader never sees
    # a partially-written (or momentarily empty) JSON document.
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(path)


def _read_state(settings: Settings, run_id: str) -> dict:
    path = _state_path(settings, run_id)
    if not path.exists():
        raise HTTPException(404, "Exploratory run not found.")
    return json.loads(path.read_text(encoding="utf-8"))


def _make_video(image: Path, target: Path, duration: float) -> None:
    # H.264 with yuv420p requires both dimensions to be divisible by two.  User
    # images (including the bundled task diagram) can have an odd pixel height.
    command = ["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(image),
               "-f", "lavfi", "-i", f"anullsrc=channel_layout=stereo:sample_rate=44100",
               "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-t", f"{duration:.3f}",
               "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-shortest", str(target)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-1000:] or "ffmpeg could not create the stimulus video")


def _volume_map(path: Path, n_vertices: int) -> np.ndarray:
    import nibabel as nib

    image = nib.load(str(path))
    volume = np.asarray(image.dataobj, dtype=np.float32)
    mean = np.nanmean(volume, axis=3).reshape(-1)
    finite = mean[np.isfinite(mean)]
    if finite.size < 2:
        raise RuntimeError("Observed fMRI volume contains insufficient finite data.")
    source = np.nan_to_num(mean, nan=float(np.nanmedian(finite)))
    indices = np.linspace(0, source.size - 1, n_vertices)
    projected = np.interp(indices, np.arange(source.size), source).astype(np.float32)
    normalized, _flat = safe_zscore(projected.reshape(1, -1), axis=1)
    return normalized.reshape(-1).astype(np.float32)


def _hosted_endpoint(settings: Settings) -> tuple[str, str, float, str] | None:
    """Return protected hosted endpoint credentials without exposing its token."""
    endpoint = str(settings.get("tribe.hosted.endpoint_url", "") or "").strip().rstrip("/")
    if not endpoint:
        return None
    if not endpoint.startswith("https://"):
        raise RuntimeError("TRIBE hosted endpoint URL must use https://")
    token_env = str(settings.get("tribe.hosted.token_env", "HF_TRIBE_ENDPOINT_TOKEN"))
    token = os.environ.get(token_env, "").strip()
    if not token:
        raise RuntimeError(f"Hosted TRIBE is configured but the {token_env} secret is missing.")
    timeout = float(settings.get("tribe.hosted.timeout_sec", 900))
    protocol = str(settings.get("tribe.hosted.protocol", "hf-toolkit")).strip().lower()
    if protocol not in {"hf-toolkit", "custom-container", "hf-space"}:
        raise RuntimeError("tribe.hosted.protocol must be 'hf-toolkit', 'custom-container', or 'hf-space'.")
    return endpoint, token, timeout, protocol


def _predict_hosted(video: Path, endpoint: tuple[str, str, float, str], settings: Settings) -> np.ndarray:
    """Send only the generated stimulus video to the chosen remote TRIBE host."""
    base_url, token, timeout, protocol = endpoint
    request_timeout = httpx.Timeout(timeout, connect=30.0)
    headers = {"Authorization": f"Bearer {token}"}
    if protocol == "hf-space":
        try:
            # A ZeroGPU Space is a public, on-demand Gradio service.  It receives
            # only the generated video; BOLD fMRI and all result processing stay
            # local.  Gradio downloads the returned .npy file to a temporary path.
            from gradio_client import Client, handle_file

            with tempfile.TemporaryDirectory(prefix="tribe-space-") as downloads:
                client = Client(base_url, verbose=False, download_files=downloads)
                output = client.predict(handle_file(str(video)), api_name="/predict")
                if isinstance(output, (list, tuple)):
                    output = output[0]
                predicted = np.load(Path(str(output)), allow_pickle=False)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Free TRIBE ZeroGPU Space request failed: {exc}") from exc
    elif protocol == "hf-toolkit":
        # Hugging Face's managed inference toolkit forwards JSON to our
        # EndpointHandler.  Base64 keeps the short generated MP4 self-contained
        # and avoids accepting arbitrary external video URLs.
        response = httpx.post(
            base_url,
            headers=headers,
            json={"inputs": {"video_base64": b64encode(video.read_bytes()).decode("ascii")}},
            timeout=request_timeout,
        )
    else:
        with video.open("rb") as handle:
            response = httpx.post(
                f"{base_url}/predict",
                headers=headers,
                files={"video": ("stimulus.mp4", handle, "video/mp4")},
                timeout=request_timeout,
            )
    if protocol != "hf-space":
        if not response.is_success:
            raise RuntimeError(f"Hosted TRIBE returned HTTP {response.status_code}: {response.text[-1000:].strip()}")
        try:
            if protocol == "hf-toolkit":
                payload = response.json()
                if isinstance(payload, list) and len(payload) == 1:
                    payload = payload[0]
                encoded = payload.get("prediction_npy_base64") if isinstance(payload, dict) else None
                if not isinstance(encoded, str):
                    raise ValueError("prediction_npy_base64 is missing")
                predicted = np.load(io.BytesIO(b64decode(encoded, validate=True)), allow_pickle=False)
            else:
                predicted = np.load(io.BytesIO(response.content), allow_pickle=False)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Hosted TRIBE returned an invalid NumPy prediction payload.") from exc
    expected_vertices = int(settings.get("surface.total_vertices", 20484))
    if predicted.ndim != 1 or predicted.size != expected_vertices:
        raise RuntimeError(f"Hosted TRIBE returned shape {predicted.shape}; expected {expected_vertices} cortical vertices.")
    return np.asarray(predicted, dtype=np.float32)


def _plain_language_reading() -> str:
    """Research-safe text shared by the on-screen and downloaded reports."""
    return """
    <h2>Plain-language reading</h2>
    <p>The observed scan and TRIBE's reference response do not match exactly. The deviation map shows the parts of the displayed cortex where that difference is relatively greater for this analysis.</p>
    <p>This is a comparison with a modelled average response to the selected stimulus; it is not a picture of brain damage, a lesion, or a confirmed abnormality. It cannot identify ADHD, a language problem, or any other diagnosis.</p>
    <p>The result is especially exploratory here because one still image was converted into a short silent clip, while the observed BOLD volume is projected onto a standard cortical display. A clinical interpretation would require the original time-matched task, participant-level surface preprocessing, quality checks, and review by a qualified clinician or researcher.</p>
    """


def _run(settings: Settings, run_id: str, bold: Path, image: Path) -> None:
    try:
        hosted = _hosted_endpoint(settings)
        device = tribe_model.resolve_device(str(settings.get("tribe.device", "auto")))
        if hosted is None and not device.cuda_available:
            raise RuntimeError(
                "Real image-to-fMRI TRIBE inference requires an NVIDIA CUDA GPU exposed to Docker. "
                "This host currently has no CUDA device available; the official V-JEPA2 visual encoder "
                "is 4.14 GB and cannot run reliably in a CPU-only container. Configure tribe.hosted "
                "to use the protected Hugging Face GPU endpoint instead."
            )
        meta = _bold_metadata(bold)
        # One still image contains no temporal information.  A single 4-second
        # visual clip is enough for TRIBE's visual encoder and avoids needlessly
        # repeating the same pixels across the entire 278-second fMRI run.
        duration = 4.0
        directory = _root(settings) / run_id
        video = directory / "stimulus.mp4"
        _write_state(settings, run_id, status="CREATING_VIDEO", progress=0.15, bold=str(bold), image=str(image), approximate=True)
        _make_video(image, video, duration)
        if hosted is not None:
            _write_state(settings, run_id, status="RUNNING_HOSTED_TRIBE", progress=0.40, bold=str(bold), image=str(image), video=str(video), approximate=True, backend="hosted-hf")
            predicted_map = _predict_hosted(video, hosted, settings)
            backend = "hosted-hf"
        else:
            _write_state(settings, run_id, status="RUNNING_TRIBE", progress=0.40, bold=str(bold), image=str(image), video=str(video), approximate=True, backend="local-gpu")
            loaded = tribe_model.load(settings, visual_only=True)
            if loaded.is_mock or loaded.handle is None:
                raise RuntimeError("Real TRIBE v2 is required for an exploratory run.")
            # This experiment deliberately makes a silent video by repeating a
            # still image.  Do not call TribeModel.get_events_dataframe(): it adds
            # a speech-transcription pipeline, and even the audio-only helper
            # downloads a 2.3 GB audio encoder.  An image-derived clip has no audio
            # information, so its event table intentionally contains Video only.
            events = standardize_events(pd.DataFrame([{
                "type": "Video", "filepath": str(video), "start": 0,
                "timeline": "default", "subject": "default",
            }]))
            predicted, _segments = loaded.handle.predict(events=events)
            predicted_map = np.asarray(predicted, dtype=np.float32).mean(axis=0)
            backend = "local-gpu"
        predicted_map, _flat = safe_zscore(predicted_map.reshape(1, -1), axis=1)
        predicted_map = predicted_map.reshape(-1).astype(np.float32)
        _write_state(settings, run_id, status="PROJECTING_FMRI", progress=0.75, bold=str(bold), image=str(image), video=str(video), approximate=True, backend=backend)
        observed_map = _volume_map(bold, predicted_map.size)
        deviation = np.abs(observed_map - predicted_map).astype(np.float32)
        np.save(directory / "observed.npy", observed_map)
        np.save(directory / "predicted.npy", predicted_map)
        np.save(directory / "deviation.npy", deviation)
        report = directory / "report.html"
        report.write_text(f"""<!doctype html><title>Exploratory TRIBE deviation report</title><h1>Exploratory TRIBE deviation report</h1><p><strong>Research use only.</strong> This report is exploratory. The observed volume was interpolated to fsaverage5 for visualization and is not a FreeSurfer/fMRIPrep surface analysis.</p>{_plain_language_reading()}<h2>Technical details</h2><ul><li>TRIBE backend: {backend}</li><li>BOLD: {bold.name}</li><li>Image: {image.name}</li><li>Generated video: {video.name}</li><li>TR: {meta['tr_sec']} s</li><li>Volumes: {meta['n_volumes']}</li><li>Mean absolute deviation: {float(np.mean(deviation)):.4f}</li></ul>""", encoding="utf-8")
        _write_state(settings, run_id, status="DONE", progress=1.0, bold=str(bold), image=str(image), video=str(video), approximate=True, backend=backend, summary={"mean_deviation": float(np.mean(deviation)), "generated_video_sec": duration, **meta})
    except Exception as exc:  # noqa: BLE001
        _write_state(settings, run_id, status="FAILED", progress=1.0, approximate=True, error=f"{type(exc).__name__}: {exc}")


def _run_imported_prediction(settings: Settings, run_id: str, bold: Path, image: Path, prediction_file: Path) -> None:
    """Complete local analysis after a one-shot free GPU notebook prediction."""
    try:
        _write_state(settings, run_id, status="READING_TRIBE_PREDICTION", progress=0.35, bold=str(bold), image=str(image), approximate=True, backend="free-gpu-import")
        predicted_map = np.load(prediction_file, allow_pickle=False)
        expected_vertices = int(settings.get("surface.total_vertices", 20484))
        if predicted_map.ndim != 1 or predicted_map.size != expected_vertices:
            raise RuntimeError(f"Uploaded TRIBE prediction has shape {predicted_map.shape}; expected {expected_vertices} cortical vertices.")
        meta = _bold_metadata(bold)
        predicted_map, _flat = safe_zscore(np.asarray(predicted_map, dtype=np.float32).reshape(1, -1), axis=1)
        predicted_map = predicted_map.reshape(-1).astype(np.float32)
        directory = _root(settings) / run_id
        _write_state(settings, run_id, status="PROJECTING_FMRI", progress=0.75, bold=str(bold), image=str(image), approximate=True, backend="free-gpu-import")
        observed_map = _volume_map(bold, predicted_map.size)
        deviation = np.abs(observed_map - predicted_map).astype(np.float32)
        np.save(directory / "observed.npy", observed_map)
        np.save(directory / "predicted.npy", predicted_map)
        np.save(directory / "deviation.npy", deviation)
        report = directory / "report.html"
        report.write_text(f"""<!doctype html><title>Exploratory TRIBE deviation report</title><h1>Exploratory TRIBE deviation report</h1><p><strong>Research use only.</strong> This report is exploratory. The observed volume was interpolated to fsaverage5 for visualization and is not a FreeSurfer/fMRIPrep surface analysis.</p>{_plain_language_reading()}<h2>Technical details</h2><ul><li>TRIBE backend: free interactive GPU import</li><li>BOLD: {bold.name}</li><li>Image: {image.name}</li><li>TR: {meta['tr_sec']} s</li><li>Volumes: {meta['n_volumes']}</li><li>Mean absolute deviation: {float(np.mean(deviation)):.4f}</li></ul>""", encoding="utf-8")
        _write_state(settings, run_id, status="DONE", progress=1.0, bold=str(bold), image=str(image), approximate=True, backend="free-gpu-import", summary={"mean_deviation": float(np.mean(deviation)), "generated_video_sec": 4.0, **meta})
    except Exception as exc:  # noqa: BLE001
        _write_state(settings, run_id, status="FAILED", progress=1.0, approximate=True, error=f"{type(exc).__name__}: {exc}")


@router.post("/runs")
def start_run(payload: dict, background: BackgroundTasks, settings: Settings = Depends(get_settings)) -> dict:
    bold = _safe_path(settings, str(payload.get("bold_id", "")))
    image = _safe_path(settings, str(payload.get("image_id", "")), images=True)
    run_id = uuid.uuid4().hex
    _write_state(settings, run_id, status="QUEUED", progress=0.0, approximate=True)
    background.add_task(_run, settings, run_id, bold, image)
    return {"id": run_id, "status": "QUEUED", "approximate": True}


@router.post("/runs/imported")
async def start_imported_run(
    background: BackgroundTasks,
    bold_id: str = Form(...),
    image_id: str = Form(...),
    prediction: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Accept the `.npy` from a free Colab/Kaggle TRIBE GPU session."""
    if Path(prediction.filename or "prediction.npy").suffix.lower() != ".npy":
        raise HTTPException(400, "Upload TRIBE's .npy prediction file.")
    payload = await prediction.read()
    if not payload or len(payload) > 2 * 1024 * 1024:
        raise HTTPException(400, "TRIBE prediction must be a non-empty .npy file smaller than 2 MB.")
    try:
        array = np.load(io.BytesIO(payload), allow_pickle=False)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, "The uploaded file is not a valid NumPy .npy array.") from exc
    expected_vertices = int(settings.get("surface.total_vertices", 20484))
    if array.ndim != 1 or array.size != expected_vertices:
        raise HTTPException(400, f"Expected a one-dimensional TRIBE map with {expected_vertices} vertices; got {array.shape}.")
    bold = _safe_path(settings, bold_id)
    image = _safe_path(settings, image_id, images=True)
    run_id = uuid.uuid4().hex
    directory = _root(settings) / run_id
    directory.mkdir(parents=True, exist_ok=True)
    prediction_path = directory / "free_gpu_prediction.npy"
    np.save(prediction_path, np.asarray(array, dtype=np.float32), allow_pickle=False)
    _write_state(settings, run_id, status="QUEUED", progress=0.0, approximate=True, backend="free-gpu-import")
    background.add_task(_run_imported_prediction, settings, run_id, bold, image, prediction_path)
    return {"id": run_id, "status": "QUEUED", "approximate": True, "backend": "free-gpu-import"}


@router.get("/runs/{run_id}")
def run_state(run_id: str, settings: Settings = Depends(get_settings)) -> dict:
    state = _read_state(settings, run_id)
    # Keep the client-side poll response self-contained.  Without the ID the
    # browser can show DONE but cannot construct URLs for maps or the report.
    return {"id": run_id, **state}


@router.get("/runs/{run_id}/maps/{kind}")
def map_file(run_id: str, kind: str, settings: Settings = Depends(get_settings)) -> Response:
    if kind not in {"observed", "predicted", "deviation"}:
        raise HTTPException(404, "Unknown map")
    path = _root(settings) / run_id / f"{kind}.npy"
    if not path.exists():
        raise HTTPException(409, "Map is not ready yet.")
    # The WebGL viewer creates a Float32Array directly from this response.
    # np.save() includes a 128-byte .npy header, which would otherwise be
    # interpreted as 32 additional float values and produce a bogus mismatch.
    values = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    return Response(values.tobytes(order="C"), media_type="application/octet-stream", headers={"X-Array-Dtype": "float32"})


@router.get("/runs/{run_id}/report")
def report_file(run_id: str, settings: Settings = Depends(get_settings)):
    path = _root(settings) / run_id / "report.html"
    if not path.exists():
        raise HTTPException(409, "Report is not ready yet.")
    return FileResponse(path, media_type="text/html", filename=f"tribe-exploratory-{run_id}.html")
