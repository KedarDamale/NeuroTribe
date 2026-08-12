"""Hardware and software readiness probe.

Detects CPU / RAM / GPU / VRAM / disk / CUDA / Docker / WSL and reports a
readiness verdict. Missing CUDA is never fatal: TRIBE falls back to the
validated CPU path rather than crashing.
"""

from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from neurotribe.config import Settings
from neurotribe.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class SystemProbeResult:
    hostname: str = ""
    platform_name: str = ""
    python_version: str = ""
    cpu_count: int = 0
    ram_gb: float | None = None
    free_disk_gb: float | None = None
    total_disk_gb: float | None = None
    disk_path: str | None = None
    gpu_name: str | None = None
    vram_gb: float | None = None
    cuda_available: bool = False
    torch_version: str | None = None
    docker_available: bool = False
    docker_version: str | None = None
    docker_cpus: int | None = None
    docker_memory_gb: float | None = None
    wsl: bool = False
    ffmpeg_available: bool = False
    tribe_available: bool = False
    freesurfer_license: bool = False
    ready: bool = False
    warnings: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "hostname": self.hostname, "platform": self.platform_name,
            "python_version": self.python_version, "cpu_count": self.cpu_count,
            "ram_gb": self.ram_gb, "free_disk_gb": self.free_disk_gb,
            "total_disk_gb": self.total_disk_gb, "disk_path": self.disk_path,
            "gpu_name": self.gpu_name,
            "vram_gb": self.vram_gb, "cuda_available": self.cuda_available,
            "torch_version": self.torch_version,
            "docker_available": self.docker_available,
            "docker_version": self.docker_version, "docker_cpus": self.docker_cpus,
            "docker_memory_gb": self.docker_memory_gb, "wsl": self.wsl,
            "ffmpeg_available": self.ffmpeg_available,
            "tribe_available": self.tribe_available,
            "freesurfer_license": self.freesurfer_license,
            "ready": self.ready, "warnings": self.warnings, "blockers": self.blockers,
        }


def _ram_gb() -> float | None:
    try:
        import psutil

        return round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except ImportError:
        pass
    try:
        if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return round(pages * page_size / (1024 ** 3), 2)
    except (ValueError, OSError):
        pass
    return None


def _detect_wsl() -> bool:
    if platform.system() != "Linux":
        return False
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        return False


def _docker_info() -> tuple[bool, str | None, int | None, float | None]:
    executable = shutil.which("docker")
    if executable is None:
        return False, None, None, None
    try:
        result = subprocess.run(
            [executable, "info", "--format", "{{.ServerVersion}}|{{.NCPU}}|{{.MemTotal}}"],
            capture_output=True, text=True, timeout=90, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, None, None, None
    if result.returncode != 0:
        return False, None, None, None
    parts = result.stdout.strip().split("|")
    version = parts[0] if parts else None
    cpus = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
    memory = None
    if len(parts) > 2 and parts[2].isdigit():
        memory = round(int(parts[2]) / (1024 ** 3), 2)
    return True, version, cpus, memory


def probe(settings: Settings) -> SystemProbeResult:
    """Full readiness probe. Never raises."""
    result = SystemProbeResult(
        hostname=socket.gethostname(),
        platform_name=f"{platform.system()} {platform.release()} ({platform.machine()})",
        python_version=sys.version.split()[0],
        cpu_count=os.cpu_count() or 1,
        ram_gb=_ram_gb(),
        wsl=_detect_wsl(),
    )

    # Measure the data directory, not the process root: under Docker the root is
    # the container overlay filesystem and would report the wrong volume.
    settings.paths.data.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(str(settings.paths.data))
    result.free_disk_gb = round(usage.free / (1024 ** 3), 1)
    result.total_disk_gb = round(usage.total / (1024 ** 3), 1)
    result.disk_path = str(settings.paths.data)

    try:
        import torch

        result.torch_version = torch.__version__
        result.cuda_available = bool(torch.cuda.is_available())
        if result.cuda_available:
            properties = torch.cuda.get_device_properties(0)
            result.gpu_name = properties.name
            result.vram_gb = round(properties.total_memory / (1024 ** 3), 2)
    except ImportError:
        result.warnings.append("PyTorch is not installed; TRIBE cannot run.")
    except Exception as exc:  # noqa: BLE001
        result.warnings.append(f"GPU probe failed: {exc}")

    if not result.cuda_available and result.torch_version:
        result.warnings.append(
            "No CUDA accelerator detected. TRIBE will use the validated CPU path; "
            "inference will be slower but correct."
        )

    (result.docker_available, result.docker_version,
     result.docker_cpus, result.docker_memory_gb) = _docker_info()
    result.ffmpeg_available = shutil.which("ffmpeg") is not None

    from neurotribe.preprocessing.fmriprep import detect_freesurfer_license
    from neurotribe.tribe.model import probe_tribe

    result.tribe_available = probe_tribe()[0]
    result.freesurfer_license = detect_freesurfer_license(settings).found

    # -- capacity checks ------------------------------------------------
    min_free = float(settings.get("autopilot.disk.min_free_gb", 20))
    if result.free_disk_gb is not None and result.free_disk_gb < min_free:
        result.blockers.append(
            f"Only {result.free_disk_gb} GB free on the data volume "
            f"({result.disk_path}); at least {min_free} GB is required before "
            "imaging acquisition or preprocessing may start."
        )

    fmriprep_mem = float(settings.get("preprocessing.fmriprep.mem_mb", 8000)) / 1024.0
    effective_ram = result.docker_memory_gb or result.ram_gb
    if effective_ram is not None and effective_ram < fmriprep_mem:
        result.warnings.append(
            f"fMRIPrep is configured for {fmriprep_mem:.1f} GB but only "
            f"{effective_ram:.1f} GB is available to containers. Reduce "
            "preprocessing.fmriprep.mem_mb or raise the Docker memory limit."
        )
    if effective_ram is not None and effective_ram < 8:
        result.warnings.append(
            f"{effective_ram:.1f} GB RAM available to containers. fMRIPrep is "
            "memory-hungry; expect failures on full-resolution runs."
        )

    if not result.docker_available:
        result.blockers.append("Docker is unavailable; fMRIPrep cannot run.")
    if not result.ffmpeg_available:
        result.warnings.append(
            "ffmpeg/ffprobe not found; stimulus validation and the synthetic smoke "
            "test video are unavailable."
        )

    result.ready = not result.blockers
    log.info("System probe complete", extra=result.to_dict())
    return result


def persist(session, settings: Settings) -> SystemProbeResult:
    """Probe and store the snapshot for the dashboard."""
    from neurotribe.database.models import SystemProbe

    result = probe(settings)
    session.add(SystemProbe(hostname=result.hostname, payload=result.to_dict(),
                            ready=result.ready, warnings=result.warnings + result.blockers))
    return result
