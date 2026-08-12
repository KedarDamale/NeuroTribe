"""TRIBE v2 model loading, device selection and version pinning.

Two backends exist:

* ``real``  - the official ``tribev2`` package. The only backend permitted to
  produce a scientific result.
* ``mock``  - a deterministic stand-in used exclusively for fixtures, smoke
  tests and UI development. Every artefact it produces is stamped
  ``backend: mock`` and is rejected by the production profile.
"""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from neurotribe.config import Settings
from neurotribe.database.enums import TribeBackend
from neurotribe.logging_setup import get_logger

log = get_logger(__name__)


class TribeUnavailable(RuntimeError):
    """The real TRIBE backend could not be loaded."""


@dataclass
class DeviceInfo:
    device: str = "cpu"
    name: str | None = None
    vram_gb: float | None = None
    cuda_available: bool = False
    mps_available: bool = False

    def to_dict(self) -> dict:
        return {
            "device": self.device, "name": self.name, "vram_gb": self.vram_gb,
            "cuda_available": self.cuda_available, "mps_available": self.mps_available,
        }


def resolve_device(preference: str = "auto") -> DeviceInfo:
    """Pick an accelerator, falling back to CPU rather than crashing."""
    info = DeviceInfo()
    try:
        import torch
    except ImportError:
        log.info("PyTorch unavailable; TRIBE will use the CPU fallback path")
        return info

    info.cuda_available = bool(torch.cuda.is_available())
    info.mps_available = bool(getattr(torch.backends, "mps", None)
                              and torch.backends.mps.is_available())

    if preference == "cuda" and not info.cuda_available:
        log.warning("CUDA requested but unavailable; falling back to CPU")
        preference = "cpu"
    if preference == "mps" and not info.mps_available:
        preference = "cpu"

    if preference == "auto":
        preference = "cuda" if info.cuda_available else ("mps" if info.mps_available else "cpu")

    info.device = preference
    if preference == "cuda":
        try:
            properties = torch.cuda.get_device_properties(0)
            info.name = properties.name
            info.vram_gb = round(properties.total_memory / (1024 ** 3), 2)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not query CUDA device", extra={"error": str(exc)})
    elif preference == "mps":
        info.name = "Apple Metal"
    else:
        info.name = platform.processor() or "cpu"
    return info


@dataclass
class TribeVersion:
    package_version: str | None = None
    commit: str | None = None
    model_id: str | None = None
    model_revision: str | None = None
    source: str | None = None

    def to_dict(self) -> dict:
        return {
            "tribe_version": self.package_version, "tribe_commit": self.commit,
            "tribe_model": self.model_id, "tribe_model_revision": self.model_revision,
            "source": self.source,
        }


def _git_commit_of(package_path: Path) -> str | None:
    """Resolve the git commit of an editable/source install of TRIBE."""
    git = shutil.which("git")
    if git is None:
        return None
    current = package_path
    for _ in range(4):
        if (current / ".git").exists():
            try:
                result = subprocess.run(
                    [git, "rev-parse", "HEAD"], cwd=str(current),
                    capture_output=True, text=True, timeout=30, check=False,
                )
                if result.returncode == 0:
                    return result.stdout.strip()
            except (OSError, subprocess.TimeoutExpired):
                return None
        current = current.parent
    return None


def probe_tribe() -> tuple[bool, TribeVersion, str | None]:
    """Check whether the real TRIBE package is importable and pin its version."""
    version = TribeVersion()
    try:
        module = importlib.import_module("tribev2")
    except ImportError as exc:
        return False, version, str(exc)

    version.package_version = getattr(module, "__version__", None)
    module_file = getattr(module, "__file__", None)
    if module_file:
        version.source = str(Path(module_file).parent)
        version.commit = _git_commit_of(Path(module_file).parent)
    if not hasattr(module, "TribeModel"):
        return False, version, "tribev2 is importable but exposes no TribeModel"
    return True, version, None


@dataclass
class LoadedModel:
    backend: TribeBackend
    handle: object | None
    version: TribeVersion
    device: DeviceInfo
    cache_folder: Path
    notes: list[str] = field(default_factory=list)

    @property
    def is_mock(self) -> bool:
        return self.backend is TribeBackend.MOCK

    def manifest(self) -> dict:
        return {
            "backend": self.backend.value,
            **self.version.to_dict(),
            "device": self.device.to_dict(),
            "cache_folder": str(self.cache_folder),
            "notes": self.notes,
        }


def load(settings: Settings) -> LoadedModel:
    """Load TRIBE according to the configured backend policy."""
    requested = str(settings.get("tribe.backend", "auto")).lower()
    cache_folder = settings.paths.models_cache
    cache_folder.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(settings.get("tribe.device", "auto")))

    available, version, error = probe_tribe()
    version.model_id = str(settings.get("tribe.model_id", "facebook/tribev2"))
    version.model_revision = str(settings.get("tribe.model_revision", "main"))

    if requested == "mock":
        if settings.is_production:
            raise TribeUnavailable(
                "The mock TRIBE backend is explicitly requested but forbidden in the "
                "production profile."
            )
        return _load_mock(settings, version, device, cache_folder,
                          ["Mock backend explicitly requested by configuration."])

    if available:
        try:
            module = importlib.import_module("tribev2")
            handle = module.TribeModel.from_pretrained(  # type: ignore[attr-defined]
                version.model_id, cache_folder=str(cache_folder),
            )
        except Exception as exc:  # noqa: BLE001 - model loading fails many ways
            if requested == "real" or settings.is_production:
                raise TribeUnavailable(
                    f"TRIBE v2 is installed but the model could not be loaded: {exc}"
                ) from exc
            log.warning("TRIBE model load failed; using mock backend",
                        extra={"error": str(exc)})
            return _load_mock(settings, version, device, cache_folder,
                              [f"Real model load failed: {exc}"])

        # Move to the selected device when the handle supports it.
        try:
            if hasattr(handle, "to"):
                handle.to(device.device)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not move TRIBE model to device",
                        extra={"device": device.device, "error": str(exc)})

        log.info("TRIBE v2 loaded", extra={"device": device.device,
                                           "commit": version.commit})
        return LoadedModel(TribeBackend.REAL, handle, version, device, cache_folder)

    message = f"tribev2 is not importable: {error}"
    if requested == "real" or settings.is_production:
        raise TribeUnavailable(message)
    return _load_mock(settings, version, device, cache_folder, [message])


def _load_mock(settings: Settings, version: TribeVersion, device: DeviceInfo,
               cache_folder: Path, notes: list[str]) -> LoadedModel:
    from neurotribe.tribe.mock import MockTribeModel

    version.package_version = f"mock-{settings.get('project.version', '0')}"
    version.commit = None
    handle = MockTribeModel(
        n_vertices=int(settings.get("surface.total_vertices", 20484)),
        hemi_order=list(settings.get("surface.hemi_order", ["L", "R"])),
    )
    log.warning("Using the MOCK TRIBE backend - outputs are NOT scientific results",
                extra={"notes": notes})
    return LoadedModel(TribeBackend.MOCK, handle, version, device, cache_folder,
                       notes + ["Outputs are synthetic and excluded from any final analysis."])


def install_hint(settings: Settings) -> dict:
    """Instructions surfaced in the UI when TRIBE is unavailable."""
    return {
        "repo_url": settings.get("tribe.repo_url"),
        "git_ref": settings.get("tribe.git_ref"),
        "model_id": settings.get("tribe.model_id"),
        "command": (
            f"pip install 'git+{settings.get('tribe.repo_url')}@"
            f"{settings.get('tribe.git_ref')}'"
        ),
        "licence": "CC-BY-NC-4.0 (research / non-commercial use)",
        "note": (
            "TRIBE v2 is used as a pretrained reference model only. This system "
            "never trains or fine-tunes it."
        ),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "hf_home": os.environ.get("HF_HOME"),
        },
    }
