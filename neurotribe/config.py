"""Configuration loading, layering and hashing.

Layering order (later wins):
    config/default.yaml
    config/<profile>.yaml
    NEUROTRIBE__<SECTION>__<KEY> environment overrides

The fully-resolved scientific configuration is hashed into
``analysis_config_hash`` which is embedded in every provenance manifest. Two
runs with different scientific parameters can therefore never be confused.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import yaml

ENV_PREFIX = "NEUROTRIBE__"

# Sections that materially affect a scientific result. Only these participate in
# ``analysis_config_hash`` so that cosmetic changes (logging, paths) do not
# invalidate an otherwise identical analysis.
SCIENTIFIC_SECTIONS = (
    "stimulus",
    "bids",
    "phenotype",
    "cohort",
    "qc",
    "preprocessing",
    "surface",
    "tribe",
    "alignment",
    "analysis",
)


def _repo_root() -> Path:
    env = os.environ.get("NEUROTRIBE_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parent.parent


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _coerce_scalar(raw: str) -> Any:
    lowered = raw.strip().lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "none", ""}:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if raw.startswith("[") or raw.startswith("{"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return raw


def _apply_env_overrides(cfg: dict) -> dict:
    """Apply ``NEUROTRIBE__A__B=value`` style overrides."""
    out = copy.deepcopy(cfg)
    for env_key, raw in sorted(os.environ.items()):
        if not env_key.startswith(ENV_PREFIX):
            continue
        path = [p.lower() for p in env_key[len(ENV_PREFIX) :].split("__") if p]
        if not path:
            continue
        cursor: dict = out
        for part in path[:-1]:
            nxt = cursor.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[part] = nxt
            cursor = nxt
        cursor[path[-1]] = _coerce_scalar(raw)
    return out


def stable_hash(obj: Any, *, length: int = 16) -> str:
    """Deterministic short hash of any JSON-serialisable object."""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


@dataclass(frozen=True)
class Paths:
    """Resolved absolute filesystem locations."""

    root: Path
    data: Path
    external: Path
    metadata: Path
    phenotype: Path
    phenotype_incoming: Path
    stimuli: Path
    stimuli_incoming: Path
    raw: Path
    derivatives: Path
    fmriprep_out: Path
    freesurfer_out: Path
    tribe: Path
    analysis: Path
    reports: Path
    cache: Path
    models_cache: Path
    downloads_cache: Path
    work: Path

    def all(self) -> Iterable[Path]:
        for field in self.__dataclass_fields__:  # type: ignore[attr-defined]
            yield getattr(self, field)

    def ensure(self) -> None:
        for path in self.all():
            path.mkdir(parents=True, exist_ok=True)


class Settings:
    """Immutable view over the layered configuration."""

    def __init__(self, raw: dict, profile: str, root: Path) -> None:
        self._raw = raw
        self.profile = profile
        self.root = root
        p = raw.get("paths", {})
        self.paths = Paths(
            root=root,
            **{
                key: (root / p[key]).resolve()
                for key in Paths.__dataclass_fields__  # type: ignore[attr-defined]
                if key != "root"
            },
        )

    # -- access ---------------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._raw[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        """Fetch a value by dotted path, e.g. ``settings.get('qc.motion.fd_threshold_mm')``."""
        cursor: Any = self._raw
        for part in dotted.split("."):
            if not isinstance(cursor, dict) or part not in cursor:
                return default
            cursor = cursor[part]
        return cursor

    def require(self, dotted: str) -> Any:
        sentinel = object()
        value = self.get(dotted, sentinel)
        if value is sentinel:
            raise KeyError(f"Missing required configuration key: {dotted}")
        return value

    @property
    def raw(self) -> dict:
        return copy.deepcopy(self._raw)

    # -- provenance -----------------------------------------------------
    @property
    def scientific_config(self) -> dict:
        return {k: self._raw.get(k) for k in SCIENTIFIC_SECTIONS}

    @property
    def analysis_config_hash(self) -> str:
        return stable_hash(self.scientific_config, length=16)

    @property
    def is_production(self) -> bool:
        return self.profile == "production"

    @property
    def research_use_only(self) -> bool:
        return bool(self.get("project.research_use_only", True))

    def subsection_hash(self, *sections: str) -> str:
        return stable_hash({s: self.get(s) for s in sections}, length=16)


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return loaded or {}


def load_settings(profile: str | None = None, root: Path | None = None) -> Settings:
    """Load configuration for ``profile`` (default from ``NEUROTRIBE_PROFILE``)."""
    root = (root or _repo_root()).resolve()
    profile = profile or os.environ.get("NEUROTRIBE_PROFILE", "development")
    if not re.fullmatch(r"[a-z0-9_-]+", profile):
        raise ValueError(f"Invalid profile name: {profile!r}")

    config_dir = root / "config"
    merged = _read_yaml(config_dir / "default.yaml")
    if not merged:
        raise FileNotFoundError(f"config/default.yaml not found under {root}")
    merged = _deep_merge(merged, _read_yaml(config_dir / f"{profile}.yaml"))
    merged = _apply_env_overrides(merged)
    return Settings(merged, profile=profile, root=root)


@lru_cache(maxsize=8)
def _cached_settings(profile: str, root: str) -> Settings:
    return load_settings(profile, Path(root))


def get_settings() -> Settings:
    """Process-wide cached settings."""
    profile = os.environ.get("NEUROTRIBE_PROFILE", "development")
    return _cached_settings(profile, str(_repo_root()))


def reset_settings_cache() -> None:
    _cached_settings.cache_clear()
