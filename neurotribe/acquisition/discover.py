"""Filesystem discovery of pre-existing HBN assets.

The operator may already have downloaded HBN files. We therefore *inspect
before we fetch*: every candidate is classified by content (not just filename),
hashed, and registered. Nothing is re-downloaded blindly.
"""

from __future__ import annotations

import csv
import io
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from neurotribe.config import Settings
from neurotribe.database.enums import AssetKind, AssetStatus
from neurotribe.database.models import DataAsset
from neurotribe.database.repository import record_audit
from neurotribe.hashing import hash_file
from neurotribe.logging_setup import get_logger

log = get_logger(__name__)

# Directory names never worth walking into.
_SKIP_DIRS = {
    ".git", ".datalad", "node_modules", "__pycache__", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build", ".next",
    ".astro", "site-packages",
}

# Signature columns used to classify a CSV by *content*.
_METADATA_SIGNATURE = {"anonymized id", "participant_id", "subject", "eid", "release_number"}
_MRIQC_SIGNATURE = {"bids_name", "efc", "fber", "fd_mean", "tsnr", "dvars_std", "snr"}
_PHENOTYPE_SIGNATURE = {"dx_01", "diagnosis_clinicianconsensus", "dx_01_cat", "dx_01_sub"}


@dataclass
class Candidate:
    path: Path
    kind: AssetKind
    confidence: float
    evidence: dict = field(default_factory=dict)
    is_directory: bool = False


def _iter_files(root: Path, max_depth: int) -> Iterable[Path]:
    root = root.resolve()
    if not root.exists():
        return
    root_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        depth = len(current.parts) - root_depth
        if depth >= max_depth:
            dirnames[:] = []
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            yield current / name


def _read_header(path: Path, max_bytes: int = 65536) -> list[str]:
    """Read the first CSV/TSV header row, tolerating BOM and odd delimiters."""
    try:
        with path.open("rb") as handle:
            blob = handle.read(max_bytes)
    except OSError:
        return []
    text = blob.decode("utf-8-sig", errors="replace")
    first_line = text.splitlines()[0] if text.splitlines() else ""
    delimiter = "\t" if first_line.count("\t") > first_line.count(",") else ","
    try:
        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        header = next(reader, [])
    except csv.Error:
        return []
    return [h.strip().strip('"').lower() for h in header]


def classify_csv(path: Path) -> Candidate | None:
    header = _read_header(path)
    if not header:
        return None
    header_set = set(header)
    name = path.name.lower()

    def overlap(signature: set[str]) -> int:
        return len(header_set & signature)

    scores = {
        AssetKind.MRIQC_FUNCTIONAL: overlap(_MRIQC_SIGNATURE) * 1.0,
        AssetKind.HBN_METADATA: overlap(_METADATA_SIGNATURE) * 1.0,
        AssetKind.PHENOTYPE_CSV: overlap(_PHENOTYPE_SIGNATURE) * 1.5,
    }
    # Filename evidence is a tiebreaker only, never the sole basis.
    if "iqm" in name or "mriqc" in name:
        scores[AssetKind.MRIQC_FUNCTIONAL] += 1.5
    if "metadata" in name:
        scores[AssetKind.HBN_METADATA] += 1.5
    if "diagnosis" in name or "consensus" in name or "phenotyp" in name:
        scores[AssetKind.PHENOTYPE_CSV] += 1.5

    kind, score = max(scores.items(), key=lambda item: item[1])
    if score < 1.0:
        return None

    # Anatomical vs functional MRIQC is decided by IQM column families.
    if kind is AssetKind.MRIQC_FUNCTIONAL:
        functional_markers = {"tsnr", "dvars_std", "fd_mean", "gsr_x", "gcor"}
        anatomical_markers = {"cjv", "cnr", "wm2max", "icvs_csf", "rpve_csf"}
        if not (header_set & functional_markers) and (header_set & anatomical_markers):
            kind = AssetKind.MRIQC_ANATOMICAL

    return Candidate(
        path=path, kind=kind, confidence=min(1.0, score / 4.0),
        evidence={"header_sample": header[:25], "n_columns": len(header)},
    )


def detect_bids_roots(search_roots: Iterable[Path], max_depth: int) -> list[Candidate]:
    """Find BIDS dataset roots by locating ``dataset_description.json``."""
    found: dict[Path, Candidate] = {}
    for root in search_roots:
        root = Path(root).resolve()
        if not root.exists():
            continue
        for path in _iter_files(root, max_depth + 2):
            if path.name != "dataset_description.json":
                continue
            bids_root = path.parent
            if bids_root in found:
                continue
            evidence = inspect_bids_root(bids_root)
            found[bids_root] = Candidate(
                path=bids_root, kind=AssetKind.BIDS_ROOT, confidence=1.0,
                evidence=evidence, is_directory=True,
            )
    return list(found.values())


def inspect_bids_root(bids_root: Path) -> dict:
    """Determine whether a BIDS tree holds real binaries or annex placeholders.

    A DataLad/git-annex dataset stores *symlinks or pointer files* rather than
    NIfTI content. Treating a placeholder as data would silently corrupt the
    pipeline, so we detect it explicitly.
    """
    info: dict = {
        "is_datalad": (bids_root / ".datalad").exists(),
        "is_git": (bids_root / ".git").exists(),
        "has_gitmodules": (bids_root / ".gitmodules").exists(),
        "n_subject_dirs": 0,
        "content_present": None,
        "annex_placeholders": 0,
        "real_nifti_files": 0,
        "sampled_nifti": 0,
    }
    try:
        subject_dirs = [d for d in bids_root.iterdir() if d.is_dir() and d.name.startswith("sub-")]
    except OSError:
        subject_dirs = []
    info["n_subject_dirs"] = len(subject_dirs)

    # Sample a handful of NIfTI files to see whether content is materialised.
    sampled = 0
    for subject_dir in subject_dirs[:5]:
        for nii in subject_dir.rglob("*.nii*"):
            sampled += 1
            try:
                if nii.is_symlink():
                    target = os.readlink(nii)
                    if ".git/annex" in str(target).replace("\\", "/"):
                        info["annex_placeholders"] += 1
                    else:
                        info["real_nifti_files"] += 1
                elif nii.stat().st_size < 4096:
                    # git-annex pointer files are tiny text stubs.
                    info["annex_placeholders"] += 1
                else:
                    info["real_nifti_files"] += 1
            except OSError:
                pass
            if sampled >= 20:
                break
        if sampled >= 20:
            break

    info["sampled_nifti"] = sampled
    if sampled == 0:
        info["content_present"] = None
    else:
        info["content_present"] = info["real_nifti_files"] > info["annex_placeholders"]
    return info


def classify_video(path: Path, allowed: set[str]) -> Candidate | None:
    suffix = path.suffix.lower().lstrip(".")
    if suffix not in allowed:
        return None
    return Candidate(path=path, kind=AssetKind.STIMULUS_VIDEO, confidence=0.6,
                     evidence={"container": suffix})


def scan(settings: Settings) -> list[Candidate]:
    """Walk configured search roots and classify everything interesting."""
    max_depth = int(settings.get("discovery.max_depth", 4))
    roots: list[Path] = []
    for entry in settings.get("discovery.search_roots", ["."]):
        candidate_root = Path(entry)
        if not candidate_root.is_absolute():
            candidate_root = settings.root / candidate_root
        roots.append(candidate_root.resolve())
    # Always include the well-known intake directories.
    roots.extend([settings.paths.phenotype_incoming, settings.paths.stimuli_incoming,
                  settings.paths.metadata, settings.paths.external])

    allowed_containers = {c.lower() for c in settings.get("stimulus.validation.allowed_containers", [])}
    seen: set[Path] = set()
    candidates: list[Candidate] = []

    for root in dict.fromkeys(roots):
        for path in _iter_files(root, max_depth):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            suffix = resolved.suffix.lower()
            candidate: Candidate | None = None
            if suffix in {".csv", ".tsv", ".txt"}:
                candidate = classify_csv(resolved)
            elif suffix.lstrip(".") in allowed_containers:
                candidate = classify_video(resolved, allowed_containers)
            if candidate is not None:
                candidates.append(candidate)

    candidates.extend(detect_bids_roots(dict.fromkeys(roots), max_depth))
    log.info("Discovery scan complete", extra={"n_candidates": len(candidates),
                                               "roots": [str(r) for r in dict.fromkeys(roots)]})
    return candidates


def register(session: Session, settings: Settings, candidates: Iterable[Candidate]) -> list[DataAsset]:
    """Upsert candidates into the data registry with hashes and timestamps."""
    full_max = int(settings.get("discovery.full_hash_max_bytes", 256 * 1024 * 1024))
    chunk = int(settings.get("discovery.partial_hash_chunk_bytes", 8 * 1024 * 1024))
    registered: list[DataAsset] = []

    for candidate in candidates:
        absolute = str(candidate.path.resolve())
        asset = session.execute(
            select(DataAsset).where(DataAsset.absolute_path == absolute)
        ).scalar_one_or_none()

        try:
            stat = candidate.path.stat()
        except OSError as exc:
            log.warning("Candidate vanished during registration",
                        extra={"path": absolute, "error": str(exc)})
            continue

        modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        if candidate.is_directory:
            digest, size, partial = None, None, False
        else:
            file_digest = hash_file(candidate.path, full_max_bytes=full_max,
                                    partial_chunk_bytes=chunk)
            digest, size, partial = file_digest.sha256, file_digest.size_bytes, file_digest.partial

        try:
            relative = str(candidate.path.resolve().relative_to(settings.root))
        except ValueError:
            relative = absolute

        if asset is None:
            asset = DataAsset(path=relative, absolute_path=absolute)
            session.add(asset)
            record_audit(session, "asset.discovered", entity_type="data_asset",
                         summary=relative, payload={"kind": candidate.kind.value})
        elif asset.sha256 != digest:
            record_audit(session, "asset.changed", entity_type="data_asset", entity_id=asset.id,
                         summary=relative, payload={"old_sha256": asset.sha256, "new_sha256": digest})

        asset.kind = candidate.kind.value
        asset.path = relative
        asset.size_bytes = size
        asset.sha256 = digest
        asset.hash_is_partial = partial
        asset.modified_at = modified
        asset.is_directory = candidate.is_directory
        # JSON column defaults apply at INSERT, so a freshly constructed row
        # still has None here until it is flushed.
        asset.validation_report = {**(asset.validation_report or {}),
                                   "discovery": candidate.evidence,
                                   "confidence": candidate.confidence}
        if asset.status == AssetStatus.MISSING.value:
            asset.status = AssetStatus.DISCOVERED.value
        # Phenotype exports are DUA-protected and must never be published.
        asset.protected = candidate.kind is AssetKind.PHENOTYPE_CSV
        registered.append(asset)

    session.flush()
    return registered


def run_discovery(session: Session, settings: Settings) -> dict:
    """Full discovery pass. Returns a summary suitable for the dashboard."""
    settings.paths.ensure()
    candidates = scan(settings)
    assets = register(session, settings, candidates)
    by_kind: dict[str, int] = {}
    for asset in assets:
        by_kind[asset.kind] = by_kind.get(asset.kind, 0) + 1
    return {
        "n_assets": len(assets),
        "by_kind": by_kind,
        "assets": [
            {"kind": a.kind, "path": a.path, "size_bytes": a.size_bytes,
             "sha256": a.sha256, "partial": a.hash_is_partial}
            for a in assets
        ],
    }
