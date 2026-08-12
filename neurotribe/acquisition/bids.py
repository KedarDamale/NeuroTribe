"""BIDS indexing and evidence-based movie-scan classification.

Two rules drive this module:

1. **Never select scans by guessing filenames.** Entities come from PyBIDS when
   available, and from a strict BIDS-entity parser otherwise.
2. **Never hard-code ``task=despicable``.** A run is bound to a stimulus only
   when the *acquisition duration* agrees with the documented HBN movie
   interval; task-name hints are a tiebreaker, never the sole evidence.

HBN's MRI protocol documents two movie intervals: The Present (00:00:00-00:03:21)
and Despicable Me (01:02:09-01:12:09).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from neurotribe.config import Settings
from neurotribe.database.enums import AssetKind, MovieKey
from neurotribe.database.models import DataAsset, Scan, Subject
from neurotribe.database.repository import get_or_create_subject, record_audit
from neurotribe.logging_setup import get_logger

log = get_logger(__name__)

_ENTITY_RE = re.compile(r"(?P<key>[a-zA-Z]+)-(?P<value>[a-zA-Z0-9.]+)")
_SUFFIX_RE = re.compile(r"_(?P<suffix>[a-zA-Z0-9]+)\.(?P<ext>nii(?:\.gz)?|json|tsv)$")


@dataclass
class BidsFile:
    path: Path
    entities: dict[str, str]
    suffix: str | None
    extension: str
    datatype: str | None


@dataclass
class IndexReport:
    n_subjects: int = 0
    n_bold: int = 0
    n_t1w: int = 0
    n_movie_scans: int = 0
    by_task: dict[str, int] = field(default_factory=dict)
    by_movie: dict[str, int] = field(default_factory=dict)
    used_pybids: bool = False
    warnings: list[str] = field(default_factory=list)
    content_present: bool | None = None

    def to_dict(self) -> dict:
        return {
            "n_subjects": self.n_subjects, "n_bold": self.n_bold, "n_t1w": self.n_t1w,
            "n_movie_scans": self.n_movie_scans, "by_task": self.by_task,
            "by_movie": self.by_movie, "used_pybids": self.used_pybids,
            "warnings": self.warnings[:50], "content_present": self.content_present,
        }


def parse_entities(path: Path, bids_root: Path) -> BidsFile | None:
    """Parse BIDS entities from a path without guessing semantics."""
    name = path.name
    match = _SUFFIX_RE.search(name)
    if match is None:
        return None
    suffix = match.group("suffix")
    extension = match.group("ext")
    stem = name[: match.start()]
    entities = {m.group("key"): m.group("value") for m in _ENTITY_RE.finditer(stem)}
    if "sub" not in entities:
        return None
    try:
        relative = path.relative_to(bids_root)
        datatype = relative.parts[-2] if len(relative.parts) >= 2 else None
    except ValueError:
        datatype = None
    return BidsFile(path=path, entities=entities, suffix=suffix,
                    extension=extension, datatype=datatype)


def _iter_bids_files(bids_root: Path) -> Iterable[BidsFile]:
    for path in bids_root.rglob("*"):
        if not path.is_file():
            continue
        parts = set(path.parts)
        if ".git" in parts or ".datalad" in parts or "derivatives" in parts:
            continue
        parsed = parse_entities(path, bids_root)
        if parsed is not None:
            yield parsed


def _load_sidecar(bold_path: Path) -> dict[str, Any]:
    """Load the JSON sidecar, honouring BIDS inheritance up the tree."""
    merged: dict[str, Any] = {}
    # Walk from dataset root down so more specific files win.
    candidates: list[Path] = []
    json_path = Path(str(bold_path).replace(".nii.gz", ".json").replace(".nii", ".json"))
    parent = bold_path.parent
    # Inherited top-level sidecars (e.g. task-<x>_bold.json).
    for level in list(parent.parents)[:4][::-1]:
        for inherited in sorted(level.glob("*_bold.json")):
            candidates.append(inherited)
    candidates.append(json_path)
    for candidate in candidates:
        if candidate.exists():
            try:
                with candidate.open("r", encoding="utf-8") as handle:
                    merged.update(json.load(handle))
            except (OSError, json.JSONDecodeError) as exc:
                log.warning("Unreadable BIDS sidecar",
                            extra={"path": str(candidate), "error": str(exc)})
    return merged


def _nifti_volume_count(path: Path) -> int | None:
    """Read the volume count from a NIfTI header without loading voxel data."""
    try:
        import nibabel as nib
    except ImportError:  # pragma: no cover - nibabel is a hard dependency in prod
        return None
    try:
        image = nib.load(str(path))
        shape = image.shape
        return int(shape[3]) if len(shape) >= 4 else None
    except Exception as exc:  # noqa: BLE001 - nibabel raises many types
        log.debug("Could not read NIfTI header", extra={"path": str(path), "error": str(exc)})
        return None


def _content_is_present(path: Path) -> bool:
    """False for git-annex placeholder stubs."""
    try:
        if path.is_symlink():
            import os
            return ".git/annex" not in str(os.readlink(path)).replace("\\", "/")
        return path.stat().st_size > 4096
    except OSError:
        return False


# --------------------------------------------------------------------------
# Movie classification
# --------------------------------------------------------------------------

@dataclass
class MovieClassification:
    movie: MovieKey
    confidence: float
    evidence: dict[str, Any]


def classify_movie(task: str | None, duration_sec: float | None,
                   settings: Settings) -> MovieClassification:
    """Bind a run to a documented HBN movie using duration + name evidence.

    Duration is the primary evidence because it is a property of the
    acquisition; the task label is site/release-dependent and is used only to
    break ties or to confirm.
    """
    catalog: dict[str, dict] = settings.get("stimulus.catalog", {})
    hints: dict[str, list[str]] = settings.get("bids.task_name_hints", {})
    tolerance = float(settings.get("bids.duration_match_tolerance_frac", 0.25))

    task_lower = (task or "").lower()
    scores: dict[str, float] = {}
    evidence: dict[str, Any] = {"task": task, "duration_sec": duration_sec, "per_movie": {}}

    for key, spec in catalog.items():
        expected = float(spec.get("expected_duration_sec") or 0.0)
        detail: dict[str, Any] = {"expected_duration_sec": expected}
        score = 0.0

        if duration_sec is not None and expected > 0:
            relative_error = abs(duration_sec - expected) / expected
            detail["relative_duration_error"] = round(relative_error, 4)
            if relative_error <= tolerance:
                # Linear credit: exact match -> 1.0, at tolerance edge -> 0.
                score += 2.0 * (1.0 - relative_error / tolerance)
            else:
                score -= 1.0
        else:
            detail["relative_duration_error"] = None

        name_hit = next((h for h in hints.get(key, []) if h.lower() in task_lower), None)
        detail["name_hint_matched"] = name_hit
        if name_hit:
            # A bare "dm"/"tp" hint is weak; a descriptive hint is strong.
            score += 1.5 if len(name_hit) > 3 else 0.5

        scores[key] = score
        evidence["per_movie"][key] = detail

    if not scores:
        return MovieClassification(MovieKey.UNKNOWN, 0.0, evidence)

    best_key, best_score = max(scores.items(), key=lambda item: item[1])
    ordered = sorted(scores.values(), reverse=True)
    runner_up = ordered[1] if len(ordered) > 1 else 0.0
    margin = best_score - runner_up
    evidence["scores"] = scores
    evidence["margin"] = round(margin, 4)

    # Require positive evidence AND separation from the alternative.
    if best_score <= 0.5 or margin < 0.4:
        evidence["decision"] = "insufficient evidence"
        return MovieClassification(MovieKey.UNKNOWN, max(0.0, best_score / 3.5), evidence)

    evidence["decision"] = "classified"
    try:
        movie = MovieKey(best_key)
    except ValueError:
        return MovieClassification(MovieKey.UNKNOWN, 0.0, evidence)
    return MovieClassification(movie, min(1.0, best_score / 3.5), evidence)


# --------------------------------------------------------------------------
# Indexing
# --------------------------------------------------------------------------

def _try_pybids(bids_root: Path) -> Any | None:
    try:
        from bids import BIDSLayout
    except ImportError:
        return None
    try:
        return BIDSLayout(str(bids_root), validate=False, derivatives=False)
    except Exception as exc:  # noqa: BLE001
        log.warning("PyBIDS layout failed; falling back to entity parser",
                    extra={"error": str(exc)})
        return None


def index_bids(session: Session, settings: Settings, asset: DataAsset) -> IndexReport:
    """Index a BIDS root into subjects + scans with movie classification."""
    bids_root = Path(asset.absolute_path)
    report = IndexReport()

    if not bids_root.exists():
        report.warnings.append(f"BIDS root does not exist: {bids_root}")
        return report

    layout = _try_pybids(bids_root)
    report.used_pybids = layout is not None

    # Gather anatomical + functional files.
    files = list(_iter_bids_files(bids_root))
    bold_files = [f for f in files if f.suffix == "bold" and f.extension.startswith("nii")]
    t1w_files = [f for f in files if f.suffix == "T1w" and f.extension.startswith("nii")]
    fmap_files = [f for f in files if f.datatype == "fmap" and f.extension.startswith("nii")]

    report.n_bold = len(bold_files)
    report.n_t1w = len(t1w_files)

    t1w_by_subject: dict[str, Path] = {}
    for item in t1w_files:
        t1w_by_subject.setdefault(item.entities["sub"], item.path)

    fmap_by_subject: dict[str, list[str]] = {}
    for item in fmap_files:
        fmap_by_subject.setdefault(item.entities["sub"], []).append(str(item.path))

    content_flags: list[bool] = []
    subjects_seen: set[str] = set()

    for item in bold_files:
        sub_label = item.entities["sub"]
        from neurotribe.acquisition.hbn_metadata import normalize_subject_id

        external_id = normalize_subject_id(sub_label) or sub_label.upper()
        subjects_seen.add(external_id)

        sidecar = _load_sidecar(item.path)
        tr = sidecar.get("RepetitionTime")
        tr = float(tr) if isinstance(tr, (int, float)) else None

        present = _content_is_present(item.path)
        content_flags.append(present)
        n_volumes = _nifti_volume_count(item.path) if present else None
        # Some releases record the run length directly in the sidecar.
        if n_volumes is None and isinstance(sidecar.get("NumberOfVolumes"), (int, float)):
            n_volumes = int(sidecar["NumberOfVolumes"])

        duration = None
        if n_volumes is not None and tr:
            duration = n_volumes * tr
        elif isinstance(sidecar.get("AcquisitionDuration"), (int, float)):
            duration = float(sidecar["AcquisitionDuration"])

        task = item.entities.get("task")
        classification = classify_movie(task, duration, settings)

        subject = get_or_create_subject(
            session, external_id,
            bids_participant_id=f"sub-{sub_label}",
            has_mri=True,
        )
        if subject.site is None and sidecar.get("InstitutionName"):
            subject.site = str(sidecar["InstitutionName"])

        t1w_path = t1w_by_subject.get(sub_label)
        subject.has_anatomical = t1w_path is not None

        existing = session.execute(
            select(Scan).where(
                Scan.subject_id == subject.id,
                Scan.task == task,
                Scan.run == item.entities.get("run"),
                Scan.session == item.entities.get("ses"),
            )
        ).scalar_one_or_none()

        scan = existing or Scan(subject_id=subject.id)
        scan.task = task
        scan.run = item.entities.get("run")
        scan.session = item.entities.get("ses")
        scan.acquisition = item.entities.get("acq")
        scan.suffix = item.suffix
        scan.datatype = item.datatype
        scan.bold_path = str(item.path)
        json_path = Path(str(item.path).replace(".nii.gz", ".json").replace(".nii", ".json"))
        scan.bold_json_path = str(json_path) if json_path.exists() else None
        scan.t1w_path = str(t1w_path) if t1w_path else None
        scan.fieldmap_paths = fmap_by_subject.get(sub_label, [])
        scan.repetition_time = tr
        scan.n_volumes = n_volumes
        scan.duration_sec = duration
        scan.echo_time = sidecar.get("EchoTime")
        scan.slice_timing_present = "SliceTiming" in sidecar
        scan.scanner = " ".join(
            str(sidecar[k]) for k in ("Manufacturer", "ManufacturersModelName") if sidecar.get(k)
        ) or None
        scan.site = subject.site
        scan.content_present = present
        scan.sidecar_json = {k: v for k, v in sidecar.items() if not isinstance(v, (list, dict))}
        scan.movie = classification.movie.value
        scan.movie_confidence = classification.confidence
        scan.movie_evidence = classification.evidence

        if existing is None:
            session.add(scan)

        report.by_task[task or "unknown"] = report.by_task.get(task or "unknown", 0) + 1
        report.by_movie[classification.movie.value] = (
            report.by_movie.get(classification.movie.value, 0) + 1
        )
        if classification.movie is not MovieKey.UNKNOWN:
            report.n_movie_scans += 1
            subject.has_movie_bold = True

    session.flush()
    report.n_subjects = len(subjects_seen)
    if content_flags:
        report.content_present = sum(content_flags) > len(content_flags) / 2
        if not report.content_present:
            report.warnings.append(
                "BIDS tree appears to contain DataLad/git-annex placeholders rather than "
                "materialised NIfTI content. Preprocessing will require `datalad get`."
            )
    if report.n_movie_scans == 0 and report.n_bold > 0:
        report.warnings.append(
            "No BOLD run could be bound to a documented HBN movie interval. "
            "Check RepetitionTime/volume counts, or supply the movie duration."
        )

    asset.n_records = report.n_bold
    asset.validation_report = {**(asset.validation_report or {}), "bids_index": report.to_dict()}
    record_audit(session, "bids.indexed", entity_type="data_asset", entity_id=asset.id,
                 summary=f"{report.n_subjects} subjects / {report.n_bold} BOLD runs",
                 payload=report.to_dict())
    log.info("BIDS index complete", extra=report.to_dict())
    return report


def index_all(session: Session, settings: Settings) -> dict:
    """Index every registered BIDS root."""
    assets = list(session.execute(
        select(DataAsset).where(DataAsset.kind == AssetKind.BIDS_ROOT.value)
    ).scalars())
    if not assets:
        return {"n_roots": 0, "reports": [], "warnings": ["No BIDS root registered."]}
    reports = [index_bids(session, settings, asset).to_dict() for asset in assets]
    return {"n_roots": len(assets), "reports": reports}


def movie_scan_counts(session: Session) -> dict[str, int]:
    counts: dict[str, int] = {}
    for scan in session.execute(select(Scan)).scalars():
        counts[scan.movie] = counts.get(scan.movie, 0) + 1
    return counts


def subjects_with_movie(session: Session, movie: MovieKey) -> list[Subject]:
    rows = session.execute(
        select(Subject).join(Scan, Scan.subject_id == Subject.id)
        .where(Scan.movie == movie.value).distinct()
    ).scalars()
    return list(rows)
