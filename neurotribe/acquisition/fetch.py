"""Selective retrieval of imaging content.

Hard rule from the specification: **never automatically fetch every HBN scan.**
Retrieval is driven by the cohort target list and pulls only the files a
subject's preprocessing actually needs (T1w, movie BOLD, sidecars, fieldmaps).

Retrieval is only possible for DataLad/git-annex BIDS trees. For a plain
directory of placeholders we surface a blocker instead of inventing content.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.orm import Session

from neurotribe.config import Settings
from neurotribe.database.enums import BlockerKind, BlockerSeverity
from neurotribe.database.models import Scan, Subject
from neurotribe.database.repository import raise_blocker, record_audit
from neurotribe.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class FetchPlan:
    subject_external_id: str
    files: list[str] = field(default_factory=list)
    estimated_bytes: int = 0
    already_present: bool = False


@dataclass
class FetchResult:
    requested: int = 0
    fetched: int = 0
    skipped_present: int = 0
    failed: list[str] = field(default_factory=list)
    method: str = "none"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "requested": self.requested, "fetched": self.fetched,
            "skipped_present": self.skipped_present, "failed": self.failed[:20],
            "method": self.method, "notes": self.notes[:20],
        }


def _datalad_available() -> bool:
    return shutil.which("datalad") is not None


def _is_placeholder(path: Path) -> bool:
    try:
        if path.is_symlink():
            import os
            return ".git/annex" in str(os.readlink(path)).replace("\\", "/")
        return path.stat().st_size <= 4096
    except OSError:
        return True


def build_plan(session: Session, scans: list[Scan]) -> list[FetchPlan]:
    """Compute the minimal file set required for the given scans."""
    plans: dict[str, FetchPlan] = {}
    for scan in scans:
        subject = session.get(Subject, scan.subject_id)
        if subject is None:
            continue
        plan = plans.setdefault(subject.external_id, FetchPlan(subject.external_id))
        for candidate in [scan.bold_path, scan.bold_json_path, scan.t1w_path, *scan.fieldmap_paths]:
            if candidate and candidate not in plan.files:
                plan.files.append(candidate)

    for plan in plans.values():
        missing = [f for f in plan.files if _is_placeholder(Path(f))]
        plan.already_present = not missing
        # Rough estimate used for the disk-space guard.
        plan.estimated_bytes = len(missing) * 200 * 1024 * 1024
    return list(plans.values())


def fetch(session: Session, settings: Settings, scans: list[Scan],
          bids_root: Path | None = None) -> FetchResult:
    """Materialise only the files the target scans need."""
    result = FetchResult()
    plans = build_plan(session, scans)
    result.requested = sum(len(p.files) for p in plans)

    pending = [p for p in plans if not p.already_present]
    result.skipped_present = sum(len(p.files) for p in plans if p.already_present)
    if not pending:
        result.method = "already_present"
        result.notes.append("All required imaging content is already materialised.")
        return result

    if bids_root is None or not (bids_root / ".datalad").exists():
        raise_blocker(
            session, BlockerKind.BIDS_MISSING,
            "Imaging content not materialised",
            "The BIDS tree contains git-annex placeholders but is not a DataLad "
            "dataset, so content cannot be retrieved automatically.",
            severity=BlockerSeverity.EXTERNAL,
            required_action=(
                "Provide a DataLad-enabled HBN BIDS clone, or materialise the NIfTI "
                "files for the target subjects into the existing tree."
            ),
            context={"n_subjects_pending": len(pending)},
        )
        result.method = "unavailable"
        result.notes.append("BIDS root is not a DataLad dataset; cannot fetch selectively.")
        return result

    if not _datalad_available():
        raise_blocker(
            session, BlockerKind.BIDS_MISSING, "DataLad not installed",
            "The BIDS tree is a DataLad dataset but the `datalad` command is unavailable.",
            severity=BlockerSeverity.ACTIONABLE,
            required_action="Install DataLad in the worker image (`pip install datalad`).",
        )
        result.method = "unavailable"
        return result

    result.method = "datalad"
    for plan in pending:
        missing = [f for f in plan.files if _is_placeholder(Path(f))]
        if not missing:
            continue
        command = ["datalad", "get", "--jobs", "2", *missing]
        try:
            completed = subprocess.run(
                command, cwd=str(bids_root), capture_output=True, text=True,
                timeout=3600, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            result.failed.append(f"{plan.subject_external_id}: {exc}")
            continue

        if completed.returncode != 0:
            result.failed.append(
                f"{plan.subject_external_id}: datalad get exited {completed.returncode}: "
                f"{completed.stderr.strip()[:200]}"
            )
            continue

        got = sum(1 for f in missing if not _is_placeholder(Path(f)))
        result.fetched += got
        if got < len(missing):
            result.failed.append(
                f"{plan.subject_external_id}: {len(missing) - got} file(s) still unavailable"
            )

    record_audit(session, "fetch.completed", entity_type="fetch",
                 summary=f"{result.fetched}/{result.requested} files", payload=result.to_dict())
    log.info("Selective fetch complete", extra=result.to_dict())
    return result


def estimate_disk_requirement(settings: Settings, n_subjects: int) -> dict:
    """Estimate disk needed for preprocessing a cohort, for the capacity guard.

    Measures the *data* directory rather than the process root. Under Docker the
    root is the container's overlay filesystem, which can report hundreds of
    free gigabytes while the bind-mounted host volume that actually receives the
    derivatives is nearly full.
    """
    per_subject_gb = float(settings.get("autopilot.disk.per_subject_estimate_gb", 12))
    required_gb = per_subject_gb * max(0, n_subjects)
    usage = shutil.disk_usage(str(settings.paths.data))
    free_gb = usage.free / (1024 ** 3)
    min_free_gb = float(settings.get("autopilot.disk.min_free_gb", 20))
    return {
        "n_subjects": n_subjects,
        "per_subject_gb": per_subject_gb,
        "required_gb": round(required_gb, 1),
        "free_gb": round(free_gb, 1),
        "total_gb": round(usage.total / (1024 ** 3), 1),
        "min_free_gb": min_free_gb,
        "sufficient": free_gb >= max(min_free_gb, required_gb * 0.5),
    }
