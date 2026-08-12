"""fMRIPrep orchestration.

The research-grade path is:

    HBN volumetric BOLD -> fMRIPrep -> FreeSurfer surfaces -> fsaverage5 GIFTI

FreeSurfer licensing and defaced-T1 failures are handled explicitly:
HBN warns that defacing can degrade some tools including FreeSurfer, so a
failure is retried once with validated settings and then recorded as
``PREPROCESSING_FAILED`` - never papered over with substituted surfaces.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from sqlalchemy.orm import Session

from neurotribe.config import Settings
from neurotribe.database.enums import (
    BlockerKind, BlockerSeverity, PreprocStatus,
)
from neurotribe.database.models import PreprocessingRun, Scan, Subject
from neurotribe.database.repository import raise_blocker, record_audit
from neurotribe.hashing import cache_key, hash_json
from neurotribe.logging_setup import get_logger

log = get_logger(__name__)

ProgressCallback = Callable[[float, str], None]


class PreprocessingError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# FreeSurfer license
# --------------------------------------------------------------------------

@dataclass
class LicenseStatus:
    found: bool
    path: Path | None = None
    source: str | None = None

    def to_dict(self) -> dict:
        return {"found": self.found, "path": str(self.path) if self.path else None,
                "source": self.source}


def detect_freesurfer_license(settings: Settings) -> LicenseStatus:
    """Locate a FreeSurfer license. Never fabricates one."""
    env_path = os.environ.get("FS_LICENSE")
    if env_path:
        candidate = Path(env_path)
        if candidate.is_file():
            return LicenseStatus(True, candidate, "FS_LICENSE")

    candidates = [
        settings.root / "config" / "license.txt",
        settings.root / "secrets" / "license.txt",
        settings.root / "license.txt",
        Path("/opt/freesurfer/license.txt"),
        Path("/run/secrets/freesurfer_license"),
        Path.home() / ".freesurfer" / "license.txt",
    ]
    for candidate in candidates:
        try:
            if candidate.is_file() and candidate.stat().st_size > 0:
                return LicenseStatus(True, candidate, "filesystem")
        except OSError:
            continue
    return LicenseStatus(False)


def ensure_license_or_block(session: Session, settings: Settings) -> LicenseStatus:
    status = detect_freesurfer_license(settings)
    if status.found:
        from neurotribe.database.repository import clear_blocker

        clear_blocker(session, BlockerKind.FREESURFER_LICENSE)
        return status

    raise_blocker(
        session, BlockerKind.FREESURFER_LICENSE,
        "FreeSurfer license required",
        "fMRIPrep's surface reconstruction requires a FreeSurfer license file. "
        "None was found via FS_LICENSE or the standard locations.",
        severity=BlockerSeverity.EXTERNAL,
        required_action=(
            "Obtain a free license from https://surfer.nmr.mgh.harvard.edu/registration.html "
            "and place it at config/license.txt (or set FS_LICENSE)."
        ),
        reference_url="https://surfer.nmr.mgh.harvard.edu/registration.html",
        blocks_stages=["preprocess_cohort", "preprocess_smoke_test"],
    )
    return status


# --------------------------------------------------------------------------
# Docker availability
# --------------------------------------------------------------------------

def docker_available() -> tuple[bool, str]:
    executable = shutil.which("docker")
    if executable is None:
        return False, "docker executable not found on PATH"
    try:
        result = subprocess.run([executable, "info", "--format", "{{.ServerVersion}}"],
                                capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"docker info failed: {exc}"
    if result.returncode != 0:
        return False, f"docker daemon unreachable: {result.stderr.strip()[:200]}"
    return True, result.stdout.strip()


def image_present(image: str) -> bool:
    executable = shutil.which("docker")
    if executable is None:
        return False
    result = subprocess.run([executable, "image", "inspect", image],
                            capture_output=True, text=True, timeout=120, check=False)
    return result.returncode == 0


def pull_image(image: str, progress: ProgressCallback | None = None) -> bool:
    executable = shutil.which("docker")
    if executable is None:
        return False
    log.info("Pulling container image", extra={"image": image})
    process = subprocess.Popen([executable, "pull", image],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert process.stdout is not None
    for line in process.stdout:
        if progress:
            progress(0.0, line.rstrip())
    return process.wait() == 0


# --------------------------------------------------------------------------
# Command construction
# --------------------------------------------------------------------------

@dataclass
class FmriprepInvocation:
    command: list[str]
    output_dir: Path
    work_dir: Path
    log_path: Path
    cache_key: str
    config_snapshot: dict = field(default_factory=dict)


def _mount(host: Path, container: str, mode: str = "ro") -> list[str]:
    return ["-v", f"{host.as_posix()}:{container}:{mode}"]


def build_invocation(settings: Settings, subject: Subject, scan: Scan,
                     bids_root: Path, license_path: Path) -> FmriprepInvocation:
    """Construct the docker command for a single participant."""
    config = settings.get("preprocessing.fmriprep", {})
    image = str(config.get("image", "nipreps/fmriprep:24.1.1"))
    participant_label = (subject.bids_participant_id or f"sub-{subject.external_id}").replace("sub-", "")

    output_dir = settings.paths.fmriprep_out
    work_dir = settings.paths.work / "fmriprep" / participant_label
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    key = cache_key(
        "fmriprep",
        subject=subject.external_id,
        scan=f"{scan.task}/{scan.run}/{scan.session}",
        bold_sha=scan.bold_path,
        version=config.get("version_pin"),
        image=image,
        output_spaces=config.get("output_spaces"),
        extra_args=config.get("extra_args"),
    )

    command = [
        shutil.which("docker") or "docker", "run", "--rm",
        "--name", f"neurotribe-fmriprep-{participant_label}",
        *_mount(bids_root, "/data", "ro"),
        *_mount(output_dir, "/out", "rw"),
        *_mount(work_dir, "/work", "rw"),
        *_mount(license_path, "/opt/freesurfer/license.txt", "ro"),
        "-e", "FS_LICENSE=/opt/freesurfer/license.txt",
        image,
        "/data", "/out", "participant",
        "--participant-label", participant_label,
        "-w", "/work",
        "--output-spaces", *[str(s) for s in config.get("output_spaces", ["fsaverage5"])],
        "--nthreads", str(config.get("nthreads", 4)),
        "--omp-nthreads", str(config.get("omp_nthreads", 2)),
        "--mem-mb", str(config.get("mem_mb", 8000)),
        "--fs-license-file", "/opt/freesurfer/license.txt",
        *[str(a) for a in config.get("extra_args", [])],
    ]
    if scan.task:
        command += ["--task-id", str(scan.task)]

    log_dir = settings.paths.derivatives / "logs" / "fmriprep"
    log_dir.mkdir(parents=True, exist_ok=True)

    return FmriprepInvocation(
        command=command, output_dir=output_dir, work_dir=work_dir,
        log_path=log_dir / f"{participant_label}_{int(time.time())}.log",
        cache_key=key,
        config_snapshot={"image": image, "config": config,
                         "config_hash": hash_json(config, length=16)},
    )


_PROGRESS_RE = re.compile(r"\[node\].*?(\d+)%|Completed.*?(\d+)/(\d+)")


def _run_streaming(command: list[str], log_path: Path,
                   progress: ProgressCallback | None = None,
                   timeout_sec: int = 60 * 60 * 24) -> int:
    """Run a subprocess, streaming combined output to a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8", errors="replace") as handle:
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            if progress and ("node" in line.lower() or "error" in line.lower()):
                progress(-1.0, line.rstrip()[:200])
            if time.monotonic() - started > timeout_sec:
                process.kill()
                raise PreprocessingError(f"fMRIPrep exceeded {timeout_sec}s timeout")
        return process.wait()


# --------------------------------------------------------------------------
# Output discovery
# --------------------------------------------------------------------------

@dataclass
class FmriprepOutputs:
    surface_lh: Path | None = None
    surface_rh: Path | None = None
    confounds: Path | None = None
    confounds_json: Path | None = None
    report: Path | None = None
    all_surfaces: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return all([self.surface_lh, self.surface_rh, self.confounds])

    def to_dict(self) -> dict:
        return {
            "surface_lh": str(self.surface_lh) if self.surface_lh else None,
            "surface_rh": str(self.surface_rh) if self.surface_rh else None,
            "confounds": str(self.confounds) if self.confounds else None,
            "confounds_json": str(self.confounds_json) if self.confounds_json else None,
            "report": str(self.report) if self.report else None,
            "complete": self.complete, "warnings": self.warnings,
        }


def _entity_match(name: str, scan: Scan) -> bool:
    """Filename must correspond to the exact run we asked for."""
    if scan.task and f"task-{scan.task}" not in name:
        return False
    if scan.run and f"run-{scan.run}" not in name:
        return False
    if scan.session and f"ses-{scan.session}" not in name:
        return False
    return True


def discover_outputs(output_dir: Path, subject: Subject, scan: Scan,
                     space: str = "fsaverage5") -> FmriprepOutputs:
    """Locate fMRIPrep derivatives programmatically.

    BIDS entity ordering and derivative layout vary between versions, so we
    search rather than assume a fixed path.
    """
    outputs = FmriprepOutputs()
    label = (subject.bids_participant_id or f"sub-{subject.external_id}")
    roots = [output_dir / label, output_dir / "fmriprep" / label]
    subject_root = next((r for r in roots if r.exists()), None)
    if subject_root is None:
        outputs.warnings.append(f"No fMRIPrep output directory found for {label}")
        return outputs

    func_files = list(subject_root.rglob("*.func.gii"))
    matching = [p for p in func_files if f"space-{space}" in p.name and _entity_match(p.name, scan)]
    outputs.all_surfaces = [p for p in func_files if f"space-{space}" in p.name]

    if not matching and outputs.all_surfaces:
        outputs.warnings.append(
            f"No {space} surface matched the requested run entities "
            f"(task={scan.task}, run={scan.run}); {len(outputs.all_surfaces)} other "
            f"{space} surfaces exist. Refusing to guess."
        )

    for path in matching:
        name = path.name
        if "hemi-L" in name:
            outputs.surface_lh = outputs.surface_lh or path
        elif "hemi-R" in name:
            outputs.surface_rh = outputs.surface_rh or path

    for path in subject_root.rglob("*desc-confounds_timeseries.tsv"):
        if _entity_match(path.name, scan):
            outputs.confounds = path
            json_path = path.with_suffix(".json")
            outputs.confounds_json = json_path if json_path.exists() else None
            break

    report = output_dir / f"{label}.html"
    outputs.report = report if report.exists() else None

    if outputs.surface_lh and not outputs.surface_rh:
        outputs.warnings.append("Left hemisphere surface found without a right hemisphere.")
    if outputs.surface_rh and not outputs.surface_lh:
        outputs.warnings.append("Right hemisphere surface found without a left hemisphere.")
    return outputs


def _is_freesurfer_failure(log_path: Path) -> bool:
    """Detect the FreeSurfer failure mode HBN warns about with defaced T1s."""
    if not log_path.exists():
        return False
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")[-200000:]
    except OSError:
        return False
    markers = (
        "recon-all exited with errors", "ERROR: mri_watershed", "Skull strip",
        "talairach_afd", "FreeSurfer failed", "mris_make_surfaces",
    )
    return any(marker.lower() in text.lower() for marker in markers)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def run_participant(session: Session, settings: Settings, subject: Subject, scan: Scan,
                    bids_root: Path, *, progress: ProgressCallback | None = None,
                    attempt: int = 1) -> PreprocessingRun:
    """Preprocess one participant, with a single validated retry on FreeSurfer failure."""
    run = PreprocessingRun(
        subject_id=subject.id, scan_id=scan.id, engine="fmriprep",
        engine_version=str(settings.get("preprocessing.fmriprep.version_pin")),
        status=PreprocStatus.RUNNING.value, attempt=attempt,
        started_at=datetime.now(timezone.utc),
    )
    session.add(run)
    session.flush()

    license_status = ensure_license_or_block(session, settings)
    if not license_status.found or license_status.path is None:
        run.status = PreprocStatus.NOT_STARTED.value
        run.error_message = "FreeSurfer license unavailable; run deferred (WAITING_EXTERNAL)."
        run.finished_at = datetime.now(timezone.utc)
        return run

    available, detail = docker_available()
    if not available:
        raise_blocker(session, BlockerKind.DOCKER_UNAVAILABLE, "Docker unavailable",
                      f"fMRIPrep runs in a container but Docker is not usable: {detail}",
                      severity=BlockerSeverity.ACTIONABLE,
                      required_action="Start Docker Desktop / the Docker daemon.")
        run.status = PreprocStatus.NOT_STARTED.value
        run.error_message = f"Docker unavailable: {detail}"
        run.finished_at = datetime.now(timezone.utc)
        return run

    invocation = build_invocation(settings, subject, scan, bids_root, license_status.path)
    run.cache_key = invocation.cache_key
    run.container_image = invocation.config_snapshot["image"]
    run.config_snapshot = invocation.config_snapshot
    run.output_dir = str(invocation.output_dir)
    run.log_path = str(invocation.log_path)

    # Cache check: a completed run with identical inputs is reused, not repeated.
    existing = discover_outputs(invocation.output_dir, subject, scan,
                                str(settings.get("surface.space", "fsaverage5")))
    if existing.complete:
        log.info("fMRIPrep outputs already present; skipping container run",
                 extra={"subject": subject.external_id})
        _record_outputs(run, existing)
        run.status = PreprocStatus.SUCCEEDED.value
        run.finished_at = datetime.now(timezone.utc)
        run.exit_code = 0
        record_audit(session, "preprocessing.cached", entity_type="preprocessing_run",
                     entity_id=run.id, summary=subject.external_id)
        return run

    if not image_present(invocation.config_snapshot["image"]):
        if not pull_image(invocation.config_snapshot["image"], progress):
            run.status = PreprocStatus.FAILED.value
            run.error_message = f"Could not pull image {invocation.config_snapshot['image']}"
            run.finished_at = datetime.now(timezone.utc)
            return run

    try:
        exit_code = _run_streaming(invocation.command, invocation.log_path, progress)
    except PreprocessingError as exc:
        run.status = PreprocStatus.FAILED.value
        run.error_message = str(exc)
        run.finished_at = datetime.now(timezone.utc)
        return run

    run.exit_code = exit_code
    outputs = discover_outputs(invocation.output_dir, subject, scan,
                               str(settings.get("surface.space", "fsaverage5")))
    _record_outputs(run, outputs)

    if exit_code == 0 and outputs.complete:
        run.status = PreprocStatus.SUCCEEDED.value
        run.finished_at = datetime.now(timezone.utc)
        record_audit(session, "preprocessing.succeeded", entity_type="preprocessing_run",
                     entity_id=run.id, summary=subject.external_id)
        return run

    run.finished_at = datetime.now(timezone.utc)
    run.status = PreprocStatus.FAILED.value
    if _is_freesurfer_failure(invocation.log_path):
        run.error_message = (
            "FreeSurfer surface reconstruction failed. HBN notes that defacing can "
            "negatively affect FreeSurfer. "
            + ("Retrying once with validated settings." if attempt == 1
               else "Retry already attempted; participant marked PREPROCESSING_FAILED.")
        )
        if attempt == 1:
            session.flush()
            record_audit(session, "preprocessing.retry", entity_type="preprocessing_run",
                         entity_id=run.id, summary=f"{subject.external_id}: FreeSurfer retry")
            return _retry_with_validated_settings(session, settings, subject, scan,
                                                  bids_root, progress)
    else:
        run.error_message = (
            f"fMRIPrep exited {exit_code}. " +
            "; ".join(outputs.warnings[:3] or ["See log for details."])
        )

    record_audit(session, "preprocessing.failed", entity_type="preprocessing_run",
                 entity_id=run.id, summary=subject.external_id,
                 payload={"exit_code": exit_code, "warnings": outputs.warnings[:5]})
    return run


def _retry_with_validated_settings(session: Session, settings: Settings, subject: Subject,
                                   scan: Scan, bids_root: Path,
                                   progress: ProgressCallback | None) -> PreprocessingRun:
    """Second attempt with settings validated against defaced-anatomy failures."""
    import copy

    from neurotribe.config import Settings as SettingsType

    raw = settings.raw
    extra = list(raw.setdefault("preprocessing", {}).setdefault("fmriprep", {}).setdefault(
        "extra_args", []))
    for flag in ("--skull-strip-t1w", "force"):
        if flag not in extra:
            extra.append(flag)
    raw["preprocessing"]["fmriprep"]["extra_args"] = extra
    retry_settings = SettingsType(copy.deepcopy(raw), profile=settings.profile, root=settings.root)

    return run_participant(session, retry_settings, subject, scan, bids_root,
                           progress=progress, attempt=2)


def _record_outputs(run: PreprocessingRun, outputs: FmriprepOutputs) -> None:
    run.surface_lh_path = str(outputs.surface_lh) if outputs.surface_lh else None
    run.surface_rh_path = str(outputs.surface_rh) if outputs.surface_rh else None
    run.confounds_path = str(outputs.confounds) if outputs.confounds else None
    run.confounds_json_path = str(outputs.confounds_json) if outputs.confounds_json else None
    run.report_path = str(outputs.report) if outputs.report else None
    run.config_snapshot = {**(run.config_snapshot or {}), "outputs": outputs.to_dict()}


def preflight(session: Session, settings: Settings) -> dict:
    """Report readiness of the preprocessing engine without running anything."""
    docker_ok, docker_detail = docker_available()
    license_status = detect_freesurfer_license(settings)
    image = str(settings.get("preprocessing.fmriprep.image"))
    return {
        "engine": settings.get("preprocessing.engine"),
        "docker_available": docker_ok,
        "docker_detail": docker_detail,
        "image": image,
        "image_present": image_present(image) if docker_ok else False,
        "freesurfer_license": license_status.to_dict(),
        "ready": docker_ok and license_status.found,
    }
