"""``python scripts/doctor.py`` - environment and pipeline diagnostics.

Prints exactly what the dashboard shows, but in a terminal, so the state of the
system is inspectable without the web UI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurotribe import RESEARCH_DISCLAIMER, __version__  # noqa: E402
from neurotribe.config import get_settings  # noqa: E402
from neurotribe.logging_setup import configure_logging  # noqa: E402

STATE_MARK = {
    "DONE": "[ok]",
    "RUNNING": "[..]",
    "PENDING": "[  ]",
    "WAITING_EXTERNAL": "[!!]",
    "BLOCKED": "[--]",
    "PARTIAL": "[~~]",
    "FAILED_RETRYABLE": "[xx]",
    "FAILED_FINAL": "[XX]",
    "SKIPPED": "[..]",
}


def rule(title: str = "") -> None:
    if title:
        print(f"\n{'=' * 76}\n{title}\n{'=' * 76}")
    else:
        print("-" * 76)


def main() -> int:
    parser = argparse.ArgumentParser(description="NeuroTRIBE diagnostics")
    parser.add_argument("--ticks", type=int, default=0,
                        help="Run N Autopilot ticks before reporting")
    parser.add_argument("--max-stages", type=int, default=25,
                        help="Maximum stages to advance per tick")
    parser.add_argument("--quiet", action="store_true", help="Suppress log output")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(level="ERROR" if args.quiet else "INFO", json_output=False)

    print(f"NeuroTRIBE-HBN v{__version__}  |  profile: {settings.profile}")
    print(RESEARCH_DISCLAIMER)
    print(f"root: {settings.root}")
    print(f"analysis config hash: {settings.analysis_config_hash}")

    from neurotribe.database.base import create_all, session_scope
    from neurotribe.jobs.autopilot import bootstrap, status, tick

    create_all()
    with session_scope() as session:
        bootstrap(session, settings)

    if args.ticks:
        rule("AUTOPILOT")
        for index in range(args.ticks):
            result = tick(settings, max_stages=args.max_stages)
            print(f"tick {index + 1}: ran {result.ran or '(nothing runnable)'}")

    report = status(settings)

    rule("SYSTEM")
    from neurotribe.system import probe

    system = probe(settings)
    print(f"  platform     : {system.platform_name}")
    print(f"  python       : {system.python_version}")
    print(f"  cpu / ram    : {system.cpu_count} cores / {system.ram_gb} GB")
    print(f"  disk free    : {system.free_disk_gb} / {system.total_disk_gb} GB")
    print(f"  gpu          : {system.gpu_name or 'none'} "
          f"(cuda={system.cuda_available})")
    print(f"  docker       : {system.docker_available} "
          f"({system.docker_version or '-'}, {system.docker_cpus} cpus, "
          f"{system.docker_memory_gb} GB)")
    print(f"  ffmpeg       : {system.ffmpeg_available}")
    print(f"  tribe v2     : {system.tribe_available}")
    print(f"  fs license   : {system.freesurfer_license}")
    for warning in system.warnings:
        print(f"  WARN  {warning}")
    for blocker in system.blockers:
        print(f"  BLOCK {blocker}")

    rule("PIPELINE")
    for stage in report["stages"]:
        mark = STATE_MARK.get(stage["state"], "[??]")
        detail = (stage["detail"] or "").replace("\n", " ")
        print(f"  {mark} {stage['key']:<26} {stage['state']:<17} {detail[:70]}")
        if stage["last_error"]:
            print(f"        error: {stage['last_error'][:100]}")

    rule("BLOCKERS")
    if not report["blockers"]:
        print("  none")
    for blocker in report["blockers"]:
        print(f"  [{blocker['severity']}] {blocker['title']}")
        print(f"      {blocker['description'][:150]}")
        if blocker["required_action"]:
            print(f"      ACTION: {blocker['required_action'][:150]}")
        if blocker["reference_url"]:
            print(f"      REF   : {blocker['reference_url']}")

    rule("DATA")
    from sqlalchemy import func, select

    from neurotribe.database.models import (
        DataAsset, Scan, Stimulus, Subject, SubjectComparison,
    )

    with session_scope() as session:
        def count(model, *where):
            stmt = select(func.count()).select_from(model)
            for clause in where:
                stmt = stmt.where(clause)
            return int(session.execute(stmt).scalar_one())

        print(f"  assets registered   : {count(DataAsset)}")
        print(f"  subjects            : {count(Subject)}")
        print(f"  with phenotype      : {count(Subject, Subject.has_phenotype.is_(True))}")
        print(f"  with movie BOLD     : {count(Subject, Subject.has_movie_bold.is_(True))}")
        print(f"  scans               : {count(Scan)}")
        print(f"  validated stimuli   : {count(Stimulus, Stimulus.validated.is_(True))}")
        print(f"  valid comparisons   : {count(SubjectComparison, SubjectComparison.valid.is_(True))}")

    rule()
    done = sum(1 for s in report["stages"] if s["state"] == "DONE")
    print(f"{done}/{len(report['stages'])} stages complete, "
          f"{len(report['blockers'])} active blocker(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
