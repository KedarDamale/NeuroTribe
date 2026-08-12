"""``python scripts/bootstrap.py`` - prepare a fresh workspace.

Creates the directory tree, initialises the database schema, registers the
Autopilot stage graph, writes the operator intake instructions, and runs an
initial discovery pass over anything already on disk.

Safe to run repeatedly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurotribe import RESEARCH_DISCLAIMER, __version__  # noqa: E402
from neurotribe.config import get_settings  # noqa: E402
from neurotribe.logging_setup import configure_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the NeuroTRIBE workspace")
    parser.add_argument("--tick", action="store_true",
                        help="Run one Autopilot iteration after bootstrapping")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(level="ERROR" if args.quiet else "INFO", json_output=False)

    print(f"NeuroTRIBE-HBN v{__version__}")
    print(RESEARCH_DISCLAIMER)
    print(f"root    : {settings.root}")
    print(f"profile : {settings.profile}\n")

    settings.paths.ensure()
    print("Directories:")
    for name in ("data", "phenotype_incoming", "stimuli_incoming", "derivatives",
                 "tribe", "analysis", "reports", "cache", "work"):
        print(f"  {name:<20} {getattr(settings.paths, name)}")

    from neurotribe.database.base import create_all, session_scope
    from neurotribe.jobs.autopilot import bootstrap

    create_all()
    print("\nDatabase schema ready.")

    with session_scope() as session:
        bootstrap(session, settings)
    print("Stage graph registered and intake instructions written.")

    from neurotribe.acquisition.discover import run_discovery

    with session_scope() as session:
        summary = run_discovery(session, settings)
    print(f"\nDiscovery: {summary['n_assets']} asset(s) found.")
    for kind, count in sorted(summary["by_kind"].items()):
        print(f"  {kind:<24} {count}")
    if not summary["by_kind"]:
        print("  (none yet - the Autopilot will report exactly what it needs)")

    if args.tick:
        from neurotribe.jobs.autopilot import tick

        print("\nRunning one Autopilot iteration...")
        result = tick(settings, max_stages=25)
        for key in result.ran:
            print(f"  ran {key} -> {result.outcomes[key]}")

    print("\nNext steps:")
    print("  1. docker compose up -d --build")
    print("  2. open http://localhost:4321")
    print("  3. python scripts/doctor.py    (terminal view of the same state)")
    print("\nThe three external gates are surfaced in the UI, never faked:")
    print(f"  - phenotype export -> {settings.paths.phenotype_incoming}")
    print(f"  - movie stimulus   -> {settings.paths.stimuli_incoming}")
    print(f"  - FreeSurfer license -> {settings.root / 'config' / 'license.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
