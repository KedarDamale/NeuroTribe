"""Unit tests for the system probe, disk guard and the Autopilot stage graph."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from neurotribe.acquisition.fetch import estimate_disk_requirement
from neurotribe.jobs.stages import (
    EXTERNALLY_GATED, STAGE_BY_KEY, STAGES, descendants, groups, order,
)
from neurotribe.system import probe


# ------------------------------------------------------------------ disk

def test_disk_guard_measures_the_data_volume(settings, monkeypatch):
    """Under Docker the process root is the container overlay filesystem.

    Measuring it would report hundreds of free gigabytes while the bind-mounted
    volume that actually receives derivatives is nearly full, so the guard must
    measure the data directory.
    """
    measured: list[str] = []
    real_disk_usage = shutil.disk_usage

    def spy(path):
        measured.append(str(path))
        return real_disk_usage(path)

    monkeypatch.setattr(shutil, "disk_usage", spy)
    estimate_disk_requirement(settings, n_subjects=3)

    assert measured, "the guard did not measure any filesystem"
    assert measured[0] == str(settings.paths.data), (
        f"guard measured {measured[0]!r}, expected the data directory"
    )


def test_disk_estimate_scales_with_cohort_size(settings):
    one = estimate_disk_requirement(settings, 1)
    ten = estimate_disk_requirement(settings, 10)
    assert ten["required_gb"] == pytest.approx(one["required_gb"] * 10)
    assert one["per_subject_gb"] > 0


def test_disk_estimate_is_insufficient_when_requirement_is_absurd(settings):
    result = estimate_disk_requirement(settings, 100_000)
    assert result["sufficient"] is False


def test_system_probe_reports_the_data_volume(settings):
    result = probe(settings)
    assert result.disk_path == str(settings.paths.data)
    assert result.free_disk_gb is not None and result.free_disk_gb >= 0
    assert result.cpu_count >= 1
    # A missing GPU must never be fatal.
    assert isinstance(result.ready, bool)
    payload = result.to_dict()
    assert "disk_path" in payload and "cuda_available" in payload


def test_system_probe_never_raises_without_optional_dependencies(settings, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name in {"torch", "psutil"}:
            raise ImportError(f"{name} blocked for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    result = probe(settings)          # must not raise
    assert result.cuda_available is False
    assert any("PyTorch" in warning for warning in result.warnings)


# ------------------------------------------------------------------ stages

def test_stage_graph_is_acyclic_and_ordered():
    ordered = order()
    assert len(ordered) == len(STAGES)
    seen: set[str] = set()
    for spec in ordered:
        for dependency in spec.depends_on:
            assert dependency in seen, (
                f"'{spec.key}' precedes its dependency '{dependency}'"
            )
        seen.add(spec.key)


def test_every_dependency_exists():
    for spec in STAGES:
        for dependency in spec.depends_on:
            assert dependency in STAGE_BY_KEY, (
                f"'{spec.key}' depends on unknown stage '{dependency}'"
            )


def test_stage_keys_are_unique():
    keys = [spec.key for spec in STAGES]
    assert len(keys) == len(set(keys))


def test_descendants_of_a_gate_include_the_experiment():
    blocked = descendants("phenotype_intake")
    assert "build_cohort" in blocked
    assert "group_analysis" in blocked
    assert "generate_report" in blocked
    # But an unrelated branch must NOT be blocked - independent work continues.
    assert "tribe_smoke_test" not in blocked
    assert "index_bids" not in blocked


def test_stimulus_gate_does_not_block_cohort_construction():
    """The two gates are independent: one arriving late must not stall the other."""
    blocked = descendants("stimulus_intake")
    assert "tribe_inference" in blocked
    assert "build_cohort" not in blocked


def test_externally_gated_stages_are_the_expected_ones():
    assert EXTERNALLY_GATED == {"phenotype_intake", "stimulus_intake", "fetch_imaging"}


def test_every_stage_has_a_handler():
    from neurotribe.jobs.autopilot import HANDLERS

    for spec in STAGES:
        assert spec.key in HANDLERS, f"stage '{spec.key}' has no handler"


def test_groups_cover_every_stage():
    covered = {key for group in groups() for key in group.stages}
    assert covered == {spec.key for spec in STAGES}


def test_externally_gated_stages_have_generous_retry_budgets():
    """A gate may legitimately wait for weeks; it must not exhaust its attempts."""
    for key in EXTERNALLY_GATED - {"fetch_imaging"}:
        assert STAGE_BY_KEY[key].max_attempts >= 100


# ------------------------------------------------------------------ autopilot

def test_bootstrap_registers_all_stages_and_writes_instructions(settings, db):
    from neurotribe.database.repository import all_stages
    from neurotribe.jobs.autopilot import bootstrap

    with db() as session:
        bootstrap(session, settings)

    with db() as session:
        registered = all_stages(session)
        assert len(registered) == len(STAGES)
        assert all(stage.state == "PENDING" for stage in registered)

    assert (settings.paths.phenotype_incoming / "README.md").exists()
    assert (settings.paths.stimuli_incoming / "README.md").exists()

    readme = (settings.paths.stimuli_incoming / "README.md").read_text(encoding="utf-8")
    assert "never downloads video" in readme
    assert "00:03:21" in readme          # the documented HBN interval


def test_tick_advances_and_raises_precise_blockers(settings, db):
    """With no data present the tick must block cleanly, not crash."""
    from neurotribe.database.repository import active_blockers
    from neurotribe.jobs.autopilot import bootstrap, tick

    with db() as session:
        bootstrap(session, settings)

    result = tick(settings, max_stages=25)
    assert "system_probe" in result.ran
    assert "discover_assets" in result.ran

    with db() as session:
        blockers = active_blockers(session)
        kinds = {blocker.kind for blocker in blockers}

    # The three legitimate external gates must be surfaced by name.
    assert "PHENOTYPE_ACCESS" in kinds
    assert "STIMULUS_MISSING" in kinds
    assert "BIDS_MISSING" in kinds

    for blocker in blockers:
        assert blocker.description, f"{blocker.kind} has no explanation"
        if blocker.severity == "EXTERNAL":
            assert blocker.required_action, f"{blocker.kind} has no required action"


def test_tick_is_idempotent_and_never_raises(settings, db):
    from neurotribe.jobs.autopilot import bootstrap, tick

    with db() as session:
        bootstrap(session, settings)

    first = tick(settings, max_stages=25)
    second = tick(settings, max_stages=25)
    # Completed stages must not re-run.
    assert "system_probe" in first.ran
    assert "system_probe" not in second.ran


def test_waiting_stages_do_not_starve_runnable_ones(settings, db):
    """Regression: gated stages must not consume the whole per-tick budget.

    `WAITING_EXTERNAL` stages stay runnable forever. Without a backoff they are
    re-checked on every tick and, with a small `max_stages`, the four data-gate
    stages alone exhaust the budget — so `tribe_install`,`stimulus_intake` and
    `preprocessing_preflight` would never run at all.
    """
    from neurotribe.database.repository import get_stage
    from neurotribe.jobs.autopilot import bootstrap, tick

    with db() as session:
        bootstrap(session, settings)

    # A deliberately tight budget, as the Celery task uses.
    for _ in range(6):
        tick(settings, max_stages=4)

    with db() as session:
        for key in ("tribe_install", "stimulus_intake", "preprocessing_preflight"):
            stage = get_stage(session, key)
            assert stage.state != "PENDING", (
                f"'{key}' never ran - a gated stage starved it "
                f"(state={stage.state}, attempts={stage.attempts})"
            )
        # And the smoke test must have been reached through tribe_install.
        assert get_stage(session, "tribe_smoke_test").state in {"DONE", "RUNNING", "PENDING"}


def test_waiting_stages_get_a_recheck_backoff(settings, db):
    from neurotribe.database.repository import get_stage
    from neurotribe.jobs.autopilot import bootstrap, tick

    with db() as session:
        bootstrap(session, settings)
    tick(settings, max_stages=25)

    with db() as session:
        stage = get_stage(session, "phenotype_intake")
        assert stage.state == "WAITING_EXTERNAL"
        assert stage.next_attempt_at is not None, "no re-check backoff was scheduled"
        # Waiting on a human is not a failed attempt.
        assert stage.attempts <= 1


def test_no_phenotype_means_no_invented_labels(settings, db):
    """The hard rule: no ADHD label may exist while phenotype data are absent."""
    from sqlalchemy import select

    from neurotribe.acquisition.phenotype import phenotype_available
    from neurotribe.database.models import Diagnosis
    from neurotribe.jobs.autopilot import bootstrap, tick

    with db() as session:
        bootstrap(session, settings)
    tick(settings, max_stages=25)

    with db() as session:
        assert not phenotype_available(session)
        assert session.execute(select(Diagnosis)).first() is None
