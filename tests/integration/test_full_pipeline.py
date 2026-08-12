"""End-to-end pipeline test on entirely synthetic data.

This is the test that proves the system works before any real, access-controlled
input exists:

    synthetic BIDS + metadata + MRIQC + phenotype + stimulus
        -> discovery / ingestion / indexing
        -> movie classification by duration
        -> cohort construction (Confirmed ADHD vs No Diagnosis Given)
        -> preprocessed-surface hand-off
        -> TRIBE inference (mock backend)
        -> spatial + temporal alignment
        -> sanity gate
        -> deviation metrics, ROI and network aggregation
        -> covariate-adjusted group statistics with FDR
        -> research report + provenance manifest

Every artefact produced here is stamped ``profile: development`` and
``backend: mock`` and is therefore never reportable as a scientific result.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import select

from neurotribe.database.enums import (
    AnalysisTier, AssetKind, CohortGroup, MovieKey, PreprocStatus,
)
from neurotribe.database.models import (
    Cohort, DataAsset, PreprocessingRun, Scan, Stimulus, Subject,
    SubjectComparison, TribeRun,
)

from tests.conftest import write_bids_dataset
from tests.fixtures.builders import make_shared_stimulus_signal, write_fmriprep_outputs

# The Present: 201 s at TR 0.8 -> 251 volumes. Chosen over Despicable Me so the
# synthetic stimulus clip stays small enough to render quickly in CI.
MOVIE = MovieKey.THE_PRESENT
TR = 0.8
N_VOLUMES = 251
TASK = "movieTP"

N_ADHD = 8
N_CONTROL = 8
N_OTHER = 4


pytestmark = pytest.mark.integration


def _find_all(haystack: str, needle: str):
    """Yield every index of ``needle`` in ``haystack``."""
    start = haystack.find(needle)
    while start != -1:
        yield start
        start = haystack.find(needle, start + 1)


@pytest.fixture()
def synthetic_project(settings, db, metadata_csv, mriqc_csv):
    """Assemble a complete synthetic project on disk and in the database."""
    subjects = [f"NDARSYN{index:05d}" for index in range(N_ADHD + N_CONTROL + N_OTHER)]

    bids_root = settings.paths.external / "HBN_BIDS"
    write_bids_dataset(bids_root, subjects, tr=TR, n_volumes=N_VOLUMES, task=TASK)

    _write_phenotype(settings.paths.phenotype_incoming / "diagnosis.csv", subjects)
    _rewrite_mriqc(mriqc_csv, subjects, task=TASK)
    _rewrite_metadata(metadata_csv, subjects)

    return {"subjects": subjects, "bids_root": bids_root}


def _write_phenotype(path: Path, subjects: list[str]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Anonymized ID", "DX_01", "DX_01_Conf"])
        writer.writeheader()
        for index, external_id in enumerate(subjects):
            if index < N_ADHD:
                row = {"DX_01": "ADHD-Combined Type", "DX_01_Conf": "Confirmed"}
            elif index < N_ADHD + N_CONTROL:
                row = {"DX_01": "No Diagnosis Given", "DX_01_Conf": "No Diagnosis Given"}
            else:
                row = {"DX_01": "Specific Learning Disorder", "DX_01_Conf": "Confirmed"}
            writer.writerow({"Anonymized ID": external_id, **row})


def _rewrite_metadata(path: Path, subjects: list[str]) -> None:
    import csv

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["Anonymized ID", "Site", "Age", "Sex",
                                "Release_Number", "Commercial_Use", "MRI"],
        )
        writer.writeheader()
        for index, external_id in enumerate(subjects):
            writer.writerow({
                "Anonymized ID": external_id,
                "Site": ["RU", "CBIC"][index % 2],
                "Age": round(9.0 + (index % 7), 1),
                "Sex": "M" if index % 2 else "F",
                "Release_Number": "SYNTHETIC", "Commercial_Use": "1", "MRI": "1",
            })


def _rewrite_mriqc(path: Path, subjects: list[str], task: str) -> None:
    import csv

    rng = np.random.default_rng(31)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["bids_name", "fd_mean", "dvars_std", "tsnr", "efc"],
        )
        writer.writeheader()
        for external_id in subjects:
            writer.writerow({
                "bids_name": f"sub-{external_id}_task-{task}_bold",
                "fd_mean": round(float(abs(rng.normal(0.15, 0.05))), 4),
                "dvars_std": 1.02, "tsnr": 44.0, "efc": 0.5,
            })


def _make_stimulus(settings) -> Path | None:
    """Render a synthetic clip matching The Present's documented duration."""
    from neurotribe.tribe.mock import synthetic_video

    spec = settings.get(f"stimulus.catalog.{MOVIE.value}")
    target = settings.paths.stimuli_incoming / "the_present_synthetic.mp4"
    # Must satisfy the configured minimum resolution (320x240) and frame rate,
    # otherwise validation correctly rejects it.
    return synthetic_video(target, duration_sec=float(spec["expected_duration_sec"]),
                           fps=12, width=320, height=240)


# --------------------------------------------------------------------------

def test_acquisition_and_indexing(settings, db, synthetic_project):
    """Discovery classifies every input by content and indexes the BIDS tree."""
    from neurotribe.acquisition import bids, discover, hbn_metadata, phenotype

    with db() as session:
        summary = discover.run_discovery(session, settings)
        kinds = summary["by_kind"]
        assert kinds.get(AssetKind.HBN_METADATA.value) == 1
        assert kinds.get(AssetKind.MRIQC_FUNCTIONAL.value) == 1
        assert kinds.get(AssetKind.PHENOTYPE_CSV.value) == 1
        assert kinds.get(AssetKind.BIDS_ROOT.value) == 1

    with db() as session:
        for asset in session.execute(
            select(DataAsset).where(DataAsset.kind == AssetKind.HBN_METADATA.value)
        ).scalars():
            report = hbn_metadata.ingest_metadata(session, settings, asset)
            assert report.n_subjects == N_ADHD + N_CONTROL + N_OTHER

    with db() as session:
        for asset in session.execute(
            select(DataAsset).where(DataAsset.kind == AssetKind.MRIQC_FUNCTIONAL.value)
        ).scalars():
            hbn_metadata.ingest_mriqc(session, settings, asset)

        result = bids.index_all(session, settings)
        assert result["n_roots"] == 1
        report = result["reports"][0]
        assert report["n_subjects"] == N_ADHD + N_CONTROL + N_OTHER
        # Movie binding must come from duration evidence, not the task label.
        assert report["by_movie"].get(MOVIE.value) == N_ADHD + N_CONTROL + N_OTHER

        qc = hbn_metadata.attach_qc_to_scans(session, settings)
        assert qc["attached"] == N_ADHD + N_CONTROL + N_OTHER

        summary = phenotype.scan_incoming(session, settings)
        assert summary["available"]

    with db() as session:
        scan = session.execute(select(Scan)).scalars().first()
        assert scan.repetition_time == pytest.approx(TR)
        assert scan.n_volumes == N_VOLUMES
        assert scan.duration_sec == pytest.approx(N_VOLUMES * TR)
        assert scan.qc is not None and scan.qc.mean_fd is not None
        # The evidence trail must justify the classification.
        assert scan.movie_evidence["per_movie"][MOVIE.value]["relative_duration_error"] < 0.05


def test_movie_classification_rejects_ambiguous_runs(settings, db):
    """A run matching no documented interval must stay UNKNOWN, not be guessed."""
    from neurotribe.acquisition.bids import classify_movie

    # Correct duration for The Present.
    good = classify_movie("movieTP", 201.0, settings)
    assert good.movie is MOVIE and good.confidence > 0.5

    # A resting-state run: right task-name shape, wrong duration.
    bad = classify_movie("rest", 350.0, settings)
    assert bad.movie is MovieKey.UNKNOWN

    # Unknown duration with only a weak name hint is not enough evidence.
    weak = classify_movie("dm", None, settings)
    assert weak.movie is MovieKey.UNKNOWN


def test_cohort_construction(settings, db, synthetic_project):
    """Confirmed-ADHD vs No-Diagnosis-Given, with every exclusion justified."""
    from neurotribe.acquisition import bids, discover, hbn_metadata, phenotype
    from neurotribe.cohort.eligibility import build_cohort

    with db() as session:
        discover.run_discovery(session, settings)
        for asset in session.execute(select(DataAsset)).scalars():
            if asset.kind == AssetKind.HBN_METADATA.value:
                hbn_metadata.ingest_metadata(session, settings, asset)
            elif asset.kind == AssetKind.MRIQC_FUNCTIONAL.value:
                hbn_metadata.ingest_mriqc(session, settings, asset)
        bids.index_all(session, settings)
        hbn_metadata.attach_qc_to_scans(session, settings)
        phenotype.scan_incoming(session, settings)

    with db() as session:
        result = build_cohort(session, settings, MOVIE, tier=AnalysisTier.PRIMARY,
                              require_preprocessing=False)
        assert result.n_case == N_ADHD
        assert result.n_control == N_CONTROL
        # The four participants with a non-ADHD diagnosis are excluded from the
        # PRIMARY contrast, each with a recorded reason.
        assert result.n_excluded == N_OTHER

        cohort = session.get(Cohort, result.cohort_id)
        for member in cohort.members:
            if not member.included:
                assert member.exclusion_reason, "every exclusion must carry a reason"

    with db() as session:
        exploratory = build_cohort(session, settings, MOVIE,
                                   tier=AnalysisTier.EXPLORATORY,
                                   require_preprocessing=False)
        # The exploratory comparison cohort absorbs the other-diagnosis group.
        assert exploratory.n_control == N_CONTROL + N_OTHER


@pytest.mark.slow
def test_full_scientific_pipeline(settings, db, synthetic_project):
    """The whole chain, ending in group statistics and a research report."""
    from neurotribe.acquisition import bids, discover, hbn_metadata, phenotype, stimulus
    from neurotribe.analysis.group import run as run_group
    from neurotribe.analysis.subject import analyze
    from neurotribe.cohort.eligibility import build_cohort
    from neurotribe.preprocessing.pipeline import prepare_and_cache
    from neurotribe.reporting.report import generate_all
    from neurotribe.tribe.inference import run as run_tribe

    video = _make_stimulus(settings)
    if video is None:
        pytest.skip("ffmpeg unavailable; cannot render the synthetic stimulus")

    # ---- acquisition -------------------------------------------------
    with db() as session:
        discover.run_discovery(session, settings)
        for asset in session.execute(select(DataAsset)).scalars():
            if asset.kind == AssetKind.HBN_METADATA.value:
                hbn_metadata.ingest_metadata(session, settings, asset)
            elif asset.kind == AssetKind.MRIQC_FUNCTIONAL.value:
                hbn_metadata.ingest_mriqc(session, settings, asset)
        bids.index_all(session, settings)
        hbn_metadata.attach_qc_to_scans(session, settings)
        phenotype.scan_incoming(session, settings)
        stimulus_summary = stimulus.scan_incoming(session, settings)
        assert MOVIE.value in stimulus_summary["available"], stimulus_summary

    # ---- TRIBE inference (once per stimulus) -------------------------
    with db() as session:
        registered = session.execute(
            select(Stimulus).where(Stimulus.key == MOVIE.value)
        ).scalar_one()
        prediction = run_tribe(session, settings, registered)
        assert prediction.n_vertices == 20484
        assert prediction.hemi_order == ["L", "R"]
        assert prediction.is_mock, "development profile must use the mock backend"
        support_low, support_high = prediction.support
        assert support_high > support_low

    # The cache must serve the second call rather than recomputing.
    with db() as session:
        n_runs_before = len(list(session.execute(select(TribeRun)).scalars()))
        registered = session.execute(
            select(Stimulus).where(Stimulus.key == MOVIE.value)
        ).scalar_one()
        run_tribe(session, settings, registered)
        assert len(list(session.execute(select(TribeRun)).scalars())) == n_runs_before

    # ---- synthetic preprocessing hand-off ----------------------------
    shared = make_shared_stimulus_signal(N_VOLUMES)
    with db() as session:
        subjects = list(session.execute(select(Subject).order_by(Subject.external_id)).scalars())
        for index, subject in enumerate(subjects):
            scan = next(s for s in subject.scans if s.movie == MOVIE.value)
            is_adhd = index < N_ADHD
            paths = write_fmriprep_outputs(
                settings.paths.fmriprep_out,
                (subject.bids_participant_id or "").replace("sub-", ""),
                TASK, n_timepoints=N_VOLUMES, tr=TR, seed=index,
                shared_signal=shared,
                # ADHD participants get a weaker stimulus-locked component, so
                # the group contrast has a real effect to recover.
                shared_weight=0.45 if is_adhd else 0.70,
                motion_scale=1.3 if is_adhd else 1.0,
            )
            run = PreprocessingRun(
                subject_id=subject.id, scan_id=scan.id, engine="fmriprep",
                engine_version="synthetic", status=PreprocStatus.SUCCEEDED.value,
                surface_lh_path=str(paths["surface_L"]),
                surface_rh_path=str(paths["surface_R"]),
                confounds_path=str(paths["confounds"]),
                confounds_json_path=str(paths["confounds_json"]),
                output_dir=str(settings.paths.fmriprep_out),
            )
            session.add(run)
            session.flush()
            prepared = prepare_and_cache(session, settings, run, scan, subject)
            assert prepared.n_vertices == 20484
            assert prepared.n_dropped == 2          # the flagged non-steady volumes
            assert 0.0 < prepared.mask.fraction <= 1.0
            assert prepared.time_sec[0] == pytest.approx(2 * TR)

    # ---- subject-level analysis --------------------------------------
    with db() as session:
        tribe_run = session.execute(
            select(TribeRun).where(TribeRun.status == "DONE")
        ).scalars().first()
        registered = session.execute(
            select(Stimulus).where(Stimulus.key == MOVIE.value)
        ).scalar_one()
        from neurotribe.tribe.inference import load_cached

        prediction = load_cached(session, settings, MOVIE.value)
        assert prediction is not None

        n_valid = 0
        for run in session.execute(
            select(PreprocessingRun).where(PreprocessingRun.denoised_path.is_not(None))
        ).scalars():
            subject = session.get(Subject, run.subject_id)
            scan = session.get(Scan, run.scan_id)
            result = analyze(session, settings, subject, scan, run, prediction,
                             tribe_run, registered)
            assert result.valid, result.invalid_reason
            assert -1.0 <= result.global_r <= 1.0
            assert result.global_mad >= 0.0
            assert result.peak_windows, "movie-moment analysis produced no windows"
            n_valid += 1
        assert n_valid == N_ADHD + N_CONTROL + N_OTHER

    with db() as session:
        comparison = session.execute(
            select(SubjectComparison).where(SubjectComparison.valid.is_(True))
        ).scalars().first()
        assert comparison.roi_metrics, "ROI aggregation produced nothing"
        assert comparison.network_metrics, "network aggregation produced nothing"
        assert comparison.sanity_report["valid"]
        # Peak windows must be ordered and carry human-readable timecodes.
        deviations = [w["deviation"] for w in comparison.peak_windows]
        assert deviations == sorted(deviations, reverse=True)
        assert ":" in comparison.peak_windows[0]["start_label"]

        # Vertex maps must be on disk and correctly shaped.
        vertex_r = np.load(comparison.vertex_r_path)
        assert vertex_r.shape == (20484,)
        finite = vertex_r[np.isfinite(vertex_r)]
        assert finite.min() >= -1.0 and finite.max() <= 1.0

    # ---- group analysis ----------------------------------------------
    with db() as session:
        build_cohort(session, settings, MOVIE, tier=AnalysisTier.PRIMARY,
                     require_preprocessing=True)

    with db() as session:
        cohort = session.execute(
            select(Cohort).where(Cohort.tier == AnalysisTier.PRIMARY.value)
            .order_by(Cohort.updated_at.desc())
        ).scalars().first()
        assert cohort.n_case == N_ADHD and cohort.n_control == N_CONTROL

        result = run_group(session, settings, cohort, tier=AnalysisTier.PRIMARY)
        assert result.sanity_passed, result.failures
        assert result.n_case == N_ADHD and result.n_control == N_CONTROL
        assert result.n_units > 0

    with db() as session:
        from neurotribe.database.models import GroupAnalysisRun, GroupResult

        analysis_run = session.execute(
            select(GroupAnalysisRun).order_by(GroupAnalysisRun.created_at.desc())
        ).scalars().first()
        rows = list(session.execute(
            select(GroupResult).where(GroupResult.run_id == analysis_run.id)
        ).scalars())
        assert rows

        for row in rows:
            if row.p_value is not None:
                assert 0.0 <= row.p_value <= 1.0
            if row.q_value is not None:
                assert 0.0 <= row.q_value <= 1.0
                # FDR adjustment can only increase a p-value.
                assert row.q_value >= row.p_value - 1e-9
            assert row.n_case == N_ADHD and row.n_control == N_CONTROL

        # The provenance manifest is mandatory and must pin the model identity.
        provenance = analysis_run.provenance
        for key in ("tribe_backend", "analysis_config_hash", "cohort_hash",
                    "denoise_strategy", "surface_space", "hemi_order",
                    "research_use_only"):
            assert key in provenance, f"provenance is missing '{key}'"
        assert provenance["tribe_backend"] == "mock"
        assert provenance["research_use_only"] is True

    # ---- reporting ----------------------------------------------------
    with db() as session:
        report = generate_all(session, settings)
        kinds = {a["kind"] for a in report["artifacts"]}
        assert "report_html" in kinds
        assert "provenance" in kinds

        html_artifact = next(a for a in report["artifacts"] if a["kind"] == "report_html")
        html = (settings.root / html_artifact["path"]).read_text(encoding="utf-8")
        assert "Research Use Only" in html
        # The system must never present itself as a diagnostic tool.
        assert "Probability of ADHD" not in html
        assert "diagnos" in html.lower()  # the disclaimer must be present

        # "healthy controls" may appear ONLY inside the sentence explaining that
        # the comparison cohort is never described that way.
        lowered = html.lower()
        for position in _find_all(lowered, "healthy control"):
            context = lowered[max(0, position - 220):position]
            assert "never" in context, (
                "the report labelled a group 'healthy controls' outside the "
                f"explanatory disclaimer, near: {html[max(0, position - 220):position + 60]}"
            )
        assert CohortGroup.NON_ADHD_COMPARISON.display in html

        # A mock-backed run must say so, loudly.
        assert "MOCK TRIBE backend" in html

        manifest = next(a for a in report["artifacts"] if a["kind"] == "provenance")
        payload = json.loads((settings.root / manifest["path"]).read_text(encoding="utf-8"))
        assert payload["research_use_only"] is True
