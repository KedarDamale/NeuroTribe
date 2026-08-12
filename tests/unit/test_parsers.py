"""Unit tests for the schema-adaptive parsers.

HBN file layouts change between releases, so these tests pin the behaviour that
protects us: known columns are found under many spellings, unknown columns are
preserved rather than dropped, and ambiguous input is refused rather than
guessed.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from neurotribe.acquisition.bids import classify_movie, parse_entities
from neurotribe.acquisition.discover import classify_csv
from neurotribe.acquisition.hbn_metadata import (
    IQM_FIELDS, SUBJECT_ID_COLUMNS, find_column, normalize_key,
    normalize_subject_id, parse_bids_name, parse_bool, parse_float, parse_mriqc,
)
from neurotribe.acquisition.phenotype import (
    is_adhd_label, is_no_diagnosis_label, parse_certainty, resolve_diagnosis_columns,
)
from neurotribe.database.enums import AssetKind, DiagnosisCertainty, MovieKey


# ------------------------------------------------------------------ ids

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("NDARAA075AMK", "NDARAA075AMK"),
        ("sub-NDARAA075AMK", "NDARAA075AMK"),
        ("ndaraa075amk", "NDARAA075AMK"),
        ("  NDARAA075AMK  ", "NDARAA075AMK"),
        ("sub-NDARAA075AMK_ses-1", "NDARAA075AMK"),
        ("", None),
        (None, None),
    ],
)
def test_normalize_subject_id(raw, expected):
    assert normalize_subject_id(raw) == expected


def test_normalize_key_ignores_punctuation_and_case():
    assert normalize_key("Anonymized ID") == "anonymizedid"
    assert normalize_key("FD_mean") == "fdmean"
    assert normalize_key("  Study-Site  ") == "studysite"


def test_find_column_matches_across_spellings():
    for header in (["Anonymized ID"], ["participant_id"], ["EID"], ["subjectID"]):
        lookup = {normalize_key(c): c for c in header}
        assert find_column(lookup, SUBJECT_ID_COLUMNS) == header[0]


def test_find_column_returns_none_when_absent():
    lookup = {normalize_key(c): c for c in ["height", "weight"]}
    assert find_column(lookup, ("framewise_displacement",)) is None


@pytest.mark.parametrize(
    "raw,expected",
    [("1", True), ("yes", True), ("TRUE", True), ("0", False), ("n/a", False),
     ("", False), ("maybe", None), (None, None)],
)
def test_parse_bool(raw, expected):
    assert parse_bool(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [("1.5", 1.5), ("1,500", 1500.0), ("n/a", None), ("", None), (".", None),
     ("NaN", None), (None, None)],
)
def test_parse_float(raw, expected):
    assert parse_float(raw) == expected


# ------------------------------------------------------------------ MRIQC

def test_parse_bids_name_extracts_entities():
    entities = parse_bids_name("sub-NDARAA075AMK_ses-1_task-movieDM_run-01_bold")
    assert entities["sub"] == "NDARAA075AMK"
    assert entities["ses"] == "1"
    assert entities["task"] == "movieDM"
    assert entities["run"] == "01"


def test_parse_mriqc_preserves_unknown_columns(tmp_path: Path):
    """An IQM we do not know about must survive, not be silently dropped."""
    path = tmp_path / "iqm.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["bids_name", "fd_mean", "tsnr", "brand_new_iqm"],
        )
        writer.writeheader()
        writer.writerow({
            "bids_name": "sub-NDARTEST0001_task-movieDM_bold",
            "fd_mean": "0.21", "tsnr": "44.5", "brand_new_iqm": "7.25",
        })

    records, report = parse_mriqc(path)
    assert len(records) == 1
    record = records[0]
    assert record.subject_external_id == "NDARTEST0001"
    assert record.task == "movieDM"
    assert record.values["mean_fd"] == pytest.approx(0.21)
    assert record.values["tsnr"] == pytest.approx(44.5)
    assert record.extra["brand_new_iqm"] == pytest.approx(7.25)
    assert report["n_records"] == 1


def test_parse_mriqc_handles_alternate_column_names(tmp_path: Path):
    path = tmp_path / "iqm.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["bids_name", "dvars_nstd"])
        writer.writeheader()
        writer.writerow({"bids_name": "sub-NDARTEST0001_bold", "dvars_nstd": "1.1"})

    records, _report = parse_mriqc(path)
    assert records[0].values["dvars"] == pytest.approx(1.1)


def test_iqm_field_map_is_not_empty():
    assert "mean_fd" in IQM_FIELDS and "tsnr" in IQM_FIELDS


# ------------------------------------------------------------------ phenotype

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Confirmed", DiagnosisCertainty.CONFIRMED),
        ("confirmed", DiagnosisCertainty.CONFIRMED),
        ("1", DiagnosisCertainty.CONFIRMED),
        ("Presumptive", DiagnosisCertainty.PRESUMPTIVE),
        ("2", DiagnosisCertainty.PRESUMPTIVE),
        ("Requires Confirmation", DiagnosisCertainty.REQUIRES_CONFIRMATION),
        ("Rule-out", DiagnosisCertainty.RULE_OUT),
        ("By History", DiagnosisCertainty.BY_HISTORY),
        ("Past", DiagnosisCertainty.PAST),
        ("No Diagnosis Given", DiagnosisCertainty.NO_DIAGNOSIS_GIVEN),
        ("Incomplete Eval", DiagnosisCertainty.INCOMPLETE_EVAL),
        ("", DiagnosisCertainty.UNKNOWN),
        ("something else", DiagnosisCertainty.UNKNOWN),
    ],
)
def test_parse_certainty(raw, expected):
    assert parse_certainty(raw) is expected


@pytest.mark.parametrize(
    "label,expected",
    [
        ("ADHD-Combined Type", True),
        ("ADHD-Inattentive Type", True),
        ("Attention-Deficit/Hyperactivity Disorder", True),
        ("attention deficit disorder", True),
        ("Autism Spectrum Disorder", False),
        ("Specific Learning Disorder", False),
        ("", False),
    ],
)
def test_is_adhd_label(label, expected):
    patterns = ["adhd", "attention[- ]deficit"]
    assert is_adhd_label(label, patterns) is expected


def test_is_no_diagnosis_label():
    patterns = ["no diagnosis given", "no diagnosis or symptoms"]
    assert is_no_diagnosis_label("No Diagnosis Given", patterns)
    assert not is_no_diagnosis_label("ADHD-Combined Type", patterns)
    assert not is_no_diagnosis_label("", patterns)


def test_resolve_diagnosis_columns_finds_the_family():
    header = [
        "Anonymized ID",
        "DX_01", "DX_01_Conf", "DX_01_Cat",
        "DX_02", "DX_02_Conf",
        "DX_10", "DX_10_Conf",
        "unrelated_column",
    ]
    families = resolve_diagnosis_columns(header, max_ordinal=10)
    ordinals = [f.ordinal for f in families]
    assert ordinals == [1, 2, 10]
    assert families[0].label == "DX_01"
    assert families[0].confidence == "DX_01_Conf"
    assert families[0].category == "DX_01_Cat"


def test_resolve_diagnosis_columns_ignores_out_of_range():
    families = resolve_diagnosis_columns(["DX_01", "DX_99"], max_ordinal=10)
    assert [f.ordinal for f in families] == [1]


# ------------------------------------------------------------------ discovery

def _write_csv(path: Path, header: list[str]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerow(["x"] * len(header))
    return path


def test_classify_csv_identifies_mriqc_by_content(tmp_path: Path):
    path = _write_csv(tmp_path / "anything.csv",
                      ["bids_name", "efc", "fber", "fd_mean", "tsnr", "dvars_std"])
    candidate = classify_csv(path)
    assert candidate is not None
    assert candidate.kind is AssetKind.MRIQC_FUNCTIONAL


def test_classify_csv_identifies_phenotype_by_content(tmp_path: Path):
    path = _write_csv(tmp_path / "export.csv", ["Anonymized ID", "DX_01", "DX_01_Cat"])
    candidate = classify_csv(path)
    assert candidate is not None
    assert candidate.kind is AssetKind.PHENOTYPE_CSV


def test_classify_csv_rejects_unrelated_tables(tmp_path: Path):
    path = _write_csv(tmp_path / "shopping.csv", ["item", "price", "quantity"])
    assert classify_csv(path) is None


# ------------------------------------------------------------------ BIDS

def test_parse_entities_reads_bids_filename(tmp_path: Path):
    root = tmp_path / "bids"
    func = root / "sub-NDARTEST0001" / "func"
    func.mkdir(parents=True)
    path = func / "sub-NDARTEST0001_task-movieDM_run-01_bold.nii.gz"
    path.write_bytes(b"")

    parsed = parse_entities(path, root)
    assert parsed is not None
    assert parsed.entities["sub"] == "NDARTEST0001"
    assert parsed.entities["task"] == "movieDM"
    assert parsed.entities["run"] == "01"
    assert parsed.suffix == "bold"
    assert parsed.datatype == "func"


def test_parse_entities_rejects_non_bids_files(tmp_path: Path):
    root = tmp_path / "bids"
    root.mkdir()
    path = root / "notes.nii.gz"
    path.write_bytes(b"")
    assert parse_entities(path, root) is None


def test_classify_movie_prefers_duration_over_name(settings):
    """A misleading task label must not override the acquisition duration."""
    # Labelled like Despicable Me but only 201 s long -> The Present wins.
    result = classify_movie("movieDM", 201.0, settings)
    assert result.movie is MovieKey.THE_PRESENT


def test_classify_movie_uses_name_only_as_tiebreaker(settings):
    exact = classify_movie("movieDM", 600.0, settings)
    assert exact.movie is MovieKey.DESPICABLE_ME
    assert exact.confidence > 0.6
    assert exact.evidence["per_movie"]["despicable_me"]["relative_duration_error"] == 0.0


def test_classify_movie_records_full_evidence(settings):
    result = classify_movie("rest", 400.0, settings)
    assert result.movie is MovieKey.UNKNOWN
    assert "per_movie" in result.evidence
    assert set(result.evidence["per_movie"]) == {"the_present", "despicable_me"}
    assert result.evidence["decision"] == "insufficient evidence"
