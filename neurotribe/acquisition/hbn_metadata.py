"""HBN release metadata + MRIQC ingestion.

HBN recommends its metadata file for determining which participants completed
imaging and phenotypic sessions, and publishes MRIQC IQMs because imaging data
are released regardless of quality. Both files have evolved across releases, so
every parser here is **schema-adaptive**: known columns are promoted, unknown
columns are preserved rather than discarded.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy.orm import Session

from neurotribe.config import Settings
from neurotribe.database.enums import AssetKind, AssetStatus, QCStatus
from neurotribe.database.models import DataAsset, ScanQC, Subject
from neurotribe.database.repository import get_or_create_subject, record_audit
from neurotribe.logging_setup import get_logger

log = get_logger(__name__)

# Candidate column names for each logical field, in priority order. Matching is
# case-insensitive and ignores punctuation.
SUBJECT_ID_COLUMNS = (
    "anonymized id", "anonymizedid", "participant_id", "participantid", "eid",
    "subject", "subjectid", "subject_id", "src_subject_id", "id",
)
SITE_COLUMNS = ("site", "study site", "studysite", "scan site", "scansite", "visit_site")
AGE_COLUMNS = ("age", "age at scan", "ageatscan", "interview_age", "basic_demos_study_site_age")
SEX_COLUMNS = ("sex", "gender", "basic_demos_sex")
RELEASE_COLUMNS = ("release_number", "releasenumber", "release", "data release")
COMMERCIAL_COLUMNS = ("commercial_use", "commercialuse", "commercial use")
MRI_FLAG_COLUMNS = ("mri", "has_mri", "mri_available", "imaging", "mri_track_scan_location")
PHENO_FLAG_COLUMNS = ("phenotype", "has_phenotype", "clinical", "diagnosis")

_TRUE_TOKENS = {"1", "1.0", "true", "yes", "y", "t", "available", "complete", "completed"}
_FALSE_TOKENS = {"0", "0.0", "false", "no", "n", "f", "na", "n/a", "", "missing", "none"}

# HBN external identifiers look like NDARXX000XXX.
_SUBJECT_ID_RE = re.compile(r"(NDAR[A-Z0-9]{8,})", re.IGNORECASE)


def normalize_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.strip().lower())


def _build_lookup(header: Iterable[str]) -> dict[str, str]:
    return {normalize_key(col): col for col in header}


def find_column(lookup: dict[str, str], candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        key = normalize_key(candidate)
        if key in lookup:
            return lookup[key]
    # Fall back to a prefix/substring match so release-specific suffixes still bind.
    for candidate in candidates:
        key = normalize_key(candidate)
        for norm, original in lookup.items():
            if norm.startswith(key) or key in norm:
                return original
    return None


def parse_bool(value: Any) -> bool | None:
    if value is None:
        return None
    token = str(value).strip().lower()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    return None


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    token = str(value).strip().replace(",", "")
    if token == "" or token.lower() in {"na", "n/a", "nan", "none", "null", "."}:
        return None
    try:
        return float(token)
    except ValueError:
        return None


def normalize_subject_id(value: Any) -> str | None:
    """Extract a canonical HBN external id from any identifier spelling."""
    if value is None:
        return None
    token = str(value).strip()
    if not token:
        return None
    token = token.replace("sub-", "").replace("sub_", "")
    match = _SUBJECT_ID_RE.search(token)
    if match:
        return match.group(1).upper()
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "", token)
    return cleaned.upper() or None


def read_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV/TSV with delimiter sniffing and BOM tolerance."""
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        sample = handle.read(65536)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        except csv.Error:
            dialect = csv.excel
            first = sample.splitlines()[0] if sample.splitlines() else ""
            if first.count("\t") > first.count(","):
                dialect = csv.excel_tab
        reader = csv.DictReader(handle, dialect=dialect)
        header = [h for h in (reader.fieldnames or []) if h is not None]
        rows = [dict(row) for row in reader]
    return header, rows


# --------------------------------------------------------------------------
# Release metadata
# --------------------------------------------------------------------------

@dataclass
class MetadataSummary:
    n_rows: int = 0
    n_subjects: int = 0
    n_with_mri: int = 0
    n_with_phenotype_flag: int = 0
    sites: dict[str, int] = field(default_factory=dict)
    columns_used: dict[str, str | None] = field(default_factory=dict)
    unmapped_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows, "n_subjects": self.n_subjects,
            "n_with_mri": self.n_with_mri,
            "n_with_phenotype_flag": self.n_with_phenotype_flag,
            "sites": self.sites, "columns_used": self.columns_used,
            "unmapped_columns": self.unmapped_columns[:50],
            "warnings": self.warnings,
        }


def ingest_metadata(session: Session, settings: Settings, asset: DataAsset) -> MetadataSummary:
    """Load HBN release metadata into the subjects table."""
    path = Path(asset.absolute_path)
    header, rows = read_table(path)
    lookup = _build_lookup(header)

    columns = {
        "subject": find_column(lookup, SUBJECT_ID_COLUMNS),
        "site": find_column(lookup, SITE_COLUMNS),
        "age": find_column(lookup, AGE_COLUMNS),
        "sex": find_column(lookup, SEX_COLUMNS),
        "release": find_column(lookup, RELEASE_COLUMNS),
        "commercial_use": find_column(lookup, COMMERCIAL_COLUMNS),
        "mri": find_column(lookup, MRI_FLAG_COLUMNS),
        "phenotype": find_column(lookup, PHENO_FLAG_COLUMNS),
    }
    summary = MetadataSummary(n_rows=len(rows), columns_used=columns)

    if columns["subject"] is None:
        summary.warnings.append(
            "No subject identifier column found; metadata cannot be linked to imaging."
        )
        asset.status = AssetStatus.INVALID.value
        asset.validation_report = {**(asset.validation_report or {}), "metadata": summary.to_dict()}
        return summary

    used = {c for c in columns.values() if c}
    summary.unmapped_columns = [c for c in header if c not in used]

    seen: set[str] = set()
    for row in rows:
        external_id = normalize_subject_id(row.get(columns["subject"]))
        if not external_id:
            continue
        if external_id in seen:
            # Duplicate participant rows are a data hazard - never silently merge.
            summary.warnings.append(f"Duplicate metadata row for {external_id}; first row kept.")
            continue
        seen.add(external_id)

        site = (row.get(columns["site"]) or "").strip() or None if columns["site"] else None
        has_mri = parse_bool(row.get(columns["mri"])) if columns["mri"] else None
        has_pheno_flag = parse_bool(row.get(columns["phenotype"])) if columns["phenotype"] else None
        commercial = parse_bool(row.get(columns["commercial_use"])) if columns["commercial_use"] else None

        subject = get_or_create_subject(
            session, external_id,
            site=site,
            age=parse_float(row.get(columns["age"])) if columns["age"] else None,
            sex=(row.get(columns["sex"]) or "").strip() or None if columns["sex"] else None,
            release=(row.get(columns["release"]) or "").strip() or None if columns["release"] else None,
        )
        if commercial is not None:
            subject.commercial_use_allowed = commercial
        if has_mri is not None:
            subject.has_mri = has_mri
        # Keep every unmapped field so nothing from the release is lost.
        subject.metadata_json = {
            **(subject.metadata_json or {}),
            "release_metadata": {k: row.get(k) for k in summary.unmapped_columns[:80]},
            "phenotype_flag": has_pheno_flag,
        }

        summary.n_subjects += 1
        if has_mri:
            summary.n_with_mri += 1
        if has_pheno_flag:
            summary.n_with_phenotype_flag += 1
        if site:
            summary.sites[site] = summary.sites.get(site, 0) + 1

    asset.status = AssetStatus.VALIDATED.value
    asset.n_records = summary.n_subjects
    asset.validation_report = {**(asset.validation_report or {}), "metadata": summary.to_dict()}
    record_audit(session, "metadata.ingested", entity_type="data_asset", entity_id=asset.id,
                 summary=f"{summary.n_subjects} subjects", payload=summary.to_dict())
    log.info("HBN metadata ingested", extra=summary.to_dict())
    return summary


# --------------------------------------------------------------------------
# MRIQC
# --------------------------------------------------------------------------

# Logical IQM name -> candidate MRIQC column names.
IQM_FIELDS = {
    "mean_fd": ("fd_mean", "meanfd", "fd_mean_mm"),
    "max_fd": ("fd_max", "maxfd"),
    "dvars": ("dvars_std", "dvars_nstd", "dvars"),
    "tsnr": ("tsnr",),
    "efc": ("efc",),
    "fber": ("fber",),
    "snr": ("snr", "snr_total"),
    "gsr_x": ("gsr_x",),
    "gsr_y": ("gsr_y",),
    "fd_perc": ("fd_perc", "fd_percent"),
}

BIDS_NAME_COLUMNS = ("bids_name", "bidsname", "filename", "file", "scan")


@dataclass
class MriqcRecord:
    subject_external_id: str
    bids_name: str | None
    task: str | None
    run: str | None
    session: str | None
    values: dict[str, float | None]
    extra: dict[str, Any]


_BIDS_ENTITY_RE = re.compile(r"(?:^|_)(?P<key>[a-zA-Z]+)-(?P<value>[a-zA-Z0-9]+)")


def parse_bids_name(name: str) -> dict[str, str]:
    """Extract BIDS entities from an MRIQC ``bids_name`` string."""
    entities = {m.group("key"): m.group("value") for m in _BIDS_ENTITY_RE.finditer(name or "")}
    return entities


def parse_mriqc(path: Path) -> tuple[list[MriqcRecord], dict]:
    header, rows = read_table(path)
    lookup = _build_lookup(header)

    subject_col = find_column(lookup, SUBJECT_ID_COLUMNS)
    bids_col = find_column(lookup, BIDS_NAME_COLUMNS)
    resolved: dict[str, str | None] = {
        logical: find_column(lookup, candidates) for logical, candidates in IQM_FIELDS.items()
    }
    mapped_cols = {c for c in [subject_col, bids_col, *resolved.values()] if c}

    records: list[MriqcRecord] = []
    for row in rows:
        external_id = normalize_subject_id(row.get(subject_col)) if subject_col else None
        bids_name = (row.get(bids_col) or "").strip() if bids_col else None
        entities = parse_bids_name(bids_name or "")
        if not external_id and entities.get("sub"):
            external_id = normalize_subject_id(entities["sub"])
        if not external_id:
            continue

        values = {logical: parse_float(row.get(col)) if col else None
                  for logical, col in resolved.items()}
        # Preserve every IQM we did not explicitly promote.
        extra = {k: parse_float(v) for k, v in row.items()
                 if k and k not in mapped_cols and parse_float(v) is not None}

        records.append(MriqcRecord(
            subject_external_id=external_id, bids_name=bids_name,
            task=entities.get("task"), run=entities.get("run"), session=entities.get("ses"),
            values=values, extra=extra,
        ))

    report = {
        "n_rows": len(rows), "n_records": len(records),
        "columns_used": {"subject": subject_col, "bids_name": bids_col, **resolved},
        "n_extra_iqms": len(records[0].extra) if records else 0,
        "detected_schema": header[:60],
    }
    return records, report


def evaluate_qc(values: dict[str, float | None], settings: Settings) -> tuple[QCStatus, str | None]:
    """Apply the configured MRIQC inclusion policy. Never silently excludes."""
    max_mean_fd = settings.get("qc.mriqc.max_mean_fd")
    min_tsnr = settings.get("qc.mriqc.min_tsnr")
    reasons: list[str] = []
    status = QCStatus.PASS

    mean_fd = values.get("mean_fd")
    if max_mean_fd is not None and mean_fd is not None and mean_fd > float(max_mean_fd):
        reasons.append(f"mean_fd={mean_fd:.3f} exceeds {float(max_mean_fd):.3f} mm")
        status = QCStatus.FAIL

    tsnr = values.get("tsnr")
    if min_tsnr is not None and tsnr is not None and tsnr < float(min_tsnr):
        reasons.append(f"tsnr={tsnr:.2f} below {float(min_tsnr):.2f}")
        status = QCStatus.FAIL

    if status is QCStatus.PASS and mean_fd is None:
        status = QCStatus.UNKNOWN
        reasons.append("mean_fd unavailable in MRIQC export")

    return status, "; ".join(reasons) or None


def ingest_mriqc(session: Session, settings: Settings, asset: DataAsset) -> dict:
    """Parse MRIQC IQMs and stage them for later attachment to indexed scans.

    Scans may not be indexed yet (BIDS indexing can run later), so records are
    buffered on the asset and attached by :func:`attach_qc_to_scans`.
    """
    path = Path(asset.absolute_path)
    records, report = parse_mriqc(path)

    buffer = [
        {
            "subject_external_id": r.subject_external_id, "bids_name": r.bids_name,
            "task": r.task, "run": r.run, "session": r.session,
            "values": r.values, "extra": r.extra,
        }
        for r in records
    ]
    asset.status = AssetStatus.VALIDATED.value if records else AssetStatus.INVALID.value
    asset.n_records = len(records)
    asset.validation_report = {**(asset.validation_report or {}), "mriqc": report, "records": buffer}
    record_audit(session, "mriqc.ingested", entity_type="data_asset", entity_id=asset.id,
                 summary=f"{len(records)} MRIQC records", payload=report)
    log.info("MRIQC ingested", extra=report)
    return report


def attach_qc_to_scans(session: Session, settings: Settings) -> dict:
    """Join buffered MRIQC records onto indexed scans.

    Matching is by (subject, task, run, session) with progressive relaxation, so
    a release whose MRIQC lacks run entities still binds correctly.
    """
    from sqlalchemy import select

    from neurotribe.database.models import Scan

    assets = list(session.execute(
        select(DataAsset).where(DataAsset.kind.in_([
            AssetKind.MRIQC_FUNCTIONAL.value, AssetKind.MRIQC_ANATOMICAL.value,
        ]))
    ).scalars())

    scans = list(session.execute(select(Scan)).scalars())
    subjects = {s.id: s for s in session.execute(select(Subject)).scalars()}
    by_subject: dict[str, list[Scan]] = {}
    for scan in scans:
        subject = subjects.get(scan.subject_id)
        if subject:
            by_subject.setdefault(subject.external_id, []).append(scan)

    attached = 0
    unmatched = 0
    for asset in assets:
        for record in asset.validation_report.get("records", []):
            candidates = by_subject.get(record["subject_external_id"], [])
            if not candidates:
                unmatched += 1
                continue

            def score(scan: Scan) -> int:
                points = 0
                if record.get("task") and scan.task == record["task"]:
                    points += 4
                if record.get("run") and (scan.run or "") == record["run"]:
                    points += 2
                if record.get("session") and (scan.session or "") == record["session"]:
                    points += 1
                return points

            best = max(candidates, key=score)
            if len(candidates) > 1 and score(best) == 0:
                unmatched += 1
                continue

            values = record["values"]
            status, reason = evaluate_qc(values, settings)
            qc = best.qc or ScanQC(scan_id=best.id)
            qc.subject_external_id = record["subject_external_id"]
            for field_name, value in values.items():
                setattr(qc, field_name, value)
            qc.extra_iqms = record.get("extra", {})
            qc.qc_status = status.value
            qc.qc_reason = reason
            qc.source_asset_id = asset.id
            if best.qc is None:
                session.add(qc)
                best.qc = qc
            attached += 1

    summary = {"attached": attached, "unmatched": unmatched, "n_scans": len(scans)}
    log.info("MRIQC attached to scans", extra=summary)
    return summary
