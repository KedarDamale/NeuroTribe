"""HBN phenotype intake (Diagnosis_ClinicianConsensus).

Full HBN phenotypic data are DUA-controlled and exported by the operator from
LORIS' Data Query Tool. This module therefore **watches an intake directory**;
it never attempts to authenticate to, scrape, or otherwise bypass LORIS.

Hard rule: while phenotype data are absent, no ADHD label is ever invented.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from neurotribe.config import Settings
from neurotribe.database.enums import AssetKind, AssetStatus, DiagnosisCertainty
from neurotribe.database.models import DataAsset, Diagnosis, Subject
from neurotribe.database.repository import get_or_create_subject, record_audit
from neurotribe.logging_setup import get_logger

from neurotribe.acquisition.hbn_metadata import (
    SUBJECT_ID_COLUMNS, find_column, normalize_key, normalize_subject_id, read_table,
)

log = get_logger(__name__)

# LORIS exports name diagnosis columns in several shapes across releases.
_DX_LABEL_PATTERNS = (
    r"^dx_?0?(\d+)$",
    r"^dx_?0?(\d+)_?(?:label|name|dx)$",
    r"^diagnosis_?0?(\d+)$",
    r"^diagnosisclinicianconsensus_?dx_?0?(\d+)$",
)
_DX_CONFIDENCE_PATTERNS = (
    r"^dx_?0?(\d+)_?(?:conf|confidence|cert|certainty|code)$",
    r"^diagnosis_?0?(\d+)_?(?:conf|confidence|certainty)$",
)
_DX_CATEGORY_PATTERNS = (
    r"^dx_?0?(\d+)_?(?:cat|category)$",
)
_DX_SUB_PATTERNS = (
    r"^dx_?0?(\d+)_?(?:sub|subtype|spec)$",
)

# Numeric certainty codes seen in LORIS exports.
_CERTAINTY_CODE_MAP = {
    "1": DiagnosisCertainty.CONFIRMED,
    "2": DiagnosisCertainty.PRESUMPTIVE,
    "3": DiagnosisCertainty.REQUIRES_CONFIRMATION,
    "4": DiagnosisCertainty.RULE_OUT,
    "5": DiagnosisCertainty.BY_HISTORY,
    "6": DiagnosisCertainty.PAST,
}


def parse_certainty(raw: Any) -> DiagnosisCertainty:
    """Map any spelling of a certainty value onto the canonical vocabulary."""
    if raw is None:
        return DiagnosisCertainty.UNKNOWN
    token = str(raw).strip()
    if not token:
        return DiagnosisCertainty.UNKNOWN

    # Numeric code path.
    numeric = token.split(".")[0]
    if numeric in _CERTAINTY_CODE_MAP:
        return _CERTAINTY_CODE_MAP[numeric]

    normalized = normalize_key(token)
    table = {
        "confirmed": DiagnosisCertainty.CONFIRMED,
        "presumptive": DiagnosisCertainty.PRESUMPTIVE,
        "requiresconfirmation": DiagnosisCertainty.REQUIRES_CONFIRMATION,
        "ruleout": DiagnosisCertainty.RULE_OUT,
        "byhistory": DiagnosisCertainty.BY_HISTORY,
        "past": DiagnosisCertainty.PAST,
        "nodiagnosisgiven": DiagnosisCertainty.NO_DIAGNOSIS_GIVEN,
        "nodiagnosis": DiagnosisCertainty.NO_DIAGNOSIS_GIVEN,
        "incompleteeval": DiagnosisCertainty.INCOMPLETE_EVAL,
        "incomplete": DiagnosisCertainty.INCOMPLETE_EVAL,
    }
    if normalized in table:
        return table[normalized]
    for key, value in table.items():
        if key in normalized:
            return value
    return DiagnosisCertainty.UNKNOWN


def is_adhd_label(label: str, patterns: Iterable[str]) -> bool:
    text = (label or "").lower()
    return any(re.search(pattern, text) for pattern in patterns)


def is_no_diagnosis_label(label: str, patterns: Iterable[str]) -> bool:
    text = (label or "").lower().strip()
    if not text:
        return False
    return any(re.search(pattern, text) for pattern in patterns)


@dataclass
class DiagnosisColumns:
    """Resolved column names for a single diagnosis ordinal."""

    ordinal: int
    label: str | None = None
    confidence: str | None = None
    category: str | None = None
    subtype: str | None = None


def resolve_diagnosis_columns(header: Iterable[str], max_ordinal: int = 10) -> list[DiagnosisColumns]:
    """Discover the dx_01..dx_10 column family regardless of export spelling."""
    resolved: dict[int, DiagnosisColumns] = {}

    def match_group(column: str, patterns: tuple[str, ...]) -> int | None:
        normalized = normalize_key(column)
        # normalize_key strips underscores, so patterns must tolerate that.
        for pattern in patterns:
            relaxed = pattern.replace("_?", "").replace("_", "")
            found = re.match(relaxed, normalized)
            if found:
                try:
                    return int(found.group(1))
                except (IndexError, ValueError):
                    return None
        return None

    for column in header:
        for patterns, attribute in (
            (_DX_CONFIDENCE_PATTERNS, "confidence"),
            (_DX_CATEGORY_PATTERNS, "category"),
            (_DX_SUB_PATTERNS, "subtype"),
            (_DX_LABEL_PATTERNS, "label"),
        ):
            ordinal = match_group(column, patterns)
            if ordinal is None or not (1 <= ordinal <= max_ordinal):
                continue
            entry = resolved.setdefault(ordinal, DiagnosisColumns(ordinal=ordinal))
            if getattr(entry, attribute) is None:
                setattr(entry, attribute, column)
            break

    return [resolved[k] for k in sorted(resolved)]


@dataclass
class PhenotypeSummary:
    n_rows: int = 0
    n_subjects: int = 0
    n_diagnoses: int = 0
    n_adhd_any: int = 0
    n_adhd_confirmed: int = 0
    n_no_diagnosis_given: int = 0
    certainty_counts: dict[str, int] = field(default_factory=dict)
    columns_detected: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows, "n_subjects": self.n_subjects,
            "n_diagnoses": self.n_diagnoses, "n_adhd_any": self.n_adhd_any,
            "n_adhd_confirmed": self.n_adhd_confirmed,
            "n_no_diagnosis_given": self.n_no_diagnosis_given,
            "certainty_counts": self.certainty_counts,
            "columns_detected": self.columns_detected,
            "warnings": self.warnings[:50],
        }


def ingest_phenotype(session: Session, settings: Settings, asset: DataAsset) -> PhenotypeSummary:
    """Parse a clinician-consensus export into the diagnoses table."""
    path = Path(asset.absolute_path)
    header, rows = read_table(path)
    summary = PhenotypeSummary(n_rows=len(rows))

    lookup = {normalize_key(c): c for c in header}
    subject_col = find_column(lookup, SUBJECT_ID_COLUMNS)
    if subject_col is None:
        summary.warnings.append("No subject identifier column; export cannot be linked.")
        asset.status = AssetStatus.INVALID.value
        asset.validation_report = {**(asset.validation_report or {}), "phenotype": summary.to_dict()}
        return summary

    max_ordinal = int(settings.get("phenotype.max_diagnoses_per_subject", 10))
    families = resolve_diagnosis_columns(header, max_ordinal)
    summary.columns_detected = [
        {"ordinal": f.ordinal, "label": f.label, "confidence": f.confidence,
         "category": f.category, "subtype": f.subtype}
        for f in families
    ]
    if not families:
        summary.warnings.append(
            "No dx_01..dx_NN diagnosis columns found. Export the "
            "Diagnosis_ClinicianConsensus instrument from the LORIS Data Query Tool."
        )
        asset.status = AssetStatus.INVALID.value
        asset.validation_report = {**(asset.validation_report or {}), "phenotype": summary.to_dict()}
        return summary

    adhd_patterns = settings.get("phenotype.adhd_patterns", ["adhd"])
    none_patterns = settings.get("phenotype.no_diagnosis_patterns", ["no diagnosis given"])
    instrument = settings.get("phenotype.instrument", "Diagnosis_ClinicianConsensus")

    seen: set[str] = set()
    for row in rows:
        external_id = normalize_subject_id(row.get(subject_col))
        if not external_id:
            continue
        if external_id in seen:
            summary.warnings.append(f"Duplicate phenotype row for {external_id}; first kept.")
            continue
        seen.add(external_id)

        subject = get_or_create_subject(session, external_id)
        # Replace any previous diagnoses for this subject - the newest authorised
        # export is authoritative.
        for old in list(subject.diagnoses):
            session.delete(old)
        subject.diagnoses = []
        subject.has_phenotype = True

        subject_has_adhd = False
        subject_adhd_confirmed = False
        subject_no_diagnosis = False

        for family in families:
            raw_label = (row.get(family.label) or "").strip() if family.label else ""
            if not raw_label or raw_label.lower() in {"na", "n/a", "nan", "none", "."}:
                continue

            raw_certainty = (row.get(family.confidence) or "") if family.confidence else ""
            certainty = parse_certainty(raw_certainty)
            adhd = is_adhd_label(raw_label, adhd_patterns)
            no_dx = is_no_diagnosis_label(raw_label, none_patterns)
            if no_dx and certainty is DiagnosisCertainty.UNKNOWN:
                # "No Diagnosis Given" appearing as a label is itself the certainty.
                certainty = DiagnosisCertainty.NO_DIAGNOSIS_GIVEN

            diagnosis = Diagnosis(
                subject_id=subject.id, ordinal=family.ordinal, raw_label=raw_label,
                normalized_label=raw_label.strip().title(),
                category=(row.get(family.category) or "").strip() or None if family.category else None,
                certainty=certainty.value, raw_certainty=str(raw_certainty).strip() or None,
                is_adhd=adhd, is_no_diagnosis=no_dx, instrument=instrument,
                source_asset_id=asset.id,
            )
            session.add(diagnosis)
            summary.n_diagnoses += 1
            summary.certainty_counts[certainty.value] = (
                summary.certainty_counts.get(certainty.value, 0) + 1
            )

            if adhd:
                subject_has_adhd = True
                if certainty is DiagnosisCertainty.CONFIRMED:
                    subject_adhd_confirmed = True
            if no_dx or certainty is DiagnosisCertainty.NO_DIAGNOSIS_GIVEN:
                subject_no_diagnosis = True

        summary.n_subjects += 1
        summary.n_adhd_any += int(subject_has_adhd)
        summary.n_adhd_confirmed += int(subject_adhd_confirmed)
        summary.n_no_diagnosis_given += int(subject_no_diagnosis)

    asset.status = AssetStatus.VALIDATED.value
    asset.n_records = summary.n_subjects
    asset.protected = True
    asset.validation_report = {**(asset.validation_report or {}), "phenotype": summary.to_dict()}
    record_audit(session, "phenotype.ingested", entity_type="data_asset", entity_id=asset.id,
                 summary=f"{summary.n_subjects} subjects, {summary.n_adhd_confirmed} confirmed ADHD",
                 payload=summary.to_dict())
    log.info("Phenotype ingested", extra=summary.to_dict())
    return summary


def scan_incoming(session: Session, settings: Settings) -> dict:
    """Detect and ingest any new phenotype export dropped into the intake dir."""
    incoming = settings.paths.phenotype_incoming
    incoming.mkdir(parents=True, exist_ok=True)

    assets = list(session.execute(
        select(DataAsset).where(DataAsset.kind == AssetKind.PHENOTYPE_CSV.value)
    ).scalars())
    ingested: list[dict] = []
    for asset in assets:
        if asset.status == AssetStatus.VALIDATED.value and asset.n_records:
            continue
        summary = ingest_phenotype(session, settings, asset)
        ingested.append({"path": asset.path, **summary.to_dict()})

    return {
        "incoming_dir": str(incoming),
        "n_assets": len(assets),
        "ingested": ingested,
        "available": any(a.status == AssetStatus.VALIDATED.value for a in assets),
    }


def phenotype_available(session: Session) -> bool:
    """True only when at least one subject carries a real, parsed diagnosis."""
    return session.execute(
        select(Subject.id).where(Subject.has_phenotype.is_(True)).limit(1)
    ).first() is not None
