"""Per-participant quality control roll-up for the QC dashboard page."""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from neurotribe.config import Settings
from neurotribe.database.enums import PreprocStatus, QCStatus
from neurotribe.database.models import (
    PreprocessingRun, Scan, Subject, SubjectComparison,
)


@dataclass
class QCRow:
    subject_external_id: str
    site: str | None = None
    group: str | None = None
    preprocessing: str = QCStatus.UNKNOWN.value
    anatomical: str = QCStatus.UNKNOWN.value
    bold: str = QCStatus.UNKNOWN.value
    motion: str = QCStatus.UNKNOWN.value
    mriqc: str = QCStatus.UNKNOWN.value
    alignment: str = QCStatus.UNKNOWN.value
    usable_frame_fraction: float | None = None
    mean_fd: float | None = None
    is_approximate: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def overall(self) -> str:
        values = [self.preprocessing, self.anatomical, self.bold, self.motion,
                  self.mriqc, self.alignment]
        if QCStatus.FAIL.value in values:
            return QCStatus.FAIL.value
        if QCStatus.WARNING.value in values:
            return QCStatus.WARNING.value
        if all(v == QCStatus.PASS.value for v in values):
            return QCStatus.PASS.value
        return QCStatus.UNKNOWN.value

    def to_dict(self) -> dict:
        return {
            "subject_external_id": self.subject_external_id, "site": self.site,
            "group": self.group, "preprocessing": self.preprocessing,
            "anatomical": self.anatomical, "bold": self.bold, "motion": self.motion,
            "mriqc": self.mriqc, "alignment": self.alignment,
            "usable_frame_fraction": self.usable_frame_fraction,
            "mean_fd": self.mean_fd, "is_approximate": self.is_approximate,
            "overall": self.overall, "notes": self.notes,
        }


def build_rows(session: Session, settings: Settings) -> list[QCRow]:
    """One QC row per participant with an imaging record."""
    min_fraction = float(settings.get("qc.motion.min_usable_frame_fraction", 0.5))
    fd_limit = settings.get("qc.mriqc.max_mean_fd")
    rows: list[QCRow] = []

    subjects = list(session.execute(
        select(Subject).where(Subject.has_mri.is_(True)).order_by(Subject.external_id)
    ).scalars())

    for subject in subjects:
        row = QCRow(subject_external_id=subject.external_id, site=subject.site)
        row.anatomical = QCStatus.PASS.value if subject.has_anatomical else QCStatus.FAIL.value
        if not subject.has_anatomical:
            row.notes.append("No T1w anatomical image.")

        scans = [s for s in subject.scans if s.movie != "unknown"]
        if scans:
            row.bold = QCStatus.PASS.value
            scan = max(scans, key=lambda s: (s.movie_confidence or 0.0))
            if not scan.content_present:
                row.bold = QCStatus.WARNING.value
                row.notes.append("BOLD content not materialised (annex placeholder).")
            if scan.qc is not None:
                row.mriqc = scan.qc.qc_status
                row.mean_fd = scan.qc.mean_fd
                if scan.qc.qc_reason:
                    row.notes.append(scan.qc.qc_reason)
            run = session.execute(
                select(PreprocessingRun)
                .where(PreprocessingRun.scan_id == scan.id)
                .order_by(PreprocessingRun.created_at.desc())
            ).scalars().first()
            if run is not None:
                row.preprocessing = {
                    PreprocStatus.SUCCEEDED.value: QCStatus.PASS.value,
                    PreprocStatus.APPROXIMATE.value: QCStatus.WARNING.value,
                    PreprocStatus.FAILED.value: QCStatus.FAIL.value,
                    PreprocStatus.RUNNING.value: QCStatus.UNKNOWN.value,
                    PreprocStatus.NOT_STARTED.value: QCStatus.UNKNOWN.value,
                }.get(run.status, QCStatus.UNKNOWN.value)
                row.is_approximate = bool(run.is_approximate)
                if run.is_approximate:
                    row.notes.append(
                        "Approximate development projection - excluded from final analysis."
                    )
                if run.error_message:
                    row.notes.append(run.error_message[:200])
                row.usable_frame_fraction = run.usable_frame_fraction
                row.mean_fd = run.mean_fd if run.mean_fd is not None else row.mean_fd

                if run.usable_frame_fraction is not None:
                    if run.usable_frame_fraction < min_fraction:
                        row.motion = QCStatus.FAIL.value
                        row.notes.append(
                            f"Usable frames {run.usable_frame_fraction:.0%} below "
                            f"{min_fraction:.0%} threshold."
                        )
                    elif run.usable_frame_fraction < min_fraction + 0.15:
                        row.motion = QCStatus.WARNING.value
                    else:
                        row.motion = QCStatus.PASS.value
                if fd_limit is not None and row.mean_fd is not None:
                    if row.mean_fd > float(fd_limit):
                        row.motion = QCStatus.FAIL.value
                    elif row.mean_fd > float(fd_limit) * 0.7 and row.motion == QCStatus.PASS.value:
                        row.motion = QCStatus.WARNING.value

            comparison = session.execute(
                select(SubjectComparison)
                .where(SubjectComparison.subject_id == subject.id)
                .order_by(SubjectComparison.created_at.desc())
            ).scalars().first()
            if comparison is not None:
                row.alignment = (QCStatus.PASS.value if comparison.valid
                                 else QCStatus.FAIL.value)
                if not comparison.valid and comparison.invalid_reason:
                    row.notes.append(comparison.invalid_reason[:200])
        else:
            row.bold = QCStatus.FAIL.value
            row.notes.append("No BOLD run bound to a documented HBN movie.")

        rows.append(row)
    return rows


def summarize(rows: list[QCRow]) -> dict:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.overall] = counts.get(row.overall, 0) + 1
    fractions = [r.usable_frame_fraction for r in rows if r.usable_frame_fraction is not None]
    motions = [r.mean_fd for r in rows if r.mean_fd is not None]
    return {
        "n_rows": len(rows),
        "by_status": counts,
        "n_approximate": sum(1 for r in rows if r.is_approximate),
        "median_usable_frame_fraction": (
            float(sorted(fractions)[len(fractions) // 2]) if fractions else None
        ),
        "median_mean_fd": float(sorted(motions)[len(motions) // 2]) if motions else None,
    }
