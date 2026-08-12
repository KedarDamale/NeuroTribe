"""ORM models.

Design rules enforced here:
  * No large scientific array is ever stored in a column - only paths + hashes.
  * Every participant exclusion carries a machine-readable reason.
  * Every derived result carries the provenance needed to reproduce it.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text,
    UniqueConstraint, JSON,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from neurotribe.database.base import Base
from neurotribe.database.enums import (
    AnalysisTier, AssetKind, AssetStatus, BlockerKind, BlockerSeverity,
    CohortGroup, DiagnosisCertainty, JobState, MovieKey, PreprocStatus,
    QCStatus, StageState,
)


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now, nullable=False,
    )


# --------------------------------------------------------------------------
# Participants and phenotype
# --------------------------------------------------------------------------

class Subject(Base, TimestampMixin):
    __tablename__ = "subjects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    # HBN external identifier, e.g. NDARAA075AMK
    external_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    # BIDS participant label, e.g. sub-NDARAA075AMK
    bids_participant_id: Mapped[str | None] = mapped_column(String(80), index=True)

    site: Mapped[str | None] = mapped_column(String(64), index=True)
    release: Mapped[str | None] = mapped_column(String(32))
    age: Mapped[float | None] = mapped_column(Float)
    sex: Mapped[str | None] = mapped_column(String(16))
    handedness: Mapped[str | None] = mapped_column(String(16))

    # HBN marks a subset of participants as restricted from commercial use.
    commercial_use_allowed: Mapped[bool | None] = mapped_column(Boolean)

    has_mri: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    has_phenotype: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    has_movie_bold: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    has_anatomical: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    diagnoses: Mapped[list["Diagnosis"]] = relationship(back_populates="subject", cascade="all, delete-orphan")
    scans: Mapped[list["Scan"]] = relationship(back_populates="subject", cascade="all, delete-orphan")
    memberships: Mapped[list["CohortMember"]] = relationship(back_populates="subject", cascade="all, delete-orphan")

    __table_args__ = (Index("ix_subjects_flags", "has_phenotype", "has_movie_bold"),)


class Diagnosis(Base, TimestampMixin):
    __tablename__ = "diagnoses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    subject_id: Mapped[str] = mapped_column(ForeignKey("subjects.id", ondelete="CASCADE"), index=True)

    # 1..10 - HBN reports up to ten diagnoses per participant.
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    raw_label: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_label: Mapped[str | None] = mapped_column(String(255), index=True)
    category: Mapped[str | None] = mapped_column(String(128), index=True)
    certainty: Mapped[str] = mapped_column(String(48), default=DiagnosisCertainty.UNKNOWN.value, index=True)
    raw_certainty: Mapped[str | None] = mapped_column(String(128))

    is_adhd: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    is_no_diagnosis: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    instrument: Mapped[str | None] = mapped_column(String(128))
    source_asset_id: Mapped[str | None] = mapped_column(ForeignKey("data_assets.id"))

    subject: Mapped[Subject] = relationship(back_populates="diagnoses")

    __table_args__ = (UniqueConstraint("subject_id", "ordinal", name="uq_diagnosis_subject_ordinal"),)


# --------------------------------------------------------------------------
# Imaging
# --------------------------------------------------------------------------

class Scan(Base, TimestampMixin):
    __tablename__ = "scans"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    subject_id: Mapped[str] = mapped_column(ForeignKey("subjects.id", ondelete="CASCADE"), index=True)

    # BIDS entities, discovered - never guessed from filenames.
    task: Mapped[str | None] = mapped_column(String(64), index=True)
    run: Mapped[str | None] = mapped_column(String(16))
    session: Mapped[str | None] = mapped_column(String(32))
    acquisition: Mapped[str | None] = mapped_column(String(64))
    suffix: Mapped[str | None] = mapped_column(String(32))
    datatype: Mapped[str | None] = mapped_column(String(16))

    movie: Mapped[str] = mapped_column(String(32), default=MovieKey.UNKNOWN.value, index=True)
    movie_confidence: Mapped[float | None] = mapped_column(Float)
    movie_evidence: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    bold_path: Mapped[str | None] = mapped_column(Text)
    bold_json_path: Mapped[str | None] = mapped_column(Text)
    t1w_path: Mapped[str | None] = mapped_column(Text)
    fieldmap_paths: Mapped[list[str]] = mapped_column(JSON, default=list)

    repetition_time: Mapped[float | None] = mapped_column(Float)
    n_volumes: Mapped[int | None] = mapped_column(Integer)
    duration_sec: Mapped[float | None] = mapped_column(Float)
    echo_time: Mapped[float | None] = mapped_column(Float)
    slice_timing_present: Mapped[bool | None] = mapped_column(Boolean)

    scanner: Mapped[str | None] = mapped_column(String(128))
    site: Mapped[str | None] = mapped_column(String(64), index=True)

    # git-annex / DataLad placeholder rather than real binary content.
    content_present: Mapped[bool] = mapped_column(Boolean, default=True)

    sidecar_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    subject: Mapped[Subject] = relationship(back_populates="scans")
    qc: Mapped["ScanQC | None"] = relationship(back_populates="scan", uselist=False, cascade="all, delete-orphan")
    preprocessing_runs: Mapped[list["PreprocessingRun"]] = relationship(
        back_populates="scan", cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint("subject_id", "task", "run", "session", name="uq_scan_entities"),
        Index("ix_scans_movie_site", "movie", "site"),
    )


class ScanQC(Base, TimestampMixin):
    """MRIQC-derived image quality metrics.

    HBN releases imaging data regardless of quality and publishes MRIQC IQMs, so
    the schema is *dynamic*: known columns are promoted, everything else is kept
    in ``extra_iqms`` rather than dropped.
    """

    __tablename__ = "scan_qc"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    scan_id: Mapped[str] = mapped_column(ForeignKey("scans.id", ondelete="CASCADE"), unique=True, index=True)
    subject_external_id: Mapped[str | None] = mapped_column(String(64), index=True)

    mean_fd: Mapped[float | None] = mapped_column(Float)
    max_fd: Mapped[float | None] = mapped_column(Float)
    dvars: Mapped[float | None] = mapped_column(Float)
    tsnr: Mapped[float | None] = mapped_column(Float)
    efc: Mapped[float | None] = mapped_column(Float)
    fber: Mapped[float | None] = mapped_column(Float)
    snr: Mapped[float | None] = mapped_column(Float)
    gsr_x: Mapped[float | None] = mapped_column(Float)
    gsr_y: Mapped[float | None] = mapped_column(Float)
    fd_perc: Mapped[float | None] = mapped_column(Float)

    extra_iqms: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    qc_status: Mapped[str] = mapped_column(String(16), default=QCStatus.UNKNOWN.value, index=True)
    qc_reason: Mapped[str | None] = mapped_column(Text)
    source_asset_id: Mapped[str | None] = mapped_column(ForeignKey("data_assets.id"))

    scan: Mapped[Scan] = relationship(back_populates="qc")


# --------------------------------------------------------------------------
# Stimuli and generic data assets
# --------------------------------------------------------------------------

class Stimulus(Base, TimestampMixin):
    __tablename__ = "stimuli"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    key: Mapped[str] = mapped_column(String(64), unique=True, index=True)  # MovieKey
    label: Mapped[str] = mapped_column(String(128))

    path: Mapped[str | None] = mapped_column(Text)
    sha256: Mapped[str | None] = mapped_column(String(64), index=True)
    size_bytes: Mapped[int | None] = mapped_column(Integer)

    duration_sec: Mapped[float | None] = mapped_column(Float)
    fps: Mapped[float | None] = mapped_column(Float)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    has_audio: Mapped[bool | None] = mapped_column(Boolean)
    container: Mapped[str | None] = mapped_column(String(16))

    expected_duration_sec: Mapped[float | None] = mapped_column(Float)
    source_interval_start: Mapped[str | None] = mapped_column(String(16))
    source_interval_end: Mapped[str | None] = mapped_column(String(16))

    validated: Mapped[bool] = mapped_column(Boolean, default=False)
    validation_notes: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    first_frame_path: Mapped[str | None] = mapped_column(Text)
    last_frame_path: Mapped[str | None] = mapped_column(Text)

    # Provenance of how the operator supplied the file. Never auto-downloaded.
    provenance_note: Mapped[str | None] = mapped_column(Text)


class DataAsset(Base, TimestampMixin):
    """Registry row for every file/directory the system knows about."""

    __tablename__ = "data_assets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    kind: Mapped[str] = mapped_column(String(48), index=True, default=AssetKind.UNKNOWN.value)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    absolute_path: Mapped[str] = mapped_column(Text, nullable=False, unique=True)

    size_bytes: Mapped[int | None] = mapped_column(Integer)
    sha256: Mapped[str | None] = mapped_column(String(64), index=True)
    hash_is_partial: Mapped[bool] = mapped_column(Boolean, default=False)
    modified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    status: Mapped[str] = mapped_column(String(24), default=AssetStatus.DISCOVERED.value, index=True)
    validation_report: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    n_records: Mapped[int | None] = mapped_column(Integer)
    is_directory: Mapped[bool] = mapped_column(Boolean, default=False)

    # True when the asset holds DUA-protected content and must never leave the box.
    protected: Mapped[bool] = mapped_column(Boolean, default=False)

    __table_args__ = (Index("ix_assets_kind_status", "kind", "status"),)


# --------------------------------------------------------------------------
# Cohorts
# --------------------------------------------------------------------------

class Cohort(Base, TimestampMixin):
    __tablename__ = "cohorts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(128), index=True)
    tier: Mapped[str] = mapped_column(String(16), default=AnalysisTier.PRIMARY.value)
    movie: Mapped[str] = mapped_column(String(32), default=MovieKey.UNKNOWN.value)

    definition: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    cohort_hash: Mapped[str] = mapped_column(String(64), index=True)
    analysis_config_hash: Mapped[str | None] = mapped_column(String(64))

    n_case: Mapped[int] = mapped_column(Integer, default=0)
    n_control: Mapped[int] = mapped_column(Integer, default=0)
    n_excluded: Mapped[int] = mapped_column(Integer, default=0)

    warnings: Mapped[list[str]] = mapped_column(JSON, default=list)
    members: Mapped[list["CohortMember"]] = relationship(back_populates="cohort", cascade="all, delete-orphan")


class CohortMember(Base, TimestampMixin):
    __tablename__ = "cohort_members"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    cohort_id: Mapped[str] = mapped_column(ForeignKey("cohorts.id", ondelete="CASCADE"), index=True)
    subject_id: Mapped[str] = mapped_column(ForeignKey("subjects.id", ondelete="CASCADE"), index=True)
    scan_id: Mapped[str | None] = mapped_column(ForeignKey("scans.id", ondelete="SET NULL"))

    group: Mapped[str] = mapped_column(String(32), default=CohortGroup.UNASSIGNED.value, index=True)
    included: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    # Mandatory whenever included is False. Enforced by the cohort builder.
    exclusion_reason: Mapped[str | None] = mapped_column(String(64), index=True)
    exclusion_detail: Mapped[str | None] = mapped_column(Text)

    # Confound snapshot frozen at cohort-build time.
    age: Mapped[float | None] = mapped_column(Float)
    sex: Mapped[str | None] = mapped_column(String(16))
    site: Mapped[str | None] = mapped_column(String(64))
    scanner: Mapped[str | None] = mapped_column(String(128))
    mean_fd: Mapped[float | None] = mapped_column(Float)
    usable_frame_fraction: Mapped[float | None] = mapped_column(Float)
    diagnostic_certainty: Mapped[str | None] = mapped_column(String(48))
    comorbidities: Mapped[list[str]] = mapped_column(JSON, default=list)
    commercial_use_allowed: Mapped[bool | None] = mapped_column(Boolean)

    cohort: Mapped[Cohort] = relationship(back_populates="members")
    subject: Mapped[Subject] = relationship(back_populates="memberships")

    __table_args__ = (UniqueConstraint("cohort_id", "subject_id", name="uq_cohort_subject"),)


# --------------------------------------------------------------------------
# Processing runs
# --------------------------------------------------------------------------

class PreprocessingRun(Base, TimestampMixin):
    __tablename__ = "preprocessing_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    subject_id: Mapped[str] = mapped_column(ForeignKey("subjects.id", ondelete="CASCADE"), index=True)
    scan_id: Mapped[str | None] = mapped_column(ForeignKey("scans.id", ondelete="SET NULL"), index=True)

    engine: Mapped[str] = mapped_column(String(32), default="fmriprep")
    engine_version: Mapped[str | None] = mapped_column(String(32))
    container_image: Mapped[str | None] = mapped_column(String(255))

    status: Mapped[str] = mapped_column(String(24), default=PreprocStatus.NOT_STARTED.value, index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    exit_code: Mapped[int | None] = mapped_column(Integer)
    attempt: Mapped[int] = mapped_column(Integer, default=0)

    cache_key: Mapped[str | None] = mapped_column(String(96), index=True)
    config_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    output_dir: Mapped[str | None] = mapped_column(Text)
    surface_lh_path: Mapped[str | None] = mapped_column(Text)
    surface_rh_path: Mapped[str | None] = mapped_column(Text)
    confounds_path: Mapped[str | None] = mapped_column(Text)
    confounds_json_path: Mapped[str | None] = mapped_column(Text)
    report_path: Mapped[str | None] = mapped_column(Text)
    log_path: Mapped[str | None] = mapped_column(Text)

    # True when surfaces came from the dev-only volume projection.
    is_approximate: Mapped[bool] = mapped_column(Boolean, default=False)
    error_message: Mapped[str | None] = mapped_column(Text)

    # Denoising outcome
    denoise_strategy: Mapped[str | None] = mapped_column(String(64))
    n_volumes: Mapped[int | None] = mapped_column(Integer)
    n_usable_frames: Mapped[int | None] = mapped_column(Integer)
    usable_frame_fraction: Mapped[float | None] = mapped_column(Float)
    mean_fd: Mapped[float | None] = mapped_column(Float)
    n_nonsteady_state: Mapped[int | None] = mapped_column(Integer)
    denoised_path: Mapped[str | None] = mapped_column(Text)
    censor_mask_path: Mapped[str | None] = mapped_column(Text)

    scan: Mapped[Scan] = relationship(back_populates="preprocessing_runs")


class TribeRun(Base, TimestampMixin):
    """TRIBE inference is executed ONCE PER STIMULUS, not once per subject."""

    __tablename__ = "tribe_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    stimulus_id: Mapped[str | None] = mapped_column(ForeignKey("stimuli.id", ondelete="SET NULL"), index=True)
    movie: Mapped[str] = mapped_column(String(32), index=True)

    backend: Mapped[str] = mapped_column(String(16), default="real")
    model_id: Mapped[str | None] = mapped_column(String(128))
    model_revision: Mapped[str | None] = mapped_column(String(64))
    tribe_commit: Mapped[str | None] = mapped_column(String(64))
    tribe_version: Mapped[str | None] = mapped_column(String(32))
    device: Mapped[str | None] = mapped_column(String(16))

    cache_key: Mapped[str] = mapped_column(String(96), unique=True, index=True)
    stimulus_sha256: Mapped[str | None] = mapped_column(String(64))

    predictions_path: Mapped[str | None] = mapped_column(Text)
    segments_path: Mapped[str | None] = mapped_column(Text)
    events_path: Mapped[str | None] = mapped_column(Text)
    manifest_path: Mapped[str | None] = mapped_column(Text)

    n_timepoints: Mapped[int | None] = mapped_column(Integer)
    n_vertices: Mapped[int | None] = mapped_column(Integer)
    surface_space: Mapped[str | None] = mapped_column(String(32))
    hemi_order: Mapped[list[str]] = mapped_column(JSON, default=list)
    time_start_sec: Mapped[float | None] = mapped_column(Float)
    time_end_sec: Mapped[float | None] = mapped_column(Float)
    applies_own_hrf_offset: Mapped[bool | None] = mapped_column(Boolean)

    geometry_validated: Mapped[bool] = mapped_column(Boolean, default=False)
    geometry_report: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    status: Mapped[str] = mapped_column(String(24), default="PENDING", index=True)
    error_message: Mapped[str | None] = mapped_column(Text)


class SubjectComparison(Base, TimestampMixin):
    """Subject-level alignment + deviation result against a TRIBE prediction."""

    __tablename__ = "subject_comparisons"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    subject_id: Mapped[str] = mapped_column(ForeignKey("subjects.id", ondelete="CASCADE"), index=True)
    scan_id: Mapped[str | None] = mapped_column(ForeignKey("scans.id", ondelete="SET NULL"))
    preprocessing_run_id: Mapped[str | None] = mapped_column(ForeignKey("preprocessing_runs.id", ondelete="SET NULL"))
    tribe_run_id: Mapped[str | None] = mapped_column(ForeignKey("tribe_runs.id", ondelete="SET NULL"), index=True)

    movie: Mapped[str] = mapped_column(String(32), index=True)
    cache_key: Mapped[str | None] = mapped_column(String(96), index=True)

    # Whole-cortex summaries
    global_agreement_r: Mapped[float | None] = mapped_column(Float)
    global_mad: Mapped[float | None] = mapped_column(Float)
    global_residual_variance: Mapped[float | None] = mapped_column(Float)

    n_shared_timepoints: Mapped[int | None] = mapped_column(Integer)
    n_usable_timepoints: Mapped[int | None] = mapped_column(Integer)
    usable_frame_fraction: Mapped[float | None] = mapped_column(Float)
    tr: Mapped[float | None] = mapped_column(Float)

    vertex_r_path: Mapped[str | None] = mapped_column(Text)
    vertex_mad_path: Mapped[str | None] = mapped_column(Text)
    residual_path: Mapped[str | None] = mapped_column(Text)
    rolling_deviation_path: Mapped[str | None] = mapped_column(Text)
    peak_windows: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)

    alignment_report: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    sanity_report: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    valid: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    invalid_reason: Mapped[str | None] = mapped_column(Text)
    is_approximate: Mapped[bool] = mapped_column(Boolean, default=False)

    roi_metrics: Mapped[list["RoiMetric"]] = relationship(back_populates="comparison", cascade="all, delete-orphan")
    network_metrics: Mapped[list["NetworkMetric"]] = relationship(back_populates="comparison", cascade="all, delete-orphan")


class RoiMetric(Base):
    __tablename__ = "roi_metrics"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    comparison_id: Mapped[str] = mapped_column(
        ForeignKey("subject_comparisons.id", ondelete="CASCADE"), index=True,
    )
    atlas: Mapped[str] = mapped_column(String(64), default="schaefer")
    n_parcels: Mapped[int] = mapped_column(Integer, default=200)
    roi_index: Mapped[int] = mapped_column(Integer, index=True)
    roi_name: Mapped[str] = mapped_column(String(128))
    network: Mapped[str | None] = mapped_column(String(64), index=True)
    hemisphere: Mapped[str | None] = mapped_column(String(2))

    agreement_r: Mapped[float | None] = mapped_column(Float)
    mad: Mapped[float | None] = mapped_column(Float)
    residual_variance: Mapped[float | None] = mapped_column(Float)
    n_vertices: Mapped[int | None] = mapped_column(Integer)

    comparison: Mapped[SubjectComparison] = relationship(back_populates="roi_metrics")

    __table_args__ = (
        UniqueConstraint("comparison_id", "roi_index", name="uq_roi_metric"),
        Index("ix_roi_metric_lookup", "comparison_id", "network"),
    )


class NetworkMetric(Base):
    __tablename__ = "network_metrics"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    comparison_id: Mapped[str] = mapped_column(
        ForeignKey("subject_comparisons.id", ondelete="CASCADE"), index=True,
    )
    network: Mapped[str] = mapped_column(String(64), index=True)
    agreement_r: Mapped[float | None] = mapped_column(Float)
    mad: Mapped[float | None] = mapped_column(Float)
    residual_variance: Mapped[float | None] = mapped_column(Float)
    n_vertices: Mapped[int | None] = mapped_column(Integer)

    comparison: Mapped[SubjectComparison] = relationship(back_populates="network_metrics")

    __table_args__ = (UniqueConstraint("comparison_id", "network", name="uq_network_metric"),)


# --------------------------------------------------------------------------
# Group analysis
# --------------------------------------------------------------------------

class GroupAnalysisRun(Base, TimestampMixin):
    __tablename__ = "group_analysis_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    cohort_id: Mapped[str | None] = mapped_column(ForeignKey("cohorts.id", ondelete="SET NULL"), index=True)
    tier: Mapped[str] = mapped_column(String(16), default=AnalysisTier.PRIMARY.value, index=True)
    name: Mapped[str] = mapped_column(String(160))

    case_group: Mapped[str] = mapped_column(String(32))
    control_group: Mapped[str] = mapped_column(String(32))
    n_case: Mapped[int] = mapped_column(Integer, default=0)
    n_control: Mapped[int] = mapped_column(Integer, default=0)

    model_formula: Mapped[str | None] = mapped_column(Text)
    covariates: Mapped[list[str]] = mapped_column(JSON, default=list)
    correction: Mapped[str | None] = mapped_column(String(32))
    alpha: Mapped[float | None] = mapped_column(Float)

    results_path: Mapped[str | None] = mapped_column(Text)
    results_summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    provenance: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    status: Mapped[str] = mapped_column(String(24), default="PENDING", index=True)
    error_message: Mapped[str | None] = mapped_column(Text)
    sanity_passed: Mapped[bool] = mapped_column(Boolean, default=False)
    sanity_report: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class GroupResult(Base):
    """One row per tested unit (ROI or network) within a group analysis."""

    __tablename__ = "group_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("group_analysis_runs.id", ondelete="CASCADE"), index=True)

    unit_type: Mapped[str] = mapped_column(String(16), index=True)  # roi | network | global
    unit_name: Mapped[str] = mapped_column(String(128), index=True)
    unit_index: Mapped[int | None] = mapped_column(Integer)
    network: Mapped[str | None] = mapped_column(String(64), index=True)
    metric: Mapped[str] = mapped_column(String(48), index=True)

    mean_case: Mapped[float | None] = mapped_column(Float)
    mean_control: Mapped[float | None] = mapped_column(Float)
    sd_case: Mapped[float | None] = mapped_column(Float)
    sd_control: Mapped[float | None] = mapped_column(Float)

    beta_adhd: Mapped[float | None] = mapped_column(Float)
    se_adhd: Mapped[float | None] = mapped_column(Float)
    t_stat: Mapped[float | None] = mapped_column(Float)
    p_value: Mapped[float | None] = mapped_column(Float)
    q_value: Mapped[float | None] = mapped_column(Float)
    effect_size: Mapped[float | None] = mapped_column(Float)
    ci_low: Mapped[float | None] = mapped_column(Float)
    ci_high: Mapped[float | None] = mapped_column(Float)
    n_case: Mapped[int | None] = mapped_column(Integer)
    n_control: Mapped[int | None] = mapped_column(Integer)

    __table_args__ = (
        Index("ix_group_result_lookup", "run_id", "unit_type", "metric"),
    )


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

class PipelineStage(Base, TimestampMixin):
    """Persistent state-machine node driving the Autopilot."""

    __tablename__ = "pipeline_stages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    label: Mapped[str] = mapped_column(String(160))
    phase: Mapped[int] = mapped_column(Integer, default=0, index=True)
    order: Mapped[int] = mapped_column(Integer, default=0)

    state: Mapped[str] = mapped_column(String(24), default=StageState.PENDING.value, index=True)
    detail: Mapped[str | None] = mapped_column(Text)
    progress: Mapped[float] = mapped_column(Float, default=0.0)

    depends_on: Mapped[list[str]] = mapped_column(JSON, default=list)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    last_error: Mapped[str | None] = mapped_column(Text)
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    result: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class Job(Base, TimestampMixin):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(160), index=True)
    kind: Mapped[str] = mapped_column(String(64), index=True)
    stage_key: Mapped[str | None] = mapped_column(String(64), index=True)
    subject_external_id: Mapped[str | None] = mapped_column(String(64), index=True)

    state: Mapped[str] = mapped_column(String(24), default=JobState.QUEUED.value, index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str | None] = mapped_column(Text)

    celery_task_id: Mapped[str | None] = mapped_column(String(64), index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    elapsed_sec: Mapped[float | None] = mapped_column(Float)
    eta_sec: Mapped[float | None] = mapped_column(Float)

    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False)

    cpu_percent: Mapped[float | None] = mapped_column(Float)
    mem_mb: Mapped[float | None] = mapped_column(Float)
    gpu_name: Mapped[str | None] = mapped_column(String(128))
    disk_mb: Mapped[float | None] = mapped_column(Float)

    log_path: Mapped[str | None] = mapped_column(Text)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text)


class Artifact(Base, TimestampMixin):
    """Any generated output file with its provenance manifest."""

    __tablename__ = "artifacts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    kind: Mapped[str] = mapped_column(String(64), index=True)
    label: Mapped[str] = mapped_column(String(200))
    path: Mapped[str] = mapped_column(Text)
    media_type: Mapped[str | None] = mapped_column(String(80))
    size_bytes: Mapped[int | None] = mapped_column(Integer)
    sha256: Mapped[str | None] = mapped_column(String(64))

    subject_external_id: Mapped[str | None] = mapped_column(String(64), index=True)
    group_run_id: Mapped[str | None] = mapped_column(String(36), index=True)
    provenance: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    tier: Mapped[str | None] = mapped_column(String(16))


class Blocker(Base, TimestampMixin):
    """Something preventing progress. Surfaced prominently in the UI."""

    __tablename__ = "blockers"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    kind: Mapped[str] = mapped_column(String(48), index=True)
    severity: Mapped[str] = mapped_column(String(16), default=BlockerSeverity.EXTERNAL.value, index=True)
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text)
    required_action: Mapped[str | None] = mapped_column(Text)
    reference_url: Mapped[str | None] = mapped_column(Text)
    blocks_stages: Mapped[list[str]] = mapped_column(JSON, default=list)

    active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    context: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    __table_args__ = (UniqueConstraint("kind", "title", name="uq_blocker_identity"),)


class AuditEvent(Base):
    """Append-only audit trail. Never updated, never deleted."""

    __tablename__ = "audit_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    actor: Mapped[str] = mapped_column(String(64), default="autopilot")
    action: Mapped[str] = mapped_column(String(96), index=True)
    entity_type: Mapped[str | None] = mapped_column(String(64), index=True)
    entity_id: Mapped[str | None] = mapped_column(String(64), index=True)
    summary: Mapped[str | None] = mapped_column(Text)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class SystemProbe(Base, TimestampMixin):
    """Latest hardware / software readiness snapshot."""

    __tablename__ = "system_probes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    hostname: Mapped[str | None] = mapped_column(String(128))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    ready: Mapped[bool] = mapped_column(Boolean, default=False)
    warnings: Mapped[list[str]] = mapped_column(JSON, default=list)
