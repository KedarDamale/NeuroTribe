"""Controlled vocabularies shared by the ORM, the API and the frontend.

These are stored as strings (not native PG enums) so that adding a value never
requires a migration, while `str, Enum` keeps them type-safe in Python and
JSON-serialisable for the API.
"""

from __future__ import annotations

from enum import Enum


class StageState(str, Enum):
    """State of a pipeline stage. Exactly the vocabulary in the specification."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    WAITING_EXTERNAL = "WAITING_EXTERNAL"
    FAILED_RETRYABLE = "FAILED_RETRYABLE"
    FAILED_FINAL = "FAILED_FINAL"
    SKIPPED = "SKIPPED"
    BLOCKED = "BLOCKED"
    PARTIAL = "PARTIAL"

    @property
    def is_terminal(self) -> bool:
        return self in {StageState.DONE, StageState.FAILED_FINAL, StageState.SKIPPED}

    @property
    def is_runnable(self) -> bool:
        return self in {StageState.PENDING, StageState.FAILED_RETRYABLE, StageState.PARTIAL}


class JobState(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    CACHED = "CACHED"


class BlockerKind(str, Enum):
    """Why a stage cannot proceed."""

    PHENOTYPE_ACCESS = "PHENOTYPE_ACCESS"
    STIMULUS_MISSING = "STIMULUS_MISSING"
    FREESURFER_LICENSE = "FREESURFER_LICENSE"
    BIDS_MISSING = "BIDS_MISSING"
    METADATA_MISSING = "METADATA_MISSING"
    MRIQC_MISSING = "MRIQC_MISSING"
    DISK_SPACE = "DISK_SPACE"
    DOCKER_UNAVAILABLE = "DOCKER_UNAVAILABLE"
    TRIBE_MODEL = "TRIBE_MODEL"
    COHORT_TOO_SMALL = "COHORT_TOO_SMALL"
    HARDWARE = "HARDWARE"
    OTHER = "OTHER"


class BlockerSeverity(str, Enum):
    EXTERNAL = "EXTERNAL"   # requires human/institutional action - cannot be automated
    ACTIONABLE = "ACTIONABLE"  # the system can fix it (e.g. free disk, pull image)
    INFO = "INFO"


class AssetKind(str, Enum):
    HBN_METADATA = "HBN_METADATA"
    MRIQC_FUNCTIONAL = "MRIQC_FUNCTIONAL"
    MRIQC_ANATOMICAL = "MRIQC_ANATOMICAL"
    BIDS_ROOT = "BIDS_ROOT"
    PHENOTYPE_CSV = "PHENOTYPE_CSV"
    STIMULUS_VIDEO = "STIMULUS_VIDEO"
    FMRIPREP_DERIVATIVE = "FMRIPREP_DERIVATIVE"
    TRIBE_PREDICTION = "TRIBE_PREDICTION"
    ANALYSIS_ARTIFACT = "ANALYSIS_ARTIFACT"
    REPORT = "REPORT"
    UNKNOWN = "UNKNOWN"


class AssetStatus(str, Enum):
    DISCOVERED = "DISCOVERED"
    VALIDATED = "VALIDATED"
    INVALID = "INVALID"
    MISSING = "MISSING"
    QUARANTINED = "QUARANTINED"


class DiagnosisCertainty(str, Enum):
    """HBN clinician-consensus certainty vocabulary."""

    CONFIRMED = "Confirmed"
    PRESUMPTIVE = "Presumptive"
    REQUIRES_CONFIRMATION = "Requires Confirmation"
    RULE_OUT = "Rule-out"
    BY_HISTORY = "By History"
    PAST = "Past"
    NO_DIAGNOSIS_GIVEN = "No Diagnosis Given"
    INCOMPLETE_EVAL = "Incomplete Eval"
    UNKNOWN = "Unknown"


class CohortGroup(str, Enum):
    """Cohort assignment. Note the deliberate absence of 'healthy controls'."""

    CONFIRMED_ADHD = "CONFIRMED_ADHD"
    NO_DIAGNOSIS_GIVEN = "NO_DIAGNOSIS_GIVEN"
    NON_ADHD_COMPARISON = "NON_ADHD_COMPARISON"
    EXCLUDED_UNCERTAIN = "EXCLUDED_UNCERTAIN"
    UNASSIGNED = "UNASSIGNED"

    @property
    def display(self) -> str:
        return {
            CohortGroup.CONFIRMED_ADHD: "Confirmed ADHD",
            CohortGroup.NO_DIAGNOSIS_GIVEN: "No Diagnosis Given",
            CohortGroup.NON_ADHD_COMPARISON: "non-ADHD comparison cohort",
            CohortGroup.EXCLUDED_UNCERTAIN: "Excluded / uncertain",
            CohortGroup.UNASSIGNED: "Unassigned",
        }[self]


class QCStatus(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


class MovieKey(str, Enum):
    DESPICABLE_ME = "despicable_me"
    THE_PRESENT = "the_present"
    UNKNOWN = "unknown"


class PreprocStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    APPROXIMATE = "APPROXIMATE"  # dev-only projection; excluded from final analysis


class AnalysisTier(str, Enum):
    PRIMARY = "PRIMARY"
    EXPLORATORY = "EXPLORATORY"


class TribeBackend(str, Enum):
    REAL = "real"
    MOCK = "mock"


class ExclusionReason(str, Enum):
    """Every exclusion must carry one of these — never a silent drop."""

    NO_PHENOTYPE = "NO_PHENOTYPE"
    NO_MOVIE_BOLD = "NO_MOVIE_BOLD"
    NO_ANATOMICAL = "NO_ANATOMICAL"
    NO_SCAN_METADATA = "NO_SCAN_METADATA"
    PREPROCESSING_FAILED = "PREPROCESSING_FAILED"
    QC_FAILED_MOTION = "QC_FAILED_MOTION"
    QC_FAILED_MRIQC = "QC_FAILED_MRIQC"
    INSUFFICIENT_USABLE_FRAMES = "INSUFFICIENT_USABLE_FRAMES"
    UNCERTAIN_DIAGNOSIS = "UNCERTAIN_DIAGNOSIS"
    ALIGNMENT_FAILED = "ALIGNMENT_FAILED"
    APPROXIMATE_SURFACE = "APPROXIMATE_SURFACE"
    COMMERCIAL_USE_RESTRICTED = "COMMERCIAL_USE_RESTRICTED"
    DUPLICATE_SUBJECT = "DUPLICATE_SUBJECT"
    NOT_ELIGIBLE_FOR_CONTRAST = "NOT_ELIGIBLE_FOR_CONTRAST"
