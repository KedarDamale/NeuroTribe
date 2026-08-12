"""Diagnosis-level logic for cohort assignment.

The primary ADHD cohort is **Confirmed ADHD only**. Certainty levels are never
pooled: combining Confirmed + Presumptive + Rule-out would silently redefine the
scientific question.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from neurotribe.database.enums import CohortGroup, DiagnosisCertainty
from neurotribe.database.models import Diagnosis, Subject


@dataclass
class DiagnosisProfile:
    """Everything cohort assignment needs to know about one participant."""

    external_id: str
    has_phenotype: bool = False
    adhd_certainties: list[str] = field(default_factory=list)
    has_no_diagnosis_given: bool = False
    all_labels: list[str] = field(default_factory=list)
    comorbidities: list[str] = field(default_factory=list)
    n_diagnoses: int = 0

    @property
    def adhd_confirmed(self) -> bool:
        return DiagnosisCertainty.CONFIRMED.value in self.adhd_certainties

    @property
    def adhd_any(self) -> bool:
        return bool(self.adhd_certainties)

    def adhd_at(self, certainties: Sequence[str]) -> bool:
        wanted = {c.value if isinstance(c, DiagnosisCertainty) else str(c) for c in certainties}
        return any(c in wanted for c in self.adhd_certainties)

    @property
    def strongest_adhd_certainty(self) -> str | None:
        priority = [
            DiagnosisCertainty.CONFIRMED.value,
            DiagnosisCertainty.PRESUMPTIVE.value,
            DiagnosisCertainty.BY_HISTORY.value,
            DiagnosisCertainty.PAST.value,
            DiagnosisCertainty.REQUIRES_CONFIRMATION.value,
            DiagnosisCertainty.RULE_OUT.value,
        ]
        for level in priority:
            if level in self.adhd_certainties:
                return level
        return self.adhd_certainties[0] if self.adhd_certainties else None


def build_profile(subject: Subject, diagnoses: Iterable[Diagnosis] | None = None) -> DiagnosisProfile:
    records = list(diagnoses if diagnoses is not None else subject.diagnoses)
    profile = DiagnosisProfile(
        external_id=subject.external_id,
        has_phenotype=bool(subject.has_phenotype),
        n_diagnoses=len(records),
    )
    for diagnosis in records:
        label = diagnosis.normalized_label or diagnosis.raw_label
        profile.all_labels.append(label)
        if diagnosis.is_adhd:
            profile.adhd_certainties.append(diagnosis.certainty)
        elif diagnosis.is_no_diagnosis or diagnosis.certainty == DiagnosisCertainty.NO_DIAGNOSIS_GIVEN.value:
            profile.has_no_diagnosis_given = True
        else:
            profile.comorbidities.append(label)
    return profile


def assign_primary_group(profile: DiagnosisProfile, config: dict) -> tuple[CohortGroup, str | None]:
    """Primary contrast: Confirmed ADHD vs No Diagnosis Given.

    Returns the group and, when the participant is excluded, a human-readable
    reason. A participant is never silently dropped.
    """
    if not profile.has_phenotype:
        return CohortGroup.EXCLUDED_UNCERTAIN, "No phenotype record available."

    case_certainties = config.get("primary", {}).get("case", {}).get("certainty", ["Confirmed"])
    if profile.adhd_at(case_certainties):
        return CohortGroup.CONFIRMED_ADHD, None

    if profile.has_no_diagnosis_given and not profile.adhd_any:
        return CohortGroup.NO_DIAGNOSIS_GIVEN, None

    if profile.adhd_any:
        return (
            CohortGroup.EXCLUDED_UNCERTAIN,
            f"ADHD present but not at required certainty "
            f"(strongest: {profile.strongest_adhd_certainty}).",
        )

    return (
        CohortGroup.EXCLUDED_UNCERTAIN,
        "Has other diagnoses; not eligible for the primary "
        "Confirmed-ADHD vs No-Diagnosis-Given contrast.",
    )


def assign_exploratory_group(profile: DiagnosisProfile, config: dict) -> tuple[CohortGroup, str | None]:
    """Exploratory contrast: Confirmed ADHD vs non-ADHD comparison cohort.

    The comparison group may carry other diagnoses. It is therefore **never**
    called 'healthy controls'.
    """
    if not profile.has_phenotype:
        return CohortGroup.EXCLUDED_UNCERTAIN, "No phenotype record available."

    case_certainties = config.get("primary", {}).get("case", {}).get("certainty", ["Confirmed"])
    if profile.adhd_at(case_certainties):
        return CohortGroup.CONFIRMED_ADHD, None

    excluded = config.get("exploratory", {}).get("control", {}).get(
        "exclude_adhd_certainty", ["Confirmed", "Presumptive", "By History", "Past"],
    )
    if profile.adhd_at(excluded):
        return (
            CohortGroup.EXCLUDED_UNCERTAIN,
            f"ADHD at certainty '{profile.strongest_adhd_certainty}' - ambiguous for "
            "either group in the exploratory contrast.",
        )

    return CohortGroup.NON_ADHD_COMPARISON, None


def group_label(group: CohortGroup) -> str:
    """Display label. Note that no path here can produce 'healthy controls'."""
    return group.display


def summarize(profiles: Iterable[DiagnosisProfile]) -> dict:
    profiles = list(profiles)
    certainty_counts: dict[str, int] = {}
    for profile in profiles:
        for certainty in profile.adhd_certainties:
            certainty_counts[certainty] = certainty_counts.get(certainty, 0) + 1
    return {
        "n_profiles": len(profiles),
        "n_with_phenotype": sum(1 for p in profiles if p.has_phenotype),
        "n_adhd_any": sum(1 for p in profiles if p.adhd_any),
        "n_adhd_confirmed": sum(1 for p in profiles if p.adhd_confirmed),
        "n_no_diagnosis_given": sum(1 for p in profiles if p.has_no_diagnosis_given),
        "adhd_certainty_counts": certainty_counts,
    }
