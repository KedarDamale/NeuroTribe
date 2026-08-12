"""Pipeline stage definitions - the Autopilot's dependency graph.

Mirrors the PHASE 0 .. PHASE 23 execution sequence from the specification. A
stage blocked on an external dependency (phenotype access, licensed stimulus,
FreeSurfer license) never crashes the application: it enters
``WAITING_EXTERNAL`` and only its *dependents* are blocked, so every independent
branch keeps building.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StageSpec:
    key: str
    label: str
    phase: int
    group: str
    depends_on: tuple[str, ...] = ()
    max_attempts: int = 3
    description: str = ""


STAGES: tuple[StageSpec, ...] = (
    # ---------------- Phase 1-2: environment and discovery ----------------
    StageSpec("system_probe", "Check OS / hardware / software", 1, "Environment",
              description="CPU, RAM, GPU, VRAM, disk, CUDA, Docker, WSL."),
    StageSpec("discover_assets", "Detect existing HBN files", 2, "Data",
              depends_on=("system_probe",),
              description="Inspect the workspace before downloading anything."),

    # ---------------- Phase 3-5: metadata and imaging index ---------------
    StageSpec("ingest_metadata", "Validate HBN release metadata", 3, "Data",
              depends_on=("discover_assets",)),
    StageSpec("ingest_mriqc", "Validate MRIQC image-quality metrics", 3, "Data",
              depends_on=("discover_assets",)),
    StageSpec("index_bids", "Index the HBN BIDS repository", 4, "Data",
              depends_on=("discover_assets",)),
    StageSpec("identify_movie_scans", "Identify movie-fMRI participants", 5, "Data",
              depends_on=("index_bids", "ingest_mriqc"),
              description="Bind BOLD runs to documented HBN movie intervals by duration."),

    # ---------------- Phase 6-8: gated inputs -----------------------------
    StageSpec("fetch_imaging", "Retrieve required MRI files only", 6, "Data",
              depends_on=("identify_movie_scans",),
              description="Selective retrieval driven by the cohort target list."),
    StageSpec("phenotype_intake", "Await / ingest ADHD phenotype data", 7, "Phenotype",
              depends_on=("discover_assets",), max_attempts=1000,
              description="DUA-controlled. Watches data/phenotype/incoming."),
    StageSpec("stimulus_intake", "Await / validate the exact movie stimulus", 8, "Stimulus",
              depends_on=("discover_assets",), max_attempts=1000,
              description="Copyrighted. Watches data/stimuli/incoming. Never downloads."),

    # ---------------- Phase 9-11: TRIBE -----------------------------------
    StageSpec("tribe_install", "Install / verify TRIBE v2", 9, "Model",
              depends_on=("system_probe",)),
    StageSpec("tribe_smoke_test", "TRIBE smoke test on a synthetic clip", 11, "Model",
              depends_on=("tribe_install",),
              description="Proves the inference path and validates output geometry."),

    # ---------------- Phase 12-14: preprocessing --------------------------
    StageSpec("preprocessing_preflight", "Verify the preprocessing pipeline", 12, "Processing",
              depends_on=("system_probe",),
              description="Docker, image, FreeSurfer license."),
    StageSpec("surface_geometry_check", "Validate fsaverage5 geometry", 14, "Processing",
              depends_on=("preprocessing_preflight",)),

    # ---------------- Phase 15-16: real inputs arrive ---------------------
    StageSpec("tribe_inference", "Run TRIBE on the real stimulus", 15, "Model",
              depends_on=("tribe_smoke_test", "stimulus_intake")),
    StageSpec("build_cohort", "Construct the ADHD cohort", 16, "Cohort",
              depends_on=("phenotype_intake", "identify_movie_scans")),

    # ---------------- Phase 17-21: the experiment -------------------------
    StageSpec("preprocess_cohort", "Preprocess the cohort", 17, "Processing",
              depends_on=("build_cohort", "preprocessing_preflight", "fetch_imaging"),
              max_attempts=5),
    StageSpec("subject_analysis", "Align TRIBE with HBN and compute deviations", 19, "Analysis",
              depends_on=("preprocess_cohort", "tribe_inference", "surface_geometry_check")),
    StageSpec("group_analysis", "ADHD group statistics", 21, "Analysis",
              depends_on=("subject_analysis",)),

    # ---------------- Phase 22-23: outputs --------------------------------
    StageSpec("generate_report", "Generate the research report", 23, "Reporting",
              depends_on=("group_analysis",)),
)

STAGE_BY_KEY: dict[str, StageSpec] = {spec.key: spec for spec in STAGES}

# Stages that legitimately wait on human/institutional action rather than failing.
EXTERNALLY_GATED = frozenset({"phenotype_intake", "stimulus_intake", "fetch_imaging"})


def order() -> list[StageSpec]:
    """Topologically ordered stage list (stable, deterministic)."""
    resolved: list[StageSpec] = []
    seen: set[str] = set()

    def visit(spec: StageSpec, trail: tuple[str, ...] = ()) -> None:
        if spec.key in seen:
            return
        if spec.key in trail:
            raise ValueError(f"Cycle in stage graph: {' -> '.join((*trail, spec.key))}")
        for dependency in spec.depends_on:
            parent = STAGE_BY_KEY.get(dependency)
            if parent is None:
                raise KeyError(f"Stage '{spec.key}' depends on unknown stage '{dependency}'")
            visit(parent, (*trail, spec.key))
        seen.add(spec.key)
        resolved.append(spec)

    for spec in sorted(STAGES, key=lambda s: (s.phase, s.key)):
        visit(spec)
    return resolved


@dataclass
class StageGroup:
    name: str
    stages: list[str] = field(default_factory=list)


def groups() -> list[StageGroup]:
    """Stage groups as rendered on the dashboard pipeline graphic."""
    ordered: dict[str, StageGroup] = {}
    for spec in order():
        group = ordered.setdefault(spec.group, StageGroup(spec.group))
        group.stages.append(spec.key)
    return list(ordered.values())


def descendants(key: str) -> set[str]:
    """Every stage transitively blocked when ``key`` cannot complete."""
    blocked: set[str] = set()
    frontier = {key}
    while frontier:
        current = frontier.pop()
        for spec in STAGES:
            if current in spec.depends_on and spec.key not in blocked:
                blocked.add(spec.key)
                frontier.add(spec.key)
    return blocked
