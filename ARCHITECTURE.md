# Architecture

> **Research Use Only.** Not a diagnostic or medical device.

---

## Services

```
Browser
   │  http://localhost:4321
   ▼
web        Astro 5 (SSR) + React 19 islands + Three.js + Tailwind 4
   │  /api → proxied
   ▼
api        FastAPI + SQLAlchemy 2 + Pydantic 2            :8000
   ├─► postgres   state, provenance, audit                :5432
   ├─► redis      Celery broker + result backend          :6379
   └─► ./data     scientific artefacts on disk
                    │
worker     Celery — acquisition, fMRIPrep, TRIBE, analysis, reporting
beat       Celery beat — Autopilot tick + intake watcher
```

`worker` mounts the host Docker socket so it can launch the fMRIPrep container
on the host daemon rather than nesting containers.

---

## Package layout

```
neurotribe/
├── config.py         layered configuration + analysis_config_hash
├── numerics.py       scale-aware flatness / z-scoring (float32-safe)
├── hashing.py        content digests and cache keys
├── system.py         hardware & toolchain probe
├── logging_setup.py  structured logging
│
├── database/         base · enums · models · repository
├── acquisition/      discover · hbn_metadata · bids · phenotype · stimulus · fetch
├── cohort/           diagnoses · eligibility · matching
├── preprocessing/    fmriprep · confounds · censoring · surfaces · pipeline · qc
├── tribe/            model · mock · geometry · inference
├── alignment/        temporal · spatial · validate
├── analysis/         residuals · roi · subject · statistics · group
├── reporting/        report
└── jobs/             stages · autopilot · celery_app · tasks

apps/api/             FastAPI app + 12 routers
apps/web/             Astro pages + React islands
```

Dependency direction is strictly downward: `analysis` may import `alignment`,
which may import `preprocessing`; nothing imports `jobs` or `apps`.

---

## Data model

17 tables. The rule that shapes all of them: **large scientific arrays never
enter PostgreSQL.** Rows hold paths, metadata, hashes and provenance; NIfTI,
GIFTI and `.npy` stay on disk.

| Group | Tables |
|---|---|
| Participants | `subjects`, `diagnoses` |
| Imaging | `scans`, `scan_qc` |
| Inputs | `stimuli`, `data_assets` |
| Cohorts | `cohorts`, `cohort_members` |
| Processing | `preprocessing_runs`, `tribe_runs` |
| Results | `subject_comparisons`, `roi_metrics`, `network_metrics` |
| Group | `group_analysis_runs`, `group_results` |
| Orchestration | `pipeline_stages`, `jobs`, `artifacts`, `blockers` |
| Integrity | `audit_events`, `system_probes` |

`scan_qc.extra_iqms` and `subjects.metadata_json` preserve columns the parsers
do not explicitly promote, so a future HBN release never loses information.

---

## Autopilot

A persistent state machine in `pipeline_stages`, driven by a Celery beat tick.

**19 stages**, each `PENDING` · `RUNNING` · `DONE` · `WAITING_EXTERNAL` ·
`FAILED_RETRYABLE` · `FAILED_FINAL` · `SKIPPED` · `BLOCKED` · `PARTIAL`.

```
system_probe
  └─ discover_assets
       ├─ ingest_metadata ──┐
       ├─ ingest_mriqc ─────┤
       ├─ index_bids ───────┴─ identify_movie_scans ─ fetch_imaging ─┐
       ├─ phenotype_intake ───────────────────────────┐              │
       └─ stimulus_intake ──────────────┐             │              │
                                        │             │              │
tribe_install ─ tribe_smoke_test ───────┴─ tribe_inference           │
preprocessing_preflight ─ surface_geometry_check                     │
                                        build_cohort ────────────────┴─ preprocess_cohort
                                                                          └─ subject_analysis
                                                                               └─ group_analysis
                                                                                    └─ generate_report
```

Each tick:

1. Recompute which stages have satisfied dependencies.
2. Run the first ready stage; **recompute again** (completing one stage often
   unblocks the next, so a single tick walks several steps forward).
3. Repeat up to `max_stages`.

Design guarantees:

- **A missing external dependency never crashes the run.** The stage enters
  `WAITING_EXTERNAL`; only its dependants are blocked, and every independent
  branch keeps building.
- **Every handler is exception-contained.** A raised handler is logged, audited
  with a traceback, and scheduled for exponential-backoff retry — the Autopilot
  itself cannot die.
- **Resumable.** State lives in PostgreSQL, so a reboot resumes rather than
  restarting.
- **Idempotent.** Re-running a `DONE` stage is a no-op; gated stages re-scan on
  every tick so a newly dropped file is noticed within one interval.

---

## Caching

Every expensive computation keys on its inputs **and** its configuration, so a
changed parameter can never reuse a stale artefact.

| Artefact | Cache key |
|---|---|
| TRIBE prediction | stimulus SHA-256 + model id/revision + TRIBE commit + backend + surface config |
| fMRIPrep run | subject + scan entities + BOLD path + image + version + output spaces + extra args |
| Prepared time series | run id + surface paths + confounds path + denoise + motion + surface config |
| Subject comparison | subject + run + tribe run + denoised path + analysis + alignment + surface config |

TRIBE inference runs **once per stimulus**, not once per subject — every
participant watched the same clip, so per-subject inference would be pure waste.

---

## API surface

25 endpoints across 12 routers. Every response carries
`X-Research-Use-Only: true`.

| Router | Purpose |
|---|---|
| `dashboard` | Full home state in one round trip; tick and stage retry |
| `system` | Hardware probe, resolved scientific config, TRIBE and preprocessing readiness |
| `data` | Source table, asset registry, scans, movie-classification evidence |
| `stimulus` | Catalog, validation status, frame thumbnails, range-enabled media streaming |
| `cohort` | Composition, balance diagnostics, exclusion accounting, matching |
| `subjects` | Detail, vertex maps (JSON or float32 binary), timelines, per-frame maps |
| `groups` | Runs, results, ROI detail, cortical **effect-size** map |
| `qc` | Per-participant QC table with filters and the active policy |
| `jobs` | Progress, resources, per-job logs |
| `logs` | INFO/WARNING/ERROR stream, download, append-only audit trail |
| `reports` | Artefact listing, download, provenance manifest |
| `surface` | `fsaverage5` mesh as packed binary buffers + parcellation |

### Binary transport

Per-vertex maps are served as raw little-endian float32 (`?format=binary`) —
roughly 8× smaller than JSON for 20 484 values and parsed directly into a
`Float32Array`. The mesh ships as three buffers (positions, normals, indices);
`api_smoke.py` asserts their exact byte counts against the manifest.

---

## Frontend

Astro SSR shells with React islands, so each page arrives populated and only the
interactive parts hydrate.

| Island | Role |
|---|---|
| `CorticalViewer` | Three.js `fsaverage5` renderer — orbit, zoom, vertex picking, six anatomical presets, L/R/both, three colour ramps |
| `PipelineBoard` | Live stage board, polls the dashboard, manual tick and per-stage retry |
| `BlockersPanel` | The external gates, explained with required action and reference |
| `SubjectExplorer` | Roster, maps, deviation timeline, peak moments, synchronised movie player |
| `GroupResults` | Tier switch, effect map, results table with CI and q |
| `QCTable`, `JobsTable`, `LogStream` | Filterable tables and the live log stream |

Viewer correctness details that matter scientifically:

- Vertex ordering follows the server manifest's `hemi_order` — the same
  convention verified against TRIBE. The client never assumes L-then-R.
- **NaN renders as inert grey**, outside every colour ramp, so "no data" can
  never be misread as "strong negative effect".
- A map whose length disagrees with the manifest is **refused**, with the
  mismatch stated, rather than rendered against the wrong geometry.
- Colour domains default to a robust 2nd–98th percentile, symmetric for
  diverging maps so sign stays meaningful.

Theme is token-driven with a pre-paint inline script (no flash), and
`prefers-reduced-motion` disables all animation.

---

## Failure handling

| Failure | Response |
|---|---|
| API unreachable | Pages render from a fallback and say so; they never blank |
| Stage handler raises | Logged, audited with traceback, exponential-backoff retry, capped by `max_attempts` |
| fMRIPrep FreeSurfer failure | One validated retry, then `PREPROCESSING_FAILED` — never substituted surfaces |
| Geometry mismatch | `GeometryError`, analysis aborted |
| Sanity-gate failure | `ANALYSIS INVALID`, stored with reason, excluded from group models |
| Insufficient disk | Acquisition pauses with required-vs-available reported |
| One subject fails | Contained; the remaining cohort continues, failure recorded |

---

## Configuration

`config/default.yaml` → `config/<profile>.yaml` → `NEUROTRIBE__SECTION__KEY`.

Ten sections are *scientific* (`stimulus`, `bids`, `phenotype`, `cohort`, `qc`,
`preprocessing`, `surface`, `tribe`, `alignment`, `analysis`) and are hashed into
`analysis_config_hash`. Cosmetic sections (logging, paths) are excluded, so
changing a log level does not invalidate an otherwise identical analysis.
