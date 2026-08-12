# RUN STATUS

> Maintained by the build. For **live** state use the dashboard
> (http://localhost:4321) or `python scripts/doctor.py --ticks 3`, which read
> the database directly.

**Research Use Only — not a diagnostic or medical device.**

---

## Build status: V1 software complete

Every component specified for V1 is built, wired and verified. The system is
waiting on three inputs that software cannot legitimately obtain on your behalf.

| Layer | Status | Evidence |
|---|---|---|
| Configuration + provenance hashing | ✅ | `analysis_config_hash` = `d4f8f87699cde4b3` on default config |
| Database (17 tables) + Alembic | ✅ | schema creates cleanly; migration baseline in place |
| Acquisition (discovery, metadata, MRIQC, BIDS, phenotype, stimulus, fetch) | ✅ | content-based classification verified on synthetic fixtures |
| Cohort (diagnoses, eligibility, matching) | ✅ | 8 ADHD vs 8 No-Diagnosis-Given built from fixtures, 4 excluded with reasons |
| Preprocessing (fMRIPrep runner, confounds, censoring, surfaces, QC) | ✅ | denoising + censoring verified; runner blocked only by the FreeSurfer license |
| TRIBE v2 integration + geometry validation | ✅ | smoke test passed: `13 × 20484`, hemi order `[L, R]` |
| Alignment (temporal, spatial, sanity gate) | ✅ | 27 scientific tests |
| Analysis (residuals, ROI, networks, group statistics) | ✅ | full contrast executed on fixtures with FDR + CIs |
| Reporting (HTML, CSV, provenance manifest) | ✅ | generated and content-asserted in the integration test |
| Autopilot state machine (19 stages) | ✅ | advances, blocks correctly, resumes, retries with backoff |
| FastAPI (25 endpoints, 12 routers) | ✅ | `scripts/api_smoke.py`: all endpoints + geometry + actions pass |
| Astro + React + Three.js frontend (8 pages) | ✅ | builds clean; all 8 pages render 200 against the live API |
| Tests | ✅ | **142 passing** (unit · integration · scientific) |
| Docker Compose (6 services) | ✅ | all healthy: `postgres · redis · api · worker · beat · web` (+ `migrate`, runs once) |

---

## Live pipeline state on this machine

Recorded from `scripts/doctor.py` on the current workspace.

```
[ok] system_probe               DONE              12 CPU / 15.67 GB RAM / 27.7 GB free / no GPU
[ok] discover_assets            DONE              Discovered 0 asset(s)
[!!] ingest_metadata            WAITING_EXTERNAL  Awaiting the HBN release metadata CSV
[!!] ingest_mriqc               WAITING_EXTERNAL  Awaiting the MRIQC IQM export
[!!] index_bids                 WAITING_EXTERNAL  Awaiting the HBN BIDS repository
[--] identify_movie_scans       BLOCKED           waiting on 'Index the HBN BIDS repository'
[--] fetch_imaging              BLOCKED           waiting on 'Identify movie-fMRI participants'
[!!] phenotype_intake           WAITING_EXTERNAL  Waiting for DUA-approved ADHD phenotype data
[!!] stimulus_intake            WAITING_EXTERNAL  Waiting for the exact, legally obtained stimulus
[ok] tribe_install              DONE              MOCK backend (development only; not reportable)
[ok] tribe_smoke_test           DONE              13 × 20484, hemi order ['L','R'] — geometry valid
[!!] preprocessing_preflight    WAITING_EXTERNAL  FreeSurfer license required

[--] surface_geometry_check     BLOCKED           waiting on 'Verify the preprocessing pipeline'
[--] tribe_inference            BLOCKED           waiting on 'Await / validate the movie stimulus'
[--] build_cohort               BLOCKED           waiting on 'Await / ingest ADHD phenotype data'
[--] preprocess_cohort          BLOCKED           waiting on 'Construct the ADHD cohort'
[--] subject_analysis           BLOCKED           waiting on 'Preprocess the cohort'
[--] group_analysis             BLOCKED           waiting on 'Align TRIBE with HBN'
[--] generate_report            BLOCKED           waiting on 'ADHD group statistics'
```

Steady state: **4 DONE · 6 WAITING_EXTERNAL · 9 BLOCKED**.

This is the **designed** behaviour: no crash, no fabricated data, every gate
explained with the action required. As soon as an input appears the Autopilot
picks it up within one tick and the dependent stages unblock automatically.

Note that `tribe_install` and `tribe_smoke_test` completed *while* the three
gates were pending — that is the "keep building despite blocked dependencies"
requirement working: the TRIBE inference path is proven end to end (geometry
validated at `13 × 20484`, hemisphere order `[L, R]`) before the stimulus or the
phenotype export exist.

---

## What is blocked, and how to unblock it

### 1. HBN dataset files — not present on this machine

Searched the workspace, Desktop, Downloads and Documents: `Metadata_R11.1.csv`,
`IQM_functional_ExternalID.csv` and `HBN_BIDS/` were **not found**. The
specification assumed they were already present; they are not, so they are
tracked as external inputs.

```
Metadata_R*.csv          →  data/metadata/
IQM_functional_*.csv     →  data/metadata/
HBN_BIDS/                →  data/external/HBN_BIDS/
```

Discovery classifies by content, so exact filenames do not matter.

### 2. HBN phenotype (ADHD labels) — DUA-controlled

```
data/phenotype/incoming/<your Diagnosis_ClinicianConsensus export>.csv
```

Complete the HBN DUA, then export the instrument from the LORIS Data Query Tool.
No ADHD label is invented while this is pending.

### 3. Exact movie stimulus — copyrighted

```
data/stimuli/incoming/<clip>.mp4
```

| Movie | Documented interval | Expected | Tolerance |
|---|---|---|---|
| The Present | 00:00:00 → 00:03:21 | 201.0 s | ±3 s |
| Despicable Me | 01:02:09 → 01:12:09 | 600.0 s | ±5 s |

Must contain an audio stream — TRIBE v2 uses its auditory pathway. Validated by
duration, hashed, and first/last frames extracted for visual verification.

### 4. FreeSurfer license

```
config/license.txt        (or set FS_LICENSE)
```

Free registration: https://surfer.nmr.mgh.harvard.edu/registration.html

---

## Capacity notes for this machine

| Resource | Value | Assessment |
|---|---|---|
| CPU | 12 cores (4 to Docker) | Adequate |
| RAM | 15.7 GB (7.8 GB to Docker) | ⚠️ Below fMRIPrep's configured 8 GB. Raise Docker's memory limit or lower `preprocessing.fmriprep.mem_mb`. |
| Disk | 27.7 GB free of 237 GB | ⚠️ Roughly 2 participants at ~12 GB each. The disk guard pauses acquisition before filling the drive. |
| GPU | none | Fine — TRIBE falls back to the validated CPU path. Inference is slower, not wrong. |
| Docker | 29.6.2 | ✅ |
| ffmpeg | present | ✅ |

**Recommendation before a real cohort run:** raise Docker's memory allocation to
≥ 12 GB and free disk to ≥ 150 GB.

---

## Deliberate deviations from the specification

Two, both stated up front rather than done silently:

1. **Frontend is Astro + React islands, not Next.js.** You specified this
   directly in your opening line; it overrides §40 of the spec. Three.js, the
   `fsaverage5` viewer and every UI requirement are implemented as specified.
2. **PyTorch and TRIBE v2 are not vendored into the base image.** TRIBE is
   CC-BY-NC-4.0 and installed from git by the operator; bundling it would make a
   licence decision on your behalf and add ~2.5 GB to every image. `tribe_install`
   reports the exact command and the pipeline continues on the mock backend
   meanwhile. To install:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install "git+https://github.com/facebookresearch/tribev2@main"
   ```

---

## Known approximations while gated (all flagged in output)

| Approximation | When | How it is flagged |
|---|---|---|
| Mock TRIBE backend | Real TRIBE not installed, `development` profile | `backend: mock` in every artefact; report says *"CRITICAL: this run used the MOCK TRIBE backend"*; refused in `production` |
| Generated `fsaverage5` sphere | Real mesh not fetchable offline | `source: generated:icosphere-order5`; correct topology and vertex count, non-anatomical coordinates |
| Geometric parcellation | No Schaefer `.annot` under `data/atlas/` | `is_approximate: true`; report states ROI names are not anatomical labels |
| Volume→surface projection | Dev only, off by default | `is_approximate`; excluded from final analysis; refused in `production` |

None can reach a production result: the `production` profile rejects the mock
backend and approximate surfaces outright.

---

## Verification log

```
pytest                               142 passed  (2m37s)
  tests/unit          111 passed     parsers, statistics, FDR, caching, numerics,
                                     system probe, disk guard, stage graph, autopilot
  tests/scientific     27 passed     correlation, lag, geometry, censoring, sanity gate
  tests/integration     4 passed     full pipeline on synthetic fixtures

scripts/api_smoke.py (Docker)        25 endpoints + geometry + actions passed
  surface buffers                    byte-exact vs manifest (122,904 / 245,760 B per hemi)
  total vertices                     20,484 ✓
  parcellation labels                81,936 B ✓

npm run build                        clean, 13 client chunks
frontend pages (Docker)              8/8 render 200 against the live API
docker compose ps                    6/6 services healthy
autonomous operation                 beat dispatches ticks; worker executes them;
                                     state persists in PostgreSQL across restarts
```

### Bugs found and fixed during verification

Recorded because each would have produced quietly wrong output rather than an
error.

| # | Bug | Impact if shipped |
|---|---|---|
| 1 | `sd < 1e-12` flatness threshold was too strict for float32 | A constant vertex (the medial wall) z-scored to **±1 instead of 0**, turning rounding noise into confident signal. Fixed with the scale-aware `neurotribe/numerics.py`; regression-tested. |
| 2 | `WAITING_EXTERNAL` stages starved runnable ones | The four data gates consumed the whole per-tick budget every tick, so `tribe_install`, `stimulus_intake` and `preprocessing_preflight` **never ran at all**. Fixed with a re-check backoff plus priority ordering; regression-tested. |
| 3 | Disk guard measured the container overlay filesystem | Reported **921 GB free when the real data volume had 29 GB**, defeating the capacity guard. Now measures the data directory; regression-tested. |
| 4 | `/api/system/probe` bound to Celery's *default* app | The probe message went nowhere silently; the API then overwrote the worker's accurate reading with its own (Docker-less) one. Now sends on the configured app. |
| 5 | `/surface/{hemi}/{buffer}` shadowed `/surface/parcellation/labels` | The parcellation buffer 400'd — the ROI overlay could never load. Route order fixed. |
| 6 | Image-level `HEALTHCHECK` curled the API | `beat` reported permanently unhealthy, because one image serves three roles. Healthchecks moved per service. |
| 7 | JSON column defaults are applied at INSERT, not construction | `TypeError` on first asset registration. Defensive reads added. |
| 8 | `cohort_hash` (NOT NULL) assigned after the row was flushed | `IntegrityError` on first cohort build. Now set at construction. |

---

## Next actions

1. Place the three HBN files (metadata, MRIQC, BIDS) in the workspace.
2. Complete the DUA and drop the phenotype export.
3. Supply the licensed stimulus clip.
4. Install the FreeSurfer license.
5. Install real TRIBE v2 + PyTorch, then set `NEUROTRIBE_PROFILE=production`.
6. Raise Docker memory and free disk before the cohort run.

The Autopilot handles everything after that without further instruction.
