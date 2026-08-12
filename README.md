# NeuroTRIBE-HBN

**Stimulus-conditioned normative cortical response analysis for ADHD using
TRIBE v2 and Healthy Brain Network movie-fMRI.**

> ### ⚠️ Research Use Only — not a diagnostic or medical device.
> This system does **not** diagnose ADHD, does **not** produce a per-person
> probability of ADHD, and deviation from a normative model is **not**
> pathology. TRIBE v2 is used as a *pretrained reference encoder* and is never
> retrained here.
>
> V1 is **research / non-commercial**: the official TRIBE v2 repository is
> CC-BY-NC-4.0, and HBN flags a subset of participants as restricted from
> commercial use.

---

## The question this system asks

> Do participants with **confirmed ADHD** show systematically different
> stimulus-evoked cortical responses from an appropriate comparison cohort,
> **relative to the response predicted by TRIBE v2**?

```
                    Exact HBN movie
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
         TRIBE v2                HBN participant
             │                     movie fMRI
             ▼                         │
   Expected average                    ▼
   cortical response  ──►  temporal + spatial alignment
                                       │
                                       ▼
                              deviation analysis
                                       │
                    ┌──────────────────┼──────────────────┐
                    ▼                  ▼                  ▼
                 vertex               ROI              network
                    └──────────────────┼──────────────────┘
                                       ▼
                    Confirmed ADHD  vs  comparison cohort
                                       ▼
                       covariate-adjusted statistics
                                       ▼
                    interactive dashboard + research report
```

---

## Quick start

```bash
cp .env.example .env
docker compose up -d --build
```

Then open **http://localhost:4321**.

Nothing else is required. The Autopilot inspects the machine, finds whatever
data already exists, and reports precisely what it still needs.

Terminal equivalent of the dashboard:

```bash
python scripts/bootstrap.py --tick     # prepare + advance once
python scripts/doctor.py --ticks 3     # full diagnostics
python scripts/api_smoke.py            # exercise every API endpoint
```

---

## The three external gates

Software cannot legitimately resolve these. They are surfaced honestly in the
UI as `WAITING_EXTERNAL`, they never crash the run, and **every other part of
the system keeps building and testing while they are pending**.

| Gate | Why it is blocked | What you do |
|---|---|---|
| **HBN phenotype (ADHD labels)** | Full HBN phenotypic data are DUA-controlled and exported via LORIS. NeuroTRIBE never authenticates to LORIS and never bypasses access control. | Complete the HBN DUA, export `Diagnosis_ClinicianConsensus` as CSV from the LORIS Data Query Tool into `data/phenotype/incoming/`. |
| **Exact movie stimulus** | The clip shown during scanning is copyrighted. NeuroTRIBE never downloads video, never touches torrent/piracy sites, and never scrapes unverified uploads. | Obtain the clip legally (HBN documents the intervals and asks researchers to contact CMI for exact-clip information) and drop it in `data/stimuli/incoming/`. |
| **FreeSurfer license** | fMRIPrep's surface reconstruction requires it. A license is never fabricated. | Free registration at surfer.nmr.mgh.harvard.edu, then save to `config/license.txt` (or set `FS_LICENSE`). |

A dropped file is auto-detected within one Autopilot tick: validated, hashed,
matched against the documented HBN interval by duration, and registered.

### Data you supply

| Input | Where |
|---|---|
| `Metadata_R*.csv` (HBN release metadata) | anywhere under the project root; `data/metadata/` is conventional |
| `IQM_functional_*.csv` (MRIQC) | same |
| HBN BIDS tree | `data/external/HBN_BIDS/` (a DataLad clone enables selective fetch) |

Discovery classifies files by **content**, not filename, so unusual names still
bind correctly.

---

## What makes this trustworthy

The failure modes that produce *plausible but wrong* neuroscience are blocked
explicitly rather than hoped away:

- **Movie binding is evidence-based.** A BOLD run is bound to a stimulus by
  *acquisition duration* against the documented HBN intervals. Task labels are
  only a tiebreaker, so `task=movieDM` on a 201-second run is correctly
  classified as *The Present*, not Despicable Me.
- **Hemisphere ordering is verified, never assumed.** TRIBE's own source is
  inspected to establish `[L, R]`; a mismatch is adopted and recorded rather
  than silently accepted. A left/right swap would invert every spatial
  conclusion while leaving summary statistics looking perfect.
- **No double hemodynamic shift.** TRIBE v2 states its prediction timing already
  includes a 5 s hemodynamic offset, so its timestamps are used verbatim and no
  second shift is applied. A residual lag is *reported*, never silently absorbed.
- **No extrapolation.** Predictions are interpolated onto the scanner's clock
  and refused outside TRIBE's support.
- **Censoring is real.** High-FD, DVARS-outlier, non-steady-state and padded
  frames are excluded from every correlation, residual and aggregate — never
  interpolated over.
- **Scale-aware flatness.** Constant float32 vertices (the medial wall) become
  exact zeros instead of being amplified into a confident ±1 by rounding noise.
- **Nothing is dropped silently.** Every excluded participant carries a
  machine-readable `ExclusionReason`.
- **The sanity gate is a gate.** Vertex-count mismatch, duplicated participants,
  out-of-range correlations, wrong stimulus duration, uncensored comparison or a
  NaN explosion produce `ANALYSIS INVALID` — not a warning.
- **No raw group-mean claims.** Every contrast is adjusted for age, sex,
  acquisition site and mean head motion, FDR-corrected, and reported with effect
  sizes, confidence intervals and sample sizes.
- **The comparison group is never called "healthy controls."** In the
  exploratory tier it may include participants with other diagnoses, so it is
  the *non-ADHD comparison cohort*.

## Vocabulary the UI enforces

| Never | Always |
|---|---|
| "healthy controls" | *No Diagnosis Given* / *non-ADHD comparison cohort* |
| "Probability of ADHD: 87%" | group-level, covariate-adjusted effect sizes |
| "TRIBE detects ADHD" | deviation from a normative encoding model |
| "patients" | *participants* |

---

## Architecture

```
Browser ──► Astro + React islands + Three.js  (apps/web, :4321)
                     │
                     ▼
              FastAPI  (apps/api, :8000)
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   PostgreSQL      Redis      data/ on disk
                     │
                     ▼
             Celery worker + beat
                     │
   ┌────────┬────────┼────────┬────────────┐
   ▼        ▼        ▼        ▼            ▼
acquisition fMRIPrep TRIBE  analysis   reporting
```

Large scientific arrays (NIfTI, GIFTI, `.npy`) **never** enter PostgreSQL — the
database holds paths, metadata, hashes and provenance only.

See [ARCHITECTURE.md](ARCHITECTURE.md), [SCIENTIFIC_METHOD.md](SCIENTIFIC_METHOD.md)
and [DATA_POLICY.md](DATA_POLICY.md).

---

## Autopilot

19 stages across the phases of the specification. Each is `PENDING`, `RUNNING`,
`DONE`, `WAITING_EXTERNAL`, `FAILED_RETRYABLE`, `FAILED_FINAL`, `SKIPPED`,
`BLOCKED` or `PARTIAL`. State lives in PostgreSQL, so a reboot **resumes**
rather than restarting, and every expensive step is cached on its inputs *and*
its configuration.

TRIBE inference runs **once per stimulus**, not once per subject — all
participants watched the same clip.

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest                       # 122 tests
pytest tests/scientific -q   # the correctness battery
```

The scientific suite proves the analysis behaves correctly: identical signals
correlate at 1, random signals at ~0, a deliberate temporal shift is detected, a
reversed hemisphere is caught, censored frames provably cannot influence a
metric, and each sanity-gate failure mode is exercised.

`tests/integration/test_full_pipeline.py` runs the **entire** chain on synthetic
fixtures — acquisition → cohort → TRIBE → alignment → deviation → ROI/network →
covariate-adjusted group statistics → report — so the system is verifiable
before any access-controlled input exists.

Synthetic data lives only under `tests/fixtures/`, is stamped
`profile: development` + `backend: mock`, and can never be reported as a result.

---

## Configuration

Layered: `config/default.yaml` → `config/<profile>.yaml` →
`NEUROTRIBE__SECTION__KEY` environment overrides.

Every scientific parameter lives in configuration and is hashed into
`analysis_config_hash`, which appears in every provenance manifest. Two runs
with different parameters can never be confused.

| Profile | Behaviour |
|---|---|
| `development` | Mock TRIBE fallback allowed; approximate surfaces allowed (always flagged). Never reportable. |
| `production` | Real TRIBE required; approximations refused; stricter disk floor. |

---

## Definition of V1 complete

V1 is complete when the exact HBN movie → TRIBE v2 → `fsaverage5` prediction is
aligned with HBN movie BOLD → fMRIPrep → `fsaverage5` observation; deviations
are computed at vertex, ROI and network level; Confirmed ADHD is contrasted
against No Diagnosis Given under a covariate-adjusted model; and the whole
pipeline is automatic, cached, resumable, tested, auditable and reproducible.

Everything above is built and verified today. The remaining work is supplying
the three gated inputs.

---

## Licence and citation

Research / non-commercial (V1). If you publish, cite TRIBE v2, fMRIPrep,
FreeSurfer, the Schaefer parcellation and the Healthy Brain Network, and include
the `analysis_config_hash`, `cohort_hash` and TRIBE commit from the provenance
manifest.
