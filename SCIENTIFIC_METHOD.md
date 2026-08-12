# Scientific method

This document states exactly what NeuroTRIBE computes, what it claims, and —
just as importantly — what it refuses to claim.

> **Research Use Only.** Not a diagnostic or medical device.

---

## 1. The question

> Do participants with **confirmed ADHD** show systematically different
> stimulus-evoked cortical responses from an appropriate comparison cohort,
> **relative to the response predicted by TRIBE v2**?

### What this is not

This system is **not** `fMRI → ADHD diagnosis`, and **not** "TRIBE detects
ADHD". It never emits a per-person probability of ADHD. TRIBE v2 predicts an
*average-subject* cortical response; deviation from that reference is difference
from a normative model, which is not the same thing as pathology.

---

## 2. Design

| Element | Choice |
|---|---|
| Reference model | TRIBE v2, **pretrained only** — no fine-tuning, no retraining, no new architecture |
| Observed data | HBN naturalistic movie-fMRI |
| Surface space | `fsaverage5`, 20 484 vertices (10 242 per hemisphere) |
| Primary stimulus | *Despicable Me* (~10 min) preferred over *The Present* (~3:21) for timepoint count; policy is `auto` with a documented preference order |
| Primary contrast | **Confirmed ADHD** vs **No Diagnosis Given** |
| Exploratory contrast | Confirmed ADHD vs **non-ADHD comparison cohort** |

Certainty levels are **never pooled**. Combining Confirmed + Presumptive +
Rule-out would silently redefine the question.

---

## 3. Stimulus binding

HBN documents two movie intervals: *The Present* (00:00:00–00:03:21) and
*Despicable Me* (01:02:09–01:12:09).

A BOLD run is bound to a stimulus by **acquisition duration**, because duration
is a property of the acquisition, whereas the task label is site- and
release-dependent. A task-name hint contributes at most a tiebreak, and binding
requires both positive evidence *and* separation from the alternative. Runs that
match nothing stay `UNKNOWN`; the full evidence trail is stored per scan and is
inspectable at `/api/data/scans/{id}/evidence`.

A supplied clip is likewise matched to the catalog by duration within a
configured tolerance, must carry an audio stream (TRIBE v2 uses its auditory
pathway), and its first and last frames are extracted for visual verification.

---

## 4. Preprocessing

```
HBN volumetric BOLD → fMRIPrep → FreeSurfer surfaces → fsaverage5 GIFTI
```

Then, per run:

1. Drop fMRIPrep-flagged non-steady-state volumes.
2. Build the censor mask.
3. Confound regression.
4. Detrend + discrete-cosine high-pass.
5. Standardise.

**Denoising strategy is configuration, not a constant.** HBN itself notes there
is no single consensus approach to motion correction, so the applied strategy —
including the exact resolved confound column list — is recorded with every run
and embedded in the provenance manifest.

**Global signal regression is off by default.** It remains available, but
enabling it is a non-default scientific choice, is logged as a warning, and
appears in the report's limitations.

### Censoring

Every frame carries `usable = true/false` with a reason: `nonsteady_state`,
`high_fd`, `dvars_outlier`, `censor_pad`, `missing_data`. The DVARS threshold is
robust (median + k·MAD), so the outliers being detected cannot inflate the
threshold that detects them. Censored frames are excluded from every
correlation, residual and aggregate — they are **never interpolated over**.

### Defaced anatomy

HBN warns that defacing can degrade FreeSurfer. A FreeSurfer failure is retried
**once** with validated settings; if it fails again the participant is marked
`PREPROCESSING_FAILED`. Substituted or approximated surfaces are never silently
used in their place. A development-only volume projection exists, is always
flagged `is_approximate`, is excluded from final analysis, and is refused
outright in the `production` profile.

---

## 5. Alignment

This is where a plausible-looking but meaningless result is most easily created.

### Temporal

Inputs: stimulus time base, TRIBE's own segment timestamps, the acquisition TR,
volume count, removed initial volumes, and the censor mask.

- TRIBE v2 documents that its prediction timing **already incorporates a 5 s
  hemodynamic-lag offset**. Its timestamps are therefore used verbatim and **no
  second shift is applied**. (`additional_hrf_shift_sec` defaults to 0 and is
  logged loudly if ever set.)
- Predictions are interpolated onto the **acquisition** grid — the scanner's
  clock is authoritative, so no observed data is invented.
- **Extrapolation is refused.** Target samples outside TRIBE's support are
  marked non-finite and censored.
- Frame *i* of the retained series is timestamped `(i + n_dropped) · TR`,
  because dropping non-steady-state volumes removes time from the start of the
  run, not from the stimulus.
- A cross-correlation lag search runs as a **diagnostic only**. A residual lag
  beyond tolerance is surfaced as a warning; the pipeline never silently
  "corrects" timing.

### Spatial

Both sides live on `fsaverage5`, so no resampling occurs — but hemisphere
ordering and medial-wall handling must match exactly. `neurotribe/tribe/geometry.py`
inspects TRIBE's own source to establish the convention, checks vertex counts,
per-hemisphere symmetry of the masked region, timestamp monotonicity and the
non-finite fraction. Failure is `ANALYSIS INVALID`, not a warning.

---

## 6. Subject-level metrics

Both series are z-scored **over usable timepoints only** — raw BOLD amplitude
and arbitrary model output are not on a comparable scale.

| Metric | Definition |
|---|---|
| Temporal agreement | vertex-wise Pearson `r(predicted, observed)` |
| Standardized residual | `residual(t,v) = observed_z(t,v) − predicted_z(t,v)` |
| Mean absolute deviation | `MAD(v) = mean_t |residual(t,v)|` |
| Residual variance | `var_t(residual(t,v))` |

A constant vertex yields exactly zero, not amplified rounding noise (see
`neurotribe/numerics.py`), and a vertex whose correlation is undefined yields
NaN rather than an arbitrary finite number.

### Movie-moment analysis

A 10 s sliding window (1 s step) gives global deviation over time. Windows with
insufficient usable-frame coverage yield NaN rather than a value computed from a
handful of surviving frames. Peak windows are ranked with overlap suppression,
so the UI can answer *what was happening when the brain deviated?*

---

## 7. Aggregation

Vertex maps are aggregated to a cortical parcellation (Schaefer 200 / 7
networks) and to the canonical networks: Visual, Somatomotor, Dorsal Attention,
Salience/Ventral Attention, Limbic, Control, Default.

If no Schaefer `.annot` is available, a deterministic geometric parcellation is
used instead — flagged `is_approximate`, with a warning that ROI names are *not*
anatomical labels, propagated into the report's limitations.

**A single network score is never given a medical interpretation on its own.**

---

## 8. Group analysis

Primary outcomes are pre-specified: ROI and network agreement and deviation.

```
ROI_metric ~ ADHD + age + sex + site + mean_FD
```

- A raw `mean(ADHD) − mean(control)` is **never** reported as evidence about
  ADHD biology. HBN explicitly warns that scanner/site differences introduce
  technical variance and that motion matters especially in developmental and
  clinical imaging.
- Categorical covariates are dummy-coded; levels present in only one group are
  dropped **with a recorded note**, because they cannot be estimated.
- Missing covariate values are mean-imputed **with a missingness indicator**, so
  imputation can never masquerade as observation.
- FDR (Benjamini–Hochberg) is applied **within** each `(unit_type, metric)`
  family, not across everything at once.
- Every result carries effect size, 95% CI, per-group sample sizes, p and q.
  Significance markers never stand alone.

### Primary vs exploratory

`PRIMARY` is the pre-specified ROI/network contrast between Confirmed ADHD and
No Diagnosis Given. `EXPLORATORY` covers vertex-level maps, the non-ADHD
comparison cohort, alternative motion thresholds, comorbidity subsets and
matched subsets. Exploratory output is labelled everywhere it appears and never
modifies the primary result.

---

## 9. The sanity gate

Before any result may be declared:

- TRIBE vertex count == observed vertex count
- hemisphere ordering confirmed
- movie duration matches the documented interval
- TR plausible; volume count present
- temporal overlap valid; sufficient usable frames
- motion censoring applied
- no NaN explosion
- correlations within [−1, 1]
- group sizes sensible; p-values within [0, 1]
- no participant duplicated

Any failure yields **`ANALYSIS INVALID`**. The comparison is stored with
`valid = False` and an explicit reason, and is excluded from every group model —
it is never silently discarded, and never quietly included.

---

## 10. Limitations (always reported)

1. Research analysis, not diagnosis; no individual-level probability is produced.
2. TRIBE predicts an average-subject response; deviation ≠ pathology.
3. Residual motion effects cannot be fully excluded despite censoring and
   adjustment.
4. Multi-site, multi-scanner acquisition introduces technical variance; site is
   a covariate but not a complete remedy.
5. No consensus denoising strategy exists; results may depend on the recorded
   choice.
6. Observational design → associative, not causal.

Plus, when triggered: mock TRIBE backend, approximate parcellation, approximate
surfaces, underpowered groups, or global signal regression enabled.

---

## 11. Reproducibility

Every result carries a mandatory manifest:

```json
{
  "tribe_commit": "…", "tribe_model": "…", "tribe_model_revision": "…",
  "tribe_backend": "real|mock",
  "fmriprep_version": "…", "fmriprep_image": "…",
  "dataset_release": "…", "stimulus_sha256": "…",
  "cohort_hash": "…", "analysis_config_hash": "…",
  "denoise_strategy": "…", "global_signal_regression": false,
  "atlas": {…}, "surface_space": "fsaverage5", "hemi_order": ["L","R"],
  "model_formula": "…", "multiple_comparisons": "fdr_bh",
  "profile": "development|production",
  "timestamp": "…", "research_use_only": true
}
```

`cohort_hash` pins exactly which participants entered the analysis.
`analysis_config_hash` covers every scientific parameter, so a changed threshold
invalidates the cache and is visible in the output.
