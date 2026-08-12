# Data policy

> **Research Use Only.** Not a diagnostic or medical device.

This document states what NeuroTRIBE will and will not do with data. These are
implemented constraints, not aspirations — each links to the code that enforces
it.

---

## 1. Hard prohibitions

The system **never**:

| Prohibition | Enforcement |
|---|---|
| Bypasses HBN access controls | No LORIS client exists. `acquisition/phenotype.py` only reads files an authorised operator places in `data/phenotype/incoming/`. |
| Pirates the movie stimulus | No downloader exists for video. `acquisition/stimulus.py` only *validates* what is supplied. No torrent, piracy-site or arbitrary-upload code path exists anywhere in the repository. |
| Fabricates a FreeSurfer license | `preprocessing/fmriprep.py::detect_freesurfer_license` searches known locations and raises a blocker. It cannot synthesise one. |
| Invents ADHD labels | While phenotype data are absent, no diagnosis row is created. The cohort builder refuses to run and reports why. |
| Fabricates scientific data | The mock TRIBE backend exists only for fixtures and UI development, is stamped `backend: mock` in every artefact, and is refused by the `production` profile. |
| Silently discards participants | Every exclusion carries a machine-readable `ExclusionReason` plus a human-readable detail. |
| Silently changes scientific parameters | Every parameter lives in configuration and is hashed into `analysis_config_hash`. |
| Downloads every HBN scan | Retrieval is driven by the cohort target list and pulls only T1w, movie BOLD, sidecars and fieldmaps. |

---

## 2. Protected data classes

### DUA-controlled phenotype

- Lives under `data/phenotype/` — **git-ignored**.
- `DataAsset.protected = True`.
- The API withholds row-level phenotype content: `/api/data/assets/{id}` strips
  the buffered records and substitutes a note.
- Never included in any exported artefact beyond aggregate group statistics.

### Copyrighted stimulus

- Lives under `data/stimuli/` — **git-ignored**.
- Streamed only to the local research UI for scene-synchronised review; never
  re-encoded, re-hosted or re-published.
- Identified by SHA-256 in the provenance manifest; the file itself is never
  copied into an artefact.

### Imaging and derivatives

- `data/raw/`, `data/derivatives/`, `data/external/` are **git-ignored**.
- All NIfTI/GIFTI/MGZ extensions are git-ignored globally as a second line of
  defence.

---

## 3. Egress

Scientific processing is entirely local. No phenotype, imaging or derivative
data is sent to:

- OpenAI, Anthropic or any other LLM API
- Sentry or any error-reporting service
- Weights & Biases or any experiment tracker
- Any analytics or telemetry endpoint

`config/default.yaml` sets `privacy.allow_external_telemetry: false` and
`privacy.allow_external_llm_calls: false`. The compose stack sets
`HF_HUB_DISABLE_TELEMETRY=1` and `WANDB_DISABLED=true`.

The only outbound network traffic in normal operation is pulling the fMRIPrep
container image and the pretrained TRIBE weights — both operator-initiated,
both cached to a named volume so they are fetched once.

---

## 4. Commercial-use restriction

HBN marks a subset of participants as restricted from commercial use. This is
parsed into `Subject.commercial_use_allowed`, frozen into the cohort snapshot,
and surfaced in the Subject Explorer. It is **recorded, not silently applied** —
excluding participants without saying so would be its own integrity failure.

V1 as a whole is research / non-commercial: the official TRIBE v2 repository is
CC-BY-NC-4.0.

---

## 5. Version control

`.gitignore` hard-excludes `data/phenotype/**`, `data/stimuli/**`, `data/raw/**`,
`data/derivatives/**`, `data/external/**`, `data/metadata/**`, `cache/**`,
`work/**`, every neuroimaging binary extension, `.env`, and `license.txt`.

Only two files under `data/` are ever tracked: the two intake `README.md` files
that tell the operator what to place where.

**Before pushing to any remote, verify:**

```bash
git status --porcelain | grep -E 'data/(phenotype|stimuli|raw|derivatives)' && echo "STOP"
```

---

## 6. Audit trail

`audit_events` is append-only: never updated, never deleted. It records asset
discovery and content changes, every stage transition, every blocker raised and
cleared, cohort builds with case/control counts, exclusion decisions, TRIBE
cache hits and completions, and every invalid analysis with its reason.

Readable at `/api/logs/audit`.

---

## 7. Retention

| Class | Location | Retention |
|---|---|---|
| Protected phenotype | `data/phenotype/` | Operator-controlled; delete to revoke |
| Stimulus | `data/stimuli/` | Operator-controlled |
| fMRIPrep derivatives | `data/derivatives/fmriprep/` | Cached; safe to delete (recomputed on demand) |
| TRIBE predictions | `data/tribe/` | Cached per stimulus hash + model revision |
| Analysis artefacts | `data/analysis/` | Keyed by cache key; safe to delete |
| Reports | `data/reports/` | Timestamped; keep for the record |
| Logs | `data/logs/` | Rotating, 5 × 32 MB |

Deleting a cache directory is always safe: the Autopilot detects the missing
artefact and recomputes it.

---

## 8. Operator responsibilities

1. Complete the HBN Data Usage Agreement and any institutional approvals before
   placing phenotype data on this machine.
2. Obtain the movie stimulus lawfully. HBN documents the intervals and asks
   researchers to contact the Child Mind Institute for exact-clip information.
3. Register for a FreeSurfer license under its own terms.
4. Do not commit `data/` contents, and do not deploy this UI to a public network
   without adding authentication — it has none by design, as a local research
   tool.
5. When publishing, include `analysis_config_hash`, `cohort_hash` and the TRIBE
   commit from the provenance manifest.
