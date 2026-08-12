"""Research report generation.

Every report carries the RESEARCH USE ONLY banner, the reproducibility manifest,
and an explicit limitations section. A report is never emitted without stating
which cohort, stimulus, model revision and configuration produced it.
"""

from __future__ import annotations

import csv
import html
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from neurotribe import RESEARCH_DISCLAIMER, __version__
from neurotribe.analysis.group import load_results
from neurotribe.config import Settings
from neurotribe.database.enums import AnalysisTier, CohortGroup
from neurotribe.database.models import (
    Cohort, CohortMember, GroupAnalysisRun, GroupResult, Stimulus, Subject,
    SubjectComparison, TribeRun,
)
from neurotribe.database.repository import active_blockers, register_artifact
from neurotribe.hashing import hash_file
from neurotribe.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class ReportContext:
    settings: Settings
    generated_at: str
    profile: str
    provenance: dict
    cohort: Cohort | None
    diagnostics: dict
    group_run: GroupAnalysisRun | None
    results: list[dict]
    qc_summary: dict
    blockers: list[dict]
    stimulus: Stimulus | None
    tribe_run: TribeRun | None
    limitations: list[str]


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

def _limitations(context_bits: dict) -> list[str]:
    """Always-present limitations, plus any triggered by this specific run."""
    items = [
        "This is a research analysis system. It is not a diagnostic tool and does "
        "not estimate any individual's probability of having ADHD.",
        "TRIBE v2 predicts an average-subject cortical response. Deviation from it "
        "reflects difference from a normative model, not pathology.",
        "Head motion is a well-known confound in developmental and clinical imaging. "
        "Motion is censored and adjusted for, but residual motion effects cannot be "
        "fully excluded.",
        "HBN acquisitions span multiple sites and scanners, which introduces technical "
        "variance. Site is included as a model covariate.",
        "There is no single consensus denoising strategy; the exact strategy used is "
        "recorded in the provenance manifest and results may depend on it.",
        "Group sizes and the observational design mean these results are associative, "
        "not causal.",
    ]
    if context_bits.get("mock_backend"):
        items.insert(0, (
            "CRITICAL: this run used the MOCK TRIBE backend. The numbers below "
            "demonstrate the pipeline only and are NOT scientific results."
        ))
    if context_bits.get("approximate_atlas"):
        items.append(
            "The parcellation is a deterministic geometric fallback rather than the "
            "Schaefer atlas; ROI names are not anatomical labels."
        )
    if context_bits.get("approximate_surfaces"):
        items.append(
            "Some participants used the approximate development surface projection; "
            "they are excluded from the final analysis."
        )
    if context_bits.get("small_groups"):
        items.append(
            "One or both groups fall below the configured minimum size; the analysis "
            "is underpowered and should be treated as preliminary."
        )
    if context_bits.get("gsr"):
        items.append(
            "Global signal regression was enabled - a non-default choice that changes "
            "the interpretation of deviation measures."
        )
    return items


def build_context(session: Session, settings: Settings,
                  tier: AnalysisTier = AnalysisTier.PRIMARY) -> ReportContext:
    from neurotribe.analysis.group import build_provenance
    from neurotribe.cohort.matching import diagnose
    from neurotribe.preprocessing.qc import build_rows, summarize

    group_run = session.execute(
        select(GroupAnalysisRun).where(GroupAnalysisRun.tier == tier.value)
        .order_by(GroupAnalysisRun.created_at.desc())
    ).scalars().first()

    cohort = session.get(Cohort, group_run.cohort_id) if group_run and group_run.cohort_id else None
    if cohort is None:
        cohort = session.execute(
            select(Cohort).where(Cohort.tier == tier.value)
            .order_by(Cohort.updated_at.desc())
        ).scalars().first()

    results: list[dict] = []
    if group_run is not None:
        payload = load_results(group_run)
        results = payload.get("results", [])
        if not results:
            results = [
                {
                    "unit_type": r.unit_type, "unit_name": r.unit_name,
                    "network": r.network, "metric": r.metric,
                    "mean_case": r.mean_case, "mean_control": r.mean_control,
                    "beta_adhd": r.beta_adhd, "p_value": r.p_value,
                    "q_value": r.q_value, "effect_size": r.effect_size,
                    "ci_low": r.ci_low, "ci_high": r.ci_high,
                    "n_case": r.n_case, "n_control": r.n_control,
                }
                for r in session.execute(
                    select(GroupResult).where(GroupResult.run_id == group_run.id)
                ).scalars()
            ]

    diagnostics = diagnose(cohort.members).to_dict() if cohort else {}
    qc_summary = summarize(build_rows(session, settings))

    tribe_run = session.execute(
        select(TribeRun).where(TribeRun.status == "DONE")
        .order_by(TribeRun.created_at.desc())
    ).scalars().first()
    stimulus = None
    if tribe_run and tribe_run.stimulus_id:
        stimulus = session.get(Stimulus, tribe_run.stimulus_id)

    provenance = (group_run.provenance if group_run and group_run.provenance
                  else (build_provenance(session, settings, cohort, group_run)
                        if cohort and group_run else _minimal_provenance(settings, tribe_run)))

    from neurotribe.preprocessing.surfaces import load_parcellation

    parcellation = load_parcellation(settings)
    minimum = int(settings.get("cohort.min_group_size", 10))

    limitations = _limitations({
        "mock_backend": bool(tribe_run and tribe_run.backend == "mock"),
        "approximate_atlas": parcellation.is_approximate,
        "approximate_surfaces": any(
            c.is_approximate for c in session.execute(select(SubjectComparison)).scalars()
        ),
        "small_groups": bool(cohort and (cohort.n_case < minimum or cohort.n_control < minimum)),
        "gsr": bool(settings.get("preprocessing.denoise.global_signal_regression", False)),
    })

    return ReportContext(
        settings=settings,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        profile=settings.profile, provenance=provenance, cohort=cohort,
        diagnostics=diagnostics, group_run=group_run, results=results,
        qc_summary=qc_summary,
        blockers=[{"kind": b.kind, "title": b.title, "severity": b.severity,
                   "required_action": b.required_action}
                  for b in active_blockers(session)],
        stimulus=stimulus, tribe_run=tribe_run, limitations=limitations,
    )


def _minimal_provenance(settings: Settings, tribe_run: TribeRun | None) -> dict:
    return {
        "neurotribe_version": __version__,
        "profile": settings.profile,
        "analysis_config_hash": settings.analysis_config_hash,
        "tribe_backend": tribe_run.backend if tribe_run else None,
        "tribe_commit": tribe_run.tribe_commit if tribe_run else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "research_use_only": True,
        "note": "No completed group analysis; this manifest describes the environment only.",
    }


# --------------------------------------------------------------------------
# Renderers
# --------------------------------------------------------------------------

def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "&mdash;"
    if isinstance(value, float):
        if value != value:  # NaN
            return "&mdash;"
        if abs(value) < 0.001 and value != 0:
            return f"{value:.2e}"
        return f"{value:.{digits}f}"
    return html.escape(str(value))


def _table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "<p class='muted'>No data available.</p>"
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body = []
    for row in rows:
        cells = "".join(f"<td>{_fmt(row.get(key))}</td>" for key, _ in columns)
        body.append(f"<tr>{cells}</tr>")
    return (f"<div class='scroll'><table><thead><tr>{head}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></div>")


HTML_STYLE = """
:root{--bg:#ffffff;--fg:#12151c;--muted:#5d6470;--line:#e2e5ec;--accent:#3b5bdb;
--warn:#b45309;--bad:#b91c1c;--good:#15803d;--card:#f7f8fa;}
@media (prefers-color-scheme:dark){:root{--bg:#0d1017;--fg:#e6e9ef;--muted:#98a0ae;
--line:#232936;--accent:#7d95f5;--card:#141924;}}
*{box-sizing:border-box}
body{margin:0;padding:2.5rem 1.5rem;background:var(--bg);color:var(--fg);
font:15px/1.65 -apple-system,BlinkMacSystemFont,'Segoe UI',Inter,system-ui,sans-serif;}
.wrap{max-width:1080px;margin:0 auto}
h1{font-size:1.9rem;letter-spacing:-.02em;margin:0 0 .35rem}
h2{font-size:1.25rem;margin:2.5rem 0 .75rem;padding-bottom:.4rem;
border-bottom:1px solid var(--line)}
h3{font-size:1rem;margin:1.5rem 0 .5rem}
.sub{color:var(--muted);margin:0 0 1.5rem}
.banner{background:#fef3c7;color:#78350f;border:1px solid #fcd34d;border-radius:10px;
padding:.85rem 1rem;font-weight:600;margin:0 0 1.75rem}
@media (prefers-color-scheme:dark){.banner{background:#2a1f07;color:#fbbf24;
border-color:#78350f}}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:.85rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:.9rem 1rem}
.card .k{color:var(--muted);font-size:.75rem;text-transform:uppercase;letter-spacing:.06em}
.card .v{font-size:1.5rem;font-weight:600;margin-top:.2rem}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:.87rem;min-width:520px}
th,td{padding:.5rem .65rem;text-align:left;border-bottom:1px solid var(--line);
white-space:nowrap}
th{color:var(--muted);font-weight:600;font-size:.75rem;text-transform:uppercase;
letter-spacing:.05em}
tbody tr:hover{background:var(--card)}
.muted{color:var(--muted)}
ul{padding-left:1.15rem}
li{margin:.35rem 0}
code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.82rem}
pre{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:1rem;overflow-x:auto}
.tag{display:inline-block;padding:.15rem .5rem;border-radius:999px;font-size:.72rem;
font-weight:600;letter-spacing:.03em}
.tag.primary{background:#dbeafe;color:#1e40af}
.tag.exploratory{background:#f3e8ff;color:#6b21a8}
.warn{color:var(--warn)}
footer{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--line);
color:var(--muted);font-size:.82rem}
"""


def render_html(context: ReportContext) -> str:
    settings = context.settings
    cohort = context.cohort
    run = context.group_run

    networks = [r for r in context.results
                if r.get("unit_type") == "network" and r.get("metric") == "mad"]
    networks.sort(key=lambda r: (r.get("q_value") if r.get("q_value") is not None else 1.0))
    rois = [r for r in context.results
            if r.get("unit_type") == "roi" and r.get("metric") == "mad"]
    rois.sort(key=lambda r: (r.get("q_value") if r.get("q_value") is not None else 1.0))

    tier_class = "primary" if (run and run.tier == "PRIMARY") else "exploratory"
    stimulus = context.stimulus

    cards = [
        ("Confirmed ADHD", cohort.n_case if cohort else "—"),
        ("Comparison cohort", cohort.n_control if cohort else "—"),
        ("Excluded", cohort.n_excluded if cohort else "—"),
        ("Units tested", len({(r["unit_type"], r["unit_name"]) for r in context.results})),
        ("QC pass", context.qc_summary.get("by_status", {}).get("PASS", 0)),
        ("Profile", context.profile),
    ]
    card_html = "".join(
        f"<div class='card'><div class='k'>{html.escape(str(k))}</div>"
        f"<div class='v'>{html.escape(str(v))}</div></div>"
        for k, v in cards
    )

    blocker_html = "<p class='muted'>No active blockers.</p>"
    if context.blockers:
        blocker_html = "<ul>" + "".join(
            f"<li><strong>{html.escape(b['title'])}</strong> "
            f"<span class='tag'>{html.escape(b['severity'])}</span><br>"
            f"<span class='muted'>{html.escape(b.get('required_action') or '')}</span></li>"
            for b in context.blockers
        ) + "</ul>"

    warnings_html = ""
    if cohort and cohort.warnings:
        warnings_html = "<ul class='warn'>" + "".join(
            f"<li>{html.escape(w)}</li>" for w in cohort.warnings
        ) + "</ul>"

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NeuroTRIBE-HBN Research Report</title>
<style>{HTML_STYLE}</style></head><body><div class="wrap">

<div class="banner">{html.escape(RESEARCH_DISCLAIMER)}</div>

<h1>NeuroTRIBE-HBN Research Report</h1>
<p class="sub">{html.escape(str(settings.get('project.title', '')))}<br>
Generated {html.escape(context.generated_at)} &middot; NeuroTRIBE v{__version__}
&middot; profile <code>{html.escape(context.profile)}</code></p>

<div class="grid">{card_html}</div>

<h2>1. Dataset</h2>
<p>Healthy Brain Network movie-fMRI. Stimulus:
<strong>{html.escape(stimulus.label if stimulus else 'not yet supplied')}</strong>
{f"({stimulus.duration_sec:.1f} s, documented interval {html.escape(str(stimulus.source_interval_start))}–{html.escape(str(stimulus.source_interval_end))})" if stimulus and stimulus.duration_sec else ""}.
Reference model: TRIBE v2 as a pretrained normative encoder (never fine-tuned).</p>

<h2>2. Cohort construction</h2>
<p>Primary contrast: <strong>Confirmed ADHD</strong> vs
<strong>{html.escape(CohortGroup.NO_DIAGNOSIS_GIVEN.display)}</strong>. Certainty
levels are never pooled. The exploratory contrast uses the
<em>{html.escape(CohortGroup.NON_ADHD_COMPARISON.display)}</em>, which may include
participants with other diagnoses and is therefore never described as
&ldquo;healthy controls&rdquo;.</p>
{warnings_html}

<h3>Covariate balance</h3>
{_table(context.diagnostics.get('continuous', []),
        [('variable', 'Variable'), ('case_mean', 'ADHD mean'),
         ('control_mean', 'Comparison mean'), ('std_mean_difference', 'SMD'),
         ('n_case', 'n ADHD'), ('n_control', 'n comparison')])}

<h2>3. Quality control</h2>
{_table([context.qc_summary],
        [('n_rows', 'Participants'), ('n_approximate', 'Approximate surfaces'),
         ('median_usable_frame_fraction', 'Median usable frames'),
         ('median_mean_fd', 'Median mean FD (mm)')])}

<h2>4. Preprocessing</h2>
<p>fMRIPrep {html.escape(str(settings.get('preprocessing.fmriprep.version_pin')))}
&rarr; FreeSurfer surfaces &rarr;
<code>{html.escape(str(settings.get('surface.space')))}</code>. Denoising strategy:
<code>{html.escape(str(settings.get('preprocessing.denoise.strategy')))}</code>;
global signal regression:
<strong>{'enabled' if settings.get('preprocessing.denoise.global_signal_regression') else 'disabled'}</strong>.
Motion censoring at FD &gt; {html.escape(str(settings.get('qc.motion.fd_threshold_mm')))} mm.</p>

<h2>5. TRIBE model</h2>
<p>Backend <code>{html.escape(str(context.provenance.get('tribe_backend')))}</code>,
commit <code>{html.escape(str(context.provenance.get('tribe_commit'))[:12])}</code>,
model <code>{html.escape(str(context.provenance.get('tribe_model')))}</code>.
Inference was executed once per stimulus and cached.</p>

<h2>6. Temporal and spatial alignment</h2>
<p>TRIBE predictions are interpolated onto each participant's acquisition time
grid using the timestamps TRIBE returns. TRIBE's published timing already
incorporates a 5&nbsp;s hemodynamic-lag offset, so <strong>no additional shift is
applied</strong>. Extrapolation beyond the prediction support is refused, and
censored frames are excluded from every metric. Hemisphere ordering was
established as
<code>{html.escape(str(context.provenance.get('hemi_order')))}</code>.</p>

<h2>7. Subject-level results</h2>
<p>Per participant we compute vertex-wise Pearson agreement, standardized
residuals, mean absolute deviation and residual variance, then aggregate to ROIs
and canonical networks.</p>

<h2>8. ADHD group results
<span class="tag {tier_class}">{html.escape(run.tier if run else 'PRIMARY')}</span></h2>
<p>Model: <code>{html.escape(run.model_formula if run else str(settings.get('analysis.group.model')))}</code>
&middot; correction: {html.escape(run.correction if run else 'fdr_bh')}
&middot; alpha {_fmt(run.alpha if run else 0.05, 2)}</p>

<h3>Networks (mean absolute deviation)</h3>
{_table(networks[:20],
        [('unit_name', 'Network'), ('mean_case', 'ADHD'), ('mean_control', 'Comparison'),
         ('effect_size', "Cohen's d"), ('ci_low', 'CI low'), ('ci_high', 'CI high'),
         ('p_value', 'p'), ('q_value', 'q (FDR)'), ('n_case', 'n ADHD'),
         ('n_control', 'n comp.')])}

<h3>Top ROIs by FDR rank</h3>
{_table(rois[:25],
        [('unit_name', 'ROI'), ('network', 'Network'), ('mean_case', 'ADHD'),
         ('mean_control', 'Comparison'), ('effect_size', "Cohen's d"),
         ('p_value', 'p'), ('q_value', 'q (FDR)')])}

<h2>9. Sensitivity analyses</h2>
<p>Exploratory outputs (vertex-level maps, the non-ADHD comparison cohort,
alternative motion thresholds, comorbidity subsets) are produced separately and
are always labelled EXPLORATORY. They do not modify the primary result.</p>

<h2>10. Limitations</h2>
<ul>{''.join(f'<li>{html.escape(item)}</li>' for item in context.limitations)}</ul>

<h2>11. Reproducibility</h2>
<pre>{html.escape(json.dumps(context.provenance, indent=2, default=str))}</pre>

<h3>Active blockers</h3>
{blocker_html}

<footer>{html.escape(RESEARCH_DISCLAIMER)} &middot; NeuroTRIBE-HBN v{__version__}
&middot; analysis config hash
<code>{html.escape(str(context.provenance.get('analysis_config_hash')))}</code></footer>

</div></body></html>"""


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def render_pdf(html_path: Path, pdf_path: Path) -> Path | None:
    """Render the HTML report to PDF when a renderer is available."""
    try:
        from weasyprint import HTML  # type: ignore

        HTML(filename=str(html_path)).write_pdf(str(pdf_path))
        return pdf_path
    except Exception as exc:  # noqa: BLE001 - weasyprint has heavy native deps
        log.info("PDF rendering unavailable; HTML report is authoritative",
                 extra={"error": str(exc)})
        return None


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def generate_all(session: Session, settings: Settings) -> dict:
    """Produce every report artefact and register it in the database."""
    out_dir = settings.paths.reports / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict] = []

    for tier in (AnalysisTier.PRIMARY, AnalysisTier.EXPLORATORY):
        context = build_context(session, settings, tier)
        if context.group_run is None and tier is AnalysisTier.EXPLORATORY:
            continue

        suffix = tier.value.lower()
        html_path = out_dir / f"report_{suffix}.html"
        html_path.write_text(render_html(context), encoding="utf-8")
        artifacts.append(_register(session, settings, "report_html",
                                   f"Research report ({tier.value})", html_path,
                                   "text/html", tier, context.provenance))

        pdf_path = render_pdf(html_path, out_dir / f"report_{suffix}.pdf")
        if pdf_path is not None:
            artifacts.append(_register(session, settings, "report_pdf",
                                       f"Research report PDF ({tier.value})", pdf_path,
                                       "application/pdf", tier, context.provenance))

        roi_rows = [r for r in context.results if r.get("unit_type") == "roi"]
        network_rows = [r for r in context.results if r.get("unit_type") == "network"]
        columns = ["unit_type", "unit_name", "network", "metric", "mean_case",
                   "mean_control", "sd_case", "sd_control", "beta_adhd", "se_adhd",
                   "t_stat", "p_value", "q_value", "effect_size", "ci_low", "ci_high",
                   "n_case", "n_control"]

        if roi_rows:
            path = write_csv(out_dir / f"roi_results_{suffix}.csv", roi_rows, columns)
            artifacts.append(_register(session, settings, "roi_csv",
                                       f"ROI results ({tier.value})", path,
                                       "text/csv", tier, context.provenance))
        if network_rows:
            path = write_csv(out_dir / f"network_results_{suffix}.csv", network_rows, columns)
            artifacts.append(_register(session, settings, "network_csv",
                                       f"Network results ({tier.value})", path,
                                       "text/csv", tier, context.provenance))

        manifest_path = out_dir / f"provenance_{suffix}.json"
        manifest_path.write_text(
            json.dumps(context.provenance, indent=2, default=str), encoding="utf-8",
        )
        artifacts.append(_register(session, settings, "provenance",
                                   f"Provenance manifest ({tier.value})", manifest_path,
                                   "application/json", tier, context.provenance))

    log.info("Report generation complete",
             extra={"out_dir": str(out_dir), "n_artifacts": len(artifacts)})
    return {"out_dir": str(out_dir), "artifacts": artifacts}


def _register(session: Session, settings: Settings, kind: str, label: str, path: Path,
              media_type: str, tier: AnalysisTier, provenance: dict) -> dict:
    digest = hash_file(path)
    try:
        relative = str(path.relative_to(settings.root))
    except ValueError:
        relative = str(path)
    register_artifact(session, kind, label, str(path), media_type=media_type,
                      sha256=digest.sha256, size_bytes=digest.size_bytes,
                      tier=tier.value, provenance=provenance)
    return {"kind": kind, "label": label, "path": relative,
            "sha256": digest.sha256, "size_bytes": digest.size_bytes}
