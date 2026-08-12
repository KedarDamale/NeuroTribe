"""Subject-level analysis orchestration.

    prepared BOLD  +  TRIBE prediction
            -> spatial alignment
            -> temporal alignment
            -> sanity gate
            -> deviation metrics
            -> ROI / network aggregation
            -> rolling deviation + peak windows
            -> persisted comparison

A comparison that fails the sanity gate is stored with ``valid = False`` and an
explicit reason. It is never silently discarded and never enters a group model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sqlalchemy import select
from sqlalchemy.orm import Session

from neurotribe.alignment import spatial, temporal, validate as sanity
from neurotribe.analysis import residuals as residuals_mod, roi as roi_mod
from neurotribe.config import Settings
from neurotribe.database.models import (
    NetworkMetric, PreprocessingRun, RoiMetric, Scan, Stimulus, Subject,
    SubjectComparison, TribeRun,
)
from neurotribe.database.repository import record_audit
from neurotribe.hashing import cache_key
from neurotribe.logging_setup import get_logger
from neurotribe.preprocessing import pipeline as preproc_pipeline, surfaces
from neurotribe.tribe.inference import TribePrediction

log = get_logger(__name__)


@dataclass
class SubjectAnalysisResult:
    comparison_id: str
    subject_external_id: str
    valid: bool
    global_r: float | None = None
    global_mad: float | None = None
    n_usable_frames: int = 0
    peak_windows: list[dict] = field(default_factory=list)
    top_rois: list[dict] = field(default_factory=list)
    top_networks: list[dict] = field(default_factory=list)
    invalid_reason: str | None = None
    is_approximate: bool = False
    reports: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "comparison_id": self.comparison_id,
            "subject_external_id": self.subject_external_id,
            "valid": self.valid, "global_agreement_r": self.global_r,
            "global_mad": self.global_mad, "n_usable_frames": self.n_usable_frames,
            "peak_windows": self.peak_windows, "top_rois": self.top_rois,
            "top_networks": self.top_networks, "invalid_reason": self.invalid_reason,
            "is_approximate": self.is_approximate, "reports": self.reports,
        }


def _artifact_dir(settings: Settings, subject: Subject, movie: str) -> Path:
    path = settings.paths.analysis / "subjects" / subject.external_id / movie
    path.mkdir(parents=True, exist_ok=True)
    return path


def analyze(session: Session, settings: Settings, subject: Subject, scan: Scan,
            run: PreprocessingRun, prediction: TribePrediction,
            tribe_run: TribeRun, stimulus: Stimulus | None = None,
            *, force: bool = False) -> SubjectAnalysisResult:
    """Run the full comparison for one participant."""
    key = cache_key(
        "comparison",
        subject=subject.external_id, run=run.id, tribe_run=tribe_run.id,
        denoised=run.denoised_path, analysis=settings.get("analysis"),
        alignment=settings.get("alignment"), surface=settings.get("surface"),
    )

    existing = session.execute(
        select(SubjectComparison).where(SubjectComparison.cache_key == key)
    ).scalar_one_or_none()
    if existing is not None and not force:
        log.info("Subject comparison served from cache",
                 extra={"subject": subject.external_id})
        return SubjectAnalysisResult(
            comparison_id=existing.id, subject_external_id=subject.external_id,
            valid=existing.valid, global_r=existing.global_agreement_r,
            global_mad=existing.global_mad,
            n_usable_frames=existing.n_usable_timepoints or 0,
            peak_windows=existing.peak_windows or [],
            invalid_reason=existing.invalid_reason,
            is_approximate=existing.is_approximate,
            reports={"cached": True},
        )

    comparison = existing or SubjectComparison(cache_key=key)
    comparison.subject_id = subject.id
    comparison.scan_id = scan.id
    comparison.preprocessing_run_id = run.id
    comparison.tribe_run_id = tribe_run.id
    comparison.movie = scan.movie
    comparison.tr = scan.repetition_time
    if existing is None:
        session.add(comparison)
    session.flush()

    def fail(reason: str, reports: dict | None = None) -> SubjectAnalysisResult:
        comparison.valid = False
        comparison.invalid_reason = reason[:2000]
        comparison.sanity_report = (reports or {}).get("sanity", {})
        comparison.alignment_report = (reports or {}).get("alignment", {})
        record_audit(session, "analysis.subject_invalid", entity_type="subject_comparison",
                     entity_id=comparison.id, summary=f"{subject.external_id}: {reason[:120]}")
        log.error("Subject comparison invalid",
                  extra={"subject": subject.external_id, "reason": reason})
        return SubjectAnalysisResult(
            comparison_id=comparison.id, subject_external_id=subject.external_id,
            valid=False, invalid_reason=reason, reports=reports or {},
        )

    # -- load prepared observation ------------------------------------
    try:
        prepared = preproc_pipeline.load_prepared(run)
    except (FileNotFoundError, OSError) as exc:
        return fail(f"Prepared time series unavailable: {exc}")

    comparison.is_approximate = prepared.is_approximate
    reports: dict = {}

    # -- spatial alignment --------------------------------------------
    try:
        spatially = spatial.align(
            prediction.predictions, prepared.data, settings,
            tribe_hemi_order=prediction.hemi_order,
        )
    except (surfaces.SurfaceError, ValueError) as exc:
        return fail(f"Spatial alignment failed: {exc}")
    reports["spatial"] = spatially.report

    # -- temporal alignment -------------------------------------------
    try:
        pair = temporal.align(
            spatially.predicted, prediction.times, prediction.ends,
            spatially.observed, prepared.time_sec, prepared.mask.usable, settings,
        )
    except temporal.AlignmentError as exc:
        return fail(f"Temporal alignment failed: {exc}", {"spatial": spatially.report})
    reports["alignment"] = pair.report

    lag = temporal.estimate_lag(
        pair.predicted[pair.usable], pair.observed[pair.usable], prepared.tr,
        float(settings.get("alignment.validation_lag_search_sec", 8.0)),
    )
    lag_sanity = sanity.check_alignment_lag(lag, settings)
    reports["lag"] = {**lag, **lag_sanity.to_dict()}

    # -- metrics -------------------------------------------------------
    try:
        metrics = residuals_mod.compute(
            pair.predicted, pair.observed, pair.usable, spatially.valid_vertices,
        )
    except ValueError as exc:
        return fail(f"Metric computation failed: {exc}", reports)

    # -- sanity gate ---------------------------------------------------
    duplicate = session.execute(
        select(SubjectComparison).where(
            SubjectComparison.subject_id == subject.id,
            SubjectComparison.movie == scan.movie,
            SubjectComparison.id != comparison.id,
            SubjectComparison.valid.is_(True),
        )
    ).scalars().first() is not None

    sanity_report = sanity.check_subject_comparison(
        predicted=pair.predicted, observed=pair.observed,
        tribe_vertices=prediction.n_vertices, observed_vertices=prepared.n_vertices,
        hemi_order_verified=bool(tribe_run.geometry_validated),
        movie_duration_sec=stimulus.duration_sec if stimulus else None,
        stimulus_expected_sec=stimulus.expected_duration_sec if stimulus else None,
        tr=prepared.tr, n_volumes=scan.n_volumes,
        shared_frames=pair.n_shared, usable_frames=pair.n_usable,
        correlations=metrics.vertex_r, censoring_applied=True,
        settings=settings, subject_seen_before=duplicate,
    )
    sanity_report.warnings.extend(lag_sanity.warnings)
    reports["sanity"] = sanity_report.to_dict()

    if not sanity_report.valid:
        return fail("ANALYSIS INVALID: " + "; ".join(sanity_report.failures), reports)

    # -- ROI / network aggregation -------------------------------------
    parcellation = surfaces.load_parcellation(settings)
    aggregation = roi_mod.aggregate(
        metrics.vertex_r, metrics.vertex_mad, metrics.vertex_residual_variance,
        parcellation, settings,
    )

    # -- movie-moment analysis -----------------------------------------
    rolling = residuals_mod.rolling_deviation(
        metrics.residual, pair.time_sec, pair.usable,
        window_sec=float(settings.get("analysis.rolling_window_sec", 10.0)),
        step_sec=float(settings.get("analysis.rolling_step_sec", 1.0)),
    )
    peaks = residuals_mod.peak_windows(
        rolling, top_n=int(settings.get("analysis.top_peak_windows", 10)),
    )

    # -- persist artefacts ---------------------------------------------
    directory = _artifact_dir(settings, subject, scan.movie)
    np.save(directory / "vertex_r.npy", metrics.vertex_r)
    np.save(directory / "vertex_mad.npy", metrics.vertex_mad)
    np.save(directory / "vertex_residual_variance.npy", metrics.vertex_residual_variance)
    np.savez_compressed(
        directory / "residual.npz",
        residual=metrics.residual.astype(np.float32),
        time_sec=pair.time_sec.astype(np.float32),
        usable=pair.usable,
    )
    np.savez_compressed(
        directory / "rolling_deviation.npz",
        starts=rolling.window_starts, ends=rolling.window_ends,
        deviation=rolling.deviation, coverage=rolling.coverage,
    )

    comparison.global_agreement_r = metrics.global_r
    comparison.global_mad = metrics.global_mad
    comparison.global_residual_variance = metrics.global_residual_variance
    comparison.n_shared_timepoints = pair.n_shared
    comparison.n_usable_timepoints = pair.n_usable
    comparison.usable_frame_fraction = (pair.n_usable / pair.n_shared) if pair.n_shared else 0.0
    comparison.vertex_r_path = str(directory / "vertex_r.npy")
    comparison.vertex_mad_path = str(directory / "vertex_mad.npy")
    comparison.residual_path = str(directory / "residual.npz")
    comparison.rolling_deviation_path = str(directory / "rolling_deviation.npz")
    comparison.peak_windows = peaks
    comparison.alignment_report = {**reports.get("alignment", {}),
                                   "spatial": reports.get("spatial", {}),
                                   "lag": reports.get("lag", {})}
    comparison.sanity_report = sanity_report.to_dict()
    comparison.valid = True
    comparison.invalid_reason = None

    for existing_roi in list(comparison.roi_metrics):
        session.delete(existing_roi)
    for existing_network in list(comparison.network_metrics):
        session.delete(existing_network)
    session.flush()

    for summary in aggregation.rois:
        session.add(RoiMetric(
            comparison_id=comparison.id,
            atlas=str(settings.get("surface.atlas.name", "schaefer")),
            n_parcels=int(settings.get("surface.atlas.n_parcels", 200)),
            roi_index=summary.roi_index, roi_name=summary.roi_name,
            network=summary.network, hemisphere=summary.hemisphere,
            agreement_r=summary.agreement_r, mad=summary.mad,
            residual_variance=summary.residual_variance, n_vertices=summary.n_vertices,
        ))
    for summary in aggregation.networks:
        session.add(NetworkMetric(
            comparison_id=comparison.id, network=summary.network,
            agreement_r=summary.agreement_r, mad=summary.mad,
            residual_variance=summary.residual_variance, n_vertices=summary.n_vertices,
        ))

    record_audit(session, "analysis.subject_completed", entity_type="subject_comparison",
                 entity_id=comparison.id, summary=subject.external_id,
                 payload={"global_r": metrics.global_r, "global_mad": metrics.global_mad,
                          "n_usable": pair.n_usable})
    log.info("Subject comparison complete", extra={
        "subject": subject.external_id, "global_r": round(metrics.global_r, 4),
        "global_mad": round(metrics.global_mad, 4), "n_usable": pair.n_usable,
    })

    return SubjectAnalysisResult(
        comparison_id=comparison.id, subject_external_id=subject.external_id,
        valid=True, global_r=metrics.global_r, global_mad=metrics.global_mad,
        n_usable_frames=pair.n_usable, peak_windows=peaks,
        top_rois=[r.to_dict() for r in aggregation.top_deviation_rois(3)],
        top_networks=[n.to_dict() for n in aggregation.top_deviation_networks(3)],
        is_approximate=prepared.is_approximate, reports=reports,
    )


def load_vertex_map(comparison: SubjectComparison, which: str) -> np.ndarray | None:
    """Load a persisted vertex map for the 3D viewer."""
    path = {
        "agreement": comparison.vertex_r_path,
        "deviation": comparison.vertex_mad_path,
    }.get(which)
    if not path or not Path(path).exists():
        return None
    return np.load(path)


def load_timecourses(comparison: SubjectComparison) -> dict | None:
    """Load residual time courses for the synchronised subject timeline."""
    if not comparison.residual_path or not Path(comparison.residual_path).exists():
        return None
    with np.load(comparison.residual_path, allow_pickle=False) as payload:
        residual = payload["residual"]
        time_sec = payload["time_sec"]
        usable = payload["usable"]
    with np.errstate(invalid="ignore"):
        global_deviation = np.nanmean(np.abs(residual), axis=1)
    return {
        "time_sec": time_sec.tolist(),
        "usable": usable.astype(bool).tolist(),
        "global_deviation": [None if not np.isfinite(v) else round(float(v), 5)
                             for v in global_deviation],
    }
