"""Scientific validation (specification section 74).

These are the tests that would catch a silently wrong result:

    identical signals      -> correlation ~ 1
    random signals         -> correlation ~ 0
    shifted signals        -> the alignment diagnostic detects the shift
    one hemisphere flipped -> the geometry test fails
    missing frames         -> the censor mask is applied correctly
"""

from __future__ import annotations

import numpy as np
import pytest

from neurotribe.alignment import spatial, temporal
from neurotribe.alignment.validate import check_subject_comparison
from neurotribe.analysis import residuals
from neurotribe.preprocessing import censoring
from neurotribe.preprocessing.surfaces import (
    SurfaceError, concatenate_hemispheres, split_hemispheres,
)
from neurotribe.tribe import geometry

from tests.conftest import make_timeseries


# --------------------------------------------------------------------------
# Correlation behaviour
# --------------------------------------------------------------------------

def test_identical_signals_correlate_at_one(rng):
    """A vertex whose observation equals its prediction must give r = 1."""
    data = make_timeseries(150, 32, rng)
    usable = np.ones(150, dtype=bool)

    r = residuals.pearson_per_vertex(
        residuals.zscore_usable(data, usable),
        residuals.zscore_usable(data.copy(), usable),
        usable,
    )
    assert np.allclose(r, 1.0, atol=1e-4), f"max deviation {np.abs(r - 1).max()}"


def test_random_signals_correlate_near_zero(rng):
    """Independent signals must not produce spurious agreement."""
    predicted = make_timeseries(4000, 40, rng, autocorr=0.0)
    observed = make_timeseries(4000, 40, np.random.default_rng(999), autocorr=0.0)
    usable = np.ones(4000, dtype=bool)

    r = residuals.pearson_per_vertex(
        residuals.zscore_usable(predicted, usable),
        residuals.zscore_usable(observed, usable),
        usable,
    )
    assert abs(float(np.mean(r))) < 0.05
    # No individual vertex should look strongly correlated by chance.
    assert np.abs(r).max() < 0.25


def test_identical_signals_give_zero_deviation(rng):
    data = make_timeseries(120, 24, rng)
    usable = np.ones(120, dtype=bool)
    z = residuals.zscore_usable(data, usable)

    residual = residuals.standardized_residual(z, z)
    assert np.allclose(residual, 0.0, atol=1e-5)
    assert np.allclose(residuals.mean_absolute_deviation(residual, usable), 0.0, atol=1e-5)


def test_correlations_stay_within_bounds(rng):
    """Floating-point drift must never push r outside [-1, 1]."""
    predicted = make_timeseries(300, 128, rng)
    observed = predicted * 3.0 - 7.0          # perfectly affine-related
    usable = np.ones(300, dtype=bool)

    r = residuals.pearson_per_vertex(
        residuals.zscore_usable(predicted, usable),
        residuals.zscore_usable(observed, usable),
        usable,
    )
    assert r.min() >= -1.0 and r.max() <= 1.0
    assert np.allclose(r, 1.0, atol=1e-4)


def test_anticorrelated_signals_give_minus_one(rng):
    data = make_timeseries(200, 16, rng)
    usable = np.ones(200, dtype=bool)
    r = residuals.pearson_per_vertex(
        residuals.zscore_usable(data, usable),
        residuals.zscore_usable(-data, usable),
        usable,
    )
    assert np.allclose(r, -1.0, atol=1e-4)


# --------------------------------------------------------------------------
# Temporal shift detection
# --------------------------------------------------------------------------

def test_shifted_signals_are_detected_by_the_lag_diagnostic(rng):
    """A deliberate temporal offset must be reported, not silently absorbed."""
    tr = 0.8
    shift_frames = 5
    base = make_timeseries(400, 20, rng, autocorr=0.9)

    # observed[t] == predicted[t - shift_frames]: the observation trails the
    # prediction, so the peak correlation sits at a positive lag.
    predicted = base[shift_frames:]
    observed = base[:-shift_frames]

    report = temporal.estimate_lag(predicted, observed, tr, max_lag_sec=10.0)
    assert report["ok"]
    assert abs(report["best_lag_frames"] - shift_frames) <= 1, report
    assert abs(report["best_lag_sec"] - shift_frames * tr) < tr * 1.5
    # A real offset must be clearly better than assuming no offset at all.
    assert report["best_correlation"] > report["zero_lag_correlation"]

    # And the sanity gate must flag it rather than absorb it.
    from neurotribe.alignment.validate import check_alignment_lag
    from neurotribe.config import load_settings

    sanity = check_alignment_lag(report, load_settings("development"))
    assert not sanity.checks["residual_lag_acceptable"]


def test_aligned_signals_show_no_lag(rng):
    tr = 0.8
    base = make_timeseries(400, 20, rng, autocorr=0.9)
    report = temporal.estimate_lag(base, base.copy(), tr, max_lag_sec=8.0)
    assert report["best_lag_frames"] == 0
    assert report["best_lag_sec"] == 0.0


def test_interpolation_refuses_to_extrapolate():
    """Inventing predictions outside TRIBE's support would fabricate data."""
    source_times = np.linspace(5.0, 15.0, 11)
    values = np.tile(np.linspace(0, 1, 11).reshape(-1, 1), (1, 4))
    target_times = np.array([0.0, 5.0, 10.0, 15.0, 20.0])

    resampled, in_support = temporal.interpolate_to_grid(
        values, source_times, target_times, allow_extrapolation=False,
    )
    assert in_support.tolist() == [False, True, True, True, False]
    assert np.isnan(resampled[0]).all()
    assert np.isnan(resampled[-1]).all()
    assert np.isfinite(resampled[1:4]).all()


def test_alignment_errors_when_scan_and_stimulus_do_not_overlap(settings):
    predicted = np.random.default_rng(1).standard_normal((50, 8))
    prediction_times = np.linspace(0.0, 40.0, 50)
    prediction_ends = prediction_times + 0.8
    observed = np.random.default_rng(2).standard_normal((30, 8))
    observed_times = np.linspace(500.0, 520.0, 30)   # entirely disjoint

    with pytest.raises(temporal.AlignmentError, match="trimmed prediction support"):
        temporal.align(predicted, prediction_times, prediction_ends,
                       observed, observed_times, np.ones(30, dtype=bool), settings)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def test_hemisphere_round_trip_is_lossless(rng):
    left = make_timeseries(20, 10242, rng)
    right = make_timeseries(20, 10242, rng)

    combined = concatenate_hemispheres(left, right, ["L", "R"])
    assert combined.shape == (20, 20484)

    parts = split_hemispheres(combined, ["L", "R"])
    assert np.array_equal(parts["L"], left)
    assert np.array_equal(parts["R"], right)


def test_reversed_hemisphere_order_changes_the_data(rng):
    """A silent L/R swap must be detectable, not invisible."""
    left = make_timeseries(10, 10242, rng)
    right = make_timeseries(10, 10242, rng)

    forward = concatenate_hemispheres(left, right, ["L", "R"])
    reversed_ = concatenate_hemispheres(left, right, ["R", "L"])
    assert not np.array_equal(forward, reversed_)

    # Reordering back must recover the original exactly.
    restored = spatial.reorder_to(reversed_, ["R", "L"], ["L", "R"], 10242)
    assert np.array_equal(restored, forward)


def test_geometry_rejects_wrong_vertex_count(settings):
    """A prediction in the wrong surface space must abort the analysis."""
    import pandas as pd

    predictions = np.random.default_rng(4).standard_normal((30, 9999)).astype(np.float32)
    segments = pd.DataFrame({"start": np.arange(30) * 1.49,
                             "end": np.arange(30) * 1.49 + 1.49})

    report = geometry.validate(predictions, segments, settings)
    assert not report.ok
    assert any("vertices" in message for message in report.errors)


def test_geometry_accepts_valid_fsaverage5_output(settings):
    import pandas as pd

    generator = np.random.default_rng(5)
    predictions = generator.standard_normal((40, 20484)).astype(np.float32)
    segments = pd.DataFrame({"start": np.arange(40) * 1.49,
                             "end": np.arange(40) * 1.49 + 1.49})

    report = geometry.validate(predictions, segments, settings)
    assert report.ok, report.errors
    assert report.n_vertices == 20484
    assert report.per_hemi_vertices == 10242
    assert report.hemi_order == ["L", "R"]


def test_geometry_rejects_non_monotonic_timestamps(settings):
    import pandas as pd

    predictions = np.random.default_rng(6).standard_normal((10, 20484)).astype(np.float32)
    starts = np.arange(10, dtype=float)
    starts[5] = 2.0                       # out of order
    segments = pd.DataFrame({"start": starts, "end": starts + 1.0})

    report = geometry.validate(predictions, segments, settings)
    assert not report.ok
    assert any("increasing" in message for message in report.errors)


def test_assert_compatible_blocks_mismatched_spaces():
    with pytest.raises(geometry.GeometryError, match="Vertex-count mismatch"):
        geometry.assert_compatible(20484, 32492)


def test_spatial_align_raises_when_no_vertex_is_usable(settings):
    """All-flat input must abort rather than produce empty 'results'."""
    predicted = np.zeros((20, 20484), dtype=np.float32)
    observed = np.zeros((20, 20484), dtype=np.float32)
    with pytest.raises(SurfaceError, match="No vertex carries usable signal"):
        spatial.align(predicted, observed, settings)


# --------------------------------------------------------------------------
# Censoring
# --------------------------------------------------------------------------

def test_censor_mask_flags_motion_spike(confounds_frame, settings):
    mask = censoring.build_mask(confounds_frame, settings, n_nonsteady=2)

    assert mask.n_total == 120
    assert not mask.usable[0] and not mask.usable[1]        # non-steady-state
    assert mask.reasons[0] == censoring.REASON_NONSTEADY
    assert not mask.usable[60]                              # the injected spike
    assert mask.reasons[60] == censoring.REASON_FD
    assert not mask.usable[61]                              # pad_after = 1
    assert mask.reasons[61] == censoring.REASON_PAD
    assert mask.n_usable < mask.n_total


def test_censored_frames_do_not_influence_metrics(rng):
    """Corrupted frames must be invisible to the correlation once censored."""
    n_timepoints, n_vertices = 200, 16
    predicted = make_timeseries(n_timepoints, n_vertices, rng)
    observed = predicted.copy()

    usable = np.ones(n_timepoints, dtype=bool)
    corrupted = slice(50, 70)
    observed[corrupted] = 1e4 * rng.standard_normal((20, n_vertices))
    usable[corrupted] = False

    metrics = residuals.compute(predicted, observed, usable,
                                np.ones(n_vertices, dtype=bool))
    # With the corruption censored, agreement is still essentially perfect.
    assert metrics.global_r > 0.999
    assert metrics.global_mad < 1e-3
    assert metrics.n_usable_frames == n_timepoints - 20


def test_zscore_uses_only_usable_frames(rng):
    data = make_timeseries(100, 8, rng)
    usable = np.ones(100, dtype=bool)
    usable[80:] = False
    data[80:] += 500.0                    # huge offset in censored frames only

    z = residuals.zscore_usable(data, usable)
    # The standardisation statistics must come from the usable frames.
    assert abs(float(z[usable].mean())) < 1e-4
    assert abs(float(z[usable].std()) - 1.0) < 1e-3


def test_flat_vertices_become_zero_not_nan(rng):
    data = make_timeseries(60, 5, rng)
    data[:, 2] = 3.14                     # constant vertex (e.g. medial wall)
    usable = np.ones(60, dtype=bool)

    z = residuals.zscore_usable(data, usable)
    assert np.isfinite(z).all()
    assert np.allclose(z[:, 2], 0.0)


def test_intersect_requires_matching_lengths(confounds_frame, settings):
    a = censoring.build_mask(confounds_frame, settings)
    b = censoring.build_mask(confounds_frame.iloc[:50], settings)
    with pytest.raises(ValueError, match="differing lengths"):
        censoring.intersect([a, b])


# --------------------------------------------------------------------------
# The sanity gate itself
# --------------------------------------------------------------------------

def test_sanity_gate_rejects_vertex_mismatch(settings):
    report = check_subject_comparison(
        predicted=np.zeros((10, 5)), observed=np.zeros((10, 5)),
        tribe_vertices=20484, observed_vertices=32492,
        hemi_order_verified=True, movie_duration_sec=600.0,
        stimulus_expected_sec=600.0, tr=0.8, n_volumes=750,
        shared_frames=100, usable_frames=90, correlations=np.array([0.5]),
        censoring_applied=True, settings=settings,
    )
    assert not report.valid
    assert report.verdict == "ANALYSIS INVALID"
    assert any("vertex count" in f.lower() for f in report.failures)


def test_sanity_gate_rejects_duplicate_participant(settings):
    report = check_subject_comparison(
        predicted=np.zeros((10, 5)), observed=np.zeros((10, 5)),
        tribe_vertices=20484, observed_vertices=20484,
        hemi_order_verified=True, movie_duration_sec=600.0,
        stimulus_expected_sec=600.0, tr=0.8, n_volumes=750,
        shared_frames=100, usable_frames=90, correlations=np.array([0.5]),
        censoring_applied=True, settings=settings, subject_seen_before=True,
    )
    assert not report.valid
    assert not report.checks["subject_not_duplicated"]


def test_sanity_gate_rejects_uncensored_comparison(settings):
    report = check_subject_comparison(
        predicted=np.zeros((10, 5)), observed=np.zeros((10, 5)),
        tribe_vertices=20484, observed_vertices=20484,
        hemi_order_verified=True, movie_duration_sec=600.0,
        stimulus_expected_sec=600.0, tr=0.8, n_volumes=750,
        shared_frames=100, usable_frames=90, correlations=np.array([0.5]),
        censoring_applied=False, settings=settings,
    )
    assert not report.valid
    assert not report.checks["censoring_applied"]


def test_sanity_gate_rejects_out_of_range_correlations(settings):
    report = check_subject_comparison(
        predicted=np.zeros((10, 5)), observed=np.zeros((10, 5)),
        tribe_vertices=20484, observed_vertices=20484,
        hemi_order_verified=True, movie_duration_sec=600.0,
        stimulus_expected_sec=600.0, tr=0.8, n_volumes=750,
        shared_frames=100, usable_frames=90,
        correlations=np.array([0.5, 1.7]),        # impossible
        censoring_applied=True, settings=settings,
    )
    assert not report.valid
    assert not report.checks["correlations_in_bounds"]


def test_sanity_gate_rejects_wrong_stimulus_duration(settings):
    report = check_subject_comparison(
        predicted=np.zeros((10, 5)), observed=np.zeros((10, 5)),
        tribe_vertices=20484, observed_vertices=20484,
        hemi_order_verified=True,
        movie_duration_sec=201.0,          # The Present ...
        stimulus_expected_sec=600.0,       # ... but Despicable Me was expected
        tr=0.8, n_volumes=750, shared_frames=100, usable_frames=90,
        correlations=np.array([0.5]), censoring_applied=True, settings=settings,
    )
    assert not report.valid
    assert not report.checks["movie_duration_matches"]


def test_sanity_gate_passes_a_clean_comparison(settings):
    report = check_subject_comparison(
        predicted=np.random.default_rng(1).standard_normal((100, 20)),
        observed=np.random.default_rng(2).standard_normal((100, 20)),
        tribe_vertices=20484, observed_vertices=20484,
        hemi_order_verified=True, movie_duration_sec=600.0,
        stimulus_expected_sec=600.0, tr=0.8, n_volumes=750,
        shared_frames=100, usable_frames=95,
        correlations=np.array([0.2, 0.4, -0.1]),
        censoring_applied=True, settings=settings,
    )
    assert report.valid, report.failures
    assert report.verdict == "VALID"
