"""Unit tests for statistics, caching, config hashing and ROI aggregation."""

from __future__ import annotations

import numpy as np
import pytest

from neurotribe.analysis import statistics as stats
from neurotribe.analysis.roi import aggregate
from neurotribe.config import stable_hash
from neurotribe.hashing import cache_key, hash_json, hash_many
from neurotribe.numerics import is_flat_sd, safe_zscore
from neurotribe.preprocessing.surfaces import Parcellation


# ------------------------------------------------------------------ FDR

def test_fdr_never_decreases_a_p_value():
    p = np.array([0.001, 0.01, 0.02, 0.04, 0.2, 0.9])
    q = stats.fdr_bh(p)
    assert np.all(q >= p - 1e-12)
    assert np.all(q <= 1.0)


def test_fdr_is_monotone_in_p():
    p = np.array([0.001, 0.008, 0.02, 0.3, 0.7])
    q = stats.fdr_bh(p)
    assert np.all(np.diff(q) >= -1e-12), q


def test_fdr_matches_the_known_benjamini_hochberg_result():
    # Worked example: n = 4.
    p = np.array([0.01, 0.02, 0.03, 0.04])
    q = stats.fdr_bh(p)
    # raw * n/rank = [0.04, 0.04, 0.04, 0.04] after monotonicity enforcement.
    assert np.allclose(q, 0.04, atol=1e-12)


def test_fdr_handles_nan_without_corrupting_others():
    p = np.array([0.01, np.nan, 0.5])
    q = stats.fdr_bh(p)
    assert np.isnan(q[1])
    assert np.isfinite(q[0]) and np.isfinite(q[2])
    assert q[0] == pytest.approx(0.02)


def test_fdr_on_all_nan_returns_all_nan():
    assert np.all(np.isnan(stats.fdr_bh(np.array([np.nan, np.nan]))))


# ------------------------------------------------------------------ OLS

def test_ols_recovers_a_known_coefficient():
    rng = np.random.default_rng(0)
    n = 200
    adhd = np.array([1] * 100 + [0] * 100)
    age = rng.normal(11, 2, n)

    true_effect = 0.8
    y = 2.0 + true_effect * adhd + 0.35 * age + rng.normal(0, 0.25, n)

    design = stats.build_design(adhd, {"age": age.tolist()})
    fit = stats.ols(y, design)
    term = fit.term("adhd")

    assert term["beta"] == pytest.approx(true_effect, abs=0.12)
    assert term["p"] < 1e-6
    assert fit.r_squared > 0.85
    assert not fit.rank_deficient


def test_ols_reports_a_null_effect_as_non_significant():
    rng = np.random.default_rng(3)
    n = 200
    adhd = np.array([1] * 100 + [0] * 100)
    y = rng.normal(0, 1, n)          # no group effect at all

    design = stats.build_design(adhd, {})
    fit = stats.ols(y, design)
    assert fit.term("adhd")["p"] > 0.05


def test_ols_refuses_a_model_with_no_residual_degrees_of_freedom():
    design = stats.DesignMatrix(matrix=np.eye(3), names=["a", "b", "c"])
    with pytest.raises(ValueError, match="degrees of freedom"):
        stats.ols(np.array([1.0, 2.0, 3.0]), design)


# ------------------------------------------------------------------ design

def test_build_design_dummy_codes_categorical_covariates():
    design = stats.build_design(
        [1, 1, 0, 0],
        {"site": ["RU", "CBIC", "RU", "CBIC"], "age": [10, 11, 12, 13]},
    )
    assert "intercept" in design.names
    assert "adhd" in design.names
    assert "site[RU]" in design.names        # CBIC is the reference level
    assert "age" in design.names
    assert design.matrix.shape[0] == 4


def test_build_design_drops_single_level_factors():
    design = stats.build_design([1, 1, 0, 0], {"site": ["RU", "RU", "RU", "RU"]})
    assert "site" in design.dropped
    assert any("fewer than two levels" in note for note in design.notes)


def test_build_design_drops_constant_numeric_covariates():
    design = stats.build_design([1, 1, 0, 0], {"age": [10, 10, 10, 10]})
    assert "age" in design.dropped


def test_build_design_flags_imputed_missing_values():
    """Imputation must be visible, never silent."""
    design = stats.build_design([1, 1, 0, 0], {"mean_fd": [0.1, None, 0.3, 0.4]})
    assert any("mean-imputed" in note for note in design.notes)
    assert "mean_fd_missing" in design.names


def test_build_design_rejects_wrong_length_covariates():
    with pytest.raises(ValueError, match="expected 4"):
        stats.build_design([1, 1, 0, 0], {"age": [10, 11]})


# ------------------------------------------------------------------ effects

def test_cohens_d_sign_and_magnitude():
    case = np.array([2.0, 2.1, 1.9, 2.0, 2.05])
    control = np.array([1.0, 1.1, 0.9, 1.0, 1.05])
    d = stats.cohens_d(case, control)
    assert d is not None and d > 5.0        # separated by ~1 with tiny SD


def test_cohens_d_is_zero_for_identical_groups():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert stats.cohens_d(values, values.copy()) == pytest.approx(0.0, abs=1e-9)


def test_cohens_d_returns_none_for_tiny_samples():
    assert stats.cohens_d(np.array([1.0]), np.array([2.0, 3.0])) is None


def test_bootstrap_ci_brackets_the_true_difference():
    rng = np.random.default_rng(11)
    case = rng.normal(1.0, 0.3, 120)
    control = rng.normal(0.0, 0.3, 120)
    low, high = stats.bootstrap_ci(case, control, n_boot=800)
    assert low is not None and high is not None
    assert low < 1.0 < high
    assert low > 0                        # a real, clearly non-zero difference


def test_coefficient_ci_brackets_the_estimate():
    low, high = stats.coefficient_ci(beta=0.5, se=0.1, df=100)
    assert low < 0.5 < high
    assert high - low == pytest.approx(2 * 1.984 * 0.1, rel=0.05)


# ------------------------------------------------------------------ numerics

def test_safe_zscore_zeroes_constant_float32_columns():
    """The float32 flatness bug: a constant column must become exactly zero."""
    data = np.full((50, 3), 3.14, dtype=np.float32)
    data[:, 1] = np.linspace(0, 1, 50, dtype=np.float32)

    z, flat = safe_zscore(data)
    assert np.allclose(z[:, 0], 0.0)
    assert np.allclose(z[:, 2], 0.0)
    assert flat.tolist() == [True, False, True]
    assert abs(float(z[:, 1].mean())) < 1e-5


def test_safe_zscore_handles_large_constant_values():
    """A large constant has a large absolute rounding error; still flat."""
    data = np.full((40, 2), 10_000.0, dtype=np.float32)
    data[:, 1] = np.arange(40, dtype=np.float32)
    z, flat = safe_zscore(data)
    assert flat[0] and not flat[1]
    assert np.allclose(z[:, 0], 0.0)


def test_is_flat_sd_uses_a_relative_tolerance():
    assert is_flat_sd(np.array([1e-4]), np.array([1000.0]))[0]
    assert not is_flat_sd(np.array([1.0]), np.array([1000.0]))[0]


# ------------------------------------------------------------------ caching

def test_cache_key_is_deterministic_and_order_independent():
    a = cache_key("tribe", model="x", sha="abc", config={"b": 2, "a": 1})
    b = cache_key("tribe", sha="abc", model="x", config={"a": 1, "b": 2})
    assert a == b
    assert a.startswith("tribe:")


def test_cache_key_changes_when_any_input_changes():
    base = cache_key("tribe", model="x", sha="abc")
    assert cache_key("tribe", model="y", sha="abc") != base
    assert cache_key("tribe", model="x", sha="def") != base
    assert cache_key("fmriprep", model="x", sha="abc") != base


def test_hash_many_is_order_independent():
    assert hash_many(["a", "b", "c"]) == hash_many(["c", "a", "b"])


def test_hash_json_is_stable_across_key_order():
    assert hash_json({"a": 1, "b": 2}) == hash_json({"b": 2, "a": 1})


def test_analysis_config_hash_tracks_scientific_parameters(settings):
    """Changing a scientific parameter must invalidate the config hash."""
    from neurotribe.config import Settings

    baseline = settings.analysis_config_hash

    raw = settings.raw
    raw["qc"]["motion"]["fd_threshold_mm"] = 0.9
    changed = Settings(raw, profile=settings.profile, root=settings.root)
    assert changed.analysis_config_hash != baseline

    # A cosmetic change must NOT invalidate it.
    raw2 = settings.raw
    raw2["logging"]["level"] = "DEBUG"
    cosmetic = Settings(raw2, profile=settings.profile, root=settings.root)
    assert cosmetic.analysis_config_hash == baseline


def test_stable_hash_is_length_bounded():
    assert len(stable_hash({"a": 1}, length=16)) == 16


# ------------------------------------------------------------------ ROI

def _toy_parcellation(n_vertices: int = 40) -> Parcellation:
    labels = np.zeros(n_vertices, dtype=np.int32)
    labels[:10] = 1
    labels[10:20] = 2
    labels[20:30] = 3
    # The final 10 vertices stay unassigned (medial wall).
    return Parcellation(
        labels=labels,
        names={1: "L_A_Visual", 2: "L_B_Default", 3: "R_C_Visual"},
        networks={1: "Visual", 2: "Default", 3: "Visual"},
        hemispheres={1: "L", 2: "L", 3: "R"},
        source="test", n_parcels=3,
    )


def test_roi_aggregation_averages_within_parcels(settings):
    parcellation = _toy_parcellation()
    r = np.zeros(40, dtype=np.float32)
    r[:10] = 0.5
    r[10:20] = 0.1
    r[20:30] = 0.9
    r[30:] = np.nan          # unassigned vertices

    mad = np.abs(r)
    variance = mad * 2

    result = aggregate(r, mad, variance, parcellation, settings)
    by_name = {roi.roi_name: roi for roi in result.rois}

    assert by_name["L_A_Visual"].agreement_r == pytest.approx(0.5)
    assert by_name["L_B_Default"].agreement_r == pytest.approx(0.1)
    assert by_name["R_C_Visual"].agreement_r == pytest.approx(0.9)
    assert by_name["L_A_Visual"].n_vertices == 10

    networks = {n.network: n for n in result.networks}
    # Visual pools parcels 1 and 3: mean of 0.5 and 0.9.
    assert networks["Visual"].agreement_r == pytest.approx(0.7)
    assert networks["Visual"].n_vertices == 20


def test_roi_aggregation_ignores_nan_vertices(settings):
    parcellation = _toy_parcellation()
    values = np.full(40, np.nan, dtype=np.float32)
    values[:5] = 1.0                        # half of parcel 1 has data

    result = aggregate(values, values, values, parcellation, settings)
    by_name = {roi.roi_name: roi for roi in result.rois}
    assert by_name["L_A_Visual"].agreement_r == pytest.approx(1.0)
    assert by_name["L_B_Default"].agreement_r is None


def test_roi_aggregation_rejects_mismatched_vertex_counts(settings):
    parcellation = _toy_parcellation()
    with pytest.raises(ValueError, match="Vertex map has 39 entries"):
        aggregate(np.zeros(39), np.zeros(39), np.zeros(39), parcellation, settings)


def test_top_deviation_ordering(settings):
    parcellation = _toy_parcellation()
    r = np.zeros(40, dtype=np.float32)
    mad = np.zeros(40, dtype=np.float32)
    mad[:10] = 0.2
    mad[10:20] = 0.9
    mad[20:30] = 0.5

    result = aggregate(r, mad, mad, parcellation, settings)
    top = result.top_deviation_rois(3)
    assert [roi.roi_name for roi in top] == ["L_B_Default", "R_C_Visual", "L_A_Visual"]
