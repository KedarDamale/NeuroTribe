"""Statistical primitives: covariate-adjusted models, FDR, effect sizes, CIs.

Specification section 35: never report a raw ``mean(ADHD) - mean(control)`` as
evidence about ADHD biology. Every group inference here goes through a model
that adjusts for age, sex, acquisition site and head motion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class DesignMatrix:
    matrix: np.ndarray                # (n_subjects, n_terms)
    names: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def n_subjects(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def n_terms(self) -> int:
        return int(self.matrix.shape[1])

    def index_of(self, name: str) -> int:
        return self.names.index(name)


def build_design(group: Sequence[int], covariates: dict[str, Sequence],
                 *, categorical: Sequence[str] = ("sex", "site", "scanner")) -> DesignMatrix:
    """Assemble ``[intercept, adhd, covariates...]`` with dummy-coded factors.

    Levels that appear in only one group, or that are constant, are dropped with
    an explicit note - they cannot be estimated and would make the design
    rank-deficient.
    """
    group_array = np.asarray(group, dtype=float).reshape(-1, 1)
    n = group_array.shape[0]
    columns = [np.ones((n, 1)), group_array]
    names = ["intercept", "adhd"]
    dropped: list[str] = []
    notes: list[str] = []

    for name, raw_values in covariates.items():
        values = list(raw_values)
        if len(values) != n:
            raise ValueError(f"Covariate '{name}' has {len(values)} values, expected {n}")

        if name in categorical:
            levels = sorted({str(v) for v in values if v is not None})
            if len(levels) < 2:
                dropped.append(name)
                notes.append(f"Covariate '{name}' has fewer than two levels; dropped.")
                continue
            # Reference level = first alphabetically; k-1 dummies.
            for level in levels[1:]:
                column = np.array([[1.0 if str(v) == level else 0.0] for v in values])
                if column.std() < 1e-12:
                    dropped.append(f"{name}[{level}]")
                    continue
                columns.append(column)
                names.append(f"{name}[{level}]")
            notes.append(f"Covariate '{name}' dummy-coded with reference level '{levels[0]}'.")
        else:
            numeric = np.array(
                [np.nan if v is None else float(v) for v in values], dtype=float
            ).reshape(-1, 1)
            if not np.isfinite(numeric).any():
                dropped.append(name)
                notes.append(f"Covariate '{name}' has no finite values; dropped.")
                continue
            if np.isnan(numeric).any():
                # Mean-impute and add a missingness indicator so the imputation
                # cannot masquerade as observed data.
                mean = float(np.nanmean(numeric))
                missing = np.isnan(numeric).astype(float)
                numeric = np.where(np.isnan(numeric), mean, numeric)
                notes.append(
                    f"Covariate '{name}': {int(missing.sum())} missing value(s) "
                    "mean-imputed with a missingness indicator."
                )
                if missing.std() > 1e-12:
                    columns.append(missing)
                    names.append(f"{name}_missing")
            if numeric.std() < 1e-12:
                dropped.append(name)
                notes.append(f"Covariate '{name}' is constant; dropped.")
                continue
            columns.append(numeric)
            names.append(name)

    return DesignMatrix(matrix=np.hstack(columns), names=names,
                        dropped=dropped, notes=notes)


@dataclass
class OLSResult:
    beta: np.ndarray
    se: np.ndarray
    t: np.ndarray
    p: np.ndarray
    residual_df: int
    r_squared: float
    names: list[str] = field(default_factory=list)
    rank_deficient: bool = False

    def term(self, name: str) -> dict:
        index = self.names.index(name)
        return {
            "beta": float(self.beta[index]), "se": float(self.se[index]),
            "t": float(self.t[index]), "p": float(self.p[index]),
        }


def _t_sf(t_values: np.ndarray, df: int) -> np.ndarray:
    """Two-sided p-values for a t statistic, with a normal fallback."""
    try:
        from scipy import stats

        return 2.0 * stats.t.sf(np.abs(t_values), df)
    except ImportError:
        from math import erfc, sqrt

        return np.array([erfc(abs(float(t)) / sqrt(2.0)) for t in np.atleast_1d(t_values)])


def ols(y: np.ndarray, design: DesignMatrix) -> OLSResult:
    """Ordinary least squares with rank-deficiency detection."""
    X = design.matrix
    y = np.asarray(y, dtype=float).ravel()
    if y.size != X.shape[0]:
        raise ValueError(f"y has {y.size} observations, design has {X.shape[0]}")

    rank = int(np.linalg.matrix_rank(X))
    residual_df = X.shape[0] - rank
    if residual_df <= 0:
        raise ValueError(
            f"Model has {X.shape[1]} terms for {X.shape[0]} subjects; no residual "
            "degrees of freedom remain."
        )

    beta, _residuals, _rank, _sv = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    residual = y - fitted
    sigma_squared = float(residual @ residual) / residual_df

    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma_squared, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_values = np.where(se > 1e-12, beta / se, 0.0)
    p_values = _t_sf(t_values, residual_df)

    total_ss = float(((y - y.mean()) ** 2).sum())
    residual_ss = float((residual ** 2).sum())
    r_squared = 1.0 - residual_ss / total_ss if total_ss > 1e-12 else 0.0

    return OLSResult(
        beta=beta, se=se, t=np.asarray(t_values), p=np.asarray(p_values),
        residual_df=residual_df, r_squared=r_squared, names=list(design.names),
        rank_deficient=rank < X.shape[1],
    )


def fdr_bh(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted q-values, NaN-safe and monotone."""
    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan)
    finite = np.isfinite(p)
    if not finite.any():
        return q

    values = p[finite]
    n = values.size
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    # Enforce monotonicity from the largest p downwards.
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    restored = np.empty(n)
    restored[order] = adjusted
    q[finite] = restored
    return q


def cohens_d(case: np.ndarray, control: np.ndarray) -> float | None:
    """Standardised mean difference with pooled SD."""
    case = np.asarray(case, dtype=float)
    control = np.asarray(control, dtype=float)
    case = case[np.isfinite(case)]
    control = control[np.isfinite(control)]
    if case.size < 2 or control.size < 2:
        return None
    pooled = ((case.size - 1) * case.var(ddof=1) + (control.size - 1) * control.var(ddof=1))
    pooled /= (case.size + control.size - 2)
    if pooled <= 0:
        return 0.0
    return float((case.mean() - control.mean()) / np.sqrt(pooled))


def bootstrap_ci(case: np.ndarray, control: np.ndarray, *, n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 20240101) -> tuple[float | None, float | None]:
    """Percentile bootstrap CI for the difference in means."""
    case = np.asarray(case, dtype=float)
    control = np.asarray(control, dtype=float)
    case = case[np.isfinite(case)]
    control = control[np.isfinite(control)]
    if case.size < 3 or control.size < 3:
        return None, None

    rng = np.random.default_rng(seed)
    differences = np.empty(n_boot)
    for index in range(n_boot):
        differences[index] = (
            rng.choice(case, case.size, replace=True).mean()
            - rng.choice(control, control.size, replace=True).mean()
        )
    low = float(np.percentile(differences, 100 * alpha / 2))
    high = float(np.percentile(differences, 100 * (1 - alpha / 2)))
    return low, high


def coefficient_ci(beta: float, se: float, df: int, alpha: float = 0.05
                   ) -> tuple[float, float]:
    """Wald confidence interval for a model coefficient."""
    try:
        from scipy import stats

        critical = float(stats.t.ppf(1 - alpha / 2, df))
    except ImportError:
        critical = 1.959963985 if df > 30 else 2.2
    return beta - critical * se, beta + critical * se
