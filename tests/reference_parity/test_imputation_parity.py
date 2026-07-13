"""Analytical parity: sp.mice + sp.mi_estimate (Rubin's rules).

DGP: y = 1 + 2*x1 - 1.5*x2 + e with e ~ N(0,1), x2 correlated with x1,
n = 400, and ~25% MCAR missingness imposed on x1. Because the imputation
model (Bayesian linear regression, method='norm') is congenial with the
linear-normal DGP, the Rubin-pooled OLS coefficient on x1 must recover the
true value 2.0 up to sampling + imputation noise, which is exactly what the
pooled SE quantifies (Rubin 1987): we assert |b_hat - 2| < 4 * pooled SE.

Exact identities checked in addition:
- observed cells are untouched by imputation (cell-for-cell equality) and
  only originally-missing cells are filled;
- the pooled point estimate equals the arithmetic mean of per-imputation
  estimates (Rubin's Q-bar), to machine precision;
- pooled variance T = U_bar + (1 + 1/m) B >= within-imputation variance
  U_bar, componentwise;
- PMM ('pmm') imputations are donor draws, i.e. every imputed value is an
  element of the observed-value set of that column.

Analytical evidence tier (no external reference numbers needed: the
identities are algebraic and the recovery target is the known DGP slope).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

TRUE_B_X1 = 2.0
N = 400
M = 10


def _make_data(seed: int = 42):
    """Linear-normal DGP with MCAR missingness in x1."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=N)
    x2 = 0.4 * x1 + rng.normal(size=N)
    y = 1.0 + TRUE_B_X1 * x1 - 1.5 * x2 + rng.normal(size=N)
    df_full = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    df = df_full.copy()
    miss_idx = rng.choice(N, size=100, replace=False)  # 25% MCAR
    df.loc[miss_idx, "x1"] = np.nan
    return df_full, df, np.sort(miss_idx)


@pytest.fixture(scope="module")
def mice_fit():
    df_full, df, miss_idx = _make_data()
    res = sp.mice(df, m=M, max_iter=5, method="norm", seed=123)
    return {"full": df_full, "df": df, "miss_idx": miss_idx, "res": res}


@pytest.fixture(scope="module")
def pooled(mice_fit):
    return sp.mi_estimate(mice_fit["res"], sp.regress, formula="y ~ x1 + x2")


def test_observed_cells_untouched_and_missing_filled(mice_fit):
    # Exact identity: imputation may only write originally-missing cells.
    df = mice_fit["df"]
    miss_mask = df["x1"].isna().values
    for i in range(mice_fit["res"].n_imputations):
        comp = mice_fit["res"].complete(i)
        # Observed x1 cells identical to input (bitwise equality).
        np.testing.assert_array_equal(
            comp.loc[~miss_mask, "x1"].values, df.loc[~miss_mask, "x1"].values
        )
        # Fully observed columns are byte-for-byte unchanged.
        np.testing.assert_array_equal(comp["y"].values, df["y"].values)
        np.testing.assert_array_equal(comp["x2"].values, df["x2"].values)
        # All originally-missing cells filled, no NaN remains.
        assert comp["x1"].notna().all()


def test_pooled_coefficient_recovers_truth(pooled):
    # Congenial norm-imputation + MCAR: pooled OLS slope on x1 is a
    # consistent estimate of 2.0. The pooled SE (Rubin's T) is precisely
    # the sampling+imputation uncertainty, so a 4-SE band is a ~1e-4
    # false-failure test under correct specification. A 0.3 absolute
    # cap guards against a degenerate (near-zero) SE.
    j = pooled["var_names"].index("x1")
    b, se = float(pooled["params"][j]), float(pooled["se"][j])
    assert abs(b - TRUE_B_X1) < max(4.0 * se, 1e-6)
    assert abs(b - TRUE_B_X1) < 0.3
    # And the pooled slope must be strongly significant.
    assert pooled["pvalues"][j] < 1e-6


def test_pooled_estimate_is_mean_of_per_imputation_estimates(mice_fit, pooled):
    # Rubin's rules: Q_bar = (1/m) sum_i Q_i, an exact algebraic identity.
    res = mice_fit["res"]
    params = np.array(
        [
            sp.regress("y ~ x1 + x2", data=res.complete(i)).params.values
            for i in range(res.n_imputations)
        ]
    )
    np.testing.assert_allclose(pooled["params"], params.mean(axis=0), atol=1e-12)


def test_total_variance_dominates_within_variance(mice_fit, pooled):
    # T = U_bar + (1 + 1/m) B with B PSD, so diag(T) >= diag(U_bar).
    res = mice_fit["res"]
    within = np.array(
        [
            sp.regress("y ~ x1 + x2", data=res.complete(i)).std_errors.values ** 2
            for i in range(res.n_imputations)
        ]
    ).mean(axis=0)
    total = np.asarray(pooled["se"], dtype=float) ** 2
    assert np.all(total >= within - 1e-12)
    # Missingness in x1 must create genuine between-imputation variance
    # for its coefficient (strict inequality, not just >=).
    j = pooled["var_names"].index("x1")
    assert total[j] > within[j]
    # FMI is a fraction in [0, 1].
    fmi = np.asarray(pooled["fmi"], dtype=float)
    assert np.all((fmi >= 0.0) & (fmi <= 1.0))


def test_pmm_imputations_are_donor_values(mice_fit):
    # PMM draws each imputed value from observed donors, so every imputed
    # cell must be an element of the observed x1 values (exact identity).
    df = mice_fit["df"]
    miss_mask = df["x1"].isna().values
    observed = set(df.loc[~miss_mask, "x1"].values)
    res_pmm = sp.mice(df, m=3, max_iter=3, method="pmm", seed=7)
    for i in range(res_pmm.n_imputations):
        imputed = res_pmm.complete(i).loc[miss_mask, "x1"].values
        assert all(v in observed for v in imputed)


def test_mice_bookkeeping(mice_fit):
    res = mice_fit["res"]
    assert res.n_imputations == M
    assert res.n_obs == N
    assert res.variables_imputed == ["x1"]
    assert res.n_missing["x1"] == 100
