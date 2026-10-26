"""Option scan (Oct 2026) of the inference estimators: conley, front_door, suest.

Each test pins a defect found by the scan and fixed with it; the ground
truths are identities (a Conley covariance with no neighbour pairs is the
HC0 sandwich; rows no equation uses cannot change suest) or a known truth
(the conditional front-door functional of a population law).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import MethodIncompatibility


def _geo_frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            # distinct coordinates: no two rows are within 1 metre
            "lat": rng.uniform(30, 45, size=n),
            "lon": rng.uniform(-120, -100, size=n),
            "w": rng.uniform(0.2, 3.0, size=n),
            "g": np.repeat(np.arange(30), n // 30),
        }
    )
    df["y"] = 1.0 + 0.5 * df.x1 - 0.3 * df.x2 + rng.normal(size=n) * (1 + df.w)
    return df


# A cutoff below every pairwise distance leaves only the diagonal of the
# kernel: the Conley matrix is then the HC0 sandwich (no small-sample factor).
_NO_NEIGHBOURS_KM = 1e-6


def test_conley_weighted_fit_is_weighted_hc0_without_neighbours():
    df = _geo_frame()
    fit = sp.regress("y ~ x1 + x2", data=df, weights="w")
    out = sp.conley(fit, data=df, lat="lat", lon="lon", dist_cutoff=_NO_NEIGHBOURS_KM)

    X = np.column_stack([np.ones(len(df)), df.x1, df.x2])
    y = df.y.to_numpy()
    w = df.w.to_numpy()
    bread = np.linalg.inv(X.T @ (X * w[:, None]))
    b = bread @ (X.T @ (w * y))
    e = y - X @ b
    s = X * (w * e)[:, None]
    V = bread @ (s.T @ s) @ bread
    # identity: same estimator, so agreement to rounding
    np.testing.assert_allclose(
        out.std_errors.to_numpy(), np.sqrt(np.diag(V)), rtol=1e-8
    )


def test_conley_refuses_a_likelihood_fit():
    # The OLS bread inv(X'X) is not the Hessian of a logit / Poisson
    # likelihood; before the fix the logit SEs at a no-neighbour cutoff were
    # 0.0264 / 0.0230 / 0.0270 against a logit HC0 sandwich of several times
    # that. The likelihood fits are refused.
    df = _geo_frame(seed=1)
    rng = np.random.default_rng(11)
    p = 1.0 / (1.0 + np.exp(-(0.2 + 0.8 * df.x1 - 0.5 * df.x2)))
    df["d"] = (rng.uniform(size=len(df)) < p).astype(float)
    df["k"] = rng.poisson(np.exp(0.3 + 0.2 * df.x1))
    for fit in (
        sp.logit("d ~ x1 + x2", data=df),
        sp.probit("d ~ x1 + x2", data=df),
        sp.poisson("k ~ x1 + x2", data=df),
    ):
        with pytest.raises(MethodIncompatibility, match="likelihood"):
            sp.conley(fit, data=df, lat="lat", lon="lon", dist_cutoff=100)


def test_conley_after_a_gaussian_glm_equals_conley_after_ols():
    df = _geo_frame(seed=1)
    ols = sp.conley(sp.regress("y ~ x1 + x2", data=df), df, "lat", "lon", 300.0)
    glm = sp.conley(
        sp.glm("y ~ x1 + x2", data=df, family="gaussian"), df, "lat", "lon", 300.0
    )
    np.testing.assert_allclose(
        glm.std_errors.to_numpy(), ols.std_errors.to_numpy(), rtol=1e-8
    )


def test_conley_result_carries_its_own_covariance_and_df():
    df = _geo_frame(seed=2)
    fit = sp.regress("y ~ x1 + x2", data=df, cluster="g")
    out = sp.conley(fit, data=df, lat="lat", lon="lon", dist_cutoff=150)
    V = np.asarray(out.data_info["vcov"], dtype=float)
    np.testing.assert_allclose(
        np.asarray(out.data_info["var_cov"], dtype=float), V, rtol=1e-12
    )
    n, k = len(df), 3
    t = out.params.to_numpy() / out.std_errors.to_numpy()
    # the interval _finalise reports is t(n - k); the p-value must agree
    crit = (out.conf_int_upper - out.params).to_numpy() / out.std_errors.to_numpy()
    np.testing.assert_allclose(crit, stats.t.ppf(0.975, n - k), rtol=1e-10)
    np.testing.assert_allclose(
        np.asarray(out.pvalues, dtype=float),
        2 * stats.t.sf(np.abs(t), n - k),
        rtol=1e-10,
    )
    new = df.iloc[:5]
    pr = out.predict(new, what="confidence")
    Xn = np.column_stack([np.ones(5), new.x1, new.x2])
    np.testing.assert_allclose(
        pr["se"].to_numpy(), np.sqrt(np.einsum("ij,jk,ik->i", Xn, V, Xn)), rtol=1e-10
    )


def _front_door_population(n: int, seed: int) -> pd.DataFrame:
    """Binary X, D, M; every working model of sp.front_door is correct.

    E[Y | d, m, x] = alpha_d + beta_d m + x, alpha = (0, 1), beta = (1, 5);
    P(D=1 | X=0, 1) = 0.2, 0.8; P(M=1 | d, x) = 0.3, 0.4 (x=0), 0.1, 0.9 (x=1).
    The conditional front-door functional
    sum_x P(x) sum_m P(m|d,x) sum_d' P(d'|x) E[Y|d',m,x] gives the ATE
    0.7 * 0.1 * 1.8 + 0.3 * 0.8 * 4.2 = 1.134; with the marginal P(d') in
    place of P(d'|x) it would be 0.7812.
    """
    rng = np.random.default_rng(seed)
    x = (rng.uniform(size=n) < 0.3).astype(float)
    d = (rng.uniform(size=n) < np.where(x == 1, 0.8, 0.2)).astype(float)
    q = np.where(x == 0, np.where(d == 1, 0.4, 0.3), np.where(d == 1, 0.9, 0.1))
    m = (rng.uniform(size=n) < q).astype(float)
    alpha = np.where(d == 1, 1.0, 0.0)
    beta = np.where(d == 1, 5.0, 1.0)
    y = alpha + beta * m + x + rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "m": m, "x": x})


def test_front_door_covariates_use_conditional_treatment_probability():
    df = _front_door_population(n=200_000, seed=3)
    r = sp.front_door(
        df, y="y", treat="d", mediator="m", covariates=["x"], n_boot=5, seed=0
    )
    truth = 0.7 * 0.1 * 1.8 + 0.3 * 0.8 * 4.2  # 1.134
    # sampling SD of the estimate at n = 200k is ~0.01; the marginal-P(d')
    # functional is 0.35 away, so 0.05 separates the two by a wide margin
    assert abs(r.estimate - truth) < 0.05
