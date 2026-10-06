"""``tvp_var`` against KFAS, ``sp.dlm``, least squares and a second filter.

``method='kalman'``. Every equation is a Gaussian state space model, so
at given variances filter, smoother and likelihood are deterministic.
They are compared (a) with the R package KFAS on the committed synthetic
file ``_fixtures/tvp_var.csv`` (``_generate_tvp_var_data.py``; reference
``tvp_var_R.json``, R 4.5.2, KFAS 1.6.0, ``_generate_tvp_var_R.R``) and
(b) with ``sp.dlm`` fitted to the same equation, including the
maximum-likelihood variances.

``method='forgetting'``. No package reference is committed: the one R
implementation found (ConnectednessApproach 1.0.4, ``TVPVAR``) demeans
the data, has no intercept and uses an error-covariance recursion that
black-box probing could not pin down. The evidence is instead (c) a
second implementation of the recursions written here in covariance
(Kalman-gain) form, while the package runs the information form, and
(d) closed forms: ordinary least squares at ``lam = 1``, discounted
least squares at ``lam < 1``, discounted generalised least squares when
the error covariance moves, and the moving-average recursion of the
error covariance.

Tolerances. ``EXACT`` = 1e-9 relative wherever the prior is proper.
Under the diffuse prior ``C0 = 1e7`` a covariance of order 1e7 collapses
to order one in the first ``k`` dates and both sides lose digits there
(the same caveat as ``test_dlm_parity.py``): coefficients are compared
after date ``k`` at 1e-5, the likelihood at 1e-9.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.timeseries.tvp_var import tvp_var

FIX = Path(__file__).parent / "_fixtures"
EXACT = 1e-9


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.read_csv(FIX / "tvp_var.csv")


@pytest.fixture(scope="module")
def ref() -> dict:
    return json.loads((FIX / "tvp_var_R.json").read_text(encoding="utf-8"))


def _rel(a: np.ndarray, b: np.ndarray, floor: float = 1e-6) -> float:
    """Largest error relative to ``max(|b|, floor)``."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), floor)))


def _design(df: pd.DataFrame, lags: int) -> tuple[np.ndarray, np.ndarray]:
    d = df.to_numpy(dtype=float)
    n = len(d)
    X = np.column_stack([d[lags - j : n - j] for j in range(1, lags + 1)])
    return d[lags:], np.column_stack([X, np.ones(n - lags)])


# --------------------------------------------------------------------- #
#  kalman: KFAS
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("case,C0", [("proper", 4.0), ("diffuse", 1e7)])
def test_kalman_matches_kfas(df: pd.DataFrame, ref: dict, case: str, C0: float) -> None:
    fit = tvp_var(
        df,
        lags=ref["lags"],
        obs_var=ref["obs_var"],
        state_var=np.array(ref["state_var"]),
        C0=C0,
    )
    rows = np.array(ref["rows"]) - 1
    k = len(fit.terms)
    if case == "diffuse":
        rows_ok = rows > k  # the first k dates are dominated by the prior
        tol_coef, tol_var, tol_ll = 1e-5, 1e-6, EXACT
    else:
        rows_ok = np.ones(rows.size, dtype=bool)
        tol_coef = tol_var = tol_ll = EXACT  # observed <= 6e-12
    for i, eq in enumerate(ref[case]):
        name = fit.var_names[i]
        ll = fit.model_info["equation_loglik"][name]
        assert abs(ll - eq["loglik"]) <= tol_ll * abs(eq["loglik"])
        r = rows[rows_ok]
        for ours, theirs, tol in (
            (fit.coef_filtered, eq["filtered"], tol_coef),
            (fit.coef_smoothed, eq["smoothed"], tol_coef),
            (fit.se_filtered**2, eq["filtered_var"], tol_var),
            (fit.se_smoothed**2, eq["smoothed_var"], tol_var),
        ):
            assert _rel(ours[r, i], np.array(theirs)[rows_ok]) <= tol
    assert fit.loglik == pytest.approx(
        sum(eq["loglik"] for eq in ref[case]), rel=tol_ll
    )


# --------------------------------------------------------------------- #
#  kalman: each equation is sp.dlm
# --------------------------------------------------------------------- #


def _dlm_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({c: df[c].to_numpy()[1:] for c in df.columns})
    for c in df.columns:
        out[f"lag_{c}"] = df[c].to_numpy()[:-1]
    return out


def test_each_equation_is_sp_dlm_at_the_same_variances(df: pd.DataFrame) -> None:
    V = [1.1, 0.8, 0.6]
    W = np.array([[0.004, 0.0, 0.001, 0.02], [0.0, 0.003, 0.0, 0.0], [0.002] * 4])
    fit = tvp_var(df, lags=1, obs_var=V, state_var=W, C0=9.0, m0=0.1)
    frame = _dlm_frame(df)
    rhs = "lag_y1 + lag_y2 + lag_y3"
    for i, name in enumerate(fit.var_names):
        # dlm lists the intercept first, tvp_var last
        d = sp.dlm(
            f"{name} ~ {rhs}",
            frame,
            obs_var=V[i],
            state_var=np.r_[W[i, 3], W[i, :3]],
            m0=0.1,
            C0=9.0,
        )
        cols = ["lag_y1", "lag_y2", "lag_y3", "Intercept"]
        sd = [f"{c}_sd" for c in cols]
        # same recursions, proper prior: agreement to rounding
        assert _rel(fit.coef_smoothed[:, i], d.smoothed[cols]) <= EXACT
        assert _rel(fit.coef_filtered[:, i], d.filtered[cols]) <= EXACT
        assert _rel(fit.se_smoothed[:, i], d.smoothed[sd]) <= EXACT
        assert _rel(fit.se_filtered[:, i], d.filtered[sd]) <= EXACT
        ll = fit.model_info["equation_loglik"][name]
        assert ll == pytest.approx(d.loglik, rel=EXACT)


def test_maximum_likelihood_is_sp_dlm_equation_by_equation(df: pd.DataFrame) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = tvp_var(df, lags=1)
        frame = _dlm_frame(df)
        k = len(fit.terms)
        total = 0.0
        for i, name in enumerate(fit.var_names):
            d = sp.dlm(f"{name} ~ lag_y1 + lag_y2 + lag_y3", frame)
            est = d.variances["estimate"].to_numpy()
            ours = fit.variances.loc[name].to_numpy()
            # the variances are taken from sp.dlm: identical
            assert ours[0] == pytest.approx(est[0], rel=1e-12)
            np.testing.assert_allclose(ours[1:], np.r_[est[2:], est[1]], rtol=1e-12)
            cols = ["lag_y1", "lag_y2", "lag_y3", "Intercept"]
            # diffuse prior: compare after the first k dates (see module
            # docstring); the regressor order differs, so rounding does too
            assert (
                _rel(fit.coef_smoothed[k:, i], d.smoothed[cols].to_numpy()[k:]) <= 1e-6
            )
            assert (
                _rel(fit.coef_filtered[k:, i], d.filtered[cols].to_numpy()[k:]) <= 1e-6
            )
            total += d.loglik
        assert fit.loglik == pytest.approx(total, rel=EXACT)


# --------------------------------------------------------------------- #
#  forgetting: a second implementation, covariance form
# --------------------------------------------------------------------- #


def _reference_filter(
    Y: np.ndarray,
    X: np.ndarray,
    lam: float,
    kappa: float,
    b0: np.ndarray,
    P0: np.ndarray,
    S0: np.ndarray,
    update: str,
) -> dict:
    """Forgetting-factor filter in Kalman-gain form (Joseph update).

    State ``beta`` = rows of the coefficient matrix stacked;
    ``Z_t = I (x) x_t'``; ``P_{t|t-1} = P_{t-1|t-1} / lam``; the error
    covariance of date ``t - 1`` enters the update of date ``t`` and is
    then moved by ``S_t = kappa S_{t-1} + (1 - kappa) e_t e_t'``.
    """
    T, K = Y.shape
    k = X.shape[1]
    beta, P, S = b0.reshape(-1).copy(), P0.copy(), S0.copy()
    coef, se, sig = [], [], []
    ll = 0.0
    for t in range(T):
        Z = np.kron(np.eye(K), X[t][None, :])
        Pp = P / lam
        e = Y[t] - Z @ beta
        F = Z @ Pp @ Z.T + S
        Fi = np.linalg.inv(F)
        ll -= 0.5 * (K * np.log(2 * np.pi) + np.log(np.linalg.det(F)) + e @ Fi @ e)
        G = Pp @ Z.T @ Fi
        beta = beta + G @ e
        A = np.eye(K * k) - G @ Z
        P = A @ Pp @ A.T + G @ S @ G.T
        r = Y[t] - Z @ beta if update == "filtered" else e
        S = kappa * S + (1 - kappa) * np.outer(r, r)
        coef.append(beta.reshape(K, k).copy())
        se.append(np.sqrt(np.diag(P)).reshape(K, k))
        sig.append(S.copy())
    return {
        "coef": np.array(coef),
        "se": np.array(se),
        "sigma": np.array(sig),
        "loglik": ll,
    }


@pytest.mark.parametrize("update", ["filtered", "predicted"])
@pytest.mark.parametrize("lags", [1, 2])
def test_forgetting_matches_second_implementation(
    df: pd.DataFrame, update: str, lags: int
) -> None:
    lam, kappa = 0.97, 0.94
    fit = tvp_var(
        df,
        lags=lags,
        method="forgetting",
        lam=lam,
        kappa=kappa,
        prior="minnesota",
        prior_tightness=0.2,
        prior_own_lag=0.5,
        C0=4.0,
        sigma_update=update,
    )
    Y, X = _design(df, lags)
    K, k = 3, 3 * lags + 1
    b0 = np.zeros((K, k))
    b0[np.arange(K), np.arange(K)] = 0.5
    v = np.r_[np.repeat([0.2 / lag**2 for lag in range(1, lags + 1)], K), 4.0]
    P0 = np.diag(np.tile(v, K))
    resid = Y - X @ np.linalg.lstsq(X, Y, rcond=None)[0]
    S0 = resid.T @ resid / len(Y)
    want = _reference_filter(Y, X, lam, kappa, b0, P0, S0, update)
    # two algebraically equal recursions under a proper prior
    assert _rel(fit.coef_filtered, want["coef"]) <= 1e-8
    assert _rel(fit.se_filtered, want["se"]) <= 1e-8
    assert _rel(fit.sigma_t, want["sigma"]) <= 1e-8
    assert fit.loglik == pytest.approx(want["loglik"], rel=1e-10)
    np.testing.assert_allclose(fit.sigma.to_numpy(), want["sigma"][-1], rtol=1e-8)


# --------------------------------------------------------------------- #
#  forgetting: closed forms
# --------------------------------------------------------------------- #


def test_lam_one_is_ols_of_sp_var(df: pd.DataFrame) -> None:
    for lags in (1, 2):
        fit = tvp_var(df, lags=lags, method="forgetting", lam=1.0, kappa=1.0)
        v = sp.var(df, lags=lags)
        for i, name in enumerate(fit.var_names):
            ols = v.coefs[name]["coef"]
            assert list(ols.index) == fit.terms
            # recursive least squares from a prior of variance 1e7: the
            # prior moves the estimate by about 1e-7 / (X'X) ~ 1e-9
            np.testing.assert_allclose(
                fit.coef_filtered[-1, i], ols.to_numpy(), rtol=1e-8, atol=1e-8
            )
        # with kappa = 1 the error covariance stays at the OLS residual
        # covariance, which is sp.var's sigma_u (divisor T)
        np.testing.assert_allclose(
            fit.sigma.to_numpy(), v.sigma_u.to_numpy(), rtol=1e-10
        )


def test_lam_below_one_is_discounted_least_squares(df: pd.DataFrame) -> None:
    lam = 0.96
    fit = tvp_var(df, lags=2, method="forgetting", lam=lam, kappa=1.0)
    Y, X = _design(df, 2)
    for t in (40, 90, len(Y) - 1):
        w = lam ** (t - np.arange(t + 1))
        Xw = X[: t + 1] * w[:, None]
        B = np.linalg.solve(Xw.T @ X[: t + 1], Xw.T @ Y[: t + 1]).T
        # the diffuse prior has weight lam**t * 1e-7 against X'WX
        np.testing.assert_allclose(fit.coef_filtered[t], B, rtol=1e-8, atol=1e-8)


def test_moving_covariance_gives_discounted_gls(df: pd.DataFrame) -> None:
    lam, kappa = 0.97, 0.9
    fit = tvp_var(df, lags=1, method="forgetting", lam=lam, kappa=kappa, C0=50.0)
    Y, X = _design(df, 1)
    K, k = 3, 4
    S_used = np.concatenate([fit.model_info["sigma0"][None], fit.sigma_t[:-1]])
    t = len(Y) - 1
    Om = lam ** (t + 1) * np.eye(K * k) / 50.0
    h = np.zeros(K * k)
    for s in range(t + 1):
        Si = np.linalg.inv(S_used[s])
        Om += lam ** (t - s) * np.kron(Si, np.outer(X[s], X[s]))
        h += lam ** (t - s) * np.kron(Si @ Y[s], X[s])
    want = np.linalg.solve(Om, h).reshape(K, k)
    # sum form against the recursion: rounding only
    np.testing.assert_allclose(fit.coef_filtered[-1], want, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(
        fit.se_filtered[-1],
        np.sqrt(np.diag(np.linalg.inv(Om))).reshape(K, k),
        rtol=1e-9,
    )


@pytest.mark.parametrize("update", ["filtered", "predicted"])
def test_error_covariance_is_the_ewma_recursion(df: pd.DataFrame, update: str) -> None:
    kappa = 0.93
    fit = tvp_var(
        df,
        lags=1,
        method="forgetting",
        lam=0.98,
        kappa=kappa,
        C0=10.0,
        sigma_update=update,
    )
    Y, X = _design(df, 1)
    coef = fit.coef_filtered
    S = fit.model_info["sigma0"].copy()
    prev = np.zeros_like(coef[0])
    for t in range(len(Y)):
        B = coef[t] if update == "filtered" else prev
        e = Y[t] - B @ X[t]
        S = kappa * S + (1 - kappa) * np.outer(e, e)
        np.testing.assert_allclose(fit.sigma_t[t], S, rtol=1e-10, atol=1e-13)
        prev = coef[t]
