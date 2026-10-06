"""``sp.kalman_filter`` / ``sp.statespace`` against KFAS and statsmodels.

Given the system matrices, filter, smoother and likelihood are
deterministic, so they are compared digit for digit on the committed
synthetic file ``_fixtures/statespace.csv`` with the system matrices in
``statespace_spec.json`` (``_generate_statespace_data.py``):

* R ``KFAS`` 1.6.0 (R 4.5.2), ``statespace_R.json``, written by
  ``_generate_statespace_R.R``;
* statsmodels ``KalmanSmoother``, run here on the same arrays.

Six models: a local level with a proper prior; an AR(2) plus noise from
its stationary distribution; two correlated observables with scattered
missing values and whole rows missing; a regression with random-walk
coefficients (time-varying ``G``); a model with singular ``Q`` and ``R``
and one observable seen every fourth date; and one with every system
matrix time-varying, which pins the timing convention.

Timing. Ours is ``X_t = F_t X_{t-1} + V_t`` with ``X_0 ~ (x0, P0)``. Both
references write ``alpha_{t+1} = T_t alpha_t + eta_t`` and start from the
first *predicted* state, so ``T_t = F_{t+1}``, ``Q_t^{ref} = Q_{t+1}``,
``a1 = F_1 x0`` and ``P1 = F_1 P0 F_1' + Q_1``.

Tolerance ``EXACT`` = 1e-9, relative to ``max(|reference|, 1)`` so that
entries that are exactly zero in one implementation and 1e-17 in the
other compare as equal. Observed gaps are 1e-11 or less.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from statspai.timeseries._statespace_core import stationary_cov
from statspai.timeseries.dlm import dlm
from statspai.timeseries.statespace import kalman_filter, statespace

FIX = Path(__file__).parent / "_fixtures"
EXACT = 1e-9
CASES = {
    "level": ["level"],
    "ar2": ["ar2"],
    "biv": ["biv1", "biv2"],
    "tvp": ["tvp_y"],
    "mixed": ["mixed1", "mixed2", "mixed3"],
    "tv": ["tv"],
}


def gap(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float).reshape(a.shape)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1.0)))


@pytest.fixture(scope="module")
def R() -> dict:
    return json.loads((FIX / "statespace_R.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.read_csv(FIX / "statespace.csv")


@pytest.fixture(scope="module")
def spec() -> dict:
    raw = json.loads((FIX / "statespace_spec.json").read_text(encoding="utf-8"))
    return {
        name: {k: (None if v is None else np.array(v, float)) for k, v in s.items()}
        for name, s in raw.items()
    }


def ours(spec: dict, df: pd.DataFrame, case: str):
    return kalman_filter(df[CASES[case]], **spec[case])


def statsmodels_reference(s: dict, y: np.ndarray):
    """The same model in statsmodels, by the mapping in the module docstring."""
    from statsmodels.tsa.statespace.kalman_smoother import KalmanSmoother

    T, n = y.shape
    m = s["F"].shape[-1]

    def stack(key: str, ndim: int, lead: bool) -> np.ndarray:
        arr = s[key]
        if arr.ndim == ndim:
            return arr
        if lead:  # transition into date t+1 sits at index t
            arr = np.concatenate([arr[1:], arr[-1:]])
        return np.moveaxis(arr, 0, -1)

    F1 = s["F"] if s["F"].ndim == 2 else s["F"][0]
    Q1 = s["Q"] if s["Q"].ndim == 2 else s["Q"][0]
    P0 = stationary_cov(F1, Q1) if s["P0"] is None else s["P0"]
    mod = KalmanSmoother(k_endog=n, k_states=m, k_posdef=m)
    mod.bind(np.ascontiguousarray(y))
    mod["design"] = stack("G", 2, False)
    mod["obs_intercept"] = stack("A", 1, False)
    mod["obs_cov"] = stack("R", 2, False)
    mod["transition"] = stack("F", 2, True)
    mod["selection"] = np.eye(m)
    mod["state_cov"] = stack("Q", 2, True)
    mod.initialize_known(F1 @ s["x0"], F1 @ P0 @ F1.T + Q1)
    return mod.smooth()


@pytest.mark.parametrize("case", list(CASES))
def test_states_and_likelihood_match_kfas(R, df, spec, case):
    out = ours(spec, df, case)
    ref = R[case]
    assert gap(out.loglik, ref["loglik"]) < EXACT
    assert gap(out.predicted_state, ref["predicted"]) < EXACT
    assert gap(out.predicted_cov, ref["P_pred"]) < EXACT
    assert gap(out.filtered_state, ref["filtered"]) < EXACT
    assert gap(out.filtered_cov, ref["P_filt"]) < EXACT
    assert gap(out.smoothed_state, ref["smoothed"]) < EXACT
    assert gap(out.smoothed_cov, ref["P_smooth"]) < EXACT


@pytest.mark.parametrize("case", ["level", "ar2", "tvp", "tv"])
def test_prediction_errors_match_kfas(R, df, spec, case):
    # KFAS reports joint prediction errors only for a single observable
    out = ours(spec, df, case)
    v = np.array([np.nan if x is None else x for x in R[case]["v"]])
    seen = np.isfinite(out.innovations[:, 0])
    assert np.array_equal(seen, np.isfinite(v))
    assert gap(out.innovations[seen, 0], v[seen]) < EXACT
    assert gap(out.innovations_cov[seen, 0, 0], np.array(R[case]["S"])[seen]) < EXACT


@pytest.mark.parametrize("case", list(CASES))
def test_everything_matches_statsmodels(df, spec, case):
    y = df[CASES[case]].to_numpy(float)
    out = ours(spec, df, case)
    ref = statsmodels_reference(spec[case], y)
    T = len(y)
    assert gap(out.loglik, ref.llf) < EXACT
    assert gap(out.loglik_obs, ref.llf_obs) < EXACT
    assert gap(out.predicted_state, ref.predicted_state.T[:T]) < EXACT
    assert (
        gap(out.predicted_cov, np.moveaxis(ref.predicted_state_cov, -1, 0)[:T]) < EXACT
    )
    assert gap(out.filtered_state, ref.filtered_state.T) < EXACT
    assert gap(out.filtered_cov, np.moveaxis(ref.filtered_state_cov, -1, 0)) < EXACT
    assert gap(out.smoothed_state, ref.smoothed_state.T) < EXACT
    assert gap(out.smoothed_cov, np.moveaxis(ref.smoothed_state_cov, -1, 0)) < EXACT
    # joint prediction errors, their covariance, and the Cholesky-standardised
    # errors, at the dates where every observable is seen (statsmodels fills
    # the rest by its own rule)
    full = np.all(np.isfinite(y), axis=1)
    assert gap(out.innovations[full], ref.forecasts_error.T[full]) < EXACT
    assert (
        gap(
            out.innovations_cov[full], np.moveaxis(ref.forecasts_error_cov, -1, 0)[full]
        )
        < EXACT
    )
    assert (
        gap(out.std_innovations[full], ref.standardized_forecasts_error.T[full]) < EXACT
    )


@pytest.mark.parametrize("case", ["ar2", "biv", "mixed"])
def test_forecasts_match_statsmodels(df, spec, case):
    y = df[CASES[case]].to_numpy(float)
    out = ours(spec, df, case)
    h = 6
    padded = np.vstack([y, np.full((h, y.shape[1]), np.nan)])
    ref = statsmodels_reference(spec[case], padded)
    fc = out.forecast(h)
    # appended missing rows make the reference's predictions h-step forecasts
    assert (
        gap(fc["state"].to_numpy(), ref.predicted_state.T[len(y) : len(y) + h]) < EXACT
    )
    assert (
        gap(
            fc["state_cov"],
            np.moveaxis(ref.predicted_state_cov, -1, 0)[len(y) : len(y) + h],
        )
        < EXACT
    )
    assert gap(fc["obs"].to_numpy(), ref.forecasts.T[len(y) :]) < EXACT
    assert (
        gap(fc["obs_cov"], np.moveaxis(ref.forecasts_error_cov, -1, 0)[len(y) :])
        < EXACT
    )


def test_time_varying_regression_matches_sp_dlm(df, spec):
    s = spec["tvp"]
    out = ours(spec, df, "tvp")
    frame = pd.DataFrame({"y": df["tvp_y"], "x": df["tvp_x"]})
    ref = dlm(
        "y ~ x", frame, obs_var=0.3, state_var=[0.05, 0.02], m0=s["x0"], C0=s["P0"]
    )
    # same recursion in a different module: rounding error only
    assert gap(out.loglik, ref.loglik) < 1e-12
    assert gap(out.filtered_state, ref.filtered[["Intercept", "x"]].to_numpy()) < 1e-11
    assert gap(out.smoothed_state, ref.smoothed[["Intercept", "x"]].to_numpy()) < 1e-11
    sd = np.sqrt(np.einsum("tii->ti", out.smoothed_cov))
    assert gap(sd, ref.smoothed[["Intercept_sd", "x_sd"]].to_numpy()) < 1e-10
    one_step = (s["G"][:, 0, :] * out.predicted_state).sum(axis=1)
    assert gap(one_step, ref.fitted.to_numpy()) < 1e-11


def test_maximum_likelihood_matches_statsmodels(df):
    """Mixed-frequency model (singular Q and R, missing data), 9 parameters.

    The structure of Neusser's quarterly-GDP example on simulated data.
    Both sides maximise the same exact likelihood (checked to 1e-9 at a
    common parameter vector), from the same start. The optimisers differ,
    so estimates are compared to 1e-5 and standard errors, each from its
    own finite differences, to 1e-4.
    """
    from statsmodels.tsa.statespace.mlemodel import MLEModel

    y = df[CASES["mixed"]].to_numpy(float)
    shift = np.eye(4, k=-1)

    def build(th: np.ndarray) -> dict:
        F = shift.copy()
        F[0, 0] = th[7]
        G = np.zeros((3, 4))
        G[0] = 0.25
        G[1, 0], G[2, 0] = th[3], th[4]
        return {
            "F": F,
            "G": G,
            "Q": np.diag([np.exp(th[8]), 0.0, 0.0, 0.0]),
            "R": np.diag([0.0, np.exp(th[5]), np.exp(th[6])]),
            "A": th[:3],
        }

    class Ref(MLEModel):
        def __init__(self, endog):
            super().__init__(endog, k_states=4, k_posdef=1)
            self["design", 0, :] = 0.25
            self["selection", 0, 0] = 1.0
            self["transition"] = shift.copy()
            self.initialize_stationary()

        def update(self, params, **kwargs):
            params = super().update(params, **kwargs)
            self["obs_intercept", :, 0] = params[:3]
            self["design", 1, 0] = params[3]
            self["design", 2, 0] = params[4]
            self["obs_cov", 1, 1] = np.exp(params[5])
            self["obs_cov", 2, 2] = np.exp(params[6])
            self["transition", 0, 0] = params[7]
            self["state_cov", 0, 0] = np.exp(params[8])

    start = np.array([0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.5, 0.0])
    ref_model = Ref(y)
    common = np.array([0.4, 0.8, -1.5, 1.7, -1.2, 0.3, 0.8, 0.7, -0.4])
    at_common = kalman_filter(y, **build(common)).loglik
    assert gap(at_common, ref_model.loglike(common)) < EXACT

    names = ["a1", "a2", "a3", "g2", "g3", "lr2", "lr3", "phi", "lq"]
    fit = statespace(y, build, start, param_names=names)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = ref_model.fit(start, method="bfgs", maxiter=2000, gtol=1e-9, disp=False)
        ref = ref_model.fit(
            ref.params, method="nm", maxiter=20000, xtol=1e-10, ftol=1e-13, disp=False
        )
    assert fit.converged
    assert fit.loglik >= ref.llf - 1e-8
    assert gap(fit.loglik, ref.llf) < 1e-9
    assert gap(fit.params.to_numpy(), ref.params) < 1e-5
    # standard errors at our estimate: numerical Hessian, outer product of
    # scores, and sandwich, each against statsmodels' estimator of the same
    # name (finite differences on both sides: 1e-4)
    for vce, cov_type in (
        ("hessian", "approx"),
        ("opg", "opg"),
        ("robust", "robust_approx"),
    ):
        mine = statespace(y, build, fit.params.to_numpy(), vce=vce)
        theirs = ref_model.smooth(fit.params.to_numpy(), cov_type=cov_type)
        assert gap(mine.se.to_numpy(), theirs.bse) < 1e-4, vce
    assert gap(fit.aic, ref.aic) < 1e-9
    assert gap(fit.bic, ref.bic) < 1e-9


# --- exact diffuse initial state ---------------------------------------------
#
# ``init='exact'`` / ``diffuse=`` against R ``KFAS`` 1.6.0 (``P1inf``;
# ``statespace_exact_R.json`` from ``_generate_statespace_exact_R.R``) and
# statsmodels' exact diffuse initialisation, on ``statespace_exact.csv``
# (``_generate_statespace_exact_data.py``). Six models: local level; local
# linear trend (two diffuse states); regression with random-walk
# coefficients and missing ``y`` inside the diffuse period; a diffuse random
# walk plus a stationary AR(1); two observables with correlated errors, two
# diffuse levels, a stationary component and missing values in the first
# periods; and every system matrix time-varying with ``det F_1 != 1``.
#
# What is compared, and from which date.
#
# * Smoothed states and covariances: every date.
# * Predicted and filtered moments: from the date the diffuse part of the
#   covariance is zero. Inside the diffuse period the filtered mean in a
#   direction that is not yet identified depends on which vector is called
#   diffuse. Ours is ``X_0``, so the first predicted state has
#   ``P_inf = F_1 P0_inf F_1'``; both references put ``P_inf = I`` on the
#   first predicted state. Where ``F_1 P0_inf F_1'`` is ``diag(diffuse)``
#   (level, tvp, mixed, biv) the two agree at every date and that is
#   asserted; in ``trend`` and ``tv`` they agree from the end of the diffuse
#   period (our early dates are checked against the large-variance limit in
#   ``tests/test_statespace.py``). The finite part of the covariance inside
#   the diffuse period is not compared: KFAS wants it zero on diffuse states.
# * Log-likelihood. With ``n_d = n_diffuse`` absorbed observations,
#       ours = statsmodels llf            - 0.5 log pdet(F_1 P0_inf F_1')
#       ours = KFAS logLik - 0.5 n_d log(2 pi) - 0.5 log pdet(F_1 P0_inf F_1')
#   KFAS drops the normal constant of the absorbed observations; the last
#   term is zero except in ``tv``, where it is ``log|det F_1|``. KFAS's
#   ``logLik(marginal = TRUE)`` adds a further model-dependent term (for the
#   local level, ``0.5 log T``) and is not offered here.
#
# Tolerance: ``EXACT`` (1e-9) against KFAS, observed 1e-12 or less. 1e-8
# against statsmodels: on the time-invariant level and trend models its
# filter differs from KFAS and from us by about 4e-10, on the others by
# 1e-15. On ``tv`` statsmodels' *smoothed* moments at the two dates inside
# the diffuse period differ from KFAS and from us by order one (KFAS and the
# generalised-least-squares form in ``tests/test_statespace.py`` agree with
# us to 1e-11), so they are compared from the end of the diffuse period.

XCASES = {
    "level": ["level"],
    "trend": ["trend"],
    "tvp": ["tvp_y"],
    "mixed": ["mixed"],
    "biv": ["biv1", "biv2"],
    "tv": ["tv"],
}
SAME_SPACE = ("level", "tvp", "mixed", "biv")


@pytest.fixture(scope="module")
def XR() -> dict:
    return json.loads((FIX / "statespace_exact_R.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def xdf() -> pd.DataFrame:
    return pd.read_csv(FIX / "statespace_exact.csv")


@pytest.fixture(scope="module")
def xspec() -> dict:
    text = (FIX / "statespace_exact_spec.json").read_text(encoding="utf-8")
    out = {}
    for name, s in json.loads(text).items():
        flags = s.pop("diffuse")
        out[name] = {k: np.array(v, float) for k, v in s.items()}
        out[name]["diffuse"] = flags
    return out


def first_slice(a: np.ndarray) -> np.ndarray:
    return a if a.ndim == 2 else a[0]


def diffuse_shift(s: dict) -> float:
    """``0.5 log pdet(F_1 P0_inf F_1')``."""
    F1 = first_slice(s["F"])
    M = F1 @ np.diag(np.array(s["diffuse"], float)) @ F1.T
    w = np.linalg.eigvalsh(M)
    return 0.5 * float(np.sum(np.log(w[w > 1e-12])))


def identified(out) -> int:
    """First date whose filtered covariance has no diffuse part."""
    return int(np.flatnonzero(~np.any(out.filtered_cov_inf, axis=(1, 2)))[0])


@pytest.mark.parametrize("case", list(XCASES))
def test_exact_diffuse_matches_kfas(XR, xdf, xspec, case):
    s = xspec[case]
    out = kalman_filter(xdf[XCASES[case]], **s)
    ref = XR[case]
    done = identified(out)
    # KFAS's d is the last date of its diffuse period, counted from one
    assert done + 1 == ref["d"]
    assert out.n_diffuse == sum(s["diffuse"])
    assert gap(out.smoothed_state, ref["smoothed"]) < EXACT
    assert gap(out.smoothed_cov, ref["P_smooth"]) < EXACT
    a = 0 if case in SAME_SPACE else done
    b = 0 if case in SAME_SPACE else done + 1
    assert gap(out.filtered_state[a:], np.array(ref["filtered"])[a:]) < EXACT
    assert gap(out.filtered_cov[done:], np.array(ref["P_filt"])[done:]) < EXACT
    assert gap(out.predicted_state[b:], np.array(ref["predicted"])[b:]) < EXACT
    c = done + 1
    assert gap(out.predicted_cov[c:], np.array(ref["P_pred"])[c:]) < EXACT
    if case in SAME_SPACE:
        k = len(ref["Pinf_pred"])
        assert gap(out.predicted_cov_inf[:k], ref["Pinf_pred"]) < EXACT
        assert not out.predicted_cov_inf[k:].any()
    const = 0.5 * out.n_diffuse * np.log(2 * np.pi) + diffuse_shift(s)
    assert gap(out.loglik, ref["loglik"] - const) < EXACT


def statsmodels_exact(s: dict, y: np.ndarray):
    """The same model in statsmodels with an exact diffuse initial state."""
    from statsmodels.tsa.statespace.initialization import Initialization
    from statsmodels.tsa.statespace.kalman_smoother import KalmanSmoother

    T, n = y.shape
    m = s["F"].shape[-1]

    def stack(key: str, ndim: int, lead: bool) -> np.ndarray:
        arr = s[key]
        if arr.ndim == ndim:
            return arr
        if lead:
            arr = np.concatenate([arr[1:], arr[-1:]])
        return np.moveaxis(arr, 0, -1)

    F1, Q1 = first_slice(s["F"]), first_slice(s["Q"])
    flags = np.array(s["diffuse"], bool)
    nd = int(flags.sum())
    assert flags[:nd].all()  # diffuse states come first in every fixture
    mod = KalmanSmoother(k_endog=n, k_states=m, k_posdef=m)
    mod.bind(np.ascontiguousarray(y))
    mod["design"] = stack("G", 2, False)
    mod["obs_intercept"] = stack("A", 1, False)
    mod["obs_cov"] = stack("R", 2, False)
    mod["transition"] = stack("F", 2, True)
    mod["selection"] = np.eye(m)
    mod["state_cov"] = stack("Q", 2, True)
    init = Initialization(m)
    init.set((0, nd), "diffuse")
    if nd < m:
        Fk, Qk = F1[nd:, nd:], Q1[nd:, nd:]
        P1 = Fk @ stationary_cov(Fk, Qk) @ Fk.T + Qk
        init.set((nd, m), "known", constant=(F1 @ s["x0"])[nd:], stationary_cov=P1)
    mod.initialize(init)
    return mod.smooth()


@pytest.mark.parametrize("case", list(XCASES))
def test_exact_diffuse_matches_statsmodels(xdf, xspec, case):
    s = xspec[case]
    y = xdf[XCASES[case]].to_numpy(float)
    T = len(y)
    out = kalman_filter(y, **s)
    ref = statsmodels_exact(s, y)
    tol = 1e-8
    done = identified(out)
    assert ref.nobs_diffuse == done + 1
    assert gap(out.loglik, ref.llf - diffuse_shift(s)) < tol
    a = 0 if case in SAME_SPACE else done
    b = 0 if case in SAME_SPACE else done + 1
    assert gap(out.filtered_state[a:], ref.filtered_state.T[a:]) < tol
    assert gap(out.predicted_state[b:], ref.predicted_state.T[b:T]) < tol
    Pf = np.moveaxis(ref.filtered_state_cov, -1, 0)
    Pp = np.moveaxis(ref.predicted_state_cov, -1, 0)[:T]
    assert gap(out.filtered_cov[done:], Pf[done:]) < tol
    assert gap(out.predicted_cov[done + 1 :], Pp[done + 1 :]) < tol
    if case in SAME_SPACE:
        assert gap(out.loglik_obs, ref.llf_obs) < tol
    c = done if case == "tv" else 0
    assert gap(out.smoothed_state[c:], ref.smoothed_state.T[c:]) < tol
    Ps = np.moveaxis(ref.smoothed_state_cov, -1, 0)
    assert gap(out.smoothed_cov[c:], Ps[c:]) < tol


@pytest.mark.parametrize("case", list(XCASES))
def test_large_variance_approximation_is_close_after_the_diffuse_period(
    xdf, xspec, case
):
    # The large-variance filter with the default kappa = 1e7 against the
    # exact one. The difference is of order 1 / kappa times the scale of
    # the problem. Observed on these fixtures (absolute): filtered states
    # 2e-8 to 8e-4, filtered covariances 2e-9 to 7e-3 (largest in `tv`),
    # smoothed states 2e-8 to 9e-6, likelihood after adding back
    # 0.5 n_d log(kappa) 1e-7 to 7e-6. The smoothed covariances of the
    # approximation are worse, up to 6e-3 in `tvp`: they are differences
    # of terms of order kappa. Hence the loose 1e-2 here.
    s = {k: v for k, v in xspec[case].items() if k != "diffuse"}
    flags = np.array(xspec[case]["diffuse"], bool)
    y = xdf[XCASES[case]].to_numpy(float)
    out = kalman_filter(y, diffuse=flags, **s)
    P0 = out.P0 + 1e7 * np.diag(flags.astype(float))
    approx = kalman_filter(y, P0=P0, **s)
    done = identified(out)
    assert gap(approx.filtered_state[done:], out.filtered_state[done:]) < 1e-2
    assert gap(approx.filtered_cov[done:], out.filtered_cov[done:]) < 1e-2
    assert gap(approx.smoothed_state, out.smoothed_state) < 1e-4
    assert gap(approx.smoothed_cov, out.smoothed_cov) < 1e-2
    shifted = approx.loglik + 0.5 * out.n_diffuse * np.log(1e7)
    assert abs(shifted - out.loglik) < 1e-4


def test_exact_diffuse_maximum_likelihood_matches_statsmodels(xdf):
    """Local linear trend, three variances, exact diffuse likelihood.

    The likelihood is compared at a common parameter vector (1e-8, the
    statsmodels filter tolerance noted above); estimates from different
    optimisers to 1e-4 and standard errors from separate finite
    differences to 1e-3.
    """
    from statsmodels.tsa.statespace.mlemodel import MLEModel

    y = xdf["trend"].to_numpy(float)

    def build(th: np.ndarray) -> dict:
        return {
            "F": [[1.0, 1.0], [0.0, 1.0]],
            "G": [1.0, 0.0],
            "Q": np.diag(np.exp(th[:2])),
            "R": np.exp(th[2]),
        }

    class Ref(MLEModel):
        def __init__(self, endog):
            super().__init__(endog, k_states=2, k_posdef=2)
            self["design", 0, 0] = 1.0
            self["transition"] = np.array([[1.0, 1.0], [0.0, 1.0]])
            self["selection"] = np.eye(2)
            self.ssm.initialize_diffuse()

        def update(self, params, **kwargs):
            params = super().update(params, **kwargs)
            self["state_cov", 0, 0] = np.exp(params[0])
            self["state_cov", 1, 1] = np.exp(params[1])
            self["obs_cov", 0, 0] = np.exp(params[2])

    ref_model = Ref(y)
    common = np.array([-1.2, -3.0, 0.1])
    at_common = kalman_filter(y, init="exact", **build(common)).loglik
    assert gap(at_common, ref_model.loglike(common)) < 1e-8

    start = np.array([-1.0, -2.0, 0.0])
    fit = statespace(y, build, start, init="exact")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = ref_model.fit(start, method="bfgs", maxiter=2000, gtol=1e-9, disp=False)
        ref = ref_model.fit(
            ref.params, method="nm", maxiter=20000, xtol=1e-10, ftol=1e-13, disp=False
        )
    assert fit.converged
    assert fit.loglik >= ref.llf - 1e-7
    assert gap(fit.loglik, ref.llf) < 1e-8
    assert gap(fit.params.to_numpy(), ref.params) < 1e-4
    theirs = ref_model.smooth(fit.params.to_numpy(), cov_type="approx")
    assert gap(fit.se.to_numpy(), theirs.bse) < 1e-3
