"""The sorted risk-set kernel against the definitions it replaces.

``survival/_cox_core.py`` computes every risk-set sum as a running total.
Each test below recomputes the same quantity from its definition, one
risk set at a time, on designs with tied times, strata and both tie
rules, and requires agreement to rounding.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.survival._cox_core import CoxKernel, dominance_counts, harrell_c
from statspai.survival.models import _km_table

# Rounding only: the two computations add the same terms in another order.
RTOL = 1e-10


def _design(n, seed, ties=0, n_strata=0):
    rng = np.random.default_rng(seed)
    X = np.column_stack(
        [rng.normal(size=n), rng.integers(0, 2, n), rng.normal(size=n)]
    ).astype(float)
    T = rng.exponential(np.exp(-0.5 * X[:, 0] + 0.3 * X[:, 1]))
    if ties:
        T = np.ceil(T * ties) / ties
    E = (rng.uniform(size=n) < 0.7).astype(float)
    S = rng.integers(0, n_strata, n) if n_strata else None
    return X, T, E, S


def _terms(beta, X, T, E, S, breslow):
    """Yield, for each Efron term, the risk set, the deaths and ``c``."""
    S = np.zeros(len(T), dtype=int) if S is None else S
    r = np.exp(X @ beta)
    for s in np.unique(S):
        in_s = S == s
        for t in np.unique(T[in_s & (E == 1)]):
            risk = np.flatnonzero(in_s & (T >= t))
            dead = np.flatnonzero(in_s & (T == t) & (E == 1))
            for ell in range(len(dead)):
                c = 0.0 if breslow else ell / len(dead)
                yield r, risk, dead, c


def _brute(beta, X, T, E, S, breslow):
    n, p = X.shape
    nll, score, info = 0.0, np.zeros(p), np.zeros((p, p))
    resid = np.zeros((n, p))
    for r, risk, dead, c in _terms(beta, X, T, E, S, breslow):
        w = r[risk] * np.where(np.isin(risk, dead), 1.0 - c, 1.0)
        s0 = w.sum()
        xbar = X[risk].T @ w / s0
        nll += np.log(s0) - (X[dead] @ beta).sum() / len(dead)
        score += X[dead].sum(axis=0) / len(dead) - xbar
        info += (X[risk].T * w) @ X[risk] / s0 - np.outer(xbar, xbar)
        resid[risk] -= (w / s0)[:, None] * (X[risk] - xbar)
        resid[dead] += (X[dead] - xbar) / len(dead)
    return nll, score, info, resid


@pytest.mark.parametrize("breslow", [False, True])
@pytest.mark.parametrize("ties", [0, 4, 1])
@pytest.mark.parametrize("n_strata", [0, 3, 40])
def test_kernel_matches_the_definition(breslow, ties, n_strata):
    X, T, E, S = _design(160, 7 + ties + n_strata, ties, n_strata)
    beta = np.array([0.4, -0.3, 0.1])
    nll, score, info, resid = _brute(beta, X, T, E, S, breslow)
    kern = CoxKernel(X, T, E, S, breslow=breslow)

    assert kern.neg_loglik(beta) == pytest.approx(nll, rel=RTOL)
    got_score, got_hess = kern.score_hessian(beta)
    np.testing.assert_allclose(got_score, score, rtol=RTOL, atol=1e-10)
    np.testing.assert_allclose(-got_hess, info, rtol=RTOL, atol=1e-10)
    got_resid = kern.score_residuals(beta)
    np.testing.assert_allclose(got_resid, resid, rtol=RTOL, atol=1e-11)
    # the residuals are a decomposition of the score
    np.testing.assert_allclose(got_resid.sum(axis=0), score, atol=1e-9)


def test_weighted_information_is_the_weighted_sum_over_event_times():
    X, T, E, S = _design(120, 3, ties=3, n_strata=2)
    beta = np.array([0.2, 0.1, -0.4])
    kern = CoxKernel(X, T, E, S, breslow=False)
    rs = kern.rs
    w = np.zeros(rs.n_groups)
    w[rs.ev] = np.sqrt(1.0 + rs.g_time[rs.ev])

    p = X.shape[1]
    want = np.zeros((p, p))
    for r, risk, dead, c in _terms(beta, X, T, E, S, False):
        wt = r[risk] * np.where(np.isin(risk, dead), 1.0 - c, 1.0)
        xbar = X[risk].T @ wt / wt.sum()
        v = (X[risk].T * wt) @ X[risk] / wt.sum() - np.outer(xbar, xbar)
        want += np.sqrt(1.0 + T[dead[0]]) * v
    np.testing.assert_allclose(kern.information(beta, w), want, rtol=RTOL)


@pytest.mark.parametrize("round_to", [None, 1, 0])
def test_harrell_c_matches_the_pair_count(round_to):
    rng = np.random.default_rng(11)
    n = 140
    T = rng.exponential(1.0, n)
    score = rng.normal(size=n)
    if round_to is not None:
        T = np.round(T, round_to)
        score = np.round(score, round_to)
    E = (rng.uniform(size=n) < 0.6).astype(float)

    conc = disc = tied = 0
    for i in np.flatnonzero(E == 1):
        for j in np.flatnonzero((T > T[i]) | ((T == T[i]) & (E == 0))):
            conc += score[i] > score[j]
            disc += score[i] < score[j]
            tied += score[i] == score[j]
    assert harrell_c(score, T, E) == (conc + 0.5 * tied) / (conc + disc + tied)


def test_dominance_counts_against_all_pairs():
    rng = np.random.default_rng(5)
    ip, iv = rng.integers(0, 50, 200), rng.integers(0, 12, 200).astype(float)
    qp, qv = rng.integers(0, 50, 90), rng.integers(0, 12, 90).astype(float)
    before = ip[:, None] < qp[None, :]
    less = int((before & (iv[:, None] < qv[None, :])).sum())
    equal = int((before & (iv[:, None] == qv[None, :])).sum())
    assert dominance_counts(ip, iv, qp, qv) == (less, equal)


@pytest.mark.parametrize("conf_type", ["plain", "log", "log-log"])
def test_km_table_is_the_product_limit_estimate(conf_type):
    rng = np.random.default_rng(2)
    T = np.round(rng.exponential(2.0, 90), 1)
    E = (rng.uniform(size=90) < 0.7).astype(float)
    T[T.argmax()], E[T.argmax()] = T.max() + 1.0, 1.0  # the curve reaches zero
    table = _km_table(T, E, 0.05, conf_type).iloc[1:].reset_index(drop=True)

    surv, var = 1.0, 0.0
    for k, t in enumerate(np.unique(T[E == 1])):
        n_risk = int((T >= t).sum())
        d = int(((T == t) & (E == 1)).sum())
        surv *= 1 - d / n_risk
        if n_risk > d:
            var += d / (n_risk * (n_risk - d))
        row = table.iloc[k]
        assert (row["time"], row["n_risk"], row["n_event"]) == (t, n_risk, d)
        assert row["n_censor"] == int(((T == t) & (E == 0)).sum())
        assert row["survival"] == pytest.approx(surv, rel=1e-13, abs=0)
        assert row["std_err"] == pytest.approx(surv * np.sqrt(var), rel=1e-13)
    last = table.iloc[-1]
    assert last["survival"] == 0.0
    if conf_type == "plain":
        assert (last["ci_lower"], last["ci_upper"]) == (0.0, 0.0)
    else:
        assert np.isnan(last["ci_lower"]) and np.isnan(last["ci_upper"])


def test_logrank_matches_the_hypergeometric_sums():
    rng = np.random.default_rng(9)
    n = 150
    df = pd.DataFrame(
        {
            "t": np.round(rng.exponential(1.0, n), 1),
            "e": (rng.uniform(size=n) < 0.7).astype(int),
            "g": rng.choice(["a", "b", "c"], n),
        }
    )
    got = sp.logrank_test(df, "t", "e", "g")
    T, E, G = df["t"].values, df["e"].values, df["g"].values
    groups = list(df["g"].unique())
    o_e = np.zeros(2)
    V = np.zeros((2, 2))
    for t in np.unique(T[E == 1]):
        n_t = (T >= t).sum()
        d_t = ((T == t) & (E == 1)).sum()
        n_g = np.array([((G == g) & (T >= t)).sum() for g in groups[:2]])
        d_g = np.array([((G == g) & (T == t) & (E == 1)).sum() for g in groups[:2]])
        o_e += d_g - n_g * d_t / n_t
        if n_t > 1:
            f = d_t * (n_t - d_t) / (n_t**2 * (n_t - 1))
            V += f * (np.diag(n_g * n_t) - np.outer(n_g, n_g))
    assert got["test_statistic"] == pytest.approx(
        o_e @ np.linalg.solve(V, o_e), rel=1e-11
    )


def test_cox_is_invariant_to_shifting_a_regressor():
    # A regressor with a large mean (a calendar year) used to overflow the
    # risk scores on the first Newton step.
    X, T, E, _ = _design(300, 21)
    df = pd.DataFrame({"t": T, "e": E, "a": X[:, 0], "b": X[:, 1]})
    base = sp.cox(data=df, duration="t", event="e", x=["a", "b"])
    far = sp.cox(
        data=df.assign(a=df["a"] + 2000.0), duration="t", event="e", x=["a", "b"]
    )
    np.testing.assert_allclose(far.params.values, base.params.values, rtol=1e-9)
    shifted = sp.cox(
        data=df.assign(a=df["a"] + 20.0), duration="t", event="e", x=["a", "b"]
    )
    np.testing.assert_allclose(shifted.params.values, base.params.values, rtol=1e-9)
    np.testing.assert_allclose(
        shifted.std_errors.values, base.std_errors.values, rtol=1e-9
    )
    # the baseline hazard is the one quantity that moves with the shift
    ratio = (
        shifted.baseline_hazard()["baseline_cumhaz"].iloc[-1]
        / base.baseline_hazard()["baseline_cumhaz"].iloc[-1]
    )
    assert ratio == pytest.approx(np.exp(-20.0 * base.params["a"]), rel=1e-7)


def test_cox_cost_does_not_grow_with_the_square_of_n():
    # 20,000 continuous times: seconds of margin over ~0.1 s, against two
    # minutes when each risk set was rebuilt from the data.
    X, T, E, _ = _design(20_000, 1)
    df = pd.DataFrame({"t": T, "e": E, "a": X[:, 0], "b": X[:, 1]})
    df["cl"] = np.arange(len(df)) % 50
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start = time.perf_counter()
        sp.cox(data=df, duration="t", event="e", x=["a", "b"], cluster="cl")
        sp.kaplan_meier(df, "t", "e", conf_type="log-log")
        elapsed = time.perf_counter() - start
    assert elapsed < 20.0
