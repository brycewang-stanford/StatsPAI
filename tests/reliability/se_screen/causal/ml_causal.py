"""SE screen: meta-learners, CATE tests and double machine learning.

Design constants (fixed before the first run, 2026-11-13):

* ``N = 1000`` (``_dgp.obs`` / ``_dgp.rct``). Rows with parametric
  learners (OLS, unpenalised logit) put the estimator in the regime its
  theory covers exactly, so a miss there is a formula problem. Rows with
  the default learners show what the default call delivers at this ``n``.
* ``B = 400``; ``B = 200`` where a fit takes several seconds.
* The DML-PLR default row and the SDID row are already explained in
  ``tests/coverage_monte_carlo/FINDINGS.md`` and are not repeated.

Run: ``PYTHONPATH=src python tests/reliability/se_screen/causal/ml_causal.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _dgp import OBS_TRUTH, obs, obs_clustered, rct  # noqa: E402
from _harness import Row, ci_from, main  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.special import expit  # noqa: E402

import statspai as sp  # noqa: E402

N = 1000
X = ["x1", "x2"]
ATE, ATT = OBS_TRUTH["ATE"], OBS_TRUTH["ATT"]
# E[x1 | x1 above its upper quartile] for a standard normal
Q75 = stats.norm.ppf(0.75)
TOP_MEAN = stats.norm.pdf(Q75) / 0.25
GATE_TOP = 1 + 0.5 * TOP_MEAN
GATE_DIFF = 0.5 * 2 * TOP_MEAN


def _ols():
    from sklearn.linear_model import LinearRegression

    return LinearRegression()


def _logit():
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(C=1e6)


def _metalearner(linear=False, cluster=False):
    def fit(rng):
        df = obs_clustered(rng, 100, 10) if cluster else obs(rng, N)
        kw = {}
        if linear:
            kw.update(
                outcome_model=_ols(), propensity_model=_logit(), cate_model=_ols()
            )
        if cluster:
            kw["cluster"] = "g"
        return ci_from(sp.metalearner(df, "y", "d", X, **kw))

    return fit


def _cddf(which):
    """GATES and BLP in a Bernoulli(0.5) experiment with known propensity."""

    def fit(rng):
        df = rct(rng, N)
        df["p"] = 0.5
        m = sp.metalearner(
            df,
            "y",
            "d",
            X,
            outcome_model=_ols(),
            propensity_model=_logit(),
            cate_model=_ols(),
        )
        seed = int(rng.integers(2**31))
        if which == "blp":
            b = sp.blp_test(m, df, "y", "d", X, propensity="p", seed=seed)
            lo, hi = b["beta1_ci"]
            return {"est": b["beta1"], "se": b["beta1_se"], "lo": lo, "hi": hi}
        g = sp.gate_test(
            m, df, by="x1", y="y", treat="d", covariates=X, propensity="p", seed=seed
        )
        if which == "gate_top":
            row = g["gate_table"].iloc[-1]
            return {
                "est": row["gate"],
                "se": row["se"],
                "lo": row["ci_lower"],
                "hi": row["ci_upper"],
            }
        est, se = g["top_vs_bottom_diff"], g["top_vs_bottom_se"]
        z = stats.norm.ppf(0.975)
        return {"est": est, "se": se, "lo": est - z * se, "hi": est + z * se}

    return fit


def _dml(model, hetero=True, default_learners=False, cluster=False, **kw):
    def fit(rng):
        if cluster:
            df = obs_clustered(rng, 100, 10)
            kw["cluster"] = "g"
        else:
            df = obs(rng, N, hetero=hetero)
        if not default_learners:
            kw["model_y"] = _ols()
            kw["model_d"] = _logit() if model == "irm" else _ols()
        kw["random_state"] = int(rng.integers(2**31))
        return ci_from(sp.dml(df, "y", "d", X, model=model, **kw))

    return fit


def _plr_cluster(rng):
    """Continuous treatment, constant effect 1, cluster-correlated errors."""
    G, m = 100, 10
    g = np.repeat(np.arange(G), m)
    n = G * m
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    d = 0.5 * x1 + np.sqrt(0.5) * rng.normal(size=G)[g] + rng.normal(size=n)
    y = d + x1 + 0.5 * x2 + np.sqrt(0.5) * rng.normal(size=G)[g] + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2, "g": g})
    r = sp.dml(
        df,
        "y",
        "d",
        X,
        model="plr",
        model_y=_ols(),
        model_d=_ols(),
        cluster="g",
        random_state=int(rng.integers(2**31)),
    )
    return ci_from(r)


def _pliv(rng):
    """Endogenous continuous treatment, one instrument, effect 1."""
    n = N
    x1, x2, z = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    u = rng.normal(size=n)
    d = 0.8 * z + 0.5 * x1 + 0.6 * u + rng.normal(size=n)
    y = d + x1 + 0.5 * x2 + u + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2, "z": z})
    r = sp.dml(
        df,
        "y",
        "d",
        X,
        model="pliv",
        instrument="z",
        model_y=_ols(),
        model_d=_ols(),
        random_state=int(rng.integers(2**31)),
    )
    return ci_from(r)


def _iivm(rng):
    """Binary instrument with one-sided non-compliance; LATE = 1."""
    n = 2000
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    z = (rng.uniform(size=n) < expit(0.3 * x1)).astype(int)
    complier = rng.uniform(size=n) < expit(0.5 + 0.5 * x2)
    d = (z * complier).astype(int)
    y = 1 + x1 + 0.5 * x2 + 1.0 * d + 0.5 * complier + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2, "z": z})
    r = sp.dml(
        df,
        "y",
        "d",
        X,
        model="iivm",
        instrument="z",
        random_state=int(rng.integers(2**31)),
    )
    return ci_from(r)


def _dml_panel(rng):
    """Unit effects correlated with the treatment, AR(1) errors, effect 0.8."""
    units, T = 200, 5
    rows = []
    a = rng.normal(size=units)
    for i in range(units):
        e = rng.normal()
        for t in range(T):
            e = 0.5 * e + np.sqrt(0.75) * rng.normal()
            x1, x2 = rng.normal(), rng.normal()
            d = 0.4 * x1 + a[i] + rng.normal()
            y = 0.8 * d + x1 + 0.5 * x2 + a[i] + e
            rows.append((i, t, y, d, x1, x2))
    df = pd.DataFrame(rows, columns=["id", "t", "y", "d", "x1", "x2"])
    r = sp.dml_panel(
        df,
        "y",
        "d",
        X,
        unit="id",
        time="t",
        ml_g=_ols(),
        ml_m=_ols(),
        seed=int(rng.integers(2**31)),
    )
    return {
        "est": float(r.estimate),
        "se": float(r.se),
        "lo": float(r.ci_lower),
        "hi": float(r.ci_upper),
    }


def _dml_did(rng):
    """Two-period DiD in differences: trend depends on x, ATT = 2."""
    n = N
    x = rng.normal(size=(n, 2))
    d = rng.binomial(1, expit(0.5 * x[:, 0]))
    dy = 1.0 + x[:, 0] + 0.5 * x[:, 1] + 2.0 * d + rng.normal(size=n)
    df = pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "d": d, "dy": dy})
    r = sp.dml_did(
        df,
        "dy",
        "d",
        X,
        ml_g=_ols(),
        ml_m=_logit(),
        random_state=int(rng.integers(2**31)),
    )
    return ci_from(r)


def _dynamic(which):
    """Three periods; the state responds to the last treatment.

    ``w_t = 0.6 w_{t-1} + 0.7 d_{t-1} + e``, ``d_t = 0.8 w_t + 0.5 d_{t-1}
    + e``, ``y = sum_t (0.5 w_t + 0.4 d_t) + e``. Holding later treatments
    fixed, a unit of treatment at period 0 / 1 / 2 moves the outcome by
    0.4 + 0.5 (0.7 + 0.7 * 0.6) = 0.96, 0.4 + 0.5 * 0.7 = 0.75 and 0.4.
    """

    def fit(rng):
        n = 500
        w = rng.normal(size=n)
        prev = np.zeros(n)
        hist = []
        for period in range(3):
            if period:
                w = 0.6 * w + 0.7 * prev + rng.normal(size=n)
            d = 0.8 * w + 0.5 * prev + rng.normal(size=n)
            hist.append((w, d))
            prev = d
        out = sum(0.5 * w_ + 0.4 * d_ for w_, d_ in hist) + rng.normal(size=n)
        df = pd.concat(
            [
                pd.DataFrame({"id": np.arange(n), "t": p, "y": out, "d": d_, "w": w_})
                for p, (w_, d_) in enumerate(hist)
            ]
        )
        res = sp.dynamic_dml(
            df,
            y="y",
            treat="d",
            id="id",
            time="t",
            covariates=["w"],
            model_y=_ols(),
            model_t=_ols(),
            random_state=int(rng.integers(2**31)),
        )
        if which == "total":
            return {
                "est": float(res.estimate),
                "se": float(res.se),
                "lo": float(res.ci_lower),
                "hi": float(res.ci_upper),
            }
        row = res.periods.iloc[0]
        return {
            "est": float(row["estimate"]),
            "se": float(row["se"]),
            "lo": float(row["ci_low"]),
            "hi": float(row["ci_high"]),
        }

    return fit


def _dml_avg(rng):
    df = obs(rng, 600, hetero=False)
    r = sp.dml_model_averaging(df, "y", "d", X, seed=int(rng.integers(2**31)))
    return ci_from(r)


D = "_dgp.obs"
ROWS = [
    Row(
        "metalearner.ate",
        "metalearner",
        "sp.metalearner(df, y, d, X)",
        N,
        ATE,
        _metalearner(),
        B=200,
        design=D + " (default learners, B = 200)",
    ),
    Row(
        "metalearner.ate.linear",
        "metalearner",
        "sp.metalearner(..., outcome_model=OLS, propensity_model=logit, "
        "cate_model=OLS)",
        N,
        ATE,
        _metalearner(linear=True),
        design=D,
    ),
    Row(
        "metalearner.ate.cluster",
        "metalearner",
        "sp.metalearner(..., cluster='g', linear learners)",
        N,
        ATE,
        _metalearner(linear=True, cluster=True),
        design="_dgp.obs_clustered, 100 clusters of 10",
    ),
    Row(
        "blp_test.beta1",
        "blp_test",
        "sp.blp_test(m, df, y, d, X, propensity='p')['beta1']",
        N,
        ATE,
        _cddf("blp"),
        design="_dgp.rct with the known propensity 0.5",
    ),
    Row(
        "gate_test.top_group",
        "gate_test",
        "sp.gate_test(m, df, by='x1', y=, treat=, propensity='p'), top quartile",
        N,
        GATE_TOP,
        _cddf("gate_top"),
        design="_dgp.rct; groups are quartiles of x1, effect 1 + 0.5 x1",
    ),
    Row(
        "gate_test.top_vs_bottom",
        "gate_test",
        "sp.gate_test(...)['top_vs_bottom_diff'] +/- 1.96 se",
        N,
        GATE_DIFF,
        _cddf("gate_diff"),
        design="as gate_test.top_group",
    ),
    Row(
        "dml.irm.ate.linear",
        "dml",
        "sp.dml(..., model='irm', model_y=OLS, model_d=logit)",
        N,
        ATE,
        _dml("irm"),
        design=D,
    ),
    Row(
        "dml.irm.atte.linear",
        "dml",
        "sp.dml(..., model='irm', score='ATTE', model_y=OLS, model_d=logit)",
        N,
        ATT,
        _dml("irm", score="ATTE"),
        design=D,
    ),
    Row(
        "dml.irm.ate.normalize_ipw",
        "dml",
        "sp.dml(..., model='irm', normalize_ipw=True, linear learners)",
        N,
        ATE,
        _dml("irm", normalize_ipw=True),
        design=D,
    ),
    Row(
        "dml.irm.ate.n_rep",
        "dml",
        "sp.dml(..., model='irm', n_rep=5, linear learners)",
        N,
        ATE,
        _dml("irm", n_rep=5),
        design=D,
    ),
    Row(
        "dml.irm.ate.default",
        "dml",
        "sp.dml(df, y, d, X, model='irm')",
        N,
        ATE,
        _dml("irm", default_learners=True),
        design=D + " (default learners)",
    ),
    Row(
        "dml.irm.ate.cluster",
        "dml",
        "sp.dml(..., model='irm', cluster='g', linear learners)",
        N,
        ATE,
        _dml("irm", cluster=True),
        design="_dgp.obs_clustered, 100 clusters of 10",
    ),
    Row(
        "dml.plr.cluster",
        "dml",
        "sp.dml(..., model='plr', cluster='g', linear learners)",
        N,
        1.0,
        _plr_cluster,
        design="continuous treatment, constant effect, cluster components "
        "in treatment and error, 100 clusters of 10",
    ),
    Row(
        "dml.pliv",
        "dml",
        "sp.dml(..., model='pliv', instrument='z', linear learners)",
        N,
        1.0,
        _pliv,
        design="endogenous continuous treatment, first-stage coefficient 0.8",
    ),
    Row(
        "dml.iivm",
        "dml",
        "sp.dml(..., model='iivm', instrument='z')",
        2000,
        1.0,
        _iivm,
        B=200,
        design="binary instrument, one-sided non-compliance, about 60% "
        "compliers, LATE = 1 (n = 2000, default learners, B = 200)",
    ),
    Row(
        "dml_panel",
        "dml_panel",
        "sp.dml_panel(df, y, d, X, unit='id', time='t', OLS learners)",
        N,
        0.8,
        _dml_panel,
        design="200 units x 5 periods, unit effect in treatment and outcome, "
        "AR(1) errors with coefficient 0.5",
    ),
    Row(
        "dml_did",
        "dml_did",
        "sp.dml_did(df, dy, d, X, ml_g=OLS, ml_m=logit)",
        N,
        2.0,
        _dml_did,
        design="two periods in differences, trend linear in x, ATT = 2",
    ),
    Row(
        "dml_model_averaging",
        "dml_model_averaging",
        "sp.dml_model_averaging(df, y, d, X)",
        600,
        ATE,
        _dml_avg,
        B=200,
        design=D
        + " with a constant effect (partially linear target; n = 600, B = 200)",
    ),
    Row(
        "dynamic_dml.total",
        "dynamic_dml",
        "sp.dynamic_dml(df, y=, treat=, id=, time=, covariates=['w'], OLS learners)",
        500,
        0.96 + 0.75 + 0.4,
        _dynamic("total"),
        design="500 units x 3 periods, linear state dynamics (see _dynamic)",
    ),
    Row(
        "dynamic_dml.first_period",
        "dynamic_dml",
        "sp.dynamic_dml(...).periods.iloc[0]",
        500,
        0.96,
        _dynamic("first"),
        design="as dynamic_dml.total",
    ),
]

MECHANISMS: dict = {}

if __name__ == "__main__":
    main(__file__, ROWS, MECHANISMS)
