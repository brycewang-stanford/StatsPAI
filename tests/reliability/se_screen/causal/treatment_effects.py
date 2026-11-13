"""SE screen: IPW, AIPW, regression adjustment, TMLE and experiment estimators.

Design constants (fixed before the first run, 2026-11-13):

* ``N = 1000`` for every row unless the row says otherwise. The design in
  ``_dgp.obs`` has two covariates and propensities within about
  [0.1, 0.9], so first-order asymptotics should hold comfortably.
* ``B = 400`` draws per row; the slow rows (collaborative TMLE, learner
  propensities) use ``B = 200`` and say so.
* Truths: population ATE / ATT / ATC by Gauss-Hermite quadrature
  (``_dgp.OBS_TRUTH``).

Run: ``PYTHONPATH=src python tests/reliability/se_screen/causal/treatment_effects.py``
(``--only <row id or prefix>``, ``--confirm`` to rerun flagged rows at
B = 1000, ``--list``). Two worker processes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from _dgp import OBS_TRUTH, obs, obs_clustered, rct, sample_truth  # noqa: E402
from _harness import Row, ci_from, main  # noqa: E402

import statspai as sp  # noqa: E402

N = 1000
X = ["x1", "x2"]
ATE, ATT, ATC = OBS_TRUTH["ATE"], OBS_TRUTH["ATT"], OBS_TRUTH["ATC"]


def _obs_fit(fn, target="ATE", n=N, **kw):
    def fit(rng):
        df = obs(rng, n)
        out = ci_from(fn(df, "y", "d", X, **kw))
        out["truth_s"] = sample_truth(df, target)
        return out

    return fit


def _weighted_fit(fn, **kw):
    """Sampling weights independent of everything: the truth is unchanged."""

    def fit(rng):
        df = obs(rng, N)
        df["w"] = rng.uniform(0.5, 1.5, size=len(df))
        return ci_from(fn(df, "y", "d", X, weights="w", **kw))

    return fit


def _cluster_fit(fn, **kw):
    def fit(rng):
        df = obs_clustered(rng, 100, 10)
        return ci_from(fn(df, "y", "d", X, cluster="g", **kw))

    return fit


def _ipw_boot(rng):
    df = obs(rng, N)
    return ci_from(sp.ipw(df, "y", "d", X, seed=int(rng.integers(2**31))))


def _ipw_learner(rng):
    from sklearn.linear_model import LogisticRegression

    df = obs(rng, 600)
    r = sp.ipw(
        df,
        "y",
        "d",
        X,
        ps_learner=LogisticRegression(C=1e6),
        n_bootstrap=200,
        seed=int(rng.integers(2**31)),
    )
    return ci_from(r)


def _gcomp(estimand="ATE", hetero=True, **kw):
    def fit(rng):
        df = obs(rng, N, hetero=hetero)
        r = sp.g_computation(
            df, "y", "d", X, estimand=estimand, seed=int(rng.integers(2**31)), **kw
        )
        return ci_from(r)

    return fit


def _rct_fit(call):
    def fit(rng):
        df = rct(rng, N)
        out = ci_from(call(df))
        out["truth_s"] = float(df["tau"].mean())
        return out

    return fit


def _rct_blocks(rng):
    """Block-randomised: four blocks, treatment share 0.3 / 0.5 / 0.5 / 0.7."""
    df = rct(rng, N)
    b = rng.integers(0, 4, size=N)
    share = np.array([0.3, 0.5, 0.5, 0.7])[b]
    d = (rng.uniform(size=N) < share).astype(int)
    # block-specific level so ignoring the blocks would be biased
    y = 1 + df["x1"] + 0.5 * df["x2"] + 0.8 * b + df["tau"] * d + rng.normal(size=N)
    df = df.assign(d=d, y=y, block=b)
    return ci_from(sp.difference_in_means(df, "y", "d", blocks="block"))


def _rct_cluster(rng):
    """Cluster-randomised: 100 clusters of 10, intra-cluster correlation 1/3."""
    G, m = 100, 10
    g = np.repeat(np.arange(G), m)
    d = (rng.uniform(size=G) < 0.5).astype(int)[g]
    y = 1 + d + np.sqrt(0.5) * rng.normal(size=G)[g] + rng.normal(size=G * m)
    import pandas as pd

    df = pd.DataFrame({"y": y, "d": d, "g": g})
    return ci_from(sp.difference_in_means(df, "y", "d", cluster="g"))


def _lm_lin_cluster(rng):
    G, m = 100, 10
    g = np.repeat(np.arange(G), m)
    d = (rng.uniform(size=G) < 0.5).astype(int)[g]
    x1 = rng.normal(size=G * m)
    y = (
        1
        + x1
        + (1 + 0.5 * x1) * d
        + np.sqrt(0.5) * rng.normal(size=G)[g]
        + rng.normal(size=G * m)
    )
    import pandas as pd

    df = pd.DataFrame({"y": y, "d": d, "g": g, "x1": x1})
    return ci_from(sp.lm_lin(df, "y", "d", ["x1"], cluster="g"))


def _tmle(estimand="ATE", n=N, **kw):
    def fit(rng):
        df = obs(rng, n)
        r = sp.tmle(
            df,
            "y",
            "d",
            X,
            estimand=estimand,
            random_state=int(rng.integers(2**31)),
            **kw,
        )
        return ci_from(r)

    return fit


def _tmle_glm(estimand="ATE", cluster=False, **kw):
    """TMLE with one parametric learner per nuisance (no Super Learner)."""

    def fit(rng):
        from sklearn.linear_model import LinearRegression, LogisticRegression

        df = obs_clustered(rng, 100, 10) if cluster else obs(rng, N)
        if cluster:
            kw["cluster"] = "g"
        r = sp.tmle(
            df,
            "y",
            "d",
            X,
            estimand=estimand,
            outcome_library=[LinearRegression()],
            propensity_library=[LogisticRegression(C=1e6)],
            random_state=int(rng.integers(2**31)),
            **kw,
        )
        return ci_from(r)

    return fit


def _ctmle(rng):
    df = obs(rng, 500)
    return ci_from(sp.ctmle(df, "y", "d", X, random_state=int(rng.integers(2**31))))


def _hal(rng):
    df = obs(rng, N)
    return ci_from(sp.hal_tmle(df, "y", "d", X, random_state=int(rng.integers(2**31))))


def _ltmle(rng):
    """Two decision points; E[Y(1, 1)] - E[Y(0, 0)] = 1 + 1 + 0.5 = 2.5."""
    from scipy.special import expit

    n = N
    l1 = rng.normal(size=n)
    a1 = (rng.uniform(size=n) < expit(0.5 * l1)).astype(int)
    l2 = 0.5 * l1 + 0.5 * a1 + rng.normal(size=n)
    a2 = (rng.uniform(size=n) < expit(0.5 * l2 + 0.5 * a1 - 0.25)).astype(int)
    y = 1 + a1 + a2 + l2 + l1 + rng.normal(size=n)
    import pandas as pd

    df = pd.DataFrame({"y": y, "a1": a1, "a2": a2, "l1": l1, "l2": l2})
    r = sp.ltmle(df, "y", ["a1", "a2"], [["l1"], ["l2"]])
    lo, hi = r.ci
    return {"est": float(r.ate), "se": float(r.se), "lo": float(lo), "hi": float(hi)}


def _lmtp(which, **kw):
    """Shift a continuous exposure by one unit: A = W + v, Y = A + W + e."""

    def fit(rng):
        import pandas as pd

        n = N
        w = rng.normal(size=n)
        a = w + rng.normal(size=n)
        y = a + w + rng.normal(size=n)
        df = pd.DataFrame({"W": w, "A": a, "Y": y})
        r = sp.lmtp(df, "Y", ["A"], shift=1.0, baseline=["W"], **kw)
        if which == "mean":
            return ci_from(r)
        est, se = float(r.model_info["effect"]), float(r.model_info["effect_se"])
        return {
            "est": est,
            "se": se,
            "lo": est - 1.959964 * se,
            "hi": est + 1.959964 * se,
        }

    return fit


def _bcf(rng):
    df = obs(rng, 500)
    return ci_from(sp.bcf(df, "y", "d", X, random_state=int(rng.integers(2**31))))


D = "_dgp.obs"
ROWS = [
    Row("ipw.ate.bootstrap", "ipw", "sp.ipw(df, y, d, X)", N, ATE, _ipw_boot, design=D),
    Row(
        "ipw.ate.sandwich",
        "ipw",
        "sp.ipw(..., se_method='sandwich')",
        N,
        ATE,
        _obs_fit(sp.ipw, se_method="sandwich"),
        design=D,
    ),
    Row(
        "ipw.att.sandwich",
        "ipw",
        "sp.ipw(..., estimand='ATT', se_method='sandwich')",
        N,
        ATT,
        _obs_fit(sp.ipw, "ATT", estimand="ATT", se_method="sandwich"),
        design=D,
    ),
    Row(
        "ipw.atc.sandwich",
        "ipw",
        "sp.ipw(..., estimand='ATC', se_method='sandwich')",
        N,
        ATC,
        _obs_fit(sp.ipw, "ATC", estimand="ATC", se_method="sandwich"),
        design=D,
    ),
    Row(
        "ipw.ate.ht.sandwich",
        "ipw",
        "sp.ipw(..., normalize=False, se_method='sandwich')",
        N,
        ATE,
        _obs_fit(sp.ipw, normalize=False, se_method="sandwich"),
        design=D,
    ),
    Row(
        "ipw.ate.weights.sandwich",
        "ipw",
        "sp.ipw(..., weights='w', se_method='sandwich')",
        N,
        ATE,
        _weighted_fit(sp.ipw, se_method="sandwich"),
        design=D + " + independent sampling weights U(0.5, 1.5)",
    ),
    Row(
        "ipw.ate.cluster.sandwich",
        "ipw",
        "sp.ipw(..., cluster='g', se_method='sandwich')",
        N,
        ATE,
        _cluster_fit(sp.ipw, se_method="sandwich"),
        design="_dgp.obs_clustered, 100 clusters of 10",
    ),
    Row(
        "ipw.ate.cluster.bootstrap",
        "ipw",
        "sp.ipw(..., cluster='g')",
        N,
        ATE,
        _cluster_fit(sp.ipw, seed=1),
        design="_dgp.obs_clustered, 100 clusters of 10",
    ),
    Row(
        "ipw.ate.ps_learner",
        "ipw",
        "sp.ipw(..., ps_learner=LogisticRegression(C=1e6), n_bootstrap=200)",
        600,
        ATE,
        _ipw_learner,
        B=200,
        design=D + " (n = 600, 200 bootstrap draws, B = 200: each draw refits)",
    ),
    Row(
        "aipw.ate", "aipw", "sp.aipw(df, y, d, X)", N, ATE, _obs_fit(sp.aipw), design=D
    ),
    Row(
        "aipw.att",
        "aipw",
        "sp.aipw(..., estimand='ATT')",
        N,
        ATT,
        _obs_fit(sp.aipw, "ATT", estimand="ATT"),
        design=D,
    ),
    Row(
        "aipw.ate.no_cross_fit",
        "aipw",
        "sp.aipw(..., cross_fit=False)",
        N,
        ATE,
        _obs_fit(sp.aipw, cross_fit=False),
        design=D,
    ),
    Row(
        "aipw.ate.sandwich",
        "aipw",
        "sp.aipw(..., cross_fit=False, se_method='sandwich')",
        N,
        ATE,
        _obs_fit(sp.aipw, cross_fit=False, se_method="sandwich"),
        design=D,
    ),
    Row(
        "aipw.ate.weights",
        "aipw",
        "sp.aipw(..., weights='w')",
        N,
        ATE,
        _weighted_fit(sp.aipw),
        design=D + " + independent sampling weights U(0.5, 1.5)",
    ),
    Row(
        "aipw.ate.cluster",
        "aipw",
        "sp.aipw(..., cluster='g')",
        N,
        ATE,
        _cluster_fit(sp.aipw),
        design="_dgp.obs_clustered, 100 clusters of 10",
    ),
    Row(
        "g_computation.ate",
        "g_computation",
        "sp.g_computation(df, y, d, X)",
        N,
        ATE,
        _gcomp(hetero=False),
        design=D + " with a constant effect (the default single regression "
        "has no treatment-covariate interaction)",
    ),
    Row(
        "g_computation.ate.by_arm",
        "g_computation",
        "sp.g_computation(..., by_arm=True)",
        N,
        ATE,
        _gcomp(by_arm=True),
        design=D,
    ),
    Row(
        "g_computation.att.by_arm",
        "g_computation",
        "sp.g_computation(..., estimand='ATT', by_arm=True)",
        N,
        ATT,
        _gcomp("ATT", by_arm=True),
        design=D,
    ),
    Row("tmle.ate", "tmle", "sp.tmle(df, y, d, X)", N, ATE, _tmle(), design=D),
    Row(
        "tmle.att",
        "tmle",
        "sp.tmle(..., estimand='ATT')",
        N,
        ATT,
        _tmle("ATT"),
        design=D,
    ),
    Row(
        "tmle.atc.glm",
        "tmle",
        "sp.tmle(..., estimand='ATC', outcome_library=[OLS], "
        "propensity_library=[logit])",
        N,
        ATC,
        _tmle_glm("ATC"),
        design=D,
    ),
    Row(
        "tmle.ate.per_arm.glm",
        "tmle",
        "sp.tmle(..., fluctuation='per_arm', GLM libraries)",
        N,
        ATE,
        _tmle_glm(fluctuation="per_arm"),
        design=D,
    ),
    Row(
        "tmle.ey1.glm",
        "tmle",
        "sp.tmle(..., estimand='EY1', GLM libraries)",
        N,
        2.0,
        _tmle_glm("EY1"),
        design=D + " (E[Y(1)] = 1 + TAU0 = 2)",
    ),
    Row(
        "tmle.ate.cluster.glm",
        "tmle",
        "sp.tmle(..., cluster='g', GLM libraries)",
        N,
        ATE,
        _tmle_glm(cluster=True),
        design="_dgp.obs_clustered, 100 clusters of 10",
    ),
    Row(
        "ctmle.ate",
        "ctmle",
        "sp.ctmle(df, y, d, X)",
        500,
        ATE,
        _ctmle,
        B=200,
        design=D + " (n = 500, B = 200: 15 s per fit at n = 1000)",
    ),
    Row("hal_tmle.ate", "hal_tmle", "sp.hal_tmle(df, y, d, X)", N, ATE, _hal, design=D),
    Row(
        "difference_in_means",
        "difference_in_means",
        "sp.difference_in_means(df, y, d)",
        N,
        ATE,
        _rct_fit(lambda df: sp.difference_in_means(df, "y", "d")),
        design="_dgp.rct (Bernoulli(0.5) assignment)",
    ),
    Row(
        "difference_in_means.blocks",
        "difference_in_means",
        "sp.difference_in_means(..., blocks='block')",
        N,
        ATE,
        _rct_blocks,
        design="four blocks with treatment shares 0.3 / 0.5 / 0.5 / 0.7 and a "
        "block effect",
    ),
    Row(
        "difference_in_means.cluster",
        "difference_in_means",
        "sp.difference_in_means(..., cluster='g')",
        N,
        1.0,
        _rct_cluster,
        design="cluster-randomised, 100 clusters of 10, ICC 1/3",
    ),
    Row(
        "lm_lin",
        "lm_lin",
        "sp.lm_lin(df, y, d, X)",
        N,
        ATE,
        _rct_fit(lambda df: sp.lm_lin(df, "y", "d", X)),
        design="_dgp.rct",
    ),
    Row(
        "lm_lin.superpopulation",
        "lm_lin",
        "sp.lm_lin(..., superpopulation=True)",
        N,
        ATE,
        _rct_fit(lambda df: sp.lm_lin(df, "y", "d", X, superpopulation=True)),
        design="_dgp.rct",
    ),
    Row(
        "lm_lin.cluster",
        "lm_lin",
        "sp.lm_lin(..., cluster='g')",
        N,
        1.0,
        _lm_lin_cluster,
        design="cluster-randomised, 100 clusters of 10, ICC 1/3, effect 1 + 0.5 x1",
    ),
    Row(
        "ltmle.ate",
        "ltmle",
        "sp.ltmle(df, y, ['a1', 'a2'], [['l1'], ['l2']])",
        N,
        2.5,
        _ltmle,
        design="two decision points, time-varying confounder affected by "
        "the first treatment, logit treatment models",
    ),
    Row(
        "lmtp.mean",
        "lmtp",
        "sp.lmtp(df, Y, ['A'], shift=1.0, baseline=['W'])",
        N,
        1.0,
        _lmtp("mean"),
        design="A = W + v, Y = A + W + e; E[Y(A + 1)] = 1",
    ),
    Row(
        "lmtp.effect",
        "lmtp",
        "sp.lmtp(...).model_info['effect'] +/- 1.96 effect_se",
        N,
        1.0,
        _lmtp("effect"),
        design="as lmtp.mean; shift effect against the natural course = 1",
    ),
    Row(
        "lmtp.mean.sdr.folds",
        "lmtp",
        "sp.lmtp(..., estimator='sdr', folds=5)",
        N,
        1.0,
        _lmtp("mean", estimator="sdr", folds=5),
        design="as lmtp.mean",
    ),
    Row(
        "bcf.ate",
        "bcf",
        "sp.bcf(df, y, d, X)",
        500,
        ATE,
        _bcf,
        B=200,
        design=D + " (n = 500, B = 200: 12 s per fit)",
    ),
]

MECHANISMS: dict = {}

if __name__ == "__main__":
    main(__file__, ROWS, MECHANISMS)
