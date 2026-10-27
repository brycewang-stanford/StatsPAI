"""Result classes that became registered in 1.31 return what they claim.

These 24 classes (and their estimators) were reachable as ``sp.<name>`` but
absent from ``__all__``, so agents never saw them. Registering them makes a
promise -- ``sp.describe_function('QLearningResult')`` says it is what
``sp.q_learning`` returns -- and this file holds each producer to it, and to
the agent-native serialization every result must support.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _dtr():
    rng = np.random.default_rng(0)
    n = 300
    x1 = rng.normal(size=n)
    a1 = rng.integers(0, 2, n)
    x2 = x1 + rng.normal(size=n)
    a2 = rng.integers(0, 2, n)
    y = x1 + a1 * (1 + x1) + a2 * (0.5 - x2) + rng.normal(size=n)
    return pd.DataFrame({"x1": x1, "a1": a1, "x2": x2, "a2": a2, "y": y})


def _dtr_kw():
    return dict(y="y", actions=["a1", "a2"], stage_covariates=[["x1"], ["x2"]])


def _snp2():
    rng = np.random.default_rng(0)
    bx1, bx2 = rng.uniform(0.1, 0.5, 30), rng.uniform(0.1, 0.5, 30)
    return pd.DataFrame(
        {
            "beta_x1": bx1,
            "beta_x2": bx2,
            "beta_y": 0.4 * bx1 + rng.normal(0, 0.02, 30),
            "se_y": rng.uniform(0.01, 0.05, 30),
        }
    )


def _snp_med():
    rng = np.random.default_rng(1)
    bx = rng.uniform(0.1, 0.5, 30)
    bm = 0.5 * bx + rng.normal(0, 0.02, 30)
    return pd.DataFrame(
        {
            "beta_x": bx,
            "se_x": rng.uniform(0.01, 0.05, 30),
            "beta_m": bm,
            "se_m": rng.uniform(0.01, 0.05, 30),
            "beta_y": 0.2 * bx + 0.6 * bm + rng.normal(0, 0.02, 30),
            "se_y": rng.uniform(0.01, 0.05, 30),
        }
    )


def _bp():
    rng = np.random.default_rng(0)
    z = rng.integers(0, 2, 400)
    d = (z & (rng.random(400) < 0.8)).astype(int)
    y = (rng.random(400) < 0.3 + 0.4 * d).astype(int)
    return sp.balke_pearl(
        pd.DataFrame({"y": y, "d": d, "z": z}), y="y", treat="d", instrument="z"
    )


def _gnn():
    rng = np.random.default_rng(0)
    n = 200
    A = np.triu((rng.random((n, n)) < 0.02).astype(float), 1)
    df = pd.DataFrame({"x": rng.normal(size=n), "t": rng.integers(0, 2, n)})
    df["y"] = df.x + df.t + rng.normal(size=n)
    return sp.gnn_causal(
        df, y="y", treat="t", covariates=["x"], adjacency=A + A.T, n_trees=30
    )


def _bayes_dml():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x1": rng.normal(size=200), "x2": rng.normal(size=200)})
    df["d"] = df.x1 + rng.normal(size=200)
    df["y"] = 0.5 * df.d + df.x2 + rng.normal(size=200)
    return sp.bayes_dml(df, y="y", treatment="d", covariates=["x1", "x2"], n_folds=2)


def _cfpolicy():
    rng = np.random.default_rng(0)
    s = rng.normal(size=150)
    a = 0.5 * s + rng.normal(size=150)
    df = pd.DataFrame({"s": s, "a": a, "r": s + 2 * a + rng.normal(size=150)})
    return sp.counterfactual_policy_optimization(
        df, state="s", action="a", reward="r", target_policy=lambda si: si + 1.0
    )


def _bandit():
    rng = np.random.default_rng(0)
    return sp.causal_bandit(
        ["A", "B"],
        n_samples=50,
        rng_seed=0,
        reward_fn=lambda arm, ctx: {"A": 1.0, "B": 0.3}[arm] + rng.normal(),
    )


def _policy_forest():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(150, 2))
    A = rng.integers(0, 2, 150)
    R = (A == (X[:, 0] > 0)).astype(float) + rng.normal(0, 0.3, 150)
    df = pd.DataFrame({"x0": X[:, 0], "x1": X[:, 1], "a": A, "r": R})
    return sp.causal_policy_forest(
        df, actions="a", rewards="r", covariates=["x0", "x1"], n_trees=3, depth=2
    )


def _cont_conformal():
    rng = np.random.default_rng(0)
    t = rng.uniform(0, 5, 200)
    x = rng.normal(size=200)
    train = pd.DataFrame(
        {"y": 2 + 0.7 * t + 0.5 * x + rng.normal(0, 0.5, 200), "t": t, "x": x}
    )
    return sp.conformal_continuous(
        train,
        y="y",
        treatment="t",
        covariates=["x"],
        test_data=pd.DataFrame({"t": [1.0], "x": [0.0]}),
    )


def _int_conformal():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "cluster": np.repeat(np.arange(12), 8),
            "treat": rng.integers(0, 2, 96),
            "x": rng.normal(size=96),
        }
    )
    df["y"] = 1 + 0.6 * df.treat + 0.4 * df.x + rng.normal(0, 0.3, 96)
    return sp.conformal_interference(
        df,
        y="y",
        treatment="treat",
        cluster="cluster",
        covariates=["x"],
        test_clusters=[0, 1],
    )


def _ewi():
    rng = np.random.default_rng(0)
    A = rng.integers(0, 2, 200)
    df = pd.DataFrame({"A": A, "credit": 600 + 100 * A + rng.normal(0, 30, 200)})

    def predictor(d):
        return 1 / (1 + np.exp(-(d["credit"] / 100 - 6)))

    def intervene(d, a):
        out = d.copy()
        out["A"] = a
        out["credit"] = 600 + 100 * a + (d["credit"] - (600 + 100 * d["A"]))
        return out

    return sp.evidence_without_injustice(
        df,
        predictor,
        protected="A",
        admissible_features=["credit"],
        scm_intervention=intervene,
        n_boot=99,
        random_state=0,
    )


def _fair_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({"g": rng.integers(0, 2, 150), "pred": rng.integers(0, 2, 150)})


def _ml_bounds():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=200), "d": rng.integers(0, 2, 200)})
    df["y"] = df.x + 0.5 * df.d + rng.normal(size=200)
    return sp.ml_bounds(
        df,
        y="y",
        treat="d",
        covariates=["x"],
        n_splits=2,
        n_bootstrap=10,
        random_state=0,
    )


def _sharp_ope():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {"a": rng.integers(0, 2, 150), "r": rng.normal(size=150), "e": 0.5, "pi": 0.5}
    )
    return sp.sharp_ope_unobserved(
        df, actions="a", rewards="r", logging_prob="e", target_prob="pi", gamma=1.5
    )


def _mdp():
    rng = np.random.default_rng(0)
    s = rng.normal(size=150)
    a = rng.normal(size=150)
    df = pd.DataFrame(
        {
            "s": s,
            "a": a,
            "ns": 0.8 * s + 0.2 * a + rng.normal(0, 0.1, 150),
            "r": s + 0.5 * a + rng.normal(0, 0.1, 150),
        }
    )
    return sp.structural_mdp(
        df, state_cols=["s"], action_cols=["a"], reward="r", next_state_cols=["ns"]
    )


CASES = {
    "QLearningResult": lambda: sp.q_learning(_dtr(), **_dtr_kw()),
    "ALearningResult": lambda: sp.a_learning(_dtr(), **_dtr_kw()),
    "SNMMResult": lambda: sp.snmm(_dtr(), **_dtr_kw()),
    "BalkePearlResult": _bp,
    "GNNCausalResult": _gnn,
    "AssimilationResult": lambda: sp.causal_kalman([0.45, 0.55], [0.1, 0.1]),
    "BayesianDMLResult": _bayes_dml,
    "CFPolicyResult": _cfpolicy,
    "CausalBanditResult": _bandit,
    "CausalMASResult": lambda: sp.causal_mas(
        ["age", "treatment", "mortality"],
        treatment="treatment",
        outcome="mortality",
        rounds=1,
    ),
    "CausalPolicyForestResult": _policy_forest,
    "ConcordanceResult": lambda: sp.rwd_rct_concordance(
        rct_estimate=0.5, rct_se=0.2, rwd_estimate=0.42
    ),
    "EvidenceSynthesisResult": lambda: sp.synthesise_evidence(
        rct_estimate=0.5, rct_se=0.2, rwd_estimate=0.42, rwd_se=0.1
    ),
    "ContinuousConformalResult": _cont_conformal,
    "InterferenceConformalResult": _int_conformal,
    "EvidenceWithoutInjusticeResult": _ewi,
    "FairnessAudit": lambda: sp.fairness_audit(
        _fair_df(), predictions="pred", protected="g"
    ),
    "FairnessResult": lambda: sp.demographic_parity(
        _fair_df(), predictions="pred", protected="g"
    ),
    "MLBoundsResult": _ml_bounds,
    "MVMRResult": lambda: sp.mr_multivariable(_snp2()),
    "MRBMAResult": lambda: sp.mr_bma(_snp2()),
    "MediationMRResult": lambda: sp.mr_mediation(_snp_med()),
    "SharpOPEResult": _sharp_ope,
    "StructuralMDPResult": _mdp,
}


@pytest.mark.parametrize("cls_name", sorted(CASES))
def test_producer_returns_registered_class_and_serializes(cls_name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = CASES[cls_name]()
    assert isinstance(res, getattr(sp, cls_name))
    assert cls_name in sp.list_functions()
    to_dict = getattr(res, "to_dict", None)
    if to_dict is not None:
        json.dumps(to_dict(), default=str)
