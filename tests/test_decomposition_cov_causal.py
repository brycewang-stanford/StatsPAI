"""Coverage campaign (decomposition) — causal & nonlinear decompositions.

Covers ``causal.py`` (gap-closing, natural-effects mediation, JVW disparity),
``nonlinear.py`` (Fairlie, Bauer–Sinning), ``kitagawa.py`` (rate decomposition),
and ``yu_elwert.py``. Each test pins the method's defining additive identity:

* gap_closing:  ``closed_gap == observed_gap - counterfactual_gap``.
* mediation:    ``total == nde + nie``  and  ``propn_mediated == nie / total``.
* disparity:    ``mediator_attributable == total_disparity - initial_disparity``.
* Fairlie / Bauer–Sinning:  ``gap == explained + unexplained == rate_A - rate_B``.
* Kitagawa:     ``gap == rate_effect + composition_effect + interaction`` and
                ``gap == rate_A - rate_B``.
* Yu–Elwert:    ``disparity == baseline + prevalence + effect + selection``.

All exact algebraic identities of the estimators (no mocking, CLAUDE.md §5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def causal_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 1500
    x1 = rng.normal(size=n)
    g = (rng.uniform(size=n) < 0.5).astype(int)
    tr = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.5 * x1)))).astype(int)
    med = 0.5 * tr + 0.3 * x1 + rng.normal(size=n)
    y = 1 + 0.8 * tr + 0.5 * med + 0.4 * x1 - 0.3 * g + rng.normal(size=n)
    ybin = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.3 + 0.5 * x1 - 0.4 * g)))).astype(
        int
    )
    return pd.DataFrame({"y": y, "ybin": ybin, "g": g, "tr": tr, "med": med, "x1": x1})


# ── gap-closing (Lundberg) ───────────────────────────────────────────


def test_gap_closing_identity(causal_df):
    r = sp.decompose(
        "gap_closing", data=causal_df, y="y", group="g", x=["x1"], inference="none"
    )
    assert r.closed_gap == pytest.approx(
        r.observed_gap - r.counterfactual_gap, rel=1e-9, abs=1e-9
    )


# ── natural-effects mediation ────────────────────────────────────────


def test_mediation_total_equals_direct_plus_indirect(causal_df):
    r = sp.decompose(
        "mediation",
        data=causal_df,
        y="y",
        treatment="tr",
        mediator="med",
        covariates=["x1"],
        inference="none",
    )
    assert r.total == pytest.approx(r.nde + r.nie, rel=1e-7, abs=1e-9)
    assert r.propn_mediated == pytest.approx(r.nie / r.total, rel=1e-7)


# ── JVW disparity reduction ──────────────────────────────────────────


def test_disparity_attributable_identity(causal_df):
    r = sp.decompose(
        "disparity", data=causal_df, y="y", group="g", mediator="med", covariates=["x1"]
    )
    assert r.mediator_attributable == pytest.approx(
        r.total_disparity - r.initial_disparity, rel=1e-7, abs=1e-9
    )


# ── nonlinear (binary outcome) decompositions ────────────────────────


def test_fairlie_identity(causal_df):
    r = sp.decompose(
        "fairlie", data=causal_df, y="ybin", group="g", x=["x1"], n_sim=120
    )
    assert r.gap == pytest.approx(r.explained + r.unexplained, rel=1e-7, abs=1e-9)
    assert r.gap == pytest.approx(r.rate_a - r.rate_b, rel=1e-7, abs=1e-9)


@pytest.mark.parametrize("variant", ["yun", "fairlie"])
def test_bauer_sinning_identity(causal_df, variant):
    r = sp.decompose(
        "bauer_sinning", data=causal_df, y="ybin", group="g", x=["x1"], variant=variant
    )
    assert r.gap == pytest.approx(r.explained + r.unexplained, rel=1e-7, abs=1e-9)


# ── Kitagawa rate decomposition ──────────────────────────────────────


def test_kitagawa_threeway_identity():
    rng = np.random.default_rng(1)
    n = 1200
    df = pd.DataFrame(
        {
            "rate": rng.uniform(0, 1, n),
            "period": (rng.uniform(size=n) < 0.5).astype(int),
            "agecat": rng.integers(0, 3, n),
        }
    )
    r = sp.decompose("kitagawa", data=df, rate="rate", group="period", by="agecat")
    assert r.gap == pytest.approx(
        r.rate_effect + r.composition_effect + r.interaction, rel=1e-9, abs=1e-9
    )
    assert r.gap == pytest.approx(r.rate_a - r.rate_b, rel=1e-9, abs=1e-9)


# ── Yu–Elwert four-way causal decomposition ──────────────────────────


def test_yu_elwert_fourway_identity(causal_df):
    r = sp.decompose(
        "yu_elwert",
        data=causal_df,
        y="ybin",
        treatment="tr",
        group="g",
        x=["x1"],
        inference="none",
        method="plugin",
    )
    assert r.disparity == pytest.approx(
        r.baseline + r.prevalence + r.effect + r.selection, rel=1e-7, abs=1e-9
    )


# ── result rendering ─────────────────────────────────────────────────


def test_causal_result_rendering(causal_df):
    for r in (
        sp.decompose(
            "mediation",
            data=causal_df,
            y="y",
            treatment="tr",
            mediator="med",
            covariates=["x1"],
            inference="none",
        ),
        sp.decompose(
            "gap_closing", data=causal_df, y="y", group="g", x=["x1"], inference="none"
        ),
    ):
        assert isinstance(r.summary(), str)
        assert isinstance(repr(r), str)


# ── Das Gupta multi-factor standardization ───────────────────────────


def test_das_gupta_factor_effects_sum_to_gap():
    rng = np.random.default_rng(2)
    a = pd.DataFrame({"f1": rng.uniform(0.2, 0.8, 6), "f2": rng.uniform(0.2, 0.8, 6)})
    b = pd.DataFrame({"f1": rng.uniform(0.2, 0.8, 6), "f2": rng.uniform(0.2, 0.8, 6)})
    r = sp.decompose("das_gupta", data_a=a, data_b=b, factor_names=["f1", "f2"])
    eff_sum = float(np.asarray(r.factor_effects["effect"], dtype=float).sum())
    assert eff_sum == pytest.approx(r.gap, rel=1e-7, abs=1e-9)
    assert r.gap == pytest.approx(r.rate_a - r.rate_b, rel=1e-9, abs=1e-9)


# ── bootstrap-inference paths (small n_boot for speed) ───────────────


def test_gap_closing_bootstrap_inference(causal_df):
    r = sp.decompose(
        "gap_closing",
        data=causal_df,
        y="y",
        group="g",
        x=["x1"],
        inference="bootstrap",
        n_boot=40,
        seed=0,
    )
    # SE / CI populated and internally consistent on the bootstrap path
    # (se / ci are dicts keyed by observed / counterfactual / closed).
    assert np.isfinite(r.se["closed"])
    lo, hi = r.ci["closed"]
    assert lo <= r.closed_gap <= hi


def test_mediation_bootstrap_inference(causal_df):
    r = sp.decompose(
        "mediation",
        data=causal_df,
        y="y",
        treatment="tr",
        mediator="med",
        covariates=["x1"],
        inference="bootstrap",
        n_boot=40,
        seed=0,
    )
    assert r.total == pytest.approx(r.nde + r.nie, rel=1e-6, abs=1e-7)


def test_yu_elwert_bootstrap_inference(causal_df):
    r = sp.decompose(
        "yu_elwert",
        data=causal_df,
        y="ybin",
        treatment="tr",
        group="g",
        x=["x1"],
        inference="bootstrap",
        n_boot=40,
        method="plugin",
    )
    assert r.disparity == pytest.approx(
        r.baseline + r.prevalence + r.effect + r.selection, rel=1e-6, abs=1e-7
    )
    assert r.se is not None


def test_fairlie_probit_model(causal_df):
    r = sp.decompose(
        "fairlie",
        data=causal_df,
        y="ybin",
        group="g",
        x=["x1"],
        model="probit",
        n_sim=100,
    )
    assert r.gap == pytest.approx(r.explained + r.unexplained, rel=1e-7, abs=1e-9)
