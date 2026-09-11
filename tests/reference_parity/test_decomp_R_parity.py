"""Decomposition family vs the R packages that define the methods.

Reference: ``_fixtures/decomp_R.json`` written by ``_generate_decomp_R.R``
(DasGuptR 2.2.0, ddecompose 1.0.0, cdgd 1.0.1) from Das Gupta's own worked
examples (exported by the same script) and the simulated
``decomp_gap.csv`` / ``decomp_ye.csv`` (``_generate_decomp_data.py``).

Every comparison is on identical bytes at 1e-10 relative or tighter, except
where a logistic MLE sits in the path (1e-9: IRLS on the R side and Newton
on ours converge to the same optimum by different routes).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "decomp_R.json").read_text(encoding="utf-8"))


def _close(ours, ref, rtol=1e-10, atol=0.0):
    np.testing.assert_allclose(float(ours), float(ref), rtol=rtol, atol=atol)


def _effects(res):
    return dict(zip(res.factor_effects["factor"], res.factor_effects["effect"]))


# --------------------------------------------------------------------------
# Das Gupta / Kitagawa (DasGuptR)
# --------------------------------------------------------------------------


def test_das_gupta_two_factors_matches_dasguptr():
    d = pd.read_csv(_FIX / "decomp_dg2_1.csv")
    f = ["avg_earnings", "earner_prop"]
    r = sp.das_gupta(d[d["pop"] == "black"], d[d["pop"] == "white"], f)
    for k, v in _effects(r).items():
        _close(v, R["dg2_1"][k])


def test_das_gupta_cross_classified_sums_over_strata():
    """Four factors x six age groups: the aggregate is sum_i prod_f f_{f,i}.

    Before 1.28.0 sp.das_gupta multiplied the factor MEANS instead, which
    gave factor A 0% of the gap here (DasGuptR: 37%) and D +333% (-52%).
    """
    d = pd.read_csv(_FIX / "decomp_dg6_5.csv")
    a, b = d[d["pop"] == 1968], d[d["pop"] == 1963]
    r = sp.das_gupta(a, b, ["A", "B", "C", "D"], by="agegroup")
    _close(r.rate_a, R["dg6_5_crude"]["r1968"])
    _close(r.rate_b, R["dg6_5_crude"]["r1963"])
    for k, v in _effects(r).items():
        _close(v, R["dg6_5"][k])


def test_das_gupta_pairs_strata_by_key_not_position():
    d = pd.read_csv(_FIX / "decomp_dg6_5.csv")
    a, b = d[d["pop"] == 1968], d[d["pop"] == 1963]
    shuffled = b.sample(frac=1.0, random_state=0)
    r1 = sp.das_gupta(a, b, ["A", "B", "C", "D"], by="agegroup")
    r2 = sp.das_gupta(a, shuffled, ["A", "B", "C", "D"], by="agegroup")
    np.testing.assert_allclose(
        r1.factor_effects["effect"], r2.factor_effects["effect"], rtol=1e-13
    )


def test_das_gupta_rejects_unpaired_rows():
    d = pd.read_csv(_FIX / "decomp_dg6_5.csv")
    a, b = d[d["pop"] == 1968], d[d["pop"] == 1963]
    with pytest.raises(ValueError, match="different numbers of rows"):
        sp.das_gupta(a, b.iloc[:-1], ["A", "B", "C", "D"])
    with pytest.raises(ValueError, match="same strata"):
        sp.das_gupta(a, b.iloc[:-1], ["A", "B", "C", "D"], by="agegroup")


def test_kitagawa_matches_dasguptr_two_factor():
    d = pd.read_csv(_FIX / "decomp_dg5_1.csv")
    d["g"] = (d["pop"] != 1970).astype(int)  # group 0 = 1970 = "A"
    r = sp.kitagawa_decompose(d, rate="rate", group="g", by="age_group", weights="size")
    _close(r.composition_effect, R["kitagawa"]["size"])
    _close(r.rate_effect, R["kitagawa"]["rate"])
    _close(r.interaction, 0.0, atol=1e-12)


# --------------------------------------------------------------------------
# gap_closing (ddecompose)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gap():
    return pd.read_csv(_FIX / "decomp_gap.csv")


@pytest.mark.parametrize("target, key", [(1, "dfl_ref0"), (0, "dfl_ref1")])
def test_gap_closing_ipw_is_dfl_reweighting(gap, target, key):
    """IPW gap closing is DiNardo-Fortin-Lemieux reweighting.

    Before 1.28.0 the weights were the reciprocal of the density ratio, so
    the reweighted sample moved away from the target distribution.
    """
    r = sp.gap_closing(
        gap,
        "y",
        "group",
        ["x1", "x2"],
        method="ipw",
        target_dist=target,
        inference="none",
        trim=0.0,
    )
    ref = R[key]
    # StatsPAI's gaps are group 0 minus group 1; ddecompose's are 1 minus 0.
    _close(r.observed_gap, -ref["observed"])
    # Either way the reweighted group's counterfactual mean mu_C splits the
    # gap as ddecompose does: counterfactual gap = -structure effect, closed
    # gap = -composition effect.
    _close(r.counterfactual_gap, -ref["structure"], rtol=1e-9)
    _close(r.closed_gap, -ref["composition"], rtol=1e-9)


def test_gap_closing_regression_is_oaxaca_counterfactual(gap):
    r = sp.gap_closing(
        gap,
        "y",
        "group",
        ["x1", "x2"],
        method="regression",
        target_dist=1,
        inference="none",
    )
    ref = R["ob_ref0"]
    _close(r.closed_gap, -ref["composition"])
    _close(r.counterfactual_gap, -ref["structure"])


def test_gap_closing_aipw_is_doubly_robust():
    """Known truth: no group effect given x, so shifting x closes the gap.

    The outcome is quadratic in x and the outcome model linear, so only the
    propensity half of AIPW is right; the estimate must still be ~0. Before
    1.28.0 the reweighting half ran backwards and this held only when the
    outcome model was also right.
    """
    rng = np.random.default_rng(1)
    n = 200_000
    g = rng.integers(0, 2, n)
    x = rng.normal(g * 1.0, 1.0)
    df = pd.DataFrame({"y": x**2 + rng.normal(0, 1, n), "group": g, "x": x})
    for target in (0, 1):
        r = sp.gap_closing(
            df, "y", "group", ["x"], method="aipw", target_dist=target, inference="none"
        )
        assert abs(r.counterfactual_gap) < 0.05, (target, r.counterfactual_gap)
        assert abs(r.observed_gap) > 0.9


# --------------------------------------------------------------------------
# Yu-Elwert (cdgd)
# --------------------------------------------------------------------------


def test_yu_elwert_efficient_matches_cdgd():
    e = pd.read_csv(_FIX / "decomp_ye.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.yu_elwert_decompose(
            e,
            "y",
            "t",
            "r",
            ["x1", "x2"],
            method="efficient",
            inference="analytic",
            trim=0.0,
        )
    ref = R["cdgd"]
    names = {
        "disparity": "total",
        "baseline": "baseline",
        "prevalence": "prevalence",
        "effect": "effect",
        "selection": "selection",
    }
    for ours, theirs in names.items():
        _close(getattr(r, ours), ref["point"][theirs], rtol=1e-9)
        _close(r.se[ours], ref["se"][theirs], rtol=1e-9)


def test_yu_elwert_efficient_components_add_up():
    e = pd.read_csv(_FIX / "decomp_ye.csv")
    r = sp.yu_elwert_decompose(
        e, "y", "t", "r", ["x1", "x2"], method="efficient", inference="none"
    )
    _close(r.baseline + r.prevalence + r.effect + r.selection, r.disparity, rtol=1e-13)


def test_yu_elwert_analytic_needs_efficient():
    e = pd.read_csv(_FIX / "decomp_ye.csv")
    with pytest.raises(ValueError, match="efficient"):
        sp.yu_elwert_decompose(e, "y", "t", "r", ["x1"], inference="analytic")
