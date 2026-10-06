"""Behaviour pinned by the rerun of Das, *Causal Inference in R* (2026-10).

Smaller items of that pass; the larger ones have files of their own
(``reference_parity/test_cox_ph_test_parity.py``,
``test_effect_size_power_ttest_parity.py``).
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# --------------------------------------------------------------- sp.match
def _discrete_score_data(seed=0, n=1200):
    """Two binary covariates: the propensity score takes four values."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 2, n)
    b = rng.integers(0, 2, n)
    p = 1 / (1 + np.exp(-(-1.0 + 0.8 * a + 0.6 * b)))
    t = rng.binomial(1, p)
    y = 1.0 * t + a - b + rng.standard_normal(n)
    return pd.DataFrame({"y": y, "t": t, "a": a, "b": b})


def test_match_warns_when_ties_first_picks_among_equals():
    df = _discrete_score_data()
    with pytest.warns(UserWarning, match="equally close matches") as rec:
        fit = sp.match(df, y="y", treat="t", covariates=["a", "b"], ties="first")
    message = next(str(w.message) for w in rec if "equally close" in str(w.message))
    n_treated = int(df["t"].sum())
    assert f"{n_treated} of {n_treated} matched units" in message
    assert "rests on 4 distinct control units" in message
    assert fit.model_info["n_units_with_tied_matches"] == n_treated
    assert fit.model_info["n_tied_matches_left_out"] > n_treated


def test_match_estimate_under_ties_first_depends_on_row_order():
    """What the warning is about: the same data, reordered, another ATT."""
    df = _discrete_score_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kw = dict(y="y", treat="t", covariates=["a", "b"])
        first = sp.match(df, ties="first", **kw).estimate
        shuffled = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
        second = sp.match(shuffled, ties="first", **kw).estimate
        # the default keeps every tied control
        all_1 = sp.match(df, **kw).estimate
        all_2 = sp.match(shuffled, **kw).estimate
        assert sp.match(df, ties="all", **kw).estimate == all_1
    assert abs(first - second) > 0.05
    assert all_1 == pytest.approx(all_2, abs=1e-10)
    assert all_1 == pytest.approx(1.0, abs=0.2)


def test_match_is_quiet_without_ties_and_by_default():
    rng = np.random.default_rng(4)
    n = 400
    x = rng.standard_normal((n, 2))
    t = rng.binomial(1, 1 / (1 + np.exp(-x[:, 0])))
    df = pd.DataFrame({"y": t + x[:, 0] + rng.standard_normal(n), "t": t})
    df["x1"], df["x2"] = x[:, 0], x[:, 1]
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fit = sp.match(df, y="y", treat="t", covariates=["x1", "x2"])
        tied = sp.match(_discrete_score_data(), y="y", treat="t", covariates=["a", "b"])
    assert not [w for w in rec if "equally close" in str(w.message)]
    assert fit.model_info["n_units_with_tied_matches"] == 0
    # ties are still counted when they are pooled
    assert tied.model_info["n_units_with_tied_matches"] > 0
    assert tied.model_info["n_tied_matches_left_out"] == 0


# ------------------------------------------------------- sp.adjust_pvalues
P = [0.001, 0.02, 0.03, 0.04, 0.2]
# R 4.5.2: p.adjust(c(0.001, 0.02, 0.03, 0.04, 0.2), method)
R_P_ADJUST = {
    "bonferroni": [0.005, 0.1, 0.15, 0.2, 1.0],
    "holm": [0.005, 0.08, 0.09, 0.09, 0.2],
    "hochberg": [0.005, 0.08, 0.08, 0.08, 0.2],
    "hommel": [0.005, 0.06, 0.06, 0.08, 0.2],
    "bh": [0.005, 0.05, 0.05, 0.05, 0.2],
    "by": [
        0.0114166666666667,
        0.114166666666667,
        0.114166666666667,
        0.114166666666667,
        0.456666666666667,
    ],
}


@pytest.mark.parametrize("method", sorted(R_P_ADJUST))
def test_adjust_pvalues_equals_r_p_adjust(method):
    np.testing.assert_allclose(
        sp.adjust_pvalues(P, method=method), R_P_ADJUST[method], rtol=1e-13
    )


def test_adjust_pvalues_orderings_and_edges():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=25) ** 2
    holm, hoch, homm = (sp.adjust_pvalues(p, m) for m in ("holm", "hochberg", "hommel"))
    # each is uniformly at least as powerful as the one before
    assert np.all(hoch <= holm + 1e-15) and np.all(homm <= hoch + 1e-15)
    assert np.all(homm >= p - 1e-15)
    assert np.all(sp.adjust_pvalues(p, "by") >= sp.adjust_pvalues(p, "bh") - 1e-15)
    np.testing.assert_allclose(
        sp.adjust_pvalues([0.01, 0.04], "sidak"), [1 - 0.99**2, 1 - 0.96**2]
    )
    # a single test is its own family
    for m in ("hochberg", "hommel", "by", "sidak"):
        assert sp.adjust_pvalues([0.03], m)[0] == pytest.approx(0.03)
    # a missing p-value stays missing and still counts towards the family
    out = sp.adjust_pvalues([0.01, np.nan, 0.04], "hommel")
    assert np.isnan(out[1])
    np.testing.assert_allclose(out[[0, 2]], [0.03, 0.08])
    with pytest.raises(ValueError, match=r"\['bonferroni', 'sidak', 'holm'"):
        sp.adjust_pvalues(P, "tukey")
