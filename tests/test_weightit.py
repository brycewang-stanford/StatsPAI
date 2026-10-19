"""``sp.weightit`` / ``sp.bal_tab``: identities and input handling.

Numerical agreement with R WeightIt / cobalt is in
``tests/reference_parity/test_weightit_cobalt_parity.py``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


def _design(n=600, seed=3):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    x3 = rng.binomial(1, 0.4, size=n)
    eta = np.column_stack([np.zeros(n), 0.6 * x1 + 0.3 * x3, -0.5 * x2 + 0.2])
    p = np.exp(eta) / np.exp(eta).sum(1, keepdims=True)
    g = np.array([rng.choice(3, p=pr) for pr in p])
    dose = 0.8 * x1 - 0.5 * x2 + 0.4 * x3 + rng.normal(size=n)
    y = 2.0 * dose + 1.0 * (g == 1) - 0.5 * (g == 2) + x1 + x2 + x3 + rng.normal(size=n)
    return pd.DataFrame(
        {
            "y": y,
            "g": np.array(["lo", "mid", "hi"])[g],
            "dose": dose,
            "x1": x1,
            "x2": x2,
            "x3": x3,
        }
    )


def _wls(y, X, w):
    sw = np.sqrt(w)
    return np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]


def test_continuous_ebal_makes_simple_slope_equal_adjusted_slope():
    # Weighted covariance of the dose with every covariate is zero, so by
    # Frisch-Waugh-Lovell the weighted simple slope equals the weighted
    # slope with the covariates.
    df = _design()
    w = sp.weightit(df, "dose", ["x1", "x2", "x3"], method="ebal").weights.to_numpy()
    one = np.ones(len(df))
    simple = _wls(df.y.to_numpy(), np.column_stack([one, df.dose]), w)[1]
    full = _wls(
        df.y.to_numpy(), np.column_stack([one, df.dose, df[["x1", "x2", "x3"]]]), w
    )[1]
    assert abs(simple - full) < 1e-8 * abs(full)


def test_multinomial_ebal_means_equal_regression_adjusted_contrasts():
    df = _design()
    W = sp.weightit(df, "g", ["x1", "x2", "x3"], method="ebal")
    w = W.weights.to_numpy()
    y = df.y.to_numpy()
    means = {
        lv: np.average(y[df.g == lv], weights=w[df.g == lv])
        for lv in ("lo", "mid", "hi")
    }
    D = np.column_stack(
        [np.ones(len(df)), df.g == "mid", df.g == "hi", df[["x1", "x2", "x3"]]]
    )
    b = _wls(y, D.astype(float), w)
    assert abs(b[1] - (means["mid"] - means["lo"])) < 1e-8
    assert abs(b[2] - (means["hi"] - means["lo"])) < 1e-8
    # every group is moved to the full-sample covariate means
    for lv in ("lo", "mid", "hi"):
        m = df.g == lv
        np.testing.assert_allclose(
            np.average(df.loc[m, ["x1", "x2", "x3"]], axis=0, weights=w[m]),
            df[["x1", "x2", "x3"]].mean().to_numpy(),
            atol=1e-9,
        )


def test_cbps_ate_equalizes_weighted_sums_across_groups():
    df = _design()
    w = sp.weightit(df, "g", ["x1", "x2"], method="cbps").weights.to_numpy()
    Xt = np.column_stack([np.ones(len(df)), df[["x1", "x2"]]])
    sums = [(Xt[df.g == lv] * w[df.g == lv, None]).sum(0) for lv in ("lo", "mid", "hi")]
    np.testing.assert_allclose(sums[0], sums[1], rtol=1e-9, atol=1e-8)
    np.testing.assert_allclose(sums[0], sums[2], rtol=1e-9, atol=1e-8)


def test_binary_glm_matches_ps_weights():
    df = _design()
    df["d"] = (df.g == "hi").astype(int)
    W = sp.weightit(df, "d", ["x1", "x2", "x3"], estimand="ATO")
    ref = sp.ps_weights(W.ps.to_numpy(), df.d.to_numpy(), estimand="ATO")
    np.testing.assert_allclose(W.weights.to_numpy(), ref, rtol=1e-12)


def test_integer_coded_treatment_needs_a_type():
    df = _design()
    df["gi"] = df.g.map({"lo": 0, "mid": 1, "hi": 2})
    with pytest.raises(MethodIncompatibility, match="treat_type"):
        sp.weightit(df, "gi", ["x1"])
    a = sp.weightit(df, "gi", ["x1", "x2"], treat_type="multinomial").weights
    b = sp.weightit(df, "g", ["x1", "x2"]).weights
    np.testing.assert_allclose(a, b, rtol=1e-10)


def test_rejects_undefined_combinations():
    df = _design()
    with pytest.raises(MethodIncompatibility, match="continuous"):
        sp.weightit(df, "dose", ["x1"], estimand="ATT")
    with pytest.raises(MethodIncompatibility, match="cbps"):
        sp.weightit(df, "dose", ["x1"], method="cbps")
    with pytest.raises(MethodIncompatibility, match="focal"):
        sp.weightit(df, "g", ["x1"], estimand="ATT")
    with pytest.raises(MethodIncompatibility, match="focal"):
        sp.weightit(df, "g", ["x1"], estimand="ATT", focal="nope")
    with pytest.raises(MethodIncompatibility):
        sp.weightit(df, "g", ["x1"], method="cbps", estimand="ATO")


def test_missing_rows_get_nan_weight_and_duplicate_index_is_positional():
    df = _design(n=300)
    df.loc[5, "x1"] = np.nan
    df.index = [0] * 150 + [1] * 150  # duplicated labels
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        W = sp.weightit(df, "g", ["x1", "x2"], method="ebal")
    assert any("dropped" in str(r.message) for r in rec)
    w = W.weights.to_numpy()
    assert np.isnan(w[5]) and np.isfinite(np.delete(w, 5)).all()
    sub = _design(n=300).drop(index=5)
    ref = sp.weightit(sub, "g", ["x1", "x2"], method="ebal").weights.to_numpy()
    np.testing.assert_allclose(np.delete(w, 5), ref, rtol=1e-10)


def test_categorical_covariate_is_expanded():
    df = _design()
    df["region"] = np.where(df.x3 == 1, "north", "south")
    a = sp.weightit(df, "g", ["x1", "region"], method="ebal").weights
    b = sp.weightit(df, "g", ["x1", "x3"], method="ebal").weights
    np.testing.assert_allclose(a, b, rtol=1e-10)
    bt = sp.bal_tab(df, "g", ["x1", "region"], weights=a)
    assert {"region_north", "region_south"} <= set(bt.table.index)


def test_bal_tab_unweighted_and_flags():
    df = _design()
    bt = sp.bal_tab(df, "g", ["x1", "x2", "x3"])
    assert "diff_adj" not in bt.table.columns
    assert set(bt.pairs) == {"hi vs. lo", "hi vs. mid", "lo vs. mid"}
    # the summary is the largest absolute pairwise difference
    for v in ("x1", "x2"):
        assert bt.table.loc[v, "diff_un"] == pytest.approx(
            max(abs(p.loc[v, "diff_un"]) for p in bt.pairs.values()), rel=1e-12
        )
    W = sp.weightit(df, "g", ["x1", "x2", "x3"], method="ebal")
    bt2 = W.balance(threshold=0.05)
    assert bt2.table["balanced"].all()
    assert bt2.summary_stats["n_imbalanced_adj"] == 0


def test_bal_tab_rejects_bad_weights():
    df = _design(n=100)
    with pytest.raises(MethodIncompatibility):
        sp.bal_tab(df, "g", ["x1"], weights=np.ones(99))
    with pytest.raises(MethodIncompatibility):
        sp.bal_tab(df, "g", ["x1"], weights=-np.ones(100))
