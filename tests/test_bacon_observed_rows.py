"""``sp.bacon_decomposition(balance='observed_rows')``: edges and identities.

The Stata ``ddtiming`` reference is in
``tests/reference_parity/test_bacon_ddtiming_Stata_parity.py``.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


def _panel(drop=(), seed=0):
    rng = np.random.default_rng(seed)
    cohorts = {u: g for u, g in enumerate([1] * 3 + [4] * 4 + [6] * 4 + [0] * 5)}
    rows = []
    for u, g in cohorts.items():
        a = rng.normal()
        for t in range(1, 9):
            d = int(g > 0 and t >= g)
            rows.append((u, t, d, a + 0.2 * t + d * (1 + 0.2 * (t - g)) + rng.normal()))
    df = pd.DataFrame(rows, columns=["unit", "t", "d", "y"])
    keep = ~pd.Series(list(zip(df.unit, df.t))).isin(list(drop)).to_numpy()
    return df[keep].reset_index(drop=True)


def _bacon(df, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.bacon_decomposition(df, y="y", treat="d", time="t", id="unit", **kw)


def test_balanced_panel_equals_default():
    df = _panel()
    a = _bacon(df)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning on a balanced panel
        b = sp.bacon_decomposition(
            df, y="y", treat="d", time="t", id="unit", balance="observed_rows"
        )
    key = ["type", "treated"]
    m = a["decomposition"].merge(b["decomposition"], on=key)
    assert len(m) == len(a["decomposition"]) == len(b["decomposition"])
    np.testing.assert_allclose(m["estimate_x"], m["estimate_y"], atol=1e-12)
    np.testing.assert_allclose(m["weight_x"], m["weight_y"], atol=1e-12)
    assert b["beta_twfe"] == pytest.approx(a["beta_twfe"], abs=1e-12)
    # on a balanced panel Theorem 1 is an identity
    assert b["weighted_sum"] == pytest.approx(b["beta_twfe"], abs=1e-12)


def test_unbalanced_panel_warns_with_the_gap_and_twfe_is_the_regression():
    df = _panel(drop=[(4, 2), (9, 7), (14, 3)])
    with pytest.warns(UserWarning, match="not an identity"):
        r = sp.bacon_decomposition(
            df, y="y", treat="d", time="t", id="unit", balance="observed_rows"
        )
    fe = sp.feols("y ~ d | unit + t", df)
    assert r["beta_twfe"] == pytest.approx(float(fe.params["d"]), rel=1e-10)
    assert r["weights"] is None and r["dropped_units"] == []
    assert r["decomposition"]["weight"].sum() == pytest.approx(1.0, abs=1e-12)


def test_unit_first_observed_treated_is_an_already_treated_control():
    # unit 3 (cohort 4) is missing periods 1-3: first observed treated in 4
    df = _panel(drop=[(3, 1), (3, 2), (3, 3)])
    r = _bacon(df, balance="observed_rows")
    dec = r["decomposition"]
    assert set(dec.loc[dec["type"] == "Later vs Always Treated", "control"]) == {
        "Always"
    }
    # the already-treated group is now 3 cohort-1 units plus unit 3
    from statspai.did._bacon_unbalanced import ALWAYS, timing_groups

    g = timing_groups(df, "d", "t", "unit")
    assert sorted(g.index[g == ALWAYS]) == [0, 1, 2, 3]


def test_unidentified_pair_is_nan_and_left_out_of_type_average():
    # cohort 6 units lose period 5, the last pre-period inside the 4-vs-6
    # window (t < 6) where cohort 4 is treated; the never-treated are not
    # in that 2x2, so year 4-5 effects absorb the treatment there
    drop = [(u, 5) for u in range(7, 11)]
    df = _panel(drop=drop)
    df = df[~((df["unit"].between(7, 10)) & (df["t"] == 4))]
    r = _bacon(df, balance="observed_rows")
    dec = r["decomposition"]
    bad = dec[~dec["identified"]]
    assert len(bad) >= 1 and bad["estimate"].isna().all()
    ok = dec[dec["identified"]]
    assert r["weighted_sum"] == pytest.approx(
        float((ok["weight"] * ok["estimate"]).sum()), abs=1e-15
    )
    s = r["summary"].set_index("type")
    for kind, part in dec.groupby("type"):
        known = part[part["identified"]]
        if known.empty:
            continue
        expect = (known["weight"] * known["estimate"]).sum() / known["weight"].sum()
        assert s.loc[kind, "estimate"] == pytest.approx(expect, abs=1e-14)
        assert s.loc[kind, "weight"] == pytest.approx(part["weight"].sum(), abs=1e-15)


@pytest.mark.parametrize("kw", [{"covariates": ["x"]}, {"weights": "w"}])
def test_observed_rows_refuses_covariates_and_weights(kw):
    df = _panel().assign(x=1.0, w=1.0)
    df["x"] = np.arange(len(df)) % 7
    with pytest.raises(MethodIncompatibility, match="observed_rows"):
        _bacon(df, balance="observed_rows", **kw)


def test_unknown_balance_value_lists_the_options():
    with pytest.raises(MethodIncompatibility, match="observed_rows"):
        _bacon(_panel(), balance="pairwise")


def test_default_still_refuses_an_unbalanced_panel():
    with pytest.raises(MethodIncompatibility, match="Unbalanced"):
        _bacon(_panel(drop=[(4, 2)]))
