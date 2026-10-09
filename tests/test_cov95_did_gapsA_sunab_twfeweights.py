"""Coverage gaps in ``sp.sun_abraham`` and ``sp.twowayfeweights``: input
contracts, cells with zero weight, the two-way within transformation and
the iterated fixed-effect projection.
"""

import importlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility

sa_mod = importlib.import_module("statspai.did.sun_abraham")
tw_mod = importlib.import_module("statspai.did.twowayfeweights")

SK = dict(y="y", g="g", t="time", i="unit")
TK = dict(y="y", group="unit", time="time", treat="d")


@pytest.fixture(scope="module")
def panel():
    df = sp.dgp_did(n_units=60, n_periods=8, staggered=True, seed=3)
    df["g"] = df["first_treat"].fillna(0).astype(int)
    df["d"] = df["treated"]
    rng = np.random.default_rng(0)
    df["x"] = rng.normal(size=len(df))
    df["z"] = (df["unit"] % 3).astype(float)
    return df


# ---------------------------------------------------------------------------
#  sun_abraham
# ---------------------------------------------------------------------------


def test_sa_contracts(panel):
    with pytest.raises(ValueError, match="Reference group is empty"):
        sp.sun_abraham(panel[panel["g"] > 0], **SK)
    with pytest.raises(MethodIncompatibility, match="bounds reversed"):
        sp.sun_abraham(panel, event_window=(3, -3), **SK)
    infinite = panel.copy()
    infinite.loc[4, "y"] = np.inf
    with pytest.raises(MethodIncompatibility, match="must be non-missing"):
        sp.sun_abraham(infinite, **SK)
    holed = panel.assign(cl=panel["unit"].astype(float))
    holed.loc[holed["unit"] == 2, "cl"] = np.nan
    with pytest.raises(MethodIncompatibility, match="must be non-missing"):
        sp.sun_abraham(holed, cluster="cl", **SK)


def test_sa_window_without_leads_has_no_pretrend_test(panel):
    full = sp.sun_abraham(panel, **SK)
    post_only = sp.sun_abraham(panel, event_window=(0, 3), **SK)
    assert full.model_info["pretrend_test"] is not None
    assert post_only.model_info["pretrend_test"] is None
    es = post_only.model_info["event_study"]
    assert list(es["relative_time"]) == [0, 1, 2, 3]
    # window_rule='report' only trims the table: the regression is the same.
    ref = full.model_info["event_study"].set_index("relative_time")["att"]
    assert np.allclose(es["att"], ref.loc[[0, 1, 2, 3]], atol=1e-12)


def test_sa_zero_weight_cells_drop_out_like_deleted_rows(panel):
    df = panel.assign(w=1.0)
    zeroed = (df["g"] > 0) & (df["time"] - df["g"] == 2)
    df.loc[zeroed, "w"] = 0.0
    weighted = sp.sun_abraham(df, weights="w", **SK)
    deleted = sp.sun_abraham(df[~zeroed], weights="w", **SK)
    es_w = weighted.model_info["event_study"].set_index("relative_time")
    es_d = deleted.model_info["event_study"].set_index("relative_time")
    # Relative time 2 has no mass, so it is not reported at all ...
    assert 2 not in es_w.index and 1 in es_w.index and 3 in es_w.index
    assert list(es_w.index) == list(es_d.index)
    # ... and every other estimate is the one from the data without it.
    assert np.allclose(es_w["att"], es_d["att"], atol=1e-10)
    assert weighted.estimate == pytest.approx(deleted.estimate, abs=1e-10)
    assert weighted.model_info["att_fixest_att"] == pytest.approx(
        deleted.model_info["att_fixest_att"], abs=1e-10
    )


def _dummy_residual(x, u, t, w=None):
    """Residual of x on unit and period dummies by (weighted) least squares."""
    D = np.column_stack(
        [pd.get_dummies(u).to_numpy(float), pd.get_dummies(t).to_numpy(float)[:, 1:]]
    )
    sw = np.ones(len(x)) if w is None else np.sqrt(w)
    coef, *_ = np.linalg.lstsq(D * sw[:, None], x * sw, rcond=None)
    return x - D @ coef


@pytest.fixture(scope="module")
def unbalanced():
    rng = np.random.default_rng(5)
    u = np.repeat(np.arange(12), 6)
    t = np.tile(np.arange(6), 12)
    keep = rng.uniform(size=u.size) > 0.2
    u, t = u[keep], t[keep]
    x = rng.normal(size=u.size) + 0.3 * u + 0.5 * t
    w = rng.uniform(0.5, 2.0, size=u.size)
    return x, u, t, w


# ---------------------------------------------------------------------------
#  twowayfeweights
# ---------------------------------------------------------------------------


def test_iterated_projection_matches_the_exact_design(panel, monkeypatch):
    kw = dict(covariates=["x"], test_random_weights=["z"])
    exact = sp.twowayfeweights(panel, **TK, **kw)
    monkeypatch.setattr(tw_mod, "_EXACT_CELLS", 0)
    iterated = sp.twowayfeweights(panel, **TK, **kw)
    assert iterated.estimate == pytest.approx(exact.estimate, abs=1e-10)
    assert iterated.se == pytest.approx(exact.se, abs=1e-10)
    assert np.allclose(iterated.detail["weight"], exact.detail["weight"], atol=1e-10)
    for key in ("n_negative", "sum_negative", "sigma_fe"):
        assert iterated.model_info[key] == pytest.approx(exact.model_info[key])
    # The weights of the decomposition sum to one.
    assert float(exact.detail["weight"].sum()) == pytest.approx(1.0, abs=1e-10)


def test_fe_residuals_accepts_a_vector():
    rng = np.random.default_rng(0)
    v = rng.normal(size=20)
    codes = np.repeat(np.arange(4), 5)
    out = tw_mod._fe_residuals(v, [codes], np.ones(20))
    assert out.shape == (20, 1)
    assert np.allclose(out[:, 0], v - v.reshape(4, 5).mean(axis=1)[codes])


def test_single_treated_cell_has_no_sensitivity_measures(panel):
    df = panel.copy()
    df["d"] = ((df["unit"] == 0) & (df["time"] == df["time"].max())).astype(float)
    res = sp.twowayfeweights(df, test_random_weights=["z"], **TK)
    info = res.model_info
    assert info["n_treated_cells"] == 1
    assert (info["n_positive"], info["n_negative"]) == (1, 0)
    assert info["sum_positive"] == pytest.approx(1.0)
    # One weight: no dispersion of weights, so sigma_fe is undefined, and
    # no regression of a cell variable on it either.
    assert np.isnan(info["sigma_fe"]) and np.isnan(info["sigma_fe_2"])
    random_table = info["random_weights"]
    assert list(random_table.index) == ["z"]
    assert random_table.loc["z"].isna().all()


def test_twfeweights_contracts(panel):
    with pytest.raises(MethodIncompatibility, match=r"variable\(s\) \['W'\]") as err:
        sp.twowayfeweights(panel.assign(W=1.0), test_random_weights=["W"], **TK)
    assert err.value.diagnostics == {"clash": ["W"]}
    with pytest.raises(DataInsufficient, match="no complete observation"):
        sp.twowayfeweights(panel.assign(y=np.nan), **TK)


def test_first_difference_regression_on_one_group_is_not_identified():
    # One group, one observation per period: the period effects absorb the
    # differenced treatment exactly.
    df = pd.DataFrame(
        {
            "unit": 0,
            "time": [1, 2, 3, 4],
            "D": [0.0, 0.0, 1.0, 1.0],
            "dd": [np.nan, 0.0, 1.0, 0.0],
            "dy": [np.nan, 0.2, 1.3, 0.1],
        }
    )
    with pytest.raises(DataInsufficient, match="no variation left after the fixed"):
        sp.twowayfeweights(
            df,
            y="dy",
            group="unit",
            time="time",
            treat="dd",
            type="fdTR",
            treat_level="D",
        )


def test_twowayfeweights_refuses_a_treatment_the_group_effects_absorb():
    # A treatment that is constant within group leaves only rounding noise
    # after the group effects. The residual sum of squares is then about
    # 1e-28, not zero, and the ratio used to come back as -4.1e12.
    df = sp.dgp_did(n_units=30, n_periods=6, staggered=True, seed=3)
    df["d"] = (df["unit"] < 10).astype(int)
    with pytest.raises(DataInsufficient, match="no variation left"):
        sp.twowayfeweights(df, y="y", group="unit", time="time", treat="d")
