"""Coverage tests for statspai.panel.panel_diagnostics (live-but-unwired).

panel_diagnostics.py holds the private helpers (`_hausman_from_data`,
`_bp_lm_test`, `_f_test_effects`, `_pesaran_cd`, `_within_estimator`,
`_re_estimator`) that back the PanelResults diagnostic methods. They are
fully functional but only `_`-prefixed, so they are reached here by
importing the module directly and feeding real synthetic panel data.
We assert structural properties (shapes, keys, finite stats, valid
p-values, recommendation strings) — not fabricated numbers.
"""

import numpy as np
import pandas as pd
import pytest

from statspai.panel import panel_diagnostics as pd_diag


@pytest.fixture
def panel_df():
    rng = np.random.default_rng(7)
    n_id, T = 30, 8
    rows = []
    for i in range(n_id):
        alpha = rng.normal(3, 2)
        for t in range(T):
            x1 = rng.normal()
            x2 = rng.normal()
            y = alpha + 1.5 * x1 - 0.7 * x2 + rng.normal(0, 0.5)
            rows.append({"id": i, "time": t, "y": y, "x1": x1, "x2": x2})
    return pd.DataFrame(rows)


def test_within_estimator_shapes(panel_df):
    beta, vcov = pd_diag._within_estimator(panel_df, "y", ["x1", "x2"], "id")
    assert beta.shape == (2,)
    assert vcov.shape == (2, 2)
    # FE recovers slopes well (alpha_i wiped out)
    assert abs(beta[0] - 1.5) < 0.3
    assert abs(beta[1] + 0.7) < 0.3
    assert np.all(np.diag(vcov) >= 0)


def test_re_estimator_shapes(panel_df):
    beta_re, vcov_re = pd_diag._re_estimator(panel_df, "y", ["x1", "x2"], "id")
    assert beta_re.shape == (2,)
    assert vcov_re.shape == (2, 2)
    assert np.all(np.diag(vcov_re) >= 0)


def test_hausman_from_data(panel_df):
    out = pd_diag._hausman_from_data(panel_df, "y", ["x1", "x2"], "id", "time")
    for key in (
        "statistic",
        "df",
        "pvalue",
        "recommendation",
        "beta_fe",
        "beta_re",
        "interpretation",
    ):
        assert key in out
    assert out["df"] == 2
    assert out["statistic"] >= 0
    assert 0.0 <= out["pvalue"] <= 1.0
    assert out["recommendation"] in ("FE", "RE")
    assert isinstance(out["beta_fe"], pd.Series)
    assert "chi2" in out["interpretation"]


def test_hausman_alpha_one_forces_fe(panel_df):
    # alpha=1.0 => pvalue < alpha always true => recommends FE branch
    out = pd_diag._hausman_from_data(
        panel_df, "y", ["x1", "x2"], "id", "time", alpha=1.0
    )
    assert out["recommendation"] == "FE"
    assert "Fixed Effects" in out["interpretation"]


def test_bp_lm_test(panel_df):
    out = pd_diag._bp_lm_test(panel_df, "y", ["x1", "x2"], "id", "time")
    assert out["df"] == 1
    assert out["statistic"] >= 0
    assert 0.0 <= out["pvalue"] <= 1.0
    assert out["recommendation"] in ("RE", "Pooled OLS")
    assert "LM" in out["interpretation"]


def test_f_test_effects(panel_df):
    out = pd_diag._f_test_effects(panel_df, "y", ["x1", "x2"], "id", "time")
    for key in ("statistic", "df1", "df2", "pvalue", "interpretation"):
        assert key in out
    assert out["df1"] == 30 - 1
    assert np.isfinite(out["statistic"])
    assert "F(" in out["interpretation"]


def test_f_test_effects_insufficient_df():
    # 2 units, 1 period each, 2 regressors -> df2 = n - N - k <= 0
    df = pd.DataFrame(
        {
            "id": [0, 1],
            "time": [0, 0],
            "y": [1.0, 2.0],
            "x1": [0.5, 1.5],
            "x2": [0.1, 0.2],
        }
    )
    out = pd_diag._f_test_effects(df, "y", ["x1", "x2"], "id", "time")
    assert np.isnan(out["statistic"])
    assert "Insufficient" in out["interpretation"]


def test_pesaran_cd(panel_df):
    # build residuals aligned with the dataframe order
    resids = pd.Series(np.asarray(panel_df["y"]) - float(panel_df["y"].mean()))
    out = pd_diag._pesaran_cd(resids, "id", "time", panel_df)
    assert "statistic" in out and "pvalue" in out
    assert np.isfinite(out["statistic"])
    assert 0.0 <= out["pvalue"] <= 1.0
    assert "CD" in out["interpretation"]


def test_pesaran_cd_insufficient_data():
    # Only 1 entity / too few periods -> insufficient-data branch
    df = pd.DataFrame(
        {
            "id": [0, 0],
            "time": [0, 1],
            "y": [1.0, 2.0],
        }
    )
    resids = pd.Series([0.1, -0.1])
    out = pd_diag._pesaran_cd(resids, "id", "time", df)
    assert np.isnan(out["statistic"])
    assert "Insufficient" in out["interpretation"]


def test_pesaran_cd_no_valid_pairs():
    # 2 entities but only 2 common periods each -> T_common may pass but
    # per-pair < 3 leaves count == 0. Construct entities with disjoint times
    # so no overlapping pair has >= 3 observations.
    df = pd.DataFrame(
        {
            "id": [0, 0, 0, 1, 1, 1],
            "time": [0, 1, 2, 3, 4, 5],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    resids = pd.Series([0.1, -0.2, 0.3, -0.1, 0.2, -0.3])
    out = pd_diag._pesaran_cd(resids, "id", "time", df)
    assert np.isnan(out["statistic"])
    assert (
        "No valid pairs" in out["interpretation"]
        or "Insufficient" in out["interpretation"]
    )
