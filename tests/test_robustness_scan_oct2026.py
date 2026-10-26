"""Option scan of robustness/ (2026-10-27).

Each test asserts the correct behaviour of an option combination that was
wrong before: the interval level of ``sp.spec_curve`` under ``alpha``, the
t(G - 1) reference of the clustered check in ``sp.robustness_report``, the
sample of the unclustered specifications when ``cluster_var`` has missing
ids, the breakpoints of ``sp.copula_sensitivity`` / ``sp.survival_sensitivity``
when the interval crosses zero between grid values, and specifications
with linearly dependent regressors in ``sp.spec_curve``.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp


def _frame(n=200, seed=0, n_clusters=5):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "g": np.arange(n) % n_clusters,
        }
    )
    df["y"] = 1.0 + 0.3 * df["x"] + 0.5 * df["a"] + rng.normal(size=n)
    return df


# A1 -------------------------------------------------------------------------
@pytest.mark.parametrize("alpha", [0.10, 0.01])
def test_spec_curve_ci_level_follows_alpha(alpha):
    df = _frame()
    res = sp.spec_curve(
        data=df, y="y", x="x", controls=[[]], se_types=["nonrobust"], alpha=alpha
    )
    row = res.results_df.iloc[0]
    df_resid = int(row["nobs"]) - 2
    half = stats.t.ppf(1 - alpha / 2, df_resid) * row["se"]
    assert row["ci_upper"] - row["estimate"] == pytest.approx(half, rel=1e-12)
    assert row["estimate"] - row["ci_lower"] == pytest.approx(half, rel=1e-12)


def test_spec_curve_ci_level_follows_alpha_with_fixed_effects():
    df = _frame()
    df["f"] = np.arange(len(df)) % 10
    res = sp.spec_curve(
        data=df,
        y="y",
        x="x",
        controls=[[]],
        fe=[["f"]],
        se_types=["cluster"],
        cluster_var="g",
        alpha=0.10,
    )
    row = res.results_df.iloc[0]
    G = df["g"].nunique()
    # the interval and the p-value refer to the same t(G - 1)
    half = stats.t.ppf(0.95, G - 1) * row["se"]
    assert row["ci_upper"] - row["estimate"] == pytest.approx(half, rel=1e-12)
    p = 2 * stats.t.sf(abs(row["estimate"] / row["se"]), G - 1)
    assert row["pvalue"] == pytest.approx(p, rel=1e-12)


# A2 -------------------------------------------------------------------------
def test_robustness_report_cluster_pvalue_uses_G_minus_1():
    df = _frame(n_clusters=5)
    rep = sp.robustness_report(df, formula="y ~ x + a", x="x", cluster_var="g")
    row = rep.results_df.set_index("check").loc["SE: Clustered (g)"]
    G = df["g"].nunique()
    p_ref = 2 * stats.t.sf(abs(row["estimate"] / row["se"]), G - 1)
    assert row["pvalue"] == pytest.approx(p_ref, rel=1e-10)
    half = stats.t.ppf(0.975, G - 1) * row["se"]
    assert row["ci_upper"] - row["estimate"] == pytest.approx(half, rel=1e-10)
    # the same numbers as sp.regress with the same CR1 variance
    fit = sp.regress("y ~ x + a", df, cluster="g")
    assert row["se"] == pytest.approx(float(fit.std_errors["x"]), rel=1e-10)
    assert row["pvalue"] == pytest.approx(float(fit.pvalues["x"]), rel=1e-8)


# A3 -------------------------------------------------------------------------
def test_spec_curve_missing_cluster_id_does_not_shrink_unclustered_specs():
    df = _frame()
    df["cl"] = df["g"].astype(float)
    df.loc[:19, "cl"] = np.nan  # 20 rows without a cluster id
    res = sp.spec_curve(
        data=df, y="y", x="x", controls=[[]], se_types=["nonrobust"], cluster_var="cl"
    )
    rows = res.results_df.set_index("se_type")
    fit = sp.regress("y ~ x", df)
    assert int(rows.loc["nonrobust", "nobs"]) == len(df)
    assert rows.loc["nonrobust", "estimate"] == pytest.approx(
        float(fit.params["x"]), rel=1e-10
    )
    # the clustered specification still drops the rows without an id
    fit_cl = sp.regress("y ~ x", df.dropna(subset=["cl"]), cluster="cl")
    assert int(rows.loc["cluster", "nobs"]) == len(df) - 20
    assert rows.loc["cluster", "se"] == pytest.approx(
        float(fit_cl.std_errors["x"]), rel=1e-10
    )


# A4 -------------------------------------------------------------------------
def test_copula_breakpoint_catches_sign_flip_between_grid_points():
    est, se = 0.32, 0.001
    res = sp.copula_sensitivity(est, se)  # grid -0.5 .. 0.5 step 0.05
    # analytic crossing: (est - z se) / (sigma_u sigma_y) = 0.318; rho = 0.30
    # leaves [0.018, 0.022], rho = 0.35 gives [-0.032, -0.028]
    assert res.breakpoint is not None
    assert abs(res.breakpoint) == pytest.approx(0.35, abs=1e-12)
    assert "robust" not in res.interpretation


def test_copula_breakpoint_of_a_negative_effect():
    res = sp.copula_sensitivity(-0.32, 0.001)
    assert res.breakpoint == pytest.approx(-0.35, abs=1e-12)


def test_survival_breakpoint_catches_sign_flip_between_grid_points():
    res = sp.survival_sensitivity(0.4, 0.001)  # grid 1.0 .. 3.0 step 0.1
    # analytic crossing: exp(|log_hr| - z se) = 1.489
    assert res.breakpoint is not None
    assert res.breakpoint == pytest.approx(1.5, abs=1e-12)
    assert "robust" not in res.interpretation
    protective = sp.survival_sensitivity(-0.4, 0.001)
    assert protective.breakpoint == pytest.approx(1.5, abs=1e-12)


# A5 -------------------------------------------------------------------------
def test_spec_curve_numerically_collinear_controls_are_left_out_and_recorded():
    df = _frame()
    df["c"] = df["a"] + df["b"]  # redundant up to floating-point rounding
    with pytest.warns(sp.AssumptionWarning, match="linearly dependent"):
        res = sp.spec_curve(
            data=df,
            y="y",
            x="x",
            controls=[["a", "b"], ["a", "b", "c"]],
            se_types=["nonrobust"],
        )
    # only the identified specification is on the curve
    assert res.n_specs == 1
    fit = sp.regress("y ~ x + a + b", df)
    row = res.results_df.iloc[0]
    assert row["estimate"] == pytest.approx(float(fit.params["x"]), rel=1e-10)
    (dropped,) = res.model_info["dropped_specs"]
    assert dropped["controls"] == "a, b, c"
    assert dropped["collinear"] == ["c"]


def test_spec_curve_without_dependence_warns_nothing():
    df = _frame()
    with warnings.catch_warnings():
        warnings.simplefilter("error", sp.AssumptionWarning)
        res = sp.spec_curve(
            data=df, y="y", x="x", controls=[[], ["a"]], se_types=["nonrobust"]
        )
    assert res.model_info["dropped_specs"] == []


# A6 -------------------------------------------------------------------------
# R sensemakr 0.1.6, black box: adjusted_estimate / adjusted_se with r2dz.x,
# r2yz.dx from k = 1 and the benchmark R2 of 0.1, dof = 30. sensemakr is not
# consistent about the interval: ovb_bounds() refers the adjusted SE to
# t(dof) (the convention kept, pinned by test_misc_sens_R_parity.py), while
# the stand-alone adjusted_ci() uses t(dof - 1) (0.0230879796302288 here).
_SENSEMAKR = {
    "adjusted_estimate": 0.227831216351297,
    "adjusted_se": 0.100107700623568,
}


def test_calibrate_adjusted_ci_follows_ovb_bounds():
    dof = 30
    res = sp.calibrate_confounding_strength(
        0.3,
        0.1,
        observed_r2_outcome=0.1,
        observed_r2_treatment=0.1,
        dof=dof,
        multipliers=[1.0],
    )
    row = res.curve.iloc[0]
    for key, value in _SENSEMAKR.items():
        assert row[key] == pytest.approx(value, rel=1e-12), key
    crit = stats.t.ppf(0.975, dof)
    assert row["adjusted_ci_low"] == pytest.approx(
        row["adjusted_estimate"] - crit * row["adjusted_se"], rel=1e-12
    )


# A7 -------------------------------------------------------------------------
@pytest.mark.parametrize(
    "rhs", ["x + C(g)", "x + I(a**2)", "x + np.log(b + 10)", "x + x:a"]
)
@pytest.mark.parametrize("fn", ["robustness_report", "subgroup_analysis"])
def test_formula_terms_that_are_not_columns_are_refused_by_name(fn, rhs):
    df = _frame()
    df["s"] = (np.arange(len(df)) % 2).astype(float)
    kwargs = {"by": {"S": "s"}} if fn == "subgroup_analysis" else {}
    with pytest.raises(sp.MethodIncompatibility, match="not a column"):
        getattr(sp, fn)(df, formula=f"y ~ {rhs}", x="x", **kwargs)


@pytest.mark.parametrize("fn", ["robustness_report", "subgroup_analysis"])
def test_absorbed_fixed_effects_are_refused_not_dropped(fn):
    # the '| g' part used to be parsed away and every check ran without it
    df = _frame()
    df["s"] = (np.arange(len(df)) % 2).astype(float)
    kwargs = {"by": {"S": "s"}} if fn == "subgroup_analysis" else {}
    with pytest.raises(sp.MethodIncompatibility, match="fixed effects"):
        getattr(sp, fn)(df, formula="y ~ x + a | g", x="x", **kwargs)
