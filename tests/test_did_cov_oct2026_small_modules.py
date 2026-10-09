"""Coverage campaign (did, Oct 2026) -- guards of the smaller DiD modules.

``twowayfeweights``, ``did_balance``, ``xtevent`` and ``cdlz_bunching``. Each
refusal is checked for exception type and message; the numerical branches
are checked against a closed form (two-way within residual, normalized
difference, delta-method standard error).
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility

tw_mod = importlib.import_module("statspai.did.twowayfeweights")
bal_mod = importlib.import_module("statspai.did.balance")
xt_mod = importlib.import_module("statspai.did.xtevent")
cdlz_mod = importlib.import_module("statspai.did.cdlz_bunching")


def _staggered(seed=0, n_units=30, T=8, cohorts=(4, 6, 0), effect=2.0, noise=0.3):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        g = cohorts[i % len(cohorts)]
        a = rng.normal()
        x = rng.normal()
        for t in range(1, T + 1):
            d = float(g > 0 and t >= g)
            rows.append(
                {
                    "unit": i,
                    "t": t,
                    "g": g,
                    "d": d,
                    "x": x + 0.2 * t * (g > 0),
                    "z": float(i % 2),
                    "y": a + 0.3 * t + effect * d + noise * rng.normal(),
                }
            )
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
#  twowayfeweights
# ══════════════════════════════════════════════════════════════════════


def _big_codes(n_units=7600, n_times=8):
    u = np.repeat(np.arange(n_units), n_times)
    t = np.tile(np.arange(n_times), n_units)
    return u, t


def test_fe_residuals_alternating_projection_balanced_closed_form():
    u, t = _big_codes()
    n = u.size
    # large enough that the dummy design is not built
    assert n * (7600 + 8 - 1) > tw_mod._EXACT_CELLS
    rng = np.random.default_rng(0)
    v = rng.normal(size=n) + 0.01 * u + 0.5 * t
    out = tw_mod._fe_residuals(v, [u, t], np.ones(n))
    assert out.shape == (n, 1)  # a vector is treated as one column
    m = v.reshape(7600, 8)
    closed = m - m.mean(1, keepdims=True) - m.mean(0, keepdims=True) + m.mean()
    # balanced panel, equal weights: the two-way within transformation
    np.testing.assert_allclose(out[:, 0], closed.ravel(), atol=1e-10)


def test_fe_residuals_alternating_projection_weighted_is_the_wls_residual():
    u, t = _big_codes()
    n = u.size
    rng = np.random.default_rng(1)
    v = rng.normal(size=(n, 2))
    w = rng.uniform(0.5, 2.0, size=n)
    out = tw_mod._fe_residuals(v, [u, t], w)
    # (i) weighted-orthogonal to every unit and period dummy; the sweep
    # stops at a relative change of 1e-13, 1e-8 on sums of ~8 terms is loose
    for c, k in ((u, 7600), (t, 8)):
        for j in range(2):
            sums = np.bincount(c, weights=w * out[:, j], minlength=k)
            assert np.abs(sums).max() < 1e-8
    # (ii) what was removed is additive in unit and period: its double
    # differences vanish. Together with (i) this is the WLS residual.
    fitted = (v - out)[:, 0].reshape(7600, 8)
    dd = fitted[1:, 1:] - fitted[1:, :-1] - fitted[:-1, 1:] + fitted[:-1, :-1]
    assert np.abs(dd).max() < 1e-9


def test_twfe_weights_sensitivity_undefined_with_a_single_weighted_cell():
    cells = pd.DataFrame({"nat_weight": [1.0, 0.0], "W": [1.0, 0.0]})
    out = tw_mod._sensitivity(cells, 1.0, False)
    assert np.isnan(out[0]) and np.isnan(out[1])


def test_twfe_weights_random_weights_test_needs_three_cells():
    cells = pd.DataFrame(
        {
            "group": [0, 1, 2, 3],
            "nat_weight": [0.5, 0.5, 0.0, 0.0],
            "W": [1.0, 1.0, 0.0, 0.0],
            "v": [1.0, 2.0, 3.0, 4.0],
        }
    )
    tab = tw_mod._random_weights_test(cells, ["v"])
    # two weighted cells cannot carry a clustered regression
    assert len(tab) == 1 and tab.iloc[0, 1:].isna().all()


def test_twfe_weights_guards():
    df = _staggered()
    kw = dict(y="y", group="unit", time="t", treat="d")
    with pytest.raises(MethodIncompatibility, match="name of a column"):
        sp.twowayfeweights(df.assign(W=1.0), test_random_weights=["W"], **kw)
    with pytest.raises(MethodIncompatibility, match="columns not found"):
        sp.twowayfeweights(df, covariates=["nope"], **kw)
    with pytest.raises(DataInsufficient, match="no complete observation"):
        sp.twowayfeweights(df.assign(y=np.nan), **kw)


def test_twfe_weights_sum_to_one_and_reproduce_the_coefficient():
    # Homogeneous effect: every weighting of the cell effects gives it back.
    df = _staggered(noise=0.0)
    r = sp.twowayfeweights(df, y="y", group="unit", time="t", treat="d")
    assert r.detail["weight"].sum() == pytest.approx(1.0, abs=1e-10)
    assert (r.detail.loc[r.detail["D"] == 0, "weight"] == 0).all()
    assert r.estimate == pytest.approx(2.0, abs=1e-10)


# ══════════════════════════════════════════════════════════════════════
#  did_balance
# ══════════════════════════════════════════════════════════════════════

B_KW = dict(g="g", t="t", i="unit")


def test_balance_normalized_difference_closed_form_and_text():
    df = _staggered()
    r = sp.did_balance(df, ["z"], cohort=4, **B_KW)
    base = df[df["t"] == 3].groupby("unit").first()
    tr, co = base.loc[base["g"] == 4, "z"], base.loc[base["g"] == 0, "z"]
    nd = (tr.mean() - co.mean()) / np.sqrt((tr.var(ddof=1) + co.var(ddof=1)) / 2)
    row = r.levels.set_index("covariate").loc["z"]
    # Imbens-Rubin normalized difference, sample variances
    assert row["norm_diff"] == pytest.approx(nd, rel=1e-10)
    assert r.max_abs_norm_diff == pytest.approx(abs(nd), rel=1e-10)
    # z never changes: the changes panel has nothing to say about it
    assert r.diagnostics["constant_in_changes"] == ["z"]
    text = r.summary()
    assert "variable" in text and "n.diff" in text
    tex = r.to_latex()
    assert "Variable & Comparison & Treated & Norm. diff." in tex


def test_balance_summary_says_so_when_nothing_is_flagged():
    df = _staggered()
    r = sp.did_balance(df, ["z"], cohort=4, threshold=50.0, **B_KW)
    assert r.flagged == []
    text = r.summary()
    assert "No covariate breaches |norm. diff| > 50.0" in text
    assert "not proof" in text


def test_balance_not_yet_treated_comparison_adds_the_later_cohort():
    df = _staggered()
    nev = sp.did_balance(df, ["x"], cohort=4, **B_KW)
    nyt = sp.did_balance(df, ["x"], cohort=4, control_group="notyettreated", **B_KW)
    n_later = df.loc[df["g"] == 6, "unit"].nunique()
    assert nyt.n_comparison == nev.n_comparison + n_later
    assert nyt.n_treated == nev.n_treated


def test_balance_weighted_property_reads_the_weighted_column():
    df = _staggered().assign(w=1.0)
    plain = sp.did_balance(df, ["x"], cohort=4, **B_KW)
    wtd = sp.did_balance(df, ["x"], cohort=4, weights="w", **B_KW)
    assert wtd.weighted
    # unit weights: the weighted statistic is the unweighted one
    assert wtd.max_abs_norm_diff == pytest.approx(plain.max_abs_norm_diff, rel=1e-10)


@pytest.mark.parametrize(
    "covs,kwargs,exc,match",
    [
        ([], {}, MethodIncompatibility, "at least one covariate"),
        (["x"], dict(weights="nope"), MethodIncompatibility, "nope"),
        (["x"], dict(control_group="clean"), MethodIncompatibility, "control_group"),
        (["x"], dict(cohort=5), MethodIncompatibility, "not a treated cohort"),
        (["x"], dict(cohort=4, base_period=8), DataInsufficient, "after the base"),
    ],
)
def test_balance_guards(covs, kwargs, exc, match):
    with pytest.raises(exc, match=match):
        sp.did_balance(_staggered(), covs, **{**B_KW, **kwargs})


def test_balance_data_guards():
    df = _staggered()
    with pytest.raises(DataInsufficient, match="No treated cohorts"):
        sp.did_balance(df.assign(g=0), ["x"], **B_KW)
    first = _staggered(cohorts=(1, 0))
    with pytest.raises(DataInsufficient, match="no pre-treatment period"):
        sp.did_balance(first, ["x"], **B_KW)
    only = _staggered(cohorts=(4,))
    with pytest.raises(DataInsufficient, match="Empty treated"):
        sp.did_balance(only, ["x"], **B_KW)
    neg = df.assign(w=-1.0)
    with pytest.raises(MethodIncompatibility, match="negative values"):
        sp.did_balance(neg, ["x"], weights="w", **B_KW)
    gap = df.assign(w=np.nan)
    with pytest.raises(MethodIncompatibility, match="missing/non-finite"):
        sp.did_balance(gap, ["x"], weights="w", **B_KW)
    with pytest.raises(DataInsufficient, match="No covariate had enough"):
        sp.did_balance(df.assign(x=np.nan), ["x"], **B_KW)


def test_balance_private_formatters():
    assert bal_mod._fmt_nd(np.nan) == "nan"
    assert bal_mod._fmt_nd(np.inf) == "+inf"
    assert bal_mod._fmt_nd(-np.inf) == "-inf"
    assert bal_mod._fmt_nd(0.12345) == "0.123"
    assert np.isnan(bal_mod._wvar(np.array([1.0, 2.0]), np.zeros(2)))


# ══════════════════════════════════════════════════════════════════════
#  xtevent
# ══════════════════════════════════════════════════════════════════════

X_KW = dict(y="y", policy="d", panel="unit", time="t")


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(window=2, engine="xtreg"), "engine must be"),
        (dict(window=2, absorb="z"), "requires engine='reghdfe'"),
        (dict(window=2, vce="hc3"), "vce must be 'robust'"),
        (dict(window=2, covariates=["nope"]), "columns not found"),
        (dict(), "pass window="),
        (dict(window=0), "positive integer"),
        (dict(window="ab"), "positive integer or a pair"),
        (dict(window=(-1, 1), norm=-3), "inside the window"),
    ],
)
def test_xtevent_guards(kwargs, match):
    with pytest.raises(MethodIncompatibility, match=match):
        sp.xtevent(_staggered(), **{**X_KW, **kwargs})


def test_xtevent_needs_integer_periods_and_complete_windows():
    df = _staggered()
    with pytest.raises(MethodIncompatibility, match="integer periods"):
        sp.xtevent(df.assign(t=df["t"] + 0.5), window=1, **X_KW)
    # an 8-period panel has no row with 6 leads and 6 lags
    with pytest.raises(DataInsufficient, match="every lead and lag"):
        sp.xtevent(df, window=6, **X_KW)


def test_xtevent_robust_reghdfe_matches_areg_point_estimates():
    df = _staggered()
    a = sp.xtevent(df, window=1, **X_KW)
    b = sp.xtevent(df, window=1, engine="reghdfe", vce="robust", **X_KW)
    ea = a.model_info["event_study"] if "event_study" in a.model_info else a.detail
    eb = b.model_info["event_study"] if "event_study" in b.model_info else b.detail
    col = "att" if "att" in ea.columns else "estimate"
    # same regression, two absorbers
    np.testing.assert_allclose(ea[col], eb[col], atol=1e-8)


def test_xtevent_two_sided_p():
    assert np.isnan(xt_mod._two_sided_p(np.nan, 10.0))
    assert xt_mod._two_sided_p(1.96, np.inf) == pytest.approx(0.05, abs=1e-4)
    assert xt_mod._two_sided_p(2.0, 5.0) > xt_mod._two_sided_p(2.0, np.inf)


# ══════════════════════════════════════════════════════════════════════
#  cdlz_bunching
# ══════════════════════════════════════════════════════════════════════


def _bunching_inputs():
    terms = pd.DataFrame(
        {
            "term": ["b_m1_0", "b_p0_0", "b_m1_1", "b_p0_1", "b_m1_pre"],
            "year": [0, 0, 1, 1, -1],
            "bin": [-1, 0, -1, 0, -1],
        }
    )
    params = pd.Series([-0.02, 0.015, -0.03, 0.02, 0.001], index=terms["term"])
    cov = pd.DataFrame(
        np.diag([1e-5, 2e-5, 1e-5, 2e-5, 1e-5]),
        index=params.index,
        columns=params.index,
    )
    scal = dict(epop=0.6, below_share=0.08, wage_bill=5.0, pct_mw=0.1, mw_level=8.0)
    return terms, params, cov, scal


class _Fit:
    """A fitted-result stand-in: ``params`` plus one covariance accessor."""

    def __init__(self, params, cov, style):
        self.params = params
        if style == "cov_params":
            self.cov_params = lambda: cov
        elif style == "vcov_method":
            self.vcov = lambda: cov.to_numpy()
        else:
            self.vcov = cov


@pytest.mark.parametrize("style", ["cov_params", "vcov_method", "vcov_attr"])
def test_bunching_from_a_fitted_result_equals_params_and_covariance(style):
    terms, params, cov, scal = _bunching_inputs()
    ref = sp.cdlz_bunching(terms=terms, params=params, covariance=cov, **scal)
    got = sp.cdlz_bunching(_Fit(params, cov, style), terms=terms, **scal)
    pd.testing.assert_frame_equal(got.detail, ref.detail)


def test_bunching_linear_statistics_closed_form():
    terms, params, cov, scal = _bunching_inputs()
    r = sp.cdlz_bunching(terms=terms, params=params, covariance=cov, **scal)
    tab = r.detail.set_index("statistic")
    # 4 bins per wage unit, averaged over the two post years, per capita
    s = 4.0 / 2 / 0.6
    assert tab.loc["missing_jobs_below", "estimate"] == pytest.approx(
        s * (-0.02 - 0.03), rel=1e-12
    )
    assert tab.loc["excess_jobs_above", "estimate"] == pytest.approx(
        s * (0.015 + 0.02), rel=1e-12
    )
    assert tab.loc["missing_jobs_below", "se"] == pytest.approx(
        s * np.sqrt(2e-5), rel=1e-12
    )


def test_bunching_guards():
    terms, params, cov, scal = _bunching_inputs()
    with pytest.raises(MethodIncompatibility, match="terms needs columns"):
        sp.cdlz_bunching(
            terms=terms.drop(columns="bin"), params=params, covariance=cov, **scal
        )
    extra = pd.concat(
        [terms, pd.DataFrame({"term": ["ghost"], "year": [0], "bin": [1]})]
    )
    with pytest.raises(MethodIncompatibility, match="not in the regression"):
        sp.cdlz_bunching(terms=extra, params=params, covariance=cov, **scal)
    with pytest.raises(MethodIncompatibility, match="no post-period terms"):
        sp.cdlz_bunching(
            terms=terms, params=params, covariance=cov, post_years=[7], **scal
        )
    with pytest.raises(MethodIncompatibility, match="pass a fitted result"):
        sp.cdlz_bunching(terms=terms, params=params, **scal)
