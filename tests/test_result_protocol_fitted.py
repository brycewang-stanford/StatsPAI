"""Result protocol for the regression family (rOpenSci RE1.3, RE4.4, RE4.7,
RE4.10, RE6.2; audit gaps #9 and #17).

Every estimator below drops incomplete rows. Its fitted values and
residuals must come back on the labels of the rows it used, so that they
line up with the data; the formula it was given must be on the result; an
iterative fit must say whether it converged and in how many iterations.
"""

import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest

import statspai as sp

matplotlib.use("Agg")

DROPPED = (1003, 1010)


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 120
    out = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "z": rng.normal(size=n),
            "id": np.repeat(np.arange(30), 4),
            "t": np.tile(np.arange(4), 30),
        }
    )
    out["d"] = out.z + rng.normal(size=n)
    out["y"] = 1 + out.x1 - out.x2 + 0.5 * out.d + rng.normal(size=n)
    out["yb"] = (out.y > 1).astype(int)
    out["yc"] = rng.poisson(np.exp(0.3 * out.x1.to_numpy()))
    out["ycens"] = np.maximum(out.y, 0.5)
    # labels that are not positions, and two incomplete rows
    out.index = out.index + 1000
    out.loc[list(DROPPED), "x1"] = np.nan
    return out


# (name, outcome column, formula, call)
CASES = [
    ("regress", "y", "y ~ x1 + x2", lambda f, d: sp.regress(f, data=d)),
    ("ivreg", "y", "y ~ x1 + (d ~ z)", lambda f, d: sp.ivreg(f, data=d)),
    ("logit", "yb", "yb ~ x1 + x2", lambda f, d: sp.logit(f, data=d)),
    ("probit", "yb", "yb ~ x1 + x2", lambda f, d: sp.probit(f, data=d)),
    ("poisson", "yc", "yc ~ x1 + x2", lambda f, d: sp.poisson(f, data=d)),
    ("nbreg", "yc", "yc ~ x1 + x2", lambda f, d: sp.nbreg(f, data=d)),
    (
        "glm",
        "yc",
        "yc ~ x1 + x2",
        lambda f, d: sp.glm(f, data=d, family="poisson"),
    ),
    ("feols", "y", "y ~ x1 + x2 | id", lambda f, d: sp.feols(f, data=d)),
    ("feols_iv", "y", "y ~ x1 | id | d ~ z", lambda f, d: sp.feols(f, data=d)),
    ("fepois", "yc", "yc ~ x1 + x2 | id", lambda f, d: sp.fepois(f, data=d)),
    (
        "feglm",
        "yb",
        "yb ~ x1 + x2 | t",
        lambda f, d: sp.feglm(f, data=d, family="logit"),
    ),
    ("qreg", "y", "y ~ x1 + x2", lambda f, d: sp.qreg(f, data=d)),
    ("tobit", "ycens", "ycens ~ x1 + x2", lambda f, d: sp.tobit(f, data=d, ll=0.5)),
    (
        "panel_re",
        "y",
        "y ~ x1 + x2",
        lambda f, d: sp.panel(d, f, entity="id", time="t", method="re"),
    ),
    (
        "panel_pooled",
        "y",
        "y ~ x1 + x2",
        lambda f, d: sp.panel(d, f, entity="id", time="t", method="pooled"),
    ),
]


def _fit(call, formula, data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return call(formula, data)


@pytest.mark.parametrize("name, ycol, formula, call", CASES, ids=[c[0] for c in CASES])
def test_series_are_on_the_estimation_sample(df, name, ycol, formula, call):
    res = _fit(call, formula, df)
    resid = res.residuals(as_series=True)
    fitted = res.fitted_values(as_series=True)
    used = df.index[df[["x1", "x2"]].notna().all(axis=1)]
    # the rows used, in the order of the stored arrays, labelled
    assert isinstance(resid, pd.Series) and isinstance(fitted, pd.Series)
    assert resid.index.equals(res.sample_index)
    assert fitted.index.equals(res.sample_index)
    assert set(resid.index) == set(used)
    assert not set(DROPPED) & set(resid.index)
    # the arrays are still the default return, with the same numbers
    assert isinstance(res.residuals(), np.ndarray)
    np.testing.assert_array_equal(res.residuals(), resid.to_numpy())
    np.testing.assert_array_equal(res.fitted_values(), fitted.to_numpy())
    # response residuals: observed = fitted + residual on each labelled row,
    # which fails if a residual is paired with another row's label
    y = df.loc[resid.index, ycol].to_numpy(dtype=float)
    np.testing.assert_allclose(
        fitted.to_numpy() + resid.to_numpy(), y, rtol=0, atol=1e-9 * np.abs(y).max()
    )


@pytest.mark.parametrize("name, ycol, formula, call", CASES, ids=[c[0] for c in CASES])
def test_formula_is_recorded(df, name, ycol, formula, call):
    assert _fit(call, formula, df).model_info["formula"] == formula


ITERATIVE = ["logit", "probit", "poisson", "nbreg", "glm", "fepois", "feglm", "tobit"]


@pytest.mark.parametrize("name", ITERATIVE)
def test_iterative_fits_report_convergence(df, name):
    _, _, formula, call = next(c for c in CASES if c[0] == name)
    mi = _fit(call, formula, df).model_info
    assert mi["converged"] is True
    assert isinstance(mi["n_iter"], int) and mi["n_iter"] >= 1


@pytest.mark.parametrize(
    "call",
    [
        lambda d: sp.logit("yb ~ x1 + x2", data=d, maxiter=1),
        lambda d: sp.probit("yb ~ x1 + x2", data=d, maxiter=1),
        lambda d: sp.glm("yc ~ x1 + x2", data=d, family="poisson", maxiter=1),
        lambda d: sp.poisson("yc ~ x1 + x2", data=d, maxiter=1),
    ],
    ids=["logit", "probit", "glm", "poisson"],
)
def test_one_iteration_is_reported_as_not_converged(df, call):
    # RE3.1: the warning can be silenced; the result still says so
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mi = call(df).model_info
    assert mi["converged"] is False
    assert mi["n_iter"] == 1


def test_other_iterative_regressions_report_convergence(df):
    complete = df.dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fits = {
            "ologit": sp.ologit(
                "yo ~ x1 + x2", data=complete.assign(yo=complete.yc.clip(0, 2))
            ),
            "gee": sp.gee("y ~ x1 + x2", data=complete, id="id", corstr="exchangeable"),
            "xtregar": sp.xtregar(complete, y="y", x=["x1", "x2"], id="id", time="t"),
            "fgls_two_step": sp.panel_fgls(
                complete, y="y", x=["x1", "x2"], id="id", time="t", panels="hetero"
            ),
            "fgls_igls": sp.panel_fgls(
                complete,
                y="y",
                x=["x1", "x2"],
                id="id",
                time="t",
                panels="hetero",
                igls=True,
            ),
        }
    for name, res in fits.items():
        assert res.model_info["converged"] is True, name
        assert int(res.model_info["n_iter"]) >= 0, name


def test_igls_stopped_early_is_flagged(df):
    complete = df.dropna()
    with pytest.warns(sp.ConvergenceWarning, match="igls did not converge"):
        res = sp.panel_fgls(
            complete,
            y="y",
            x=["x1", "x2"],
            id="id",
            time="t",
            panels="hetero",
            igls=True,
            maxiter=1,
        )
    assert res.model_info["converged"] is False


def test_panel_within_residuals_line_up_with_entities(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.panel(df, "y ~ x1 + x2", entity="id", time="t", method="fe")
    e = res.residuals(as_series=True)
    assert not set(DROPPED) & set(e.index)
    # within residuals sum to zero inside every entity; with the labels
    # mapped to the wrong rows the sums would not vanish
    sums = e.groupby(df.loc[e.index, "id"].to_numpy()).sum()
    assert np.abs(sums.to_numpy()).max() <= 1e-10 * np.abs(e.to_numpy()).max()


def test_first_difference_residuals_sit_on_the_later_period(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.panel(df, "y ~ x1 + x2", entity="id", time="t", method="fd")
    e = res.residuals(as_series=True)
    rows = df.loc[e.index]
    used = df.loc[df[["x1", "x2"]].notna().all(axis=1)]
    pairs = set(zip(used.id, used.t))
    # each first difference belongs to a row whose previous period is used too
    assert all((i, t - 1) in pairs for i, t in zip(rows.id, rows.t))
    assert len(e) == sum((i, t - 1) in pairs for i, t in pairs)


def test_dates_label_a_time_series_fit():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2001-01-01", periods=48, freq="MS")
    ts = pd.DataFrame({"x": rng.normal(size=48)}, index=idx)
    ts["y"] = 0.5 * ts.x + rng.normal(size=48)
    ts.iloc[7, 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e = sp.regress("y ~ x", data=ts).residuals(as_series=True)
    assert isinstance(e.index, pd.DatetimeIndex)
    assert e.index.equals(idx.delete(7))


def test_array_input_has_no_labels_to_give():
    rng = np.random.default_rng(2)
    X = np.column_stack([np.ones(30), rng.normal(size=30)])
    y = X @ [1.0, 2.0] + rng.normal(size=30)
    from statspai.regression.ols import OLSRegression

    res = OLSRegression(y=y, X=X, var_names=["const", "x"]).fit()
    assert res.sample_index is None
    with pytest.raises(sp.MethodIncompatibility, match="did not record"):
        res.residuals(as_series=True)


# ---------------------------------------------------------------- RE6.2


def test_fitted_plot_draws_fitted_values_with_their_interval(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.regress("y ~ x1 + x2", data=df)
    fig, ax = res.plot(type="fitted")
    band = res.predict(what="confidence")
    order = np.argsort(res.fitted_values(), kind="stable")
    line = ax.lines[0]
    np.testing.assert_array_equal(line.get_ydata(), res.fitted_values()[order])
    observed = df.loc[res.sample_index, "y"].to_numpy()[order]
    np.testing.assert_allclose(ax.collections[1].get_offsets()[:, 1], observed)
    # the shaded band runs from the lower to the upper confidence bound
    verts = ax.collections[0].get_paths()[0].vertices
    lo, hi = band["lower"].to_numpy()[order], band["upper"].to_numpy()[order]
    assert np.isclose(verts[:, 1].min(), lo.min())
    assert np.isclose(verts[:, 1].max(), hi.max())
    assert ax.get_xlabel() == "observation, ordered by fitted value"


def test_fitted_plot_prediction_band_is_wider_by_the_error_variance(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.regress("y ~ x1 + x2", data=df)
    conf = res.predict(what="confidence")
    pred = res.predict(what="prediction")
    s2 = res.data_info["rss"] / res.data_info["df_resid"]
    np.testing.assert_allclose(pred["se"] ** 2, conf["se"] ** 2 + s2, rtol=1e-10)
    _, ax = res.plot(type="fitted", interval="prediction")
    verts = ax.collections[0].get_paths()[0].vertices
    assert np.isclose(verts[:, 1].max(), pred["upper"].max())


def test_fitted_plot_after_absorbed_effects_needs_the_data(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.feols("y ~ x1 + x2 | id", data=df)
    with pytest.raises(sp.MethodIncompatibility, match="data="):
        res.plot(type="fitted")
    _, ax = res.plot(type="fitted", data=df)
    np.testing.assert_allclose(
        np.sort(ax.lines[0].get_ydata()), np.sort(res.fitted_values()), rtol=1e-12
    )


def test_default_plot_is_still_the_coefficient_plot(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.regress("y ~ x1 + x2", data=df)
    _, ax_default = res.plot()
    _, ax_coef = sp.coefplot(res)
    assert [t.get_text() for t in ax_default.get_yticklabels()] == [
        t.get_text() for t in ax_coef.get_yticklabels()
    ]
    with pytest.raises(sp.MethodIncompatibility, match="type="):
        res.plot(type="residuals")
