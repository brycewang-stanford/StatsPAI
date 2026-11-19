"""Fixes from running the Python Polars edition of Gow and Lubberink,
*Empirical Research in Accounting* (https://iangow.github.io/era_pl_book/),
through StatsPAI chapter by chapter (2026-11-19).

References are the statsmodels / pyfixest calls the book itself makes.

* A column read only as category labels may hold ``inf``: the book codes a
  never-treated firm's event time as ``-inf`` inside ``C(t,
  Treatment(reference=-1))`` (chapter 21). Numeric uses of ``inf`` are
  still refused.
* A text regressor (Fama-French industry ``ff12`` as ``"1"`` ... ``"12"``,
  chapter 24) is expanded into indicators by the count, ordered and
  quantile models; it used to be cast to float and fitted as one number.
* ``sp.regress(cluster=[a, b], <unknown>=...)`` raises like the one-way
  path does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility

smf = pytest.importorskip("statsmodels.formula.api")
sm = pytest.importorskip("statsmodels.api")


# ---------------------------------------------------------------- inf labels
@pytest.fixture
def event_panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 600
    return pd.DataFrame(
        {
            "t": rng.choice([-np.inf, -2.0, -1.0, 0.0, 1.0], n),
            "y": rng.normal(size=n),
            "g": rng.integers(0, 25, n),
        }
    )


def test_regress_accepts_inf_as_a_factor_level(event_panel):
    f = "y ~ C(t, Treatment(reference=-1))"
    ref = smf.ols(f, event_panel).fit()
    ours = sp.regress(f, event_panel)
    assert list(ours.params.index) == list(ref.params.index)
    np.testing.assert_allclose(ours.params, ref.params, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(ours.std_errors, ref.bse, rtol=1e-10)


def test_feols_accepts_inf_as_a_factor_level(event_panel):
    pf = pytest.importorskip("pyfixest")
    for f in ("y ~ C(t, Treatment(reference=-1)) | g", "y ~ i(t, ref=-1) | g"):
        ref = pf.feols(f, data=event_panel, vcov="iid")
        ours = sp.feols(f, data=event_panel, vcov="iid")
        np.testing.assert_allclose(
            ours.params.reindex(ref.coef().index), ref.coef(), rtol=1e-10
        )
        np.testing.assert_allclose(
            ours.std_errors.reindex(ref.se().index), ref.se(), rtol=1e-10
        )


@pytest.mark.parametrize("formula", ["y ~ t", "y ~ C(t) + t", "t ~ C(g)"])
def test_inf_in_a_numeric_use_is_still_refused(event_panel, formula):
    with pytest.raises(DataInsufficient):
        sp.regress(formula, event_panel)


def test_inf_in_a_numeric_feols_regressor_is_still_refused(event_panel):
    with pytest.raises(DataInsufficient):
        sp.feols("y ~ t | g", event_panel)


def test_inf_label_column_used_as_weights_is_still_refused(event_panel):
    d = event_panel.assign(w=np.where(np.isinf(event_panel["t"]), np.inf, 1.0))
    with pytest.raises(DataInsufficient):
        sp.regress("y ~ C(t)", d, weights="w")


def test_label_only_columns_reads_the_formula():
    from statspai._input_validation import label_only_columns

    cols = ["y", "t", "g", "x", "z"]
    assert label_only_columns("y ~ C(t) + x | g", cols) == {"t", "g"}
    assert label_only_columns("y ~ C(t) + t", cols) == set()
    assert label_only_columns("y ~ x | g | x ~ z", cols) == {"g"}
    assert label_only_columns("t ~ C(t)", cols) == set()


# ------------------------------------------------------------- text regressor
@pytest.fixture(params=[["1", "2", "3", "10"], ["a", "b", "c", "d"]])
def labelled(request) -> pd.DataFrame:
    labels = request.param
    rng = np.random.default_rng(3)
    n = 800
    d = pd.DataFrame({"x": rng.normal(size=n), "g": rng.choice(labels, n)})
    d["y"] = rng.poisson(np.exp(0.3 * d["x"] + (d["g"] == labels[1]) * 0.5))
    d["yc"] = d["y"] + rng.normal(size=n)
    return d


def _aligned(ours: pd.Series, ref: pd.Series) -> np.ndarray:
    ref = ref.rename({"Intercept": "_cons"})
    return (
        ours.rename({"const": "_cons", "Intercept": "_cons"})
        .reindex(ref.index)
        .to_numpy()
    )


def test_poisson_expands_a_text_regressor(labelled):
    ref = smf.glm("y ~ x + g", labelled, family=sm.families.Poisson()).fit(tol=1e-12)
    ours = sp.poisson("y ~ x + g", labelled)
    np.testing.assert_allclose(_aligned(ours.params, ref.params), ref.params, rtol=1e-7)
    np.testing.assert_allclose(_aligned(ours.std_errors, ref.bse), ref.bse, rtol=1e-6)


def test_nbreg_expands_a_text_regressor(labelled):
    ours = sp.nbreg("y ~ x + g", labelled)
    assert {"g[T.%s]" % lv for lv in sorted(labelled["g"].unique())[1:]} <= set(
        ours.params.index
    )
    assert "g" not in ours.params.index


def test_qreg_expands_a_text_regressor(labelled):
    ref = smf.quantreg("yc ~ x + g", labelled).fit(q=0.5)
    ours = sp.qreg("yc ~ x + g", labelled)
    np.testing.assert_allclose(_aligned(ours.params, ref.params), ref.params, atol=1e-5)


def test_ologit_expands_a_text_regressor(labelled):
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    d = labelled.assign(yo=np.clip(labelled["y"], 0, 3))
    levels = sorted(d["g"].unique())
    exog = pd.DataFrame({"x": d["x"]})
    for lv in levels[1:]:  # patsy's treatment coding: first sorted level is base
        exog[f"g[T.{lv}]"] = (d["g"] == lv).astype(float)
    ref = OrderedModel(d["yo"], exog, distr="logit").fit(
        method="bfgs", disp=0, maxiter=5000, gtol=1e-10
    )
    ours = sp.ologit("yo ~ x + g", d)
    slopes = list(exog.columns)
    np.testing.assert_allclose(
        ours.params.reindex(slopes), ref.params.reindex(slopes), rtol=1e-4
    )


@pytest.mark.parametrize("fn", ["zinb", "hurdle"])
def test_models_without_factor_expansion_refuse_text(labelled, fn):
    with pytest.raises(MethodIncompatibility, match="category labels"):
        getattr(sp, fn)("y ~ x + g", data=labelled)


def test_column_list_entry_refuses_text(labelled):
    with pytest.raises(MethodIncompatibility, match="category labels"):
        sp.poisson(data=labelled, y="y", x=["x", "g"])
    with pytest.raises(MethodIncompatibility, match="category labels"):
        sp.qreg(data=labelled, y="yc", x=["x", "g"])


# ------------------------------------------------------- regress **kwargs
def test_two_way_cluster_rejects_unknown_keywords():
    rng = np.random.default_rng(1)
    d = pd.DataFrame(
        {
            "y": rng.normal(size=60),
            "x": rng.normal(size=60),
            "a": np.repeat(range(12), 5),
            "b": np.tile(range(5), 12),
        }
    )
    with pytest.raises(TypeError, match="ssc"):
        sp.regress("y ~ x", d, cluster=["a", "b"], ssc={"G_adj": False})
    with pytest.raises(TypeError, match="foo"):
        sp.regress("y ~ x", d, cluster="a", vce="cr2", foo=1)
    with pytest.raises(TypeError, match="wild_reps"):
        sp.regress("y ~ x", d, cluster="a", wild_reps=99)


# ------------------------------------------------- second pass (2026-11-19)
def test_ivreg_expands_a_text_exogenous_regressor(labelled):
    pf = pytest.importorskip("pyfixest")
    d = labelled.assign(
        z=labelled["x"] + np.random.default_rng(9).normal(size=len(labelled))
    )
    ours = sp.ivreg("yc ~ g + (x ~ z)", d)
    ref = pf.feols("yc ~ C(g) | x ~ z", data=d)
    ref_coef = ref.coef().rename(lambda k: k.replace("C(g)", "g"))
    np.testing.assert_allclose(ours.params.reindex(ref_coef.index), ref_coef, rtol=1e-9)
    with pytest.raises(MethodIncompatibility, match="category labels"):
        sp.ivreg("yc ~ x + (g ~ z)", d)


@pytest.mark.parametrize(
    "call",
    [
        lambda d: sp.hdfe_ols("yc ~ x + g", d),
        lambda d: sp.rifreg("yc ~ x + g", d),
        lambda d: sp.heckman(d, y="yc", x=["x", "g"], z=["x", "g", "y"], select=None),
        lambda d: sp.fracreg(
            d.assign(f=d["yc"] / (1 + d["yc"].max())), y="f", x=["x", "g"]
        ),
        lambda d: sp.logit(
            data=d.assign(b=(d["y"] > 0).astype(int)), y="b", x=["x", "g"]
        ),
        lambda d: sp.tobit(data=d, y="yc", x=["x", "g"]),
        lambda d: sp.mixed(
            d.assign(grp=np.arange(len(d)) // 8),
            y="yc",
            x_fixed=["x", "g"],
            group="grp",
        ),
        lambda d: sp.liml(
            data=d.assign(z=d["x"]), y="yc", x_endog=["x"], x_exog=["g"], z=["z"]
        ),
    ],
    ids=[
        "hdfe_ols",
        "rifreg",
        "heckman",
        "fracreg",
        "logit_xy",
        "tobit_xy",
        "mixed",
        "liml",
    ],
)
def test_estimators_without_expansion_refuse_text(labelled, call):
    with pytest.raises(MethodIncompatibility, match="category labels"):
        call(labelled)


def test_cox_expands_a_text_covariate(labelled):
    rng = np.random.default_rng(4)
    d = labelled.assign(
        dur=rng.exponential(1.0, len(labelled)) * np.exp(-0.3 * labelled["x"]),
        ev=(rng.random(len(labelled)) < 0.8).astype(int),
    )
    ours = sp.cox("dur ~ x + g", d, event="ev")
    ref = smf.phreg("dur ~ x + g", d, status=d["ev"].to_numpy(), ties="efron").fit()
    ref_params = pd.Series(ref.params, index=ref.model.exog_names)
    np.testing.assert_allclose(
        ours.params.reindex(ref_params.index), ref_params, rtol=1e-6
    )


def test_regtable_keep_and_drop_take_patterns():
    rng = np.random.default_rng(2)
    d = pd.DataFrame({"x": rng.normal(size=200), "yr": rng.choice([1, 2, 3], 200)})
    d["y"] = d["x"] + rng.normal(size=200)
    fit = sp.regress("y ~ x * C(yr)", d)
    names = list(fit.params.index)
    from statspai.output.regression_table import _match_coef_names

    assert _match_coef_names(names, ["yr"], "drop") == {n for n in names if "yr" in n}
    assert _match_coef_names(names, ["x"], "keep") == {"x"}  # exact name wins
    assert _match_coef_names(names, ["x:*"], "keep") == {
        n for n in names if n.startswith("x:")
    }
    with pytest.warns(UserWarning, match="matched no coefficient"):
        _match_coef_names(names, ["nothere"], "keep")
    text = str(sp.regtable(fit, drop=["yr"]))
    assert "C(yr)" not in text and "x" in text


def test_rdplot_passes_masspoints():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    rng = np.random.default_rng(5)
    x = np.round(rng.uniform(-1, 1, 800), 1)  # heavy mass points
    d = pd.DataFrame({"x": x, "y": x + (x >= 0) + rng.normal(0, 0.3, 800)})
    from statspai.rd._rdplot_numbers import rdplot_numbers

    for mp in ("off", "adjust"):
        fig, _ = sp.rdplot(d, y="y", x="x", c=0, masspoints=mp)
        ref = rdplot_numbers(d["y"].to_numpy(), d["x"].to_numpy(), c=0, masspoints=mp)
        assert tuple(fig.rdplot_data["J"]) == tuple(ref["J"])
