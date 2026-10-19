"""Summary / indicator rows of ``sp.regtable`` (2026-10-19 audit).

* "Cluster SE" said ``No`` for clustered ``sp.feols`` columns, whether the
  clustering came from ``cluster=`` or ``vcov={"CRV1": ...}`` (the fixest
  family records it under ``model_info['clustervar']``).
* "Mean of Y" printed ``-0.000`` for fixed-effect models: it averaged the
  demeaned outcome.
* Unknown ``stats=`` names printed as empty rows; they now raise with a
  did-you-mean.
* Logit columns showed empty ``R² / Adj. R² / F`` rows and no pseudo-R² or
  log-likelihood.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def df():
    r = np.random.default_rng(0)
    n = 400
    x1 = r.normal(size=n)
    d = (r.random(n) < 0.5).astype(int)
    g = r.integers(0, 25, n)
    y = 1 + 2 * d + x1 + 0.3 * g / 10 + r.normal(size=n)
    out = pd.DataFrame(dict(y=y, d=d, x1=x1, g=g))
    out["yb"] = (out.y > out.y.median()).astype(int)
    out["cnt"] = r.poisson(np.exp(0.3 + 0.3 * x1 + 0.2 * d))
    return out


def _rows(table):
    return {row["term"]: row for row in table.to_dict()["table"] if row["term"]}


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


@pytest.mark.parametrize(
    "fit",
    [
        lambda df: sp.feols("y ~ d + x1 | g", data=df, vcov={"CRV1": "g"}),
        lambda df: sp.feols("y ~ d + x1 | g", data=df, cluster="g"),
        lambda df: sp.fepois("cnt ~ d + x1 | g", data=df, vcov={"CRV1": "g"}),
        lambda df: sp.hdfe_ols("y ~ d + x1 | g", data=df, cluster="g"),
        lambda df: sp.regress("y ~ d + x1", data=df, cluster="g"),
    ],
    ids=["feols_vcov_dict", "feols_cluster", "fepois", "hdfe_ols", "regress"],
)
def test_cluster_row_names_the_cluster_variable(df, fit):
    m = _quiet(fit, df)
    rows = _rows(_quiet(sp.regtable, m))
    assert rows["Cluster SE"]["(1)"] == "g"


def test_cluster_row_two_way_lists_both(df):
    m = _quiet(sp.feols, "y ~ d + x1 | g", data=df, vcov={"CRV1": "g+d"})
    assert _rows(_quiet(sp.regtable, m))["Cluster SE"]["(1)"] == "g, d"


def test_unclustered_column_says_no_beside_a_clustered_one(df):
    m1 = _quiet(sp.feols, "y ~ d + x1 | g", data=df, vcov={"CRV1": "g"})
    m2 = _quiet(sp.feols, "y ~ d + x1 | g", data=df, vcov="hetero")
    row = _rows(_quiet(sp.regtable, m1, m2))["Cluster SE"]
    assert (row["(1)"], row["(2)"]) == ("g", "No")


def test_hdfe_ols_fe_row_says_yes(df):
    m = _quiet(sp.hdfe_ols, "y ~ d + x1 | g", data=df)
    assert _rows(_quiet(sp.regtable, m))["G FE"]["(1)"] == "Yes"


@pytest.mark.parametrize(
    "fit, col",
    [
        (lambda df: sp.feols("y ~ d + x1 | g", data=df), "y"),
        (lambda df: sp.fepois("cnt ~ d + x1 | g", data=df), "cnt"),
        (lambda df: sp.feglm("yb ~ d + x1 | g", data=df, family="logit"), "yb"),
    ],
    ids=["feols", "fepois", "feglm_logit"],
)
def test_mean_of_y_is_the_outcome_mean_with_fixed_effects(df, fit, col):
    m = _quiet(fit, df)
    table = _quiet(sp.regtable, m, stats=["N", "ymean"])
    stats = table.to_dict()["models"][0]["stats"]
    # Estimation sample = all rows here (no singletons, no separation).
    assert stats["N"] == len(df)
    assert stats["Mean of Y"] == pytest.approx(float(df[col].mean()), rel=1e-12)


def test_unknown_stat_raises_with_suggestion(df):
    m = _quiet(sp.regress, "y ~ d + x1", data=df)
    with pytest.raises(sp.MethodIncompatibility) as info:
        sp.regtable(m, stats=["N", "r_sqared"])
    assert "r_squared" in info.value.recovery_hint


@pytest.mark.parametrize("name", ["bogus", "Estimator"])
def test_unknown_stat_names_are_refused(df, name):
    m = _quiet(sp.regress, "y ~ d + x1", data=df)
    with pytest.raises(sp.MethodIncompatibility):
        sp.regtable(m, stats=["N", name])


def test_long_spellings_render_values(df):
    m = _quiet(sp.regress, "y ~ d + x1", data=df)
    table = _quiet(sp.regtable, m, stats=["n_obs", "r_squared", "adj_r_squared"])
    stats = table.to_dict()["models"][0]["stats"]
    rows = _rows(table)
    assert rows["N"]["(1)"] == str(len(df))
    assert rows["R²"]["(1)"] == f"{stats['R-squared']:.3f}"
    assert stats["R-squared"] == pytest.approx(m.diagnostics["R-squared"], abs=0)


def test_fe_cluster_and_dv_mean_names_are_accepted(df):
    m = _quiet(sp.feols, "y ~ d + x1 | g", data=df, vcov={"CRV1": "g"})
    rows = _rows(_quiet(sp.regtable, m, stats=["N", "R2", "Cluster", "FE", "DV mean"]))
    assert rows["Cluster SE"]["(1)"] == "g"
    assert rows["G FE"]["(1)"] == "Yes"
    assert rows["Mean of Y"]["(1)"] == f"{df.y.mean():.3f}"
    assert "Cluster" not in rows and "FE" not in rows


def test_logit_default_rows_are_pseudo_r2_and_loglik(df):
    m = _quiet(sp.logit, "yb ~ d + x1", data=df)
    rows = _rows(_quiet(sp.regtable, m))
    assert rows["Pseudo R²"]["(1)"] == f"{m.model_info['pseudo_r2']:.3f}"
    assert rows["Log-Lik."]["(1)"] == f"{m.model_info['ll']:.3f}"
    for empty in ("R²", "Adj. R²", "F"):
        assert empty not in rows


def test_ols_default_rows_unchanged(df):
    m = _quiet(sp.regress, "y ~ d + x1", data=df)
    rows = _rows(_quiet(sp.regtable, m))
    assert {"N", "R²", "Adj. R²", "F"} <= set(rows)
    assert "Pseudo R²" not in rows


def test_causal_column_has_no_empty_r2_rows(df):
    m = _quiet(sp.ipw, df, y="y", treat="d", covariates=["x1"], n_bootstrap=10)
    rows = _rows(_quiet(sp.regtable, m))
    assert "N" in rows
    for empty in ("R²", "Adj. R²", "F"):
        assert empty not in rows


def test_esttab_and_modelsummary_share_the_cluster_row(df):
    m = _quiet(sp.feols, "y ~ d + x1 | g", data=df, vcov={"CRV1": "g"})
    text_e = _quiet(sp.esttab, m, output="text")
    text_m = _quiet(sp.modelsummary, m)
    for text in (str(text_e), str(text_m)):
        line = next(ln for ln in text.splitlines() if ln.startswith("Cluster SE"))
        assert line.split()[-1] == "g"
