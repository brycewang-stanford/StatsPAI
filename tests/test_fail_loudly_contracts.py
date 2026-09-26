"""Regression guards for options that used to be accepted and then ignored.

Each case here returned a plausible number while silently dropping part of
the request (CLAUDE.md §3.7):

* ``alpha=`` on 31 ``EconometricResults`` estimators: the stored interval,
  ``conf_int()`` and ``tidy()`` stayed at 95%;
* ``summary(alpha=)`` relabelled the interval columns without recomputing;
* ``sp.synth(method='mc', covariates=)`` and ``(method='augmented',
  placebo=False)`` dropped the option; ``augsynth`` ignored every unknown
  keyword; ``synth_compare`` skipped failing methods without a trace;
* listwise deletion of incomplete rows happened with no note;
* a singular variance matrix left NaN standard errors with no warning.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd
import pytest
from scipy import stats

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

from statspai.exceptions import AssumptionWarning, ConvergenceWarning
from statspai.workflow._degradation import WorkflowDegradedWarning

# --------------------------------------------------------------------------
# alpha= reaches the reported interval
# --------------------------------------------------------------------------


def _xsec() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 600
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "z1": rng.normal(size=n),
            "z2": rng.normal(size=n),
            "g": rng.integers(0, 30, n),
            "id": np.repeat(np.arange(60), 10),
            "time": np.tile(np.arange(10), 60),
            "lat": rng.uniform(30, 40, n),
            "lon": rng.uniform(-100, -90, n),
        }
    )
    df["d"] = df.z1 + 0.5 * df.z2 + rng.normal(size=n)
    df["y"] = 1 + df.x1 - 0.5 * df.x2 + 0.5 * df.d + rng.normal(size=n)
    df["yb"] = (df.x1 + rng.logistic(size=n) > 0).astype(int)
    df["yb2"] = (df.x2 + rng.normal(size=n) > 0).astype(int)
    df["yc"] = rng.poisson(np.exp(0.3 + 0.3 * df.x1))
    df["yo"] = np.digitize(df.x1 + rng.logistic(size=n), [-1, 0, 1])
    df["ym"] = rng.integers(0, 3, n)
    df["yf"] = 1 / (1 + np.exp(-(0.3 * df.x1 + rng.normal(scale=0.5, size=n))))
    df["dur"] = rng.exponential(np.exp(-0.3 * df.x1))
    df["ev"] = (rng.uniform(size=n) < 0.8).astype(int)
    df["ylog"] = (
        2
        + 0.5 * df.x1
        - np.abs(rng.normal(scale=0.5, size=n))
        + rng.normal(scale=0.3, size=n)
    )
    df["treat"] = (df.z1 + rng.normal(size=n) > 0).astype(int)
    df["grp"] = np.repeat(np.arange(150), 4)
    df["ck"] = 0
    df.loc[df.groupby("grp").x1.idxmax(), "ck"] = 1
    return df


_DF = _xsec()


def _base():
    return sp.regress("y ~ x1 + x2", data=_DF)


ALPHA_CASES: Dict[str, Callable[[float], Any]] = {
    "logit": lambda a: sp.logit("yb ~ x1 + x2", data=_DF, alpha=a),
    "probit": lambda a: sp.probit("yb ~ x1 + x2", data=_DF, alpha=a),
    "poisson": lambda a: sp.poisson("yc ~ x1 + x2", data=_DF, alpha=a),
    "nbreg": lambda a: sp.nbreg("yc ~ x1 + x2", data=_DF, alpha=a),
    "zip_model": lambda a: sp.zip_model("yc ~ x1", data=_DF, inflate=["x2"], alpha=a),
    "hurdle": lambda a: sp.hurdle("yc ~ x1", data=_DF, alpha=a),
    "ppmlhdfe": lambda a: sp.ppmlhdfe("yc ~ x1", data=_DF, absorb="g", alpha=a),
    "glm": lambda a: sp.glm("y ~ x1 + x2", data=_DF, alpha=a),
    "fracreg": lambda a: sp.fracreg(_DF, y="yf", x=["x1"], alpha=a),
    "betareg": lambda a: sp.betareg(_DF, y="yf", x=["x1"], alpha=a),
    "mlogit": lambda a: sp.mlogit("ym ~ x1", data=_DF, alpha=a),
    "ologit": lambda a: sp.ologit("yo ~ x1 + x2", data=_DF, alpha=a),
    "clogit": lambda a: sp.clogit("ck ~ x1 + x2", data=_DF, group="grp", alpha=a),
    "truncreg": lambda a: sp.truncreg(_DF[_DF.y > 0], y="y", x=["x1"], ll=0, alpha=a),
    "biprobit": lambda a: sp.biprobit(_DF, "yb", "yb2", ["x1"], ["x2"], alpha=a),
    "etregress": lambda a: sp.etregress(_DF, "y", ["x1"], "treat", ["z1"], alpha=a),
    "iv": lambda a: sp.iv("y ~ x1 + (d ~ z1 + z2)", data=_DF, alpha=a),
    "liml": lambda a: sp.liml("y ~ x1 + (d ~ z1 + z2)", data=_DF, alpha=a),
    "jive": lambda a: sp.jive(_DF, "y", ["d"], ["x1"], ["z1", "z2"], alpha=a),
    "panel_fgls": lambda a: sp.panel_fgls(_DF, "y", ["x1"], alpha=a),
    "panel_logit": lambda a: sp.panel_logit(_DF, "yb", ["x1"], method="re", alpha=a),
    "interactive_fe": lambda a: sp.interactive_fe(_DF, "y", ["x1"], alpha=a),
    "frontier": lambda a: sp.frontier(_DF, "ylog", ["x1"], alpha=a),
    "zisf": lambda a: sp.zisf(_DF, "ylog", ["x1"], alpha=a),
    "cox": lambda a: sp.cox(data=_DF, duration="dur", event="ev", x=["x1"], alpha=a),
    "survreg": lambda a: sp.survreg(
        data=_DF, duration="dur", event="ev", x=["x1"], alpha=a
    ),
    "twoway_cluster": lambda a: sp.twoway_cluster(_base(), _DF, "g", "time", alpha=a),
    "conley": lambda a: sp.conley(
        _base(), _DF, lat="lat", lon="lon", dist_cutoff=100, alpha=a
    ),
    "jackknife_se": lambda a: sp.jackknife_se(_base(), _DF, "g", alpha=a),
    "cr2_se": lambda a: sp.cr2_se(_base(), _DF, "g", alpha=a),
}


@pytest.mark.parametrize("name", sorted(ALPHA_CASES))
def test_fit_time_alpha_sets_every_interval(name: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ALPHA_CASES[name](0.10)
    assert res.alpha == pytest.approx(0.10)
    term = res.params.index[-1]
    se = float(res.std_errors[term])
    crit = stats.t.ppf(0.95, res._inference_df())
    want = 2 * crit * se
    ci = res.conf_int()
    np.testing.assert_allclose(
        float(ci.iloc[:, 1][term] - ci.iloc[:, 0][term]), want, rtol=1e-8
    )
    np.testing.assert_allclose(
        float(res.conf_int_upper[term] - res.conf_int_lower[term]), want, rtol=1e-8
    )
    tidy = res.tidy().set_index("term")
    np.testing.assert_allclose(
        float(tidy.loc[term, "conf_high"] - tidy.loc[term, "conf_low"]),
        want,
        rtol=1e-8,
    )


def test_default_alpha_is_unchanged() -> None:
    res = sp.regress("y ~ x1 + x2", data=_DF)
    assert res.alpha == 0.05
    crit = stats.t.ppf(0.975, res.data_info["df_resid"])
    lo, hi = res.conf_int().loc["x1"]
    np.testing.assert_allclose(hi - lo, 2 * crit * res.std_errors["x1"], rtol=1e-12)


def test_summary_alpha_recomputes_the_interval_it_labels() -> None:
    res = sp.regress("y ~ x1 + x2", data=_DF)
    text = str(res.summary(alpha=0.10))
    assert "[0.050" in text and "0.950]" in text
    lo90, hi90 = res.conf_int(alpha=0.10).loc["x1"]
    row = next(line for line in text.splitlines() if line.startswith("x1"))
    printed = [float(tok) for tok in row.split()[1:]]
    np.testing.assert_allclose(printed[-2:], [lo90, hi90], atol=5e-5)


def test_pickled_results_without_alpha_still_work() -> None:
    res = sp.regress("y ~ x1 + x2", data=_DF)
    del res.__dict__["alpha"]  # a result pickled before the attribute existed
    assert res.alpha == 0.05
    assert res.conf_int().shape == (3, 2)


# --------------------------------------------------------------------------
# synthetic control: options that were dropped
# --------------------------------------------------------------------------


def _synth_panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n_units, n_periods = 20, 15
    df = pd.DataFrame(
        {
            "unit": np.repeat(np.arange(n_units), n_periods),
            "time": np.tile(np.arange(n_periods), n_units),
        }
    )
    df["x1"] = rng.normal(size=len(df))
    df["y"] = (
        rng.normal(size=len(df))
        + 0.1 * df.time
        + np.repeat(rng.normal(size=n_units), n_periods)
        + 1.5 * df.x1
    )
    df.loc[(df.unit == 0) & (df.time >= 10), "y"] += 2
    return df


_SC = dict(outcome="y", unit="unit", time="time", treated_unit=0, treatment_time=10)


def test_synth_mc_uses_covariates() -> None:
    df = _synth_panel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = sp.synth(df, method="mc", placebo=False, **_SC)
        adj = sp.synth(df, method="mc", covariates=["x1"], placebo=False, **_SC)
    assert abs(adj.estimate - plain.estimate) > 0.1


def test_synth_augmented_honours_placebo_and_rejects_typos() -> None:
    df = _synth_panel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        off = sp.synth(df, method="augmented", placebo=False, **_SC)
        on = sp.synth(df, method="augmented", placebo=True, **_SC)
    assert np.isnan(off.pvalue) and np.isfinite(on.pvalue)
    with pytest.raises(TypeError, match="ridge_lamda"):
        sp.synth(df, method="augmented", ridge_lamda=1.0, **_SC)
    with pytest.raises(TypeError, match="v_methd"):
        sp.synth(df, method="classic", v_methd="nested", **_SC)


def test_synth_compare_reports_failures_and_rejects_typos() -> None:
    df = _synth_panel()
    with pytest.raises(TypeError, match="unexpected keyword"):
        sp.synth_compare(df, methods=["classic", "augmented"], typo_opt=1, **_SC)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        comp = sp.synth_compare(
            df, methods=["classic", "no_such_method"], placebo=False, **_SC
        )
    assert list(comp.comparison_table["method"]) == ["classic"]
    assert [d["section"] for d in comp.degradations] == [
        "synth_compare method='no_such_method'"
    ]
    assert any(issubclass(w.category, WorkflowDegradedWarning) for w in caught)


# --------------------------------------------------------------------------
# listwise deletion is announced
# --------------------------------------------------------------------------


def _with_missing() -> pd.DataFrame:
    df = _DF[["y", "treat", "x1", "x2", "d", "z1"]].copy()
    df.loc[:9, "x2"] = np.nan
    df.loc[20:24, "y"] = np.nan
    df["unused"] = np.nan
    return df


@pytest.mark.parametrize(
    "fit",
    [
        lambda df: sp.regress("y ~ treat + x1 + x2", data=df),
        lambda df: sp.aipw(df, "y", "treat", ["x1", "x2"]),
        lambda df: sp.dml(df, "y", "treat", ["x1", "x2"]),
        lambda df: sp.iv("y ~ x2 + (d ~ z1)", data=df),
    ],
    ids=["regress", "aipw", "dml", "iv"],
)
def test_listwise_deletion_is_announced(fit: Callable[[pd.DataFrame], Any]) -> None:
    df = _with_missing()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = fit(df)
    notes = [
        w
        for w in caught
        if issubclass(w.category, AssumptionWarning)
        and "listwise deletion" in str(w.message)
    ]
    assert len(notes) == 1
    assert res.model_info["listwise_deletion"] == {
        "n_rows_dropped_missing": 15,
        "missing_by_column": {"y": 5, "x2": 10},
    }


def test_missing_by_design_gets_no_note() -> None:
    """Lee bounds use the missing outcomes (attrition); they are not dropped."""
    rng = np.random.default_rng(1)
    n = 800
    df = pd.DataFrame({"d": rng.integers(0, 2, n)})
    df["y"] = 1 + df.d + rng.normal(size=n)
    df.loc[rng.uniform(size=n) < 0.2 + 0.1 * df.d, "y"] = np.nan
    df["s"] = df["y"].notna().astype(int)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sp.lee_bounds(df, y="y", treat="d", selection="s")
    assert not [w for w in caught if "listwise deletion" in str(w.message)]


def test_complete_data_gets_no_note() -> None:
    df = _with_missing().dropna(subset=["y", "x2"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = sp.regress("y ~ treat + x1 + x2", data=df)
    assert not [w for w in caught if "listwise deletion" in str(w.message)]
    assert "listwise_deletion" not in res.model_info


# --------------------------------------------------------------------------
# non-finite standard errors are flagged
# --------------------------------------------------------------------------


def test_nonfinite_se_warns_and_is_recorded() -> None:
    # Poisson data: the NB2 dispersion goes to its boundary and its SE is NaN.
    with pytest.warns(ConvergenceWarning, match="not finite"):
        res = sp.zinb("yc ~ x1", data=_DF, inflate=["x2"])
    assert res.model_info["nonfinite_se_terms"] == ["ln_alpha"]
