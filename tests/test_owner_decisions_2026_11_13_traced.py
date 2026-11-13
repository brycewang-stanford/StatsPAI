"""Owner decisions of 2026-11-13, part two
(``docs/dev/2026-11-13-owner-decisions.md``).

The items whose source files are on a traced Track A estimation path:
the announced ``crit_dist`` default of ``sp.stacked_did`` (a
``FutureWarning`` on the default path, silent when explicit, numbers
unchanged, and the announced value equal to R ``fixest``), the PLIV
label, the SC jackknife warning, the HAC reference distribution, the
labelled balance tables of ``sp.match``, the single convergence warning
of ``sp.tmle`` and the SMD denominator each balance function names. The
other announcements are in ``test_owner_decisions_2026_11_13.py``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import AssumptionWarning, MethodIncompatibility

FIX = Path(__file__).resolve().parent / "reference_parity" / "_fixtures"
ROOT = Path(__file__).resolve().parents[1]


def _run(fn, *args, **kw):
    """Call ``fn`` and return ``(result, warnings recorded)``."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = fn(*args, **kw)
    return res, list(rec)


def _future(rec, option):
    return [
        w
        for w in rec
        if issubclass(w.category, FutureWarning) and f"{option}=" in str(w.message)
    ]


def _assert_announced(rec, option, today, later):
    """Exactly one announcement, naming both values, at the caller's line."""
    found = _future(rec, option)
    assert len(found) == 1, [str(w.message) for w in rec]
    text = str(found[0].message)
    assert f"{option}={today}" in text and f"{option}={later}" in text
    assert "MIGRATION.md" in text
    # attributed to this file, not to a wrapper inside the package
    assert Path(found[0].filename).name == Path(__file__).name


# ---------------------------------------------------------------------------
# A2. sp.stacked_did(crit_dist=): normal today, t(G-1) later
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stack_panel():
    return pd.read_csv(ROOT / "tests" / "r_parity" / "data" / "75_stacked.csv")


def _stacked(df, **kw):
    return sp.stacked_did(
        df,
        y="y",
        group="id",
        time="year",
        first_treat="first_treat",
        window=(-3, 3),
        **kw,
    )


def test_stacked_did_default_announces_t(stack_panel):
    default, rec = _run(_stacked, stack_panel)
    _assert_announced(rec, "crit_dist", "'normal'", "'t'")
    normal, rec_n = _run(_stacked, stack_panel, crit_dist="normal")
    t, rec_t = _run(_stacked, stack_panel, crit_dist="t")
    assert not _future(rec_n, "crit_dist") and not _future(rec_t, "crit_dist")
    # today's numbers, exactly
    assert (default.estimate, default.se, default.pvalue, default.ci) == (
        normal.estimate,
        normal.se,
        normal.pvalue,
        normal.ci,
    )
    pd.testing.assert_frame_equal(default.detail, normal.detail)
    assert default.model_info["conventions"] == {
        "parity": None,
        "crit_dist": "normal",
        "crit_df": None,
    }
    # estimates and standard errors do not depend on the rule
    assert (t.estimate, t.se) == (normal.estimate, normal.se)
    np.testing.assert_array_equal(t.detail["att"], normal.detail["att"])
    np.testing.assert_array_equal(t.detail["se"], normal.detail["se"])
    # the headline follows t(G - 1) with G = 240 units
    assert t.model_info["conventions"]["crit_df"] == 239
    assert t.pvalue == pytest.approx(
        2 * stats.t.sf(abs(t.estimate / t.se), 239), rel=1e-12
    )
    half = stats.t.ppf(0.975, 239) * t.se
    assert t.ci == pytest.approx((t.estimate - half, t.estimate + half), rel=1e-12)


@pytest.mark.parametrize("tag,alpha", [("never_95", 0.05), ("never_90", 0.10)])
def test_stacked_did_t_matches_fixest(stack_panel, tag, alpha):
    """``crit_dist='t'`` reproduces the p-values and intervals R fixest
    prints for the same stacked regression (fixture
    ``stacked_crit_R.csv`` from ``_generate_stacked_crit_R.R``)."""
    ref = pd.read_csv(FIX / "stacked_crit_R.csv")
    ref = ref[ref["tag"] == tag].set_index("rel")
    fit = _stacked(stack_panel, crit_dist="t", alpha=alpha)
    es = fit.detail.set_index("relative_time").drop(index=-1)
    assert sorted(es.index) == sorted(ref.index)
    assert fit.model_info["conventions"]["crit_df"] == int(ref["df"].iloc[0])
    for k in ref.index:
        # Coefficients and SEs are Track A module 75's (1e-6); the
        # reference distribution adds nothing to that gap.
        assert es.loc[k, "att"] == pytest.approx(ref.loc[k, "b"], rel=1e-8)
        assert es.loc[k, "se"] == pytest.approx(ref.loc[k, "se"], rel=1e-8)
        assert es.loc[k, "pvalue"] == pytest.approx(ref.loc[k, "pvalue"], rel=1e-6)
        assert es.loc[k, "ci_lower"] == pytest.approx(ref.loc[k, "ll"], rel=1e-7)
        assert es.loc[k, "ci_upper"] == pytest.approx(ref.loc[k, "ul"], rel=1e-7)


def test_stacked_did_poisson_and_library_callers_do_not_warn(stack_panel):
    d = stack_panel.assign(count=np.exp(stack_panel["y"] / 4).round())
    _, rec = _run(
        sp.stacked_did,
        d,
        y="count",
        group="id",
        time="year",
        first_treat="first_treat",
        window=(-3, 3),
        family="poisson",
    )
    assert not _future(rec, "crit_dist")
    # sp.did_summary calls stacked_did on the user's behalf with today's
    # rule pinned: no announcement, and the row holds today's numbers
    summary, rec = _run(
        sp.did_summary,
        stack_panel,
        y="y",
        time="year",
        first_treat="first_treat",
        group="id",
        methods=["stacked"],
    )
    assert not _future(rec, "crit_dist")
    row = summary.detail.iloc[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        direct = sp.stacked_did(
            stack_panel,
            y="y",
            group="id",
            time="year",
            first_treat="first_treat",
            aggregation="treated_obs",
            crit_dist="normal",
        )
    assert (row["estimate"], row["se"], row["pvalue"]) == (
        direct.estimate,
        direct.se,
        direct.pvalue,
    )
    with pytest.raises(MethodIncompatibility, match="crit_dist"):
        _stacked(stack_panel, crit_dist="student")


# ---------------------------------------------------------------------------
# B6. sp.dml(model='pliv'): the coefficient is not labelled a LATE
# ---------------------------------------------------------------------------


def test_pliv_estimand_label_is_theta():
    rng = np.random.default_rng(0)
    n = 600
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    d = z + u + 0.5 * x + rng.normal(size=n)
    y = 0.5 * d + x + u + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "d": d, "z": z, "x": x})
    from sklearn.linear_model import LinearRegression

    lr = LinearRegression()
    fit = sp.dml(
        df,
        y="y",
        treat="d",
        covariates=["x"],
        model="pliv",
        instrument="z",
        ml_g=lr,
        ml_m=lr,
        ml_r=lr,
    )
    assert fit.estimand == "theta"
    assert "LATE" not in fit.summary()
    # With linear nuisances fitted by least squares the cross-fitted PLIV
    # estimate is close to 2SLS on the same data; the label change moved
    # no number (the DoubleML parity of the estimate is pinned in
    # tests/reference_parity/test_ml_probe_nov2026.py).
    iv = sp.ivreg("y ~ x + (d ~ z)", df)
    assert fit.estimate == pytest.approx(float(iv.params["d"]), abs=3 * fit.se)
    # the interactive IV model still estimates a LATE
    zb = (z > 0).astype(int)
    db = ((0.8 * zb + u + rng.normal(size=n)) > 0.5).astype(int)
    df2 = pd.DataFrame({"y": 1.5 * db + x + u, "d": db, "z": zb, "x": x})
    assert (
        sp.dml(
            df2, y="y", treat="d", covariates=["x"], model="iivm", instrument="z"
        ).estimand
        == "LATE"
    )


# ---------------------------------------------------------------------------
# B7. sp.sdid(method='sc', se_method='jackknife') warns
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sc_panel():
    rng = np.random.default_rng(5)
    rows = []
    for u in range(30):
        load = rng.normal()
        for t in range(12):
            treated = u < 4 and t >= 8
            rows.append(
                {
                    "unit": u,
                    "time": t,
                    "y": load * (1 + 0.1 * t) + 0.2 * t + treated + rng.normal(0, 0.3),
                }
            )
    return pd.DataFrame(rows)


def _assumption(rec, needle):
    return [
        w
        for w in rec
        if issubclass(w.category, AssumptionWarning) and needle in str(w.message)
    ]


def test_sdid_sc_jackknife_warns_and_keeps_the_number(sc_panel):
    kw = dict(
        outcome="y",
        unit="unit",
        time="time",
        treated_unit=[0, 1, 2, 3],
        treatment_time=8,
    )
    res, rec = _run(sp.sdid, sc_panel, method="sc", se_method="jackknife", **kw)
    found = _assumption(rec, "jackknife")
    assert len(found) == 1
    assert "placebo" in found[0].message.recovery_hint
    entry = res.model_info["diagnostics"]["sc_jackknife_se"]
    assert entry["valid"] is False and entry["se"] == res.se
    assert entry["use_instead"] == ["placebo", "bootstrap"]
    # the number is the fixed-weight leave-one-unit-out jackknife: with
    # the unit weights held fixed and no time weights or intercept, the
    # SC estimate without unit i is a reweighted mean, recomputed here.
    w = res.model_info["unit_weights"].set_index("unit")["weight"]
    wide = sc_panel.pivot(index="unit", columns="time", values="y")
    post = wide.loc[:, 8:].mean(axis=1)
    treated, controls = [0, 1, 2, 3], [u for u in wide.index if u > 3]
    omega = w.reindex(controls).to_numpy()

    def tau(drop):
        tr = [u for u in treated if u != drop]
        keep = np.array([u != drop for u in controls])
        om = omega[keep] / omega[keep].sum()
        co = post.reindex(controls).to_numpy()[keep]
        return post.reindex(tr).mean() - om @ co

    assert res.estimate == pytest.approx(tau(None), rel=1e-10)
    loo = np.array([tau(u) for u in controls] + [tau(u) for u in treated])
    n_loo = len(loo)
    jack = np.sqrt((n_loo - 1) / n_loo * np.sum((loo - loo.mean()) ** 2))
    assert res.se == pytest.approx(jack, rel=1e-8)
    # the valid combinations stay silent
    for method, se_method in (("sdid", "jackknife"), ("sc", "placebo")):
        _, rec2 = _run(
            sp.sdid, sc_panel, method=method, se_method=se_method, seed=1, **kw
        )
        assert not _assumption(rec2, "jackknife")


# ---------------------------------------------------------------------------
# B8. sp.regress(robust='hac') refers to t(N - K)
# ---------------------------------------------------------------------------


def test_regress_hac_uses_t_reference_and_says_so():
    rng = np.random.default_rng(0)
    n = 60
    x = rng.normal(size=n)
    df = pd.DataFrame({"y": 1 + 0.3 * x + rng.normal(size=n), "x": x})
    fit = sp.regress("y ~ x", df, robust="hac")
    t = fit.params["x"] / fit.std_errors["x"]
    assert fit.pvalues["x"] == pytest.approx(2 * stats.t.sf(abs(t), n - 2), rel=1e-12)
    half = stats.t.ppf(0.975, n - 2) * fit.std_errors["x"]
    lo, hi = fit.conf_int().loc["x"]
    assert (lo, hi) == pytest.approx(
        (fit.params["x"] - half, fit.params["x"] + half), rel=1e-12
    )
    doc = " ".join(sp.regress.__doc__.split())
    assert "Newey-West with normal critical values" not in doc
    assert "t critical values with ``N - K`` degrees of freedom" in doc


# ---------------------------------------------------------------------------
# B9 / B12. sp.match: labelled balance tables, named SMD denominators
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def matched():
    df = sp.datasets.nsw_dw()
    covs = ["age", "education", "re74", "re75"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.match(df, y="re78", treat="treat", covariates=covs, method="psm")
    return df, covs, res


def test_match_summary_labels_both_balance_tables(matched):
    _, covs, res = matched
    text = res.summary()
    before = text.index("Covariate Balance, before matching")
    after = text.index("Covariate Balance, after matching")
    assert before < after
    assert "Detailed Estimates" not in text
    assert text.count("smd = difference in means / pooled SD of the two arms") == 2
    # each printed block holds the numbers of the table it is labelled as
    b_tab, a_tab = res.model_info["balance_before"], res.model_info["balance_after"]
    assert res.detail is b_tab
    block_before, block_after = text[before:after], text[after:]
    for name in covs:
        assert f"{b_tab.set_index('variable').loc[name, 'smd']:.4f}" in block_before
        assert f"{a_tab.set_index('variable').loc[name, 'smd']:.4f}" in block_after
    # a result whose detail is not a staged balance table is unchanged
    df = sp.dgp_did(n_units=60, n_periods=6, staggered=True, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        other = sp.did_imputation(
            df, y="y", group="unit", time="time", first_treat="first_treat"
        )
    assert "Covariate Balance" not in other.summary()


def test_each_balance_function_names_its_smd_denominator(matched):
    """One set of matching weights, three conventions, each one stated by
    the function and reproduced here by hand."""
    df, covs, res = matched
    md = res.model_info["matched_data"]
    w = md["_weight"].fillna(0.0).to_numpy()
    t = md["_treated"].to_numpy().astype(bool)

    def wvar(x, wt):
        m = np.average(x, weights=wt)
        return np.sum(wt * (x - m) ** 2) * wt.sum() / (wt.sum() ** 2 - np.sum(wt**2))

    after = res.model_info["balance_after"].set_index("variable")["smd"]
    before = res.model_info["balance_before"].set_index("variable")["smd"]
    assert res.model_info["smd_denominator"] == (
        "pooled SD of the two arms in the raw sample"
    )
    for table in (res.model_info["balance_after"], res.model_info["balance_before"]):
        assert table.attrs["smd_denominator"] == (
            "pooled SD of the two arms in the raw sample"
        )
    ps_tab = sp.ps_balance(md, "_treated", covs, weights=md["_weight"]).table
    assert ps_tab.attrs["smd_denominator"]["smd_weighted"] == (
        "pooled SD of the two arms, weighted sample"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bal_att = sp.bal_tab(
            md, treat="_treated", covariates=covs, weights=w, estimand="ATT"
        )
        bal_pool = sp.bal_tab(
            md,
            treat="_treated",
            covariates=covs,
            weights=w,
            estimand="ATT",
            s_d_denom="pooled",
        )
    assert bal_att.summary_stats["s_d_denom"] == "treated"
    assert bal_pool.summary_stats["s_d_denom"] == "pooled"
    for name in covs:
        x = md[name].to_numpy(dtype=float)
        xt, xc, wt, wc = x[t], x[~t], w[t], w[~t]
        diff_raw = xt.mean() - xc.mean()
        diff_w = np.average(xt, weights=wt) - np.average(xc, weights=wc)
        pooled_raw = np.sqrt((xt.var(ddof=1) + xc.var(ddof=1)) / 2)
        pooled_w = np.sqrt((wvar(xt, wt) + wvar(xc, wc)) / 2)
        # sp.match: pooled SD of the raw sample, before and after
        assert before[name] == pytest.approx(diff_raw / pooled_raw, abs=6e-5)
        assert after[name] == pytest.approx(diff_w / pooled_raw, abs=6e-5)
        # sp.love_plot / sp.ps_balance: pooled SD of the weighted sample
        assert ps_tab.loc[name, "smd_weighted"] == pytest.approx(
            diff_w / pooled_w, rel=1e-10
        )
        assert ps_tab.loc[name, "smd_raw"] == pytest.approx(
            diff_raw / pooled_raw, rel=1e-10
        )
    # sp.bal_tab: the treated arm's raw SD for an ATT (cobalt), and with
    # s_d_denom='pooled' the numbers sp.match prints
    adj_att = _bal_tab_adjusted(bal_att)
    adj_pool = _bal_tab_adjusted(bal_pool)
    for name in covs:
        x = md[name].to_numpy(dtype=float)
        xt, xc, wt, wc = x[t], x[~t], w[t], w[~t]
        diff_w = np.average(xt, weights=wt) - np.average(xc, weights=wc)
        assert adj_att[name] == pytest.approx(diff_w / xt.std(ddof=1), rel=1e-10)
        assert adj_pool[name] == pytest.approx(after[name], abs=6e-5)


def _bal_tab_adjusted(result) -> pd.Series:
    """Adjusted standardized differences of a ``BalTabResult`` by term."""
    return result.table["diff_adj"]


# ---------------------------------------------------------------------------
# B10. sp.tmle: one warning for the default library's convergence failures
# ---------------------------------------------------------------------------


def test_tmle_default_library_convergence_is_reported_once(monkeypatch):
    """The default library is replaced by one whose logistic learner stops
    after a single iteration, so the convergence warning is certain on
    every platform; everything else is the shipped code path."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

    from statspai.tmle.super_learner import SuperLearner

    def library(task):
        if task == "classification":
            return [
                LogisticRegression(max_iter=1, random_state=0),
                LogisticRegression(C=0.01, max_iter=2000, random_state=0),
            ]
        return [LinearRegression(), Ridge(alpha=1.0)]

    monkeypatch.setattr(
        SuperLearner, "_default_library", lambda self: library(self.task)
    )
    rng = np.random.default_rng(3)
    n = 400
    x1, x2 = rng.normal(size=n), 50 * rng.normal(size=n)
    a = (rng.uniform(size=n) < 1 / (1 + np.exp(-0.5 * x1 - 0.01 * x2))).astype(int)
    y = 1.0 * a + x1 + 0.01 * x2 + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "a": a, "x1": x1, "x2": x2})
    kw = dict(y="y", treat="a", covariates=["x1", "x2"])

    res, rec = _run(sp.tmle, df, **kw)
    assert not [w for w in rec if issubclass(w.category, ConvergenceWarning)]
    collapsed = _assumption(rec, "did not converge")
    assert len(collapsed) == 1
    message = collapsed[0].message
    assert "LogisticRegression in the propensity model g" in str(message)
    hint = message.recovery_hint
    assert "scales" in hint and "propensity_library=" in hint
    entry = res.model_info["diagnostics"]["super_learner_convergence"]
    assert entry == message.diagnostics["super_learner_convergence"]
    assert [(e["nuisance"], e["learner"]) for e in entry] == [
        ("propensity model g", "LogisticRegression")
    ]
    # attributed to the caller's line, so it shows once per call site
    assert Path(collapsed[0].filename).name == Path(__file__).name

    # The same learners passed as a library: that fit is not wrapped, so
    # sklearn's warnings come through one per internal fit, as many as
    # the collapsed warning counted, and the numbers are identical.
    by_hand, rec2 = _run(
        sp.tmle,
        df,
        outcome_library=library("regression"),
        propensity_library=library("classification"),
        **kw,
    )
    raw = [w for w in rec2 if issubclass(w.category, ConvergenceWarning)]
    assert len(raw) == entry[0]["n_warnings"] >= 1
    assert not _assumption(rec2, "did not converge")
    assert "diagnostics" not in by_hand.model_info
    assert (by_hand.estimate, by_hand.se) == (res.estimate, res.se)

    # A default library that converges reports nothing.
    monkeypatch.setattr(
        SuperLearner,
        "_default_library",
        lambda self: (
            [LogisticRegression(C=0.01, max_iter=2000, random_state=0)]
            if self.task == "classification"
            else [LinearRegression()]
        ),
    )
    quiet, rec3 = _run(sp.tmle, df, **kw)
    assert not _assumption(rec3, "did not converge")
    assert "diagnostics" not in quiet.model_info
