"""Survival / epidemiology / smoothing family vs R and Stata on identical bytes.

References
----------
``_fixtures/survival_epi_R.json`` is written by ``_generate_survival_epi_R.R``
(survival, cmprsk, epitools, DescTools, pROC, epiR; versions in the file's
``provenance``); ``_fixtures/survival_epi_stata.json`` by
``_fixtures/_generate_survival_epi_stata.do`` (Stata 18 with the SSC packages
stcompet 1.0.7 and diagt 2.032). Both read the CSVs that
``_fixtures/_generate_survival_epi_data.py`` writes.

Conventions each number depends on
----------------------------------
* ``cuminc``: the CIF is the Aalen-Johansen estimator (no convention).
  ``variance="gray"`` is cmprsk's asymptotic variance; ``variance="delta"``
  (default) is the Marubini-Valsecchi delta method that stcompet reports,
  and ``conf_type="log-log"`` stcompet's bounds. Gray's test is cmprsk's
  (unstratified, ``rho`` as given).
* ``cox`` (used as the Fine-Gray no-competing-risk identity): ``ties``
  as R ``coxph(ties=)`` / Stata ``stcox`` (Breslow default, ``efron``).
* ``finegray``: censoring KM at left limits (cmprsk and stcrreg both).
  ``vce="robust"`` (default) is cmprsk's ``var``; ``small_sample=True`` is
  stcrreg's ``N/(N-1)`` scaling of it; ``vce="model"`` is cmprsk's
  ``invinf``. Breslow ties throughout.
* ``cox_frailty``: gamma frailty with variance theta. At a fixed theta the
  fit equals R ``frailty(theta=, sparse=FALSE)`` and Stata ``stcox,
  shared()`` (full penalised information). The theta maximiser is compared
  with R ``optimize`` on the integrated likelihood (tolerance 1e-12) and
  with Stata's ``e(theta)``.
* ``direct_standardize``: ``ci_method="gamma"`` + Poisson variance is
  epitools; ``ci_method="normal"`` + ``variance="binomial"`` is dstdize.
  ``indirect_standardize``: exact Poisson CI is istdize, log-normal is
  epitools.
* ``breslow_day_test``: Mantel-Haenszel common OR; ``tarone_correction``.
* ``roc_curve``: AUC counts ties one half; DeLong variance (pROC, roctab
  default); ``se_method="hanley-empirical"`` is ``roctab, hanley``.
* ``sensitivity_specificity``: Wilson or Clopper-Pearson (epiR ``method``;
  diagt is exact and reports percentages).
* ``kdensity``: ``'epanechnikov'`` here is Stata's ``epan2``,
  ``'uniform'`` its ``rectangle``, ``'triangular'`` its ``triangle``.
  Bandwidths: ``'silverman'`` = R ``bw.nrd0``; ``'stata'`` = kdensity's
  default; ``'sheather-jones'`` = R ``bw.SJ`` without binning.
* ``lpoly``: ``se_method="stata"`` with ``pwidth``.
* ``power_case_control(test="chi2")``: Stata ``power twoproportions``
  (controls = group 1) and, for 1:1, R ``power.prop.test(strict=TRUE)``.

Tolerances
----------
1e-10 relative wherever both sides evaluate the same closed form or run the
same deterministic recursion (observed: 1e-16 to 1e-12). Looser, each with
its mechanism:

* stcrreg coefficients 1e-7 and SEs 1e-8: Stata's ``ml`` stops at its own
  gradient tolerance (coefficients observed 5e-9 off; the sandwich is
  evaluated there, 1.4e-10); the log likelihood agrees to 1e-16.
* Stata ``power`` 1e-11: Stata's normal CDF (observed 4e-13).
* Frailty theta: R ``optimize`` 1e-6 (observed 5e-8; the integrated
  likelihood is flat at the maximum, so theta is determined far less
  precisely than the likelihood). Stata's ``e(theta)`` 5e-5 (observed
  7.5e-6): Stata stops earlier on the flat maximum, and our integrated
  log likelihood there is at least as high as Stata's.
* Sheather-Jones vs ``bw.SJ(nb = 1e7, tol = 1e-14)`` 1e-6 (observed 2e-7):
  R bins the pairwise differences into ``nb`` classes, an O(1/nb) error.
  R's default (``nb = 1000``, ``tol = 0.1 * lower``) differs by 1.8e-3;
  that gap is asserted as a bound, not as parity.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

_FIX = Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "survival_epi_R.json").read_text(encoding="utf-8"))
S = json.loads((_FIX / "survival_epi_stata.json").read_text(encoding="utf-8"))

TOL = 1e-10


def _close(ours, ref, rtol=TOL, atol=0.0):
    np.testing.assert_allclose(
        np.asarray(ours, dtype=float),
        np.asarray(ref, dtype=float),
        rtol=rtol,
        atol=atol,
    )


def _csv(name):
    return pd.read_csv(_FIX / name)


# --------------------------------------------------------------------------
# cumulative incidence
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cr():
    return _csv("survival_epi_cr.csv")


def _at(res, cause, times, group=None):
    t = res.cif_table
    t = t[t["cause"] == cause]
    if group is not None:
        t = t[t["group"] == group]
    return t.set_index("time").loc[times]


@pytest.mark.parametrize("cause", [1, 2])
def test_cuminc_matches_cmprsk_estimate_and_gray_variance(cr, cause):
    ref = R["cuminc"][f"cause{cause}"]
    res = sp.cuminc(cr, duration="time", event="status", variance="gray")
    m = _at(res, cause, ref["time"])
    _close(m["cif"], ref["est"])
    _close(m["se"] ** 2, ref["var"])


@pytest.mark.parametrize("cause", [1, 2])
def test_cuminc_delta_variance_matches_stcompet(cr, cause):
    ref = S[f"stcompet_cause{cause}"]
    res = sp.cuminc(cr, duration="time", event="status", conf_type="log-log")
    m = _at(res, cause, ref["time"])
    _close(m["cif"], ref["ci"])
    _close(m["se"], ref["se"])
    if "lo" in ref:
        _close(m["ci_lower"], ref["lo"])
        _close(m["ci_upper"], ref["hi"])


def test_cuminc_by_group_matches_cmprsk(cr):
    res = sp.cuminc(cr, duration="time", event="status", group="grp", variance="gray")
    for g in (0, 1):
        ref = R["cuminc_grp"][f"g{g}_cause1"]
        m = _at(res, 1, ref["time"], group=g)
        _close(m["cif"], ref["est"])
        _close(m["se"] ** 2, ref["var"])


def test_gray_test_matches_cmprsk(cr):
    """Before this fix the statistic was a log-rank on unweighted
    subdistribution risk sets with a hypergeometric variance: 2.395 for
    cause 1 here against cmprsk's 1.850."""
    ref = R["cuminc_grp"]["tests"]
    res = sp.cuminc(cr, duration="time", event="status", group="grp")
    _close([res.gray_test[c]["statistic"] for c in (1, 2)], ref["stat"])
    _close([res.gray_test[c]["p_value"] for c in (1, 2)], ref["pv"])


def test_gray_test_rho_and_three_groups_match_cmprsk(cr):
    r1 = sp.cuminc(cr, duration="time", event="status", group="grp", rho=1.0)
    _close([r1.gray_test[c]["statistic"] for c in (1, 2)], R["gray_rho1"]["stat"])
    r3 = sp.cuminc(cr, duration="time", event="status", group="g3")
    _close([r3.gray_test[c]["statistic"] for c in (1, 2)], R["gray_g3"]["stat"])
    _close([r3.gray_test[c]["p_value"] for c in (1, 2)], R["gray_g3"]["pv"])
    assert [r3.gray_test[c]["df"] for c in (1, 2)] == R["gray_g3"]["df"]


def test_cuminc_identities(cr):
    """CIFs of all causes plus the all-cause KM sum to one; at the first
    event time the delta variance is the binomial p(1-p)/n."""
    res = sp.cuminc(cr, duration="time", event="status")
    t = res.cif_table
    wide = t.pivot(index="time", columns="cause", values="cif")
    time = cr["time"].to_numpy()
    fail = cr["status"].to_numpy() > 0
    surv = np.cumprod(
        [1.0 - np.sum(fail & (time == u)) / np.sum(time >= u) for u in wide.index]
    )
    _close(wide.sum(axis=1).to_numpy() + surv, 1.0, rtol=1e-13)
    first = t[t["cause"] == 1].iloc[0]
    n = len(cr)
    d1 = int(((cr["time"] == first["time"]) & (cr["status"] == 1)).sum())
    p = d1 / n
    _close(first["se"] ** 2, p * (1 - p) / n, rtol=1e-12)


# --------------------------------------------------------------------------
# Fine-Gray
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cause", [1, 2])
def test_finegray_matches_crr(cr, cause):
    """Before this fix the censoring KM was right-continuous, so with the
    censorings tied to event times here the coefficient on x1 was 0.44074
    (crr: 0.44060), and the SEs were the inverse information only."""
    ref = R[f"crr_cause{cause}"]
    x = ["x1", "x2", "grp"]
    res = sp.finegray(cr, duration="time", event="status", x=x, cause=cause)
    _close(res.params, ref["coef"])
    _close(res.bse, ref["se_robust"])
    _close(res.loglik, ref["loglik"])
    model = sp.finegray(
        cr, duration="time", event="status", x=x, cause=cause, vce="model"
    )
    _close(model.bse, ref["se_model"])


@pytest.mark.parametrize("cause", [1, 2])
def test_finegray_matches_stcrreg(cr, cause):
    ref = S[f"stcrreg_cause{cause}"]
    res = sp.finegray(
        cr,
        duration="time",
        event="status",
        x=["x1", "x2", "grp"],
        cause=cause,
        small_sample=True,
    )
    _close(res.params, ref["b"], rtol=1e-7)
    _close(res.bse, ref["se"], rtol=1e-8)
    _close(res.loglik, ref["ll"])


def test_finegray_without_competing_events_is_breslow_cox(cr):
    """No competing events: every weight is 1 and the model is Cox's."""
    d = cr[cr["status"] != 2]
    fg = sp.finegray(d, duration="time", event="status", x=["x1", "x2"], vce="model")
    cox = sp.cox(
        data=d, duration="time", event="status", x=["x1", "x2"], ties="breslow"
    )
    _close(fg.params, np.asarray(cox.params, dtype=float), rtol=1e-8)
    _close(fg.params, R["cox_breslow"]["coef"], rtol=1e-8)


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_cox_ties_match_coxph_and_stcox(cr, ties):
    """Before this fix sp.cox(ties="breslow") silently ran Efron: 0.3557
    for x1 here where coxph / stcox (Breslow) give 0.3356."""
    d = cr[cr["status"] != 2]
    res = sp.cox(data=d, duration="time", event="status", x=["x1", "x2"], ties=ties)
    for ref, b, ll, ll0 in (
        (R[f"cox_{ties}"], "coef", "loglik", "loglik0"),
        (S[f"stcox_{ties}"], "b", "ll", "ll_0"),
    ):
        # sp.cox stops Newton at a 1e-9 absolute step (x2 is 0.045).
        _close(res.params, ref[b], rtol=1e-8)
        _close(res.std_errors, ref["se"], rtol=1e-8)
        _close(res.diagnostics["Log-likelihood"], ref[ll])
        _close(res.diagnostics["Log-likelihood (null)"], ref[ll0])


# --------------------------------------------------------------------------
# shared gamma frailty
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fr():
    return _csv("survival_epi_frailty.csv")


_FORMULA = "time + event ~ x1 + x2"


@pytest.mark.parametrize(
    "key", ["frailty_fixed_050", "frailty_fixed_024", "frailty_ml"]
)
def test_cox_frailty_fixed_theta_matches_coxph(fr, key):
    """Before this fix sp.cox_frailty ignored the frailties when updating
    beta (it returned the ordinary Cox fit) and searched theta on [0.5, 100]
    of a mis-specified likelihood."""
    ref = R[key]
    res = sp.cox_frailty(_FORMULA, fr, cluster="cid", theta=ref["theta"])
    _close(res.beta, ref["coef"], rtol=1e-9)
    _close(res.se, ref["se"], rtol=1e-9)
    _close(res.log_frailties, ref["log_frailty"], rtol=1e-8, atol=1e-10)
    _close(res.log_likelihood, ref["c_loglik"])


def test_cox_frailty_at_stata_theta_matches_stcox_shared(fr):
    ref = S["stcox_shared"]
    res = sp.cox_frailty(_FORMULA, fr, cluster="cid", theta=ref["theta"])
    _close(res.beta, ref["b"])
    _close(res.se, ref["se"])
    _close(res.log_likelihood, ref["ll"])
    _close(res.loglik_cox, ref["ll_c"], rtol=1e-9)
    _close(res.loglik_cox, R["cox_loglik"])


def test_cox_frailty_theta_maximiser(fr):
    res = sp.cox_frailty(_FORMULA, fr, cluster="cid")
    _close(res.theta, R["frailty_ml"]["theta"], rtol=1e-6)
    _close(res.log_likelihood, R["frailty_ml"]["c_loglik"])
    _close(res.theta, S["stcox_shared"]["theta"], rtol=5e-5)
    assert res.log_likelihood >= S["stcox_shared"]["ll"] - 1e-9
    lr, p = res.lr_theta0
    _close(lr, S["stcox_shared"]["chi2_c"], rtol=1e-6)
    assert 0.0 < p < 0.05


def test_cox_frailty_identities(fr):
    """theta = 0 is the Cox model; at any theta > 0 the fitted frailties
    average one (the stationarity condition of the gamma penalty)."""
    zero = sp.cox_frailty(_FORMULA, fr, cluster="cid", theta=0.0)
    cox = sp.cox(data=fr, duration="time", event="event", x=["x1", "x2"])
    _close(zero.beta, np.asarray(cox.params, dtype=float), rtol=1e-8)
    res = sp.cox_frailty(_FORMULA, fr, cluster="cid", theta=0.5)
    _close(np.mean(res.frailties), 1.0, rtol=1e-8)


# --------------------------------------------------------------------------
# standardisation
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def std():
    return _csv("survival_epi_std.csv")


def test_direct_standardize_matches_epitools(std):
    ref = R["direct"]
    res = sp.direct_standardize(
        std["events"], std["pop"], std["std_pop"], ci_method="gamma"
    )
    _close(res.rate, ref["adj.rate"])
    _close(res.ci, [ref["lci"], ref["uci"]])


def test_direct_standardize_matches_dstdize(std):
    ref = S["dstdize"]
    res = sp.direct_standardize(
        std["events"],
        std["pop"],
        std["std_pop"],
        ci_method="normal",
        variance="binomial",
    )
    _close(res.rate, ref["adj"])
    _close(res.se, ref["se"])
    _close(res.ci, [ref["lb"], ref["ub"]])


def test_indirect_standardize_matches_istdize_and_epitools(std):
    obs = float(std["events"].sum())
    args = (obs, std["ref_events"], std["ref_pop"], std["pop"])
    ex = sp.indirect_standardize(*args)
    ref = S["istdize"]
    _close(ex.smr, ref["smr"])
    _close(ex.expected, ref["cases_exp"])
    _close(ex.ci, [ref["lb"], ref["ub"]])
    ln = sp.indirect_standardize(*args, ci_method="lognormal")
    ref = R["indirect"]
    _close(ln.smr, ref["sir"])
    _close(ln.expected, ref["exp"])
    _close(ln.ci, [ref["lci"], ref["uci"]])


# --------------------------------------------------------------------------
# Breslow-Day
# --------------------------------------------------------------------------
def test_breslow_day_matches_desctools_and_stata():
    bd = _csv("survival_epi_bd.csv")
    tabs = [[[r.a, r.b], [r.c, r.d]] for r in bd.itertuples()]
    c0, p0 = sp.breslow_day_test(tabs, tarone_correction=False)
    c1, p1 = sp.breslow_day_test(tabs)
    ref = R["breslow_day"]
    _close(
        [c0, p0, c1, p1], [ref["stat"], ref["p"], ref["stat_tarone"], ref["p_tarone"]]
    )
    ref = S["cc_bd"]
    _close([c0, p0, c1, p1], [ref["bd"], ref["p_bd"], ref["tarone"], ref["p_tarone"]])
    _close(sp.mantel_haenszel(tabs).estimate, ref["or_mh"])
    _close(sp.mantel_haenszel(tabs).estimate, R["breslow_day"]["or_mh"])


# --------------------------------------------------------------------------
# ROC / AUC
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def roc():
    return _csv("survival_epi_roc.csv")


@pytest.mark.parametrize(
    "col,rkey,skey",
    [("score", "roc", "roctab_score"), ("score_tied", "roc_tied", "roctab_score_tied")],
)
def test_roc_matches_proc_and_roctab(roc, col, rkey, skey):
    """Before this fix tied scores were traversed one observation at a
    time in sort order, so the AUC depended on the (unstable) order of
    tied cases and controls: 0.669 vs the Mann-Whitney 0.667 on one draw."""
    dl = sp.roc_curve(roc["y"], roc[col], se_method="delong")
    ref = R[rkey]
    _close(dl.auc, ref["auc"])
    _close(dl.auc_se**2, ref["var_delong"])
    _close(dl.auc_ci, [ref["ci_lo"], ref["ci_hi"]])
    ref = S[skey]
    _close(dl.auc, ref["area"])
    _close(dl.auc_se, ref["se"])
    _close(dl.auc_ci, [ref["lb"], ref["ub"]])
    he = sp.roc_curve(roc["y"], roc[col], se_method="hanley-empirical")
    _close(he.auc_se, ref["se_hanley"])
    _close(he.auc_ci, [ref["lb_hanley"], ref["ub_hanley"]])
    _close(sp.auc(roc["y"], roc[col]), R[rkey]["auc"])


def test_auc_with_ties_is_mann_whitney_and_order_free(roc):
    y = roc["y"].to_numpy()
    s = roc["score_tied"].to_numpy()
    r = rankdata(s)
    n1 = y.sum()
    n0 = len(y) - n1
    mw = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    _close(sp.auc(y, s), mw, rtol=1e-14)
    perm = np.random.default_rng(0).permutation(len(y))
    _close(sp.auc(y[perm], s[perm]), mw, rtol=1e-14)
    rc = sp.roc_curve(y, s)
    assert len(rc.thresholds) == len(np.unique(s))
    _close(np.trapezoid(np.r_[0.0, rc.tpr], np.r_[0.0, rc.fpr]), mw, rtol=1e-12)


# --------------------------------------------------------------------------
# sensitivity / specificity
# --------------------------------------------------------------------------
@pytest.mark.parametrize("k", [0, 1, 2])
def test_sensitivity_specificity_matches_epir_and_diagt(k):
    r = _csv("survival_epi_diag.csv").iloc[k]
    cells = dict(tp=int(r.tp), fn=int(r.fn), fp=int(r.fp), tn=int(r.tn))
    for method in ("wilson", "exact"):
        o = sp.sensitivity_specificity(**cells, ci_method=method)
        ref = R["diag"][k][method]
        _close([o.sensitivity, *o.sensitivity_ci], ref["se"], rtol=1e-12)
        _close([o.specificity, *o.specificity_ci], ref["sp"], rtol=1e-12)
        _close([o.ppv, o.npv], [ref["pv_pos"][0], ref["pv_neg"][0]])
        _close([o.lr_pos, o.lr_neg], [ref["lr_pos"], ref["lr_neg"]])
    o = sp.diagnostic_test(**cells, ci_method="exact")
    ref = S[f"diagti_{k + 1}"]
    _close(100 * np.array([o.sensitivity, *o.sensitivity_ci]), ref["sens"], rtol=1e-12)
    _close(100 * np.array([o.specificity, *o.specificity_ci]), ref["spec"], rtol=1e-12)
    _close(
        100 * np.array([o.ppv, o.npv, o.prevalence]),
        [ref["ppv"], ref["npv"], ref["prev"]],
    )
    _close([o.lr_pos, o.lr_neg], [ref["lrp"], ref["lrn"]])


# --------------------------------------------------------------------------
# kernel density / local polynomial
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def kd():
    return _csv("survival_epi_kd.csv")


def test_kdensity_bandwidth_rules(kd):
    _close(sp.kdensity(kd, x="x").bandwidth, R["bw"]["nrd0"])
    _close(
        sp.kdensity(kd, x="x", bw_method="stata").bandwidth,
        S["kdensity_default"]["bwidth"],
    )
    sj = sp.kdensity(kd, x="x", bw_method="sheather-jones").bandwidth
    _close(sj, R["bw"]["sj_nb1e7"], rtol=1e-6)
    # R's default bw.SJ bins and stops early; the gap is bounded, not parity.
    assert abs(sj / R["bw"]["sj_default"] - 1) < 3e-3
    # Before this fix 'sheather-jones' returned the Silverman rule.
    assert abs(sj / sp.kdensity(kd, x="x").bandwidth - 1) > 0.05


@pytest.mark.parametrize(
    "ours,stata",
    [
        ("epanechnikov", "epan2"),
        ("biweight", "biweight"),
        ("gaussian", "gaussian"),
        ("uniform", "rectangle"),
        ("triangular", "triangle"),
    ],
)
def test_kdensity_matches_stata_kernels(kd, ours, stata):
    ref = S["kdensity_at"]
    res = sp.kdensity(kd, x="x", bandwidth=0.7, kernel=ours, grid=np.array(ref["at"]))
    _close(res.density, ref[stata])


@pytest.mark.parametrize(
    "degree,kernel,stata",
    [
        (1, "epanechnikov", "epan2"),
        (0, "epanechnikov", "epan2"),
        (2, "gaussian", "gaussian"),
        (1, "biweight", "biweight"),
    ],
)
def test_lpoly_matches_stata(kd, degree, kernel, stata):
    ref = S["lpoly_at"]
    res = sp.lpoly(
        kd,
        y="y",
        x="x",
        bandwidth=0.8,
        degree=degree,
        kernel=kernel,
        grid=np.array(ref["at"]),
        se_method="stata",
        pwidth=1.2,
    )
    _close(res.fitted, ref[f"fit_{degree}_{stata}"])
    _close(res.se, ref[f"se_{degree}_{stata}"])


def test_lpoly_stata_se_with_constant_variance_identity(kd):
    """With a straight line and no noise the local residual variance is 0."""
    d = kd.assign(y=1.0 + 2.0 * kd["x"])
    res = sp.lpoly(
        d,
        y="y",
        x="x",
        bandwidth=0.8,
        degree=1,
        grid=np.array([2.0, 3.0]),
        se_method="stata",
        pwidth=1.2,
    )
    _close(res.fitted, [5.0, 7.0], rtol=1e-12)
    _close(res.se, [0.0, 0.0], atol=1e-6)


# --------------------------------------------------------------------------
# case-control power
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "key,nc,ratio,alt",
    [
        ("n200_r2", 200, 2, "two-sided"),
        ("n200_r1", 200, 1, "two-sided"),
        ("n200_r1_one", 200, 1, "one-sided"),
        ("n30_r2", 30, 2, "two-sided"),
        ("n50_r3", 50, 3, "two-sided"),
    ],
)
def test_power_case_control_chi2_matches_stata(key, nc, ratio, alt):
    res = sp.power_case_control(
        n_cases=nc,
        odds_ratio=2.0,
        exposure_prevalence=0.3,
        ratio=ratio,
        alternative=alt,
        test="chi2",
    )
    _close(res.power, S["power_cc"][key], rtol=1e-11)


def test_power_case_control_chi2_matches_power_prop_test():
    ref = R["power_prop"]
    kw = dict(odds_ratio=2.0, exposure_prevalence=0.3, test="chi2")
    _close(sp.power_case_control(n_cases=200, **kw).power, ref["n200"])
    _close(sp.power_case_control(n_cases=60, **kw).power, ref["n60"])
    _close(
        sp.power_case_control(n_cases=200, alternative="one-sided", **kw).power,
        ref["n200_one"],
    )


def test_power_case_control_sample_size_is_minimal():
    kw = dict(odds_ratio=2.0, exposure_prevalence=0.3, ratio=2.0, test="chi2")
    n = sp.power_case_control(power_target=0.8, **kw).n
    assert sp.power_case_control(n_cases=n, **kw).power >= 0.8
    assert sp.power_case_control(n_cases=n - 1, **kw).power < 0.8
