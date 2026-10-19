"""``sp.from_r`` mappings added 2026-10-19, each executed and compared with
the R function it translates.

R 4.5 with AER, survival, pwr, survey, sandwich + lmtest, etwfe 0.6.2 +
fixest 0.14.0, on the data built here (written with ``%.17g``, read back
with ``read.csv``), printed with ``sprintf("%.15e")``. Tolerance 1e-8
except for ``etwfe``: ``emfx`` takes its standard errors from
marginaleffects' numerical Jacobian, which agrees with StatsPAI's analytic
delta method to about 3e-6 (1e-5 here); the point estimates agree to 1e-12.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

RTOL = 1e-8


@pytest.fixture(scope="module")
def rx():
    r = np.random.default_rng(21)
    n = 500
    z = r.normal(size=n)
    u = r.normal(size=n)
    x1 = r.normal(size=n)
    x2 = 0.6 * z + 0.5 * u + r.normal(size=n)
    y = 1 + 0.5 * x1 - 0.7 * x2 + u + r.normal(size=n)
    st = r.integers(1, 5, n)
    psu = st * 100 + r.integers(1, 6, n)
    w = r.uniform(0.5, 3, n)
    yb = (y > 1).astype(int)
    return pd.DataFrame(dict(y=y, x1=x1, x2=x2, z=z, st=st, psu=psu, w=w, yb=yb))


@pytest.fixture(scope="module")
def surv():
    r = np.random.default_rng(11)
    n = 300
    x1 = r.normal(size=n)
    x2 = r.binomial(1, 0.4, n)
    g = r.integers(1, 31, n)
    st = r.integers(0, 2, n)
    t = np.ceil(r.exponential(1 / np.exp(0.5 * x1 - 0.4 * x2)) * 10)
    c = np.ceil(r.exponential(1.5, n) * 10)
    time = np.minimum(t, c)
    d = (t <= c).astype(int)
    return pd.DataFrame(dict(time=time, d=d, x1=x1, x2=x2, g=g, st=st))


def _run(line: str, data: pd.DataFrame, **extra):
    out = sp.from_r(line)
    assert out["ok"], out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return out, getattr(sp, out["tool"])(data=data, **out["arguments"], **extra)


def _close(a, b, rtol=RTOL):
    return bool(np.isclose(float(a), b, rtol=rtol, atol=0))


@pytest.mark.parametrize(
    "line",
    [
        "AER::ivreg(y ~ x1 + x2 | x1 + z, data = d)",
        "ivreg::ivreg(y ~ x1 | x2 | z, data = d)",
    ],
)
def test_ivreg(rx, line):
    out, r = _run(line, rx)
    assert out["arguments"]["formula"] == "y ~ x1 + (x2 ~ z)"
    assert _close(r.params["x2"], -5.899032950789854e-01)
    assert _close(r.std_errors["x2"], 1.216532599580228e-01)
    assert _close(r.std_errors["x1"], 6.813154590885113e-02)


@pytest.mark.parametrize(
    "line, name, b, se",
    [
        (
            "coxph(Surv(time, d) ~ x1 + x2, data = s)",
            "x1",
            4.213519729292841e-01,
            8.366412080940303e-02,
        ),
        (
            "coxph(Surv(time, d) ~ x1 + x2, data = s, robust = TRUE)",
            "x2",
            -5.911050684392980e-01,
            1.591406602223233e-01,
        ),
        (
            'coxph(Surv(time, d) ~ x1 + x2 + strata(st), data = s, ties = "breslow", '
            "robust = TRUE)",
            "x1",
            4.030835565514496e-01,
            8.457381534429873e-02,
        ),
    ],
)
def test_coxph(surv, line, name, b, se):
    out, r = _run(line, surv)
    assert out["complete"]
    assert _close(r.params[name], b) and _close(r.std_errors[name], se)


def test_coxph_cluster_maps_to_no_small_sample_factor():
    """coxph's clustered variance has no G/(G-1): sp.cox(ssc='r'); the
    numbers are pinned in reference_parity/test_survival_weighted_parity.py."""
    out = sp.from_r("coxph(Surv(time, d) ~ x1, data = s, cluster = g)")
    assert out["complete"] is True
    assert out["arguments"]["cluster"] == "g" and out["arguments"]["ssc"] == "r"
    # the cluster() term and id= with robust = TRUE are the same variance
    for line in (
        "coxph(Surv(time, d) ~ x1 + cluster(g), data = s)",
        "coxph(Surv(time, d) ~ x1, data = s, id = g, robust = TRUE)",
    ):
        args = sp.from_r(line)["arguments"]
        assert args["cluster"] == "g" and args["ssc"] == "r"
    args = sp.from_r("coxph(Surv(t0, t1, d) ~ x1, data = s)")["arguments"]
    assert args["start"] == "t0" and args["duration"] == "t1"


@pytest.mark.parametrize(
    "line, n, power",
    [
        # pwr: n = 63.77 per group; sp reports the total rounded up per group
        ("pwr.t.test(d = 0.5, power = 0.8)", 128, None),
        ("pwr.t.test(n = 50, d = 0.5)", 100, 6.968934055335295e-01),
        ('pwr.t.test(n = 30, d = 0.5, type = "paired")', 30, 7.539647157438569e-01),
        (
            'pwr.t.test(n = 30, d = 0.5, type = "one.sample", alternative = "greater")',
            30,
            8.482541909291499e-01,
        ),
    ],
)
def test_pwr_t_test(line, n, power):
    out = sp.from_r(line)
    r = sp.power_ttest(**out["arguments"])
    assert int(r.n) == n
    if power is not None:
        assert _close(r.power, power)
    else:
        assert int(np.ceil(6.376561044413734e01)) * 2 == n


def test_svydesign_and_svyglm(rx):
    design_out = sp.from_r(
        "svydesign(ids = ~psu, strata = ~st, weights = ~w, data = d, nest = TRUE)"
    )
    assert design_out["arguments"] == {
        "weights": "w",
        "strata": "st",
        "cluster": "psu",
        "nest": True,
    }
    des = sp.svydesign(rx, **design_out["arguments"])
    out = sp.from_r("svyglm(y ~ x1 + x2, design = des)")
    r = sp.svyglm(design=des, **out["arguments"])
    assert _close(r.params["x2"], -3.742239331170247e-01)
    assert _close(r.std_errors["x2"], 4.127721042373272e-02)
    # dof='residual' is R's summary.svyglm reference distribution
    assert _close(
        r.p_values[list(r.params.index).index("x2")], 3.101749967912252e-07, rtol=1e-6
    )
    out = sp.from_r("svyglm(yb ~ x1 + x2, design = des, family = quasibinomial())")
    r = sp.svyglm(design=des, **out["arguments"])
    # IRLS stopping rules differ in the eighth digit
    assert _close(r.params["x1"], 7.074211756648576e-01, rtol=1e-7)
    assert _close(r.std_errors["x1"], 1.585514668111425e-01, rtol=1e-7)


@pytest.mark.parametrize(
    "vcov, se",
    [
        ('vcovHC(m, type = "HC1")', 7.156477998303300e-02),
        ("vcovHC(m)", 7.203341822250398e-02),  # sandwich's default is HC3
        ("vcovCL(m, cluster = ~psu)", 8.395324516895447e-02),
    ],
)
def test_coeftest_with_sandwich(rx, vcov, se):
    out = sp.from_r(f"coeftest(m, vcov = {vcov})")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = sp.regress("y ~ x1 + x2", data=rx)
        r = sp.vcov(m, out["arguments"]["vce"])
    assert _close(r.params["x1"], 4.688171965246452e-01)
    assert _close(r.std_errors["x1"], se)


@pytest.fixture(scope="module")
def mpdta():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.datasets.mpdta()


@pytest.mark.parametrize(
    "extra, b, se",
    [
        ("", -3.510827660810594e-02, 6.925108473840879e-03),
        (', cgroup = "never"', -3.297651380965732e-02, 7.777058201813744e-03),
        (", ivar = countyreal", -3.510827660810419e-02, 6.920906434828933e-03),
    ],
)
def test_etwfe(mpdta, extra, b, se):
    line = (
        "etwfe(lemp ~ 0, tvar = year, gvar = first_treat, data = d, "
        f"vcov = ~countyreal{extra})"
    )
    out, r = _run(line, mpdta)
    assert out["complete"]
    assert _close(r.estimate, b, rtol=1e-10)
    assert _close(r.se, se, rtol=1e-5)


def test_emfx_event(mpdta):
    _, fit = _run(
        "etwfe(lemp ~ 0, tvar = year, gvar = first_treat, data = d, "
        "vcov = ~countyreal)",
        mpdta,
    )
    out = sp.from_r('emfx(m, type = "event")')
    ev = sp.etwfe_emfx(fit, **out["arguments"]).detail
    ref = {
        0: (-3.755145672511043e-02, 6.269770215591061e-03),
        1: (-3.158364223395704e-02, 8.160421777958445e-03),
        3: (-2.542051734427808e-02, 1.343160144915184e-02),
    }
    got = {int(e): (b, s) for e, b, s in zip(ev.event_time, ev.estimate, ev.se)}
    for e, (b, s) in ref.items():
        assert _close(got[e][0], b, rtol=1e-10) and _close(got[e][1], s, rtol=1e-5)


def test_etwfe_without_vcov_is_incomplete():
    out = sp.from_r(
        "etwfe(lemp ~ 0, tvar = year, gvar = first_treat, data = d, ivar = id)"
    )
    assert out["complete"] is False and out["dropped"] == ["vcov"]


def test_translations_to_mirrored_functions():
    out = sp.from_r('aggte(out, type = "dynamic", min_e = -5, max_e = 5)')
    assert out["python_code"] == "sp.aggte(out, type='dynamic', min_e=-5.0, max_e=5.0)"
    out = sp.from_r("aggte(out)")
    assert out["arguments"]["type"] == "group"  # did::aggte's default
    out = sp.from_r(
        'did_multiplegt_dyn(df, "y", "g", "t", "d", effects = 3, placebo = 2, '
        'cluster = "g")'
    )
    assert out["arguments"] == {
        "y": "y",
        "group": "g",
        "time": "t",
        "treatment": "d",
        "dynamic": 3,
        "placebo": 2,
        "se_method": "analytic",
        "cluster": "g",
    }
    # R's effects defaults to 1, sp's dynamic to 3: written out
    assert (
        sp.from_r('did_multiplegt_dyn(df, "y", "g", "t", "d")')["arguments"]["dynamic"]
        == 1
    )
    out = sp.from_r('modelsummary(list("A" = m1, "B" = m2))')
    assert out["python_code"] == "sp.modelsummary(m1, m2, model_names=['A', 'B'])"


@pytest.mark.parametrize(
    "line, function",
    [
        ("weightit(d ~ x, data = df)", "ps_weights"),
        ("synthdid_estimate(setup$Y, setup$N0, setup$T0)", "sdid"),
        ('mediation::mediate(med, out, treat = "d", mediator = "m")', "mediate"),
        ("CausalImpact(data, pre.period, post.period)", "causal_impact"),
        ("spatialreg::lagsarlm(y ~ x, data = df, listw = lw)", "sar"),
        (
            "pcalg::pc(suffStat, indepTest = gaussCItest, alpha = 0.01, labels = v)",
            "pc_algorithm",
        ),
        ("MendelianRandomization::mr_ivw(inp)", "mr_ivw"),
    ],
)
def test_guidance_names_an_existing_function(line, function):
    out = sp.from_r(line)
    assert out["ok"] is False
    assert function in out["statspai_functions"]
    assert callable(getattr(sp, function))
