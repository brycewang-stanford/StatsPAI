"""``sp.four_way_decomposition`` with logistic / Poisson outcomes and a
logistic mediator, against Stata ``med4way``.

Reference: Stata 18, ``med4way`` 2.3.2 (GPL-3, used as a black box) on
``_fixtures/four_way_nonlinear.csv`` from
``_fixtures/_generate_four_way_nonlinear_stata.do``, eight settings: every
outcome model (linear, logistic, Poisson) with a normal and with a logistic
mediator, covariates at their means or fixed by ``c()``, a non-unit
exposure contrast. ``med4way`` fits linear models by maximum likelihood,
so the comparison uses ``vce='ml'``. The regressions are run to tight
tolerances on both sides; estimates are held to 1e-7 and delta-method SEs
to 1e-6 (observed: 5e-9 and 6e-8). Its ``op_e`` SE is the exception: it
must equal the ``p_cde`` SE (``op_e = 1 - p_cde``), and ``med4way``'s
numerical derivatives give the two values 2e-6 apart in one setting, so
``prop_eliminated`` is held to the identity and to Stata's ``p_cde`` SE.

A second test checks the estimand itself: with a Poisson outcome and a
logistic mediator the components equal the excess relative risks computed
by simulating the potential outcomes of the data-generating process.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"

_CASES = {
    "1": dict(y="ybc", mediator="mc", yreg="logistic", mreg="linear"),
    "2": dict(y="ybb", mediator="mb", yreg="logistic", mreg="logistic"),
    "3": dict(y="ypc", mediator="mc", yreg="poisson", mreg="linear", m0=1.0),
    "4": dict(y="ypb", mediator="mb", yreg="poisson", mreg="logistic", m0=1.0),
    "5": dict(y="yc", mediator="mb", yreg="linear", mreg="logistic"),
    "6": dict(
        y="ybb",
        mediator="mb",
        yreg="logistic",
        mreg="logistic",
        covariate_values={"c1": 0.5, "c2": 1.0},
    ),
    "7": dict(y="ybc", mediator="mc", yreg="logistic", mreg="linear", m0=0.5),
    "8": dict(
        y="ypc",
        mediator="mc",
        yreg="poisson",
        mreg="linear",
        a0=-0.5,
        a1=1.5,
        m0=2.0,
        covariate_values={"c2": 0.0},
    ),
}

_STATA_NAME = {
    "cde": "cde",
    "int_ref": "intref",
    "int_med": "intmed",
    "pie": "pie",
    "prop_cde": "p_cde",
    "prop_int_ref": "p_intref",
    "prop_int_med": "p_intmed",
    "prop_pie": "p_pie",
    "prop_mediated": "op_m",
    "prop_interaction": "op_ati",
    "prop_eliminated": "op_e",
}


@pytest.fixture(scope="module")
def ref():
    return json.loads(
        (_FIX / "four_way_nonlinear_Stata.json").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "four_way_nonlinear.csv", float_precision="round_trip")


def _stata_key(ours, yreg):
    if ours == "total_effect":
        return "te" if yreg == "linear" else "tereri"
    name = _STATA_NAME[ours]
    if ours.startswith("prop_") or yreg == "linear":
        return name
    return "ereri_" + name


@pytest.mark.parametrize("case", sorted(_CASES))
def test_matches_med4way(ref, data, case):
    kw = dict(_CASES[case])
    cov = [] if case == "7" else ["c1", "c2"]
    r = sp.four_way_decomposition(data, treat="a", covariates=cov, vce="ml", **kw)
    R = ref[case]
    assert r.n_obs == R["N"]
    est = {
        "total_effect": r.total_effect,
        "cde": r.cde,
        "int_ref": r.int_ref,
        "int_med": r.int_med,
        "pie": r.pie,
    }
    est.update({k: r.proportions[k[5:]] for k in _STATA_NAME if k.startswith("prop_")})
    for ours, value in est.items():
        key = _stata_key(ours, kw["yreg"])
        assert value == pytest.approx(R["b"][key], rel=1e-7, abs=1e-10), ours
        if ours == "prop_eliminated":
            assert r.se[ours] == pytest.approx(r.se["prop_cde"], rel=1e-12)
            assert r.se[ours] == pytest.approx(R["se"]["p_cde"], rel=1e-6)
        else:
            assert r.se[ours] == pytest.approx(R["se"][key], rel=1e-6), ours
    if kw["yreg"] != "linear":
        assert r.detail["total_rr"] == pytest.approx(R["b"]["terira"], rel=1e-7)
        assert r.detail["scale"] == "excess_relative_risk"


def test_ols_and_ml_differ_only_through_the_variance_divisor(data):
    """With a normal mediator and a log-linear outcome the mediator variance
    enters the estimate: vce='ols' (RSS/(n-p), CMAverse) and 'ml' (RSS/n)
    give the components implied by those two variances."""
    kw = dict(
        y="ybc", treat="a", mediator="mc", covariates=["c1", "c2"], yreg="logistic"
    )
    ols = sp.four_way_decomposition(data, vce="ols", **kw)
    ml = sp.four_way_decomposition(data, vce="ml", **kw)
    n, p = len(data), 4
    assert ols.detail["mediator_variance"] == pytest.approx(
        ml.detail["mediator_variance"] * n / (n - p), rel=1e-12
    )
    # PIE = g(a0, a1) / g(a0, a0) - 1: both means carry the same factor
    # exp(theta2^2 s2 / 2) at a0 = 0, so PIE does not depend on s2; the CDE
    # (divided by g(a0, a0) only) does.
    assert ols.pie == pytest.approx(ml.pie, rel=1e-12)
    assert ols.cde != pytest.approx(ml.cde, rel=1e-6)


def test_poisson_logistic_components_equal_simulated_potential_outcomes():
    """Known truth: E[Y(a, M(a'))] by simulating the DGP's potential outcomes."""
    th = np.array([-0.4, 0.3, 0.5, 0.25])  # 1, a, m, a*m (log link)
    be = np.array([-0.3, 1.0])  # 1, a (logit)

    def g(a, ap, rng, n):
        m = rng.binomial(1, 1 / (1 + np.exp(-(be[0] + be[1] * ap))), n)
        return np.exp(th[0] + th[1] * a + (th[2] + th[3] * a) * m).mean()

    rng = np.random.default_rng(1)
    big = 4_000_000
    g00, g10, g01, g11 = (
        g(a, ap, rng, big) for a, ap in [(0, 0), (1, 0), (0, 1), (1, 1)]
    )
    gm = lambda a: np.exp(th[0] + th[1] * a)  # noqa: E731  (m* = 0)
    truth = {
        "cde": (gm(1) - gm(0)) / g00,
        "int_ref": (g10 - g00 - gm(1) + gm(0)) / g00,
        "int_med": (g11 - g01 - g10 + g00) / g00,
        "pie": (g01 - g00) / g00,
    }
    n = 100_000
    a = rng.integers(0, 2, n).astype(float)
    m = rng.binomial(1, 1 / (1 + np.exp(-(be[0] + be[1] * a))), n).astype(float)
    y = rng.poisson(np.exp(th[0] + th[1] * a + th[2] * m + th[3] * a * m))
    df = pd.DataFrame({"y": y, "a": a, "m": m})
    r = sp.four_way_decomposition(
        df, y="y", treat="a", mediator="m", yreg="poisson", mreg="logistic"
    )
    for k, v in truth.items():
        assert abs(getattr(r, k) - v) < 4 * r.se[k], k


def test_logistic_model_needs_binary_variable(data):
    with pytest.raises(sp.MethodIncompatibility, match="0/1"):
        sp.four_way_decomposition(
            data, y="ybc", treat="a", mediator="mc", mreg="logistic"
        )


def test_unknown_covariate_value_is_refused(data):
    with pytest.raises(sp.MethodIncompatibility, match="covariate_values"):
        sp.four_way_decomposition(
            data,
            y="ybc",
            treat="a",
            mediator="mc",
            covariates=["c1"],
            covariate_values={"c3": 1.0},
        )


def test_zero_total_effect_gives_nan_proportions_with_warning():
    a = np.tile([0.0, 1.0], 50)
    m = np.repeat(np.linspace(-1, 1, 50), 2)
    df = pd.DataFrame({"y": 1 + 2 * m, "a": a, "m": m})
    with pytest.warns(UserWarning, match="proportions are undefined"):
        r = sp.four_way_decomposition(df, y="y", treat="a", mediator="m")
    assert np.isnan(r.proportions["cde"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert abs(r.total_effect) <= 1e-10


@pytest.mark.parametrize("case", ["1", "2", "3", "4"])
def test_ols_estimates_match_cmaverse(data, case):
    """R CMAverse 0.1.0 (GPL-3, black box), ``cmest(estimation='paramfunc',
    inference='delta')`` from ``_generate_four_way_nonlinear_R.R``: its
    normal mediator variance is RSS/(n-p), our ``vce='ols'``. Estimates to
    1e-8. Standard errors only with a logistic mediator, to 5e-5: R ``glm``
    stops at a relative deviance change of 1e-8, which moves the
    covariance by up to 1.6e-5 here. With a normal mediator CMAverse omits
    the variance's sampling error, and its ERcde / ERintref SEs disagree
    with both ``med4way`` and ours (factors 2.8 and 0.8, see
    docs/dev/modules/treatment_effects.md), so they are not compared."""
    R = json.loads((_FIX / "four_way_nonlinear_R.json").read_text(encoding="utf-8"))[
        case
    ]
    kw = dict(_CASES[case])
    r = sp.four_way_decomposition(data, treat="a", covariates=["c1", "c2"], **kw)
    ours = {
        "Rte": (r.total_effect + 1.0, "total_effect"),
        "ERcde": (r.cde, "cde"),
        "ERintref": (r.int_ref, "int_ref"),
        "ERintmed": (r.int_med, "int_med"),
        "ERpnie": (r.pie, "pie"),
    }
    for name, (value, key) in ours.items():
        assert value == pytest.approx(R["b"][name], rel=1e-8), name
        if kw["mreg"] == "logistic":
            assert r.se[key] == pytest.approx(R["se"][name], rel=5e-5), name
