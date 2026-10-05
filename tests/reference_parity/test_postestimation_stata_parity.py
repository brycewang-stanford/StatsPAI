"""Cross-language parity: ``test`` / ``lincom`` / p-values / ``margins`` against Stata.

Fixture: ``_fixtures/_generate_postestimation_stata.do`` (Stata 18, official
commands) on ``vce_grammar_data.csv``.

Three defects this file pins, each of which produced a plausible-looking
number rather than an error:

1. ``sp.test`` / ``sp.lincom`` rebuilt the coefficient covariance from the
   standard errors as a *diagonal* matrix. ``test x1 = x2`` after
   ``regress, vce(robust)`` gave F = 154.4 against Stata's 167.0.
2. Every ``EconometricResults`` reported t(N-K) p-values and intervals.
   Stata uses t(G-1) under ``regress, vce(cluster)`` (a 1e-32 p-value
   against Stata's 9e-15 on this design) and z after likelihood-based
   commands.
3. A misspelled coefficient in ``test`` / ``lincom`` was silently read as 0.

All comparisons are closed-form functionals of the fit, so the budget is the
CLAUDE.md default 1e-6 (observed agreement is ~1e-13).
"""

from __future__ import annotations

import json
import pathlib
import warnings
from functools import lru_cache
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

from statspai.exceptions import MethodIncompatibility

_FIX = pathlib.Path(__file__).parent / "_fixtures"
RTOL = 1e-6


@lru_cache(maxsize=None)
def _ref() -> Dict[str, Any]:
    path = _FIX / "postestimation_stata.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_postestimation_stata.do first")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _data() -> pd.DataFrame:
    return pd.read_csv(_FIX / "vce_grammar_data.csv")


def _fits() -> Dict[str, Any]:
    d = _data
    return {
        "regress_ols": lambda: sp.regress("yl ~ x1 + x2", d()),
        "regress_robust": lambda: sp.regress("yl ~ x1 + x2", d(), vce="robust"),
        "regress_cluster": lambda: sp.regress("yl ~ x1 + x2", d(), vce="cluster g"),
        "ivregress_small_cluster": lambda: sp.ivreg(
            "yiv ~ x1 + x2 + (endo ~ z1 + z2)", d(), cluster="g"
        ),
        "logit_oim": lambda: sp.logit("yb ~ x1 + x2", d()),
        "logit_cluster": lambda: sp.logit("yb ~ x1 + x2", d(), vce="cluster g"),
        "poisson_robust": lambda: sp.poisson("yc ~ x1 + x2", d(), vce="robust"),
    }


BLOCKS = sorted(_fits())


@lru_cache(maxsize=None)
def _fit(block: str) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _fits()[block]()


def _position(result: Any, term: str) -> int:
    return list(result.params.index).index(term)


@pytest.mark.parametrize("block", BLOCKS)
def test_coefficient_p_value_and_interval(block):
    ref, r = _ref()[block], _fit(block)
    i = _position(r, "x1")
    assert float(np.asarray(r.pvalues)[i]) == pytest.approx(ref["p_x1"], rel=RTOL)
    assert float(np.asarray(r.conf_int_lower)[i]) == pytest.approx(
        ref["ll_x1"], rel=RTOL
    )
    assert float(np.asarray(r.conf_int_upper)[i]) == pytest.approx(
        ref["ul_x1"], rel=RTOL
    )
    ci = r.conf_int()
    assert float(ci.iloc[i, 0]) == pytest.approx(ref["ll_x1"], rel=RTOL)
    tidy = r.tidy().set_index("term")
    assert float(tidy.loc["x1", "conf_low"]) == pytest.approx(ref["ll_x1"], rel=RTOL)


@pytest.mark.parametrize("block", BLOCKS)
def test_wald_tests(block):
    ref, r = _ref()[block], _fit(block)
    eq = sp.test(r, "x1 = x2")
    assert eq["statistic"] == pytest.approx(ref["test_eq_stat"], rel=RTOL)
    assert eq["pvalue"] == pytest.approx(ref["test_eq_p"], rel=RTOL, abs=1e-300)
    joint = r.test("x1 x2")
    assert joint["statistic"] == pytest.approx(ref["test_joint_stat"], rel=RTOL)
    chain = r.test("x1 = x2 = 0")
    assert chain["pvalue"] == pytest.approx(ref["test_chain_p"], rel=RTOL, abs=1e-300)
    expected = "F" if ref["df_r"] > 0 else "chi2"
    assert eq["distribution"] == expected


@pytest.mark.parametrize("block", BLOCKS)
def test_lincom(block):
    ref, r = _ref()[block], _fit(block)
    s = r.lincom("x1 + x2")
    assert s["estimate"] == pytest.approx(ref["lincom_sum_est"], rel=RTOL)
    assert s["se"] == pytest.approx(ref["lincom_sum_se"], rel=RTOL)
    assert s["pvalue"] == pytest.approx(ref["lincom_sum_p"], rel=RTOL, abs=1e-300)
    assert s["ci"][0] == pytest.approx(ref["lincom_sum_lb"], rel=RTOL)
    mix = sp.lincom(r, "x1 - 2*x2 + _cons")
    assert mix["estimate"] == pytest.approx(ref["lincom_mix_est"], rel=RTOL)
    assert mix["se"] == pytest.approx(ref["lincom_mix_se"], rel=RTOL)


MARGINS = {
    "logit_margins": lambda: sp.logit("yb ~ x1 + x2", _data()),
    "probit_margins": lambda: sp.probit("yp ~ x1 + x2", _data()),
    "poisson_margins": lambda: sp.poisson("yc ~ x1 + x2", _data(), vce="robust"),
    "regress_interaction_margins": lambda: sp.regress("yl ~ x1 + x2 + x1:x2", _data()),
    "logit_interaction_margins": lambda: sp.logit("yb ~ x1 + x2 + x1:x2", _data()),
}


@pytest.mark.parametrize("block", sorted(MARGINS))
def test_margins_dydx_matches_stata(block):
    """Average marginal effects on the prediction scale, delta-method SEs."""
    ref = _ref()[block]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = sp.margins(MARGINS[block](), data=_data()).set_index("variable")
    for v in ("x1", "x2"):
        assert float(m.loc[v, "dy/dx"]) == pytest.approx(ref[f"dydx_{v}"], rel=RTOL)
        assert float(m.loc[v, "se"]) == pytest.approx(ref[f"se_{v}"], rel=RTOL)
    # p-values in the far tail are compared on the log scale.
    assert np.log(float(m.loc["x1", "pvalue"])) == pytest.approx(
        np.log(ref["p_x1"]), rel=1e-4
    )


def test_margins_is_not_the_index_coefficient_after_logit():
    r = sp.logit("yb ~ x1 + x2", _data())
    m = sp.margins(r).set_index("variable")
    assert abs(float(m.loc["x1", "dy/dx"]) - float(r.params["x1"])) > 0.1


def test_margins_refuses_unsupported_models():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.ologit("yo ~ x1 + x2", _data())
    with pytest.raises(MethodIncompatibility, match="prediction scale"):
        sp.margins(r, data=_data())
    with pytest.raises(MethodIncompatibility, match="do not enter the model"):
        sp.margins(sp.regress("yl ~ x1", _data()), variables=["x2"])


def test_cluster_degrees_of_freedom_are_g_minus_1():
    assert _ref()["regress_cluster"]["df_r"] == 39
    assert _fit("regress_cluster").test("x1 = 0")["df"] == (1, 39)


@pytest.mark.parametrize(
    "bad", ["nope = 0", "x1 = x1", "x1*x2 = 0", "x1 = 0 = 1", "", "x1 + = 0"]
)
def test_bad_hypotheses_raise(bad):
    with pytest.raises(MethodIncompatibility):
        sp.test(_fit("regress_ols"), bad)


def test_multi_coefficient_restriction_needs_covariance():
    class _SEOnly:
        params = pd.Series([1.0, 2.0], index=["a", "b"])
        std_errors = pd.Series([0.1, 0.2], index=["a", "b"])
        data_info: Dict[str, Any] = {}

    assert sp.test(_SEOnly(), "a = 0")["chi2"] == pytest.approx(100.0)
    with pytest.raises(MethodIncompatibility, match="covariance"):
        sp.test(_SEOnly(), "a = b")
    with pytest.raises(MethodIncompatibility, match="covariance"):
        sp.lincom(_SEOnly(), "a + b")


def test_stale_covariance_is_not_used():
    """CR2 replaces std_errors after the fit; the covariance a test uses must
    be the CR2 one (stored in full since vce='cr2' keeps its matrix), never
    the OLS matrix the fit started from."""
    r = sp.regress("yl ~ x1 + x2", _data(), vce="CR2", cluster="g")
    single = sp.test(r, "x1 = 0")
    se = float(r.std_errors["x1"])
    assert single["chi2"] == pytest.approx((float(r.params["x1"]) / se) ** 2, rel=1e-12)
    V = np.asarray(r.data_info["var_cov"], dtype=float)
    names = list(r.params.index)
    assert np.sqrt(np.diag(V)) == pytest.approx(r.std_errors.to_numpy(), rel=1e-12)
    c = np.zeros(len(names))
    c[names.index("x1")], c[names.index("x2")] = 1.0, -1.0
    wald = float(c @ r.params.to_numpy()) ** 2 / float(c @ V @ c)
    assert sp.test(r, "x1 = x2")["chi2"] == pytest.approx(wald, rel=1e-10)
