"""Cross-language parity: ``sp.margins`` factor / transform / weight / exposure
extensions (1.32) against Stata 18 ``margins``.

Fixtures: ``_fixtures/_generate_margins_ext_data.py`` -> ``margins_ext_data.csv``
-> ``_fixtures/_generate_margins_ext_stata.do`` -> ``margins_ext_stata.json``
(whole ``r(table)`` at 17 significant digits).

=========================================================  ==================================
StatsPAI                                                   Stata
=========================================================  ==================================
``regress("yl ~ x + I(x**2) + C(g) + z")``                 ``regress yl c.x##c.x i.g z``
``margins(r, d)`` (dy/dx of x incl. 2 b2 x; ``2.g`` rows)  ``margins, dydx(x g z)``
``method="mem"``                                           ``... atmeans``
``regress(..., weights="w")`` + margins / margins_at /     ``regress ... [aw=w]`` +
contrast                                                   ``margins`` (weighted average)
``logit`` / ``probit`` with I(x**2), C(g)*z                ``logit`` / ``probit``, tight tol.
``poisson(..., exposure="expo")``, ``glm(..., exposure=)`` ``poisson/glm ..., exposure(expo)``
=========================================================  ==================================

Tolerances: OLS margins are closed-form functionals of (b, V) except the x^2
path, which StatsPAI differentiates by central differences on the rebuilt
design (exact for a polynomial up to rounding): asserted at 1e-8. ML fits in
Stata use 1e-14 convergence tolerances; the coefficient gap is then ~1e-10
and margins are asserted at 1e-7.
"""

from __future__ import annotations

import json
import pathlib
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
LIN = 1e-8
ML = 1e-7


@lru_cache(maxsize=None)
def _stata():
    return json.loads((_FIX / "margins_ext_stata.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _data() -> pd.DataFrame:
    return pd.read_csv(_FIX / "margins_ext_data.csv")


def T(key: str) -> pd.DataFrame:
    m = _stata()[key]
    v = np.array([[np.nan if a is None else a for a in row] for row in m["v"]])
    return pd.DataFrame(v, index=m["rows"], columns=m["cols"])


FITS = {
    "ols_quad": lambda d: sp.regress("yl ~ x + I(x**2) + C(g) + z", d),
    "ols_aw": lambda d: sp.regress("yl ~ C(g)*x + z", d, weights="w"),
    "logit_quad": lambda d: sp.logit("yb ~ x + I(x**2) + C(g) + z", d),
    "probit_gz": lambda d: sp.probit("yb ~ x + C(g)*z", d),
    "poisson_expo": lambda d: sp.poisson("yc ~ x + z + C(g)", d, exposure="expo"),
    "glm_expo": lambda d: sp.glm("yc ~ x + z", d, family="poisson", exposure="expo"),
}


@lru_cache(maxsize=None)
def _fit(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return FITS[name](_data())


def _stata_rows(key):
    """Stata r(table) as {label: (b, se)}, base-level columns dropped."""
    t = T(key)
    out = {}
    for col in t.columns:
        if col.startswith("1b."):
            continue
        out[col] = (t.loc["b", col], t.loc["se", col], t.loc["pvalue", col])
    return out


def _check(m, key, rtol):
    ref = _stata_rows(key)
    got = m.set_index("variable")
    assert set(got.index) == set(ref), (sorted(got.index), sorted(ref))
    for lab, (b, se, pv) in ref.items():
        assert got.loc[lab, "dy/dx"] == pytest.approx(b, rel=rtol, abs=1e-12), lab
        assert got.loc[lab, "se"] == pytest.approx(se, rel=rtol), lab
        assert np.log(got.loc[lab, "pvalue"]) == pytest.approx(np.log(pv), rel=1e-4)


@pytest.mark.parametrize(
    "fit,key,rtol",
    [
        ("ols_quad", "ols_quad__dydx", LIN),
        ("ols_aw", "ols_aw__dydx", LIN),
        ("logit_quad", "logit_quad__dydx", ML),
        ("probit_gz", "probit_gz__dydx", ML),
        ("poisson_expo", "poisson_expo__dydx", ML),
        ("glm_expo", "glm_expo__dydx", ML),
    ],
)
def test_dydx_all_variables_matches_stata(fit, key, rtol):
    _check(sp.margins(_fit(fit), _data()), key, rtol)


@pytest.mark.parametrize(
    "fit,key,rtol",
    [
        ("ols_quad", "ols_quad__dydx_atmeans", LIN),
        ("logit_quad", "logit_quad__dydx_atmeans", ML),
    ],
)
def test_atmeans_with_quadratic_and_factor_matches_stata(fit, key, rtol):
    _check(sp.margins(_fit(fit), _data(), method="mem"), key, rtol)


def test_dydx_of_quadratic_at_values_matches_stata():
    ref = T("ols_quad__dydx_x_at_x")
    for j, v in enumerate([-1.0, 0.0, 1.0]):
        m = sp.margins(_fit("ols_quad"), _data(), variables=["x"], at={"x": v})
        assert float(m["dy/dx"].iloc[0]) == pytest.approx(ref.loc["b"].iloc[j], rel=LIN)
        assert float(m["se"].iloc[0]) == pytest.approx(ref.loc["se"].iloc[j], rel=LIN)


def test_quadratic_ame_identity():
    """Reference-free (T1): AME of x in x + x^2 is b1 + 2 b2 mean(x)."""
    r, d = _fit("ols_quad"), _data()
    m = sp.margins(r, d, variables=["x"])
    hand = r.params["x"] + 2 * r.params["I(x ** 2)"] * d["x"].mean()
    assert float(m["dy/dx"].iloc[0]) == pytest.approx(hand, rel=1e-9)


def test_log_transform_ame_identity():
    """Reference-free (T1): AME of x under b log(x) is b mean(1/x)."""
    d = _data().assign(xp=lambda f: np.exp(f["x"]))
    r = sp.regress("yl ~ np.log(xp) + z", d)
    m = sp.margins(r, d, variables=["xp"])
    hand = r.params["np.log(xp)"] * np.mean(1.0 / d["xp"])
    assert float(m["dy/dx"].iloc[0]) == pytest.approx(hand, rel=1e-8)


def test_weighted_predictive_margins_and_contrast_match_stata():
    r, d = _fit("ols_aw"), _data()
    m = sp.margins_at(r, d, at={"x": [0.0, 1.0]})
    ref = T("ols_aw__at_x")
    np.testing.assert_allclose(m["margin"], ref.loc["b"], rtol=LIN)
    np.testing.assert_allclose(m["se"], ref.loc["se"], rtol=LIN)
    c = sp.contrast(r, d, "g", method="r")
    ref = T("ols_aw__r_g")
    np.testing.assert_allclose(c["contrast"], ref.loc["b"], rtol=LIN)
    np.testing.assert_allclose(c["se"], ref.loc["se"], rtol=LIN)


def test_exposure_enters_predictive_margins():
    m = sp.margins_at(_fit("poisson_expo"), _data(), at={"x": [0.0, 1.0]})
    ref = T("poisson_expo__at_x")
    np.testing.assert_allclose(m["margin"], ref.loc["b"], rtol=ML)
    np.testing.assert_allclose(m["se"], ref.loc["se"], rtol=ML)


def test_weights_needed_but_missing_fail_loudly():
    r = _fit("ols_aw")
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="weights"):
        sp.margins(r, _data().drop(columns="w"), variables=["x"])


def test_transformed_model_without_data_fails_loudly():
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.margins(_fit("ols_quad"))


def test_factor_rows_record_base_level():
    m = sp.margins(_fit("ols_quad"), _data())
    assert m.attrs["base_levels"] == {"g": 1}
    assert m.attrs["design_backend"] == "formula"
