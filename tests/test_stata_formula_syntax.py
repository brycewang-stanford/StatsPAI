"""Stata factor-variable and time-series terms inside formulas.

``sp.regress("y ~ i.g##c.x", df)`` is Stata's ``regress y i.g##c.x``; after
``sp.xtset(df, id=, time=)``, ``y ~ L.x + D.z`` is ``regress y L.x D.z``.
The expected coefficients were printed by Stata 18 MP (``%21.17g``,
2026-10-19) on the data built below (a panel of 30 units over 6 periods
with 25 rows removed, so lags meet gaps, in shuffled row order).
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core.utils import r_formula_idioms
from statspai.exceptions import MethodIncompatibility


@pytest.fixture()
def fv():
    r = np.random.default_rng(5)
    ids = np.repeat(np.arange(1, 31), 6)
    t = np.tile(np.arange(1, 7), 30)
    d = pd.DataFrame({"id": ids, "t": t})
    d = d.drop(index=r.choice(len(d), 25, replace=False)).reset_index(drop=True)
    n = len(d)
    d["g"] = r.integers(1, 4, n)
    d["h"] = r.integers(0, 2, n)
    d["x"] = np.round(r.normal(size=n), 6)
    d["z"] = np.round(r.normal(size=n), 6)
    d["y"] = np.round(
        1 + 0.5 * d.x - 0.3 * d.g + 0.2 * d.g * d.x + 0.4 * d.h + r.normal(size=n), 6
    )
    return d.sample(frac=1, random_state=2).reset_index(drop=True)


def _coefs(formula, data):
    return sp.regress(formula, data=data).params.to_numpy()


# reg y i.g##c.x: 2.g 3.g x 2.g#c.x 3.g#c.x _cons
STATA_G_X = [
    -0.20715324190344558, -0.6182999564002809, 0.70734241522733121,
    0.11323496816344254, 0.34936805726629017, 0.84866365330388738,
]  # fmt: skip


def test_full_factorial_with_a_slope(fv):
    got = sp.regress("y ~ i.g##c.x", data=fv).params
    # intercept, 2.g, 3.g, x, 2.g#x, 3.g#x in the formula's order
    order = [1, 2, 3, 4, 5, 0]
    np.testing.assert_allclose(got.to_numpy()[order], STATA_G_X, rtol=1e-9)


def test_interaction_alone_gives_a_slope_per_level(fv):
    # reg y i.g#c.x: 1.g#c.x 2.g#c.x 3.g#c.x _cons
    want = [0.70936174895298731, 0.81491815807439227, 1.0147559472859999]
    got = _coefs("y ~ i.g#c.x", fv)
    np.testing.assert_allclose(got[1:], want, rtol=1e-9)
    assert got[0] == pytest.approx(0.56159440273880201, rel=1e-9)


def test_continuous_square_and_factor(fv):
    # reg y c.x##c.x i.h: x, c.x#c.x, 1.h, _cons
    got = _coefs("y ~ c.x##c.x + i.h", fv)
    want = [0.30938625188349789, 0.84812038823268976, 0.015570735297156049]
    np.testing.assert_allclose(sorted(got[[0, 2, 3]]), sorted(want), rtol=1e-9)
    assert 0.49199088212872366 == pytest.approx(got[1], rel=1e-9)


def test_base_level(fv):
    # reg y ib2.g x: 1.g 3.g x _cons
    got = _coefs("y ~ ib2.g + x", fv)
    np.testing.assert_allclose(
        got,
        [0.64517150955205627, 0.20239611881235872, -0.39117667004569762,
         0.8668909708703505],
        rtol=1e-9,
    )  # fmt: skip


def test_two_factors_crossed(fv):
    # reg y i.g##i.h: 2.g 3.g 1.h 2.g#1.h 3.g#1.h _cons
    got = _coefs("y ~ i.g##i.h", fv)
    want = [
        0.60223352173913014, -0.47006752173913008, -0.25671589210950058,
        0.45058761619190441, 0.6852996060303177, -0.41942987915486735,
    ]  # fmt: skip
    np.testing.assert_allclose(got, want, rtol=1e-9)


def test_lags_read_the_declared_panel(fv):
    sp.xtset(fv, id="id", time="t")
    res = sp.regress("y ~ L.x + D.z + F.x", data=fv)
    # Stata: reg y L.x D.z F.x after xtset id t; N = 71
    np.testing.assert_allclose(
        res.params.to_numpy(),
        [0.42212280768631549, -0.0040977704093067731, -0.13057466849073343,
         -0.084620563217463168],
        rtol=1e-9,
    )  # fmt: skip
    assert list(res.params.index) == ["Intercept", "x_L1", "z_D1", "x_F1"]
    assert int(res.data_info["nobs"]) == 71
    assert "x_L1" not in fv.columns  # the caller's frame is left alone


def test_lag_without_a_declaration_names_the_fix(fv):
    with pytest.raises(MethodIncompatibility) as err:
        sp.regress("y ~ L.x", data=fv)
    assert "sp.xtset" in err.value.recovery_hint


def test_lag_name_collision_is_refused(fv):
    sp.xtset(fv, id="id", time="t")
    with pytest.raises(MethodIncompatibility, match="already have"):
        sp.regress("y ~ L.x", data=fv.assign(x_L1=0.0))


def test_parsers_without_data_refuse_lags_with_the_right_syntax():
    with pytest.raises(MethodIncompatibility, match="time-series operator") as err:
        r_formula_idioms("y ~ L.x")
    assert "sp.lag" in err.value.recovery_hint


@pytest.mark.parametrize(
    "formula",
    ["y ~ i.g*x", "y ~ x + g#h#"],
)
def test_unreadable_stata_terms_are_refused_with_the_spelling(formula):
    with pytest.raises(MethodIncompatibility) as err:
        r_formula_idioms(formula)
    assert "i.g##c.x" in err.value.recovery_hint


@pytest.mark.parametrize(
    "formula",
    [
        "y ~ x + np.log(z) + C(g)",
        "y ~ I(x * 1.5) + Q('c.x')",
        'y ~ Q("L.x") + z',
        "y ~ x:z - 1",
    ],
)
def test_ordinary_formulas_are_untouched(formula):
    from statspai.core._stata_formula import has_stata_ts_terms, stata_factor_syntax

    assert stata_factor_syntax(formula) == formula
    assert not has_stata_ts_terms(formula)
    r_formula_idioms(formula)  # no refusal


def test_lag_lists_and_grouped_terms(fv):
    sp.xtset(fv, id="id", time="t")
    one = sp.regress("y ~ L(1/2).x", data=fv).params
    assert list(one.index) == ["Intercept", "x_L1", "x_L2"]
    lagged = fv.assign(x_L1=sp.lag(fv, "x"), x_L2=sp.lag(fv, "x", 2)).dropna(
        subset=["x_L1", "x_L2"]
    )
    two = sp.regress("y ~ x_L1 + x_L2", data=lagged).params
    np.testing.assert_allclose(one.to_numpy(), two.to_numpy(), rtol=1e-12)
    grouped = sp.regress("y ~ c.(x z)#i.h", data=fv).params
    spelled = sp.regress("y ~ x:C(h) + z:C(h)", data=fv).params
    np.testing.assert_allclose(grouped.to_numpy(), spelled.to_numpy(), rtol=1e-12)


def test_other_likelihood_estimators_read_the_syntax(fv):
    sp.xtset(fv, id="id", time="t")
    fv = fv.assign(yb=(fv["y"] > 1).astype(int))
    a = sp.logit("yb ~ L.x + i.g", data=fv).params
    lagged = fv.assign(x_L1=sp.lag(fv, "x"))
    b = sp.logit("yb ~ x_L1 + C(g)", data=lagged).params
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=1e-10)
