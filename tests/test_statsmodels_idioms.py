"""Spellings a user arriving from statsmodels / linearmodels writes first.

Found by working through a Python econometrics textbook whose every example
is written for those two libraries.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(42)
    n = 400
    z1, z2, w = rng.normal(size=(3, n))
    u = rng.normal(size=n)
    x = 0.8 * z1 + 0.5 * z2 + 0.4 * w + 0.6 * u + rng.normal(size=n)
    inc = np.exp(rng.normal(size=n))
    y = 1.0 + 0.7 * x - 0.5 * w + 0.3 * inc - 0.02 * inc**2 + u
    return pd.DataFrame({"y": y, "x": x, "w": w, "z1": z1, "z2": z2, "inc": inc})


def test_comma_separated_restrictions_are_a_joint_test(df):
    res = sp.regress("y ~ x + w + inc", df, robust="hc1")
    listed = sp.test(res, ["x = 0", "w = 0"])
    comma = sp.test(res, "x = 0, w = 0")
    assert comma["statistic"] == pytest.approx(listed["statistic"], rel=1e-12)
    assert comma["df"] == listed["df"] == (2, len(df) - 4)
    mixed = sp.test(res, "x = w, inc = 0.3")
    assert mixed["statistic"] == pytest.approx(
        sp.test(res, ["x = w", "inc = 0.3"])["statistic"], rel=1e-12
    )
    with pytest.raises(MethodIncompatibility):
        sp.test(res, "x = 0, , w = 0")
    # Stata's list of names keeps its meaning, commas or not
    assert sp.test(res, "x w")["statistic"] == pytest.approx(listed["statistic"])


def test_coefficient_names_ignore_blanks_inside_them(df):
    res = sp.regress("y ~ x + inc + I(inc**2) + np.log(inc)", df)
    assert "I(inc ** 2)" in res.params.index  # how the design names it
    canonical = sp.test(res, "I(inc ** 2) = 0")
    for spelled in ("I(inc**2) = 0", "I( inc ** 2 )=0", "I(inc **2) = 0"):
        assert sp.test(res, spelled)["statistic"] == pytest.approx(
            canonical["statistic"], rel=1e-12
        )
    joint = sp.test(res, "I(inc**2) = 0, np.log( inc ) = 0")
    assert joint["df"][0] == 2
    # a name that only contains another is still not that other name
    with pytest.raises(MethodIncompatibility, match="Unknown coefficient"):
        sp.test(res, "inc2 = 0")
    assert sp.test(res, "inc = 0")["df"][0] == 1


def test_linearmodels_bracket_block_is_the_endogenous_block(df):
    paren = sp.ivreg("y ~ w + (x ~ z1 + z2)", df, robust="hc0")
    for formula in (
        "y ~ 1 + w + [x ~ z1 + z2]",
        "y ~ w + [x ~ z1 + z2]",
        "y ~ [x ~ z1 + z2] + w",
    ):
        got = sp.ivreg(formula, df, robust="hc0")
        np.testing.assert_allclose(
            got.params[paren.params.index], paren.params, rtol=1e-12
        )
        np.testing.assert_allclose(
            got.std_errors[paren.params.index], paren.std_errors, rtol=1e-12
        )
    assert sp.iv("y ~ 1 + w + [x ~ z1]", df).params["x"] == pytest.approx(
        sp.iv("y ~ w + (x ~ z1)", df).params["x"], rel=1e-12
    )


def test_statsmodels_attribute_names_say_where_to_look(df):
    res = sp.regress("y ~ x + w", df)
    for name, pointer in [
        ("bse", "std_errors"),
        ("rsquared", ".r2"),
        ("resid", "residuals()"),
        ("f_test", ".test("),
        ("get_margeff", "sp.margins"),
    ]:
        with pytest.raises(AttributeError, match=name) as err:
            getattr(res, name)
        assert pointer in str(err.value)
    # the names stay absent: code that probes with hasattr reads as before
    assert not hasattr(res, "bse") and not hasattr(res, "estimate")
    with pytest.raises(AttributeError) as err:
        res.no_such_thing
    assert ";" not in str(err.value)
    import copy
    import pickle

    assert pickle.loads(pickle.dumps(res)).params.equals(res.params)
    assert copy.deepcopy(res).params.equals(res.params)
