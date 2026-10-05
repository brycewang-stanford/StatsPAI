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


def test_iv_reports_formula_terms_under_the_names_ols_gives_them(df):
    data = df.assign(g=np.arange(len(df)) % 3, xp=np.exp(df["x"] / 4))
    rhs = "w + inc + I(inc**2) + C(g)"
    ols = sp.regress(f"y ~ {rhs} + np.log(xp)", data)
    iv = sp.ivreg(f"y ~ {rhs} + [np.log(xp) ~ z1 + z2]", data)
    # the same regressors, the same labels: one row each in a joint table
    assert set(iv.params.index) == set(ols.params.index)
    assert list(iv.std_errors.index) == list(iv.params.index)
    assert list(iv.vcov().index) == list(iv.params.index)
    assert "First-stage F (np.log(xp))" in iv.diagnostics
    table = str(sp.regtable(ols, iv))
    assert table.count("np.log(xp)") == 1 and "np.log[xp]" not in table
    assert "I[inc" not in table and "g[1]" not in table
    # names written as in the formula, and Stata's factor-variable names
    assert sp.test(iv, "np.log(xp) = 0")["df"][0] == 1
    assert sp.test(iv, "I(inc**2) = 0, np.log(xp) = 0")["df"][0] == 2
    assert sp.test(iv, "i.g")["df"][0] == 2
    # the estimates are those of the fit with the columns built by hand
    built = data.assign(lx=np.log(data.xp), inc2=data.inc**2)
    dummies = pd.get_dummies(built.g, prefix="g", drop_first=True).astype(float)
    built = pd.concat([built, dummies], axis=1)
    ref = sp.ivreg("y ~ w + inc + inc2 + g_1 + g_2 + (lx ~ z1 + z2)", built)
    pairs = {"np.log(xp)": "lx", "I(inc ** 2)": "inc2", "C(g)[T.1]": "g_1"}
    for ours, theirs in pairs.items():
        assert iv.params[ours] == pytest.approx(ref.params[theirs], rel=1e-10)
        assert iv.std_errors[ours] == pytest.approx(ref.std_errors[theirs], rel=1e-9)


def test_iv_predictions_use_the_structural_equation(df):
    import patsy

    data = df.assign(xp=np.exp(df["x"] / 4))
    iv = sp.ivreg("y ~ w + I(inc**2) + (np.log(xp) ~ z1 + z2)", data)
    X = patsy.dmatrix("w + I(inc**2) + np.log(xp)", data, return_type="dataframe")
    manual = X[list(iv.params.index)].to_numpy() @ iv.params.to_numpy()
    np.testing.assert_allclose(iv.predict(data), manual, rtol=1e-12)
    np.testing.assert_allclose(iv.predict(), manual, rtol=1e-10)
    # d y / d xp = b / xp, averaged over the sample
    me = sp.margins(iv, data).set_index("variable")["dy/dx"]
    assert me["xp"] == pytest.approx(
        iv.params["np.log(xp)"] * (1 / data.xp).mean(), rel=1e-4
    )


def test_bracket_names_of_panel_fits_answer_to_the_formula_spelling():
    rng = np.random.default_rng(3)
    rows = []
    for i in range(50):
        a = rng.normal()
        for t in range(5):
            x = np.exp(rng.normal())
            rows.append(
                {"id": i, "t": t, "x": x, "y": a + 0.5 * np.log(x) + rng.normal()}
            )
    panel = pd.DataFrame(rows)
    fit = sp.panel(panel, "y ~ np.log(x) + I(x**2)", entity="id", time="t")
    stored = sp.test(fit, "np.log[x] = 0.5")["statistic"]
    assert sp.test(fit, "np.log(x) = 0.5")["statistic"] == pytest.approx(stored)
    assert sp.test(fit, "I(x**2) = 0")["df"][0] == 1


def test_a_name_from_another_library_points_at_ours():
    for name, target in [
        ("vecm", "sp.vec"),
        ("IV2SLS", "sp.ivreg"),
        ("adfuller", "sp.unitroot"),
        ("arch_model", "sp.garch"),
    ]:
        with pytest.raises(AttributeError, match=name) as err:
            getattr(sp, name)
        assert target in str(err.value)
        assert not hasattr(sp, name)
    with pytest.raises(AttributeError, match="did you mean sp.regress"):
        sp.regres
    for target in sp._ELSEWHERE_NAMES.values():
        attr = target.split("(")[0].split(" ")[0].replace("sp.", "")
        assert hasattr(sp, attr), target
