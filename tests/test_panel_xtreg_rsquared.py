"""sp.panel reports the within / between / overall R-squared as Stata's
``xtreg`` does after fe, re and be.

Before, the labels carried linearmodels' 1 - SSR/SST quantities: on a
fixed-effects fit the "between" R-squared was 0.47 where ``xtreg, fe``
reports 0.87. Those values are kept in ``model_info['r2_linearmodels']``.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _unbalanced_panel():
    rng = np.random.default_rng(7)
    rows = []
    for i in range(60):
        a = rng.normal()
        for t in range(rng.integers(3, 9)):
            x1 = 0.6 * a + rng.normal()
            x2 = rng.normal() + 0.3 * t
            rows.append((i, t, x1, x2, 1 + 0.8 * x1 - 0.5 * x2 + a + rng.normal()))
    return pd.DataFrame(rows, columns=["id", "t", "x1", "x2", "y"])


# Stata 18 MP on the same 352 rows (written with DataFrame.to_csv, read
# with `import delimited ..., asdouble`):
#   xtset id t
#   foreach m in fe re be {
#       quietly xtreg y x1 x2, `m'
#       display %18.15f e(r2_w) " " %18.15f e(r2_b) " " %18.15f e(r2_o)
#   }
STATA_XTREG = {
    "fe": (0.482601120118227, 0.872974017003197, 0.603614724318982),
    "re": (0.465009246356360, 0.902144603636874, 0.633635896928227),
    "be": (0.427310986883804, 0.908045115183176, 0.630991252650276),
}


@pytest.mark.parametrize("method", ["fe", "re", "be"])
def test_matches_stata_xtreg(method):
    r = sp.panel(
        _unbalanced_panel(), "y ~ x1 + x2", entity="id", time="t", method=method
    )
    got = tuple(
        r.diagnostics[f"R-squared ({k})"] for k in ("within", "between", "overall")
    )
    # Stata prints 15 decimals; the rest is display rounding.
    np.testing.assert_allclose(got, STATA_XTREG[method], rtol=0, atol=1e-14)
    assert r.model_info["r2_convention"] == "xtreg"


def test_linearmodels_values_are_kept():
    df = _unbalanced_panel()
    r = sp.panel(df, "y ~ x1 + x2", entity="id", time="t", method="fe")
    from linearmodels.panel import PanelOLS

    lm = PanelOLS(
        df.set_index(["id", "t"])["y"],
        df.set_index(["id", "t"])[["x1", "x2"]],
        entity_effects=True,
    ).fit()
    assert r.model_info["r2_linearmodels"]["between"] == pytest.approx(
        lm.rsquared_between, abs=1e-14
    )
    # The headline R-squared of a within fit is unchanged.
    assert r.diagnostics["R-squared"] == pytest.approx(lm.rsquared, abs=1e-14)


def test_one_regressor_gives_the_same_r2_after_fe_re_be():
    # Squared correlations do not depend on the scale of a single slope,
    # so xtreg prints identical R-squareds after fe, re and be.
    df = _unbalanced_panel()
    vals = [
        tuple(
            sp.panel(df, "y ~ x1", entity="id", time="t", method=m).diagnostics[
                f"R-squared ({k})"
            ]
            for k in ("within", "between", "overall")
        )
        for m in ("fe", "re", "be")
    ]
    np.testing.assert_allclose(vals[0], vals[1], rtol=1e-13)
    np.testing.assert_allclose(vals[0], vals[2], rtol=1e-13)


def test_other_fits_keep_the_linearmodels_convention():
    df = _unbalanced_panel()
    df["w"] = 1.0 + (df["id"] % 3)
    for kw in (
        {"method": "twoway"},
        {"method": "pooled"},
        {"method": "fe", "weights": "w"},
    ):
        r = sp.panel(df, "y ~ x1 + x2", entity="id", time="t", **kw)
        assert r.model_info["r2_convention"] == "linearmodels", kw
        assert "R-squared (overall)" not in r.diagnostics
