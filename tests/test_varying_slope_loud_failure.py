"""Varying-slope fixed effects must fail loudly, never silently.

pyfixest (through 0.50.1) accepts fixest's ``fe[slope]`` syntax but does not
absorb the slope. The fit comes back bit-identical to one that never mentioned
the slope at all, with no warning — so a user asking for
``y ~ d | county + pref[year]`` gets the ``y ~ d | county`` answer and no
indication anything was dropped. On the fixture below that is 0.337 against a
Stata ``reghdfe`` truth of 0.067.

These tests pin the *refusal*, not a number: whatever the backend does later,
StatsPAI must not hand back a quietly-wrong coefficient.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

pytest.importorskip("pyfixest")


@pytest.fixture(scope="module")
def slope_panel() -> pd.DataFrame:
    """30 counties x 20 years, nested in 6 prefectures / 3 provinces."""
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame(
        {
            "county": np.repeat(np.arange(30), 20),
            "year": np.tile(np.arange(20), 30),
        }
    )
    df["prov"] = df["county"] // 10
    df["pref"] = df["county"] // 5
    df["x"] = rng.normal(size=n)
    df["d"] = (df["county"] < 15).astype(int) * (df["year"] >= 10).astype(int)
    df["y"] = 1 + 0.3 * df["d"] + 0.5 * df["x"] + rng.normal(size=n)
    return df


# --------------------------------------------------------------------------- #
#  The refusal
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "fml",
    [
        "y ~ d | county + pref[year]",
        "y ~ d | county + pref[[year]]",
        "y ~ d | county + i.pref#c.year",
        "y ~ d | pref[year]",
    ],
)
def test_feols_refuses_varying_slope_fe(slope_panel, fml):
    with pytest.raises(MethodIncompatibility, match="varying slope"):
        sp.feols(fml, data=slope_panel)


def test_error_names_both_working_alternatives(slope_panel):
    """An agent must be able to recover from the message without guessing."""
    with pytest.raises(MethodIncompatibility) as exc:
        sp.feols("y ~ d | county + pref[year]", data=slope_panel)
    msg = str(exc.value)
    assert "i(pref, year)" in msg
    assert "hdfe_ols" in msg
    assert "i.pref#c.year" in msg


@pytest.mark.parametrize("fn_name", ["fepois", "feglm"])
def test_glm_paths_refuse_too(slope_panel, fn_name):
    df = slope_panel.assign(
        y=(slope_panel["y"] > slope_panel["y"].median()).astype(int)
    )
    fn = getattr(sp, fn_name)
    with pytest.raises(MethodIncompatibility, match="varying slope"):
        fn("y ~ d | county + pref[year]", data=df)


# --------------------------------------------------------------------------- #
#  Guard must not over-trigger on syntax that *is* supported
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "fml",
    [
        "y ~ d + x | county + year",
        "y ~ d + x:year | county",
        "y ~ d + x*year | county",
        "y ~ d | county + prov^year",
        "y ~ d + i(pref, year) | county",
        "y ~ d + x",
    ],
)
def test_supported_syntax_still_runs(slope_panel, fml):
    res = sp.feols(fml, data=slope_panel)
    assert "d" in res.params.index


def test_rhs_slope_matches_stata_reghdfe(slope_panel):
    """`i(pref, year)` on the RHS is the correct escape hatch.

    Stata: reghdfe y d, absorb(county i.pref#c.year) -> d = 0.066701865575718
    Agreement is ~1e-8 (alternating-projection tolerance), not bit-exact.
    """
    res = sp.feols("y ~ d + i(pref, year) | county", data=slope_panel)
    assert float(res.params["d"]) == pytest.approx(0.066701865575718, abs=1e-7)
