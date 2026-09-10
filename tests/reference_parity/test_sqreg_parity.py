"""Cross-language parity: ``sp.sqreg`` vs R ``quantreg::rq``.

Regenerate the reference with::

    Rscript tests/reference_parity/_generate_sqreg_R.R

Two claims, at two very different tolerances, because they are two
different kinds of agreement:

* **Coefficients are bit-exact** (3.5e-14). Both sides minimise the same
  pinball loss with the same Barrodale-Roberts simplex, so there is no
  reason for them to differ and they do not.

  Until 1.27.0 this file could only assert four decimal places, because
  ``sp.sqreg`` rounded its returned coefficients to four decimals before
  handing them back — not for display, in the value itself. On this
  fixture that capped agreement at 1.2e-04 on ``x2`` and **7.8e-03** on
  ``x3``: a coefficient near zero loses every significant digit to a fixed
  number of decimal places. The old test recorded that as the estimator's
  accuracy, which it never was.

* **Standard errors differ by one scalar per quantile, and that is a
  documented sparsity-estimator convention, not a discrepancy.**
  ``sp.sqreg`` computes the Powell-type kernel sandwich; R's ``se="iid"``
  uses Koenker-Bassett with a Siddiqui/Hall-Sheather bandwidth; R's
  default ``se="nid"`` is the Hendricks-Koenker difference quotient, which
  is what Stata ``qreg`` reports. The test below asserts the strong form
  of "same sandwich, different f(0)": the ratio of our standard errors to
  R's ``iid`` ones is **constant across coefficients to 6e-16** within
  each quantile. A structural difference could not produce a constant
  ratio, so this pins far more than a loose numerical band would.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

_FIX = Path(__file__).parent / "_fixtures"
NAMES = ("(Intercept)", "x1", "x2", "x3")


@pytest.fixture(scope="module")
def ref():
    return json.loads((_FIX / "sqreg_R.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fitted():
    df = pd.read_csv(_FIX / "sqreg_data.csv")
    return sp.sqreg(df, "y", ["x1", "x2", "x3"], quantiles=[0.25, 0.5, 0.75])


def _sp_name(name):
    return "const" if name == "(Intercept)" else name


TAUS = [("tau_025", 0.25), ("tau_050", 0.5), ("tau_075", 0.75)]


@pytest.mark.parametrize("tau_str,tau_val", TAUS)
def test_coefficients_match_quantreg(fitted, ref, tau_str, tau_val):
    col = f"Q({tau_val})"
    got = {row["variable"]: float(row[col]) for _, row in fitted.iterrows()}
    for name in NAMES:
        assert got[_sp_name(name)] == pytest.approx(
            ref[tau_str]["coef"][name], rel=1e-10
        )


@pytest.mark.parametrize("tau_str,tau_val", TAUS)
def test_coefficients_are_returned_at_full_precision(fitted, ref, tau_str, tau_val):
    """The rounding guard.

    Four-decimal rounding would satisfy the assertion above only for the
    large coefficients, so this checks the small one explicitly: ``x3`` is
    ~1e-3 here, where four decimals leaves a single significant digit.
    """
    col = f"Q({tau_val})"
    got = {row["variable"]: float(row[col]) for _, row in fitted.iterrows()}
    x3 = got["x3"]
    assert x3 != pytest.approx(round(x3, 4), rel=1e-12, abs=0.0) or abs(x3) > 1.0
    assert x3 == pytest.approx(ref[tau_str]["coef"]["x3"], rel=1e-10)


@pytest.mark.parametrize("tau_str,tau_val", TAUS)
def test_standard_errors_are_one_scalar_from_quantregs_iid_sandwich(
    fitted, ref, tau_str, tau_val
):
    """Same sandwich, different sparsity estimate.

    Asserting a constant ratio is a much sharper statement than asserting
    the standard errors are "close": any difference in the meat or bread
    of the sandwich would vary across coefficients.
    """
    col = f"SE({tau_val})"
    got = {row["variable"]: float(row[col]) for _, row in fitted.iterrows()}
    ratios = np.array([got[_sp_name(n)] / ref[tau_str]["se_iid"][n] for n in NAMES])
    spread = (ratios.max() - ratios.min()) / ratios.mean()
    assert spread < 1e-13, f"ratio is not constant across coefficients: {ratios}"
    # And the scalar itself is a sparsity estimate, so it is near 1 rather
    # than arbitrary -- a factor of 2 would mean a different estimand.
    assert 0.8 < ratios.mean() < 1.25


@pytest.mark.parametrize("tau_str,tau_val", TAUS)
def test_standard_errors_are_not_the_nid_convention(fitted, ref, tau_str, tau_val):
    """Records which convention we do *not* implement.

    Stata's ``qreg`` and R's ``summary.rq`` default both report ``nid``.
    Users migrating from either will see different standard errors, and
    that is a documented difference rather than a defect -- but if
    ``sp.sqreg`` ever silently switches conventions, this fails.
    """
    col = f"SE({tau_val})"
    got = {row["variable"]: float(row[col]) for _, row in fitted.iterrows()}
    nid_ratios = np.array([got[_sp_name(n)] / ref[tau_str]["se_nid"][n] for n in NAMES])
    assert (nid_ratios.max() - nid_ratios.min()) / nid_ratios.mean() > 1e-6


def test_quantile_ordering_reflects_the_planted_dgp(fitted):
    """Kept from the original analytical test: x2's effect declines in tau."""
    row = fitted[fitted["variable"] == "x2"].iloc[0]
    assert row["Q(0.25)"] > row["Q(0.5)"] > row["Q(0.75)"]
