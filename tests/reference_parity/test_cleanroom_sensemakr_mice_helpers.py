"""Clean-room helpers against black-box R outputs.

``_robustness_value`` (Cinelli and Hazlett 2020, eqs. 18-20) and
``_barnard_rubin_df`` (Barnard and Rubin 1999) were re-implemented from the
published formulas (docs/dev/cleanroom/sensemakr_mice_helpers.md). The
fixtures hold R sensemakr ``robustness_value`` and R mice pooling degrees of
freedom on grids chosen by us, printed to 15 significant digits.

Tolerances: both quantities are closed form, so the only error left is the
15-digit printing of the reference (rtol 1e-10 for RV, which also goes
through a t quantile; rtol 1e-12 for the df; atol 1e-12 for RV rows that are
exactly zero in the reference).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from statspai.diagnostics.sensemakr import _robustness_value
from statspai.imputation.mice import _barnard_rubin_df

_FIX = Path(__file__).parent / "_fixtures"
_RV = pd.read_csv(_FIX / "sensemakr_rv_blackbox_R.csv")
_BR = pd.read_csv(_FIX / "mice_barnard_rubin_blackbox_R.csv")


def _as_bool(v) -> bool:
    if isinstance(v, str):
        return v.strip().upper() == "TRUE"
    return bool(v)


@pytest.mark.parametrize("row", list(_RV.itertuples(index=False)), ids=str)
def test_robustness_value_matches_r_blackbox(row):
    got = _robustness_value(
        float(row.t),
        int(row.dof),
        q=float(row.q),
        alpha=float(row.alpha),
        invert=_as_bool(row.invert),
    )
    assert isinstance(got, float)
    if row.rv == 0:
        assert abs(got) <= 1e-12
    else:
        np.testing.assert_allclose(got, row.rv, rtol=1e-10, atol=0)


def test_robustness_value_fixture_covers_both_branches():
    inv = _RV["invert"].map(_as_bool)
    assert inv.any() and (~inv).any()
    assert (_RV["alpha"] < 1).any() and (_RV["alpha"] == 1).any()
    assert (_RV["rv"] == 0).any() and (_RV["rv"] > 0).any()


@pytest.mark.parametrize("row", list(_BR.itertuples(index=False)), ids=str)
def test_barnard_rubin_df_matches_r_blackbox(row):
    got = _barnard_rubin_df(
        int(row.m), np.array([row.b]), np.array([row.t]), float(row.dfcom)
    )
    np.testing.assert_allclose(np.asarray(got)[0], row.df, rtol=1e-12, atol=0)


def test_barnard_rubin_df_vectorised_equals_rowwise():
    for (m, dfcom), g in _BR.groupby(["m", "dfcom"]):
        got = _barnard_rubin_df(
            int(m), g["b"].to_numpy(), g["t"].to_numpy(), float(dfcom)
        )
        np.testing.assert_allclose(got, g["df"].to_numpy(), rtol=1e-12, atol=0)


def test_barnard_rubin_df_limits():
    # b = 0: nu_old = inf, so nu = nu_obs = (dfcom+1)/(dfcom+3) * dfcom
    got = _barnard_rubin_df(5, np.array([0.0]), np.array([1.0]), 10.0)
    np.testing.assert_allclose(got, [11.0 / 13.0 * 10.0], rtol=1e-15)
    # dfcom = inf: nu = nu_old = (m-1)/lambda^2, and inf when b = 0
    got = _barnard_rubin_df(5, np.array([0.0, 0.5]), np.array([1.0, 1.0]), np.inf)
    assert np.isinf(got[0])
    np.testing.assert_allclose(got[1], 4.0 / (1.2 * 0.5) ** 2, rtol=1e-15)
