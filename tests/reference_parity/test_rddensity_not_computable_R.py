"""``sp.rddensity`` when the density test cannot be computed: NaN, as R.

R ``rddensity`` 2.6 (GPL-2, black box) on the design below, with
``h = c(0.5, 0.5)``, reports ``hat = (NA, NA, NA)``, ``t_jk = NA`` and
``p_jk = NA``: one side's window holds a single observation, too few for
the order-``p + 1`` local polynomial. Run::

    X <- c(seq(-5, -1, length.out = 20), -0.3, -0.1, 0.2,
           seq(1, 5, length.out = 20))
    r <- rddensity(X, c = 0, h = c(0.5, 0.5))
    r$test$t_jk; r$test$p_jk          # NA NA

StatsPAI reported ``T = 0, p = 1`` here, i.e. "no evidence of
manipulation", a conclusion the data cannot support. It now reports NaN,
warns, and records why in ``model_info['diagnostics']``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import AssumptionWarning

X = np.r_[np.linspace(-5, -1, 20), -0.3, -0.1, 0.2, np.linspace(1, 5, 20)]


def test_not_computable_statistic_is_nan_like_r():
    df = pd.DataFrame({"x": X})
    with pytest.warns(AssumptionWarning, match="not computable"):
        res = sp.rddensity(df, x="x", c=0, h=0.5)
    assert np.isnan(res.estimate) and np.isnan(res.pvalue)
    assert np.isnan(res.model_info["density_left"])
    assert np.isnan(res.model_info["density_right"])
    info = res.model_info["diagnostics"]["not_computable"]
    assert (info["n_eff_left"], info["n_eff_right"]) == (2, 1)
    assert info["polynomial_order_fitted"] == 3


def test_computable_windows_are_unchanged():
    """Boundary: one more point per side inside h=1.5 gives a finite test
    and no ``not_computable`` diagnostic."""
    df = pd.DataFrame({"x": np.r_[X, -0.7, -0.9, -1.2, 0.6, 0.9, 1.2]})
    res = sp.rddensity(df, x="x", c=0, h=1.5)
    assert np.isfinite(res.estimate) and 0.0 <= res.pvalue <= 1.0
    assert "diagnostics" not in res.model_info
