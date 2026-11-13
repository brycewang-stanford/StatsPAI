"""``sp.dose_response(method='dr')`` warns when one density weight dominates.

The pseudo-outcome divides by the conditional density of the dose. Its
variance is a linear regression of squared residuals on the covariates,
which can fall towards zero at extreme covariate values even when the
truth is homoskedastic; the density of that unit's dose then collapses
and its weight explodes. In the SE calibration screen
(``tests/reliability/se_screen/causal/dose_qte.py``, row
``dose_response.dr``) this made the reported standard error 0.6 of the
sampling SD of the 25th-to-75th percentile contrast. The estimator now
says when it happens.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import statspai as sp
from statspai.dose_response import dr
from statspai.exceptions import AssumptionWarning


def _draw(seed: int, n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    t = 0.5 * x1 + rng.normal(size=n)
    y = 1 + 0.7 * t + x1 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "t": t, "x1": x1, "x2": x2})


def _fit(df, seed):
    return sp.dose_response(
        df,
        "y",
        "t",
        ["x1", "x2"],
        method="dr",
        treatment_model=LinearRegression(),
        outcome_model=LinearRegression(),
        random_state=seed,
    )


def test_warns_when_a_density_weight_is_above_the_alarm():
    # On this draw the largest weight is about 213 (homoskedastic truth).
    with pytest.warns(AssumptionWarning, match="density weight"):
        r = _fit(_draw(1), 1)
    assert r.model_info["max_density_ratio"] > dr._DENSITY_RATIO_ALARM


def test_silent_when_the_weights_are_moderate():
    # On this draw the largest weight is about 15.
    with warnings.catch_warnings():
        warnings.simplefilter("error", AssumptionWarning)
        r = _fit(_draw(0), 0)
    assert r.model_info["max_density_ratio"] < dr._DENSITY_RATIO_ALARM
