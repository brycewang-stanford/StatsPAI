"""Count models drop rows with a missing outcome or regressor, as Stata does.

Until 2026-10 the plain-column path of ``sp.poisson`` / ``sp.nbreg`` /
``sp.ppmlhdfe`` passed missing values into the IRLS, which stopped with
``LinAlgError: SVD did not converge``; formulas with ``C()`` / ``I()`` /
interactions went through patsy and already dropped them. Found by
fitting every regression-family estimator on a frame with holes and on
the same frame cleaned beforehand, and comparing.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient

HOLES_X = [3, 50, 51, 200, 333]
HOLES_Y = [7, 8]


@pytest.fixture(scope="module")
def frames():
    rng = np.random.default_rng(11)
    n = 400
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    df = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "g": np.arange(n) % 40,
            "k": np.arange(n) % 3,
            "w": rng.uniform(0.5, 2.0, size=n),
            "cnt": rng.poisson(np.exp(0.3 * x1 - 0.2 * x2 + 0.4)).astype(float),
        }
    )
    holed = df.copy()
    holed.loc[HOLES_X, "x1"] = np.nan
    holed.loc[HOLES_Y, "cnt"] = np.nan
    clean = holed.dropna(subset=["x1", "cnt"]).reset_index(drop=True)
    return holed, clean


def _fit(fn, df, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(df, **kw)


CASES = {
    "poisson": lambda df, **kw: sp.poisson("cnt ~ x1 + x2", df, **kw),
    "poisson_xy": lambda df, **kw: sp.poisson(data=df, y="cnt", x=["x1", "x2"], **kw),
    "nbreg": lambda df, **kw: sp.nbreg("cnt ~ x1 + x2", df, **kw),
    "ppmlhdfe": lambda df, **kw: sp.ppmlhdfe("cnt ~ x1 + x2 | k", df, **kw),
}
OPTIONS = {
    "default": {},
    "cluster": dict(cluster="g"),
    "weights_cluster": dict(weights="w", cluster="g"),
    "exposure": dict(exposure="w"),
}


@pytest.mark.parametrize("model", sorted(CASES))
@pytest.mark.parametrize("option", ["default", "cluster"])
def test_holes_equal_the_cleaned_frame(frames, model, option):
    holed, clean = frames
    a = _fit(CASES[model], holed, **OPTIONS[option])
    b = _fit(CASES[model], clean, **OPTIONS[option])
    assert len(clean) == 400 - len(HOLES_X) - len(HOLES_Y)
    np.testing.assert_allclose(a.params, b.params, rtol=1e-12)
    np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=1e-12)


@pytest.mark.parametrize("option", ["weights_cluster", "exposure"])
def test_weights_exposure_and_clusters_follow_the_kept_rows(frames, option):
    """The side columns are read on the rows that survive, not positionally."""
    holed, clean = frames
    a = _fit(CASES["poisson"], holed, **OPTIONS[option])
    b = _fit(CASES["poisson"], clean, **OPTIONS[option])
    np.testing.assert_allclose(a.params, b.params, rtol=1e-12)
    np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=1e-12)
    assert int(a.data_info["nobs"]) == len(clean)


def test_nothing_observed_is_an_error_not_an_empty_fit(frames):
    holed, _ = frames
    empty = holed.assign(x1=np.nan)
    with pytest.raises(DataInsufficient, match="no row"):
        sp.poisson("cnt ~ x1 + x2", empty)
