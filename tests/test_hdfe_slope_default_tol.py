"""``sp.hdfe_ols`` tightens its default tolerance when a slope is absorbed.

Varying-slope sweeps converge slowly. On the 1.5-million-row replication of
Zheng, Huang and Zhu (2026, Table 5B col. 3: four FE groups plus
``i.quarter#c.covid_exposure``) the default ``tol=1e-8`` left the clustered
SE 1.05e-5 (relative) from reghdfe's, ``1e-10`` 1.5e-7 and ``1e-12`` 3.8e-8,
while the coefficient agreed at every tolerance. reghdfe itself converges at
its own 1e-8, so the default now is ``1e-12`` whenever a slope is absorbed;
reghdfe's collinearity screen keeps its ``1e-8``-based threshold.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(4)
    n = 3000
    df = pd.DataFrame(
        {
            "firm": rng.integers(0, 150, n),
            "q": rng.integers(0, 8, n),
            "city": rng.integers(0, 10, n),
            "z": rng.normal(size=n),
            "x": rng.normal(size=n),
        }
    )
    df["y"] = 0.5 * df.x + 0.3 * df.z * df.q + rng.normal(size=n)
    return df


def test_default_tol_is_tight_only_with_a_slope(data):
    plain = sp.hdfe_ols("y ~ x | firm + city^q", data, cluster="firm")
    slope = sp.hdfe_ols("y ~ x | firm + city^q + i.q#c.z", data, cluster="firm")
    assert plain.absorber.tol == 1e-8
    assert slope.absorber.tol == 1e-12
    explicit = sp.hdfe_ols(
        "y ~ x | firm + city^q + i.q#c.z", data, cluster="firm", tol=1e-8
    )
    assert explicit.absorber.tol == 1e-8


def test_default_equals_explicit_tight_fit(data):
    f = "y ~ x | firm + city^q + i.q#c.z"
    a = sp.hdfe_ols(f, data, cluster="firm")
    b = sp.hdfe_ols(f, data, cluster="firm", tol=1e-12)
    assert float(a.coef["x"]) == float(b.coef["x"])
    assert float(a.se["x"]) == float(b.se["x"])


def test_collinearity_screen_unchanged_by_the_tighter_sweep(data):
    """Within-FE variation of relative size ~5e-12 lies between the screen's
    1e-9 (from reghdfe's tol 1e-8) and 1e-13 (from 1e-12): the default keeps
    reghdfe's rule and omits the column; an explicit tol=1e-12 is the user
    asking for the looser screen, as reghdfe's tol() would."""
    rng = np.random.default_rng(1)
    d = data.assign(w=data.firm.astype(float) + 1e-4 * rng.normal(size=len(data)))
    f = "y ~ x + w | firm + city^q + i.q#c.z"
    with pytest.warns(UserWarning, match="omitted"):
        r = sp.hdfe_ols(f, d, cluster="firm")
    assert np.isnan(float(r.coef["w"]))
    kept = sp.hdfe_ols(f, d, cluster="firm", tol=1e-12)
    assert np.isfinite(float(kept.coef["w"]))
