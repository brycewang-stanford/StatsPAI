"""Arithmetic inside a formula must not wrap around on small integers.

``pd.read_stata`` returns a Stata ``byte`` / ``int`` as ``int8`` / ``int16``.
numpy arithmetic stays in that type, so ``I(x**2)`` overflowed silently for
values above 11 (``int8``) or 181 (``int16``) and the regression was fitted
on garbage. Stata and R compute in double precision. Found on the
restaurant-inspection regressions of Huntington-Klein, *The Effect*, ch. 13
(``c.numberoflocations##c.numberoflocations``, an ``int`` up to 646).
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def frames():
    rng = np.random.default_rng(20261005)
    n = 600
    x = rng.integers(1, 600, n)
    z = rng.normal(size=n)
    y = 1e-5 * x.astype(float) ** 2 + 0.002 * x + z + rng.normal(size=n)
    base = pd.DataFrame(
        {
            "y": y,
            "x": x,
            "z": z,
            "w": z + rng.normal(size=n),
            "g": np.repeat(np.arange(30), 20),
            "t": np.tile(np.arange(20), 30),
        }
    )
    base["count"] = rng.poisson(np.exp(0.2 + 1e-6 * x**2))
    narrow = base.assign(x=base["x"].astype("int16"), g=base["g"].astype("int8"))
    wide = base.assign(x=base["x"].astype("float64"))
    return narrow, wide


CALLS = {
    "regress": lambda d: sp.regress("y ~ x + I(x**2) + z", data=d),
    "feols": lambda d: sp.feols("y ~ x + I(x**2) + z", data=d),
    "feols_fe": lambda d: sp.feols("y ~ x + I(x**2) + z | g", data=d),
    "fepois": lambda d: sp.fepois("count ~ x + I(x**2) | g", data=d),
    "poisson": lambda d: sp.poisson("count ~ x + I(x**2)", data=d),
    "glm": lambda d: sp.glm("count ~ x + I(x**2)", data=d, family="poisson"),
    "ivreg": lambda d: sp.ivreg("y ~ x + I(x**2) + (z ~ w)", data=d),
    "panel": lambda d: sp.panel(d, "y ~ x + I(x**2) + z", entity="g", time="t"),
}


@pytest.mark.parametrize("name", sorted(CALLS))
def test_square_of_an_int16_column_is_the_square(frames, name):
    narrow, wide = frames
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = CALLS[name](narrow)
        b = CALLS[name](wide)
    # same numbers, so the coefficients agree to rounding; before the fix
    # they differed in the first digit
    np.testing.assert_allclose(
        np.asarray(a.params, dtype=float),
        np.asarray(b.params, dtype=float),
        rtol=1e-7,
        atol=1e-10,
    )


def test_the_square_is_recovered(frames):
    narrow, _ = frames
    fit = sp.regress("y ~ x + I(x**2) + z", data=narrow)
    np.testing.assert_allclose(fit.params["I(x ** 2)"], 1e-5, atol=3e-6)


def test_factor_levels_keep_their_names(frames):
    """Widening to int64 (not to float) leaves C(g) levels as 'T.1'."""
    narrow, _ = frames
    fit = sp.regress("y ~ z + C(g)", data=narrow)
    assert "C(g)[T.1]" in fit.params.index


def test_the_caller_frame_is_not_modified(frames):
    narrow, _ = frames
    before = narrow.dtypes.copy()
    sp.regress("y ~ x + I(x**2)", data=narrow)
    assert (narrow.dtypes == before).all()
