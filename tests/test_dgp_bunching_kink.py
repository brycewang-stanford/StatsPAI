"""``sp.dgp_bunching`` produces bunching at the kink.

It used to shrink earnings above the kink by ``(1 - elasticity)`` without
moving anyone onto the kink, so the density had no excess mass there and
``sp.bunching`` found nothing (estimate ~0.02, p ~0.9) on the package's own
demonstration data. It now follows the iso-elastic kink model: counterfactual
earnings between the kink and ``kink * ((1 - t0) / (1 - t1))^e`` locate at the
kink, higher earnings scale by ``((1 - t1) / (1 - t0))^e``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def df():
    return sp.dgp_bunching(n=6000, kink_point=50000.0, elasticity=0.3, seed=3)


def test_earnings_follow_the_kink_model(df):
    k, z_star, z = 50000.0, df["counterfactual_income"], df["income"]
    response = (0.8 / 1.0) ** 0.3
    upper = k / response
    assert df.attrs["bunching_upper"] == pytest.approx(upper)
    below, above = z_star <= k, z_star >= upper
    between = ~below & ~above
    np.testing.assert_array_equal(z[below], z_star[below])
    np.testing.assert_allclose(z[above], z_star[above] * response)
    assert (z[between] == k).all()
    assert int(between.sum()) == df.attrs["n_bunchers"] > 0


def test_bunching_estimator_detects_the_excess_mass(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.bunching(
            df,
            running_var="income",
            threshold=50000.0,
            dt=df.attrs["dt"],
            n_bootstrap=50,
            random_state=0,
        )
    assert res.estimate > 1
    assert res.pvalue < 0.01


def test_invalid_tax_schedule_raises():
    with pytest.raises(ValueError, match="t0 < t1"):
        sp.dgp_bunching(n=100, t0=0.3, t1=0.2)
