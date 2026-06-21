"""Tests for ``sp.kitagawa_test`` — Kitagawa (2015) test of instrument
validity / the LATE testable implication (previously untested public
function).

The robust, low-noise property exercised here is the test *statistic*:
a valid (randomly assigned, excluded) instrument yields a near-zero
violation statistic, while a gross exclusion violation (Z entering the
outcome directly) drives the statistic up by two orders of magnitude.

References
----------
Kitagawa, T. (2015). "A Test for Instrument Validity." Econometrica 83(5),
2043-2063.
"""

import numpy as np
import pandas as pd

import statspai as sp


def _iv_data(violate=False, seed=0, n=2500):
    rng = np.random.default_rng(seed)
    Z = rng.integers(0, 2, n)
    U = rng.normal(size=n)
    # Monotone first stage in Z.
    D = ((0.3 + 0.4 * Z + 0.3 * U) > rng.random(n)).astype(int)
    direct = 4.0 * Z if violate else 0.0  # exclusion-restriction violation
    Y = 1.0 + 0.5 * D + direct + U + rng.normal(0, 0.5, n)
    return pd.DataFrame({"y": Y, "d": D, "z": Z})


def test_kitagawa_result_contract():
    res = sp.kitagawa_test(
        _iv_data(), y="y", treatment="d", instrument="z", n_boot=200, seed=0
    )
    assert 0.0 <= res.p_value <= 1.0
    assert res.statistic >= 0.0
    assert res.n_obs == 2500
    assert hasattr(res, "first_stage")


def test_kitagawa_valid_instrument_not_rejected():
    res = sp.kitagawa_test(
        _iv_data(violate=False),
        y="y",
        treatment="d",
        instrument="z",
        n_boot=300,
        seed=0,
    )
    # A genuinely valid instrument should not be rejected at 5%.
    assert res.p_value > 0.05
    assert res.statistic < 0.05


def test_kitagawa_statistic_rises_under_exclusion_violation():
    valid = sp.kitagawa_test(
        _iv_data(violate=False),
        y="y",
        treatment="d",
        instrument="z",
        n_boot=200,
        seed=0,
    )
    invalid = sp.kitagawa_test(
        _iv_data(violate=True),
        y="y",
        treatment="d",
        instrument="z",
        n_boot=200,
        seed=0,
    )
    # The measured density violation must be far larger when Z enters Y.
    assert invalid.statistic > 10 * max(valid.statistic, 1e-4)


def test_kitagawa_seed_reproducible():
    data = _iv_data()
    a = sp.kitagawa_test(data, y="y", treatment="d", instrument="z", n_boot=200, seed=7)
    b = sp.kitagawa_test(data, y="y", treatment="d", instrument="z", n_boot=200, seed=7)
    assert a.statistic == b.statistic
    assert a.p_value == b.p_value
