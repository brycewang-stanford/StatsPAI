"""The rest of Stata's ``rddensity`` output: the conventional statistic and
the binomial tests.

``rddensity x, all`` on ``sp.datasets.lee_2008_senate()`` (Stata 18,
``rddensity`` with its defaults: unrestricted model, ``comb`` bandwidths,
triangular kernel, jackknife variance)::

            Method |      T          P>|T|
      Conventional |   -1.6506      0.0988
            Robust |   -0.8753      0.3814

followed by a table of exact binomial tests in ten symmetric windows.
``sp.rddensity`` returned the robust statistic only. The robust one is the
test; the conventional one is not bias corrected and is reported because
Stata prints it.
"""

import numpy as np
import pytest

import statspai as sp

RTOL = 1e-9

# e(f_pl) e(f_pr) e(se_pl) e(se_pr) e(se_p) e(T_p) e(pv_p)
CONVENTIONAL = {
    "density_left": 0.022189160419733,
    "density_right": 0.0180376324316414,
    "se_left": 0.0020165555127367,
    "se_right": 0.001503192714492,
    "se": 0.0025151708635498,
    "statistic": -1.650594815746339,
    "pvalue": 0.09882133818025,
}
# e(se_ql) e(se_qr) e(T_q) e(pv_q)
ROBUST = (
    0.0032884781492836,
    0.0023705365869852,
    -0.8752729886032767,
    0.3814253876174171,
)

# half-width (3 decimals), < c, >= c, p-value (4 decimals), as printed
BINOMIAL = [
    (0.430, 8, 12, 0.5034),
    (0.861, 17, 25, 0.2800),
    (1.291, 25, 34, 0.2976),
    (1.722, 45, 47, 0.9170),
    (2.152, 51, 55, 0.7709),
    (2.583, 66, 65, 1.0000),
    (3.013, 79, 71, 0.5678),
    (3.444, 94, 86, 0.6020),
    (3.874, 105, 94, 0.4785),
    (4.305, 115, 107, 0.6386),
]


@pytest.fixture(scope="module")
def fit():
    return sp.rddensity(sp.datasets.lee_2008_senate(), "x")


def test_conventional_statistic_matches_stata(fit):
    got = fit.model_info["conventional"]
    for key, ref in CONVENTIONAL.items():
        assert got[key] == pytest.approx(ref, rel=RTOL), key


def test_robust_statistic_is_still_the_test(fit):
    se_l, se_r, t, p = ROBUST
    assert fit.model_info["se_left"] == pytest.approx(se_l, rel=RTOL)
    assert fit.model_info["se_right"] == pytest.approx(se_r, rel=RTOL)
    assert fit.estimate == pytest.approx(t, rel=RTOL)
    assert fit.pvalue == pytest.approx(p, rel=RTOL)
    # without the bias correction the same data look more suspicious
    assert fit.model_info["conventional"]["pvalue"] < fit.pvalue


def test_binomial_table_matches_stata(fit):
    table = fit.model_info["binomial_tests"]
    assert len(table) == len(BINOMIAL)
    for row, (half, n_left, n_right, pvalue) in zip(table.itertuples(), BINOMIAL):
        assert row.half_width == pytest.approx(half, abs=5e-4)
        assert (row.n_left, row.n_right) == (n_left, n_right)
        assert row.pvalue == pytest.approx(pvalue, abs=5e-5)
    # windows are multiples of the first, which holds twenty observations
    first = table["half_width"].iloc[0]
    np.testing.assert_allclose(table["half_width"], first * np.arange(1, 11))
    assert table.loc[0, ["n_left", "n_right"]].sum() == 20


def test_cutoff_other_than_zero(fit):
    lee = sp.datasets.lee_2008_senate()
    shifted = sp.rddensity(lee.assign(x=lee["x"] + 7.5), "x", c=7.5)
    assert shifted.model_info["conventional"]["statistic"] == pytest.approx(
        fit.model_info["conventional"]["statistic"], rel=1e-8
    )
    assert shifted.model_info["binomial_tests"]["n_left"].tolist() == (
        fit.model_info["binomial_tests"]["n_left"].tolist()
    )


def test_few_observations_give_an_empty_table():
    import sys

    module = sys.modules["statspai.diagnostics.rddensity"]
    table = module._binomial_tests(np.linspace(-1, 1, 15))
    assert len(table) == 0
    assert list(table.columns) == ["half_width", "n_left", "n_right", "pvalue"]
