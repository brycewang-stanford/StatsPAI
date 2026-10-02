"""Bandwidth selection when the interquartile range sets the pilot bandwidth.

``rdbwselect`` starts from ``C * min(sd, IQR / 1.349) * n^(-1/5)``, with the
quartiles taken by Hyndman and Fan's definition 2 (R ``quantile(type = 2)``).
StatsPAI used numpy's ``method="lower"``, which is a different order statistic
at the upper quartile whenever ``floor(0.75 (n - 1)) != ceil(0.75 n) - 1``.
Nothing showed while the standard deviation was the smaller of the two; on
the Lee (2008) House data the IQR binds and every data-driven bandwidth was
off by 5.6e-5 (``h = 0.1356186677`` against ``0.1356263072`` in both
references), which moved the estimates by 2e-5 to 2e-4.

The data below are built without a random number generator so that R, Stata
and Python read the same numbers: ``x = t^3`` on an even grid has
``sd = 0.378`` and ``IQR / 1.349 = 0.185``, and ``n = 1002`` is a sample size
at which the two quantile rules differ.

Reference values: R ``rdrobust`` 4.0.0 (R 4.5.2) and Stata ``rdrobust``
10.0.0 (Stata 18), which agree with each other to all 13 printed digits.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

# rtol: the selectors are deterministic and the references print 13 digits;
# 1e-9 leaves room for BLAS differences in three nested local regressions.
RTOL = 1e-9

# method: (h_left, h_right, b_left, b_right)
REFERENCE = {
    "mserd": (0.3703423375657, 0.3703423375657, 0.5153728640684, 0.5153728640684),
    "msetwo": (0.2756485169525, 0.2762957410317, 0.5154362324242, 0.5153813944607),
    "msesum": (0.2461775418889, 0.2461775418889, 0.5154432019530, 0.5154432019530),
    "cerrd": (0.2621561059205, 0.2621561059205, 0.5153728640684, 0.5153728640684),
    "certwo": (0.1951247115899, 0.1955828653764, 0.5154362324242, 0.5153813944607),
    "cersum": (0.1742629432294, 0.1742629432294, 0.5154432019530, 0.5154432019530),
}


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    n = 1002
    i = np.arange(1, n + 1)
    t = -1 + 2 * (i - 0.5) / n
    x = t**3
    y = (
        0.5
        + 0.8 * (x >= 0)
        + 1.5 * x
        - 2.0 * x**2
        + 0.3 * np.sin(12.9898 * i) * np.cos(78.233 * i)
    )
    return pd.DataFrame({"x": x, "y": y})


def test_the_iqr_is_what_binds(data):
    x = data["x"].to_numpy()
    iqr2 = np.quantile(x, 0.75, method="averaged_inverted_cdf") - np.quantile(
        x, 0.25, method="averaged_inverted_cdf"
    )
    iqr_lower = np.quantile(x, 0.75, method="lower") - np.quantile(
        x, 0.25, method="lower"
    )
    assert iqr2 / 1.349 < np.std(x, ddof=1)
    # The design only tests the fix if the two rules disagree here.
    assert iqr2 != iqr_lower


@pytest.mark.parametrize("method", sorted(REFERENCE))
def test_bandwidths_match_r_and_stata(data, method):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = sp.rdbwselect(data, "y", "x", all=True).set_index("method")
    row = table.loc[method]
    got = (row["h_left"], row["h_right"], row["b_left"], row["b_right"])
    assert got == pytest.approx(REFERENCE[method], rel=RTOL)


def test_default_estimate_matches_r_and_stata(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.rdrobust(data, "y", "x")
    info = res.model_info
    assert info["conventional"]["estimate"] == pytest.approx(0.801714670501, rel=RTOL)
    assert info["robust"]["estimate"] == pytest.approx(0.801942960855, rel=RTOL)
    assert info["conventional"]["se"] == pytest.approx(0.019005495153, rel=RTOL)
    assert info["robust"]["se"] == pytest.approx(0.020682195744, rel=RTOL)
