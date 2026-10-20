"""Reference parity: the grid density estimator vs R ``lpdensity`` 2.5.

``statspai.rd._lpdensity.lpdensity_numbers`` draws the curves of
``sp.rdplotdensity``. Until the 2026-10 clean-room rewrite it had no
reference fixture at all (license-provenance audit, row H7). The fixture
``lpdensity_R.json`` (``_fixtures/_generate_lpdensity_R.R``) records R
``lpdensity(...)$Estimate`` on two committed samples: 300 normal draws
(no ties) and the same rounded to one decimal (mass points), across the
three kernels, polynomial orders 1-4, the CDF (``v=0``), the second
derivative (``v=2``), per-point bandwidths, ``scale`` and both settings
of ``massPoints``. R was run as a black box.

Measured 2026-10-20: estimates and standard errors agree to 1.5e-11 or
better in every cell (the worst is ``f_q`` of the order-4 fit for the
second derivative). ``RTOL`` leaves room for BLAS differences across
platforms, nothing more: both sides solve the same small weighted least
squares problems.

The analytic checks below need no reference: on an equally spaced sample
the empirical CDF is exactly linear, so every interior estimate equals
the spacing's reciprocal density whatever the kernel and order.

References
----------
cattaneo2020simple
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from statspai.rd._lpdensity import lpdensity_numbers

_FIX = pathlib.Path(__file__).parent / "_fixtures"
RTOL = 1e-9


@pytest.fixture(scope="module")
def ref():
    return json.loads((_FIX / "lpdensity_R.json").read_text(encoding="utf-8"))


def _sample(name):
    import pandas as pd

    return pd.read_csv(_FIX / f"lpdensity_{name}.csv")["x"].to_numpy()


def _cells(ref):
    return sorted(k for k in ref if not k.startswith("_"))


def _as_float(values):
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def test_samples_are_the_r_side_bytes(ref):
    assert len(_sample("cont")) == ref["_meta"]["n_cont"]
    assert len(_sample("tied")) == ref["_meta"]["n_tied"]
    # the tied sample really has mass points, the continuous one none
    assert np.unique(_sample("tied")).size < 100
    assert np.unique(_sample("cont")).size == ref["_meta"]["n_cont"]


@pytest.mark.parametrize(
    "cell",
    [
        "tri_p2",
        "uni_p2",
        "epa_p2",
        "tri_p1",
        "tri_p3v2",
        "tri_cdf",
        "tri_bwvec",
        "tri_scale",
        "tied_mp",
        "tied_nomp",
    ],
)
def test_estimates_and_standard_errors_match_r(ref, cell):
    want = ref[cell]
    a = want["args"]
    got = lpdensity_numbers(
        _sample(a["data"]),
        np.asarray(a["grid"], dtype=float),
        np.asarray(a["bw"], dtype=float),
        p=a["p"],
        q=a["q"],
        v=a["v"],
        kernel=a["kernel"],
        scale=a["scale"],
        mass_points=a["mass_points"],
    )
    np.testing.assert_array_equal(got["nh"], want["nh"])
    for key in ("f_p", "f_q", "se_p", "se_q"):
        np.testing.assert_allclose(
            got[key], _as_float(want[key]), rtol=RTOL, equal_nan=True, err_msg=key
        )


def test_every_fixture_cell_is_exercised(ref):
    assert len(_cells(ref)) == 10


def test_mass_points_switch_matters_on_tied_data(ref):
    """The two tied cells must differ, or the switch is not being tested."""
    mp = _as_float(ref["tied_mp"]["f_p"])
    no_mp = _as_float(ref["tied_nomp"]["f_p"])
    assert np.max(np.abs(mp - no_mp)) > 1e-3


@pytest.mark.parametrize("kernel", ["triangular", "uniform", "epanechnikov"])
@pytest.mark.parametrize("p", [1, 2, 3])
def test_equally_spaced_sample_gives_its_exact_density(kernel, p):
    """ECDF of x_i = a + (b - a)(i - 1)/(n - 1) is i/n, linear in x, so the
    local polynomial reproduces it exactly: f = (n - 1) / (n (b - a))."""
    a, b, n = -1.0, 3.0, 401
    x = np.linspace(a, b, n)
    grid = np.array([-0.2, 0.5, 1.0, 2.1])
    out = lpdensity_numbers(x, grid, 0.6, p=p, kernel=kernel, mass_points=False)
    want = (n - 1) / (n * (b - a))
    np.testing.assert_allclose(out["f_p"], want, rtol=1e-10)
    np.testing.assert_allclose(out["f_q"], want, rtol=1e-10)
    # the CDF itself (v = 0) is the line through the ECDF points
    cdf = lpdensity_numbers(x, grid, 0.6, p=p, v=0, kernel=kernel)
    np.testing.assert_allclose(
        cdf["f_p"], (1 + (n - 1) * (grid - a) / (b - a)) / n, rtol=1e-10
    )


def test_scale_multiplies_estimates_and_standard_errors():
    x = _sample("cont")
    grid = np.array([-0.5, 0.0, 0.7])
    base = lpdensity_numbers(x, grid, 0.8)
    half = lpdensity_numbers(x, grid, 0.8, scale=0.5)
    for key in ("f_p", "f_q", "se_p", "se_q"):
        np.testing.assert_allclose(half[key], 0.5 * base[key], rtol=1e-14)


def test_unidentified_windows_are_nan_not_numbers():
    """Fewer distinct positive-weight points than coefficients: R returns NA."""
    x = np.r_[np.zeros(30), np.ones(30)]
    out = lpdensity_numbers(x, np.array([0.5, 7.0]), 0.8, p=2)
    assert out["nh"].tolist() == [60, 0]
    assert np.isnan(out["f_p"]).all() and np.isnan(out["se_p"]).all()


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"bw": -0.5}, "positive"),
        ({"bw": [0.5, 0.6]}, "one value per grid point"),
        ({"bw": 0.5, "kernel": "gaussian"}, "kernel"),
        ({"bw": 0.5, "p": 1, "v": 2}, "v must be"),
    ],
)
def test_invalid_arguments_raise(kwargs, match):
    from statspai.exceptions import MethodIncompatibility

    x = _sample("cont")
    with pytest.raises(MethodIncompatibility, match=match):
        lpdensity_numbers(x, np.array([0.0, 0.5, 1.0]), **kwargs)
