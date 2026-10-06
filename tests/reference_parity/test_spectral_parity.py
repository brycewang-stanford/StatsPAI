"""``periodogram`` and ``cumulative_periodogram_test`` against R and Stata.

Everything here is deterministic, so the comparison is digit for digit on
the committed synthetic file ``_fixtures/spectral.csv``
(``_generate_spectral_data.py``).

* R 4.5.2 ``stats::spec.pgram`` and ``stats::spec.ar``
  (``_generate_spectral_R.R`` -> ``spectral_R.json``): frequencies,
  ordinates, equivalent degrees of freedom, bandwidth, and the interval
  ``plot.spec`` draws.
* Stata 18 ``pergram, generate()`` and ``wntestb``
  (``_generate_spectral_Stata.do`` -> ``spectral_Stata.csv``,
  ``spectral_Stata_scalars.csv``).

Tolerance ``EXACT`` = 1e-10 relative: the two sides run different FFT
codes and the kernel sums differ in order; observed errors are below
2e-12. The one looser assert is Stata's p-value, explained where it is
made.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import special

from statspai.timeseries.spectral import cumulative_periodogram_test, periodogram

FIX = Path(__file__).parent / "_fixtures"
EXACT = 1e-10


def rel(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape
    return float(np.max(np.abs(a - b) / np.abs(b)))


@pytest.fixture(scope="module")
def R() -> dict:
    return json.loads((FIX / "spectral_R.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.read_csv(FIX / "spectral.csv")


PGRAM_CASES = {
    # R: spec.pgram(x), with fast = TRUE a no-op because 150 = 2 * 3 * 5^2
    "default": {},
    "textbook": {"taper": 0, "detrend": "mean"},
    "none": {"taper": 0, "detrend": "none"},
    "spans35": {"method": "smoothed", "spans": (3, 5)},
    "spans7_taper25": {
        "method": "smoothed",
        "spans": 7,
        "taper": 0.25,
        "detrend": "mean",
    },
    "spans4": {"method": "smoothed", "spans": 4},
    "pad1": {"method": "smoothed", "spans": 5, "pad": 1},
}


@pytest.mark.parametrize("case", sorted(PGRAM_CASES))
def test_spec_pgram(R: dict, df: pd.DataFrame, case: str) -> None:
    res = periodogram(df, "x", **PGRAM_CASES[case])
    ref = R[case]
    assert rel(res.table["freq"], ref["freq"]) < EXACT
    assert rel(res.table["spectrum"], ref["spec"]) < EXACT
    assert rel(res.df, ref["df"]) < EXACT
    assert rel(res.bandwidth, ref["bandwidth"]) < EXACT
    # plot.spec's 95% interval: spec * df / qchisq(c(.975, .025), df)
    assert rel(res.table["lower"], ref["lower"]) < EXACT
    assert rel(res.table["upper"], ref["upper"]) < EXACT


@pytest.mark.parametrize("fast", [True, False])
def test_spec_pgram_prime_length(R: dict, df: pd.DataFrame, fast: bool) -> None:
    # n = 127 is prime: R's default pads to 128, fast = FALSE does not
    x = df["x"].to_numpy()[:127]
    res = periodogram(x, method="smoothed", spans=(3, 3), fast=fast)
    ref = R["odd_fast" if fast else "odd_nofast"]
    assert len(res.table) == (64 if fast else 63)
    assert rel(res.table["freq"], ref["freq"]) < EXACT
    assert rel(res.table["spectrum"], ref["spec"]) < EXACT
    assert rel(res.df, ref["df"]) < EXACT
    assert rel(res.bandwidth, ref["bandwidth"]) < EXACT


def test_spec_ar_aic(R: dict, df: pd.DataFrame) -> None:
    res = periodogram(df, "x", method="ar")
    assert res.order == R["ar_aic_order"] == 3
    assert rel(res.var_pred, R["ar_aic_varpred"]) < EXACT
    freq = np.asarray(R["ar_aic"]["freq"])
    assert np.max(np.abs(res.table["freq"] - freq)) < 1e-15
    assert rel(res.table["spectrum"], R["ar_aic"]["spec"]) < EXACT


def test_spec_ar_fixed_order(R: dict, df: pd.DataFrame) -> None:
    res = periodogram(df, "x", method="ar", order=3, n_freq=101)
    assert rel(res.table["spectrum"], R["ar_fixed3"]["spec"]) < EXACT


def test_spec_ar_selects_order_zero_for_white_noise(R: dict, df: pd.DataFrame) -> None:
    res = periodogram(df, "z", method="ar", max_order=4)
    assert res.order == R["ar_max4_order"] == 0
    assert rel(res.table["spectrum"], R["ar_max4"]["spec"]) < EXACT


def test_stata_pergram(df: pd.DataFrame) -> None:
    """``pergram, generate()`` saves ``|sum_t (x_t - mean) e^{..}|^2`` at
    ``(k - 1) / n``: ``n`` times our untapered periodogram per cycle."""
    stata = pd.read_csv(FIX / "spectral_Stata.csv")
    for col, n in (("x", 150), ("z", 150), ("x", 127)):
        name = f"pg_{col}" if n == 150 else "pg_x127"
        x = df[col].to_numpy()[:n]
        res = periodogram(x, taper=0, detrend="mean")
        ref = stata[name].to_numpy()[1 : n // 2 + 1]
        assert rel(res.table["spectrum"] * n, ref) < EXACT


@pytest.mark.parametrize("series", ["x", "z", "x127", "x101"])
def test_stata_wntestb(df: pd.DataFrame, series: str) -> None:
    scalars = pd.read_csv(FIX / "spectral_Stata_scalars.csv")
    ref = scalars[scalars["series"] == series].set_index("stat")["value"]
    x = {
        "x": df["x"].to_numpy(),
        "z": df["z"].to_numpy(),
        "x127": df["x"].to_numpy()[:127],
        "x101": df["z"].to_numpy()[:101],
    }[series]
    res = cumulative_periodogram_test(x)
    assert rel(res.statistic, ref["stat"]) < EXACT
    # Stata sums the alternating Kolmogorov series until a term is small
    # and stops: for z it omits the fifth term, 2 exp(-50 B^2) = 1.02e-8,
    # which is the whole gap. Ours is the full series (scipy), checked
    # below; 2e-8 absolute bounds the truncation.
    assert abs(res.pvalue - ref["p"]) < 2e-8
    j = np.arange(1, 200)
    exact = 2.0 * np.sum((-1.0) ** (j - 1) * np.exp(-2.0 * j**2 * res.statistic**2))
    assert abs(res.pvalue - exact) < 1e-14
    assert res.pvalue == special.kolmogorov(res.statistic)


def test_stata_p_gap_is_the_omitted_term(df: pd.DataFrame) -> None:
    scalars = pd.read_csv(FIX / "spectral_Stata_scalars.csv")
    ref = scalars[scalars["series"] == "z"].set_index("stat")["value"]
    res = cumulative_periodogram_test(df["z"].to_numpy())
    j = np.arange(1, 5)
    four = 2.0 * np.sum((-1.0) ** (j - 1) * np.exp(-2.0 * j**2 * res.statistic**2))
    assert abs(four - ref["p"]) < 1e-13
