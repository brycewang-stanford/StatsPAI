"""Conley spatial HAC must not fail silently.

Two ways it used to:

1. **Negative variances clamped to zero.** A kernel-weighted HAC meat matrix
   is not PSD by construction — with a uniform (indicator) spatial kernel,
   ``S'WS`` routinely lands with negative diagonal entries. The repo idiom
   ``sqrt(maximum(diag(V), 0))`` turned that into ``se = 0``, which reads
   downstream as an infinitely precise estimate (``t = inf``, ``p = 0``).
   Stata ``acreg`` reports those terms as missing; so do we now.

2. **Coordinates paired positionally with a shorter design.** When the
   estimator drops rows, coordinate *i* stops describing observation *i*.

Both are pinned here as behaviour, not as numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility, NumericalInstability
from statspai.inference._psd import psd_diagnostics, se_from_vcov

pytest.importorskip("pyfixest")


@pytest.fixture(scope="module")
def collinear_coords_panel() -> pd.DataFrame:
    """Coordinates are a function of the absorbed FE — drives the meat non-PSD.

    Every observation in a county shares one location, so the spatial weights
    carry no within-county information once county FE are partialled out.
    Stata acreg returns V[1,1] = -0.000420113683069266 here.
    """
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame(
        {
            "county": np.repeat(np.arange(30), 20),
            "year": np.tile(np.arange(20), 30),
        }
    )
    df["x"] = rng.normal(size=n)
    df["d"] = (df["county"] < 15).astype(int) * (df["year"] >= 10).astype(int)
    df["y"] = 1 + 0.3 * df["d"] + 0.5 * df["x"] + rng.normal(size=n)
    df["lat"] = 30 + df["county"] * 0.3
    df["lon"] = 110 + df["county"] * 0.25
    return df


# --------------------------------------------------------------------------- #
#  1. Non-PSD covariance -> nan + warning, never 0
# --------------------------------------------------------------------------- #


def test_negative_variance_reports_nan_not_zero(collinear_coords_panel):
    with pytest.warns(RuntimeWarning, match="non-positive-definite"):
        res = sp.feols(
            "y ~ d + x | county",
            data=collinear_coords_panel,
            vce="conley",
            conley_lat="lat",
            conley_lon="lon",
            conley_cutoff=500,
        )
    se = res.std_errors
    assert np.isnan(se["d"]), "a negative variance must surface as nan"
    assert np.isnan(se["x"])
    # The specific failure this guards: se == 0 implies t = inf, p = 0.
    assert not (se.fillna(-1) == 0).any()


def test_warning_explains_and_suggests_remedies(collinear_coords_panel):
    with pytest.warns(RuntimeWarning) as rec:
        sp.feols(
            "y ~ d + x | county",
            data=collinear_coords_panel,
            vce="conley",
            conley_lat="lat",
            conley_lon="lon",
            conley_cutoff=500,
        )
    msg = "\n".join(str(w.message) for w in rec)
    assert "bartlett" in msg
    assert "cutoff" in msg
    assert "collinear" in msg


def test_se_from_vcov_clamps_rounding_noise_silently():
    """A -1e-18 next to a variance of 1.0 is rounding, not failure."""
    V = np.diag([1.0, -1e-18])
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        se = se_from_vcov(V, ["a", "b"])
    assert se[0] == pytest.approx(1.0)
    assert se[1] == 0.0


def test_se_from_vcov_can_raise_instead():
    V = np.diag([1.0, -0.5])
    with pytest.raises(NumericalInstability, match="non-positive-definite"):
        se_from_vcov(V, ["a", "b"], on_negative="raise")


def test_psd_diagnostics_separates_noise_from_failure():
    diag = psd_diagnostics(np.diag([2.0, -1e-19, -0.75]))
    assert diag["n_negative"] == 1
    assert diag["noise_mask"].tolist() == [False, True, False]
    assert diag["negative_mask"].tolist() == [False, False, True]


# --------------------------------------------------------------------------- #
#  2. Coordinate / design alignment
# --------------------------------------------------------------------------- #


def test_dropped_rows_refuse_rather_than_mispair(collinear_coords_panel):
    df = collinear_coords_panel.copy()
    df.loc[5, "x"] = np.nan  # pyfixest drops this row; coords still have 600
    with pytest.raises(MethodIncompatibility, match="rows were dropped"):
        sp.feols(
            "y ~ d + x | county",
            data=df,
            vce="conley",
            conley_lat="lat",
            conley_lon="lon",
            conley_cutoff=500,
        )


def test_error_tells_the_caller_how_to_recover(collinear_coords_panel):
    df = collinear_coords_panel.copy()
    df.loc[5, "x"] = np.nan
    with pytest.raises(MethodIncompatibility) as exc:
        sp.feols(
            "y ~ d + x | county",
            data=df,
            vce="conley",
            conley_lat="lat",
            conley_lon="lon",
            conley_cutoff=500,
        )
    assert "dropna" in str(exc.value)


def test_missing_coordinate_is_rejected(collinear_coords_panel):
    df = collinear_coords_panel.copy()
    df.loc[7, "lat"] = np.nan
    with pytest.raises(MethodIncompatibility, match="missing values"):
        sp.feols(
            "y ~ d + x | county",
            data=df,
            vce="conley",
            conley_lat="lat",
            conley_lon="lon",
            conley_cutoff=500,
        )


# --------------------------------------------------------------------------- #
#  3. Parity: where the estimator is well defined, we match acreg
# --------------------------------------------------------------------------- #


def test_matches_stata_acreg_on_psd_design():
    """60 units x 12 periods, locations independent of any FE.

    Stata 18 MP:
        acreg y x z, spatial latitude(lat) longitude(lon) dist(500)
        se_x = 0.055300980344313  se_z = 0.026023332902857
    """
    rng = np.random.default_rng(11)
    n_unit, n_time = 60, 12
    rows = []
    for u in range(n_unit):
        lat = 25 + rng.uniform(0, 12)
        lon = 105 + rng.uniform(0, 14)
        for t in range(n_time):
            rows.append(
                {
                    "unit": u,
                    "t": t,
                    "lat": lat,
                    "lon": lon,
                    "x": rng.normal(),
                    "z": rng.normal(),
                }
            )
    df = pd.DataFrame(rows)
    df["y"] = 1 + 0.5 * df["x"] - 0.3 * df["z"] + rng.normal(size=len(df))

    res = sp.feols(
        "y ~ x + z",
        data=df,
        vce="conley",
        conley_lat="lat",
        conley_lon="lon",
        conley_cutoff=500,
    )
    assert float(res.std_errors["x"]) == pytest.approx(0.055300980344313, rel=1e-7)
    assert float(res.std_errors["z"]) == pytest.approx(0.026023332902857, rel=1e-7)
