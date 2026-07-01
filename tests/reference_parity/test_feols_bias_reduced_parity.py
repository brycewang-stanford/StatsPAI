"""Reference parity: ``feols(vce="CR2"/"CR3"/"conley")`` on the FE-absorbed design.

* **CR2 / CR3** (== ``vce="jackknife"``) are the Pustejovsky-Tipton (2018)
  bias-reduced cluster-robust SEs applied to the within-transformed (FE-absorbed)
  design that ``sp.feols`` stores. The within-transform's leverage adjustment
  reproduces R ``clubSandwich::vcovCR(plm_within_model, type=...)`` to machine
  precision — the absorbed fixed effects are handled correctly for CR2/CR3.

* **Conley** is spatial HAC on the within design (acreg planar distance). It is
  validated by the internal anchor: ``feols(vce="conley")`` on an FE model equals
  ``sp.regress(vce="conley")`` on the same data FE-demeaned by hand — and
  ``sp.regress``'s Conley is pinned to Stata ``acreg`` (see
  ``REFERENCES.md`` / test_regress SE parity).

clubSandwich reference (R 4.5, plm + clubSandwich), 500 obs / 18 clusters::

    df$firm <- factor(df$firm)
    pm <- plm(y ~ x + z, data=df, index="firm", model="within")
    sqrt(diag(vcovCR(pm, cluster=df$firm, type="CR2")))["x"]  # 0.056095873
    sqrt(diag(vcovCR(pm, cluster=df$firm, type="CR3")))["x"]  # 0.057956013
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyfixest")

from statspai.fixest import feols  # noqa: E402
from statspai.regression.ols import regress  # noqa: E402

PLM_CR2_X = 0.056095873
PLM_CR3_X = 0.057956013


def _fe_panel() -> pd.DataFrame:
    rng = np.random.default_rng(2027)
    n, g = 500, 18
    firm = rng.integers(1, g + 1, n)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    fe = rng.normal(0, 1, g + 1)[firm]
    y = 1.0 + 0.5 * x - 0.3 * z + fe + rng.normal(size=n)
    return pd.DataFrame({"y": y, "x": x, "z": z, "firm": firm})


def test_feols_cr2_cr3_match_clubsandwich_plm() -> None:
    df = _fe_panel()
    r2 = feols("y ~ x + z | firm", data=df, vce="CR2", cluster="firm")
    r3 = feols("y ~ x + z | firm", data=df, vce="CR3", cluster="firm")
    assert np.isclose(float(r2.std_errors["x"]), PLM_CR2_X, atol=1e-7)
    assert np.isclose(float(r3.std_errors["x"]), PLM_CR3_X, atol=1e-7)
    # vce="jackknife" is an alias for CR3
    rj = feols("y ~ x + z | firm", data=df, vce="jackknife", cluster="firm")
    assert np.isclose(float(rj.std_errors["x"]), PLM_CR3_X, atol=1e-7)


def test_feols_cr2_requires_cluster() -> None:
    df = _fe_panel()
    with pytest.raises(Exception):
        feols("y ~ x | firm", data=df, vce="CR2")


def _spatial_fe_panel() -> pd.DataFrame:
    rng = np.random.default_rng(21)
    n, g = 400, 15
    firm = rng.integers(1, g + 1, n)
    lat = rng.uniform(30, 45, n)
    lon = rng.uniform(-120, -100, n)
    x = rng.normal(size=n)
    fe = rng.normal(0, 1, g + 1)[firm]
    y = 1.0 + 0.5 * x + fe + rng.normal(size=n)
    return pd.DataFrame({"y": y, "x": x, "firm": firm, "lat": lat, "lon": lon})


def test_feols_conley_matches_regress_on_demeaned() -> None:
    """feols-FE Conley == regress(Conley) on the FE-demeaned data (whose Conley
    is pinned to Stata acreg). Also anchors the no-FE case to plain regress."""
    df = _spatial_fe_panel()
    f = feols(
        "y ~ x | firm",
        data=df,
        vce="conley",
        conley_lat="lat",
        conley_lon="lon",
        conley_cutoff=200,
    )
    dm = df.copy()
    for c in ("y", "x"):
        dm[c] = df[c] - df.groupby("firm")[c].transform("mean")
    r = regress(
        "y ~ x - 1",
        dm,
        vce="conley",
        conley_lat="lat",
        conley_lon="lon",
        conley_cutoff=200,
    )
    assert np.isclose(float(f.std_errors["x"]), float(r.std_errors["x"]), atol=1e-9)

    # no-FE anchor: feols("y ~ x") Conley == regress("y ~ x") Conley
    f0 = feols(
        "y ~ x",
        data=df,
        vce="conley",
        conley_lat="lat",
        conley_lon="lon",
        conley_cutoff=200,
    )
    r0 = regress(
        "y ~ x",
        df,
        vce="conley",
        conley_lat="lat",
        conley_lon="lon",
        conley_cutoff=200,
    )
    assert np.isclose(float(f0.std_errors["x"]), float(r0.std_errors["x"]), atol=1e-9)
