"""Reference parity: ``sp.hdfe_ols`` extended ``vce=`` menu.

``hdfe_ols`` (the native-numpy reghdfe-style HDFE estimator) gains the same
canonical ``vce=`` menu as ``sp.regress`` / ``sp.feols`` / ``sp.panel``,
computed on the absorber's within-transformed design:

* ``vce="robust"``/``"hc1"`` — HC1 with **reghdfe's** small-sample factor
  ``N/(N-k-df_a)``. Frozen against Stata 18
  ``reghdfe y x z, absorb(firm) vce(robust)`` on the shared frozen DGP::

      se_x = 0.0431561136   se_z = 0.0467233768   (df_r = 480 = N-k-df_a)

* ``vce="CR2"``/``"CR3"`` (== ``"jackknife"``) — Pustejovsky-Tipton on the
  within design; equals the *identical* frozen ``clubSandwich::vcovCR(plm,
  model="within")`` anchor used by ``sp.feols`` / ``sp.panel``
  (``PLM_CR2_X=0.056095873``, ``PLM_CR3_X=0.057956013``).

* ``vce="conley"`` — acreg planar spatial HAC; anchored internally:
  equals ``sp.regress(vce="conley")`` on the same data FE-demeaned by hand
  (whose Conley is pinned to Stata ``acreg``).

* ``vce="wild"`` — shorthand for the existing native ``wild=True`` WCR
  bootstrap path (grammar convergence only; the wild path itself is already
  validated against Stata ``boottest``).
"""

import numpy as np
import pandas as pd
import pytest

from statspai.panel.feols import feols as hdfe_ols
from statspai.regression.ols import regress

# Stata 18 MP: reghdfe y x z, absorb(firm) vce(robust)  [frozen DGP below]
REGHDFE_HC1_X = 0.0431561136
REGHDFE_HC1_Z = 0.0467233768
# R clubSandwich::vcovCR(plm within) — same anchor as feols/panel CR2 tests.
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


def test_hdfe_ols_hc1_matches_reghdfe_robust() -> None:
    df = _fe_panel()
    r = hdfe_ols("y ~ x + z | firm", data=df, vce="robust")
    assert np.isclose(float(r.std_errors["x"]), REGHDFE_HC1_X, atol=1e-9)
    assert np.isclose(float(r.std_errors["z"]), REGHDFE_HC1_Z, atol=1e-9)
    # hc1 is an alias for robust; hc0 drops the small-sample factor.
    r1 = hdfe_ols("y ~ x + z | firm", data=df, vce="hc1")
    assert float(r1.std_errors["x"]) == float(r.std_errors["x"])
    r0 = hdfe_ols("y ~ x + z | firm", data=df, vce="hc0")
    assert float(r0.std_errors["x"]) < float(r.std_errors["x"])


def test_hdfe_ols_cr2_cr3_match_clubsandwich_plm() -> None:
    df = _fe_panel()
    r2 = hdfe_ols("y ~ x + z | firm", data=df, vce="CR2", cluster="firm")
    r3 = hdfe_ols("y ~ x + z | firm", data=df, vce="CR3", cluster="firm")
    rj = hdfe_ols("y ~ x + z | firm", data=df, vce="jackknife", cluster="firm")
    assert np.isclose(float(r2.std_errors["x"]), PLM_CR2_X, atol=1e-7)
    assert np.isclose(float(r3.std_errors["x"]), PLM_CR3_X, atol=1e-7)
    assert float(rj.std_errors["x"]) == float(r3.std_errors["x"])


def test_hdfe_ols_conley_matches_regress_on_demeaned() -> None:
    df = _spatial_fe_panel()
    f = hdfe_ols(
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
    assert np.isclose(float(f.std_errors["x"]), float(r.std_errors["x"]), atol=1e-12)


def test_hdfe_ols_vce_wild_alias_and_guards() -> None:
    df = _spatial_fe_panel()
    rw = hdfe_ols("y ~ x | firm", data=df, vce="wild", cluster="firm", wild_seed=7)
    assert rw.se_type == "wild_cluster"
    assert "wild_p" in rw.cluster_info
    with pytest.raises(Exception):
        hdfe_ols("y ~ x | firm", data=df, vce="CR2")  # cluster required
    with pytest.raises(Exception):
        hdfe_ols("y ~ x | firm", data=df, vce="conley")  # coords required
    with pytest.raises(Exception):
        hdfe_ols("y ~ x", data=df, vce="robust")  # no FE -> use sp.regress
