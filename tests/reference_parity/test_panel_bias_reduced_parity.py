"""Reference parity: ``sp.panel(method="fe", vce=...)`` extended SE menu.

The panel one-way (entity) fixed-effects estimator gets the same bias-reduced /
spatial-HAC / two-way cluster SE menu as ``sp.feols``, computed on the
entity-within (FE-absorbed) design:

* **CR2 / CR3** (== ``vce="jackknife"``) are the Pustejovsky-Tipton (2018)
  bias-reduced cluster-robust SEs.  Because OLS on the entity-demeaned design
  reproduces the linearmodels FE coefficients, and the within-transform's
  leverage adjustment reproduces R ``clubSandwich::vcovCR(plm, model="within")``
  exactly, the panel FE SEs equal the *identical* clubSandwich reference already
  frozen for ``sp.feols`` (``test_feols_bias_reduced_parity``):

      df$firm <- factor(df$firm)
      pm <- plm(y ~ x + z, data=df, index="firm", model="within")
      sqrt(diag(vcovCR(pm, cluster=df$firm, type="CR2")))["x"]  # 0.056095873
      sqrt(diag(vcovCR(pm, cluster=df$firm, type="CR3")))["x"]  # 0.057956013

* **Conley** spatial HAC and **two-way** clustering are anchored internally:
  ``sp.panel(method="fe", ...)`` equals ``sp.regress(...)`` run on the same data
  FE-demeaned by hand — and ``sp.regress``'s Conley is pinned to Stata ``acreg``
  and its two-way cluster to ``reghdfe`` / ``ivreg2`` (see ``REFERENCES.md`` and
  the ``sp.regress`` SE-parity tests).
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("linearmodels")

from statspai.panel import panel  # noqa: E402
from statspai.regression.ols import regress  # noqa: E402

# Frozen clubSandwich plm(model="within") reference — identical data / model as
# ``test_feols_bias_reduced_parity`` so the two estimators share one anchor.
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
    df = pd.DataFrame({"y": y, "x": x, "z": z, "firm": firm})
    df["t"] = df.groupby("firm").cumcount() + 1
    return df


def test_panel_fe_cr2_cr3_match_clubsandwich_plm() -> None:
    df = _fe_panel()
    r2 = panel(
        df, "y ~ x + z", entity="firm", time="t", method="fe", vce="CR2", cluster="firm"
    )
    r3 = panel(
        df, "y ~ x + z", entity="firm", time="t", method="fe", vce="CR3", cluster="firm"
    )
    rj = panel(
        df,
        "y ~ x + z",
        entity="firm",
        time="t",
        method="fe",
        vce="jackknife",
        cluster="firm",
    )
    assert np.isclose(float(r2.std_errors["x"]), PLM_CR2_X, atol=1e-7)
    assert np.isclose(float(r3.std_errors["x"]), PLM_CR3_X, atol=1e-7)
    assert np.isclose(float(rj.std_errors["x"]), PLM_CR3_X, atol=1e-7)
    # FE point estimates unaffected by the SE choice.
    assert np.isclose(float(r2.params["x"]), 0.500223, atol=1e-5)


def test_panel_fe_extended_vce_requires_fe() -> None:
    df = _fe_panel()
    with pytest.raises(Exception):
        panel(
            df,
            "y ~ x + z",
            entity="firm",
            time="t",
            method="re",
            vce="CR2",
            cluster="firm",
        )


def _spatial_fe_panel() -> pd.DataFrame:
    rng = np.random.default_rng(21)
    n, g = 400, 15
    firm = rng.integers(1, g + 1, n)
    lat = rng.uniform(30, 45, n)
    lon = rng.uniform(-120, -100, n)
    x = rng.normal(size=n)
    fe = rng.normal(0, 1, g + 1)[firm]
    y = 1.0 + 0.5 * x + fe + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "x": x, "firm": firm, "lat": lat, "lon": lon})
    df["t"] = df.groupby("firm").cumcount() + 1
    df["region"] = rng.integers(1, 8, n)
    return df


def test_panel_fe_conley_matches_regress_on_demeaned() -> None:
    df = _spatial_fe_panel()
    f = panel(
        df,
        "y ~ x",
        entity="firm",
        time="t",
        method="fe",
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


def test_panel_fe_twoway_matches_regress_on_demeaned() -> None:
    df = _spatial_fe_panel()
    f = panel(
        df, "y ~ x", entity="firm", time="t", method="fe", cluster=["firm", "region"]
    )
    dm = df.copy()
    for c in ("y", "x"):
        dm[c] = df[c] - df.groupby("firm")[c].transform("mean")
    r = regress("y ~ x - 1", dm, cluster=["firm", "region"])
    assert np.isclose(float(f.std_errors["x"]), float(r.std_errors["x"]), atol=1e-9)
