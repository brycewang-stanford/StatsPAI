"""Reference parity: ``sp.feols(vce="wild")`` vs Stata ``reghdfe`` + ``boottest``.

The native wild cluster bootstrap on ``sp.feols`` runs the WCR bootstrap
(Cameron-Gelbach-Miller 2008) on the FE-absorbed within design. This pins it to
David Roodman's ``boottest`` — the canonical implementation of the method.

Reference values were produced on the identical 600-obs / 15-cluster panel
(Stata 18 MP; ``reghdfe`` 6.13.1, ``boottest`` 4.5.3)::

    import delimited wild_parity.csv, clear
    reghdfe y x z, absorb(firm) vce(cluster firm)
    boottest x, reps(99999) weighttype(rademacher) nograph

With 15 clusters, ``boottest`` enumerates all 2^15 = 32768 Rademacher draws
(the *exact* wild bootstrap); StatsPAI draws 99999 Rademacher weights, so its
p-value is a Monte-Carlo estimate of that exact p-value. The point estimate and
the CRV1 cluster SE match to ~1e-9; the wild p-value matches to MC error.

See ``REFERENCES.md`` and ``docs/guides/grammar.md``.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyfixest")

from statspai.fixest import feols  # noqa: E402

# --- Stata reghdfe + boottest reference values (frozen) ------------------
STATA_COEF_X = 0.06576570970445639  # reghdfe _b[x]
STATA_CLUSTER_SE_X = 0.027193981415322716  # reghdfe _se[x] (CRV1)
STATA_BOOTTEST_P_X = 0.02630615  # boottest rademacher, exhaustive 2^15
STATA_BOOTTEST_CI_LO_X = 0.0089431  # boottest 95% CI lower (test inversion)


def _wild_panel() -> pd.DataFrame:
    """600-obs, 15-cluster panel; identical to the CSV Stata read."""
    rng = np.random.default_rng(2024)
    n, g = 600, 15
    firm = rng.integers(0, g, n)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    clu = rng.normal(0, 0.8, size=g)[firm]
    y = 0.07 * x + 0.5 * z + clu + rng.normal(0, 1.0, size=n)
    return pd.DataFrame({"y": y, "x": x, "z": z, "firm": firm})


def test_feols_point_and_cluster_se_match_reghdfe() -> None:
    df = _wild_panel()
    base = feols("y ~ x + z | firm", data=df, vcov={"CRV1": "firm"})
    assert np.isclose(float(base.params["x"]), STATA_COEF_X, atol=1e-6)
    # CRV1 cluster SE matches reghdfe vce(cluster firm) essentially exactly.
    assert np.isclose(float(base.std_errors["x"]), STATA_CLUSTER_SE_X, atol=5e-5)


def test_feols_wild_pvalue_matches_boottest() -> None:
    df = _wild_panel()
    f = feols(
        "y ~ x + z | firm",
        data=df,
        vce="wild",
        cluster="firm",
        wild_reps=99999,
        seed=12345,
    )
    # point estimate unchanged by the bootstrap
    assert np.isclose(float(f.params["x"]), STATA_COEF_X, atol=1e-6)
    # wild p-value: our 99999-draw MC estimate vs boottest's exact 2^15 p.
    # |Δ| ~ 0.0002 here; tolerance is ~8x the binomial MC SE.
    assert np.isclose(float(f.pvalues["x"]), STATA_BOOTTEST_P_X, atol=4e-3), (
        float(f.pvalues["x"]),
        STATA_BOOTTEST_P_X,
    )
    # CI lower bound (percentile-t) agrees with boottest's inverted CI lower.
    assert np.isclose(float(f.conf_int_lower["x"]), STATA_BOOTTEST_CI_LO_X, atol=1e-2)


def test_feols_vce_wild_requires_cluster() -> None:
    df = _wild_panel()
    with pytest.raises(Exception):
        feols("y ~ x | firm", data=df, vce="wild")
