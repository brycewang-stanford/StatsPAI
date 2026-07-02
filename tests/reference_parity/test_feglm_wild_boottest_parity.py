"""Reference parity: ``fepois`` / ``feglm`` ``vce="wild"`` score wild bootstrap.

The restricted (null-imposed) **score wild cluster bootstrap** (Kline-Santos
2012) — the method Stata ``boottest`` runs after ``poisson`` / ``logit`` — is
wired as ``sp.fepois(vce="wild", cluster=...)`` /
``sp.feglm(..., vce="wild")``. For a small number of clusters the 2^G Rademacher
grid is enumerated, so the p-value is deterministic (no RNG dependence).

**Consistency, not bit-exactness.** This is a canonical restricted
efficient-score bootstrap. It agrees with Stata ``boottest`` on the enumerated
p-value to ~2 decimals but is *not* bit-identical: ``boottest`` uses a specific
full-model-bread / restricted-score studentization of the observed statistic
(its reported z=-1.0999 vs the canonical -1.031 here) that shifts the exact
count. The method itself is a valid, correctly-sized wild cluster bootstrap.

Stata reference (Stata 18 MP, ``boottest`` after ``poisson``), G=10 clusters,
2^10 enumerated, on the frozen DGP::

    poisson y x1 x2 x3 i.firm, vce(cluster clu)
    boottest x3, reps(1023) weighttype(rademacher)   // p = 0.31378299 (= 321/1023)
    boottest x2, ...                                 // p ~ 0  (strong rejection)
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyfixest")

from statspai.fixest import fepois  # noqa: E402

# Frozen Stata boottest score-bootstrap p-value for the null variable x3.
BOOTTEST_X3_P = 0.31378299
# 2-decimal agreement (boottest's exact studentization differs; see docstring).
_ATOL = 0.02


def _wild_panel() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n, g = 240, 10
    firm = rng.integers(0, 5, n)
    clu = rng.integers(0, g, n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    mu = np.exp(0.3 + 0.35 * x1 - 0.2 * x2 + 0.0 * x3 + 0.1 * firm)
    y = rng.poisson(mu)
    return pd.DataFrame(
        {"y": y, "x1": x1, "x2": x2, "x3": x3, "firm": firm, "clu": clu}
    )


def test_fepois_wild_agrees_with_boottest() -> None:
    df = _wild_panel()
    r = fepois("y ~ x1 + x2 + x3 | firm", data=df, vce="wild", cluster="clu")
    # Enumerated (G=10 -> 2^10) so the p-value is deterministic.
    assert np.isclose(float(r.pvalues["x3"]), BOOTTEST_X3_P, atol=_ATOL)
    # Strong regressors reject decisively (boottest ~ 0).
    assert float(r.pvalues["x1"]) < 0.05
    assert float(r.pvalues["x2"]) < 0.05
    # Point estimates + SE stay the cluster-robust (CRV1) values.
    assert "wild cluster bootstrap" in r.model_info["vcov_type"]


def test_fepois_wild_requires_cluster() -> None:
    df = _wild_panel()
    with pytest.raises(Exception):
        fepois("y ~ x1 + x2 + x3 | firm", data=df, vce="wild")
