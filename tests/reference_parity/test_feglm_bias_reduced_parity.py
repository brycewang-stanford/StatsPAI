"""Reference parity: ``fepois`` / ``feglm`` ``vce="CR2"/"CR3"/"jackknife"``.

The GLM bias-reduced cluster-robust SEs (Pustejovsky-Tipton 2018) are computed
on the **fixed-effects-as-dummies** design and reproduce R
``clubSandwich::vcovCR(glm_fit, type="CR2"/"CR3")`` to convergence precision.

Unlike OLS, the weighted-projection Frisch-Waugh-Lovell equivalence does *not*
carry the CR2 leverage adjustment through fixed-effect absorption for a GLM, so
the dummy design is required to match the clubSandwich reference (verified
directly: the absorbed design differs by ~1%). ``glm_cr_vcov`` implements the
IRLS-weighted adjustment for any exponential family (``d = dμ/dη``,
``V = Var(μ)``, working weight ``w = d²/V``).

clubSandwich references (R 4.5, clubSandwich 0.6.2)::

    # Poisson, FE=firm as factor, cluster=clu  (n=300)
    g <- glm(y ~ x1 + x2 + firm, family=poisson, data=d)
    sqrt(diag(vcovCR(g, cluster=d$clu, type="CR2")))[c("x1","x2")]
    #   x1=0.030557580  x2=0.039172730
    sqrt(diag(vcovCR(g, cluster=d$clu, type="CR3")))[c("x1","x2")]
    #   x1=0.032057722  x2=0.040952671

    # Binomial/logit, FE=firm as factor, cluster=clu  (n=350)
    g <- glm(y ~ x1 + x2 + firm, family=binomial, data=d)
    sqrt(diag(vcovCR(g, cluster=d$clu, type="CR2")))[c("x1","x2")]
    #   x1=0.098410759  x2=0.131109523
    sqrt(diag(vcovCR(g, cluster=d$clu, type="CR3")))[c("x1","x2")]
    #   x1=0.102811150  x2=0.136655962
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyfixest")

from statspai.fixest import feglm, fepois  # noqa: E402

# Frozen clubSandwich glm references (see module docstring).
POIS_CR2 = {"x1": 0.030557580, "x2": 0.039172730}
POIS_CR3 = {"x1": 0.032057722, "x2": 0.040952671}
LOGIT_CR2 = {"x1": 0.098410759, "x2": 0.131109523}
LOGIT_CR3 = {"x1": 0.102811150, "x2": 0.136655962}

# atol=1e-4 (~0.1-0.3% of the SE) is a strong parity claim: the *absorbed*
# (non-reference) design differs by ~1%, so this cleanly separates correct from
# wrong while tolerating pyfixest-vs-R IRLS μ-convergence noise (~1e-6).
_ATOL = 1e-4


def _pois_panel() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n, g = 300, 12
    firm = rng.integers(0, 6, n)
    clu = rng.integers(0, g, n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    mu = np.exp(0.2 + 0.5 * x1 - 0.3 * x2 + 0.15 * firm)
    y = rng.poisson(mu)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "firm": firm, "clu": clu})


def _logit_panel() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 350
    firm = rng.integers(0, 5, n)
    clu = rng.integers(0, 14, n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-(0.2 + 0.7 * x1 - 0.4 * x2 + 0.3 * firm)))
    y = (rng.random(n) < p).astype(int)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "firm": firm, "clu": clu})


def test_fepois_cr2_cr3_match_clubsandwich() -> None:
    df = _pois_panel()
    r2 = fepois("y ~ x1 + x2 | firm", data=df, vce="CR2", cluster="clu")
    r3 = fepois("y ~ x1 + x2 | firm", data=df, vce="CR3", cluster="clu")
    rj = fepois("y ~ x1 + x2 | firm", data=df, vce="jackknife", cluster="clu")
    for v in ("x1", "x2"):
        assert np.isclose(float(r2.std_errors[v]), POIS_CR2[v], atol=_ATOL)
        assert np.isclose(float(r3.std_errors[v]), POIS_CR3[v], atol=_ATOL)
        assert np.isclose(float(rj.std_errors[v]), POIS_CR3[v], atol=_ATOL)


def test_feglm_logit_cr2_cr3_match_clubsandwich() -> None:
    df = _logit_panel()
    r2 = feglm("y ~ x1 + x2 | firm", data=df, family="logit", vce="CR2", cluster="clu")
    r3 = feglm("y ~ x1 + x2 | firm", data=df, family="logit", vce="CR3", cluster="clu")
    for v in ("x1", "x2"):
        assert np.isclose(float(r2.std_errors[v]), LOGIT_CR2[v], atol=_ATOL)
        assert np.isclose(float(r3.std_errors[v]), LOGIT_CR3[v], atol=_ATOL)


def test_fepois_cr2_requires_cluster() -> None:
    df = _pois_panel()
    with pytest.raises(Exception):
        fepois("y ~ x1 + x2 | firm", data=df, vce="CR2")
