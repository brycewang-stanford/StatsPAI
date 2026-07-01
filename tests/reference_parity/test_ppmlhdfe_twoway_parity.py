"""Reference parity: ``sp.ppmlhdfe(cluster=[a, b])`` two-way clustering.

Two-way cluster-robust SEs (Cameron-Gelbach-Miller 2011) for the PPML HDFE
estimator match Stata ``ppmlhdfe y x1 x2, absorb(o d) cluster(ca cb)`` to the
printed precision. The variance is ``(G_min/(G_min-1)) · bread · (M_ca + M_cb -
M_{ca∩cb}) · bread`` on the FE-residualised design — the single ``G_min/(G_min-1)``
small-sample factor is Stata's ppmlhdfe/reghdfe multiway convention (the one-way
``cluster(ca)`` path reduces to ``G/(G-1)`` and already matches Stata exactly).

Stata references (Stata 18 MP, ppmlhdfe), n=600, FE=o+d, on the frozen DGP::

    ppmlhdfe y x1 x2, absorb(o d) cluster(ca)      # x1=.0197737  x2=.01600301
    ppmlhdfe y x1 x2, absorb(o d) cluster(cb)      # x1=.02052703 x2=.02349358
    ppmlhdfe y x1 x2, absorb(o d) cluster(ca cb)   # x1=.01820187 x2=.01793071
"""

import numpy as np
import pandas as pd
import pytest

from statspai.regression.count import ppmlhdfe

# Frozen Stata ppmlhdfe references (see module docstring).
STATA_ONEWAY_CA = {"x1": 0.0197737, "x2": 0.01600301}
STATA_TWOWAY = {"x1": 0.01820187, "x2": 0.01793071}
_ATOL = 5e-7  # matches Stata's printed precision (SEs pinned to ~7 digits)


def _ppml_panel() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 600
    o = rng.integers(0, 8, n)
    d = rng.integers(0, 8, n)
    ca = rng.integers(0, 11, n)
    cb = rng.integers(0, 13, n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    mu = np.exp(1.0 + 0.4 * x1 - 0.3 * x2 + 0.1 * o - 0.1 * d)
    y = rng.poisson(mu)
    return pd.DataFrame(
        {"y": y, "x1": x1, "x2": x2, "o": o, "d": d, "ca": ca, "cb": cb}
    )


def test_ppmlhdfe_twoway_matches_stata() -> None:
    df = _ppml_panel()
    r = ppmlhdfe("y ~ x1 + x2 | o + d", data=df, cluster=["ca", "cb"])
    for v in ("x1", "x2"):
        assert np.isclose(float(r.std_errors[v]), STATA_TWOWAY[v], atol=_ATOL)
    # Two-way inference binds on the smaller cluster dimension.
    assert r.data_info["n_clusters"] == 11


def test_ppmlhdfe_oneway_unchanged() -> None:
    """The one-way path is untouched and still matches Stata exactly."""
    df = _ppml_panel()
    r = ppmlhdfe("y ~ x1 + x2 | o + d", data=df, cluster="ca")
    for v in ("x1", "x2"):
        assert np.isclose(float(r.std_errors[v]), STATA_ONEWAY_CA[v], atol=_ATOL)


def test_ppmlhdfe_twoway_requires_pair() -> None:
    df = _ppml_panel()
    with pytest.raises(Exception):
        ppmlhdfe("y ~ x1 + x2 | o + d", data=df, cluster=["ca", "cb", "o"])
