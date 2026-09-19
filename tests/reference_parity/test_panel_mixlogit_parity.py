"""``sp.mixlogit`` against Stata ``mixlogit`` (Hole) on *identical* Halton draws.

Reference
---------
``_fixtures/panel_glmm_Stata.json`` (``mixlogit_*``) --
``_generate_panel_glmm_stata.do``: SSC ``mixlogit`` 1.4.0 in a private ado
directory, ``group(cs) id(pid) rand(quality comfort) nrep(50) burn(15)``,
tolerance 1e-10 / ltolerance 1e-12.  Data ``_fixtures/mixlogit_data.csv``
(150 individuals x 4 choice situations x 3 alternatives).

Why this is a deterministic comparison, not a Monte-Carlo one
-------------------------------------------------------------
A simulated likelihood is only comparable across implementations when the
draws are the same.  Stata ``mixlogit`` gives individual n the standard-normal
draws ``invnormal(halton(nrep, krnd, 1 + burn + nrep*(n-1)))`` -- prime
bases 2, 3, ..., ``burn`` leading points dropped, consecutive blocks per
individual, no scrambling.  ``sp.mixlogit(n_draws=50, halton_burn=15,
halton_shift=False)`` builds exactly that matrix, so the two maximise the
same function: the log-likelihoods agree to ~1e-12 and the estimates to the
optimisers' precision.  With the default (randomly shifted) draws the two
are different simulators of the same integral and would agree only within
simulation error.

Conventions
-----------
* Default SEs in Stata are ``oim`` (inverse Hessian) -> ``robust=False``;
  Stata's ``robust`` is the individual-level sandwich times N/(N−1)
  -> ``robust=True`` (``small_sample=True``, added in this version; without
  it the robust variance was 0.67% = 1/149 below Stata's).
* The standard deviations / Cholesky diagonal enter through their absolute
  value; both sides report the positive representative (before this version
  StatsPAI could report e.g. ``sd_quality = -0.671``).
* ``ln(1)``: the last random coefficient (comfort) is lognormal,
  β = exp(m + s z) -- ``random_dist={'comfort': 'lognormal'}``.
* ``corr``: Cholesky L with Σ = LL'; compared on Σ (invariant to the sign
  of each column) and on the means.

Tolerance: rtol 1e-6 on means, SDs, Σ and SEs (observed <= 2.2e-7: BFGS on
a finite-difference gradient here, Newton-Raphson in Stata); 1e-10 on the
log-likelihood.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = Path(__file__).parent / "_fixtures"
S = json.loads((_FIX / "panel_glmm_Stata.json").read_text(encoding="utf-8"))
D = pd.read_csv(_FIX / "mixlogit_data.csv")
STATA_DRAWS = dict(n_draws=50, halton_burn=15, halton_shift=False)
BASE = dict(
    y="chosen",
    chid="cs",
    x_fixed=["price"],
    x_random=["quality", "comfort"],
    panel_id="pid",
    tol=1e-10,
    maxiter=1000,
    **STATA_DRAWS,
)
RTOL = 1e-6
NAMES = [
    ("price", "Mean:price"),
    ("mean_quality", "Mean:quality"),
    ("mean_comfort", "Mean:comfort"),
    ("sd_quality", "SD:quality"),
    ("sd_comfort", "SD:comfort"),
]


def _fit(**kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.mixlogit(D, **{**BASE, **kw})


@pytest.mark.parametrize(
    "robust,spec", [(False, "mixlogit_oim"), (True, "mixlogit_robust")]
)
def test_matches_stata_mixlogit(robust, spec):
    r = _fit(robust=robust)
    ref = S[spec]
    for ours, theirs in NAMES:
        np.testing.assert_allclose(r.params[ours], ref["b"][theirs], rtol=RTOL)
        np.testing.assert_allclose(r.std_errors[ours], ref["se"][theirs], rtol=RTOL)
    np.testing.assert_allclose(r.model_info["log_likelihood"], ref["ll"], rtol=1e-10)
    assert r.model_info["converged"]


def test_lognormal_matches_stata():
    r = _fit(robust=False, random_dist={"comfort": "lognormal"})
    ref = S["mixlogit_lognormal"]
    for ours, theirs in NAMES:
        np.testing.assert_allclose(r.params[ours], ref["b"][theirs], rtol=RTOL)
        np.testing.assert_allclose(r.std_errors[ours], ref["se"][theirs], rtol=RTOL)
    np.testing.assert_allclose(r.model_info["log_likelihood"], ref["ll"], rtol=1e-10)


def test_correlated_matches_stata_on_sigma():
    r = _fit(robust=False, correlated=True)
    ref = S["mixlogit_corr"]["b"]

    def sigma(l11, l21, l22):
        L = np.array([[l11, 0.0], [l21, l22]])
        return L @ L.T

    ours = sigma(r.params["L[0,0]"], r.params["L[1,0]"], r.params["L[1,1]"])
    theirs = sigma(ref["l11:_cons"], ref["l21:_cons"], ref["l22:_cons"])
    np.testing.assert_allclose(ours, theirs, rtol=RTOL)
    for ours_nm, theirs_nm in NAMES[:3]:
        np.testing.assert_allclose(r.params[ours_nm], ref[theirs_nm], rtol=RTOL)
    np.testing.assert_allclose(
        r.model_info["log_likelihood"], S["mixlogit_corr"]["ll"], rtol=1e-10
    )


def test_robust_small_sample_factor():
    a = _fit(robust=True)
    b = _fit(robust=True, small_sample=False)
    np.testing.assert_allclose(a.std_errors**2 / b.std_errors**2, 150 / 149, rtol=1e-12)


def test_standard_deviations_reported_positive_by_default():
    r = sp.mixlogit(D, **{k: v for k, v in BASE.items() if k not in STATA_DRAWS})
    assert (r.params[["sd_quality", "sd_comfort"]] > 0).all()
