"""``sp.fmm`` vs R ``flexmix`` 2.3-20 (secondary reference; Stata ``fmm`` is
the primary one, see ``test_fmm_stata_parity.py``).

Fixture: ``_fixtures/_generate_fmm_flexmix_R.R`` runs flexmix's EM from the
simulated class with ``tolerance = 1e-13`` on two of the Stata CSVs.

* ``pois2_lcprob`` (Poisson classes, concomitant multinomial logit in z):
  flexmix maximises the same likelihood. Log-likelihood to 1e-12 relative
  (observed 1.4e-14); coefficients to 5e-6 relative (observed 7e-7): flexmix
  stops on the change in the log-likelihood, and an EM run stopped at a
  log-likelihood change of 1e-13 still leaves the parameters about
  ``sqrt`` of that away from the maximum along the slow EM direction.
* ``gauss2_plain`` (Gaussian classes): a documented convention difference.
  flexmix's Gaussian M-step estimates the class standard deviation with the
  divisor ``(n - p) * sum(post_k) / n`` (``FLXMRglm``'s ``lm.wfit``-based
  ``sigma``), not ``sum(post_k)``, so its EM stops at a fixed point that is
  not the maximum likelihood estimate. The test pins that mechanism: at
  flexmix's estimates, our posterior probabilities reproduce flexmix's
  weighted least squares coefficients, its standard deviations under that
  divisor and its class shares (observed 2e-10), and our log-likelihood at
  flexmix's point equals flexmix's ``logLik`` (observed 1e-16 relative).
  The MLE that ``sp.fmm`` and Stata report has the larger log-likelihood.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

import statspai as sp
from statspai.regression import _fmm_core

_FIX = pathlib.Path(__file__).parent / "_fixtures"
REF = json.loads((_FIX / "fmm_flexmix_R.json").read_text(encoding="utf-8"))


def _coefs(comp: dict) -> np.ndarray:
    return np.array([comp["coef.(Intercept)"], comp["coef.x1"], comp["coef.x2"]])


def test_poisson_concomitant_matches_flexmix():
    ref = REF["pois2_lcprob"]
    d = pd.read_csv(_FIX / "fmm_pois2.csv")
    res = sp.fmm("y ~ x1 + x2", d, 2, family="poisson", class_covariates="z", seed=3)
    assert abs(res.loglik - ref["ll"]) <= 1e-12 * abs(ref["ll"])
    for k, comp in enumerate(ref["coef"], start=1):
        ours = res.params[
            [f"y:{k}.Class#{t}" for t in ("_cons", "x1", "x2")]
        ].to_numpy()
        np.testing.assert_allclose(ours, _coefs(comp), rtol=5e-6)
    cons, slope = ref["concomitant"]["2"]
    np.testing.assert_allclose(
        res.params[["2.Class:_cons", "2.Class:z"]].to_numpy(), [cons, slope], rtol=5e-6
    )


def test_gaussian_flexmix_is_the_fixed_point_of_its_df_corrected_m_step():
    ref = REF["gauss2_plain"]
    d = pd.read_csv(_FIX / "fmm_gauss2.csv")
    y = d["y"].to_numpy()
    X = np.column_stack([np.ones(len(d)), d["x1"], d["x2"]])
    n, p = X.shape
    B = np.column_stack([_coefs(c) for c in ref["coef"]])
    sigma = np.array([c["sigma"] for c in ref["coef"]])
    prior = np.array(ref["prior"])
    model = _fmm_core.MixtureModel(y, X, np.ones((n, 1)), 2, "gaussian", np.ones(n))
    theta = model.pack(np.log(prior / prior[0])[None, :], B, np.log(sigma))
    ll_i, _, post, _, _ = model.joint(theta)
    assert abs(ll_i.sum() - ref["ll"]) <= 1e-12 * abs(ref["ll"])
    for k in range(2):
        w = post[:, k]
        b = np.linalg.solve((X.T * w) @ X, (X.T * w) @ y)
        r = y - X @ b
        np.testing.assert_allclose(b, B[:, k], rtol=1e-8)
        s = np.sqrt(np.sum(w * r * r) * n / np.sum(w) / (n - p))
        np.testing.assert_allclose(s, sigma[k], rtol=1e-8)
        np.testing.assert_allclose(w.mean(), prior[k], rtol=1e-8)
    res = sp.fmm("y ~ x1 + x2", d, 2, seed=3)
    assert res.loglik > ref["ll"] + 1e-3
