"""``sp.fmm`` vs Stata 18 ``fmm`` (finite mixtures of regress / logit / poisson).

Fixtures: ``_fixtures/_generate_fmm_data.py`` writes four simulated CSVs with
well-separated classes (unique maximum up to relabelling);
``_fixtures/_generate_fmm_Stata.do`` fits them with Stata 18 ``fmm``
started from the simulated class, so Stata's class labels are StatsPAI's
default order (ascending class mean), with the convergence criteria
tightened to ``nrtolerance(1e-13)``. Cases: Gaussian with 2 and 3 classes,
logit and Poisson with 2, each with and without ``lcprob(z)``; OIM, robust
and cluster covariances; frequency and sampling weights.

Compared: ``e(ll)``, every coefficient and standard error (``e(b)``,
``e(V)``), AIC / BIC (``estat ic``), ``estat lcprob`` and ``estat lcmean``
(margin, delta-method SE, interval), and posterior / prior class
probabilities and class means of the first ten rows.

Tolerances.
* Default (Gaussian, Poisson, logit with ``lcprob``): log-likelihood 1e-12
  relative, coefficients and SEs 1e-6 relative. Both programs run Newton to
  the maximum; observed gaps are 1e-16 to 1e-8 (Stata's own stop on the
  Poisson cases).
* ``estat lcprob`` / ``estat lcmean``: 1e-6. Stata's ``margins`` takes the
  delta-method Jacobian by numerical differences; ours is a complex-step
  derivative. Observed 3e-7 to 2e-9.
* ``logit2_plain``: log-likelihood 1e-9, coefficients 1e-5, SEs 2e-5
  (observed 1.3e-10, 1.2e-6, 5.4e-6). Reference numerics, not a different
  estimator: Stata's ``e(ll)`` sits 3.6e-7 above the log-likelihood that
  Stata itself computes by hand at its own ``e(b)`` from ``predict``'s
  linear predictors and class probabilities (``ll_manual``), and that hand
  computation equals ours to 1e-15. The excess is concentrated on
  observations whose class linear predictor exceeds about 13 in absolute
  value, where ``fmm``'s internal logit density is inflated by up to 2e-8
  relative. Without class covariates this likelihood is very flat (SEs near
  1 on coefficients near 3), so that 3.6e-7 moves the maximiser by 1e-6.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
REF = json.loads((_FIX / "fmm_Stata.json").read_text(encoding="utf-8"))

CASES = {
    "gauss2_plain": ("gauss2", 2, "gaussian", "y ~ x1 + x2", {}),
    "gauss2_lcprob": (
        "gauss2",
        2,
        "gaussian",
        "y ~ x1 + x2",
        {"class_covariates": "z"},
    ),
    "gauss2_lcprob_robust": (
        "gauss2",
        2,
        "gaussian",
        "y ~ x1 + x2",
        {"class_covariates": "z", "vce": "robust"},
    ),
    "gauss2_cluster": ("gauss2", 2, "gaussian", "y ~ x1 + x2", {"cluster": "g"}),
    "gauss2_lcprob_fw": (
        "gauss2",
        2,
        "gaussian",
        "y ~ x1 + x2",
        {"class_covariates": "z", "weights": "fw"},
    ),
    "gauss2_lcprob_pw": (
        "gauss2",
        2,
        "gaussian",
        "y ~ x1 + x2",
        {"class_covariates": "z", "weights": "fw", "weights_as": "sampling"},
    ),
    "gauss3_plain": ("gauss3", 3, "gaussian", "y ~ x1", {}),
    "gauss3_lcprob": ("gauss3", 3, "gaussian", "y ~ x1", {"class_covariates": ["z"]}),
    "logit2_plain": ("logit2", 2, "logit", "y ~ x1 + x2", {}),
    "logit2_lcprob": ("logit2", 2, "logit", "y ~ x1 + x2", {"class_covariates": "z"}),
    "pois2_plain": ("pois2", 2, "poisson", "y ~ x1 + x2", {}),
    "pois2_lcprob": ("pois2", 2, "poisson", "y ~ x1 + x2", {"class_covariates": "z"}),
    "pois2_lcprob_cluster": (
        "pois2",
        2,
        "poisson",
        "y ~ x1 + x2",
        {"class_covariates": "z", "cluster": "g"},
    ),
}

# (ll, coefficients, SEs) relative tolerances; see the module docstring.
_DEFAULT = (1e-12, 1e-6, 1e-6)
_TOL = {"logit2_plain": (1e-9, 1e-5, 2e-5)}


def _ours(stata_name: str) -> str:
    """Stata ``e(b)`` column name -> ``sp.fmm`` parameter name."""
    name = stata_name.replace("1bn.", "1.")
    if name.startswith("/var("):
        return name
    eq, term = name.split(":", 1)
    if eq.endswith(".Class"):
        return f"{eq}:{term}"
    if "#c." in term:
        cls, x = term.split("#c.")
        return f"y:{cls}#{x}"
    return f"y:{term}#_cons"


_FITS: dict = {}


def _fit(key: str):
    if key not in _FITS:
        ds, k, fam, formula, kw = CASES[key]
        df = pd.read_csv(_FIX / f"fmm_{ds}.csv")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _FITS[key] = sp.fmm(formula, df, k, family=fam, seed=7, **kw)
    return _FITS[key]


def _rel(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


@pytest.mark.parametrize("key", sorted(CASES))
def test_loglik_coefficients_and_se(key):
    res, ref = _fit(key), REF[key]
    tol_ll, tol_b, tol_se = _TOL.get(key, _DEFAULT)
    assert abs(res.loglik - ref["ll"]) / abs(ref["ll"]) <= tol_ll
    names = ref["b"]["names"]
    b = np.asarray(ref["b"]["values"][0], float)
    se = np.sqrt(np.diag(np.asarray(ref["V"]["values"], float)))
    free = [i for i, n in enumerate(names) if not n.startswith("1b.Class:")]
    ours = [_ours(names[i]) for i in free]
    assert len(ours) == res.df_model == ref["rank"]
    assert _rel(res.params[ours], b[free]) <= tol_b
    assert _rel(res.std_errors[ours], se[free]) <= tol_se
    if not key.endswith("_fw"):
        assert res.n_obs == ref["N"]
    assert abs(res.aic - ref["aic"]) <= 2 * tol_ll * abs(ref["aic"])
    assert abs(res.bic - ref["bic"]) <= 2 * tol_ll * abs(ref["bic"])


@pytest.mark.parametrize("key", sorted(CASES))
def test_lcprob_lcmean(key):
    res, ref = _fit(key), REF[key]
    for ours, theirs in (
        (res.class_probs, ref["lcprob"]),
        (res.class_means, ref["lcmean"]),
    ):
        t = np.asarray(theirs["values"], float)  # rows b se z p ll ul ...
        got = ours[["estimate", "std_error", "conf_low", "conf_high"]].to_numpy().T
        assert _rel(got, t[[0, 1, 4, 5]]) <= 1e-6


@pytest.mark.parametrize("key", sorted(CASES))
def test_posterior_prior_and_class_means_by_row(key):
    res, ref = _fit(key), REF[key]
    k = res.k
    head = np.asarray(ref["head"]["values"], float)
    got = np.hstack(
        [
            res.posterior.to_numpy()[:10],
            res.prior.to_numpy()[:10],
            res.fitted.to_numpy()[:10],
        ]
    )
    assert got.shape == head.shape == (10, 3 * k)
    tol = 1e-5 if key == "logit2_plain" else 1e-6
    assert np.max(np.abs(got - head) / np.maximum(np.abs(head), 1e-3)) <= tol


@pytest.mark.parametrize("key", ["logit2_plain", "logit2_lcprob"])
def test_logit_likelihood_equals_stata_hand_computation(key):
    """Our log-likelihood at the maximum equals the one Stata computes by
    hand at its own estimates (``ll_manual``); ``fmm``'s ``e(ll)`` is the
    outlier (see the module docstring)."""
    res, ref = _fit(key), REF[key]
    assert abs(res.loglik - ref["ll_manual"]) <= 1e-12 * abs(ref["ll_manual"])
    # e(ll) is the larger one: fmm's logit density is inflated, not ours deflated
    assert 0.0 < ref["ll"] - ref["ll_manual"] < 1e-6


def test_frequency_weights_count_in_n():
    res = _fit("gauss2_lcprob_fw")
    assert res.n_ic == REF["gauss2_lcprob_fw"]["N"] == 1646
    assert _fit("gauss2_lcprob_pw").n_ic == REF["gauss2_lcprob_pw"]["N"] == 800
