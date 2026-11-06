"""Proximal and negative-control estimators against R and Stata (T2).

Same bytes on every side: ``_fixtures/proximal_ref_data.csv`` (written by
``_fixtures/_generate_proximal_interference_data.py``). References:

* R 4.5.2, ``ivreg`` 0.6.7 and ``sandwich`` 3.1.1
  (``_fixtures/_generate_proximal_interference_R.R`` ->
  ``proximal_interference_R.json``);
* Stata 18 ``ivregress 2sls`` / ``regress``
  (``_fixtures/_generate_proximal_interference_Stata.do`` ->
  ``proximal_interference_Stata.json``).

Both are run as black boxes (GPL / proprietary). What is compared:

* ``sp.proximal`` is the linear outcome-bridge 2SLS: ``y`` on
  ``(1, d, X, W)`` with instruments ``(1, d, X, Z)``, classical covariance
  ``RSS / (n - k) (X_hat' X_hat)^-1``. That is ``ivreg``'s default and
  Stata's ``ivregress 2sls, small``. Just identified, over identified,
  two outcome proxies, and no covariates.
* ``sp.negative_control_outcome`` / ``sp.negative_control_exposure`` are
  OLS with HC1 (``vcovHC(type = "HC1")``, ``regress, vce(robust)``).
* ``sp.double_negative_control`` is the 2SLS of ``y`` on ``(d, nco, X)``
  with ``(d, nce, X)`` and HC1 (``ivregress 2sls, vce(robust) small``).

The errors are heteroskedastic so HC1 and classical differ; the tolerance
is 1e-9 relative on the estimate and the standard error (observed at most
2.3e-14 against R and 6e-15 against Stata, i.e. rounding).

Not compared here, and why: ``sp.fortified_pci``, ``sp.bidirectional_pci``
and ``sp.pci_mtp`` average or rescale bridge estimates and report
bootstrap standard errors, for which no reference package exists;
``sp.proximal_regression`` combines a 2SLS bridge with a penalised
(``C = 1e6``) logistic weight in a form no package implements;
``sp.select_pci_proxies`` is a ranking heuristic. Their evidence stays the
known-truth tests in ``test_proximal_parity.py`` and
``test_frontier_known_truth_recovery.py``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
import pytest

import statspai as sp

_FIX = Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "proximal_interference_R.json").read_text(encoding="utf-8"))
STATA = json.loads(
    (_FIX / "proximal_interference_Stata.json").read_text(encoding="utf-8")
)
DATA = pd.read_csv(_FIX / "proximal_ref_data.csv")
RTOL = 1e-9


def _check(est: float, se: float, r_ref: dict, stata_ref: list) -> None:
    assert est == pytest.approx(r_ref["estimate"], rel=RTOL)
    assert se == pytest.approx(r_ref["se"], rel=RTOL)
    assert est == pytest.approx(stata_ref[0], rel=RTOL)
    assert se == pytest.approx(stata_ref[1], rel=RTOL)


@pytest.mark.parametrize(
    "case,proxy_z,proxy_w,covariates",
    [
        ("just_identified", ["z1"], ["w1"], ["x1", "x2"]),
        ("over_identified", ["z1", "z2"], ["w1"], ["x1"]),
        ("two_w", ["z1", "z2"], ["w1", "w2"], ["x1", "x2"]),
        ("no_covariates", ["z1"], ["w1"], None),
    ],
)
def test_proximal_equals_ivreg_and_ivregress(case, proxy_z, proxy_w, covariates):
    # two outcome proxies: the first-stage F is not defined and a
    # RuntimeWarning says so; the estimate is unaffected
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = sp.proximal(
            DATA,
            y="y",
            treat="d",
            proxy_z=proxy_z,
            proxy_w=proxy_w,
            covariates=covariates,
        )
    _check(res.estimate, res.se, R["proximal"][case], STATA[f"proximal_{case}"])
    assert res.model_info["se_method"] == "2sls_sandwich"


@pytest.mark.parametrize(
    "case,covariates", [("covariates", ["x1", "x2"]), ("bare", None)]
)
def test_negative_control_outcome_equals_ols_hc1(case, covariates):
    res = sp.negative_control_outcome(DATA, nco="nco", treat="d", covariates=covariates)
    _check(res.estimate, res.se, R["nco"][case], STATA[f"nco_{case}"])


@pytest.mark.parametrize("case,covariates", [("covariates", ["x1"]), ("bare", None)])
def test_negative_control_exposure_equals_ols_hc1(case, covariates):
    res = sp.negative_control_exposure(DATA, y="y", nce="nce", covariates=covariates)
    _check(res.estimate, res.se, R["nce"][case], STATA[f"nce_{case}"])


@pytest.mark.parametrize(
    "case,covariates", [("covariates", ["x1", "x2"]), ("bare", None)]
)
def test_double_negative_control_equals_2sls_hc1(case, covariates):
    res = sp.double_negative_control(
        DATA, y="y", treat="d", nce="nce", nco="nco", covariates=covariates
    )
    _check(res.estimate, res.se, R["dnc"][case], STATA[f"dnc_{case}"])
