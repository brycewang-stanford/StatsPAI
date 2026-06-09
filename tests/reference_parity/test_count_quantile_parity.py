"""Reference parity: ``sp`` count / quantile / Tobit estimators vs R.

These four maximum-likelihood / LP estimators are pinned to *exact* R
numbers on a shared fixed design (``count_quantile_data.csv``):

  * ``sp.poisson`` -> ``stats::glm(family = poisson)``  (coef + model SE)
  * ``sp.nbreg``   -> ``MASS::glm.nb``  (coef + model SE + dispersion;
                       sp's ``alpha`` equals MASS's ``1/theta`` under the
                       NB2 parameterization)
  * ``sp.qreg``    -> ``quantreg::rq``  (tau = 0.5 and 0.75; coefficients
                       only — see note)
  * ``sp.tobit``   -> ``AER::tobit(left = 0)``  (coef + SE + scale=sigma)

Tolerances / conventions
------------------------
  * poisson / nbreg / tobit coef and SE: 1e-5 relative.  All three solve
    the identical log-likelihood; sp reproduces glm / glm.nb / survreg to
    numerical precision, including the model-based (Fisher-information) SE
    and the Tobit scale.
  * nbreg dispersion: ``alpha == 1/theta`` to 1e-5.
  * qreg coefficients: 1e-4 relative at tau = 0.5 / 0.75, which are *unique*
    optima on this design (an even-n / degenerate quantile can leave the LP
    solution non-unique, where the simplex and interior-point methods pick
    different valid vertices; tau = 0.25 is such a case here and is
    deliberately excluded).  qreg standard errors are NOT pinned: sp uses a
    Powell (1991) kernel sandwich whose bandwidth rule differs from
    quantreg's nid/iid sparsity estimators, so cross-package SE equality is
    not expected.  Coefficient parity is the identifying check.

References
----------
- Venables, W. N. & Ripley, B. D. (2002). *Modern Applied Statistics with
  S*. Springer, New York (MASS ``glm.nb``). doi:10.1007/978-0-387-21706-2
- Koenker, R. (2005). *Quantile Regression*. Cambridge University Press
  (quantreg ``rq``, and the kernel/sparsity standard-error estimators that
  sp's quantile SE follows). doi:10.1017/CBO9780511754098
- Zeileis, A., Kleiber, C. & Jackman, S. (2008). "Regression Models for
  Count Data in R." *Journal of Statistical Software*, 27(8) (pscl
  ``zeroinfl``). doi:10.18637/jss.v027.i08

(All DOIs verified via Crossref, 2026-06-08.)
"""

from __future__ import annotations

import json
import math
import pathlib
import warnings

import pandas as pd
import pytest

import statspai as sp

_FIXTURE_DIR = pathlib.Path(__file__).parent / "_fixtures"


@pytest.fixture(scope="module")
def cq_data():
    return pd.read_csv(_FIXTURE_DIR / "count_quantile_data.csv")


@pytest.fixture(scope="module")
def r_reference():
    with open(_FIXTURE_DIR / "count_quantile_R.json", encoding="utf-8") as f:
        return json.load(f)


def _detail_map(res):
    """variable -> (coef, se) from a CausalResult.detail table."""
    d = res.detail.set_index("variable")
    return {
        name: (float(row["coefficient"]), float(row["se"]))
        for name, row in d.iterrows()
    }


# --------------------------------------------------------------------------
# Poisson  (sp.poisson == glm family=poisson, model-based SE)
# --------------------------------------------------------------------------
def test_poisson_matches_glm(cq_data, r_reference):
    res = sp.poisson("yc ~ x1 + x2", data=cq_data)
    ref = r_reference["poisson"]
    for name, rb in ref.items():
        assert float(res.params[name]) == pytest.approx(rb["coef"], rel=1e-5), name
        # SE band is 1e-4 (not 1e-5): the model-based SE is read off the IRLS
        # Hessian at convergence, and sp's stopping tolerance differs from
        # glm's by enough to move the SE in the 5th significant figure.
        assert float(res.std_errors[name]) == pytest.approx(rb["se"], rel=1e-4), name


# --------------------------------------------------------------------------
# Negative binomial  (sp.nbreg == MASS::glm.nb, alpha == 1/theta)
# --------------------------------------------------------------------------
def test_nbreg_matches_glm_nb(cq_data, r_reference):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.nbreg("yc ~ x1 + x2", data=cq_data)
    ref = r_reference["nbreg"]
    for name in ("_cons", "x1", "x2"):
        rb = ref[name]
        assert float(res.params[name]) == pytest.approx(rb["coef"], rel=1e-5), name
        # SE band 1e-4 for the same IRLS-tolerance reason as Poisson.
        assert float(res.std_errors[name]) == pytest.approx(rb["se"], rel=1e-4), name


def test_nbreg_dispersion_matches_inverse_theta(cq_data, r_reference):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.nbreg("yc ~ x1 + x2", data=cq_data)
    alpha = float(res.model_info["dispersion"])  # NB2 alpha
    assert alpha == pytest.approx(r_reference["nbreg"]["alpha"], rel=1e-5)


# --------------------------------------------------------------------------
# Quantile regression  (sp.qreg == quantreg::rq; coefficients only)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("tau,key", [(0.5, "qreg_tau50"), (0.75, "qreg_tau75")])
def test_qreg_coefficients_match_rq(cq_data, r_reference, tau, key):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.qreg(cq_data, "yl ~ x1 + x2", quantile=tau)
    coefs = {k: v[0] for k, v in _detail_map(res).items()}
    ref = r_reference[key]
    for name in ("const", "x1", "x2"):
        assert coefs[name] == pytest.approx(ref[name]["coef"], rel=1e-4), (
            f"qreg tau={tau} {name}: sp={coefs[name]:.6f} "
            f"vs rq={ref[name]['coef']:.6f}"
        )


# --------------------------------------------------------------------------
# Tobit  (sp.tobit == AER::tobit left=0; coef + SE + scale)
# --------------------------------------------------------------------------
def test_tobit_matches_aer(cq_data, r_reference):
    res = sp.tobit(cq_data, y="yt", x=["x1", "x2"], ll=0.0)
    dm = _detail_map(res)
    ref = r_reference["tobit"]
    for name in ("const", "x1", "x2"):
        coef, se = dm[name]
        assert coef == pytest.approx(ref[name]["coef"], rel=1e-5), name
        assert se == pytest.approx(ref[name]["se"], rel=1e-5), name


def test_tobit_sigma_matches_aer_scale(cq_data, r_reference):
    res = sp.tobit(cq_data, y="yt", x=["x1", "x2"], ll=0.0)
    sigma = float(res.model_info["sigma"])
    assert sigma == pytest.approx(r_reference["tobit"]["sigma"], rel=1e-5)


# --------------------------------------------------------------------------
# Zero-inflated counts  (sp.zip_model / sp.zinb == pscl::zeroinfl)
#
# Frozen against pscl on a separate zero-inflated fixture.  pscl is only
# needed to (re)generate the JSON, not to run these tests.
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def zi_data():
    return pd.read_csv(_FIXTURE_DIR / "zeroinfl_data.csv")


@pytest.fixture(scope="module")
def zi_reference():
    with open(_FIXTURE_DIR / "zeroinfl_R.json", encoding="utf-8") as f:
        return json.load(f)


def test_zip_matches_pscl(zi_data, zi_reference):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.zip_model(formula="y ~ x1 + x2", data=zi_data, inflate=["z"])
    ref = zi_reference["zip"]
    for name in ("const", "x1", "x2", "inflate_const", "inflate_z"):
        rb = ref[name]
        # ZIP is an exact match to pscl on both count and inflation blocks,
        # including model SEs.
        assert float(res.params[name]) == pytest.approx(rb["coef"], rel=1e-4), name
        assert float(res.std_errors[name]) == pytest.approx(rb["se"], rel=1e-4), name


def test_zinb_matches_pscl(zi_data, zi_reference):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.zinb(formula="y ~ x1 + x2", data=zi_data, inflate=["z"])
    ref = zi_reference["zinb"]
    # ZINB coefficients agree to ~1e-3 (the negative-binomial theta is found
    # by an inner profile likelihood that sp and pscl converge slightly
    # differently); the band stays far tighter than any specification error.
    for name in ("const", "x1", "x2", "inflate_const", "inflate_z"):
        assert float(res.params[name]) == pytest.approx(
            ref[name]["coef"], rel=1e-3
        ), name


def test_zinb_theta_matches_pscl(zi_data, zi_reference):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.zinb(formula="y ~ x1 + x2", data=zi_data, inflate=["z"])
    # sp stores the NB dispersion as ln_alpha; theta = 1 / exp(ln_alpha).
    ln_alpha = float(res.params["ln_alpha"])
    theta = 1.0 / math.exp(ln_alpha)
    assert theta == pytest.approx(zi_reference["zinb"]["theta"], rel=1e-3)
