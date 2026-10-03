"""``sp.fast.fepois`` against R ``fixest::fepois``: variance kinds, weights, layouts.

Fixture: ``_fixtures/fast_fepois.csv`` (the ``fast_feols_weights`` panel
with an over-dispersed count outcome; 16 rows drop as all-zero firms on
both sides) and ``_fixtures/fast_fepois_R.json`` from
``_generate_fast_fepois_R.R`` (fixest 0.14.0, default ``ssc()``).

Twenty cells: one and two absorbed dimensions x no weights / weights x
``iid`` / ``hc1`` / clustered on a key that nests the firm effect (``g``),
on one that nests nothing (``c2``), and on an absorbed dimension
(``year``).

Three defects this caught (2026-10-03), none of which moved a coefficient:

* the small-sample factors were not fixest's, although the source said
  they were. ``iid`` used ``n/(n - p - Σ(G_k - 1))`` where fixest uses
  ``(n - 1)/(n - K)``; ``hc1`` was one degree of freedom short; and
  ``cr1`` charged every absorbed level even when the effects were nested
  in the clusters, which made clustered standard errors **8% too large**
  in the commonest design (unit effects, clusters of units);
* a weighted ``hc1`` fit left the observation weights out of the score:
  standard errors 43% to 46% off;
* the count for one absorbed dimension was ``G - 1`` rather than ``G``
  (shared with ``sp.fast.feols``).

Tolerance. Both sides iterate. With StatsPAI's tolerances tightened the
worst gap is 6.5e-8, which is fixest's own stopping rule; at StatsPAI's
defaults it is 1.2e-6. The first is asserted at 5e-7, the second at 5e-6.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

_FIX = pathlib.Path(__file__).parent / "_fixtures"
TIGHT = dict(tol=1e-13, fe_tol=1e-13, maxiter=200, fe_maxiter=20000)
RTOL_TIGHT = 5e-7
RTOL_DEFAULT = 5e-6
RTOL_COEF = 1e-10

_REF = json.loads((_FIX / "fast_fepois_R.json").read_text(encoding="utf-8"))
CELLS = sorted(k for k in _REF if k != "_meta")


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "fast_fepois.csv")


def _fit(data, key, **extra):
    fe, weights, kind = key.split("__")
    vcov = {"iid": dict(vcov="iid"), "hetero": dict(vcov="hc1")}.get(
        kind, dict(vcov="cr1", cluster=kind)
    )
    return sp.fast.fepois(
        "cnt ~ x1 + x2 | " + fe.replace("_", " + "),
        data,
        weights="w" if weights == "set" else None,
        **vcov,
        **extra,
    )


def _gaps(fit, cell):
    names = list(fit.coef_names)
    se = np.sqrt(np.diag(fit.vcov_matrix))
    b_gap = max(
        abs(float(fit.coef_vec[names.index(n)]) / cell[f"b_{n}"] - 1)
        for n in ("x1", "x2")
    )
    se_gap = max(
        abs(float(se[names.index(n)]) / cell[f"se_{n}"] - 1) for n in ("x1", "x2")
    )
    return b_gap, se_gap


@pytest.mark.parametrize("key", CELLS)
def test_matches_fixest(data, key):
    fit = _fit(data, key, **TIGHT)
    assert fit.n_kept == _REF[key]["nobs"]
    b_gap, se_gap = _gaps(fit, _REF[key])
    assert b_gap < RTOL_COEF
    assert se_gap < RTOL_TIGHT


@pytest.mark.parametrize("key", CELLS)
def test_default_tolerances_stay_within_the_iteration_budget(data, key):
    b_gap, se_gap = _gaps(_fit(data, key), _REF[key])
    assert b_gap < RTOL_COEF
    assert se_gap < RTOL_DEFAULT


def test_old_convention_is_still_reachable_and_is_not_fixest_s(data):
    """``ssc='statspai'``: same coefficients, clustered SEs 8% larger."""
    key = "firm__none__g"
    new = _fit(data, key, **TIGHT)
    old = _fit(data, key, ssc="statspai", **TIGHT)
    np.testing.assert_allclose(old.coef_vec, new.coef_vec, rtol=1e-12)
    ratio = np.sqrt(np.diag(old.vcov_matrix) / np.diag(new.vcov_matrix))
    n, p = new.n_kept, 2
    g = data["g"].nunique()
    firms = new.fe_cardinality[0]
    expected = np.sqrt((n - p - 1) / (n - p - (firms - 1)))
    np.testing.assert_allclose(ratio, expected, rtol=1e-10)
    assert expected > 1.07 and g == 25


def test_weighted_hc1_uses_the_weights_in_the_score(data):
    """Unit weights must reproduce the unweighted fit; they did, and still do.

    The defect showed only with non-constant weights, so the check is that
    doubling every weight leaves the HC1 standard errors unchanged (the
    bread scales by 1/2 twice, the meat by 4).
    """
    base = _fit(data, "firm__set__hetero", **TIGHT)
    doubled = sp.fast.fepois(
        "cnt ~ x1 + x2 | firm",
        data.assign(w=2 * data["w"]),
        weights="w",
        vcov="hc1",
        **TIGHT,
    )
    np.testing.assert_allclose(
        np.diag(doubled.vcov_matrix), np.diag(base.vcov_matrix), rtol=1e-9
    )


def test_unknown_ssc_is_refused(data):
    with pytest.raises(MethodIncompatibility, match="ssc"):
        sp.fast.fepois("cnt ~ x1 + x2 | firm", data, ssc="stata")
