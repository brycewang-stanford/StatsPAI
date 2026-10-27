"""``sp.akm`` against pytwoway 0.3.21 (AKM, Andrews et al., KSS).

Reference: ``_fixtures/akm_twoway_pytwoway.json`` from
``_generate_akm_twoway_pytwoway.py`` (pytwoway / bipartitepandas, MIT,
run in a throwaway environment) on ``_fixtures/akm_twoway.csv``.

What is compared, and at which tolerance:

* the leave-one-observation-out connected set: the same 1,727 rows;
* plug-in components, ``var(psi)`` and ``cov`` under the homoskedastic
  correction, the KSS ``var(psi)`` and residual variance, the leverage
  range and the fitted effects: exact computations on both sides, so
  ``1e-10`` (pytwoway's effects come from MINRES at ``1e-10``; everything
  else is a closed form);
* the homoskedastic ``var(alpha)``: pytwoway's exact trace adds the
  diagonal block ``(W'W)^{-1}`` as a vector (``Dwinv + C @ Minv @ B`` with a
  1-D ``Dwinv`` broadcasts along rows), which drops exactly
  ``(n_workers - 1) / n`` from the trace; the identity
  ``ours = pytwoway - sigma2_ho * (n_workers - 1) / n`` is asserted to
  ``1e-12``;
* KSS ``var(alpha)`` and ``cov`` (whose exact pytwoway traces carry the same
  block) and all homoskedastic components: against pytwoway's stochastic
  (Hutchinson, 4000 draws) traces, within 4 Monte Carlo standard errors
  (the stored per-draw SDs over sqrt(4000)). A stochastic screen, not an
  exact match; the exact identities with dense ``(X'X)^{-1}`` are in
  ``tests/test_akm.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FIX = Path(__file__).parent / "_fixtures"


@pytest.fixture(scope="module")
def ref():
    return json.loads((FIX / "akm_twoway_pytwoway.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fit():
    d = pd.read_csv(FIX / "akm_twoway.csv")
    return sp.akm(d, "lnw", "worker", "firm", variance="kss", leverage="exact")


def test_leave_one_out_set_matches(fit, ref):
    rows = ref["rows"]
    theirs = set(zip(rows["worker"], rows["firm"], rows["year"]))
    d = pd.read_csv(FIX / "akm_twoway.csv").loc[fit.data.index]
    ours = set(zip(d.worker, d.firm, d.year))
    assert ours == theirs
    assert fit.sample["n_obs"] == ref["n_obs"] == 1727
    assert fit.sample["n_workers"] == ref["n_workers"]
    assert fit.sample["n_firms"] == ref["n_firms"]
    assert fit.sample["n_movers"] == ref["n_movers"]


def test_effects_match_up_to_normalisation(fit, ref):
    r = pd.DataFrame(ref["rows"])
    d = pd.read_csv(FIX / "akm_twoway.csv").loc[fit.data.index]
    ours = d.assign(a=fit.data["alpha"].to_numpy(), p=fit.data["psi"].to_numpy())
    m = ours.merge(r, on=["worker", "firm", "year"])
    assert len(m) == 1727
    # alpha + psi is normalisation free; psi up to one constant
    np.testing.assert_allclose(m.a + m.p, m.alpha + m.psi, atol=1e-10)
    assert np.ptp(m.p - m.psi) < 1e-10


def test_exact_components(fit, ref):
    e = ref["exact"]
    dec = fit.decomposition
    pairs = [
        ("plugin", "var_y", "var(y)"),
        ("plugin", "var_alpha", "var(alpha)_fe"),
        ("plugin", "var_psi", "var(psi)_fe"),
        ("plugin", "cov", "cov(psi, alpha)_fe"),
        ("plugin", "var_resid", "var(eps)_fe"),
        ("homoskedastic", "var_psi", "var(psi)_ho"),
        ("homoskedastic", "cov", "cov(psi, alpha)_ho"),
        ("homoskedastic", "var_resid", "var(eps)_ho"),
        ("kss", "var_psi", "var(psi)_he"),
        ("kss", "var_resid", "var(eps)_he"),
    ]
    for col, row, key in pairs:
        assert dec.loc[row, col] == pytest.approx(e[key], abs=1e-10), (col, row)
    assert fit.leverage["min_P"] == pytest.approx(e["min_lev"], abs=1e-12)
    assert fit.leverage["max_P"] == pytest.approx(e["max_lev"], abs=1e-12)


def test_homoskedastic_var_alpha_differs_by_the_missing_diagonal(fit, ref):
    e = ref["exact"]
    n, n_w = fit.sample["n_obs"], fit.sample["n_workers"]
    sigma2 = fit.decomposition.loc["var_resid", "homoskedastic"]
    ours = fit.decomposition.loc["var_alpha", "homoskedastic"]
    assert ours == pytest.approx(e["var(alpha)_ho"] - sigma2 * (n_w - 1) / n, abs=1e-12)


@pytest.mark.parametrize(
    "col, row, key, sd_key, scale",
    [
        ("kss", "var_alpha", "var(alpha)_he", "tr_var(alpha)_he_sd", None),
        ("kss", "var_psi", "var(psi)_he", "tr_var(psi)_he_sd", None),
        ("kss", "cov", "cov(psi, alpha)_he", "tr_cov(psi, alpha)_he_sd", None),
        ("homoskedastic", "var_alpha", "var(alpha)_ho", "tr_var(alpha)_ho_sd", "ho"),
        ("homoskedastic", "var_psi", "var(psi)_ho", "tr_var(psi)_ho_sd", "ho"),
        (
            "homoskedastic",
            "cov",
            "cov(psi, alpha)_ho",
            "tr_cov(psi, alpha)_ho_sd",
            "ho",
        ),
    ],
)
def test_corrected_components_within_hutchinson_error(
    fit, ref, col, row, key, sd_key, scale
):
    h = ref["hutchinson"]
    # Monte Carlo SE of the mean of the per-draw traces; the homoskedastic
    # correction multiplies the trace by sigma^2
    mc_se = h[sd_key] / np.sqrt(h["ndraw_trace"])
    if scale == "ho":
        mc_se *= h["var(eps)_ho"]
    assert abs(fit.decomposition.loc[row, col] - h[key]) <= 4 * mc_se
