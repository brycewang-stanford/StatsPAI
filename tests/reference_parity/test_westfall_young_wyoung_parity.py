"""``sp.westfall_young`` against Stata ``wyoung`` (Reif, SSC 2025-01-10).

Two designs from ``_fixtures/_generate_wyoung_data.py``: a stratified
experiment (two strata of 8, ``i.stratum`` effects, HC1) and a cluster-
randomized one (12 clusters of 5, CR1). ``_generate_wyoung_Stata.do`` runs
``wyoung ..., permute(treat)`` with ``reps(20000)``.

* Coefficients, t-statistics and the unadjusted (analytic) p-values are
  deterministic and agree to 1e-12 (T2).
* The adjusted p-values are a Monte Carlo estimate in ``wyoung`` (20,000
  random re-randomizations) and exact here (all 4,900 / 924 assignments are
  enumerated). They agree within 4 Monte Carlo standard errors of wyoung's
  estimate (S-level screen; observed within 1 SE).

wyoung's licence is not stated; it was used as a black box (outputs only).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
REF = json.loads((_FIX / "wyoung_Stata.json").read_text(encoding="utf-8"))

DESIGNS = {
    "strata": dict(file="wyoung_strata.csv", kw=dict(strata="stratum")),
    "cluster": dict(file="wyoung_cluster.csv", kw=dict(cluster="cl")),
}


@pytest.fixture(scope="module", params=sorted(DESIGNS))
def fit(request):
    name = request.param
    spec = DESIGNS[name]
    data = pd.read_csv(_FIX / spec["file"])
    out = sp.westfall_young(
        data,
        y=["y1", "y2", "y3"],
        treat="treat",
        covariates=["x"],
        n_perms=100_000,
        **spec["kw"],
    )
    return name, out, np.array(REF[name], dtype=float)


def test_enumerates_every_assignment(fit):
    name, out, _ = fit
    assert out.attrs["exact"] is True
    assert out.attrs["n_perms"] == {"strata": 4900, "cluster": 924}[name]


def test_coefficients_t_and_analytic_p_match_wyoung(fit):
    _, out, ref = fit
    coef, se, p = ref[:, 0], ref[:, 1], ref[:, 2]
    np.testing.assert_allclose(out["coef"], coef, rtol=1e-12)
    np.testing.assert_allclose(out["t"], coef / se, rtol=1e-12)
    np.testing.assert_allclose(out["p_analytic"], p, rtol=1e-12)


def test_adjusted_p_within_monte_carlo_error_of_wyoung(fit):
    _, out, ref = fit
    p_stata = ref[:, 3]
    mc_se = np.sqrt(np.maximum(p_stata * (1 - p_stata), 1e-4) / REF["reps"])
    assert np.all(np.abs(out["p_wy"].to_numpy() - p_stata) <= 4 * mc_se)


def test_translation_runs_the_same_call():
    data = pd.read_csv(_FIX / "wyoung_strata.csv")
    line = (
        "wyoung y1 y2 y3, cmd(regress OUTCOMEVAR treat x i.stratum, robust) "
        "familyp(treat) permute(treat) strata(stratum) reps(20000) seed(12345)"
    )
    tr = sp.from_stata(line)
    assert tr["ok"] and tr["tool"] == "westfall_young"
    out = sp.westfall_young(data, **tr["arguments"])
    ref = np.array(REF["strata"], dtype=float)
    np.testing.assert_allclose(out["p_analytic"], ref[:, 2], rtol=1e-12)
