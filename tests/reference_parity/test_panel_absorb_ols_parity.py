"""``sp.absorb_ols`` (the HDFE engine behind ``sp.hdfe_ols``) against
``fixest::feols`` and Stata ``reghdfe``.

References
----------
* Track A goldens, read-only: ``tests/r_parity/results/03_hdfe_R.json`` /
  ``15_hdfe_cluster_R.json`` (fixest 0.14.0) and
  ``tests/stata_parity/results/03_hdfe_Stata.json`` /
  ``15_hdfe_cluster_Stata.json`` (reghdfe), on the committed
  ``tests/r_parity/data/03_hdfe.csv`` / ``15_hdfe_cluster.csv`` bytes --
  two-way FE, iid and firm-clustered.  ``sp.absorb_ols`` is called directly
  (the Track A modules call ``sp.fast.feols`` / ``sp.hdfe_ols``).
* ``_fixtures/panel_glmm_Stata.json`` (``reghdfe_*``) --
  ``_generate_panel_glmm_stata.do`` on ``_fixtures/absorb_ols_data.csv``:
  six singleton firms (dropped by both sides), ``[aw=w]`` analytic weights,
  one- and two-way clustering, reghdfe (SSC) in a private ado directory.

Conventions
-----------
* iid: σ̂² = RSS / (n − p − Σ_k G_k + (K − 1)).
* one-way cluster CR1: G/(G−1)·(n−1)/(n−K) with FE nested in the cluster
  not charged to K.
* two-way cluster: inclusion-exclusion.  reghdfe scales *every* term by
  G_min/(G_min − 1); ``absorb_ols``'s default scales each term by its own
  G/(G − 1) (R ``sandwich::vcovCL`` and ``sp.multiway_cluster_vcov``).
  ``cluster_df='min'`` (added in this version) reproduces reghdfe; the
  default differs by 1.8e-4 / 1.2e-2 in variance on this sample, pinned
  below as the size of that convention gap.

Tolerance: coefficients 1e-12, iid SEs 1e-12, clustered SEs 1e-10 (observed
<= 5.6e-11: residual differences of order 1e-15 summed within cluster and
squared in the meat -- the same mechanism as the ``hdfe_ols`` alias proof).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

ROOT = Path(__file__).resolve().parents[2]
_FIX = Path(__file__).parent / "_fixtures"
S = json.loads((_FIX / "panel_glmm_Stata.json").read_text(encoding="utf-8"))


def _golden(path):
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    return {r["statistic"]: r for r in rows}


@pytest.mark.parametrize("module", ["03_hdfe", "15_hdfe_cluster"])
@pytest.mark.parametrize("side", ["R", "Stata"])
def test_matches_track_a_goldens(module, side):
    d = pd.read_csv(ROOT / "tests" / "r_parity" / "data" / f"{module}.csv")
    base = "r_parity" if side == "R" else "stata_parity"
    ref = _golden(ROOT / "tests" / base / "results" / f"{module}_{side}.json")
    kw = {"cluster": d["firm"].to_numpy()} if module == "15_hdfe_cluster" else {}
    out = sp.absorb_ols(
        d["y"].to_numpy(),
        d[["x1", "x2"]].to_numpy(),
        d[["firm", "year"]],
        tol=1e-12,
        **kw,
    )
    se_rtol = 1e-10 if kw else 1e-12
    for i, v in enumerate(("x1", "x2")):
        np.testing.assert_allclose(
            out["coef"][i], ref[f"beta_{v}"]["estimate"], rtol=1e-12
        )
        np.testing.assert_allclose(out["se"][i], ref[f"beta_{v}"]["se"], rtol=se_rtol)


D = pd.read_csv(_FIX / "absorb_ols_data.csv")


def _fit(**kw):
    return sp.absorb_ols(
        D["y"].to_numpy(),
        D[["x1", "x2"]].to_numpy(),
        D[["firm", "year"]],
        tol=1e-12,
        **kw,
    )


@pytest.mark.parametrize(
    "spec,kw",
    [
        ("reghdfe_iid", {}),
        ("reghdfe_cl_firm", {"cluster": "firm"}),
        ("reghdfe_aw_iid", {"weights": True}),
        ("reghdfe_aw_cl_firm", {"weights": True, "cluster": "firm"}),
        ("reghdfe_aw_cl_firm_year", {"weights": True, "cluster": ["firm", "year"]}),
    ],
)
def test_matches_reghdfe(spec, kw):
    args = {}
    if kw.get("weights"):
        args["weights"] = D["w"].to_numpy()
    cl = kw.get("cluster")
    if isinstance(cl, str):
        args["cluster"] = D[cl].to_numpy()
    elif isinstance(cl, list):
        args["cluster"] = [D[c].to_numpy() for c in cl]
        args["cluster_df"] = "min"
    out = _fit(**args)
    ref = S[spec]
    assert out["n"] == ref["N"] == 1194
    assert out["n_singletons_dropped"] == 6
    se_rtol = 1e-10 if cl else 1e-12
    for i, v in enumerate(("x1", "x2")):
        np.testing.assert_allclose(out["coef"][i], ref["b"][v], rtol=1e-12)
        np.testing.assert_allclose(out["se"][i], ref["se"][v], rtol=se_rtol)


def test_default_multiway_factor_is_per_term():
    per = _fit(
        weights=D["w"].to_numpy(), cluster=[D["firm"].to_numpy(), D["year"].to_numpy()]
    )
    ref = S["reghdfe_aw_cl_firm_year"]["se"]
    rel = np.abs(per["se"] ** 2 / np.array([ref["x1"], ref["x2"]]) ** 2 - 1.0)
    np.testing.assert_allclose(rel, [1.76e-4, 1.229e-2], rtol=2e-2)


def test_one_way_is_invariant_to_cluster_df():
    a = _fit(cluster=D["firm"].to_numpy())
    b = _fit(cluster=D["firm"].to_numpy(), cluster_df="min")
    np.testing.assert_array_equal(a["se"], b["se"])
