"""Callaway-Sant'Anna on repeated cross-sections vs Stata ``csdid``.

A replication ran ``csdid2 y, time() gvar() cluster(city#year) method(dripw)
agg(group)`` -- no ``ivar()``, so the firm panel is treated as repeated
cross-sections and the cluster may vary within firm. StatsPAI refused a
time-varying cluster, required the bootstrap for any cluster, and its
aggregation ignored csdid's estimated treated-count weights.

Reference: Stata 18 ``csdid`` (v1) and ``csdid2`` on
``_fixtures/csdid2_rc_cluster.csv`` (unbalanced firm panel used as repeated
cross-sections, cluster ``cy`` = city x year), from
``_generate_csdid_rc_cluster_Stata.do``.

* ``csdid`` and StatsPAI agree to 1e-10 on every ATT(g, t), its analytic SE
  with and without ``cluster(cy)``, the cohort ATT(g) with their SEs and
  the simple ATT (``agg_weights='csdid'``, whose SEs include the influence
  of csdid's estimated cell weights). The cell SEs also equal R
  ``DRDID::drdid_rc`` (checked by hand, 0.4102878 on the first cell).
* ``csdid2``'s point estimates are the same; its SEs are not. For control
  rows it scales the influence function by the whole 2x2 subsample's share
  instead of the control cell's (``(y - ybar) N / n_sub`` against ``(y -
  ybar) N / n_cell``), which understates the control-side variance. That
  is a documented divergence (T4), not a target: ``csdid`` and ``DRDID``
  are two independent references that agree with StatsPAI. On the
  replicated paper's data StatsPAI returns ``csdid``'s 0.033423408 where
  the paper, run with ``csdid2``, reports 0.0328111.
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
REF = json.loads((_FIX / "csdid_rc_cluster_Stata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "csdid2_rc_cluster.csv")


def _fit(data, cluster):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.callaway_santanna(
            data,
            y="y",
            g="g",
            t="year",
            i="id",
            estimator="dr",
            base_period="universal",
            panel=False,
            clustervars=None if cluster == "none" else cluster,
        )


@pytest.mark.parametrize("cluster", ["none", "cy"])
def test_cells_match_csdid(data, cluster):
    r = _fit(data, cluster)
    ref = REF[cluster]
    np.testing.assert_allclose(r.detail["att"], ref["cell_att"], rtol=1e-10)
    np.testing.assert_allclose(r.detail["se"], ref["cell_se"], rtol=1e-9)


@pytest.mark.parametrize("cluster", ["none", "cy"])
def test_aggregates_match_csdid(data, cluster):
    r = _fit(data, cluster)
    ref = REF[cluster]
    g = sp.aggte(r, type="group", agg_weights="csdid", share_variance=False)
    np.testing.assert_allclose(g.estimate, ref["group_att"][0], rtol=1e-10)
    np.testing.assert_allclose(g.se, ref["group_se"][0], rtol=1e-9)
    np.testing.assert_allclose(g.detail["att"], ref["group_att"][1:], rtol=1e-10)
    np.testing.assert_allclose(g.detail["se"], ref["group_se"][1:], rtol=1e-9)
    s = sp.aggte(r, type="simple", agg_weights="csdid", share_variance=False)
    np.testing.assert_allclose(s.estimate, ref["simple_att"], rtol=1e-10)
    np.testing.assert_allclose(s.se, ref["simple_se"], rtol=1e-9)


def test_csdid2_point_estimates_agree_and_its_se_is_smaller(data):
    """Documents the csdid2 divergence rather than chasing it."""
    r = _fit(data, "cy")
    g = sp.aggte(r, type="group", agg_weights="csdid", share_variance=False)
    ref = REF["cy"]
    np.testing.assert_allclose(g.estimate, ref["csdid2_group_att"][0], rtol=1e-10)
    assert not np.isclose(g.se, ref["csdid2_group_se"][0], rtol=1e-3)


def test_time_varying_cluster_needs_panel_false(data):
    with pytest.raises(sp.MethodIncompatibility, match="panel=False"):
        sp.callaway_santanna(
            data,
            y="y",
            g="g",
            t="year",
            i="id",
            clustervars="cy",
            allow_unbalanced_panel=True,
        )


# ---------------------------------------------------------------------------
# Panel (ivar given): time-invariant cluster, analytic SEs.
# ---------------------------------------------------------------------------
PANEL = json.loads(
    (_FIX / "csdid_panel_cluster_Stata.json").read_text(encoding="utf-8")
)


def test_panel_clustered_analytic_ses_match_csdid():
    """``csdid y, ivar() cluster(cl)`` on mpdta with a county-group cluster.

    The fixture is ``sp.datasets.mpdta()`` with ``cl = countyreal mod 23 + 1``
    (``_generate_csdid_panel_cluster_Stata.do``). csdid's group average holds
    the cohort shares fixed (``share_variance=False``) while its simple ATT
    carries the cohort-share term (``share_variance=True``), as documented
    for ``sp.aggte``.
    """
    m = pd.read_csv(_FIX / "mpdta_cluster.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.callaway_santanna(
            m,
            y="lemp",
            g="first_treat",
            t="year",
            i="countyreal",
            estimator="dr",
            base_period="universal",
            clustervars="cl",
        )
    k = len(r.detail)
    np.testing.assert_allclose(r.detail["att"], PANEL["cell_att"][:k], rtol=1e-9)
    np.testing.assert_allclose(r.detail["se"], PANEL["cell_se"][:k], rtol=1e-9)
    g = sp.aggte(r, type="group", share_variance=False)
    np.testing.assert_allclose(g.estimate, PANEL["group_att"][0], rtol=1e-9)
    np.testing.assert_allclose(g.se, PANEL["group_se"][0], rtol=1e-9)
    np.testing.assert_allclose(g.detail["se"], PANEL["group_se"][1:], rtol=1e-9)
    s = sp.aggte(r, type="simple", share_variance=True)
    np.testing.assert_allclose(s.estimate, PANEL["simple_att"], rtol=1e-9)
    np.testing.assert_allclose(s.se, PANEL["simple_se"], rtol=1e-9)
