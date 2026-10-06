"""Weighted Callaway-Sant'Anna aggregations on repeated cross-sections vs R ``did``.

The aggregation weights of ``sp.aggte`` are estimated cohort shares, and
their influence function (R ``did:::wif``) enters every aggregate that
mixes cohorts. Under observation weights that function is built from
``w_i * 1{G_i = g}``. The repeated-cross-section route (``panel=False``,
and ``allow_unbalanced_panel=True``) did not hand the weights to the
aggregation, so the share term used head counts while the shares
themselves used weight mass. Point estimates and every ATT(g, t) were
right; the SEs of the simple, event-study, calendar and group-overall
aggregates were off by up to 1.2e-4 in relative terms. The balanced-panel
route was not affected.

Found on 2026-10-06 in a three-way comparison with Stata ``csdid`` 2.0.0
(pre-release) and R ``did`` 2.5.1, which agree with each other to 1e-10.

Reference: R ``did`` 2.5.1 on ``_fixtures/csdid2_rc_cluster.csv`` with
weights ``0.5 + (id mod 7) / 7``, from
``_generate_cs_rc_weighted_aggte_R.R``.
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
REF = json.loads((_FIX / "cs_rc_weighted_aggte_R.json").read_text(encoding="utf-8"))

# Same estimator on the same bytes: the differences are floating-point
# only (observed <= 2e-12), so the gate is far below the 1e-6 parity
# budget and four orders of magnitude below the defect it guards.
RTOL = 1e-8


@pytest.fixture(scope="module")
def data():
    d = pd.read_csv(_FIX / "csdid2_rc_cluster.csv")
    d["w"] = 0.5 + (d["id"] % 7) / 7
    d["rid"] = np.arange(len(d))
    return d


def _fit(data, cluster):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.callaway_santanna(
            data,
            y="y",
            g="g",
            t="year",
            i="rid",
            weights="w",
            estimator="dr",
            control_group="nevertreated",
            base_period="universal",
            panel=False,
            clustervars=None if cluster == "none" else cluster,
        )


@pytest.mark.parametrize("cluster", ["none", "cy"])
@pytest.mark.parametrize("agg", ["simple", "dynamic", "group", "calendar"])
def test_weighted_rc_aggregates_match_did(data, cluster, agg):
    ref = REF[cluster][agg]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.aggte(_fit(data, cluster), type=agg, bstrap=False, cband=False)
    np.testing.assert_allclose(res.estimate, ref["overall_att"], rtol=RTOL)
    np.testing.assert_allclose(res.se, ref["overall_se"], rtol=RTOL)
    if agg == "simple":
        return
    got = res.detail.set_index(res.detail.columns[0])
    for key, att, se in zip(ref["egt"], ref["att"], ref["se"]):
        if se is None:  # the universal-base reference period
            continue
        np.testing.assert_allclose(got.loc[key, "att"], att, rtol=RTOL, atol=1e-12)
        np.testing.assert_allclose(got.loc[key, "se"], se, rtol=RTOL)


def test_fit_time_simple_att_uses_the_same_share_term(data):
    """The headline SE printed by the fit is the simple aggregate."""
    fit = _fit(data, "none")
    np.testing.assert_allclose(
        fit.estimate, REF["none"]["simple"]["overall_att"], rtol=RTOL
    )
    np.testing.assert_allclose(fit.se, REF["none"]["simple"]["overall_se"], rtol=RTOL)


def test_share_term_is_scale_invariant_in_the_weights(data):
    """Multiplying every weight by a constant changes nothing."""
    d2 = data.assign(w=data["w"] * 37.0)
    a = sp.aggte(_fit(data, "none"), type="calendar")
    b = sp.aggte(_fit(d2, "none"), type="calendar")
    np.testing.assert_allclose(a.se, b.se, rtol=1e-12)
