"""Reference parity: analytic cluster-robust SEs of ``sp.callaway_santanna``
against R ``did`` on unequal cluster sizes.

``sp.callaway_santanna(clustervars=, bstrap=False)`` forms its standard
errors from the cluster sums of the influence function,
``sqrt(sum_c (sum_{i in c} psi_i)^2) / n``. Until did 2.5.0 the only
reference for that was Stata ``csdid``
(``test_cs_rc_cluster_csdid_parity.py``). did 2.3.0 had no analytic
clustered standard error, and its clustered bootstrap averaged the influence
function within cluster, which equals the cluster-robust variance only when
every cluster has the same size, so ``test_cs_gaps_parity.py`` covered the
clustered path against R with property tests only. did 2.5.0 computes the
analytic clustered standard error from cluster sums at every aggregation
level, so the comparison with the canonical implementation is now direct.

Fixture: ``_fixtures/_generate_cs_cluster_analytic_R.R`` (did 2.5.1, DRDID
1.3.0) on the two panels of ``test_cs_gaps_parity.py``, whose nine clusters
have sizes 150, 90, 45, 30, 20, 15, 6, 3 and 1. Eight cases: balanced and
unbalanced panel, never-treated and not-yet-treated comparison group,
varying and universal base period. Each case pins every ATT(g, t) cell and
its standard error, the overall ATT and standard error of the simple,
dynamic, group and calendar aggregations, and the dynamic aggregation's
per-event-time standard errors.

Tolerance ``rtol = 1e-8`` throughout. Both sides evaluate the same closed
form, so nothing looser is justified; the observed worst case is 7e-13 on
the estimates and 3e-14 on the standard errors.

References
----------
- Callaway, B. and Sant'Anna, P.H.C. (2021). "Difference-in-Differences with
  Multiple Time Periods." *Journal of Econometrics*, 225(2), 200-230.
  [@callaway2021difference]
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_DIR = pathlib.Path(__file__).parent / "_fixtures"
_JSON = _DIR / "cs_cluster_analytic_R.json"
_RTOL = 1e-8
_AGGREGATIONS = ("simple", "dynamic", "group", "calendar")


def _cases():
    return json.loads(_JSON.read_text(encoding="utf-8"))["cases"]


def _id(c):
    panel = "unbalanced" if "unbalanced" in c["panel"] else "balanced"
    return f"{panel}-{c['control_group']}-{c['base_period']}"


def _fit(c):
    df = pd.read_csv(_DIR / f"{c['panel']}.csv")
    kw = dict(
        y="y",
        g="g",
        t="t",
        i="i",
        estimator="dr",
        control_group=c["control_group"],
        base_period=c["base_period"],
        clustervars="state",
        bstrap=False,
    )
    if "unbalanced" in c["panel"]:
        kw["allow_unbalanced_panel"] = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.callaway_santanna(df, **kw)


def test_fixture_was_generated_with_cluster_sums():
    """did before 2.5.0 has no analytic clustered SE to compare with."""
    meta = json.loads(_JSON.read_text(encoding="utf-8"))["meta"]
    major, minor = (int(x) for x in meta["did_version"].split(".")[:2])
    assert (major, minor) >= (2, 5)


@pytest.mark.parametrize("case", _cases(), ids=_id)
def test_clustered_cells_match_r(case):
    fit = _fit(case)
    ref = pd.DataFrame({k: case[k] for k in ("group", "time", "att", "se")})
    merged = ref.merge(
        fit.detail[["group", "time", "att", "se"]],
        on=["group", "time"],
        how="outer",
        suffixes=("_r", "_sp"),
        indicator=True,
    )
    assert (merged["_merge"] == "both").all(), "the (g, t) grids differ from R"
    np.testing.assert_allclose(merged["att_sp"], merged["att_r"], rtol=_RTOL)
    np.testing.assert_allclose(merged["se_sp"], merged["se_r"], rtol=_RTOL)


@pytest.mark.parametrize("agg_type", _AGGREGATIONS)
@pytest.mark.parametrize("case", _cases(), ids=_id)
def test_clustered_aggregates_match_r(case, agg_type):
    agg = sp.aggte(_fit(case), type=agg_type, bstrap=False)
    want = case["agg"][agg_type]
    assert agg.estimate == pytest.approx(want["att"], rel=_RTOL)
    assert agg.se == pytest.approx(want["se"], rel=_RTOL)


@pytest.mark.parametrize("case", _cases(), ids=_id)
def test_clustered_event_time_ses_match_r(case):
    agg = sp.aggte(_fit(case), type="dynamic", bstrap=False)
    want = case["agg"]["dynamic"]
    ref = pd.DataFrame(
        {
            "relative_time": want["egt"],
            "att_r": want["att_egt"],
            "se_r": [np.nan if v is None else v for v in want["se_egt"]],
        }
    ).dropna(subset=["se_r"])
    merged = ref.merge(agg.detail[["relative_time", "att", "se"]], on="relative_time")
    assert len(merged) == len(ref)
    np.testing.assert_allclose(merged["att"], merged["att_r"], rtol=_RTOL, atol=1e-12)
    np.testing.assert_allclose(merged["se"], merged["se_r"], rtol=_RTOL)


def test_clustering_moves_the_standard_errors():
    """Guard against the fixture pinning an unclustered path by accident."""
    case = _cases()[0]
    df = pd.read_csv(_DIR / f"{case['panel']}.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = sp.callaway_santanna(
            df,
            y="y",
            g="g",
            t="t",
            i="i",
            estimator="dr",
            control_group=case["control_group"],
            base_period=case["base_period"],
            bstrap=False,
        )
    clustered = sp.aggte(_fit(case), type="simple", bstrap=False).se
    unclustered = sp.aggte(plain, type="simple", bstrap=False).se
    assert abs(clustered / unclustered - 1) > 0.05
