"""``next_steps()`` follows the variance estimator the fit already used.

2026-10-19 audit: every ``sp.feols`` result was told to "re-estimate with
heteroskedasticity-robust standard errors" even when clustered, and was
handed the IV checklist (first-stage F, over-identification, Kitagawa),
because ``"iv" in "ols (statspai native)"`` is true.
"""

from __future__ import annotations

import pytest

import statspai as sp


@pytest.fixture(scope="module")
def panel():
    return sp.dgp_did(n_units=60, n_periods=8, seed=1)


def _actions(result):
    return [s["action"] for s in result.next_steps()]


def _variance_actions(result):
    keys = ("robust=", "vcov=", "wild_cluster_boot")
    return [a for a in _actions(result) if any(k in a for k in keys)]


def test_clustered_feols_gets_no_robust_refit_and_no_iv_checklist(panel):
    r = sp.feols("y ~ treated | unit + time", data=panel, vcov={"CRV1": "unit"})
    acts = _actions(r)
    assert _variance_actions(r) == []
    assert not any("firststage" in a or "kitagawa" in a for a in acts)


def test_iid_feols_is_pointed_at_its_own_hetero_vcov(panel):
    r = sp.feols("y ~ treated | unit + time", data=panel)
    (action,) = _variance_actions(r)
    assert action.startswith("sp.feols(") and "vcov='hetero'" in action
    # The suggested call runs and gives the hetero fit.
    rerun = eval(action, {"sp": sp, "df": panel})
    direct = sp.feols("y ~ treated | unit + time", data=panel, vcov="hetero")
    assert rerun.std_errors["treated"] == pytest.approx(
        direct.std_errors["treated"], rel=1e-12
    )


def test_regress_with_cluster_or_robust_needs_nothing(panel):
    for kw in ({"cluster": "unit"}, {"robust": "hc1"}):
        r = sp.regress("y ~ treated", data=panel, **kw)
        assert _variance_actions(r) == [], kw


def test_classical_regress_still_gets_the_robust_refit(panel):
    r = sp.regress("y ~ treated", data=panel)
    assert _variance_actions(r) == ["sp.regress('y ~ treated', data=df, robust='hc1')"]


def test_few_clusters_get_a_wild_cluster_bootstrap(panel):
    small = panel[panel["unit"] < 12]
    r = sp.regress("y ~ treated", data=small, cluster="unit")
    (action,) = _variance_actions(r)
    assert action.startswith("sp.wild_cluster_boot(result, data=df, cluster='unit'")


def test_iv_fit_keeps_its_checklist_and_refits_through_sp_iv(panel):
    r = sp.ivreg("y ~ (treated ~ group)", data=panel)
    acts = _actions(r)
    assert any("firststage" in a for a in acts)
    assert _variance_actions(r) == [
        "sp.iv('y ~ (treated ~ group)', data=df, robust='hc1')"
    ]
