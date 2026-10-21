"""``sp.stacked_did(cluster_stack=, aggregation=)`` vs Stata ``stackedev``.

Stata ``stackedev`` (Bleiberg, 0.0) stacks every cohort with the
never-treated units over the whole panel and clusters on
``stack * clust_unit``, a unit x sub-experiment cluster. A control unit
enters every stack, so that clustering ignores the correlation of its errors
across stacks; ``sp.stacked_did`` clusters on the unit by default and
reproduces ``stackedev`` with ``cluster_stack=True``.

Reference: Stata 18, ``stackedev y F*event L*event, cohort(first_treat)
time(year) never_treat(never) covariates(x) unit_fe(id) clust_unit(id)``
and the same stack by hand with ``reghdfe ..., cluster(id)``, on
``_fixtures/stacked_did_stackedev.csv`` (60 units, 2011-2018, cohorts
2014 / 2015 / 2016, three missing unit-periods), from
``_generate_stacked_did_stackedev_Stata.do``. The two headline averages
are Stata ``lincom`` of L0-L4 with equal weights and with weights equal to
the number of treated rows at each event time. Held to 1e-9 (observed
~2e-11).
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
R = json.loads((_FIX / "stacked_did_stackedev_Stata.json").read_text(encoding="utf-8"))
KS = (-5, -4, -3, -2, 0, 1, 2, 3, 4)


def _name(k: int) -> str:
    return f"F{-k}event" if k < 0 else f"L{k}event"


@pytest.fixture(scope="module")
def panel():
    return pd.read_csv(_FIX / "stacked_did_stackedev.csv")


def _fit(panel, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.stacked_did(
            panel,
            y="y",
            group="id",
            time="year",
            first_treat="first_treat",
            window=(-5, 4),
            controls=["x"],
            **kw,
        )


@pytest.mark.parametrize("key, stack", [("stackedev", True), ("unit", False)])
@pytest.mark.parametrize("agg", ["event_time", "treated_obs"])
def test_matches_stata(panel, key, stack, agg):
    ref = R[key]
    r = _fit(panel, cluster_stack=stack, aggregation=agg)
    es = r.model_info["event_study"].set_index("relative_time")
    assert r.n_obs == ref["N"]
    assert r.model_info["n_clusters"] == ref["N_clust"]
    for k in KS:
        assert es.loc[k, "att"] == pytest.approx(
            ref[f"b_{_name(k)}"], rel=1e-9, abs=1e-10
        )
        assert es.loc[k, "se"] == pytest.approx(ref[f"se_{_name(k)}"], rel=1e-9)
    head = "att_eq" if agg == "event_time" else "att_obs"
    assert r.estimate == pytest.approx(ref[head], rel=1e-9)
    assert r.se == pytest.approx(ref[f"{head}_se"], rel=1e-9)


def test_unit_by_stack_understates_the_control_se():
    # The point of the default: the control's SE under stackedev's
    # clustering is well below the unit-clustered one (every control unit
    # repeats in all three stacks), while the event-time SEs barely move.
    se_x = R["stackedev"]["se_x"] / R["unit"]["se_x"]
    assert se_x < 0.75
    assert R["stackedev"]["N_clust"] == 132 and R["unit"]["N_clust"] == 60


def test_treated_obs_weights_are_the_treated_row_shares(panel):
    r = _fit(panel, aggregation="treated_obs")
    n = R["n_treated"]
    tot = sum(n.values())
    w = r.model_info["aggregation_weights"]
    assert w == {int(k): pytest.approx(v / tot, rel=1e-15) for k, v in n.items()}
    es = r.model_info["event_study"].set_index("relative_time")
    assert r.estimate == pytest.approx(
        sum(w[k] * es.loc[k, "att"] for k in w), rel=1e-12
    )


def test_aggregations_coincide_when_every_post_time_has_the_same_rows():
    # Balanced event window: each cohort observed at every post event time,
    # so the treated-row weights are equal and the two averages agree.
    df = sp.dgp_did(n_units=90, n_periods=10, staggered=True, seed=3)
    kw = dict(y="y", group="unit", time="time", first_treat="first_treat")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cohorts = sorted(df.loc[df.first_treat > 0, "first_treat"].unique())
        hi = int(df.time.max() - max(cohorts))
        a = sp.stacked_did(df, window=(-2, hi), **kw)
        b = sp.stacked_did(df, window=(-2, hi), aggregation="treated_obs", **kw)
    assert np.isclose(b.estimate, a.estimate, rtol=1e-12)
    assert np.isclose(b.se, a.se, rtol=1e-12)


def test_bad_aggregation_and_pooled_are_refused(panel):
    with pytest.raises(sp.MethodIncompatibility, match="aggregation"):
        _fit(panel, aggregation="cohort")
    with pytest.raises(sp.MethodIncompatibility, match="pooled"):
        _fit(panel, spec="pooled", aggregation="treated_obs")


def test_cluster_stack_with_a_coarser_cluster(panel):
    # cluster x stack with a cluster coarser than the unit: one cluster per
    # (region, stack) pair present in the stack.
    p = panel.assign(region=panel["id"] % 7)
    r = _fit(p, cluster="region", cluster_stack=True)
    assert r.model_info["n_clusters"] == 21
    assert r.model_info["cluster_var"] == "region"
    assert r.model_info["cluster_stack"] is True
