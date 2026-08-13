"""Design-level DiD audits: cluster counts and the forward-engineering contract.

The cluster grading is graded against a published simulation grid rather
than a rule of thumb, so the tests pin the grid boundaries themselves: a
future edit that quietly moves the threshold away from what
Ulloa-Perez et al. (2025) actually ran has to move a test.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient


def _panel(
    n_clusters: int, units_per_cluster: int = 2, n_times: int = 6, seed: int = 3
):
    """Staggered panel with treatment assigned at the cluster level."""
    rng = np.random.default_rng(seed)
    cohorts = [0, 3, 4, 5]
    rows = []
    for c in range(n_clusters):
        g = cohorts[c % len(cohorts)]
        for u in range(units_per_cluster):
            uid = c * units_per_cluster + u
            base = rng.normal()
            for t in range(1, n_times + 1):
                eff = 0.0 if (g == 0 or t < g) else 1.0
                rows.append((uid, c, t, g, base + 0.1 * t + eff + rng.normal()))
    return pd.DataFrame(rows, columns=["unit", "cluster", "time", "g", "y"])


# --------------------------------------------------------------------- #
# Cluster diagnostics
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "n_clusters, expected",
    [
        (12, "below-evidence"),
        (30, "weakest-cell"),
        (49, "weakest-cell"),
        (50, "improving"),
        (99, "improving"),
        (100, "largest-cell-or-above"),
        (140, "largest-cell-or-above"),
    ],
)
def test_cluster_verdict_tracks_the_published_grid(n_clusters, expected):
    df = _panel(n_clusters)
    diag = sp.did_cluster_diagnostics(
        df, unit="unit", first_treat="g", cluster="cluster", warn=False
    )
    assert diag.n_clusters == n_clusters
    assert diag.verdict == expected
    assert tuple(diag.reference_grid) == (30, 50, 100)


def test_cluster_counts_split_treated_and_control():
    df = _panel(40)
    diag = sp.did_cluster_diagnostics(
        df, unit="unit", first_treat="g", cluster="cluster", warn=False
    )
    assert diag.n_treated_clusters + diag.n_control_clusters == diag.n_clusters
    # Cohorts 3, 4 and 5 are treated; cohort 0 is never-treated.
    assert diag.n_cohorts == 3
    assert set(diag.clusters_per_cohort) == {3, 4, 5}
    assert sum(diag.clusters_per_cohort.values()) == diag.n_treated_clusters


def test_defaulting_to_the_unit_id_warns():
    """Silently treating units as clusters is the optimistic error."""
    df = _panel(40)
    with pytest.warns(UserWarning, match="no cluster= given"):
        diag = sp.did_cluster_diagnostics(df, unit="unit", first_treat="g")
    assert diag.cluster_is_unit
    # 40 clusters of 2 units: the unit-level count is double, and it is the
    # coarser count that governs inference.
    assert diag.n_clusters == 80


def test_thin_designs_warn_and_say_they_are_outside_the_evidence():
    df = _panel(12)
    with pytest.warns(UserWarning, match="below-evidence"):
        diag = sp.did_cluster_diagnostics(
            df, unit="unit", first_treat="g", cluster="cluster"
        )
    assert "does not extend downward" in diag.interpretation
    assert "below-evidence" in diag.summary()


def test_cluster_diagnostics_rejects_unknown_columns():
    df = _panel(30)
    with pytest.raises(ValueError, match="not found in data"):
        sp.did_cluster_diagnostics(df, unit="unit", first_treat="nope", warn=False)


def test_cluster_diagnostics_rejects_an_empty_frame():
    empty = _panel(30).iloc[0:0]
    with pytest.raises((DataInsufficient, ValueError)):
        sp.did_cluster_diagnostics(
            empty, unit="unit", first_treat="g", cluster="cluster", warn=False
        )


def test_cluster_diagnostics_roundtrips_to_dict():
    df = _panel(60)
    diag = sp.did_cluster_diagnostics(
        df, unit="unit", first_treat="g", cluster="cluster", warn=False
    )
    payload = diag.to_dict()
    assert payload["verdict"] == "improving"
    assert payload["reference_grid"] == [30, 50, 100]


# --------------------------------------------------------------------- #
# Forward-engineering contract
# --------------------------------------------------------------------- #


def test_contract_covers_all_eight_steps():
    df = _panel(40)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.callaway_santanna(df, y="y", g="g", t="time", i="unit")
    contract = sp.did_design_contract(fit)
    assert [s["step"] for s in contract.steps] == list(range(1, 9))
    assert {s["status"] for s in contract.steps} <= {
        "determined",
        "undetermined",
        "not-evidenced",
    }
    # Step 8 is a judgement about the setting, not a property of the object.
    assert contract.steps[7]["status"] == "not-evidenced"


def test_comparison_group_is_reported_as_an_assumption_not_an_option():
    df = _panel(40)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.callaway_santanna(
            df, y="y", g="g", t="time", i="unit", control_group="notyettreated"
        )
    contract = sp.did_design_contract(fit)
    step2 = next(s for s in contract.steps if s["key"] == "identification")
    assert step2["status"] == "determined"
    assert "parallel trends" in step2["detail"]
    assert "not-yet-treated" in step2["detail"]


def test_aggregation_determines_the_target_parameter():
    """The raw att_gt object has not chosen a summary; aggte has."""
    df = _panel(40)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.callaway_santanna(df, y="y", g="g", t="time", i="unit")
        agg = sp.aggte(fit, type="dynamic")
    raw_step1 = sp.did_design_contract(fit).steps[0]
    agg_step1 = sp.did_design_contract(agg).steps[0]
    assert raw_step1["status"] == "undetermined"
    assert agg_step1["status"] == "determined"
    assert "dynamic" in agg_step1["detail"]


def test_missing_slots_are_undetermined_not_defaulted():
    class Bare:
        model_info: dict = {}
        estimate = None

    contract = sp.did_design_contract(Bare())
    statuses = {s["key"]: s["status"] for s in contract.steps}
    assert statuses["target_parameter"] == "undetermined"
    assert statuses["identification"] == "undetermined"
    assert statuses["uncertainty"] == "undetermined"
    assert contract.n_determined == 0
    assert "identification" in contract.undetermined


def test_contract_summary_explains_the_marks():
    df = _panel(40)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.callaway_santanna(df, y="y", g="g", t="time", i="unit")
    text = sp.did_design_contract(fit).summary()
    assert "forward-engineering contract" in text
    assert "An unstated choice is still a choice" in text
