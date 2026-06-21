"""Coverage round-3 (final) — multi-outcome SCM, sequential SDID, and the
classic SCM dispatcher (``synth/scm.py``).

Targets validation guards, the alternative ``method`` options, and the
loud failures of ``multi_outcome_synth``, ``sequential_sdid`` (staggered
cohort SDID), and the ``sp.synth`` classic dispatcher.

All pure-numpy. Assertions check real properties (finite ATT/SE,
populated per-cohort tables, correct exceptions); no estimator numbers
are fabricated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.synth.multi_outcome import multi_outcome_synth
from statspai.synth.sequential_sdid import sequential_sdid

T_TREAT = 11


def _panel(seed=0, n_donors=8, n_t=20, effect=4.0, outcomes=("y",)):
    rng = np.random.default_rng(seed)
    units = [f"u{i}" for i in range(n_donors)] + ["treated"]
    rows = []
    for u in units:
        base = rng.normal(0, 1)
        fe = rng.normal(0, 0.5)
        for t in range(1, n_t + 1):
            row = {"unit": u, "time": t}
            for k, oc in enumerate(outcomes):
                eff = effect if (u == "treated" and t >= T_TREAT) else 0.0
                row[oc] = base + 0.2 * t + fe + eff + 0.3 * k + rng.normal(0, 0.3)
            rows.append(row)
    return pd.DataFrame(rows)


def _staggered_panel(seed=0, n_units=12, n_t=16):
    """Panel with staggered adoption cohorts for sequential SDID."""
    rng = np.random.default_rng(seed)
    rows = []
    cohorts = {}
    for i in range(n_units):
        # roughly a third never-treated, rest split across two cohorts
        if i % 3 == 0:
            cohort = 0  # never treated
        elif i % 3 == 1:
            cohort = 8
        else:
            cohort = 11
        cohorts[f"u{i}"] = cohort
    for i in range(n_units):
        u = f"u{i}"
        c = cohorts[u]
        base = rng.normal(0, 1)
        for t in range(1, n_t + 1):
            eff = 3.0 if (c != 0 and t >= c) else 0.0
            rows.append(
                {
                    "unit": u,
                    "time": t,
                    "cohort": c,
                    "y": base + 0.2 * t + eff + rng.normal(0, 0.3),
                }
            )
    return pd.DataFrame(rows)


# ===========================================================================
# multi_outcome_synth
# ===========================================================================
@pytest.mark.parametrize("method", ["concatenated", "averaged"])
def test_multi_outcome_methods(method):
    df = _panel(0, outcomes=("y1", "y2"))
    r = multi_outcome_synth(
        df,
        outcomes=["y1", "y2"],
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        method=method,
        placebo=True,
    )
    assert np.isfinite(r.estimate)


def test_multi_outcome_unknown_method_raises():
    df = _panel(1, outcomes=("y1", "y2"))
    with pytest.raises(ValueError):
        multi_outcome_synth(
            df,
            outcomes=["y1", "y2"],
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=T_TREAT,
            method="bogus",
        )


def test_multi_outcome_missing_column_raises():
    df = _panel(2, outcomes=("y1",))
    with pytest.raises(ValueError):
        multi_outcome_synth(
            df,
            outcomes=["y1", "does_not_exist"],
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=T_TREAT,
        )


def test_multi_outcome_missing_treated_raises():
    df = _panel(3, outcomes=("y1", "y2"))
    with pytest.raises((ValueError, KeyError)):
        multi_outcome_synth(
            df,
            outcomes=["y1", "y2"],
            unit="unit",
            time="time",
            treated_unit="ghost",
            treatment_time=T_TREAT,
        )


# ===========================================================================
# sequential_sdid (staggered SDID)
# ===========================================================================
def test_sequential_sdid_runs_and_reports():
    df = _staggered_panel(0)
    r = sequential_sdid(
        df,
        outcome="y",
        unit="unit",
        time="time",
        cohort="cohort",
        never_treated_value=0,
        n_reps=20,
        seed=1,
    )
    assert np.isfinite(r.estimate)
    # The result __repr__ / summary paths
    txt = str(r)
    assert len(txt) > 0
    if hasattr(r, "summary"):
        assert isinstance(r.summary(), str)


def test_sequential_sdid_result_dataclass_summary_repr():
    from statspai.synth.sequential_sdid import SequentialSDIDResult

    per = pd.DataFrame({"cohort": [8, 11], "att": [3.1, 2.9], "se": [0.4, 0.5]})
    res = SequentialSDIDResult(
        aggregate_att=3.0,
        aggregate_se=0.3,
        aggregate_ci=(2.4, 3.6),
        per_cohort=per,
        model_info={"n_cohorts": 2},
    )
    s = res.summary()
    assert "Aggregate ATT" in s and "Per-cohort" in s
    r = repr(res)
    assert "SequentialSDIDResult" in r and "2 cohorts" in r


def test_sequential_sdid_cohort_weights_equal():
    df = _staggered_panel(1)
    r = sequential_sdid(
        df,
        outcome="y",
        unit="unit",
        time="time",
        cohort="cohort",
        never_treated_value=0,
        cohort_weights="equal",
        n_reps=15,
        seed=2,
    )
    assert np.isfinite(r.estimate)


def test_sequential_sdid_bad_data_type_raises():
    with pytest.raises(TypeError):
        sequential_sdid(
            [1, 2, 3], outcome="y", unit="unit", time="time", cohort="cohort"
        )


def test_sequential_sdid_missing_column_raises():
    df = _staggered_panel(2).drop(columns=["cohort"])
    with pytest.raises(ValueError):
        sequential_sdid(df, outcome="y", unit="unit", time="time", cohort="cohort")


# ===========================================================================
# classic SCM dispatcher (scm.py) — validation + branches
# ===========================================================================
def test_synth_classic_missing_args_raise():
    df = _panel(0)
    with pytest.raises((ValueError, TypeError)):
        sp.synth(df, outcome="y", unit="unit", time="time", method="classic")


def test_synth_classic_unknown_method_raises():
    df = _panel(1)
    with pytest.raises(ValueError):
        sp.synth(
            df,
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=T_TREAT,
            method="totally_unknown_method",
        )


def test_synth_classic_with_predictors_and_no_standardize():
    df = _panel(2)
    r = sp.synth(
        df,
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        method="classic",
        standardize=False,
    )
    assert np.isfinite(r.estimate)
    w = r.model_info.get("weights")
    if w is not None and isinstance(w, dict):
        vals = np.asarray(list(w.values()), dtype=float)
        assert vals.min() >= -1e-6
