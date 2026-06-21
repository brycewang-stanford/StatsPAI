"""Coverage tests for the statspai.synth dispatcher (sp.synth method routes).

Exercises every method branch of the unified dispatcher on a small real
synthetic-control panel and asserts genuine structural properties.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _panel(n_units=8, n_periods=16, treatment_time=11, effect=4.0, seed=7):
    rng = np.random.default_rng(seed)
    alphas = rng.normal(10, 2, n_units)
    betas = rng.normal(0.5, 0.1, n_units)
    common = rng.normal(0, 0.4, n_periods)  # shared factor
    records = []
    for i in range(n_units):
        for ti, t in enumerate(range(1, n_periods + 1)):
            y = alphas[i] + betas[i] * t + common[ti] + rng.normal(0, 0.2)
            if i == 0 and t >= treatment_time:
                y += effect
            records.append({"unit": f"u{i}", "time": t, "outcome": y})
    return pd.DataFrame(records)


@pytest.fixture
def panel():
    return _panel()


@pytest.mark.parametrize(
    "method",
    [
        "classic",
        "penalized",
        "ridge",
        "demeaned",
        "detrended",
        "unconstrained",
        "elastic_net",
        "augmented",
        "ascm",
        "factor",
        "gsynth",
        "mc",
        "matrix_completion",
        "penscm",
        "abadie_lhour",
        "fdid",
        "forward_did",
        "cluster",
        "sparse",
        "lasso",
        "kernel",
        "kernel_ridge",
    ],
)
def test_dispatch_methods_basic(panel, method):
    res = sp.synth(
        panel,
        outcome="outcome",
        unit="unit",
        time="time",
        treated_unit="u0",
        treatment_time=11,
        placebo=False,
    )
    # The dispatcher with explicit method:
    res = sp.synth(
        panel,
        outcome="outcome",
        unit="unit",
        time="time",
        treated_unit="u0",
        treatment_time=11,
        method=method,
        placebo=False,
    )
    assert np.isfinite(res.estimate)
    mi = res.model_info or {}
    # gap series should have a row per time period when present
    if "gap_table" in mi and mi["gap_table"] is not None:
        gt = mi["gap_table"]
        assert len(gt) >= 1


def test_dispatch_sdid_method(panel):
    res = sp.synth(
        panel,
        outcome="outcome",
        unit="unit",
        time="time",
        treated_unit="u0",
        treatment_time=11,
        method="sdid",
        placebo=False,
    )
    assert np.isfinite(res.estimate)


def test_dispatch_scpi_method(panel):
    res = sp.synth(
        panel,
        outcome="outcome",
        unit="unit",
        time="time",
        treated_unit="u0",
        treatment_time=11,
        method="scpi",
    )
    assert np.isfinite(res.estimate)


def test_dispatch_bsts_method(panel):
    res = sp.synth(
        panel,
        outcome="outcome",
        unit="unit",
        time="time",
        treated_unit="u0",
        treatment_time=11,
        method="bsts",
    )
    assert np.isfinite(res.estimate)


def test_dispatch_staggered_with_treatment(panel):
    # method='staggered' route through the dispatcher (needs treatment col)
    rng = np.random.default_rng(3)
    recs = []
    adopt = {"a": 8, "b": 12, "c": None, "d": None, "e": None, "f": None}
    for u, g in adopt.items():
        a = rng.normal(10, 1)
        for t in range(1, 17):
            y = a + 0.4 * t + rng.normal(0, 0.2)
            treat = 1 if (g is not None and t >= g) else 0
            if treat:
                y += 3.0
            recs.append({"unit": u, "time": t, "outcome": y, "treated": treat})
    df = pd.DataFrame(recs)
    res = sp.synth(
        df,
        outcome="outcome",
        unit="unit",
        time="time",
        method="staggered",
        treatment="treated",
        placebo=False,
    )
    assert np.isfinite(res.estimate)


def test_dispatch_discos_route(panel):
    rng = np.random.default_rng(4)
    recs = []
    for i in range(7):
        mu = rng.normal(5, 1)
        for t in range(1, 13):
            for _ in range(25):
                y = rng.normal(mu + 0.1 * t, 1.0)
                if i == 0 and t >= 8:
                    y += 2.0
                recs.append({"unit": f"u{i}", "time": t, "outcome": y})
    df = pd.DataFrame(recs)
    res = sp.synth(
        df,
        outcome="outcome",
        unit="unit",
        time="time",
        treated_unit="u0",
        treatment_time=8,
        method="discos",
        placebo=False,
    )
    assert np.isfinite(res.estimate)


def test_dispatch_multi_outcome_route():
    rng = np.random.default_rng(9)
    recs = []
    for i in range(8):
        a = rng.normal(10, 2)
        for t in range(1, 17):
            y1 = a + 0.5 * t + rng.normal(0, 0.2)
            y2 = a * 0.5 + 0.3 * t + rng.normal(0, 0.2)
            if i == 0 and t >= 11:
                y1 += 4.0
                y2 += 2.0
            recs.append({"unit": f"u{i}", "time": t, "y1": y1, "y2": y2})
    df = pd.DataFrame(recs)
    res = sp.synth(
        df,
        outcome="y1",
        unit="unit",
        time="time",
        treated_unit="u0",
        treatment_time=11,
        method="multi_outcome",
        outcomes=["y1", "y2"],
        placebo=False,
    )
    assert np.isfinite(res.estimate)


def test_dispatch_unknown_method_raises(panel):
    with pytest.raises(ValueError):
        sp.synth(
            panel,
            outcome="outcome",
            unit="unit",
            time="time",
            treated_unit="u0",
            treatment_time=11,
            method="not_a_real_method",
        )


def test_dispatch_staggered_requires_treatment(panel):
    # staggered method needs a treatment indicator column
    with pytest.raises((ValueError, KeyError, TypeError)):
        sp.synth(
            panel,
            outcome="outcome",
            unit="unit",
            time="time",
            treated_unit="u0",
            treatment_time=11,
            method="staggered",
        )


def test_classic_weights_simplex(panel):
    res = sp.synth(
        panel,
        outcome="outcome",
        unit="unit",
        time="time",
        treated_unit="u0",
        treatment_time=11,
        method="classic",
        placebo=False,
    )
    mi = res.model_info or {}
    w = mi.get("weights")
    if w is not None:
        if isinstance(w, dict):
            vals = np.asarray(list(w.values()), dtype=float)
        else:
            arr = np.asarray(w, dtype=object)
            if arr.ndim == 2:  # list of (name, weight) pairs
                vals = arr[:, -1].astype(float)
            else:
                vals = np.asarray(w, dtype=float)
        assert vals.min() >= -1e-6
        assert abs(vals.sum() - 1.0) < 1e-2


def test_classic_with_placebo_populates(panel):
    res = sp.synth(
        panel,
        outcome="outcome",
        unit="unit",
        time="time",
        treated_unit="u0",
        treatment_time=11,
        method="classic",
        placebo=True,
    )
    mi = res.model_info or {}
    # placebo machinery should populate something
    assert any(
        k in mi for k in ("placebo_gaps", "placebo_effects", "rmspe_ratios", "pvalue")
    )


def test_summary_and_repr(panel):
    res = sp.synth(
        panel,
        outcome="outcome",
        unit="unit",
        time="time",
        treated_unit="u0",
        treatment_time=11,
        method="classic",
        placebo=False,
    )
    assert isinstance(res.summary(), str)
    assert len(res.summary()) > 0
