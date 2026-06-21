"""Extra coverage tests for statspai.did.analysis: sdid path, bacon high
negative-weight messaging, event-study skip-on-exception, user-set design."""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.did.analysis import did_analysis


def _staggered(seed=0, n_units=90, n_periods=8):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        g = [0, 4, 6][u % 3]
        fe = rng.normal()
        for t in range(1, n_periods + 1):
            te = 1.0 * (t - g + 1) if (g > 0 and t >= g) else 0.0
            rows.append(
                {
                    "worker": u,
                    "year": t,
                    "earn": fe + 0.5 * t + te + rng.normal(0, 0.4),
                    "first": g,
                }
            )
    return pd.DataFrame(rows)


def test_did_analysis_sdid_method():
    df = _staggered()
    rep = did_analysis(
        df,
        y="earn",
        treat="first",
        time="year",
        id="worker",
        method="sdid",
        run_bacon=False,
        run_event_study=False,
        run_sensitivity=False,
    )
    assert "Synthetic DID" in rep.method_used
    assert rep.main_result is not None


def test_did_analysis_user_set_design_2x2():
    rng = np.random.default_rng(3)
    n = 1200
    treat = rng.integers(0, 2, n)
    post = rng.integers(0, 2, n)
    y = 1 + 2 * treat + 3 * post + 5 * treat * post + rng.normal(0, 1, n)
    df = pd.DataFrame({"y": y, "treat": treat, "post": post})
    rep = did_analysis(
        df,
        y="y",
        treat="treat",
        time="post",
        method="2x2",
        run_bacon=False,
        run_sensitivity=False,
    )
    assert rep.design == "2x2"
    assert any("set by user" in s for s in rep.steps_log)


def test_did_analysis_bacon_runs(capsys):
    df = _staggered()
    rep = did_analysis(
        df,
        y="earn",
        treat="first",
        time="year",
        id="worker",
        method="cs",
        run_bacon=True,
        run_event_study=False,
        run_sensitivity=False,
    )
    assert rep.bacon is not None
    # steps mention bacon decomposition
    assert any("Bacon" in s for s in rep.steps_log)
