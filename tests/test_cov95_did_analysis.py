"""Coverage tests for statspai.did.analysis (did_analysis workflow + DIDAnalysis)."""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.did.analysis import did_analysis


def _data_2x2(seed=0, n=1500):
    rng = np.random.default_rng(seed)
    treat = rng.integers(0, 2, n)
    post = rng.integers(0, 2, n)
    y = 1 + 2 * treat + 3 * post + 5 * treat * post + rng.normal(0, 1, n)
    return pd.DataFrame(
        {
            "y": y,
            "treat": treat,
            "post": post,
            "x1": rng.normal(0, 1, n),
            "cl": rng.integers(0, 10, n),
        }
    )


def _data_staggered(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(90):
        g = [0, 4, 6][u % 3]
        fe = rng.normal()
        for t in range(1, 9):
            te = 1.0 * (t - g + 1) if (g > 0 and t >= g) else 0.0
            rows.append(
                {
                    "worker": u,
                    "year": t,
                    "earn": fe + 0.5 * t + te + rng.normal(0, 0.4),
                    "first": g,
                    "x1": rng.normal(),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def stag():
    return _data_staggered()


def test_did_analysis_2x2_auto():
    df = _data_2x2()
    rep = did_analysis(
        df, y="y", treat="treat", time="post", run_bacon=False, run_sensitivity=False
    )
    assert rep.design == "2x2"
    assert abs(rep.main_result.estimate - 5.0) < 0.5
    s = rep.summary()
    assert "DID Analysis Report" in s
    assert "Main Estimate" in s


def test_did_analysis_2x2_with_covariates_cluster():
    df = _data_2x2()
    rep = did_analysis(
        df,
        y="y",
        treat="treat",
        time="post",
        covariates=["x1"],
        cluster="cl",
        run_bacon=False,
        run_sensitivity=False,
    )
    assert rep.main_result.se > 0


def test_did_analysis_staggered_cs_full(stag):
    rep = did_analysis(
        stag,
        y="earn",
        treat="first",
        time="year",
        id="worker",
        method="cs",
        run_bacon=True,
        run_event_study=True,
        run_sensitivity=True,
    )
    assert rep.design == "staggered"
    assert "Callaway" in rep.method_used
    assert rep.main_result.estimate > 0
    s = rep.summary()
    assert "Event Study" in s
    # bacon block present
    assert rep.bacon is not None
    assert "Bacon Decomposition" in s


def test_did_analysis_staggered_auto_picks_cs(stag):
    rep = did_analysis(
        stag,
        y="earn",
        treat="first",
        time="year",
        id="worker",
        method="auto",
        run_bacon=False,
        run_sensitivity=False,
    )
    assert rep.design == "staggered"
    assert "Callaway" in rep.method_used


def test_did_analysis_sun_abraham(stag):
    rep = did_analysis(
        stag,
        y="earn",
        treat="first",
        time="year",
        id="worker",
        method="sa",
        run_bacon=False,
        run_event_study=False,
        run_sensitivity=False,
    )
    assert "Sun-Abraham" in rep.method_used


def test_did_analysis_bjs_imputation(stag):
    rep = did_analysis(
        stag,
        y="earn",
        treat="first",
        time="year",
        id="worker",
        method="bjs",
        run_bacon=False,
        run_event_study=True,
        run_sensitivity=False,
        event_window=(-3, 3),
    )
    assert "Borusyak" in rep.method_used


def test_did_analysis_auto_no_id_nonbinary_raises():
    df = _data_2x2()
    df["treat"] = np.arange(len(df)) % 4  # nonbinary, no id
    with pytest.raises(ValueError):
        did_analysis(
            df,
            y="y",
            treat="treat",
            time="post",
            run_bacon=False,
            run_sensitivity=False,
        )


def test_did_analysis_cs_missing_id_raises(stag):
    from statspai.exceptions import MethodIncompatibility

    with pytest.raises(MethodIncompatibility):
        did_analysis(
            stag,
            y="earn",
            treat="first",
            time="year",
            method="cs",
            run_bacon=False,
            run_sensitivity=False,
        )


def test_did_analysis_sa_missing_id_raises(stag):
    from statspai.exceptions import MethodIncompatibility

    with pytest.raises(MethodIncompatibility):
        did_analysis(
            stag,
            y="earn",
            treat="first",
            time="year",
            method="sa",
            run_bacon=False,
            run_sensitivity=False,
        )


def test_did_analysis_bjs_missing_id_raises(stag):
    from statspai.exceptions import MethodIncompatibility

    with pytest.raises(MethodIncompatibility):
        did_analysis(
            stag,
            y="earn",
            treat="first",
            time="year",
            method="bjs",
            run_bacon=False,
            run_sensitivity=False,
        )


def test_did_analysis_unknown_method_raises(stag):
    with pytest.raises(ValueError):
        did_analysis(
            stag,
            y="earn",
            treat="first",
            time="year",
            id="worker",
            method="not_a_method",
            run_bacon=False,
            run_sensitivity=False,
        )


def test_did_analysis_plot_event_study(stag):
    import matplotlib

    matplotlib.use("Agg")
    rep = did_analysis(
        stag,
        y="earn",
        treat="first",
        time="year",
        id="worker",
        method="cs",
        run_bacon=False,
        run_event_study=True,
        run_sensitivity=False,
    )
    # should not raise
    rep.plot()


def test_did_analysis_plot_main_only():
    import matplotlib

    matplotlib.use("Agg")
    df = _data_2x2()
    rep = did_analysis(
        df,
        y="y",
        treat="treat",
        time="post",
        run_bacon=False,
        run_event_study=False,
        run_sensitivity=False,
    )
    rep.plot()
