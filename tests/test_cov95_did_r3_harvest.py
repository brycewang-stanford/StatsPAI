"""Coverage round-3 for statspai.did.harvest (sp.harvest_did).

Covers the treat-based cohort inference path, all three weighting schemes,
the HarvestDIDResult.summary() formatting, and the validation / empty-harvest
error branches. Uses a real staggered DiD panel; assertions check shapes,
ATT sign near the known positive DGP effect, p-values in [0, 1], and raised
errors.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.did.harvest import HarvestDIDResult, harvest_did


@pytest.fixture(scope="module")
def panel():
    return sp.dgp_did(n_units=90, n_periods=12, staggered=True, seed=0)


def test_harvest_treat_inference(panel):
    out = harvest_did(panel, unit="unit", time="time", outcome="y",
                      treat="treated", horizons=range(-3, 5))
    assert np.isfinite(out.estimate)
    assert 0.0 <= out.pvalue <= 1.0
    assert isinstance(out.detail, pd.DataFrame)
    assert {"cohort", "horizon", "att", "se"} <= set(out.detail.columns)
    es = out.model_info["event_study"]
    assert {"relative_time", "att", "se", "pvalue"} <= set(es.columns)
    assert "pretrend_test" in out.model_info


def test_harvest_cohort_column(panel):
    out = harvest_did(panel, unit="unit", time="time", outcome="y",
                      cohort="first_treat", never_value=np.nan)
    assert np.isfinite(out.estimate)


def test_harvest_weighting_equal(panel):
    out = harvest_did(panel, unit="unit", time="time", outcome="y",
                      treat="treated", weighting="equal")
    assert out.model_info["weighting"] == "equal"
    assert np.isfinite(out.estimate)


def test_harvest_weighting_n_treated(panel):
    out = harvest_did(panel, unit="unit", time="time", outcome="y",
                      treat="treated", weighting="n_treated")
    assert out.model_info["weighting"] == "n_treated"
    assert np.isfinite(out.estimate)


def test_harvest_missing_column_raises(panel):
    with pytest.raises(ValueError, match="not in data"):
        harvest_did(panel, unit="unit", time="time", outcome="missing",
                    treat="treated")


def test_harvest_no_treat_or_cohort_raises(panel):
    with pytest.raises(ValueError, match="treat.*cohort"):
        harvest_did(panel, unit="unit", time="time", outcome="y")


def test_harvest_bad_weighting_raises(panel):
    with pytest.raises(ValueError, match="weighting must be"):
        harvest_did(panel, unit="unit", time="time", outcome="y",
                    treat="treated", weighting="bogus")


def test_harvest_no_treated_obs_raises(panel):
    df = panel.copy()
    df["treated"] = 0  # nobody treated
    with pytest.raises(ValueError, match="No treated observations"):
        harvest_did(df, unit="unit", time="time", outcome="y", treat="treated")


def test_harvest_empty_horizons_raises(panel):
    # horizons that can never align with the panel time range -> 0 comparisons
    with pytest.raises(RuntimeError, match="0 valid"):
        harvest_did(panel, unit="unit", time="time", outcome="y",
                    treat="treated", horizons=[10_000])


def test_harvest_only_pre_horizons_raises(panel):
    # All-negative horizons leave no post-treatment horizon to aggregate.
    with pytest.raises(RuntimeError, match="post-treatment"):
        harvest_did(panel, unit="unit", time="time", outcome="y",
                    treat="treated", horizons=[-3, -2, -1])


def test_harvest_result_dataclass_summary():
    es = pd.DataFrame({"relative_time": [0, 1], "att": [0.4, 0.5],
                       "se": [0.1, 0.1], "pvalue": [0.01, 0.02],
                       "n_comparisons": [3, 3]})
    res = HarvestDIDResult(
        estimate=0.45, se=0.07, ci=(0.31, 0.59), alpha=0.05,
        n_comparisons=6,
        comparisons=pd.DataFrame({"att": [0.4]}),
        event_study=es,
        pretrend_test={"pvalue": 0.6},
    )
    s = res.summary()
    assert "Harvesting DID" in s
    assert "Aggregate ATT" in s
    assert "Event study" in s


def test_harvest_result_summary_empty_event_study():
    res = HarvestDIDResult(
        estimate=0.45, se=0.07, ci=(0.31, 0.59), alpha=0.05,
        n_comparisons=6,
        comparisons=pd.DataFrame({"att": [0.4]}),
        event_study=pd.DataFrame(),
        pretrend_test={"pvalue": 0.6},
    )
    s = res.summary()
    assert "Event study" not in s
