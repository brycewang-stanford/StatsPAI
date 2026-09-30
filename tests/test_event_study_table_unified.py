"""One event-study accessor across estimators (top-5 replication list).

``aggte(type='dynamic')`` kept its event study in ``detail`` while
Sun-Abraham, stacked DiD and BJS used ``model_info['event_study']``; and
``sp.event_study_table`` read only an ``estimate`` column, so for every
estimator that names it ``att`` it returned NaN point estimates without an
error (⚠️ fixed in 1.33). Each table must now reproduce its estimator's own
numbers.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def panel():
    rng = np.random.default_rng(0)
    units, periods = 80, 10
    df = pd.DataFrame(
        dict(
            i=np.repeat(np.arange(units), periods), t=np.tile(np.arange(periods), units)
        )
    )
    df["g"] = np.repeat(rng.choice([0, 4, 6, 8], units), periods)
    df["y"] = (df.g > 0) * (df.t >= df.g) * 1.0 + rng.normal(size=len(df))
    return df


def _fits(df):
    cs = sp.callaway_santanna(df, y="y", g="g", t="t", i="i")
    return {
        "sun_abraham": sp.sun_abraham(df, y="y", g="g", t="t", i="i"),
        "aggte_dynamic": sp.aggte(cs, type="dynamic"),
        "stacked_did": sp.stacked_did(
            df, y="y", group="i", time="t", first_treat="g", window=(-3, 3)
        ),
        "did_imputation": sp.did_imputation(
            df,
            y="y",
            group="i",
            time="t",
            first_treat="g",
            horizon=[0, 1, 2],
            pretrends=2,
        ),
        "event_study": sp.event_study(
            df, y="y", treat_time="g", time="t", unit="i", window=(-3, 3)
        ),
    }


def test_every_estimator_gives_a_table_with_its_own_numbers(panel):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fits = _fits(panel)
    for name, r in fits.items():
        es = r.model_info["event_study"]
        assert isinstance(es, pd.DataFrame), name
        col = "estimate" if "estimate" in es.columns else "att"
        tbl = sp.event_study_table(r)
        assert np.all(np.isfinite(tbl.params.to_numpy())), name
        ref = es.set_index("relative_time")[col]
        for label, value in tbl.params.items():
            t = int(str(label).split("=")[1])
            assert value == pytest.approx(float(ref.loc[t]), rel=1e-12), (name, t)


def test_aggte_dynamic_detail_and_model_info_agree(panel):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cs = sp.callaway_santanna(panel, y="y", g="g", t="t", i="i")
        ag = sp.aggte(cs, type="dynamic")
        grp = sp.aggte(cs, type="group")
    pd.testing.assert_frame_equal(ag.model_info["event_study"], ag.detail)
    assert "event_study" not in grp.model_info
