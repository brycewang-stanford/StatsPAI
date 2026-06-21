"""Regression tests: downstream tooling must work on a plain ``sp.event_study``
result.

`sp.event_study` used to emit its coefficient table with the column name
``estimate`` while the rest of the DID family (and every consumer) keys on the
canonical ``att`` column (``did._core.EVENT_STUDY_COLUMNS``). As a result the
canonical event-study estimator crashed its own library's exporters, plotters
and pre-trend tools:

* ``.tidy()`` / ``.to_markdown()`` -> ``TypeError: NoneType / float``
* ``.plot()`` / ``.event_study_plot()`` -> ``KeyError: 'att'``
* ``sp.pretrends_test`` / ``sp.pretrends_summary`` -> ``LinAlgError`` (the
  SE = 0 reference period made the diagonal VCV singular)
* ``sp.honest_did`` / ``sp.breakdown_m`` -> ``ValueError: missing {'att'}``

These guard the fixes (canonical ``att`` column + reference-period drop).
"""

import matplotlib
import numpy as np
import pandas as pd
import pytest

import statspai as sp

matplotlib.use("Agg")  # headless plotting


@pytest.fixture
def es_result():
    rng = np.random.RandomState(0)
    rows = []
    for u in range(48):
        tt = np.nan if u % 4 == 0 else [4, 6][u % 2]
        fe = rng.normal()
        for t in range(1, 11):
            post = (not np.isnan(tt)) and t >= tt
            y = fe + 0.2 * t + (1.5 if post else 0.0) + rng.normal()
            rows.append((u, t, y, tt))
    df = pd.DataFrame(rows, columns=["unit", "time", "y", "treat_time"])
    return sp.event_study(
        df,
        y="y",
        treat_time="treat_time",
        time="time",
        unit="unit",
        window=(-3, 3),
    )


def test_event_study_table_has_canonical_att_column(es_result):
    table = es_result.model_info["event_study"]
    # The canonical DID coefficient column must be present (and equal the
    # backward-compatible 'estimate' alias).
    assert "att" in table.columns
    assert np.allclose(table["att"].to_numpy(), table["estimate"].to_numpy())


def test_tidy_and_exports_do_not_crash(es_result):
    tidy = es_result.tidy()
    event_rows = tidy[tidy["type"] == "event_study"]
    assert len(event_rows) > 0
    assert np.isfinite(event_rows["estimate"]).all()
    # Exporters that delegate to tidy() previously inherited the crash.
    assert isinstance(es_result.to_markdown(), str)
    assert isinstance(es_result.to_latex(), str)


def test_plots_do_not_crash(es_result):
    import matplotlib.pyplot as plt

    es_result.plot()
    es_result.event_study_plot()
    plt.close("all")


def test_pretrends_tools_run(es_result):
    pt = sp.pretrends_test(es_result)
    assert 0.0 <= pt["pvalue"] <= 1.0
    # pretrends_summary wraps pretrends_test; it must not raise either.
    sp.pretrends_summary(es_result)
    pp = sp.pretrends_power(es_result)
    assert 0.0 <= pp["power"] <= 1.0


def test_honest_did_and_breakdown_accept_event_study(es_result):
    # Both extract the event-study table and require the 'att' column.
    sp.honest_did(es_result)
    sp.breakdown_m(es_result)
