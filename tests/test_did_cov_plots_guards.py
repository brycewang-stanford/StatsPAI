"""Coverage campaign — did/plots.py guard branches + the honest-DiD sensitivity
forest (``did/plots.py``).

Part of the core-module ≥95% coverage initiative
(see ``.coverage_campaign/CAMPAIGN.md``). Adds margin above the hard-95% line by
exercising the cheap, deterministic *input-guard* branches of the result-based
DiD plotters (each raises a specific, informative error on a malformed result)
plus the ``sensitivity_plot`` renderer fed a well-formed honest-DiD-style
DataFrame.

Guards are tested with light ``SimpleNamespace`` stand-ins for the result
object — these exercise validation control flow only, not any numerical path
(no estimator is mocked).
"""

from __future__ import annotations

import types

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


# --------------------------------------------------------------------------
# event_study_plot — missing event-study estimates
# --------------------------------------------------------------------------
def test_event_study_plot_requires_event_study():
    from statspai.did.plots import event_study_plot

    bogus = types.SimpleNamespace(model_info={})
    with pytest.raises(ValueError, match="no event study estimates"):
        event_study_plot(bogus)


# --------------------------------------------------------------------------
# group_time_plot — missing group-time detail
# --------------------------------------------------------------------------
def test_group_time_plot_requires_group_detail():
    bogus = types.SimpleNamespace(detail=pd.DataFrame({"time": [1, 2]}))
    with pytest.raises(ValueError, match="group-time detail"):
        sp.group_time_plot(bogus)


# --------------------------------------------------------------------------
# cohort_event_study_plot — three distinct guards
# --------------------------------------------------------------------------
def test_cohort_event_study_plot_requires_group_detail():
    bogus = types.SimpleNamespace(detail=pd.DataFrame({"time": [1, 2]}))
    with pytest.raises(ValueError, match="group-time detail"):
        sp.cohort_event_study_plot(bogus)


def test_cohort_event_study_plot_requires_relative_time():
    bogus = types.SimpleNamespace(
        detail=pd.DataFrame({"group": [2, 2], "att": [0.1, 0.2]})
    )
    with pytest.raises(ValueError, match="relative_time"):
        sp.cohort_event_study_plot(bogus)


def test_cohort_event_study_plot_requires_treated_cohort():
    # only never-treated (group == 0) → no treated cohorts to plot
    bogus = types.SimpleNamespace(
        detail=pd.DataFrame(
            {"group": [0, 0], "relative_time": [-1, 0], "att": [0.0, 0.0]}
        )
    )
    with pytest.raises(ValueError, match="No treated cohorts"):
        sp.cohort_event_study_plot(bogus)


# --------------------------------------------------------------------------
# sensitivity_plot — empty guard + a real honest-DiD-style render
# --------------------------------------------------------------------------
def test_sensitivity_plot_rejects_empty():
    with pytest.raises(ValueError, match="Empty sensitivity"):
        sp.sensitivity_plot(pd.DataFrame())


def test_sensitivity_plot_renders_breakdown_frontier():
    # Minimal honest-DiD sensitivity table: as the relative-magnitudes bound M
    # grows, the robust CI widens and eventually fails to reject zero.
    sens = pd.DataFrame(
        {
            "M": [0.0, 0.5, 1.0, 1.5, 2.0],
            "ci_lower": [0.20, 0.10, -0.05, -0.20, -0.40],
            "ci_upper": [0.80, 0.90, 1.00, 1.15, 1.35],
            "rejects_zero": [True, True, False, False, False],
        }
    )
    fig0, ax0 = plt.subplots()
    fig, ax = sp.sensitivity_plot(
        sens, original_estimate=0.5, original_ci=(0.2, 0.8), ax=ax0
    )
    assert ax is ax0
    assert isinstance(fig, plt.Figure)
    # the breakdown frontier draws at least the CI band / markers
    assert len(ax.collections) + len(ax.get_lines()) >= 1
