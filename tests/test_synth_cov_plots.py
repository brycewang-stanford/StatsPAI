"""Branch coverage for ``sp.synthplot`` — the unified SCM plot dispatcher.

Part of the core-module >=95% coverage campaign (``.coverage_campaign/``);
contributes the ``synth`` plotting layer. Naming follows the campaign
convention ``test_<mod>_cov_<topic>.py``.

``src/statspai/synth/plots.py`` is a single public entry point
(``synthplot(result, type=...)``) that routes to ~15 private ``_plot_*``
renderers. Before this module only ``type='trajectory'`` and ``'gap'``
were exercised; here we drive every renderer reachable from the bundled
California Proposition-99 teaching dataset so the plotting layer is
smoke-covered rather than silently untested.

These are rendering smoke tests: they assert that each plot type returns
a Matplotlib ``Figure`` (and does not raise) under the ``Agg`` backend.
They deliberately do not pin pixel output — the numerical content of the
synthetic control is covered by the estimator parity suites; here we only
guard that the visualisation code paths execute on real fitted results.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: no display needed in CI
import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

import statspai as sp


@pytest.fixture(scope="module")
def prop99():
    return sp.datasets.california_prop99()


@pytest.fixture(scope="module")
def _base():
    return dict(
        outcome="cigsale",
        unit="state",
        time="year",
        treated_unit="California",
        treatment_time=1989,
    )


@pytest.fixture(scope="module")
def basic_result(prop99, _base):
    """Classic SCM fit (default ``placebo=True``)."""
    return sp.synth(prop99, **_base)


@pytest.fixture(scope="module")
def placebo_result(prop99, _base):
    """Fit carrying a placebo distribution for placebo / rmspe plots."""
    return sp.synth(prop99, inference="placebo", **_base)


@pytest.fixture(scope="module")
def conformal_result(prop99, _base):
    """Fit carrying conformal CIs for the conformal plot."""
    return sp.conformal_synth(prop99, **_base)


def _assert_figure(out):
    """``synthplot`` returns ``(fig, ax)`` or ``(fig, axes)``."""
    assert isinstance(out, tuple) and len(out) == 2
    fig = out[0]
    assert isinstance(fig, Figure)
    plt.close("all")


# Plot types renderable from a standard SCM fit.
@pytest.mark.parametrize("plot_type", ["trajectory", "gap", "both", "weights"])
def test_synthplot_basic_types(basic_result, plot_type):
    _assert_figure(sp.synthplot(basic_result, type=plot_type))


# Plot types that need an inference / placebo distribution in the result.
@pytest.mark.parametrize("plot_type", ["placebo", "placebo_gap", "rmspe"])
def test_synthplot_placebo_types(placebo_result, plot_type):
    _assert_figure(sp.synthplot(placebo_result, type=plot_type))


def test_synthplot_conformal(conformal_result):
    _assert_figure(sp.synthplot(conformal_result, type="conformal"))


def test_synthplot_compare(prop99, _base, basic_result):
    """``type='compare'`` overlays multiple results passed as a list."""
    other = sp.synth(prop99, method="demeaned", **_base)
    _assert_figure(
        sp.synthplot(
            [basic_result, other], type="compare", labels=["Classic", "De-meaned"]
        )
    )


@pytest.mark.parametrize("plot_type", ["trajectory", "gap"])
def test_synthplot_pre_band_overlay(basic_result, plot_type):
    """``pre_band=True`` overlays the ±1.96×pre-RMSPE noise envelope."""
    _assert_figure(sp.synthplot(basic_result, type=plot_type, pre_band=True))


def test_synthplot_unknown_type_raises(basic_result):
    with pytest.raises(ValueError, match="Unknown plot type"):
        sp.synthplot(basic_result, type="definitely_not_a_plot_type")
