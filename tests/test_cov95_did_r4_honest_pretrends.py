"""Coverage round-4 — Rambachan-Roth honest DiD + pre-trends sensitivity.

Drives:
- ``sp.honest_did`` smoothness & relative-magnitude restrictions and its
  input-validation raises;
- ``sp.breakdown_m`` (the breakdown M*), incl. the bad-event-time raise;
- ``sp.sensitivity_rr`` and the ``SensitivityResult`` rich-display methods
  ``.plot()`` / ``._repr_html_()`` / ``.summary()`` — smoke-tested with the
  matplotlib Agg backend (assert a Figure/Axes and str come back, never
  pixel values);
- ``sp.pretrends_test`` / ``sp.pretrends_power`` / ``sp.pretrends_summary``.

All run on a real staggered CS result with a +2 ATT; assertions check
shapes, monotonic CI widening in M, p-values in [0, 1], and that bad
inputs raise — no fabricated numbers.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import statspai as sp  # noqa: E402
from statspai.exceptions import DataInsufficient, MethodIncompatibility  # noqa: E402


def _cs_panel(seed=0, n_units=120, n_periods=10, att=2.0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        g = [0, 5, 7][u % 3]
        fe = rng.normal()
        for t in range(1, n_periods + 1):
            treated_now = (g > 0) and (t >= g)
            y = fe + 0.3 * t + (att if treated_now else 0.0) + rng.normal(0, 0.4)
            rows.append({"unit": u, "time": t, "y": y, "g": g})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def cs_result():
    return sp.callaway_santanna(_cs_panel(), y="y", g="g", t="time", i="unit")


@pytest.mark.parametrize("method", ["smoothness", "relative_magnitude"])
def test_honest_did_methods(cs_result, method):
    tab = sp.honest_did(cs_result, e=0, method=method)
    assert len(tab) >= 1
    assert {"ci_lower", "ci_upper", "rejects_zero"}.issubset(tab.columns)
    # CI must contain (lower <= upper) and widen as the bound grows
    widths = (tab["ci_upper"] - tab["ci_lower"]).values
    assert (widths >= -1e-9).all()
    assert widths[-1] >= widths[0] - 1e-9


def test_honest_did_custom_mgrid(cs_result):
    tab = sp.honest_did(
        cs_result, e=0, method="relative_magnitude", m_grid=[0.0, 0.5, 1.0, 2.0]
    )
    assert len(tab) == 4


def test_honest_did_bad_method_raises(cs_result):
    with pytest.raises(MethodIncompatibility, match="method"):
        sp.honest_did(cs_result, e=0, method="not_a_method")


def test_honest_did_invalid_m_grid_raises(cs_result):
    with pytest.raises(MethodIncompatibility, match="m_grid"):
        sp.honest_did(cs_result, e=0, m_grid=[0.0, np.inf])


def test_honest_did_no_event_study_raises():
    # A frame with no event-study table cannot be honest-DiD'd.
    df = sp.dgp_did(n_units=40, n_periods=6, effect=1.0, seed=0)
    r = sp.did(df, y="y", treat="treated", time="time", id="unit", method="twfe")
    with pytest.raises((MethodIncompatibility, AttributeError, TypeError)):
        sp.honest_did(r, e=0)


def test_breakdown_m(cs_result):
    m_star = sp.breakdown_m(cs_result, e=0)
    assert np.isfinite(m_star)
    assert m_star >= 0.0


def test_breakdown_m_bad_e_raises(cs_result):
    with pytest.raises(DataInsufficient):
        sp.breakdown_m(cs_result, e=999)


# ── Rambachan-Roth SensitivityResult display methods ─────────────── #


@pytest.fixture(scope="module")
def sens_result(cs_result):
    return sp.sensitivity_rr(cs_result, method="C-LF")


def test_sensitivity_rr_basic(sens_result):
    assert len(sens_result.mbar_grid) >= 1
    assert len(sens_result.ci_lower) == len(sens_result.mbar_grid)
    assert (sens_result.ci_upper >= sens_result.ci_lower).all()


def test_sensitivity_summary_is_str(sens_result):
    s = sens_result.summary()
    assert isinstance(s, str) and len(s) > 0
    assert isinstance(repr(sens_result), str)


def test_sensitivity_repr_html(sens_result):
    html = sens_result._repr_html_()
    assert isinstance(html, str)
    assert "Sensitivity" in html
    assert "<table" in html


def test_sensitivity_plot_default(sens_result):
    ax = sens_result.plot()
    from matplotlib.axes import Axes

    assert isinstance(ax, Axes)
    assert ax.figure is not None
    plt.close("all")


def test_sensitivity_plot_supplied_ax(sens_result):
    fig, ax = plt.subplots()
    out = sens_result.plot(ax=ax, color="green")
    assert out is ax
    plt.close(fig)


# ── pretrends test / power / summary ─────────────────────────────── #


def test_pretrends_test(cs_result):
    res = sp.pretrends_test(cs_result)
    assert 0.0 <= res["pvalue"] <= 1.0


def test_pretrends_power(cs_result):
    res = sp.pretrends_power(cs_result)
    assert 0.0 <= res["power"] <= 1.0
    assert np.isfinite(res["noncentrality"])


def test_pretrends_summary_is_str(cs_result):
    s = sp.pretrends_summary(cs_result)
    assert isinstance(s, str) and "Pre-Trends" in s
