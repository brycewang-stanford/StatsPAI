"""Plots of results added in October 2026 that failed on their own examples.

Each test draws the plot and checks that what is drawn is a number the
result reports.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import statspai as sp  # noqa: E402


def test_best_subset_plot_draws_the_criterion_of_each_size():
    rng = np.random.default_rng(0)
    d = pd.DataFrame(rng.normal(size=(100, 4)), columns=list("abcd"))
    d["y"] = d.a + rng.normal(size=100)
    res = sp.best_subset(d, "y", list("abcd"))
    fig, ax = res.plot()
    drawn = ax.lines[0].get_ydata()
    np.testing.assert_array_equal(drawn, res.history["bic"].to_numpy())
    assert ax.get_ylabel() == "BIC"
    plt.close(fig)


def test_synth_ttest_plot_draws_the_treated_and_synthetic_paths():
    rng = np.random.default_rng(1)
    units, periods = 12, 30
    rows = []
    factor = rng.normal(size=periods).cumsum()
    for u in range(units):
        load = rng.uniform(0.5, 1.5)
        for t in range(periods):
            y = load * factor[t] + rng.normal(scale=0.3)
            if u == 0 and t >= 20:
                y += 2.0
            rows.append((u, t, y))
    df = pd.DataFrame(rows, columns=["unit", "time", "y"])
    res = sp.synth_ttest(df, "y", "unit", "time", treated_unit=0, treatment_time=20)
    out = res.plot()
    fig = out[0] if isinstance(out, tuple) else plt.gcf()
    mi = res.model_info
    treated = df[df.unit == 0].sort_values("time").y.to_numpy()
    np.testing.assert_array_equal(mi["Y_treated"], treated)
    # the post-period gap of the plotted synthetic control is att_sc
    post = np.asarray(mi["times"]) >= 20
    gap = mi["Y_treated"][post] - mi["Y_synth"][post]
    assert np.mean(gap) == mi["att_sc"]
    plt.close(fig)
