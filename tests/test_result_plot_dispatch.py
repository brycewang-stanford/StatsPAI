"""Plot dispatch on result objects (2026-10-19 audit).

* ``CausalResult.plot(type="nonsense")`` drew a coefplot; it now raises.
* ``sp.rddensity(...).plot()`` raised "'yerr' must not contain negative
  values": its headline is a T statistic while ``ci`` is on the density
  scale. It now draws the two boundary densities.
* ``sp.coefplot`` warns when the models' coefficients live on different
  scales; ``drop=`` removes terms (Stata ``coefplot, drop(_cons)``).
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import statspai as sp  # noqa: E402


@pytest.fixture(scope="module")
def df():
    r = np.random.default_rng(0)
    out = pd.DataFrame({"x": r.normal(size=400)})
    out["d"] = (r.random(400) < 0.5).astype(int)
    out["y"] = 1 + out.x + 2 * out.d + r.normal(size=400)
    out["yb"] = (out.y > out.y.median()).astype(int)
    return out


def test_unknown_type_raises_with_the_valid_list(df):
    res = sp.ipw(df, y="y", treat="d", covariates=["x"], n_bootstrap=10)
    with pytest.raises(sp.MethodIncompatibility) as info:
        res.plot(type="nonsense")
    assert "event_study" in info.value.recovery_hint


def test_econometric_results_plot(df):
    m = sp.regress("y ~ x + d", data=df)
    fig, ax = m.plot(drop=["Intercept"])
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == ["d", "x"]
    with pytest.raises(sp.MethodIncompatibility):
        m.plot(type="residuals_typo")


def test_rddensity_plot_draws_the_boundary_densities():
    r = np.random.default_rng(0)
    # Density 1/3 below the cutoff and 2/3 above (uniform pieces).
    x = np.r_[r.uniform(-1, 0, 4000), r.uniform(0, 1, 8000)]
    res = sp.rddensity(pd.DataFrame({"x": x}), x="x")
    fig, ax = res.plot()
    points = [line.get_ydata()[0] for line in ax.lines if line.get_marker() == "o"]
    mi = res.model_info
    assert points == [mi["density_left"], mi["density_right"]]
    # Known truth within four standard errors of each boundary estimate.
    assert abs(mi["density_left"] - 1 / 3) < 4 * mi["se_left"]
    assert abs(mi["density_right"] - 2 / 3) < 4 * mi["se_right"]
    fig2, _ = res.plot(type="density")
    assert fig2 is not None


def test_coefplot_warns_on_mixed_scales(df):
    ols = sp.regress("yb ~ x", data=df)
    logit = sp.logit("yb ~ x", data=df)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sp.coefplot(ols, logit)
    messages = [str(w.message) for w in caught]
    assert any("log-odds" in m and "linear" in m for m in messages)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sp.coefplot(ols, sp.regress("y ~ x", data=df))
    assert not any("different scales" in str(w.message) for w in caught)


def test_coefplot_keeps_the_intercept_unless_dropped(df):
    m = sp.regress("y ~ x", data=df)
    _, ax = sp.coefplot(m)
    assert "Intercept" in [t.get_text() for t in ax.get_yticklabels()]
    _, ax = sp.coefplot(m, drop=["Intercept"])
    assert [t.get_text() for t in ax.get_yticklabels()] == ["x"]


def test_cox_plot_type_alias_and_refusal():
    r = np.random.default_rng(1)
    d = pd.DataFrame({"x": r.normal(size=200)})
    d["t"] = r.exponential(np.exp(-0.5 * d.x))
    d["e"] = 1
    res = sp.cox(data=d, duration="t", event="e", x=["x"])
    assert res.plot(type="hr") is not None
    with pytest.raises(sp.MethodIncompatibility):
        res.plot(type="nonsense")
    with pytest.raises(ValueError):  # still a ValueError for old callers
        res.plot(kind="nonsense")
