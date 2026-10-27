"""Option scan round 6 (2026-10-27): ``sp.fast`` inference conventions.

Each test pins the documented behaviour against R ``fixest`` 0.14.0 (run
as a black box on the same data; numbers below) or against an identity.

- ``sp.fast.event_study`` charged the unit fixed effects to the CR1
  small-sample factor although they are nested in the (default, unit)
  clusters. Its docstring promises the ``reghdfe`` / ``fixest`` rule, which
  leaves nested effects out of ``K``; the standard errors were 5.5% too
  large.
- ``FeolsResult.tidy()`` referred clustered t statistics to
  ``t(df_resid)``; ``fixest`` (and Stata ``regress, vce(cluster)``) use
  ``t(G - 1)``. With 7 clusters the p-value of ``x1`` was 0.00024 against
  fixest's 0.0097. ``sp.fast.etable`` placed its stars by the same rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _fe_panel() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    n = 300
    firm = rng.integers(0, 30, n)
    year = rng.integers(0, 6, n)
    firm[:3] = [100, 101, 102]  # three singletons
    st = firm % 7  # firms nested in states
    x1 = rng.normal(size=n) + 0.1 * firm
    x2 = rng.normal(size=n)
    w = rng.uniform(0.3, 3, n)
    y = (
        1
        + 0.5 * x1
        - 0.3 * x2
        + 0.05 * firm
        + 0.2 * year
        + rng.normal(size=n) * (1 + 0.5 * np.abs(x1))
    )
    return pd.DataFrame(dict(y=y, x1=x1, x2=x2, firm=firm, year=year, st=st, w=w))


def _event_panel() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n_units, n_periods = 40, 10
    d = pd.DataFrame(
        [(u, t) for u in range(n_units) for t in range(n_periods)],
        columns=["unit", "time"],
    )
    g = np.where(np.arange(n_units) < 25, rng.integers(3, 8, n_units), -1)
    d["g"] = g[d.unit]
    d["et"] = np.where(d.g > 0, d.time - d.g, np.nan)
    d["region"] = d.unit % 5
    d["y"] = (
        0.1 * d.unit
        + 0.2 * d.time
        + np.where(d.et >= 0, 1 + 0.1 * d.et.fillna(0), 0)
        + rng.normal(size=len(d))
    )
    return d


# fixest 0.14.0: feols(y ~ i(et2, ref = c(-1, -1000)) | unit + time, d,
# cluster = ~unit / ~region), et2 = -1000 for never-treated rows. First four
# event-time coefficients (-7, -6, -5, -4).
_FIXEST_ES_COEF = [0.1949603640, -0.2079963123, 0.3931977147, -0.3233753763]
_FIXEST_ES_SE_UNIT = [0.6004056911, 0.3484367215, 0.3638886445, 0.3760742668]
_FIXEST_ES_SE_REGION = [0.5063093709, 0.3320433006, 0.4165373782, 0.5168642365]


@pytest.mark.parametrize(
    "cluster, expected",
    [(None, _FIXEST_ES_SE_UNIT), ("region", _FIXEST_ES_SE_REGION)],
)
def test_fast_event_study_cr1_leaves_nested_unit_fe_out_of_k(cluster, expected):
    d = _event_panel()
    res = sp.fast.event_study(
        d, y="y", unit="unit", time="time", event_time="et", cluster=cluster
    )
    assert list(res.event_times[:4]) == [-7, -6, -5, -4]
    # Coefficients agree to the demeaning tolerance; the standard errors
    # carry the same factor, so a wrong K shows up as a 5% gap.
    np.testing.assert_allclose(res.coefs[:4], _FIXEST_ES_COEF, rtol=1e-7)
    np.testing.assert_allclose(res.ses[:4], expected, rtol=1e-7)


def test_fast_event_study_se_equals_fast_feols_on_the_same_dummies():
    # Identity: the event study is feols on its own dummies.
    d = _event_panel()
    res = sp.fast.event_study(
        d, y="y", unit="unit", time="time", event_time="et", cluster="region"
    )
    for k in res.event_times:
        d[f"e{int(k) + 100}"] = (d.et == k).astype(float)
    rhs = " + ".join(f"e{int(k) + 100}" for k in res.event_times)
    fit = sp.fast.feols(f"y ~ {rhs} | unit + time", d, vcov="cr1", cluster="region")
    np.testing.assert_allclose(res.ses, fit.se().to_numpy(), rtol=1e-9)


# fixest 0.14.0: pvalue(feols(y ~ x1 + x2 | firm + year, d, vcov = ~st))
_FIXEST_P_CR1_ST = [0.00973237596155, 0.06771962986888]
# vcov = "hetero": t(df_resid) as before (degrees_freedom(m, "t") = 260)
_FIXEST_P_HC1 = [7.45149750359e-05, 4.32231539931e-02]


def test_fast_feols_tidy_clustered_p_values_use_g_minus_1():
    d = _fe_panel()
    fit = sp.fast.feols("y ~ x1 + x2 | firm + year", d, vcov="cr1", cluster="st")
    p = fit.tidy()["Pr(>|t|)"].to_numpy()
    np.testing.assert_allclose(p, _FIXEST_P_CR1_ST, rtol=1e-6)


def test_fast_feols_tidy_hc1_p_values_keep_df_resid():
    d = _fe_panel()
    fit = sp.fast.feols("y ~ x1 + x2 | firm + year", d, vcov="hc1")
    p = fit.tidy()["Pr(>|t|)"].to_numpy()
    np.testing.assert_allclose(p, _FIXEST_P_HC1, rtol=1e-6)


def test_fast_etable_stars_follow_the_fit_p_values():
    d = _fe_panel()
    fit = sp.fast.feols("y ~ x1 + x2 | firm + year", d, vcov="cr1", cluster="st")
    tab = sp.fast.etable(fit, fmt="%.4f")
    # fixest's p-values are 0.0097 (x1) and 0.068 (x2); on the
    # 0.10 / 0.05 / 0.01 ladder that is *** and *. On t(df_resid) x2 read
    # p = 0.027 and printed **. Each cell is "<estimate><stars> (<se>)".
    assert tab.loc["x1"].iloc[0].split()[0] == "0.5055***"
    assert tab.loc["x2"].iloc[0].split()[0] == "-0.2321*"
