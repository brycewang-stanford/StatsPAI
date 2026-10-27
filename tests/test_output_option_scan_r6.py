"""Option scan round 6 (2026-10-27): intervals in exported tables and plots.

Ground truth is the result object itself: a table, an inline citation or a
coefficient plot must print the interval the result reports through
``conf_int(alpha)`` (``EconometricResults``) or ``b ± z·se`` (a
``CausalResult`` with normal inference).

- ``alpha`` other than 0.05 recomputed the bounds on ``t(N - K)`` for every
  result: a clustered OLS fit (``t(G - 1)``) and a logit (normal) got a
  different interval in the table than from the result.
- A fit called with ``alpha=0.10`` stores 90% bounds; the table reused them
  under its default ``alpha=0.05`` and labelled them "95% CI".
- ``sp.coefplot`` / ``sp.coefplot_tikz`` always drew ``b ± 1.96·se``.
- Under ``eform=[...]`` the decimals of a row depended on column order.
"""

from __future__ import annotations

import json
import re
import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp

matplotlib.use("Agg")


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n, n_groups = 200, 10
    g = np.repeat(np.arange(n_groups), n // n_groups)
    x = rng.normal(size=n) + rng.normal(size=n_groups)[g]
    y = 1 + 0.3 * x + rng.normal(size=n_groups)[g] + rng.normal(size=n)
    d = (rng.random(n) < 1 / (1 + np.exp(-x))).astype(int)
    t = (rng.random(n) < 0.5).astype(int)
    return pd.DataFrame(dict(y=y + t, x=x, g=g, d=d, t=t))


def _ci_cell(table, row: str, col: int = 0):
    frame = table.to_dataframe()
    pos = list(frame.index).index(row)
    cell = frame.iloc[pos + 1, col]
    lo, hi = re.match(r"\[(.+), (.+)\]", cell).groups()
    return float(lo), float(hi)


def _close(got, want):
    # 6 decimals are printed; compare at that resolution.
    np.testing.assert_allclose(got, want, atol=5e-7, rtol=0)


@pytest.mark.parametrize("alpha", [0.10, 0.01])
def test_clustered_ols_interval_at_other_levels_uses_g_minus_1(data, alpha):
    m = sp.regress("y ~ x", data=data, vce="cluster g")
    assert m.inference_rule()["df"] == 9
    tab = sp.regtable(m, se_type="ci", alpha=alpha, fmt="%.6f")
    _close(_ci_cell(tab, "x"), m.conf_int(alpha).loc["x"].to_numpy())


def test_logit_interval_at_other_levels_uses_the_normal(data):
    lg = sp.logit("d ~ x", data=data)
    tab = sp.regtable(lg, se_type="ci", alpha=0.10, fmt="%.6f")
    want = lg.conf_int(0.10).loc["x"].to_numpy()
    z = stats.norm.ppf(0.95)
    b, se = lg.params["x"], lg.std_errors["x"]
    np.testing.assert_allclose(want, [b - z * se, b + z * se], rtol=1e-12)
    _close(_ci_cell(tab, "x"), want)


def test_fit_level_bounds_are_not_relabelled(data):
    lg = sp.logit("d ~ x", data=data, alpha=0.10)
    tab = sp.regtable(lg, se_type="ci", fmt="%.6f")  # asks for 95%
    _close(_ci_cell(tab, "x"), lg.conf_int(0.05).loc["x"].to_numpy())
    tab90 = sp.regtable(lg, se_type="ci", alpha=0.10, fmt="%.6f")
    _close(
        _ci_cell(tab90, "x"),
        [lg.conf_int_lower["x"], lg.conf_int_upper["x"]],
    )


def test_causal_result_fit_level_bounds_are_not_relabelled(data):
    r = sp.ipw(data, y="y", treat="t", covariates=["x"], alpha=0.10)
    z = stats.norm.ppf(0.975)
    tab = sp.regtable(r, se_type="ci", fmt="%.6f")
    row = tab.to_dataframe().index[0]
    _close(_ci_cell(tab, row), [r.estimate - z * r.se, r.estimate + z * r.se])


def test_esttab_and_cite_follow_the_same_rule(data):
    m = sp.regress("y ~ x", data=data, vce="cluster g")
    tab = sp.esttab(m, ci=True, alpha=0.10, fmt="%.6f")
    want = m.conf_int(0.10).loc["x"].to_numpy()
    frame = tab.to_dataframe()
    pos = list(frame.index).index("x")
    lo, hi = re.match(r"\[(.+), (.+)\]", frame.iloc[pos + 1, 0]).groups()
    _close([float(lo), float(hi)], want)

    text = sp.cite(m, "x", second_row="ci", alpha=0.10, fmt="%.6f")
    lo, hi = re.search(r"\[(.+), (.+)\]", text).groups()
    _close([float(lo), float(hi)], want)

    lg = sp.logit("d ~ x", data=data, alpha=0.10)
    text = sp.cite(lg, "x", second_row="ci", fmt="%.6f")
    lo, hi = re.search(r"\[(.+), (.+)\]", text).groups()
    _close([float(lo), float(hi)], lg.conf_int(0.05).loc["x"].to_numpy())


def test_vcov_rereport_carries_its_reference_distribution(data):
    m = sp.regress("y ~ x", data=data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tab = sp.regtable(m, vcov="cluster g", se_type="ci", alpha=0.10, fmt="%.6f")
        refit = sp.vcov(m, "cluster g")
    _close(_ci_cell(tab, "x"), refit.conf_int(0.10).loc["x"].to_numpy())


def test_to_dict_round_trip_keeps_the_interval_rule(data):
    m = sp.regress("y ~ x", data=data, vce="cluster g")
    lg = sp.logit("d ~ x", data=data, alpha=0.10)
    tab = sp.regtable(m, lg, se_type="ci", alpha=0.10, fmt="%.6f")
    payload = json.loads(json.dumps(tab.to_dict()))
    back = sp.output.regression_table.RegtableResult.from_dict(payload)
    assert back.to_dataframe().equals(tab.to_dataframe())


def test_coefplot_draws_the_fit_interval(data):
    import matplotlib.pyplot as plt

    m = sp.regress("y ~ x", data=data, vce="cluster g")
    for alpha in (0.05, 0.10):
        fig, ax = sp.coefplot(m, variables=["x"], alpha=alpha)
        bars = [c for c in ax.collections if hasattr(c, "get_segments")]
        seg = bars[0].get_segments()[0]
        np.testing.assert_allclose(
            sorted(seg[:, 0]), m.conf_int(alpha).loc["x"].to_numpy(), rtol=1e-12
        )
        plt.close(fig)


def test_coefplot_tikz_draws_the_fit_interval(data):
    m = sp.regress("y ~ x", data=data, vce="cluster g")
    tikz = sp.coefplot_tikz(m, variables=["x"], level=0.90)
    err = float(re.search(r"\+- \(([0-9.e-]+),0\)", tikz).group(1))
    lo, hi = m.conf_int(0.10).loc["x"].to_numpy()
    assert err == pytest.approx((hi - lo) / 2, rel=1e-5)  # printed at %.6g


def test_eform_precision_does_not_depend_on_column_order():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(size=n) * 30
    d = (rng.random(n) < 1 / (1 + np.exp(-x / 30))).astype(int)
    df = pd.DataFrame(dict(x=x, d=d))
    lg = sp.logit("d ~ x", data=df)
    ols = sp.regress("d ~ x", data=df)
    a = sp.regtable(lg, ols, eform=[True, False]).to_dataframe()
    b = sp.regtable(ols, lg, eform=[False, True]).to_dataframe()
    assert a.iloc[:, 0].tolist() == b.iloc[:, 1].tolist()
    assert a.iloc[:, 1].tolist() == b.iloc[:, 0].tolist()
