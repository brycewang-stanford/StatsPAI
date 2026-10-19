"""``sp.didregress(aggregate=...)`` against Stata 18 ``aggregate()``.

References: ``_fixtures/didregress_dlang_Stata.csv``, written by
``_fixtures/_generate_didregress_dlang_Stata.do`` on the data of
``_fixtures/_generate_didregress_dlang_data.py`` (a repeated cross-section
with ten groups and six periods; a panel of 60 units in eight groups).

The ATET and its standard error agree to 1e-10 for ``dlang``,
``dlang, varying`` and ``standard``, with within-cell covariates, a
cell-level covariate and none, on both data types. The confidence bounds
agree when the t reference uses Stata's degrees of freedom
(``dlang_df='stata'``); the default uses the residual degrees of freedom
of the cell regression, which is the exact distribution of the OLS t
statistic under normal errors (asserted below from first principles).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility

FX = Path(__file__).parent / "_fixtures"
TOL = 1e-10

CASES = {
    "rcs_dlang_xw": ("rcs", "dlang", ["x", "w"]),
    "rcs_varying_xw": ("rcs", "dlang_varying", ["x", "w"]),
    "rcs_dlang_xwz": ("rcs", "dlang", ["x", "w", "z"]),
    "rcs_varying_xwz": ("rcs", "dlang_varying", ["x", "w", "z"]),
    "rcs_dlang_none": ("rcs", "dlang", []),
    "rcs_standard_xw": ("rcs", "standard", ["x", "w"]),
    "rcs_standard_xwz": ("rcs", "standard", ["x", "w", "z"]),
    "panel_dlang_x": ("panel", "dlang", ["x"]),
    "panel_varying_x": ("panel", "dlang_varying", ["x"]),
    "panel_standard_x": ("panel", "standard", ["x"]),
}


@pytest.fixture(scope="module")
def ref():
    return pd.read_csv(FX / "didregress_dlang_Stata.csv").set_index("case")


@pytest.fixture(scope="module")
def data():
    return {
        "rcs": pd.read_csv(FX / "didregress_dlang.csv"),
        "panel": pd.read_csv(FX / "didregress_dlang_panel.csv"),
    }


def _fit(data, kind, method, covs, **kw):
    extra = {"id": "id"} if kind == "panel" else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.didregress(
            data[kind],
            "y",
            "d",
            group="g",
            time="t",
            covariates=covs,
            aggregate=method,
            **extra,
            **kw,
        )


@pytest.mark.parametrize("case", sorted(CASES))
def test_estimate_se_and_stata_interval(case, ref, data):
    kind, method, covs = CASES[case]
    r = ref.loc[case]
    res = _fit(data, kind, method, covs, dlang_df="stata")
    assert abs(res.estimate - r.b) < TOL
    assert abs(res.se - r.se) < TOL
    assert res.model_info["n_cells"] == r.N
    assert res.model_info["df_stata"] == r.df_r
    assert abs(res.ci[0] - r.lb) < 1e-9 and abs(res.ci[1] - r.ub) < 1e-9


@pytest.mark.parametrize("case", ["rcs_dlang_xw", "rcs_dlang_xwz", "panel_varying_x"])
def test_default_df_is_the_cell_regression_residual_df(case, data):
    """cells - (treatment + cell covariates + G + T - 1 effects)."""
    kind, method, covs = CASES[case]
    res = _fit(data, kind, method, covs)
    mi = res.model_info
    k = 1 + len(mi["cell_covariates"])
    assert mi["df"] == mi["n_cells"] - k - (mi["n_groups"] + mi["n_periods"] - 1)
    crit = stats.t.ppf(0.975, mi["df"])
    assert abs(res.ci[1] - (res.estimate + crit * res.se)) < 1e-12


def test_ols_t_statistic_has_the_residual_df_under_normal_errors():
    """Known truth: with normal homoskedastic cell errors the dlang t
    statistic is exactly t(residual df). A 2,000-draw check of the 99th
    percentile of |t| against t(df) and against Stata's larger df."""
    rng = np.random.default_rng(0)
    cells = pd.MultiIndex.from_product(
        [range(1, 6), range(1, 5)], names=["g", "t"]
    ).to_frame(index=False)
    cells["d"] = ((cells.g <= 2) & (cells.t >= 3)).astype(int)
    tvals = []
    for _ in range(2000):
        df = cells.assign(y=rng.normal(size=len(cells)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sp.didregress(df, "y", "d", group="g", time="t", aggregate="dlang")
        tvals.append(res.estimate / res.se)
    mi = res.model_info
    assert mi["df"] == 20 - 1 - 8 and mi["df_stata"] == 20 - 1 - 4
    q = np.quantile(np.abs(tvals), 0.99)
    # t(11) 99.5% = 3.106, t(15) 99.5% = 2.947; MC sd of the quantile ~0.07
    assert abs(q - stats.t.ppf(0.995, mi["df"])) < 0.2


def test_refusals(data):
    with pytest.raises(MethodIncompatibility):
        _fit(data, "rcs", "dlang", ["x"], cluster="g")
    with pytest.raises(MethodIncompatibility):
        _fit(data, "rcs", "bogus", ["x"])
    small = data["rcs"].groupby(["g", "t"]).head(2)
    with pytest.raises(DataInsufficient, match="dlang_varying"):
        _fit({"rcs": small}, "rcs", "dlang_varying", ["x", "w"])
