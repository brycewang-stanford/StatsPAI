"""``sp.did_imputation(pretrends=k)`` when some of the ``k`` leads are not
identified.

Without never-treated units the far leads are observed for the last cohort
only, and in the auxiliary regression their indicators are spanned by the
fixed effects and the nearer leads. Stata's ``did_imputation`` omits them
(Baker's simulated panel in Cunningham's Remix labs, ``pretrends(24)``:
``pre18`` to ``pre24`` are reported as zero and ``pre1`` to ``pre17`` are
unchanged). Up to 1.38.0 StatsPAI inverted the singular Gram matrix and
returned every lead with standard errors in the thousands, the identified
ones included. It now refuses and names the longest identified run.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    """Cohorts treated at 4, 7 and 10 observed over periods 1-9: the last
    cohort is untreated throughout and is the only one with leads past -6."""
    rng = np.random.default_rng(11)
    rows = []
    for unit in range(90):
        cohort = (4, 7, 10)[unit % 3]
        alpha = rng.normal()
        for t in range(1, 10):
            effect = 1.0 + 0.2 * (t - cohort) if t >= cohort else 0.0
            rows.append(
                {
                    "unit": unit,
                    "t": t,
                    "g": cohort,
                    "y": alpha + 0.3 * t + effect + rng.normal(scale=0.5),
                }
            )
    return pd.DataFrame(rows)


def _fit(panel: pd.DataFrame, k: int):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.did_imputation(
            panel, y="y", group="unit", time="t", first_treat="g", pretrends=k
        )


def _lead_regression(panel: pd.DataFrame, k: int) -> np.ndarray:
    """The auxiliary regression written out with dummies: untreated rows,
    unit and period effects, one indicator per lead."""
    u = panel[panel["t"] < panel["g"]].copy()
    rel = (u["t"] - u["g"]).to_numpy()
    X = np.column_stack(
        [pd.get_dummies(u["unit"]).to_numpy(float)]
        + [pd.get_dummies(u["t"]).to_numpy(float)[:, 1:]]
        + [(rel == -j).astype(float)[:, None] for j in range(1, k + 1)]
    )
    beta = np.linalg.lstsq(X, u["y"].to_numpy(), rcond=None)[0]
    return beta[-k:]


def test_identified_leads_equal_the_dummy_regression(panel):
    res = _fit(panel, 5)
    pre = res.detail[res.detail["relative_time"] < 0].sort_values(
        "relative_time", ascending=False
    )
    np.testing.assert_allclose(pre["att"], _lead_regression(panel, 5), atol=1e-8)
    assert float(pre["se"].max()) < 1.0


@pytest.mark.parametrize("k, missing", [(6, [-6]), (8, [-8, -7, -6])])
def test_unidentified_leads_are_refused_by_name(panel, k, missing):
    with pytest.raises(MethodIncompatibility) as err:
        _fit(panel, k)
    assert err.value.diagnostics["unidentified_leads"] == missing
    assert "pretrends=5" in str(err.value.recovery_hint)


def test_the_design_really_is_rank_deficient(panel):
    """Independent of the estimator: the dummy design with six leads has a
    null vector, so no coefficient vector is singled out by least squares."""
    u = panel[panel["t"] < panel["g"]]
    rel = (u["t"] - u["g"]).to_numpy()

    def rank(k: int) -> int:
        X = np.column_stack(
            [pd.get_dummies(u["unit"]).to_numpy(float)]
            + [pd.get_dummies(u["t"]).to_numpy(float)[:, 1:]]
            + [(rel == -j).astype(float)[:, None] for j in range(1, k + 1)]
        )
        return int(np.linalg.matrix_rank(X))

    assert rank(5) - rank(4) == 1
    assert rank(6) == rank(5)
