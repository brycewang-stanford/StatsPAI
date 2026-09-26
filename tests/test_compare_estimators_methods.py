"""Every ``sp.compare_estimators`` method must actually run.

The table is assembled inside a ``try`` per method, and a failing method only
leaves a ``UserWarning``. Three branches called their estimator with keywords
it does not take -- ``sp.match(treatment=)``, ``sp.dml(treatment=)`` and
``sp.causal_forest(treatment=, covariates=)`` -- so matching (in the default
method list), DML and the causal forest were silently absent from every
comparison.
"""

from __future__ import annotations

import warnings

import pytest

import statspai as sp

METHODS = ["ols", "matching", "ipw", "aipw", "g_computation", "dml", "causal_forest"]


@pytest.fixture(scope="module")
def comparison():
    df = sp.dgp_observational(n=400, seed=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = sp.compare_estimators(
            data=df,
            y="y",
            treatment="treatment",
            covariates=["x1", "x2"],
            methods=METHODS,
        )
    return out, [str(w.message) for w in caught]


def test_no_method_is_silently_dropped(comparison):
    out, messages = comparison
    assert not [m for m in messages if "failed" in m]
    assert len(out.estimates_table) == len(METHODS)


def test_default_methods_include_matching():
    df = sp.dgp_observational(n=400, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = sp.compare_estimators(
            data=df, y="y", treatment="treatment", covariates=["x1", "x2"]
        )
    assert "Propensity Score Matching" in set(out.estimates_table["method"])


def test_table_is_reproducible(comparison):
    first, _ = comparison
    df = sp.dgp_observational(n=400, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        again = sp.compare_estimators(
            data=df,
            y="y",
            treatment="treatment",
            covariates=["x1", "x2"],
            methods=["ipw"],
        )
    ipw = first.estimates_table.set_index("method").loc["Inverse Probability Weighting"]
    row = again.estimates_table.iloc[0]
    assert row["se"] == pytest.approx(ipw["se"], rel=1e-12)
