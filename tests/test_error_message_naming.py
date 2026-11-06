"""Error messages name the function that failed and the offending input.

rOpenSci G5.2a asks that a message identify where it came from. These
messages used to be raised word for word from several modules ("alpha must
be in (0, 1)", "data must be a pandas DataFrame."); each now starts with
the public name and, where it is a value, reports what was passed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "y": rng.normal(size=50),
            "d": rng.integers(0, 2, 50),
            "x": rng.normal(size=50),
        }
    )


def test_qte_names_itself_and_the_bad_quantile(df):
    with pytest.raises(ValueError) as exc:
        sp.qte(df, y="y", treat="d", quantiles=[0.5, 1.5])
    assert str(exc.value) == (
        "sp.qte: quantiles must lie strictly inside (0, 1); got [0.5, 1.5]."
    )


@pytest.mark.parametrize(
    "fn, name",
    [(sp.bayes_regress, "sp.bayes_regress"), (sp.bma, "sp.bma")],
)
def test_bayes_entry_points_name_the_wrong_data_type(fn, name):
    with pytest.raises(MethodIncompatibility) as exc:
        fn("y ~ x", data=[1, 2])
    assert str(exc.value).startswith(
        f"{name}: data must be a pandas DataFrame; got list."
    )


def test_alpha_messages_differ_by_function_and_report_the_value(df):
    with pytest.raises(MethodIncompatibility) as a:
        sp.residual_balance(df, y="y", treat="d", covariates=["x"], alpha=1.5)
    with pytest.raises(MethodIncompatibility) as b:
        sp.rd_optimized(df, y="y", x="x", alpha=1.5)
    msg_a, msg_b = str(a.value).splitlines()[0], str(b.value).splitlines()[0]
    assert msg_a == "sp.residual_balance: alpha must be in (0, 1); got 1.5."
    assert msg_b == "sp.rd_optimized: alpha must be in (0, 1); got 1.5."


def test_constant_outcome_names_the_estimator(df):
    with pytest.raises(MethodIncompatibility) as exc:
        sp.gp_regress("y ~ x", data=df.assign(y=1.0))
    assert str(exc.value).splitlines()[0] == (
        "sp.gp_regress: The outcome does not vary."
    )
