"""``sp.oaxaca``'s threefold decomposition against R ``oaxaca`` 0.1.5.

Endowments use Group B's coefficients, coefficients are evaluated at Group
B's means, and the interaction is the rest (Stata ``oaxaca, threefold``;
R ``oaxaca(...)$threefold``). It does not depend on ``reference=``. R
``oaxaca`` is GPL and used as a black box: ``_fixtures/_generate_oaxaca_threefold_R.R``
on ``tests/agent_bench/data/wage_gap.csv``.
"""

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

HERE = pathlib.Path(__file__).parent
R = json.loads(
    (HERE / "_fixtures" / "oaxaca_threefold_R.json").read_text(encoding="utf-8")
)
DATA = HERE.parent / "agent_bench" / "data" / "wage_gap.csv"


@pytest.mark.parametrize("reference", [0, 1, "pooled"])
def test_threefold_matches_r_oaxaca(reference):
    d = pd.read_csv(DATA)
    o = sp.oaxaca(
        d, y="log_wage", group="female", x=["educ", "exper"], reference=reference
    ).overall
    for key in ("gap", "endowments", "coefficients", "interaction"):
        np.testing.assert_allclose(o[key], R[key], rtol=1e-10)
    assert o["endowments"] + o["coefficients"] + o["interaction"] == pytest.approx(
        o["gap"], abs=1e-12
    )
    assert o["endowments_se"] > 0 and o["coefficients_se"] > 0
