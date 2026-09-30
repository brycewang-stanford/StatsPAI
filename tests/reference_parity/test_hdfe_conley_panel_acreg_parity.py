"""``sp.hdfe_ols(vce='conley')`` on a panel = Stata ``acreg ..., pfe1() pfe2()``.

Through 1.33 the Conley option of ``hdfe_ols`` pooled every row into one
spatial kernel: on a panel a unit's own rows in different years (distance
zero) and all cross-period pairs of neighbours were treated as correlated,
and the n x n distance matrix limited it to cross-sections (Web of Power
replication, QJE 2023). With ``conley_time`` / ``conley_unit`` /
``conley_lag`` it now follows ``acreg``'s spatial + time HAC -- the time
kernel alone within a unit, spatial kernel x time kernel across units at
``conley_lag_cross`` -- on the within design, in O(neighbour pairs) memory.

Reference: Stata ``acreg`` with ``pfe1(id) pfe2(t)`` on
``_fixtures/hdfe_conley_panel.csv`` (``_generate_hdfe_conley_panel_Stata.do``).
``acreg``'s ``hac`` is the Bartlett *time* kernel, its ``bartlett`` the
Bartlett *spatial* kernel. Coefficients and SEs agree to 1e-12.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
STATA = json.loads((_FIX / "hdfe_conley_panel_Stata.json").read_text(encoding="utf-8"))
CASES = {
    "lag2_hac_bartlett": dict(
        conley_lag=2, conley_kernel="bartlett", conley_time_kernel="bartlett"
    ),
    "lag0": dict(conley_lag=0, conley_kernel="uniform", conley_time_kernel="uniform"),
    "lag3_lagdist3_hac_bartlett": dict(
        conley_lag=3,
        conley_lag_cross=3,
        conley_kernel="bartlett",
        conley_time_kernel="bartlett",
    ),
}


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "hdfe_conley_panel.csv")


@pytest.mark.parametrize("key", list(CASES))
def test_matches_acreg(data, key):
    r = sp.hdfe_ols(
        "y ~ x1 + x2 | id + t",
        data=data,
        vce="conley",
        conley_lat="lat",
        conley_lon="lon",
        conley_cutoff=300.0,
        conley_time="t",
        conley_unit="id",
        **CASES[key],
    )
    ref = STATA[key]
    assert r.n_obs == ref["N"]
    for j, v in enumerate(["x1", "x2"]):
        assert r.params[v] == pytest.approx(ref["b"][j], rel=1e-12)
        assert r.std_errors[v] == pytest.approx(ref["se"][j], rel=1e-12)


def test_pooled_kernel_on_a_panel_warns(data):
    with pytest.warns(UserWarning, match="conley_time"):
        sp.hdfe_ols(
            "y ~ x1 + x2 | id + t",
            data=data,
            vce="conley",
            conley_lat="lat",
            conley_lon="lon",
            conley_cutoff=300.0,
        )


def test_half_specified_panel_request_raises(data):
    with pytest.raises(sp.MethodIncompatibility, match="conley_unit"):
        sp.hdfe_ols(
            "y ~ x1 + x2 | id + t",
            data=data,
            vce="conley",
            conley_lat="lat",
            conley_lon="lon",
            conley_cutoff=300.0,
            conley_time="t",
        )
