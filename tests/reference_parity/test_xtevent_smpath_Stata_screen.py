"""``sp.xtevent_smpath`` vs Stata ``xteventplot, smpath(line)`` (xtevent 3.1.0).

Stata prints, for each polynomial degree, the least Wald statistic of a path
through the normalised coefficient, then optimises the leading coefficient
numerically and documents the result as approximate. So:

* the critical value and the per-degree least Wald statistics are
  deterministic least-squares quantities and agree to the 9 significant
  digits Stata displays; the degree chosen is therefore the same;
* the path itself is a stochastic screen (S): Stata's optimiser stops near
  the optimum (here within 1e-7 of the closed form on four designs, 2e-4 on
  the clustered window(-3 4) one). Where they differ, the closed form is at
  least as good: its path is inside the Wald region and its leading
  coefficient is no larger in magnitude than that of Stata's path.

Fixture: ``_fixtures/xtevent_smpath_Stata.json`` from
``_fixtures/_generate_xtevent_smpath_Stata.do`` (Stata 18 MP; xtevent used as
a black box).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient

_FIX = pathlib.Path(__file__).parent / "_fixtures"
REF = json.loads((_FIX / "xtevent_smpath_Stata.json").read_text(encoding="utf-8"))
_CONT = pd.read_csv(_FIX / "xtevent_continuous.csv")
_BIN = pd.read_csv(_FIX / "xtevent_binary.csv")

CASES = {
    "cont_w2": (_CONT, dict(window=2, covariates=["x"]), {}),
    "cont_w34_cluster": (
        _CONT,
        dict(window=(-3, 4), covariates=["x"], cluster="cl"),
        {},
    ),
    "cont_w3_norm2": (_CONT, dict(window=3, covariates=["x"], norm=-2), {}),
    "cont_w3_postwindow2": (_CONT, dict(window=3, covariates=["x"]), {"postwindow": 2}),
    "bin_w4": (_BIN, dict(window=4), {}),
    "bin_w4_cluster_maxorder2": (_BIN, dict(window=4, cluster="cl"), {"maxorder": 2}),
}


def _fit(key):
    data, kw, smkw = CASES[key]
    r = sp.xtevent(data, y="y", policy="z", panel="id", time="t", **kw)
    return r, smkw


def _stata_path(key):
    return pd.Series({int(k): v for k, v in REF[key]["smpath"].items()}).sort_index()


@pytest.mark.parametrize("key", [k for k in CASES if REF[k]["smpath"] is not None])
def test_wald_by_degree_and_critical_value_match_stata(key):
    r, smkw = _fit(key)
    out = sp.xtevent_smpath(r, **smkw)
    c = REF[key]
    assert out.attrs["critical"] == pytest.approx(c["critical"], rel=1e-8)
    np.testing.assert_allclose(
        out.attrs["wald_by_order"], c["wald_by_order"], rtol=1e-8
    )
    assert out.attrs["order"] == len(c["wald_by_order"]) - 1


def test_no_polynomial_within_maxorder_is_refused_like_stata():
    r, smkw = _fit("bin_w4_cluster_maxorder2")
    with pytest.raises(DataInsufficient, match="Wald") as err:
        sp.xtevent_smpath(r, **smkw)
    c = REF["bin_w4_cluster_maxorder2"]
    np.testing.assert_allclose(
        err.value.diagnostics["wald_by_order"], c["wald_by_order"], rtol=1e-8
    )
    assert err.value.recovery_hint


@pytest.mark.parametrize("key", [k for k in CASES if REF[k]["smpath"] is not None])
def test_path_screen_against_stata(key):
    r, smkw = _fit(key)
    out = sp.xtevent_smpath(r, **smkw).set_index("relative_time")
    st = _stata_path(key)
    ours = out.loc[st.index, "smpath"]
    scale = float(np.max(np.abs(st)))
    # S-level screen: Stata's optimiser is approximate.
    assert np.max(np.abs(ours.to_numpy() - st.to_numpy())) <= 1e-3 * scale

    # The closed form is at least as good as Stata's path: fit Stata's path
    # exactly with a polynomial of the same degree and compare.
    deg = out.attrs["order"]
    k = st.index.to_numpy(dtype=float)
    lead_stata = np.polyfit(k, st.to_numpy(), deg)[0]
    lead_ours = out.attrs["coefficients"][-1]
    assert abs(lead_ours) <= abs(lead_stata) * (1 + 1e-6) + 1e-12
    assert out.attrs["wald"] <= out.attrs["critical"] * (1 + 1e-9)
