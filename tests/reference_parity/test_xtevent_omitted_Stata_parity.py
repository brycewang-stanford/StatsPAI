"""``sp.xtevent`` when an event-time regressor is collinear (Stata omits it).

With ``window=3`` on a ten-period panel only ``t = 5..7`` keep every lead
and lag, so the left endpoint ``1 - z_{t+3}`` is constant within every
unit and the right endpoint is collinear with the rest. Stata ``xtevent``
3.1.0 drops both (``o.`` in ``e(b)``), reports them as 0 and goes on;
before 2026-10-20 ``sp.xtevent`` raised ``KeyError``.

Reference: ``_fixtures/xtevent_omitted_Stata.json`` from
``_fixtures/_generate_xtevent_omitted_Stata.do`` on
``_fixtures/xtevent_omitted.csv``. T2 at 1e-10: every event-time
coefficient and SE, N.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FX = pathlib.Path(__file__).parent / "_fixtures"
STATA = json.loads((FX / "xtevent_omitted_Stata.json").read_text(encoding="utf-8"))
CASES = {
    "w3": dict(window=3),
    "w3_cluster": dict(window=3, cluster="id"),
    "w3_trend3_ols": dict(window=3, trend=-3, trend_method="ols"),
}


def _k(name):
    tag = name.split("_k_eq_")[1]
    return -int(tag[1:]) if tag[0] == "m" else int(tag[1:])


@pytest.mark.parametrize("engine", ["areg", "reghdfe"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_omitted_endpoints_match_stata(case, engine):
    df = pd.read_csv(FX / "xtevent_omitted.csv")
    if engine == "reghdfe" and "cluster" in CASES[case]:
        pytest.skip("reghdfe clustering has its own small-sample factor")
    with pytest.warns(UserWarning, match="omitted because of collinearity"):
        res = sp.xtevent(
            df, y="y", policy="z", panel="id", time="t", engine=engine, **CASES[case]
        )
    ref = STATA[case]
    names = ref["names"].split()
    es = res.model_info["event_study"].set_index("relative_time")
    assert res.model_info["n_obs"] == ref["N"]
    omitted_ref = sorted(_k(n) for n in names if n.startswith("o._k_eq_"))
    assert omitted_ref == [-4, 4]
    assert sorted(_k(n) for n in res.model_info["omitted"]) == omitted_ref
    assert es.loc[omitted_ref, "is_omitted"].all()
    for nm, b, se in zip(names, ref["b"], ref["se"]):
        if "_k_eq_" not in nm:
            continue
        k = _k(nm)
        np.testing.assert_allclose(es.loc[k, "estimate"], b, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(es.loc[k, "se"], se, rtol=1e-10, atol=1e-12)


def test_static_policy_collinear_is_refused():
    df = pd.read_csv(FX / "xtevent_omitted.csv")
    df["z"] = (df["id"] % 2).astype(float)  # time-invariant: absorbed by unit FE
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="not identified"):
        sp.xtevent(df, y="y", policy="z", panel="id", time="t", static=True)
