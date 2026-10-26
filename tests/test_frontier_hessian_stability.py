"""Standard errors of sp.frontier / sp.xtfrontier are not round-off.

Until 2026-10 both took the inverse of a plain h = 1e-5 central-difference
Hessian. Its round-off was ~1e-5 relative, enough that perturbing the
outcome by one ulp moved the intercept SE of Track A module 28 by 1.7e-5,
and the committed SE sat 1.3e-5 from R and Stata, which agree with each
other to 1e-8. A Richardson-extrapolated Hessian and a Newton finish to the
quasi-Newton search remove both. The truth here is the committed Stata
artifacts (analytic-Hessian ``frontier`` / ``xtfrontier``) on the same bytes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

PARITY = Path(__file__).resolve().parent / "r_parity"
STATA = Path(__file__).resolve().parent / "stata_parity" / "results"
TERMS = {"_cons": "beta_intercept", "lnk": "beta_lnk", "lnl": "beta_lnl"}


def _stata(module: str) -> dict[str, dict]:
    rows = json.loads((STATA / f"{module}_Stata.json").read_text(encoding="utf-8"))
    return {r["statistic"]: r for r in rows["rows"]}


def _data(module: str) -> pd.DataFrame:
    return pd.read_csv(PARITY / "data" / f"{module}.csv", float_precision="round_trip")


def _ulp_perturbed(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    signs = np.sign(np.random.default_rng(0).normal(size=len(out)))
    out[col] = np.nextafter(out[col].to_numpy(), signs * np.inf)
    return out


def _cross_section(df: pd.DataFrame):
    return sp.frontier(data=df, y="lny", x=["lnk", "lnl"], dist="half-normal")


def test_frontier_se_matches_stata_and_is_stable_to_one_ulp():
    df = _data("28_frontier")
    ref = _stata("28_frontier")
    fit = _cross_section(df)
    for term, stat in TERMS.items():
        assert fit.params[term] == pytest.approx(ref[stat]["estimate"], rel=1e-8)
        assert fit.std_errors[term] == pytest.approx(ref[stat]["se"], rel=1e-7)
    moved = _cross_section(_ulp_perturbed(df, "lny"))
    for term in TERMS:
        # Round-off in the old Hessian moved this by 1.7e-5.
        assert moved.std_errors[term] == pytest.approx(fit.std_errors[term], rel=1e-7)


def test_xtfrontier_pitt_lee_matches_stata_to_optimizer_precision():
    df = _data("29_panel_sfa")
    ref = _stata("29_panel_sfa")
    fit = sp.xtfrontier(
        data=df,
        y="lny",
        x=["lnk", "lnl"],
        id="unit",
        time="year",
        model="ti",
        dist="half-normal",
    )
    for term, stat in TERMS.items():
        # 2e-6 / 3e-6 before the Newton finish and the Richardson Hessian.
        assert fit.params[term] == pytest.approx(ref[stat]["estimate"], rel=1e-8)
        assert fit.std_errors[term] == pytest.approx(ref[stat]["se"], rel=1e-7)
