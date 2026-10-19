"""``sp.panel_match`` against R ``PanelMatch`` 3.1.5.

Reference numbers: ``_fixtures/panel_match_R.json``, produced by
``_fixtures/_generate_panel_match_R.R`` on the synthetic panels written by
``_fixtures/_generate_panel_match_data.py``. R is not run in CI.

Deterministic pieces are T2 (1e-10): matched sets, refinement weights,
point estimates at every lead, the conditional and unconditional analytic
standard errors, the covariate-balance tables (refined and unrefined).
Bootstrap standard errors are stochastic screens (R's ``sample()`` stream
cannot be reproduced): only that they lie within a generous band.

Three documented departures, each pinned by an identity:

* ``forbid_treatment_reversal``: when every control of a set violates the
  restriction R keeps the set with its original (violating) controls; we
  drop it. ``test_forbid_reversal_drops_the_set_r_keeps``.
* placebo test: for a treated observation within ``k - 2`` periods of the
  panel end R's estimate at ``t-k`` loses its ``Y[t-1]`` terms (the row it
  would sit on does not exist). ``test_placebo_r_truncation_at_panel_end``.
* CBPS refinement: ``sp.cbps``'s over-identified ATT fit is not R
  ``CBPS``'s (see ``sp.cbps``); screened only.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FX = Path(__file__).parent / "_fixtures"
TOL = 1e-10

CASES = {
    "none": dict(data="full", lag=4, refinement="none", qoi="att", lead=range(4)),
    "maha": dict(
        data="full",
        lag=4,
        qoi="att",
        lead=range(4),
        size_match=5,
        covariates=["x1", "x2"],
    ),
    "maha_lagged": dict(
        data="full",
        lag=3,
        qoi="att",
        lead=range(3),
        size_match=4,
        covariates=["lag(y, 1:3)", "lag(x1, 1:2)", "x2"],
    ),
    "maha_diag": dict(
        data="full",
        lag=3,
        qoi="att",
        lead=range(2),
        size_match=3,
        covariates=["x1", "x2"],
        diagonal_covariance=True,
    ),
    "ps_weight": dict(
        data="full",
        lag=4,
        refinement="ps_weight",
        qoi="att",
        lead=range(3),
        covariates=["lag(y, 1:4)", "lag(x1, 1:4)"],
    ),
    "ps_match": dict(
        data="full",
        lag=4,
        refinement="ps_match",
        qoi="att",
        lead=range(3),
        size_match=5,
        covariates=["lag(y, 1:2)", "x1", "x2"],
    ),
    "atc_maha": dict(
        data="full",
        lag=3,
        qoi="atc",
        lead=range(3),
        size_match=5,
        covariates=["x1", "x2"],
    ),
    "art_ps": dict(
        data="full",
        lag=3,
        refinement="ps_weight",
        qoi="art",
        lead=range(2),
        covariates=["lag(y, 1:2)", "x1"],
    ),
    "exact": dict(
        data="full",
        lag=3,
        qoi="att",
        lead=range(2),
        size_match=3,
        covariates=["x1", "x2"],
        exact_match=["region"],
    ),
    "restrict": dict(
        data="full",
        lag=4,
        refinement="none",
        qoi="att",
        lead=range(2),
        restrict_control_period=2,
    ),
    "gaps_missing": dict(
        data="gaps",
        lag=3,
        qoi="att",
        lead=range(3),
        size_match=5,
        covariates=["lag(y, 1:2)", "x1", "x2"],
    ),
    "gaps_strict": dict(
        data="gaps",
        lag=3,
        refinement="ps_weight",
        qoi="att",
        lead=range(2),
        match_missing=False,
        covariates=["lag(y, 1:2)", "x1"],
    ),
}


@pytest.fixture(scope="module")
def R():
    return json.loads((FX / "panel_match_R.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def panels():
    return {
        "full": pd.read_csv(FX / "panel_match_data.csv"),
        "gaps": pd.read_csv(FX / "panel_match_gaps.csv"),
    }


def _fit(panels, cfg, **extra):
    cfg = dict(cfg)
    df = panels[cfg.pop("data")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.panel_match(
            df,
            "y",
            "d",
            id="id",
            time="t",
            balance_covariates=["x1", "x2", "y"],
            **cfg,
            **extra,
        )


def _r_sets(entries):
    out = {}
    for s in entries:
        ctrl = [int(c) for c in np.atleast_1d(s["controls"])] if s["controls"] else []
        w = [float(v) for v in np.atleast_1d(s["weights"])] if s["controls"] else []
        out[s["key"]] = dict(zip(ctrl, w))
    return out


def _py_sets(frame, qoi):
    out = {}
    for (u, t), g in frame[frame.qoi == qoi].groupby(["treated_unit", "time"]):
        g = g.dropna(subset=["control_unit"])
        out[f"{int(u)}.{int(t)}"] = dict(zip(g.control_unit.astype(int), g.weight))
    return out


def _assert_same_sets(r, p):
    assert set(r) == set(p)
    for key in r:
        assert set(r[key]) == set(p[key]), key
        for c, w in r[key].items():
            assert abs(p[key][c] - w) < TOL, (key, c)


@pytest.mark.parametrize("name", sorted(CASES))
def test_sets_weights_estimates_and_analytic_se(name, R, panels):
    ref = R["cases"][name]
    cond = _fit(panels, CASES[name], se_method="conditional")
    uncond = _fit(panels, CASES[name], se_method="unconditional")
    _assert_same_sets(
        _r_sets(ref["sets"]),
        _py_sets(cond.model_info["matched_sets"], CASES[name]["qoi"]),
    )
    np.testing.assert_allclose(
        cond.detail["estimate"], ref["est"]["estimate"], rtol=0, atol=TOL
    )
    np.testing.assert_allclose(
        cond.detail["se"], ref["est"]["se_conditional"], rtol=0, atol=TOL
    )
    np.testing.assert_allclose(
        uncond.detail["se"], ref["est"]["se_unconditional"], rtol=0, atol=TOL
    )


@pytest.mark.parametrize(
    "name", sorted(n for n in CASES if CASES[n]["qoi"] in ("att", "atc"))
)
def test_covariate_balance(name, R, panels):
    ref = R["cases"][name]["balance"]
    res = _fit(panels, CASES[name], se_method="conditional")
    for key, ours in (("refined", "balance"), ("unrefined", "balance_unrefined")):
        expected = pd.DataFrame(ref[key], index=ref["rows"], columns=ref["cols"])
        got = res.model_info[ours].copy()
        got.index = [i.replace("t-", "t_") for i in got.index]
        np.testing.assert_allclose(
            got.loc[expected.index, expected.columns], expected, rtol=0, atol=TOL
        )


def test_ate_point_estimate_and_both_set_families(R, panels):
    ref = R["cases"]["ate_maha"]
    cfg = dict(
        data="full",
        lag=3,
        qoi="ate",
        lead=range(2),
        size_match=5,
        covariates=["x1", "x2"],
    )
    res = _fit(panels, cfg, n_boot=200, seed=0)
    frame = res.model_info["matched_sets"]
    _assert_same_sets(_r_sets(ref["sets_att"]), _py_sets(frame, "att"))
    _assert_same_sets(_r_sets(ref["sets_atc"]), _py_sets(frame, "atc"))
    np.testing.assert_allclose(
        res.detail["estimate"], ref["est"]["estimate"], rtol=0, atol=TOL
    )


def test_forbid_reversal_drops_the_set_r_keeps(R, panels):
    """R keeps set 15.7 although each of its controls is treated before the
    last lead; every other set and weight agrees."""
    ref = _r_sets(R["cases"]["forbid"]["sets"])
    cfg = dict(
        data="full",
        lag=4,
        qoi="att",
        lead=range(4),
        size_match=5,
        covariates=["x1", "x2"],
        forbid_treatment_reversal=True,
    )
    res = _fit(panels, cfg, se_method="conditional")
    ours = _py_sets(res.model_info["matched_sets"], "att")
    assert set(ref) - set(ours) == {"15.7"}
    D = panels["full"].pivot(index="id", columns="t", values="d")
    for c in ref["15.7"]:
        assert D.loc[c, 6] == 0 and D.loc[c, 7:10].max() == 1  # violates the rule
    del ref["15.7"]
    _assert_same_sets(ref, ours)
    # our estimate is the same ratio over the sets both agree on
    Y = panels["full"].pivot(index="id", columns="t", values="y")
    effects = []
    for f in range(4):
        tot = 0.0
        for key, ctrl in ref.items():
            if not ctrl:
                continue
            i, t = map(int, key.split("."))
            c = np.array(list(ctrl))
            w = np.array(list(ctrl.values()))
            tot += (Y.loc[i, t + f] - Y.loc[i, t - 1]) - w @ (
                Y.loc[c, t + f] - Y.loc[c, t - 1]
            ).to_numpy()
        effects.append(tot / sum(1 for v in ref.values() if v))
    np.testing.assert_allclose(res.detail["estimate"], effects, rtol=0, atol=TOL)


def _placebo_fit(panels, method):
    return _fit(
        panels,
        dict(
            data="full",
            lag=4,
            qoi="att",
            lead=range(2),
            size_match=5,
            covariates=["x1", "x2"],
        ),
        se_method=method,
        placebo_test=True,
    ).model_info["placebo_test"]


def test_placebo_matches_r_away_from_the_panel_end(R, panels):
    ref = R["placebo"]
    assert ref["names"] == ["t-4", "t-3", "t-2"]
    cond = _placebo_fit(panels, "conditional")
    uncond = _placebo_fit(panels, "unconditional")
    # t-3 and t-2 need rows up to t+1 <= last lead: no truncation in R
    np.testing.assert_allclose(
        cond["estimate"][1:], ref["estimate"][1:], rtol=0, atol=TOL
    )
    np.testing.assert_allclose(
        cond["se"][1:], ref["se_conditional"][1:], rtol=0, atol=TOL
    )
    np.testing.assert_allclose(
        uncond["se"][1:], ref["se_unconditional"][1:], rtol=0, atol=TOL
    )


def test_placebo_r_truncation_at_panel_end(R, panels):
    """Ours is the pre-trend DiD from t-4 to t-1 on R's own placebo sets;
    R's number is that formula with Y[t-1] dropped where t + 2 is past the
    last period."""
    ref = R["placebo"]
    sets = _r_sets(ref["sets"])
    Y = panels["full"].pivot(index="id", columns="t", values="y")
    last = Y.columns.max()
    full, truncated, n = 0.0, 0.0, 0
    for key, ctrl in sets.items():
        if not ctrl:
            continue
        i, t = map(int, key.split("."))
        c = np.array(list(ctrl))
        w = np.array(list(ctrl.values()))
        keep = 1.0 if t + 2 <= last else 0.0
        full += (Y.loc[i, t - 1] - Y.loc[i, t - 4]) - w @ (
            Y.loc[c, t - 1] - Y.loc[c, t - 4]
        ).to_numpy()
        truncated += (keep * Y.loc[i, t - 1] - Y.loc[i, t - 4]) - w @ (
            keep * Y.loc[c, t - 1] - Y.loc[c, t - 4]
        ).to_numpy()
        n += 1
    assert abs(truncated / n - ref["estimate"][0]) < TOL
    ours = _placebo_fit(panels, "conditional")
    assert abs(ours["estimate"].iloc[0] - full / n) < TOL


def test_cbps_refinement_screen(R, panels):
    """S-level: sp.cbps's ATT fit is not R CBPS's; estimates stay close."""
    ref = R["cases"]["cbps_weight"]
    cfg = dict(
        data="full",
        lag=3,
        refinement="cbps_weight",
        qoi="att",
        lead=range(2),
        covariates=["lag(y, 1:2)", "x1"],
    )
    res = _fit(panels, cfg, se_method="conditional")
    assert set(_r_sets(ref["sets"])) == set(
        _py_sets(res.model_info["matched_sets"], "att")
    )
    np.testing.assert_allclose(
        res.detail["estimate"], ref["est"]["estimate"], atol=0.01
    )


@pytest.mark.parametrize("name", ["maha", "atc_maha", "ps_weight"])
def test_bootstrap_se_screen(name, R, panels):
    """S-level: unit bootstrap SE against R's 500-draw SE (MC error ~3%
    plus heavy tails of a ratio); a 15% band, not a parity claim."""
    res = _fit(panels, CASES[name], se_method="bootstrap", n_boot=2000, seed=11)
    r_se = np.asarray(R["cases"][name]["est"]["boot_se"])
    assert np.all(np.abs(res.detail["se"].to_numpy() / r_se - 1) < 0.15)
