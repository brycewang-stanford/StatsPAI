"""Several treated units with staggered adoption in ``sp.scest(treat=)``.

Reference: R ``scpi`` 4.0.1 ``scdataMulti`` + ``scest`` (GPL-3), used as a
black box through its documented interface on the committed synthetic
panel (``_fixtures/_generate_scest_multi_{data.py,R.R}``).

What is checked, and why this is evidence
-----------------------------------------
* R's own outputs are reproduced from R's weights by the construction the
  implementation uses (one problem per treated unit on its pre-treatment
  periods before ``adoption - anticipation``, the never-treated units as
  donors, effects ``'unit-time'`` / ``'unit'`` / ``'time'`` as the gaps,
  their unit means, and their means across units at horizons
  ``1..min_i T1_i``) to 1e-12.
* Each unit's weights: R hands the problem to a conic solver that stops at
  a 1e-8 relative gap, StatsPAI solves it exactly (the single-unit solver of
  ``test_did_synth_scpi_parity.py``). As there, R's point is feasible, its
  objective exceeds ours by at most 1e-7 relative, and strong convexity
  bounds the weight distance by the gap; ``ols`` and ``lasso`` agree to
  1e-9 directly, and the data-driven ridge radius to 1e-10.
* StatsPAI's reported effects are the same construction applied to its own
  weights (identity).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility

FIX = Path(__file__).parent / "_fixtures"
R = json.loads((FIX / "scest_multi_R.json").read_text(encoding="utf-8"))
PANEL = pd.read_csv(FIX / "scest_multi_panel.csv")
ADOPT = {"u01": 2012, "u02": 2014, "u03": 2015}
DONORS = sorted(set(PANEL.unit) - set(ADOPT))
WIDE = PANEL.pivot(index="year", columns="unit", values="y")
CASES = sorted(k for k in R if k != "versions")


def _parse(case):
    parts = case.split("|")
    effect, constr = parts[0], parts[1]
    anticipation = 1 if "anticipation1" in parts else 0
    units = ["u01", "u03"] if "units_est" in parts else sorted(ADOPT)
    return effect, constr, anticipation, units


def _blocks(units, anticipation):
    """Per-unit (A, B, P, Y_post) from the CSV, by the scpi construction."""
    years = WIDE.index.to_numpy()
    out = {}
    for u in units:
        start = years[np.searchsorted(years, ADOPT[u]) - anticipation]
        pre, post = years < start, years >= start
        out[u] = (
            WIDE.loc[pre, u].to_numpy(),
            WIDE.loc[pre, DONORS].to_numpy(),
            WIDE.loc[post, DONORS].to_numpy(),
            WIDE.loc[post, u].to_numpy(),
        )
    return out


def _aggregate(effect, blocks, weights):
    fits = {u: blocks[u][2] @ weights[u] for u in blocks}
    if effect == "unit-time":
        return np.concatenate([fits[u] for u in blocks])
    if effect == "unit":
        return np.array([fits[u].mean() for u in blocks])
    reach = min(len(f) for f in fits.values())
    return np.mean([fits[u][:reach] for u in blocks], axis=0)


def _r_weights(case, units):
    w = np.asarray(R[case]["w"], dtype=float).reshape(len(units), len(DONORS))
    return {u: w[i] for i, u in enumerate(units)}


def _fit(case):
    effect, constr, anticipation, units = _parse(case)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.scest(
            PANEL, outcome="y", unit="unit", time="year", treat="treat",
            treated_unit=None if len(units) == 3 else units, effect=effect,
            anticipation=anticipation, w_constr=constr,
        )  # fmt: skip


@pytest.mark.parametrize("case", CASES)
def test_r_output_is_one_fit_per_unit_with_never_treated_donors(case):
    effect, _, anticipation, units = _parse(case)
    assert R[case]["donors"][: len(DONORS)] == [f"{units[0]}.{d}" for d in DONORS]
    blocks = _blocks(units, anticipation)
    w_R = _r_weights(case, units)
    want = _aggregate(effect, blocks, w_R)
    got = np.asarray(R[case]["Y_post_fit"], dtype=float)
    np.testing.assert_allclose(got, want, rtol=1e-12)
    pre = np.concatenate([blocks[u][1] @ w_R[u] for u in units])
    np.testing.assert_allclose(R[case]["Y_pre_fit"], pre, rtol=1e-12)


@pytest.mark.parametrize("case", CASES)
def test_weights_are_optimal_within_the_solver_gap(case):
    effect, constr, anticipation, units = _parse(case)
    fit = _fit(case)
    blocks = _blocks(units, anticipation)
    w_R = _r_weights(case, units)
    assert list(fit["weights"].columns) == units
    assert list(fit["weights"].index) == DONORS
    for u in units:
        A, B = blocks[u][0], blocks[u][1]
        w_py = fit["weights"][u].to_numpy()

        def f(w):
            return float(np.sum((A - B @ w) ** 2))

        gap = f(w_R[u]) - f(w_py)
        # Ours may lose to R only by the rounding of evaluating f: the error
        # of each residual, eps (|A| + |B| |w|), through the sum of squares,
        # plus the summation itself. A fixed 1e-14 * f was tighter than that
        # on the unconstrained fits (Windows CI: 48 eps, bound 153 eps).
        r = A - B @ w_py
        rounding = np.finfo(float).eps * (
            2 * np.abs(r) @ (np.abs(A) + np.abs(B) @ np.abs(w_py)) + A.size * f(w_py)
        )
        assert gap >= -rounding
        assert gap <= 1e-7 * f(w_py)
        lam_min = np.linalg.eigvalsh(B.T @ B).min()
        assert np.linalg.norm(w_R[u] - w_py) <= np.sqrt(max(gap, 0) / lam_min) + 1e-9
        if constr in ("ols", "lasso"):
            np.testing.assert_allclose(w_py, w_R[u], rtol=0, atol=1e-9)
        spec = fit["w_constr_spec"][u]
        if constr != "ols":
            assert abs(spec["Q"] - R[case]["Q"][units.index(u)]) < 1e-10


@pytest.mark.parametrize("case", CASES)
def test_effects_are_the_same_construction_on_our_weights(case):
    effect, constr, anticipation, units = _parse(case)
    fit = _fit(case)
    blocks = _blocks(units, anticipation)
    ours = {u: fit["weights"][u].to_numpy() for u in units}
    np.testing.assert_allclose(
        fit["Y_synth_post"], _aggregate(effect, blocks, ours), rtol=1e-12
    )
    assert len(fit["effects"]) == len(R[case]["Y_post_fit"])
    np.testing.assert_allclose(
        fit["effects"]["effect"],
        fit["effects"]["outcome"] - fit["effects"]["synthetic"],
        rtol=1e-12,
    )
    if constr in ("ols", "lasso"):
        np.testing.assert_allclose(
            fit["Y_synth_post"], R[case]["Y_post_fit"], rtol=1e-9
        )
    if effect == "time":
        # horizons every treated unit reaches: 1..min_i T1_i
        assert list(fit["effects"]["horizon"]) == [
            int(n) for n in R[case]["Y_post_fit_names"]
        ]
        assert set(fit["effects"]["n_units"]) == {len(units)}


def test_one_treated_unit_equals_the_single_unit_fit():
    only = PANEL[(PANEL.unit == "u02") | PANEL.unit.isin(DONORS)]
    multi = sp.scest(only, outcome="y", unit="unit", time="year", treat="treat")
    single = sp.scest(
        only, outcome="y", unit="unit", time="year", treated_unit="u02",
        treatment_time=2014,
    )  # fmt: skip
    np.testing.assert_array_equal(multi["weights"]["u02"].to_numpy(), single["weights"])
    np.testing.assert_array_equal(multi["Y_synth_post"], single["Y_synth_post"])


def test_treatment_that_switches_off_is_refused():
    bad = PANEL.copy()
    bad.loc[(bad.unit == "u01") & (bad.year == 2018), "treat"] = 0
    with pytest.raises(MethodIncompatibility, match="absorbing"):
        sp.scest(bad, outcome="y", unit="unit", time="year", treat="treat")


def test_no_never_treated_donor_is_refused():
    treated = PANEL[PANEL.unit.isin(ADOPT)]
    with pytest.raises(DataInsufficient, match="never-treated"):
        sp.scest(treated, outcome="y", unit="unit", time="year", treat="treat")


@pytest.mark.parametrize(
    "kw,match",
    [
        (dict(treat="treat", treatment_time=2012), "treatment_time"),
        (dict(treated_unit="u01", treatment_time=2012, effect="time"), "treat="),
        (dict(treat="treat", effect="avg"), "effect="),
        (dict(treat="treat", treated_unit="u05"), "never receive"),
        (dict(), "treated_unit="),
    ],
)
def test_argument_errors(kw, match):
    with pytest.raises(MethodIncompatibility, match=match):
        sp.scest(PANEL, outcome="y", unit="unit", time="year", **kw)


def test_unit_treated_from_the_first_period_is_refused():
    early = PANEL.copy()
    early.loc[early.unit == "u01", "treat"] = 1
    with pytest.raises(DataInsufficient, match="no pre-treatment period"):
        sp.scest(early, outcome="y", unit="unit", time="year", treat="treat")


@pytest.mark.parametrize(
    "edit,kw,error,match",
    [
        (lambda d: d.assign(treat=d.treat.where(d.year != 2005)), {},
         MethodIncompatibility, "missing values"),
        (lambda d: d.assign(treat=d.treat * 2), {}, MethodIncompatibility, "0/1"),
        (lambda d: d, dict(anticipation=-1), MethodIncompatibility, "anticipation"),
        (lambda d: d.assign(treat=0), {}, DataInsufficient, "no unit is ever treated"),
    ],
)  # fmt: skip
def test_treatment_column_and_anticipation_are_validated(edit, kw, error, match):
    with pytest.raises(error, match=match):
        sp.scest(
            edit(PANEL), outcome="y", unit="unit", time="year", treat="treat", **kw
        )
