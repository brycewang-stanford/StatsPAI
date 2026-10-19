"""Estimation sample of ``sp.lee_bounds`` against Stata ``leebounds``.

Reference: Stata 18, ``leebounds`` (Tauchmann, SSC v1.5) on
``_fixtures/leebounds_missing.csv`` from
``_fixtures/_generate_leebounds_missing_stata.do``. ``leebounds`` drops a row
from the whole computation when its treatment or selection indicator is
missing, or when it is selected and its outcome is missing. Before
2026-10-19 the second rule was missing here and both bounds came back NaN,
and a missing selection indicator raised an unrelated error. With
``tight()``, ``leebounds`` keeps rows whose covariate is missing in the
untrimmed arm's mean but leaves them out of every cell; ``sp.lee_bounds``
drops them and warns, and it equals Stata once the rows are dropped first
(the ``xdrop`` rows). Without ``select()`` a unit is selected when its
outcome is observed (``selection=None``); ``cieffect`` is ``result.ci``. Bounds held to 1e-12, analytic SEs to 1e-8, as in
``test_lee_bounds_leebounds_Stata_parity.py``.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"

_CASES = {
    "ymiss": dict(y="y2", treat="d", selection="s"),
    "dmiss": dict(y="y", treat="d2", selection="s"),
    "smiss": dict(y="y", treat="d", selection="s3"),
    "xdrop": dict(y="y", treat="d", selection="s", covariates=["xm"]),
    # no select(): selected = outcome observed
    "nosel": dict(y="y", treat="d", selection=None),
    "cie90": dict(y="y", treat="d", selection="s", alpha=0.10),
    "tight90": dict(y="y", treat="d", selection="s", covariates=["x"], alpha=0.10),
}


@pytest.fixture(scope="module")
def ref():
    return json.loads(
        (_FIX / "leebounds_missing_stata.json").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "leebounds_missing.csv", float_precision="round_trip")


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("case", sorted(_CASES))
def test_sample_rules_match_leebounds(ref, data, k, case):
    R = ref[f"{k}_{case}"]
    d = data[data["sample"] == k]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.lee_bounds(d, se_method="analytic", trimming="leebounds", **_CASES[case])
    mi = r.model_info
    assert mi["lower_bound"] == pytest.approx(R["lower"], rel=1e-12, abs=1e-12)
    assert mi["upper_bound"] == pytest.approx(R["upper"], rel=1e-12, abs=1e-12)
    assert mi["se_lower"] == pytest.approx(R["se_lower"], rel=1e-8)
    assert mi["se_upper"] == pytest.approx(R["se_upper"], rel=1e-8)
    if "cilower" in R:
        # cieffect is the Imbens-Manski interval, result.ci. leebounds
        # searches C_n on a grid of step 0.001 (program leebci); sp solves
        # the equation. Rebuilding leebci's grid from our bounds and SEs gives
        # Stata's interval, and our root lies within one grid step of it.
        alpha = _CASES[case].get("alpha", 0.05)
        c_grid = _leebci_grid(mi, alpha)
        assert mi["lower_bound"] - mi["se_lower"] * c_grid == pytest.approx(
            R["cilower"], rel=1e-8
        )
        assert mi["upper_bound"] + mi["se_upper"] * c_grid == pytest.approx(
            R["ciupper"], rel=1e-8
        )
        assert abs(mi["imbens_manski_cn"] - c_grid) <= 1e-3
        assert r.ci[0] == pytest.approx(
            mi["lower_bound"] - mi["se_lower"] * mi["imbens_manski_cn"]
        )


def _leebci_grid(mi, alpha):
    """``leebci``'s C_n: the point of the 0.001 grid from z_{1-a} to z_{1-a/2}
    minimising the squared coverage error (first minimum kept)."""
    from scipy.stats import norm

    lo, hi = norm.ppf(1 - alpha), norm.ppf(1 - alpha / 2)
    width = mi["upper_bound"] - mi["lower_bound"]
    smax = max(mi["se_lower"], mi["se_upper"])
    grid = lo + 0.001 * np.arange(int(np.floor((hi - lo) / 0.001 + 1e-9)) + 1)
    err = (norm.cdf(grid + width / smax) - norm.cdf(-grid) - (1 - alpha)) ** 2
    return float(grid[int(np.argmin(err))])


def test_no_missing_outcome_without_selection_raises(data):
    d = data[data["sample"] == 1].assign(y=lambda f: f["y"].fillna(0.0))
    with pytest.raises(sp.DataInsufficient, match="all observations selected"):
        sp.lee_bounds(d, y="y", treat="d", n_bootstrap=20)


def test_dropped_rows_are_reported(data):
    d = data[data["sample"] == 1]
    n_bad = int(((d["s"] == 1) & d["y2"].isna()).sum())
    assert n_bad > 0
    with pytest.warns(UserWarning, match=f"{n_bad} selected but with a missing"):
        sp.lee_bounds(d, y="y2", treat="d", selection="s", n_bootstrap=20)


@pytest.mark.parametrize("col,role", [("d", "treatment"), ("s", "selection")])
def test_indicator_not_coded_01_raises(data, col, role):
    # Stata: "selection indicator ... incorrectly specified"; a 1/2 treatment
    # is accepted by leebounds but changes its lower bound, so it is refused.
    d = data[data["sample"] == 1].copy()
    d[col] = d[col] + 1
    with pytest.raises(sp.MethodIncompatibility, match=f"{role} indicator"):
        sp.lee_bounds(d, y="y", treat="d", selection="s", n_bootstrap=20)
