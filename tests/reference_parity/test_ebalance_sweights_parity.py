"""``sp.ebalance(weights=)`` (sampling weights) vs R ``WeightIt`` 1.7.0.

``weights=`` is WeightIt's ``s.weights``, not its ``base.weights``: the
targets are the sampling-weighted moments of the target group, the
reweighted arm is balanced in the sampling-weighted sense and both
outcome means carry the sampling weights. Fixture:
``_fixtures/ebalance_sweights_R.json`` from
``_fixtures/_generate_ebalance_sweights_R.R`` (WeightIt is GPL, used as a
black box). Sampling weights ``sw = 0.4 + (row % 5) * 0.3`` and base
weights ``bw = 0.5 + (row % 4) / 4`` (rows numbered from 1).
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

_FIX = pathlib.Path(__file__).parent / "_fixtures"
REF = json.loads((_FIX / "ebalance_sweights_R.json").read_text(encoding="utf-8"))
CASES = sorted(k for k in REF if not k.startswith("_"))


def _frame(c):
    df = pd.read_csv(_FIX / c["data"])
    r = np.arange(1, len(df) + 1)
    df["sw"] = 0.4 + (r % 5) * 0.3
    df["bw"] = 0.5 + (r % 4) / 4
    return df


def _fit(key):
    c = REF[key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return c, sp.ebalance(
            _frame(c),
            y="y",
            treat="d",
            covariates=c["covariates"],
            estimand=c["estimand"],
            weights="sw",
            base_weights="bw" if c["base_weights"] else None,
        )


@pytest.mark.parametrize("key", CASES)
def test_estimate_matches_weightit(key):
    # 2e-9 relative: on Lalonde ATC with base weights WeightIt's own
    # standardized moment gap is 1.3e-10 (ours 4e-14), which moves the
    # estimate by 1.1e-9 of itself
    c, r = _fit(key)
    assert abs(r.estimate - c["est"]) <= 2e-9 * max(1.0, abs(c["est"]))


@pytest.mark.parametrize("key", CASES)
def test_mestimation_se_matches_weightit(key):
    c, r = _fit(key)
    assert abs(r.se - c["se"]) / c["se"] < 1e-7


@pytest.mark.parametrize("key", CASES)
def test_balancing_weights_match_weightit(key):
    # weights_full excludes the sampling weights, as WeightIt's $weights;
    # each reweighted arm is compared up to its normalisation
    c, r = _fit(key)
    df = _frame(c)
    w = r.model_info["weights_full"]
    ref = np.asarray(c["w"], dtype=float)
    for arm in (0, 1):
        sel = df["d"].to_numpy() == arm
        ratio = w[sel] / ref[sel]
        assert np.ptp(ratio) <= 1e-8 * ratio.mean()


def test_unit_sampling_weights_change_nothing():
    c = REF[CASES[0]]
    df = _frame(c)
    df["one"] = 1.0
    a = sp.ebalance(df, "y", "d", c["covariates"], weights="one")
    b = sp.ebalance(df, "y", "d", c["covariates"])
    assert abs(a.estimate - b.estimate) < 1e-12 and abs(a.se - b.se) < 1e-12


def test_refuses_naive_and_dof_adjust():
    c = REF[CASES[0]]
    df = _frame(c)
    with pytest.raises(MethodIncompatibility):
        sp.ebalance(df, "y", "d", c["covariates"], weights="sw", vce="naive")
    with pytest.raises(MethodIncompatibility):
        sp.ebalance(df, "y", "d", c["covariates"], weights="sw", dof_adjust=True)
