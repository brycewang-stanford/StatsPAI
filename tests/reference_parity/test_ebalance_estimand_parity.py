"""``sp.ebalance(estimand=, base_weights=)`` vs R ``WeightIt`` 1.7.0.

``WeightIt::weightit(method = "ebal", estimand = "ATT" | "ATE" | "ATC",
base.weights = )`` solves the same entropy-balancing problems (unique
solutions): the ATT reweights the controls to the treated moments, the ATC
the treated to the control moments, the ATE each arm to the pooled sample.
Base weights are the entropy reference ``q``; the targets stay unweighted.
``lm_weightit(y ~ d, vcov = "asympt")`` reports the M-estimation sandwich of
the stacked balancing and outcome equations, which ``vce='mestimation'``
computes.

Fixture: ``_fixtures/ebalance_estimand_R.json`` from
``_fixtures/_generate_ebalance_estimand_R.R`` (the three data sets of
``ebalance_weightit_R.json``; the plain ATT lives there). Tolerances: both
sides solve the dual to ~1e-10, so the estimate is held to 1e-9 relative
and the sandwich, which inherits that, to 1e-7 -- the budgets of the
existing ATT parity test.
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
REF = json.loads((_FIX / "ebalance_estimand_R.json").read_text(encoding="utf-8"))
CASES = sorted(k for k in REF if not k.startswith("_"))


def _data(name):
    df = pd.read_csv(_FIX / name)
    df["bw"] = 0.5 + (np.arange(1, len(df) + 1) % 4) / 4
    return df


@pytest.mark.parametrize("key", CASES)
def test_estimand_and_base_weights_match_weightit(key):
    c = REF[key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.ebalance(
            _data(c["data"]),
            y="y",
            treat="d",
            covariates=c["covariates"],
            estimand=c["estimand"],
            base_weights="bw" if c["base_weights"] else None,
        )
    assert r.estimand == c["estimand"]
    assert abs(r.estimate - c["est"]) / abs(c["est"]) < 1e-9
    assert abs(r.se - c["se"]) / c["se"] < 1e-7


def test_balance_identities_for_the_ate():
    """Identity: under the ATE both reweighted arms reproduce the pooled
    covariate means exactly."""
    df = _data("ebalance_sim_het.csv")
    r = sp.ebalance(df, y="y", treat="d", covariates=["X1", "X2"], estimand="ATE")
    w = r.model_info["weights_full"]
    d = df["d"].to_numpy() == 1
    for col in ("X1", "X2"):
        x = df[col].to_numpy()
        pooled = x.mean()
        assert abs(np.average(x[d], weights=w[d]) - pooled) < 1e-8 * (1 + abs(pooled))
        assert abs(np.average(x[~d], weights=w[~d]) - pooled) < 1e-8 * (1 + abs(pooled))


def test_atc_is_the_relabelled_att():
    """Identity: the ATC is minus the ATT with the arms swapped."""
    df = _data("ebalance_sim_cia.csv")
    atc = sp.ebalance(df, y="y", treat="d", covariates=["X1", "X2"], estimand="ATC")
    att_swapped = sp.ebalance(
        df.assign(d=1 - df["d"]), y="y", treat="d", covariates=["X1", "X2"]
    )
    assert abs(atc.estimate + att_swapped.estimate) < 1e-10
    assert abs(atc.se - att_swapped.se) < 1e-10 * atc.se


def test_bad_arguments_are_refused():
    df = _data("ebalance_sim_cia.csv")
    with pytest.raises(sp.MethodIncompatibility, match="estimand"):
        sp.ebalance(df, y="y", treat="d", covariates=["X1"], estimand="LATE")
    with pytest.raises(sp.MethodIncompatibility, match="ATT only"):
        sp.ebalance(
            df, y="y", treat="d", covariates=["X1"], estimand="ATE", dof_adjust=True
        )
    with pytest.raises(sp.MethodIncompatibility, match="positive"):
        sp.ebalance(
            df.assign(bw=0.0), y="y", treat="d", covariates=["X1"], base_weights="bw"
        )
    with pytest.raises(sp.MethodIncompatibility, match="coded 0/1"):
        sp.ebalance(df.assign(d=2 * df["d"]), y="y", treat="d", covariates=["X1"])
