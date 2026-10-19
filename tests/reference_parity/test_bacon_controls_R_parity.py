"""``sp.bacon_decomposition(covariates=)`` against R ``bacondecomp`` 0.1.1.

Reference: ``_fixtures/bacon_controls_R.json`` from
``_fixtures/_generate_bacon_controls_R.R`` on the panel written by
``_fixtures/_generate_bacon_controls_data.py`` (timing groups 1 = treated
throughout, 4, 7, 9 and never treated; one and two covariates).

T2: Omega, the within estimate, every pair's weight, estimate and type
agree to 1e-10 (the always-treated vs never pair has no 2x2 variation and
reduces to its covariate-mean term on both sides). Identity: the TWFE
coefficient with controls equals Omega * beta_within + (1 - Omega) *
sum(weight * estimate).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient

FX = Path(__file__).parent / "_fixtures"


@pytest.fixture(scope="module")
def ref():
    return json.loads((FX / "bacon_controls_R.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def panel():
    return pd.read_csv(FX / "bacon_controls.csv")


@pytest.mark.parametrize("key,covs", [("x1", ["x1"]), ("x1x2", ["x1", "x2"])])
def test_matches_bacondecomp(key, covs, ref, panel):
    r = sp.bacon_decomposition(
        panel, y="y", treat="d", time="t", id="id", covariates=covs
    )
    R = ref[key]
    assert abs(r["omega"] - R["Omega"]) < 1e-12
    assert abs(r["beta_within"] - R["beta_hat_w"]) < 1e-10 * max(
        1, abs(R["beta_hat_w"])
    )
    assert abs(r["beta_twfe"] - R["twfe"]) < 1e-12
    expected = pd.DataFrame(R["two_by_twos"])
    ours = r["decomposition"].copy()
    ours["untreated"] = ours["control"].map(
        lambda v: 99999.0 if v == "Never" else float(v)
    )
    merged = expected.merge(ours, on=["treated", "untreated"], suffixes=("_r", "_p"))
    assert len(merged) == len(expected) == len(ours)
    assert (merged["type_r"] == merged["type_p"]).all()
    np.testing.assert_allclose(
        merged["weight_p"], merged["weight_r"], rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        merged["estimate_p"], merged["estimate_r"], rtol=1e-10, atol=1e-10
    )


@pytest.mark.parametrize("covs", [["x1"], ["x1", "x2"]])
def test_decomposition_identity(covs, panel):
    r = sp.bacon_decomposition(
        panel, y="y", treat="d", time="t", id="id", covariates=covs
    )
    assert abs(r["weighted_sum"] - r["beta_twfe"]) < 1e-12
    assert abs(r["decomposition"]["weight"].sum() - 1) < 1e-12
    assert 0 <= r["omega"] <= 1


def test_without_covariates_is_unchanged(panel):
    a = sp.bacon_decomposition(panel, y="y", treat="d", time="t", id="id")
    b = sp.bacon_decomposition(
        panel, y="y", treat="d", time="t", id="id", covariates=[]
    )
    assert a["beta_twfe"] == b["beta_twfe"]
    assert "omega" not in a


def test_missing_covariate_refused(panel):
    bad = panel.copy()
    bad.loc[3, "x1"] = np.nan
    with pytest.raises(DataInsufficient):
        sp.bacon_decomposition(
            bad, y="y", treat="d", time="t", id="id", covariates=["x1"]
        )
