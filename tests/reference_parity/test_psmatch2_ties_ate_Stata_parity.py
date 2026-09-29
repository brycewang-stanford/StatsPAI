"""``sp.psmatch2(ties=True, ate=True)`` against Stata ``psmatch2``.

A PSM-DID replication (``psmatch2 ..., neighbor(1) ties ate common`` and a
regression on ``_weight != .``) could not be reproduced: StatsPAI had no
``ties`` (every equidistant control is a match) and no ``ate`` (controls
matched too, two-sided support, treated rows weighted only by their use as
a match), so its matched sample had 13,000 rows where Stata's had 28,628.

Reference: Stata 18 ``psmatch2`` 4.0.12 with the propensity score supplied
(``pscore(ps)``, coarse two-decimal scores so ties are everywhere) on
``_fixtures/psmatch2_ties.csv``, from ``_generate_psmatch2_ties_Stata.do``.
Supplying the score isolates the matching rule: ``_weight`` and
``_support`` agree row for row, ATT / ATU / ATE to 1e-12, and the ATT SE
to 1e-7 (psmatch2 stores the squared weights it sums as ``float``).
On the replication's own data (99,878 firm-years) the same check -- Stata
fed StatsPAI's score -- gives identical matched samples (28,602 rows); the
last difference from the paper's 28,628 comes from the logit, whose
scores differ from Stata's by 1e-7 (its convergence tolerance), which is
enough to change a few hundred nearest neighbours among 10^5 scores.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.matching._psmatch2_nn import psmatch2_nn

_FIX = pathlib.Path(__file__).parent / "_fixtures"
REF = json.loads((_FIX / "psmatch2_ties_Stata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "psmatch2_ties.csv")


@pytest.fixture(scope="module")
def ref_rows():
    return pd.read_csv(_FIX / "psmatch2_ties_Stata.csv").sort_values("row")


@pytest.mark.parametrize("ate", [False, True])
def test_matching_rule_matches_psmatch2(data, ref_rows, ate):
    tag = "ate" if ate else "ties"
    out = psmatch2_nn(
        data["ps"].to_numpy(),
        data["treat"].to_numpy(),
        data["y"].to_numpy(),
        ties=True,
        ate=ate,
        common_support=True,
    )
    np.testing.assert_array_equal(
        out["support"].astype(int), ref_rows[f"s_{tag}"].to_numpy()
    )
    np.testing.assert_allclose(
        out["weight"], ref_rows[f"w_{tag}"].to_numpy(), rtol=1e-12, equal_nan=True
    )
    ref = REF["ties_ate" if ate else "ties"]
    np.testing.assert_allclose(out["att"], ref["att"], rtol=1e-12)
    # psmatch2 squares _weight into a `gen` (float) variable before summing.
    np.testing.assert_allclose(out["se_att"], ref["seatt"], rtol=1e-7)
    if ate:
        np.testing.assert_allclose(out["atu"], ref["atu"], rtol=1e-12)
        np.testing.assert_allclose(out["ate"], ref["ate"], rtol=1e-12)


def test_public_entry_point_uses_the_same_rule():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"x": rng.integers(0, 4, n).astype(float)})
    df["d"] = (rng.random(n) < 0.2 + 0.1 * df["x"]).astype(int)
    df["y"] = df["x"] + df["d"] + rng.normal(size=n)
    m = sp.psmatch2(
        df,
        treat="d",
        covariates=["x"],
        outcome="y",
        common_support="minmax",
        ties=True,
        ate=True,
    )
    md = m.matched_data
    ref = psmatch2_nn(
        md["_pscore"].to_numpy(),
        md["_treated"].to_numpy(),
        df["y"].to_numpy(),
        ties=True,
        ate=True,
        common_support=True,
    )
    np.testing.assert_allclose(md["_weight"], ref["weight"], equal_nan=True)
    assert m.att == pytest.approx(ref["att"], rel=1e-12)
    assert m.result.model_info["ate"] == pytest.approx(ref["ate"], rel=1e-12)
    # Four covariate values -> four scores: ties put every control in play.
    assert (
        md.loc[md["_treated"] == 0, "_weight"].notna().sum()
        > 3 * (df["d"] == 1).sum() / 4
    )


def test_rejects_unsupported_combinations(data):
    with pytest.raises(Exception, match="neighbor=1"):
        sp.psmatch2(
            data.assign(x=data["ps"]),
            treat="treat",
            covariates=["x"],
            outcome="y",
            neighbor=2,
            ties=True,
        )
