"""R parity for ``sp.overlap_weights`` against ``WeightIt::weightit``.

Overlap weights (Li, Morgan & Zaslavsky 2018) give every unit the *opposite*
arm's propensity — ``1 - e`` for treated, ``e`` for control — so the estimand
is the ATO and units in the region of good overlap dominate.
``WeightIt::weightit(..., method = "glm", estimand = "ATO")`` fits the same
logistic propensity model and applies the same transformation, so the two
should agree to the precision of the two logit solvers.

The whole generalized family from Li, Li & Li (2019, Table 1) is pinned, not
just ATO. They share one propensity fit, so an ATO-only test could pass on
the weighting formula while the propensity model was wrong.

What this fixture caught
------------------------
``sp.overlap_weights`` fitted ``sklearn.LogisticRegression(C=1e6)`` — a
*penalised* likelihood, however large ``C`` is — while ``sp.match`` and
``sp.psmatch2`` had always used the unpenalised Newton-Raphson MLE.  The same
data therefore got two different propensity scores depending on which
StatsPAI function you called, and the overlap-weight theory being implemented
is derived at the score equations of the *unpenalised* logit.  Measured here:
the penalised fit sat 8.5e-06 from R's ``glm`` and pushed the four estimands
~1e-6 off ``WeightIt``; the MLE agrees to 2.6e-14 and ~4e-14.

Fixture provenance
------------------
``_fixtures/_generate_overlap_weights_R.R`` under R 4.5.2 + WeightIt 1.7.0.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIXTURE_DIR = pathlib.Path(__file__).parent / "_fixtures"
_REGEN = (
    "Regenerate with Rscript tests/reference_parity/_fixtures/"
    "_generate_overlap_weights_R.R (R >= 4.5 + WeightIt >= 1.7)."
)

#: Both sides solve the same unpenalised logit, so the gap is solver noise.
_RTOL = 1e-11

ESTIMANDS = ["ATO", "ATE", "ATT", "ATC"]


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    path = _FIXTURE_DIR / "overlap_weights_data.csv"
    if not path.exists():  # pragma: no cover - fixture ships with the repo
        pytest.skip(f"missing {path.name}. {_REGEN}")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def r_ref() -> dict:
    path = _FIXTURE_DIR / "overlap_weights_R.json"
    if not path.exists():  # pragma: no cover
        pytest.skip(f"missing {path.name}. {_REGEN}")
    return json.loads(path.read_text(encoding="utf-8"))


def _fit(data, estimand):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.overlap_weights(
            data,
            y="y",
            treat="d",
            covariates=["x1", "x2"],
            estimand=estimand,
            n_bootstrap=0,
            seed=1,
        )


@pytest.mark.parametrize("estimand", ESTIMANDS)
def test_effect_matches_weightit(data, r_ref, estimand):
    ours = _fit(data, estimand).estimate
    assert ours == pytest.approx(r_ref[estimand]["effect"], rel=_RTOL)


class TestPropensityModel:
    """The gap the fixture exists to prevent reopening."""

    def test_propensity_matches_r_glm(self, data, r_ref):
        from statspai.matching.overlap_weights import _logit_pscore

        ps = _logit_pscore(
            data[["x1", "x2"]].to_numpy(dtype=float),
            data["d"].to_numpy(dtype=int),
        )
        np.testing.assert_allclose(
            ps, np.asarray(r_ref["pscore"], dtype=float), rtol=0, atol=1e-11
        )

    def test_propensity_is_the_same_one_sp_match_uses(self, data):
        """One package, one propensity score for the same specification."""
        from statspai.matching.match import MatchEstimator
        from statspai.matching.overlap_weights import _logit_pscore

        X = data[["x1", "x2"]].to_numpy(dtype=float)
        T = data["d"].to_numpy(dtype=int)
        np.testing.assert_allclose(
            _logit_pscore(X, T),
            MatchEstimator._logit_propensity(X, T, poly=1),
            rtol=0,
            atol=1e-12,
        )

    def test_penalised_fit_would_be_visibly_worse(self, data, r_ref):
        """Pin the size of the defect, so a revert cannot pass silently."""
        pytest.importorskip("sklearn")
        from sklearn.linear_model import LogisticRegression

        X = data[["x1", "x2"]].to_numpy(dtype=float)
        T = data["d"].to_numpy(dtype=int)
        m = LogisticRegression(max_iter=1000, solver="lbfgs", C=1e6).fit(X, T)
        penalised = m.predict_proba(X)[:, 1]
        gap = float(np.max(np.abs(penalised - np.asarray(r_ref["pscore"]))))
        # ~8.5e-06 when this was written: small, but 8 orders of magnitude
        # worse than the MLE and enough to move the estimates at 1e-6.
        assert gap > 1e-7


class TestWeightConstruction:
    def test_ato_weights_are_the_opposite_arm_propensity(self, data, r_ref):
        """w = 1-e for treated, e for control — the defining property."""
        ps = np.asarray(r_ref["pscore"], dtype=float)
        t = data["d"].to_numpy(dtype=int)
        w = np.asarray(r_ref["ATO"]["weights"], dtype=float)
        expected = np.where(t == 1, 1.0 - ps, ps)
        # WeightIt reports unnormalised overlap weights.
        np.testing.assert_allclose(w, expected, rtol=1e-10)

    def test_estimands_are_genuinely_different(self, data):
        est = {e: _fit(data, e).estimate for e in ESTIMANDS}
        assert len(set(np.round(list(est.values()), 9))) == len(ESTIMANDS)

    def test_ato_sits_between_att_and_atc(self, data):
        """Not a law, but true on this DGP; a sign flip would break it."""
        e = {k: _fit(data, k).estimate for k in ESTIMANDS}
        assert min(e["ATT"], e["ATC"]) <= e["ATO"] <= max(e["ATT"], e["ATC"])
