"""Target-trial and longitudinal estimators against R, on the same data.

Reference numbers are in ``_fixtures/tte_longitudinal_R.json``, written by
``_fixtures/_generate_tte_longitudinal_R.R`` from the CSVs of
``_fixtures/_generate_tte_longitudinal_data.py``. ipw, ltmle, sandwich and
SuperLearner are GPL and were run as black boxes.

* ``sp.target_trial_emulate``: the weighted mean difference among the
  eligible equals the treatment coefficient of ``lm(Y ~ A, weights = w)``
  and its SE the HC0 sandwich (``sandwich::vcovHC``), unweighted and with
  ``ipw::ipwpoint`` weights (unstabilised and stabilised).
* ``sp.longitudinal_analyze`` / ``sp.longitudinal_contrast``, IPW path:
  ``ipw::ipwpoint`` weights from a logit in the baseline confounder, then
  weighted means (``lm(Y ~ 1)``) and the contrast (``lm(Y ~ A)``) with HC0
  SEs. G-formula path: ``ltmle::ltmle(gcomp = TRUE)`` with the linear
  learner ``SL.lm`` is the iterated conditional expectation estimator with
  linear regressions on the full history; only the point estimate is
  compared, because the StatsPAI path reports a bootstrap SE and ltmle's
  g-computation SE is, in its own words, not accurate.
* ``sp.clone_censor_weight``: per strategy, the at-risk rows (up to and
  including the first deviation) give ``ipw::ipwtm(type = "cens")`` IPC
  weights; the uncensored rows and their stabilised and unstabilised
  weights must agree.

Tolerances: 1e-9 relative. Both sides fit the logits to convergence
(``glm`` at ``epsilon = 1e-14``, Newton steps below 1e-8 here), and the
JSON stores 15 significant digits.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FX = Path(__file__).parent / "_fixtures"
REF = json.loads((FX / "tte_longitudinal_R.json").read_text(encoding="utf-8"))
RTOL = 1e-9


# ---------------------------------------------------------------------------
# sp.target_trial_emulate
# ---------------------------------------------------------------------------


class TestTargetTrialEmulate:
    ref = REF["tte"]
    data = pd.read_csv(FX / "tte_point.csv")
    proto = sp.target_trial_protocol(
        eligibility="age >= 45 and ldl > 110",
        treatment_strategies=["treat", "no treat"],
        assignment="observational emulation",
        time_zero="baseline",
        followup_end="end of study",
        outcome="Y",
    )

    def _full_weights(self, key):
        w = np.ones(len(self.data))
        pos = self.data.index[self.data["id"].isin(self.ref["eligible_id"])]
        w[pos] = self.ref[key]
        return w

    def test_eligible_set(self):
        res = sp.target_trial_emulate(
            self.proto, self.data, outcome_col="Y", treatment_col="A"
        )
        assert res.n_eligible == self.ref["n_eligible"]
        assert res.n_excluded_immortal == len(self.data) - self.ref["n_eligible"]

    @pytest.mark.parametrize(
        "case,key",
        [("unweighted", None), ("ipw", "w_ipw"), ("stabilized", "w_stab")],
    )
    def test_estimate_and_hc0_se(self, case, key):
        w = None if key is None else self._full_weights(key)
        res = sp.target_trial_emulate(
            self.proto, self.data, outcome_col="Y", treatment_col="A", weights=w
        )
        assert res.estimate == pytest.approx(self.ref[case]["est"], rel=RTOL)
        assert res.se == pytest.approx(self.ref[case]["se"], rel=RTOL)

    def test_se_is_invariant_to_rescaling_the_weights(self):
        w = self._full_weights("w_ipw")
        a = sp.target_trial_emulate(
            self.proto, self.data, outcome_col="Y", treatment_col="A", weights=w
        )
        b = sp.target_trial_emulate(
            self.proto, self.data, outcome_col="Y", treatment_col="A", weights=7 * w
        )
        assert b.se == pytest.approx(a.se, rel=1e-12)
        assert b.estimate == pytest.approx(a.estimate, rel=1e-12)


# ---------------------------------------------------------------------------
# sp.longitudinal_analyze / sp.longitudinal_contrast
# ---------------------------------------------------------------------------


class TestLongitudinalIPW:
    ref = REF["long_ipw"]
    data = pd.read_csv(FX / "long_ipw.csv")
    kw = dict(id="id", time="time", treatment="A", outcome="Y", baseline=["B"])

    def test_propensities_are_inside_the_trimming_bounds(self):
        # trim=0.01 clips propensities; R's are inside, so it is inactive
        lo, hi = self.ref["ps_range"]
        assert 0.01 < lo and hi < 0.99

    @pytest.mark.parametrize("regime", ["always_treat", "never_treat"])
    def test_regime_mean(self, regime):
        res = sp.longitudinal_analyze(self.data, regime=regime, **self.kw)
        assert res.method == "ipw"
        ref = self.ref["always" if regime == "always_treat" else "never"]
        assert res.estimate == pytest.approx(ref["est"], rel=RTOL)
        assert res.se == pytest.approx(ref["se"], rel=RTOL)

    def test_contrast(self):
        res = sp.longitudinal_contrast(
            self.data, regime_a="always_treat", regime_b="never_treat", **self.kw
        )
        # the two arms are disjoint, so the sum of the arm variances is the
        # HC0 variance of the lm(Y ~ A) coefficient
        assert res["contrast"] == pytest.approx(self.ref["contrast"]["est"], rel=RTOL)
        assert res["se"] == pytest.approx(self.ref["contrast"]["se"], rel=RTOL)


class TestLongitudinalGFormula:
    ref = REF["long_tv"]
    data = pd.read_csv(FX / "long_tv.csv")
    kw = dict(
        id="id",
        time="time",
        treatment="A",
        outcome="Y",
        time_varying=["L"],
        baseline=["B"],
    )

    @pytest.mark.parametrize("regime", ["always_treat", "never_treat"])
    def test_regime_mean_equals_ltmle_gcomp(self, regime):
        res = sp.longitudinal_analyze(self.data, regime=regime, **self.kw)
        assert res.method == "g-formula"
        ref = self.ref["always" if regime == "always_treat" else "never"]
        assert res.estimate == pytest.approx(ref, rel=RTOL)

    def test_contrast_equals_ltmle_gcomp_ate(self):
        res = sp.longitudinal_contrast(
            self.data, regime_a="always_treat", regime_b="never_treat", **self.kw
        )
        assert res["contrast"] == pytest.approx(self.ref["contrast"], rel=RTOL)


# ---------------------------------------------------------------------------
# sp.clone_censor_weight
# ---------------------------------------------------------------------------


class TestCloneCensorWeight:
    ref = REF["ccw"]
    data = pd.read_csv(FX / "ccw_panel.csv")
    strategies = {
        "always": lambda g: g["treat"].to_numpy() == 1,
        "never": lambda g: g["treat"].to_numpy() == 0,
    }

    def _fit(self, stabilize):
        return sp.clone_censor_weight(
            self.data,
            id_col="id",
            time_col="time",
            treatment_col="treat",
            strategies=self.strategies,
            censor_covariates=["L", "age"],
            stabilize=stabilize,
        )

    @pytest.mark.parametrize("stabilize,key", [(True, "w_stab"), (False, "w_unstab")])
    def test_rows_and_weights_equal_ipwtm(self, stabilize, key):
        res = self._fit(stabilize)
        out = res.cloned_data
        for strat in ("always", "never"):
            ref = self.ref[strat]
            got = out.loc[out["_strategy"] == strat].sort_values(["id", "time"])
            np.testing.assert_array_equal(got["id"].to_numpy(), ref["id"])
            np.testing.assert_array_equal(got["time"].to_numpy(), ref["time"])
            np.testing.assert_allclose(got["_ipcw"].to_numpy(), ref[key], rtol=RTOL)
        assert res.n_clones == len(self.ref["always"]["id"]) + len(
            self.ref["never"]["id"]
        )

    def test_censored_flag_marks_clones_that_deviate(self):
        out = self._fit(True).cloned_data
        for strat, target in (("always", 1), ("never", 0)):
            sub = self.data.sort_values(["id", "time"])
            ever_dev = sub.groupby("id")["treat"].apply(
                lambda s: bool((s != target).any())
            )
            got = out.loc[out["_strategy"] == strat].groupby("id")["_censored"].first()
            assert (got == ever_dev.loc[got.index].astype(int)).all()
