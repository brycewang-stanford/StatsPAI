"""Behavioural (simulation) checks for LLR matching and the bootstrap SE.

Stata parity pins these estimators to *another implementation*; it cannot
say whether either implementation recovers a known truth.  These tests do
that, on designs where the answer is known analytically.

They are deliberately small and seeded.  The tolerances are loose enough to
be stable but tight enough that a sign error, a mis-scaled kernel, or a
bootstrap that forgets to re-estimate the propensity score would fail.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _dgp(seed, n=600, tau=2.0, nonlinear=False):
    """Selection on observables with a known constant treatment effect."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    d = rng.binomial(1, 1 / (1 + np.exp(-(0.9 * x1 - 0.5 * x2))))
    base = 0.7 * x1 - 0.3 * x2
    if nonlinear:
        base = base + 0.8 * x1**2
    y = 1.0 + tau * d + base + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "d": d, "y": y})


def _fit(df, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.match(df, y="y", treat="d", covariates=["x1", "x2"], **kwargs)


class TestLLRRecoversTheTruth:
    def test_llr_is_close_to_the_true_att(self):
        est = [
            _fit(
                _dgp(s),
                method="llr",
                kernel="tricube",
                bwidth=0.2,
                se_method="ai",
            ).estimate
            for s in range(12)
        ]
        assert float(np.mean(est)) == pytest.approx(2.0, abs=0.10)

    def test_llr_beats_kernel_matching_on_a_sloped_design(self):
        """The local-linear correction exists to remove first-order bias.

        With a propensity-correlated outcome trend, plain kernel matching
        carries a boundary/slope bias that LLR is designed to remove, so LLR
        should sit closer to the truth on average.
        """
        bias_llr, bias_kern = [], []
        for s in range(12):
            df = _dgp(s, nonlinear=True)
            bias_llr.append(
                _fit(
                    df, method="llr", kernel="tricube", bwidth=0.2, se_method="ai"
                ).estimate
                - 2.0
            )
            bias_kern.append(
                _fit(df, method="kernel", kernel="tricube", bwidth=0.2).estimate - 2.0
            )
        assert abs(float(np.mean(bias_llr))) <= abs(float(np.mean(bias_kern))) + 1e-9

    def test_llr_weights_sum_to_one_per_treated_unit(self):
        r = _fit(_dgp(1), method="llr", kernel="tricube", bwidth=0.3, se_method="ai")
        md = r.matched_data
        w = md["_weight"].to_numpy(dtype=float)
        t = md["_treated"].to_numpy(dtype=float)
        n_treated = int(np.sum(np.isfinite(w) & (t == 1)))
        assert float(np.nansum(w[t == 0])) == pytest.approx(n_treated, rel=1e-8)

    def test_att_identity_holds_for_llr(self):
        r = _fit(_dgp(2), method="llr", kernel="tricube", bwidth=0.3, se_method="ai")
        md = r.matched_data
        t = (md["_treated"] == 1) & md["_y"].notna()
        assert float((md.loc[t, "y"] - md.loc[t, "_y"]).mean()) == pytest.approx(
            r.estimate, abs=1e-12
        )


class TestBootstrapBehaviour:
    def test_bootstrap_se_is_in_the_right_ballpark(self):
        """Compare the bootstrap SE to the actual sampling SD across draws."""
        est = [
            _fit(_dgp(s), method="kernel", kernel="epan", bwidth=0.1).estimate
            for s in range(25)
        ]
        sampling_sd = float(np.std(est, ddof=1))

        boot = _fit(
            _dgp(0),
            method="kernel",
            kernel="epan",
            bwidth=0.1,
            se_method="bootstrap",
            bootstrap_reps=150,
            bootstrap_seed=7,
        ).se
        # An order-of-magnitude check: a bootstrap that resampled wrongly
        # (or forgot to re-fit the propensity score) lands far outside this.
        assert 0.4 * sampling_sd < boot < 2.5 * sampling_sd

    def test_bootstrap_se_shrinks_with_sample_size(self):
        kw = dict(
            method="kernel",
            kernel="epan",
            bwidth=0.1,
            se_method="bootstrap",
            bootstrap_reps=100,
            bootstrap_seed=3,
        )
        small = _fit(_dgp(5, n=300), **kw).se
        large = _fit(_dgp(5, n=1500), **kw).se
        assert large < small

    def test_bootstrap_reestimates_the_propensity_score(self):
        """A bootstrap holding the score fixed would be systematically tighter.

        Not a proof, but it fails loudly if the replication path is ever
        refactored into reusing the parent fit.
        """
        r = _fit(
            _dgp(4),
            method="kernel",
            kernel="epan",
            bwidth=0.1,
            se_method="bootstrap",
            bootstrap_reps=120,
            bootstrap_seed=11,
        )
        assert r.model_info["bootstrap_reps_successful"] >= 100
        # The analytic psmatch2 SE conditions on the fitted score; the
        # bootstrap does not, so they must not coincide.
        analytic = _fit(_dgp(4), method="kernel", kernel="epan", bwidth=0.1).se
        assert r.se != pytest.approx(analytic, rel=1e-6)

    def test_failed_replications_are_reported_not_hidden(self):
        r = _fit(
            _dgp(6),
            method="radius",
            caliper=0.05,
            se_method="bootstrap",
            bootstrap_reps=40,
            bootstrap_seed=2,
        )
        info = r.model_info
        assert info["bootstrap_reps_successful"] + info["bootstrap_reps_failed"] == 40
        assert info["bootstrap_reps"] == 40

    def test_bootstrap_bias_is_recorded(self):
        r = _fit(
            _dgp(8),
            method="kernel",
            kernel="epan",
            bwidth=0.1,
            se_method="bootstrap",
            bootstrap_reps=60,
            bootstrap_seed=4,
        )
        assert np.isfinite(r.model_info["bootstrap_bias"])
        # Bias should be small relative to the effect for a sane estimator.
        assert abs(r.model_info["bootstrap_bias"]) < 0.5
