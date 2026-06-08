"""Tier D analytic special-case tests — partial-identification & overlap bounds.

Part of the P1 "Tier D analytic special-cases" campaign (see
``.tierd_campaign/CAMPAIGN.md``). These four estimators previously had **no
numerical-assertion test at all** (``scripts/tierd_classify.py`` graded them
``untested``). Each test below anchors the implementation to a *known truth* —
either a closed-form identity that the estimator's own math must satisfy, or a
constructed data-generating process whose identified set we can compute by hand
— rather than a smoke call.

Estimators covered (all reference-less, no R/Stata parity):
    sp.horowitz_manski   Horowitz-Manski (2000) conditional ATE bounds
    sp.iv_bounds         Nevo-Rosen (2012) LATE bounds under imperfect IV
    sp.oster_delta       Oster (2019) coefficient-stability identified set
    sp.trimming          Crump et al. (2009) / Stürmer optimal-overlap trimming

No estimator numerics are modified; these tests pin current correct behaviour
(CLAUDE.md §5 / §7, purely additive — see campaign red line).
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ---------------------------------------------------------------------------
# sp.trimming — Crump (2009) / Stürmer overlap trimming
# ---------------------------------------------------------------------------
class TestTrimmingAnalytic:
    """``trimming`` is a deterministic function of the propensity score when
    ``ps`` is supplied directly, so the kept set is a closed-form truth."""

    def test_sturmer_keeps_exactly_the_closed_interval(self):
        # Stürmer fixes the overlap region at [0.1, 0.9]; with ps supplied the
        # kept rows are exactly {ps : 0.1 <= ps <= 0.9}, an exact known set.
        ps_vals = np.array([0.02, 0.10, 0.30, 0.50, 0.70, 0.90, 0.98])
        df = pd.DataFrame({"t": [0, 1, 0, 1, 0, 1, 0], "ps": ps_vals, "row": range(7)})
        out = sp.trimming(
            df,
            treatment="t",
            covariates=["ps"],
            method="sturmer",
            ps=df["ps"],
        )
        kept = sorted(out["ps"].tolist())
        assert kept == [0.10, 0.30, 0.50, 0.70, 0.90]
        # Boundary inclusivity is part of the closed form (>= and <=).
        assert 0.10 in kept and 0.90 in kept
        assert 0.02 not in kept and 0.98 not in kept

    def test_crump_perfect_overlap_trims_nothing(self):
        # All ps == 0.5 is perfect overlap; Crump's optimal alpha <= 0.5 and
        # every unit lies strictly inside [alpha, 1 - alpha], so nothing is
        # dropped regardless of the exact threshold.
        n = 200
        df = pd.DataFrame({"t": ([0, 1] * (n // 2)), "ps": np.full(n, 0.5)})
        out = sp.trimming(
            df, treatment="t", covariates=["ps"], method="crump", ps=df["ps"]
        )
        assert len(out) == n

    def test_crump_drops_extreme_tails(self):
        # With heavy tails near 0 and 1, the Crump rule must trim them; every
        # surviving unit must lie inside the open (0, 1) interior.
        rng = np.random.default_rng(0)
        bulk = rng.uniform(0.3, 0.7, size=180)
        tails = np.concatenate([np.full(10, 0.001), np.full(10, 0.999)])
        ps = np.concatenate([bulk, tails])
        df = pd.DataFrame({"t": (np.arange(200) % 2), "ps": ps})
        out = sp.trimming(
            df, treatment="t", covariates=["ps"], method="crump", ps=df["ps"]
        )
        assert 0 < len(out) < 200
        assert out["ps"].min() > 0.001 and out["ps"].max() < 0.999

    def test_crump_better_overlap_trims_no_more(self):
        # Monotonicity: a tighter-overlap design cannot be trimmed harder than a
        # worse-overlap design.
        rng = np.random.default_rng(1)
        good = np.clip(rng.normal(0.5, 0.05, 300), 1e-4, 1 - 1e-4)
        bad = np.clip(rng.normal(0.5, 0.25, 300), 1e-4, 1 - 1e-4)
        df_g = pd.DataFrame({"t": np.arange(300) % 2, "ps": good})
        df_b = pd.DataFrame({"t": np.arange(300) % 2, "ps": bad})
        kept_g = len(
            sp.trimming(
                df_g, treatment="t", covariates=["ps"], method="crump", ps=df_g["ps"]
            )
        )
        kept_b = len(
            sp.trimming(
                df_b, treatment="t", covariates=["ps"], method="crump", ps=df_b["ps"]
            )
        )
        assert kept_g >= kept_b


# ---------------------------------------------------------------------------
# sp.horowitz_manski — conditional ATE bounds
# ---------------------------------------------------------------------------
class TestHorowitzManskiAnalytic:
    """The per-stratum Manski ATE bound width is exactly ``y_upper - y_lower``;
    summing the stratum weights (=1 when every stratum has both arms) gives a
    total width that does not depend on the data — a closed-form identity."""

    @staticmethod
    def _two_stratum_data(seed=7, n=400):
        rng = np.random.default_rng(seed)
        g = rng.integers(0, 2, size=n)  # binary covariate -> 2 strata
        d = rng.integers(0, 2, size=n)  # randomized treatment
        # Outcome strictly inside [0, 1]; both arms present in each stratum.
        y = np.clip(0.4 + 0.2 * d + 0.1 * g + rng.normal(0, 0.05, n), 0.01, 0.99)
        return pd.DataFrame({"y": y, "d": d, "g": g})

    @pytest.mark.parametrize("y_lo,y_hi", [(0.0, 1.0), (-2.0, 3.0)])
    def test_ate_bound_width_equals_outcome_range(self, y_lo, y_hi):
        df = self._two_stratum_data()
        res = sp.horowitz_manski(
            df,
            y="y",
            treatment="d",
            covariates=["g"],
            y_lower=y_lo,
            y_upper=y_hi,
            n_boot=10,
        )
        # width = sum_s w_s * (y_hi - y_lo) = (y_hi - y_lo), since every stratum
        # contains both treated and control units.
        assert res.upper - res.lower == pytest.approx(y_hi - y_lo, abs=1e-9)
        assert res.lower <= res.upper

    def test_single_stratum_matches_closed_form(self):
        # A constant covariate -> one stratum -> the conditional bound reduces
        # to the unconditional Manski ATE bound, computable in closed form.
        df = self._two_stratum_data()
        df["const"] = 1.0
        y_lo, y_hi = 0.0, 1.0
        res = sp.horowitz_manski(
            df,
            y="y",
            treatment="d",
            covariates=["const"],
            y_lower=y_lo,
            y_upper=y_hi,
            n_boot=10,
        )
        Y, D = df["y"].values, df["d"].values
        p1 = D.mean()
        p0 = 1 - p1
        e1, e0 = Y[D == 1].mean(), Y[D == 0].mean()
        lb = e1 * p1 + y_lo * p0 - e0 * p0 - y_hi * p1
        ub = e1 * p1 + y_hi * p0 - e0 * p0 - y_lo * p1
        assert res.lower == pytest.approx(lb, abs=1e-9)
        assert res.upper == pytest.approx(ub, abs=1e-9)

    def test_bounds_contain_true_ate(self):
        # Worst-case bounds must always bracket the true ATE of the DGP.
        rng = np.random.default_rng(3)
        n = 800
        d = rng.integers(0, 2, n)
        g = rng.integers(0, 2, n)
        true_ate = 0.2
        y = np.clip(0.4 + true_ate * d + 0.1 * g + rng.normal(0, 0.05, n), 0.0, 1.0)
        df = pd.DataFrame({"y": y, "d": d, "g": g})
        res = sp.horowitz_manski(
            df,
            y="y",
            treatment="d",
            covariates=["g"],
            y_lower=0.0,
            y_upper=1.0,
            n_boot=10,
        )
        assert res.lower <= true_ate <= res.upper


# ---------------------------------------------------------------------------
# sp.oster_delta — coefficient-stability identified set (Oster 2019)
# ---------------------------------------------------------------------------
class TestOsterDeltaAnalytic:

    def test_stable_coefficient_gives_degenerate_set(self):
        # A control orthogonal to the treatment leaves the treatment
        # coefficient unchanged (beta_short == beta_full), so Oster's bias
        # term (proportional to beta_short - beta_full) vanishes and the
        # identified set collapses to the point estimate.
        rng = np.random.default_rng(11)
        n = 2000
        t = rng.normal(0, 1, n)
        c = rng.normal(0, 1, n)  # orthogonal to t in expectation
        beta = 1.5
        y = beta * t + 2.0 * c + rng.normal(0, 0.3, n)
        df = pd.DataFrame({"y": y, "t": t, "c": c})
        res = sp.oster_delta(df, y="y", x_base=["t"], x_controls=["c"], n_boot=10)
        assert res.lower == pytest.approx(res.upper, abs=2e-2)
        assert res.lower == pytest.approx(beta, abs=5e-2)

    def test_matches_oster_eq3(self):
        # Re-derive Oster (2019) eq. 3 from the same OLS fits the estimator
        # uses and assert the identified set endpoints match exactly. This
        # pins the implementation to its documented analytic formula.
        rng = np.random.default_rng(12)
        n = 1500
        c = rng.normal(0, 1, n)  # confounder
        t = 0.6 * c + rng.normal(0, 1, n)  # selection on observable
        y = 1.0 * t + 1.5 * c + rng.normal(0, 0.5, n)
        df = pd.DataFrame({"y": y, "t": t, "c": c})
        r_max = 1.3

        Y = df["y"].values
        n_ = len(df)
        Xs = np.column_stack([np.ones(n_), df["t"].values])
        bs = np.linalg.lstsq(Xs, Y, rcond=None)[0]
        beta_short = bs[1]
        r2_short = 1 - np.var(Y - Xs @ bs) / np.var(Y)
        Xf = np.column_stack([np.ones(n_), df["t"].values, df["c"].values])
        bf = np.linalg.lstsq(Xf, Y, rcond=None)[0]
        beta_full = bf[1]
        r2_full = 1 - np.var(Y - Xf @ bf) / np.var(Y)

        bias = (beta_short - beta_full) * (r_max - r2_full) / (r2_full - r2_short)
        beta_star = beta_full - bias
        exp_lo, exp_hi = min(beta_full, beta_star), max(beta_full, beta_star)

        res = sp.oster_delta(
            df, y="y", x_base=["t"], x_controls=["c"], r_max=r_max, n_boot=10
        )
        assert res.lower == pytest.approx(exp_lo, abs=1e-9)
        assert res.upper == pytest.approx(exp_hi, abs=1e-9)


# ---------------------------------------------------------------------------
# sp.iv_bounds — Nevo-Rosen (2012) LATE bounds under imperfect IV
# ---------------------------------------------------------------------------
class TestIVBoundsAnalytic:

    @staticmethod
    def _valid_iv_data(seed=21, n=4000):
        rng = np.random.default_rng(seed)
        z = rng.integers(0, 2, n).astype(float)  # binary instrument
        u = rng.normal(0, 1, n)  # confounder
        # Treatment driven by z and u (endogeneity), binary.
        d = ((0.3 + 0.4 * z + 0.5 * u + rng.normal(0, 0.2, n)) > 0.5).astype(float)
        tau = 1.0  # structural effect
        y = tau * d + 0.8 * u + rng.normal(0, 0.3, n)  # exclusion holds (no z)
        return pd.DataFrame({"y": y, "d": d, "z": z})

    def test_monotone_set_equals_ols_wald_interval(self):
        # Under 'monotone_iv' the identified set is exactly [min(OLS, Wald),
        # max(OLS, Wald)] — recompute both moment estimands independently.
        df = self._valid_iv_data()
        Y, D, Z = df["y"].values, df["d"].values, df["z"].values
        wald = np.cov(Z, Y, ddof=1)[0, 1] / np.cov(Z, D, ddof=1)[0, 1]
        ols = np.cov(D, Y, ddof=1)[0, 1] / np.var(D, ddof=1)
        res = sp.iv_bounds(
            df,
            y="y",
            treatment="d",
            instrument="z",
            assumption="monotone_iv",
            n_boot=10,
        )
        assert res.lower == pytest.approx(min(ols, wald), abs=1e-9)
        assert res.upper == pytest.approx(max(ols, wald), abs=1e-9)

    def test_valid_iv_recovers_effect_ols_biased_up(self):
        # When the instrument is valid, the Wald estimand is consistent for the
        # structural effect tau=1, while OLS is biased *upward* by the positive
        # confounder. The monotone-IV set therefore runs from the consistent
        # Wald endpoint up to the biased OLS endpoint: [Wald, OLS] with Wald<OLS.
        df = self._valid_iv_data()
        Y, D, Z = df["y"].values, df["d"].values, df["z"].values
        wald = np.cov(Z, Y, ddof=1)[0, 1] / np.cov(Z, D, ddof=1)[0, 1]
        ols = np.cov(D, Y, ddof=1)[0, 1] / np.var(D, ddof=1)
        res = sp.iv_bounds(
            df,
            y="y",
            treatment="d",
            instrument="z",
            assumption="monotone_iv",
            n_boot=10,
        )
        assert wald == pytest.approx(1.0, abs=0.12)  # IV consistency
        assert ols - wald > 0.5  # known-sign confounding bias
        assert res.lower == pytest.approx(wald, abs=1e-9)
        assert res.upper == pytest.approx(ols, abs=1e-9)

    def test_less_than_late_symmetric_interval(self):
        # The 'less_than_late' set is [Wald - |OLS - Wald|, Wald + |OLS - Wald|]
        # i.e. symmetric about the Wald point — recompute and check.
        df = self._valid_iv_data()
        Y, D, Z = df["y"].values, df["d"].values, df["z"].values
        wald = np.cov(Z, Y, ddof=1)[0, 1] / np.cov(Z, D, ddof=1)[0, 1]
        ols = np.cov(D, Y, ddof=1)[0, 1] / np.var(D, ddof=1)
        diff = abs(ols - wald)
        res = sp.iv_bounds(
            df,
            y="y",
            treatment="d",
            instrument="z",
            assumption="less_than_late",
            n_boot=10,
        )
        assert res.lower == pytest.approx(wald - diff, abs=1e-9)
        assert res.upper == pytest.approx(wald + diff, abs=1e-9)
        # Wald point is the midpoint of this set.
        assert 0.5 * (res.lower + res.upper) == pytest.approx(wald, abs=1e-9)
