"""Regression tests for two inference-correctness fixes.

These lock in fixes that change *inference* (not point estimates) in two
estimators where the previous code reported the wrong size / coverage:

1. ``sp.cusum_test`` (Brown-Durbin-Evans recursive CUSUM) compared the CUSUM
   path against a *constant* 1.358 boundary -- the sup|Brownian-bridge|
   critical value that belongs to the OLS-CUSUM, not the recursive CUSUM. On
   a stable (H0) relation it rejected ~30% of the time at a nominal 5% level.
   The boundary must be the *linear* BDE boundary ``a * [1 + 2 s / (n - k)]``.

2. ``sp.lee_bounds`` labelled its confidence interval "Imbens-Manski" but
   applied the two-sided ``z_{1-alpha/2}`` to *both* endpoints, which is the
   Horowitz-Manski interval for the identified *set* and over-covers the
   parameter. The genuine Imbens & Manski (2004) interval uses a critical
   value ``C_n`` that interpolates between the one- and two-sided z.

References for the fixed methods are hand-verified (see the respective module
docstrings); Imbens & Manski (2004) Econometrica 72(6):1845-1857 was verified
via Crossref and RePEc/IDEAS.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.bounds.lee_manski import _imbens_manski_cn
from statspai.timeseries.structural_break import cusum_test


# ----------------------------------------------------------------------
# Fix 1 -- recursive CUSUM uses the Brown-Durbin-Evans linear boundary
# ----------------------------------------------------------------------
class TestCusumLinearBoundary:
    def test_critical_value_is_linear_boundary(self):
        rng = np.random.default_rng(0)
        n = 150
        x = rng.normal(size=n)
        y = 1.0 + 0.5 * x + rng.normal(scale=0.7, size=n)
        df = pd.DataFrame({"y": y, "x": x})
        res = cusum_test(df, y="y", x=["x"], alpha=0.05)

        boundary = np.asarray(res["critical_value"], dtype=float)
        m = n - 2  # n - k, k = 2 (intercept + x)
        assert boundary.shape == (m,)
        # Boundary widens linearly from a (=0.948 at 5%) to 3a across the sample.
        a = 0.948
        s = np.arange(1, m + 1)
        np.testing.assert_allclose(boundary, a * (1.0 + 2.0 * s / m), rtol=1e-12)
        np.testing.assert_allclose(boundary[0], a * (1 + 2 / m), rtol=1e-12)
        np.testing.assert_allclose(boundary[-1], 3 * a, rtol=1e-9)

    def test_size_near_nominal_under_h0(self):
        """A nominal 5% test must not reject ~30% of the time under H0."""
        rng = np.random.default_rng(12345)
        n, B = 120, 400
        rej_new = 0
        rej_old_const = 0
        for _ in range(B):
            x = rng.normal(size=n)
            y = 1.0 + 0.5 * x + rng.normal(scale=0.7, size=n)  # stable -> H0
            df = pd.DataFrame({"y": y, "x": x})
            res = cusum_test(df, y="y", x=["x"], alpha=0.05)
            rej_new += int(res["reject"])
            # Replicate the OLD constant-1.358 rule on the same path.
            rej_old_const += int(res["max_cusum"] > 1.358)

        size_new = rej_new / B
        size_old = rej_old_const / B
        # New test is close to nominal (and conservative); old rule grossly
        # over-rejects. Generous bounds keep this deterministic test stable.
        assert size_new < 0.12, f"new size {size_new:.3f} too high"
        assert size_old > 0.20, f"old const rule size {size_old:.3f} (bug guard)"

    def test_power_against_mean_shift(self):
        rng = np.random.default_rng(7)
        n = 120
        rej = 0
        for _ in range(200):
            x = rng.normal(size=n)
            shift = np.where(np.arange(n) < 60, 0.0, 2.5)
            y = 1.0 + 0.5 * x + shift + rng.normal(scale=0.7, size=n)
            df = pd.DataFrame({"y": y, "x": x})
            rej += int(cusum_test(df, y="y", x=["x"])["reject"])
        assert rej / 200 > 0.8

    def test_keys_and_doctest_contract_preserved(self):
        rng = np.random.default_rng(0)
        n = 120
        x = rng.normal(size=n)
        y = 1.0 + 0.5 * x + rng.normal(scale=0.5, size=n)
        df = pd.DataFrame({"y": y, "x": x})
        res = cusum_test(df, y="y", x=["x"])
        assert sorted(res.keys()) == [
            "critical_value",
            "cusum",
            "max_cusum",
            "n_obs",
            "reject",
        ]
        assert res["n_obs"] == 120
        assert isinstance(res["reject"], bool)


# ----------------------------------------------------------------------
# Fix 2 -- Lee bounds use the genuine Imbens-Manski C_n
# ----------------------------------------------------------------------
class TestImbensManskiCn:
    def test_point_identified_limit_is_two_sided_z(self):
        z_two = stats.norm.ppf(0.975)
        assert _imbens_manski_cn(0.0, 1.0, 0.05) == pytest.approx(z_two, abs=1e-8)

    def test_wide_bounds_limit_is_one_sided_z(self):
        z_one = stats.norm.ppf(0.95)
        assert _imbens_manski_cn(1e6, 1.0, 0.05) == pytest.approx(z_one, abs=1e-6)

    def test_monotone_and_bracketed(self):
        z_one = stats.norm.ppf(0.95)
        z_two = stats.norm.ppf(0.975)
        widths = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 50.0]
        cns = [_imbens_manski_cn(d, 1.0, 0.05) for d in widths]
        # within [z_one, z_two] and non-increasing in width
        for c in cns:
            assert z_one - 1e-9 <= c <= z_two + 1e-9
        assert all(cns[i] >= cns[i + 1] - 1e-9 for i in range(len(cns) - 1))
        # strictly interior for an intermediate width
        assert z_one < _imbens_manski_cn(0.5, 1.0, 0.05) < z_two

    def test_solves_defining_equation(self):
        delta = 1.3
        sigma = 0.8
        c = _imbens_manski_cn(delta, sigma, 0.05)
        ratio = delta / sigma
        lhs = stats.norm.cdf(c + ratio) - stats.norm.cdf(-c)
        assert lhs == pytest.approx(0.95, abs=1e-7)

    def test_degenerate_sigma(self):
        z_one = stats.norm.ppf(0.95)
        z_two = stats.norm.ppf(0.975)
        assert _imbens_manski_cn(0.0, 0.0, 0.05) == pytest.approx(z_two)
        assert _imbens_manski_cn(2.0, 0.0, 0.05) == pytest.approx(z_one)


class TestLeeBoundsCI:
    def _selection_dgp(self, seed=0, n=1500):
        rng = np.random.default_rng(seed)
        d = rng.integers(0, 2, n)
        # treatment raises retention (differential attrition)
        latent = 0.3 + 0.6 * d + rng.normal(size=n)
        s = (latent > 0).astype(int)
        y = 1.0 + 0.5 * d + rng.normal(size=n)
        return pd.DataFrame({"y": y, "d": d, "s": s})

    def test_im_ci_brackets_bounds_and_uses_cn_below_two_sided_z(self):
        df = self._selection_dgp()
        res = sp.lee_bounds(df, y="y", treat="d", selection="s", n_bootstrap=200)
        lb = res.model_info["lower_bound"]
        ub = res.model_info["upper_bound"]
        ci_lo, ci_hi = res.ci
        # The Imbens-Manski CI brackets the identified set and is ordered.
        assert ci_lo <= lb + 1e-9 <= ub + 1e-9 <= ci_hi + 1e-9
        # The per-endpoint multiplier C_n implied by these (positive-width)
        # bounds is strictly below the two-sided z that the old Horowitz-Manski
        # interval used on both ends -> the IM interval is the narrower one.
        z_two = stats.norm.ppf(0.975)
        c_n = _imbens_manski_cn(ub - lb, 1.0, 0.05)  # any sigma>0; width>0 => <z_two
        assert c_n < z_two


# ----------------------------------------------------------------------
# Matching default SE (v1.22: 'auto' moved from 'ai' to 'abadie_imbens')
#
# Before 1.22 the default deliberately reproduced 'ai' and merely warned,
# to keep published numbers stable. The coverage study in
# benchmarks/matching_se_coverage.py showed 'ai' never reaches nominal
# coverage in any of 36 designs (0.71-0.92 vs a nominal 0.95), so the
# default now resolves to the estimator that is correctly sized.
# ----------------------------------------------------------------------
class TestMatchingDefaultSE:
    def _psm_data(self, seed=0, n=400):
        rng = np.random.default_rng(seed)
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        d = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.5 * x1 + 0.3 * x2)))).astype(int)
        y = 1.0 + 2.0 * d + x1 + 0.5 * x2 + rng.normal(size=n)
        return pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})

    def _fit(self, se_method=None):
        df = self._psm_data()
        kw = {} if se_method is None else {"se_method": se_method}
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            res = sp.match(
                df, y="y", treat="d", covariates=["x1", "x2"], method="psm", **kw
            )
        return res, [str(w.message) for w in rec]

    def test_default_resolves_to_abadie_imbens(self):
        default, _ = self._fit()
        explicit, _ = self._fit("abadie_imbens")
        assert default.model_info["se_method"] == "abadie_imbens"
        assert default.se == pytest.approx(explicit.se)

    def test_default_no_longer_equals_ai(self):
        """The v1.22 change: this equality held before, and must not now."""
        default, _ = self._fit()
        ai, _ = self._fit("ai")
        assert default.se != pytest.approx(ai.se)
        assert default.se > ai.se  # the old default was too narrow

    def test_only_the_standard_error_moved(self):
        """⚠️ contract: point estimates are untouched by the default change."""
        default, _ = self._fit()
        ai, _ = self._fit("ai")
        assert default.estimate == pytest.approx(ai.estimate, rel=0, abs=0)
        assert default.n_obs == ai.n_obs

    def test_default_does_not_warn(self):
        """The default is now the recommended estimator, so it is quiet."""
        _, msgs = self._fit()
        assert not [m for m in msgs if "matched-pair SE" in m]

    def test_explicit_ai_warns_with_the_measured_shortfall(self):
        """'ai' remains available but must not pass silently."""
        _, msgs = self._fit("ai")
        hits = [m for m in msgs if "matched-pair SE" in m]
        assert hits
        assert "never reaches nominal coverage" in hits[0]
        assert "abadie_imbens" in hits[0]

    @staticmethod
    def _anti_conservative_warned(rec):
        return any("matched-pair SE" in str(w.message) for w in rec)

    def test_abadie_imbens_does_not_warn_and_is_larger(self):
        df = self._psm_data()
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            aimbens = sp.match(
                df,
                y="y",
                treat="d",
                covariates=["x1", "x2"],
                method="psm",
                se_method="abadie_imbens",
            )
        assert not self._anti_conservative_warned(rec)
        ai = sp.match(
            df, y="y", treat="d", covariates=["x1", "x2"], method="psm", se_method="ai"
        )
        # The rigorous SE is finite, positive, and (anti-conservative 'ai'
        # being too small) generally >= the naive one on this DGP.
        assert np.isfinite(aimbens.se) and aimbens.se > 0
        assert aimbens.se >= ai.se - 1e-9


# ----------------------------------------------------------------------
# IV cluster=<Series> no longer crashes; accepts a Series or column name
# ----------------------------------------------------------------------
class TestIVClusterSeriesGuard:
    def _iv_data(self, seed=0, n=600):
        rng = np.random.default_rng(seed)
        g = rng.integers(0, 40, n)
        z = rng.normal(size=n)
        x = 0.7 * z + rng.normal(size=n)
        y = 1.0 + 2.0 * x + rng.normal(size=n)
        return pd.DataFrame({"y": y, "x": x, "z": z, "g": g})

    def test_series_cluster_matches_string_cluster(self):
        df = self._iv_data()
        rs = sp.iv("y ~ (x ~ z)", data=df, cluster="g")
        rser = sp.iv("y ~ (x ~ z)", data=df, cluster=df["g"])
        # Passing the column as a Series must reproduce the column-name path
        # exactly (previously the Series raised "truth value ... ambiguous").
        np.testing.assert_allclose(
            np.atleast_1d(rs.std_errors), np.atleast_1d(rser.std_errors)
        )

    def test_length_mismatch_raises_clear_error(self):
        df = self._iv_data()
        # A misaligned cluster vector must fail loudly, not silently misalign.
        with pytest.raises(Exception, match="length"):
            sp.iv("y ~ (x ~ z)", data=df, cluster=df["g"].iloc[:500])


# ----------------------------------------------------------------------
# gardner_did: opt-in cluster bootstrap SE (default analytic understates)
# ----------------------------------------------------------------------
class TestGardnerBootstrapSE:
    def _panel(self, seed=1, nu=60, nt=6):
        rng = np.random.default_rng(seed)
        unit = np.repeat(np.arange(nu), nt)
        time = np.tile(np.arange(1, nt + 1), nu)
        cohort = np.where(unit < nu // 2, 4, 0)  # half treated at t=4
        first = cohort
        d = ((first > 0) & (time >= first)).astype(float)
        ui = np.repeat(rng.normal(size=nu), nt)
        y = ui + 0.3 * time + 2.0 * d + rng.normal(size=nu * nt)
        return pd.DataFrame({"y": y, "unit": unit, "time": time, "g": first})

    def test_bootstrap_keeps_point_but_widens_se(self):
        df = self._panel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = sp.gardner_did(
                df, y="y", group="unit", time="time", first_treat="g", vce="analytic"
            )
            b = sp.gardner_did(
                df,
                y="y",
                group="unit",
                time="time",
                first_treat="g",
                vce="bootstrap",
                n_boot=199,
                boot_seed=0,
            )
        # Bootstrap changes inference only, not the point estimate.
        assert b.estimate == pytest.approx(a.estimate)
        # ...and the analytic SE understates, so bootstrap is (weakly) larger.
        assert b.se >= a.se - 1e-9
        assert b.model_info["vce"] == "bootstrap"

    def test_analytic_default_warns_bootstrap_does_not(self):
        df = self._panel()
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            sp.gardner_did(df, y="y", group="unit", time="time", first_treat="g")
        assert any("Stage-1" in str(w.message) for w in rec)

        with warnings.catch_warnings(record=True) as rec2:
            warnings.simplefilter("always")
            sp.gardner_did(
                df,
                y="y",
                group="unit",
                time="time",
                first_treat="g",
                vce="bootstrap",
                n_boot=49,
            )
        assert not any("Stage-1" in str(w.message) for w in rec2)


# ----------------------------------------------------------------------
# did_imputation: opt-in cluster bootstrap for the overall-ATT SE
# ----------------------------------------------------------------------
class TestDidImputationBootstrapSE:
    def _panel(self, seed=2, nu=60, nt=6):
        rng = np.random.default_rng(seed)
        unit = np.repeat(np.arange(nu), nt)
        time = np.tile(np.arange(1, nt + 1), nu)
        first = np.where(unit < nu // 2, 4, 0)
        d = ((first > 0) & (time >= first)).astype(float)
        ui = np.repeat(rng.normal(size=nu), nt)
        y = ui + 0.3 * time + 2.0 * d + rng.normal(size=nu * nt)
        return pd.DataFrame({"y": y, "unit": unit, "time": time, "g": first})

    def test_bootstrap_keeps_point_but_widens_se(self):
        df = self._panel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = sp.did_imputation(
                df, y="y", group="unit", time="time", first_treat="g", vce="analytic"
            )
            b = sp.did_imputation(
                df,
                y="y",
                group="unit",
                time="time",
                first_treat="g",
                vce="bootstrap",
                n_boot=199,
                boot_seed=0,
            )
        assert b.estimate == pytest.approx(a.estimate)
        assert b.se >= a.se - 1e-9
        assert b.model_info["vce"] == "bootstrap"

    def test_analytic_default_warns(self):
        df = self._panel()
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            sp.did_imputation(df, y="y", group="unit", time="time", first_treat="g")
        assert any("anti-conservative" in str(w.message) for w in rec)


# ----------------------------------------------------------------------
# Callaway-Sant'Anna pre-trend test: Hotelling-F finite-sample correction
# ----------------------------------------------------------------------
class TestCSPretrendFCorrection:
    def _panel(self, seed=0, nu=51, nt=8):
        rng = np.random.default_rng(seed)
        cohort_by_unit = np.where(
            np.arange(nu) % 3 == 0, 0, np.where(np.arange(nu) % 3 == 1, 4, 6)
        )
        unit = np.repeat(np.arange(nu), nt)
        time = np.tile(np.arange(1, nt + 1), nu)
        first = np.repeat(cohort_by_unit, nt)
        d = ((first > 0) & (time >= first)).astype(float)
        ui = np.repeat(rng.normal(size=nu), nt)
        ti = np.tile(rng.normal(size=nt), nu)
        y = ui + ti + 2.0 * d + rng.normal(size=nu * nt)
        return pd.DataFrame({"y": y, "unit": unit, "time": time, "g": first})

    def test_pretrend_pvalue_uses_hotelling_f_not_chi2(self):
        df = self._panel()
        res = sp.callaway_santanna(df, y="y", g="g", t="time", i="unit")
        pt = res.model_info["pretrend_test"]
        W, k = pt["statistic"], pt["df"]
        assert k >= 1
        G = int(df["unit"].nunique())
        f_stat = W * (G - k) / (k * (G - 1))
        # The reported p-value is the F(k, G-k) tail, not the chi²(k) tail.
        assert pt["pvalue"] == pytest.approx(
            float(stats.f.sf(f_stat, k, G - k)), rel=1e-6
        )
        # F-correction is (weakly) more conservative than the old chi² Wald.
        assert pt["pvalue"] >= float(stats.chi2.sf(W, k)) - 1e-9


# ----------------------------------------------------------------------
# gardner_did event-study ATT is treated-obs-weighted (matches non-ES path)
# ----------------------------------------------------------------------
class TestGardnerEventStudyWeighting:
    def _hetero_panel(self, seed=0, nu=90, nt=8):
        rng = np.random.default_rng(seed)
        rows = []
        for u in range(nu):
            first = [4, 6, 0][u % 3]
            eff = {4: 1.0, 6: 3.0, 0: 0.0}[first]  # heterogeneous by cohort
            for t in range(1, nt + 1):
                d = first > 0 and t >= first
                y = 0.3 * u + 0.4 * t + (eff if d else 0.0) + rng.normal()
                rows.append({"unit": u, "time": t, "g": first, "y": y})
        return pd.DataFrame(rows)

    def test_event_study_overall_att_matches_non_event_study(self):
        df = self._hetero_panel()
        hz = list(range(-4, 5))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = sp.gardner_did(
                df, y="y", group="unit", time="time", first_treat="g", event_study=False
            )
            b = sp.gardner_did(
                df,
                y="y",
                group="unit",
                time="time",
                first_treat="g",
                event_study=True,
                horizon=hz,
            )
        # Obs-weighted ES aggregation must equal the (obs-weighted) non-ES ATT;
        # the previous unweighted mean disagreed under heterogeneity.
        assert b.estimate == pytest.approx(a.estimate, abs=1e-6)
