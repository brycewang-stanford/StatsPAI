"""Unit tests for the batch-E ``did_imputation`` options.

Covers the Stata ``did_imputation`` gap-closure additions:
``pretrends=``, ``balanced=`` (hbalance), ``min_n=`` (minn),
``hetby=``, ``save_weights=`` (saveweights), ``save_residuals=``
(saveresid).

The core imputation estimator has its own parity tests; here we lock in
the option semantics, the exact-weights identity ``ATT = w'y``, and the
error paths.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _panel(n_units=60, n_periods=9, het=False, missing_tail_unit=None, seed=0):
    """Two treated cohorts (4, 7) + never-treated; optional heterogeneity."""
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(n_units):
        first = [4, 7, 0][unit % 3]
        region = unit % 2
        for year in range(1, n_periods + 1):
            if missing_tail_unit is not None and unit == missing_tail_unit:
                if year == n_periods:
                    continue  # unbalanced tail for hbalance tests
            treated = first != 0 and year >= first
            slope = 2.0 + region if het else 2.0
            te = slope * (year - first + 1) if treated else 0.0
            rows.append(
                {
                    "county": unit,
                    "year": year,
                    "wage": unit * 0.1 + year + te + rng.normal(),
                    "first_treat": first,
                    "region": region,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def panel():
    return _panel()


def _fit(df, **kw):
    with warnings.catch_warnings():
        # The analytic-SE anti-conservativeness warning is expected here.
        warnings.simplefilter("ignore", UserWarning)
        return sp.did_imputation(
            df,
            y="wage",
            group="county",
            time="year",
            first_treat="first_treat",
            **kw,
        )


# ----------------------------------------------------------------------
# pretrends=k
# ----------------------------------------------------------------------


class TestPretrends:
    def test_adds_placebo_horizons_and_joint_test(self, panel):
        r = _fit(panel, pretrends=3)
        pt = r.model_info["pretrend_test"]
        assert pt["periods"] == [-3, -2, -1]
        assert pt["df"] == 3
        assert 0.0 <= pt["pvalue"] <= 1.0
        es = r.model_info["event_study"]
        assert set(es["relative_time"]) == {-3, -2, -1}

    def test_merges_with_horizon(self, panel):
        r = _fit(panel, pretrends=2, horizon=[0, 1, 2])
        es = r.model_info["event_study"]
        assert set(es["relative_time"]) == {-2, -1, 0, 1, 2}
        # Joint test restricted to the requested placebo window.
        assert r.model_info["pretrend_test"]["periods"] == [-2, -1]

    def test_clean_dgp_does_not_reject(self, panel):
        # Parallel-trends DGP → the placebo joint test should not reject.
        r = _fit(panel, pretrends=3)
        assert r.model_info["pretrend_test"]["pvalue"] > 0.05

    def test_broken_pretrends_reject(self):
        # Inject a strong pre-trend for eventually-treated units only.
        df = _panel(seed=3)
        bump = (df["first_treat"] > 0) & (df["year"] < df["first_treat"])
        df.loc[bump, "wage"] += 3.0 * df.loc[bump, "year"]
        r = _fit(df, pretrends=3)
        assert r.model_info["pretrend_test"]["pvalue"] < 0.01

    @pytest.mark.parametrize("bad", [0, -2, True])
    def test_invalid_pretrends_raise(self, panel, bad):
        with pytest.raises(ValueError, match="pretrends"):
            _fit(panel, pretrends=bad)


# ----------------------------------------------------------------------
# balanced (hbalance)
# ----------------------------------------------------------------------


class TestBalanced:
    def test_drops_units_missing_horizons(self):
        # Cohort-7 units only reach relative time +2 on a 9-period panel;
        # requesting horizons up to +4 must drop that whole cohort.
        df = _panel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with pytest.warns(UserWarning, match="balanced=True dropped"):
                r = sp.did_imputation(
                    df,
                    y="wage",
                    group="county",
                    time="year",
                    first_treat="first_treat",
                    horizon=list(range(0, 5)),
                    balanced=True,
                )
        assert r.model_info["balanced"] is True
        assert r.model_info["n_units_dropped_balance"] == 20  # cohort 7
        # Every reported horizon now has the same composition size.
        es = r.model_info["event_study"]
        assert es["n_obs"].nunique() == 1

    def test_negative_horizons_do_not_drop(self, panel):
        # Cohort-4 units lack rel time -4 (panel starts at year 1), but
        # hbalance balances on the post-treatment window only.
        r = _fit(panel, horizon=list(range(-4, 3)), balanced=True)
        assert r.model_info.get("n_units_dropped_balance", 0) == 0

    def test_requires_horizon(self, panel):
        with pytest.raises(ValueError, match="balanced"):
            _fit(panel, balanced=True)

    def test_requires_nonnegative_horizon(self, panel):
        with pytest.raises(ValueError, match="non-negative"):
            _fit(panel, balanced=True, pretrends=2)


# ----------------------------------------------------------------------
# min_n
# ----------------------------------------------------------------------


class TestMinN:
    def test_drops_thin_horizons(self):
        # Horizons 3-5 are reached by cohort 4 only (20 units); 0-2 by
        # both cohorts (40). min_n=21 must drop exactly the thin tail.
        df = _panel()
        with pytest.warns(UserWarning, match="min_n"):
            r = sp.did_imputation(
                df,
                y="wage",
                group="county",
                time="year",
                first_treat="first_treat",
                horizon=list(range(0, 6)),
                min_n=21,
            )
        es = r.model_info["event_study"]
        assert set(es["relative_time"]) == {0, 1, 2}
        assert (es["n_obs"] >= 21).all()

    def test_no_filter_when_all_horizons_thick(self, panel):
        r_plain = _fit(panel, horizon=[0, 1, 2])
        r_minn = _fit(panel, horizon=[0, 1, 2], min_n=5)
        pd.testing.assert_frame_equal(
            r_plain.model_info["event_study"], r_minn.model_info["event_study"]
        )

    def test_requires_horizon(self, panel):
        with pytest.raises(ValueError, match="min_n"):
            _fit(panel, min_n=10)

    @pytest.mark.parametrize("bad", [0, -1, True])
    def test_invalid_min_n_raises(self, panel, bad):
        with pytest.raises(ValueError, match="min_n"):
            _fit(panel, horizon=[0, 1], min_n=bad)


# ----------------------------------------------------------------------
# hetby
# ----------------------------------------------------------------------


class TestHetby:
    def test_recovers_heterogeneous_slopes(self):
        # DGP: region-1 effects are exactly 1.0 · (rel_time + 1) larger.
        df = _panel(het=True, seed=1)
        r = _fit(df, hetby="region")
        het = r.model_info["hetby"].set_index("region")
        assert set(het.index) == {0, 1}
        assert het.loc[1, "att"] > het.loc[0, "att"]
        # Both subgroup ATTs are estimated, positive, and significant.
        assert (het["pvalue"] < 0.01).all()
        # Subgroup ATTs average (obs-weighted) back to the overall ATT.
        w = het["n_obs"] / het["n_obs"].sum()
        assert np.isclose(float((het["att"] * w).sum()), r.estimate, atol=1e-10)

    def test_missing_column_raises(self, panel):
        with pytest.raises(ValueError, match="hetby"):
            _fit(panel, hetby="nope")

    def test_time_varying_hetby_raises(self, panel):
        df = panel.assign(tv=(panel["year"] > 4).astype(int))
        with pytest.raises(ValueError, match="time-varying"):
            _fit(df, hetby="tv")


# ----------------------------------------------------------------------
# save_weights / save_residuals
# ----------------------------------------------------------------------


class TestExports:
    def test_weights_reproduce_att_exactly(self, panel):
        r = _fit(panel, save_weights=True)
        w = r.model_info["estimation_weights"]
        att_from_w = float(w.values @ panel["wage"].values)
        assert np.isclose(att_from_w, r.estimate, atol=1e-8)

    def test_weights_reproduce_att_with_controls(self, panel):
        df = panel.assign(x1=np.random.default_rng(9).normal(size=len(panel)))
        r = _fit(df, controls=["x1"], save_weights=True)
        att_from_w = float(
            r.model_info["estimation_weights"].values @ df["wage"].values
        )
        assert np.isclose(att_from_w, r.estimate, atol=1e-6)

    def test_weight_structure(self, panel):
        r = _fit(panel, save_weights=True)
        w = r.model_info["estimation_weights"]
        treated = (panel["first_treat"] > 0) & (panel["year"] >= panel["first_treat"])
        n1 = int(treated.sum())
        # Treated rows carry exactly 1/N1.
        np.testing.assert_allclose(w[treated.values].values, 1.0 / n1)
        # Weights on y sum to ~0 (ATT is a contrast, not a level).
        assert abs(float(w.sum())) < 1e-6

    def test_weights_align_after_balancing(self):
        df = _panel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = sp.did_imputation(
                df,
                y="wage",
                group="county",
                time="year",
                first_treat="first_treat",
                horizon=[0, 1, 2, 3, 4],
                balanced=True,
                save_weights=True,
            )
        w = r.model_info["estimation_weights"]
        # Index maps back into the caller's frame (subset of df rows).
        assert w.index.isin(df.index).all()
        att_from_w = float(w.values @ df.loc[w.index, "wage"].values)
        assert np.isclose(att_from_w, r.estimate, atol=1e-8)

    def test_residuals_nan_on_treated_zero_mean_untreated(self, panel):
        r = _fit(panel, save_residuals=True)
        res = r.model_info["residuals"]
        treated = (panel["first_treat"] > 0) & (panel["year"] >= panel["first_treat"])
        assert res[treated.values].isna().all()
        assert res[~treated.values].notna().all()
        # Untreated residuals from the least-squares fit average to ~0.
        assert abs(float(res.dropna().mean())) < 1e-8

    def test_exports_off_by_default(self, panel):
        r = _fit(panel)
        assert "estimation_weights" not in r.model_info
        assert "residuals" not in r.model_info


# ----------------------------------------------------------------------
# aliases
# ----------------------------------------------------------------------


def test_bjs_alias_inherits_options(panel):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        r = sp.bjs(
            panel,
            y="wage",
            group="county",
            time="year",
            first_treat="first_treat",
            pretrends=2,
            save_weights=True,
        )
    assert r.model_info["pretrend_test"]["periods"] == [-2, -1]
    assert "estimation_weights" in r.model_info
