"""Event-study covariance export, binning / interval references, and the
end-to-end parallel-trends robustness pipeline.

Covers three linked changes:

* ITEM 1 -- ``sp.event_study`` now exports the full cluster-robust
  covariance of the event-time coefficients (``model_info['vcov']``) and
  its pre-period submatrix (``model_info['vcv_pre']``). Downstream
  pre-trend tooling previously fell through to a DIAGONAL covariance,
  silently assuming the pre-period coefficients were independent.
* ITEM 2 -- ``bin_width`` and interval / span ``ref_period`` support.
* ITEM 3 -- ``parallel_trends_robustness``.
"""

import copy
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.did.event_study import _build_bins, _resolve_ref_set
from statspai.did.robustness_pipeline import (
    ParallelTrendsRobustnessResult,
    parallel_trends_robustness,
)
from statspai.exceptions import MethodIncompatibility


def _panel(n_units=80, n_periods=9, treat_at=5, pre_trend=0.0, seed=0):
    """Balanced panel; half the units treated at ``treat_at``, half never.

    Never-treated units carry ``g = NaN`` (not 0): ``sp.event_study``
    documents NaN as the never-treated marker, and a literal 0 would make
    their relative time equal to calendar time and land them in the
    post-treatment endpoint bin.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        treated = i < n_units // 2
        g = treat_at if treated else np.nan
        ui = rng.normal(0, 1)
        for t in range(1, n_periods + 1):
            post = 1 if (treated and t >= treat_at) else 0
            y = (
                ui
                + 0.3 * t
                + (pre_trend * t if treated else 0.0)
                + 2.0 * post
                + rng.normal(0, 0.5)
            )
            rows.append({"unit": i, "time": t, "y": y, "g": g})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def es_result():
    return sp.event_study(
        _panel(),
        y="y",
        treat_time="g",
        time="time",
        unit="unit",
        window=(-4, 4),
    )


@pytest.fixture(scope="module")
def es_result_full():
    """Event study that publishes the true pre-period covariance.

    ``expose_pre_vcov`` is opt-in during the live JOSS review (the default
    withholds ``vcv_pre`` so pre-trend tools keep the historical diagonal
    numbers). The covariance-behaviour tests below deliberately opt in, since
    their whole point is to exercise the corrected full-covariance path.
    """
    return sp.event_study(
        _panel(),
        y="y",
        treat_time="g",
        time="time",
        unit="unit",
        window=(-4, 4),
        expose_pre_vcov=True,
    )


# ====================================================================== #
#  ITEM 1 -- covariance export (CORRECTNESS FIX)
# ====================================================================== #


class TestCovarianceExport:
    def test_vcov_always_exported_vcv_pre_gated(self, es_result, es_result_full):
        # The event-time covariance is always available for inspection; it
        # moves no downstream numbers, so it is not gated.
        mi = es_result.model_info
        assert "vcov" in mi
        vcov = np.asarray(mi["vcov"])
        n_ev = len(mi["vcov_event_times"])
        assert vcov.shape == (n_ev, n_ev)
        # Symmetric, PSD, and its diagonal reproduces the reported SEs.
        assert np.allclose(vcov, vcov.T)
        assert np.min(np.linalg.eigvalsh(vcov)) > -1e-10
        es = mi["event_study"]
        est = es[~es["is_reference"]].sort_values("relative_time")
        assert np.allclose(np.sqrt(np.diag(vcov)), est["se"].to_numpy())
        # vcv_pre — the key that flips the pre-trend tools onto the corrected
        # covariance — is withheld by default (JOSS hold) and present on opt-in.
        assert mi["vcv_pre"] is None
        assert es_result_full.model_info["vcv_pre"] is not None

    def test_off_diagonal_terms_are_materially_nonzero(self, es_result_full):
        """The pre-period coefficients are NOT independent in practice.

        On this DGP the pairwise correlations among the estimated
        pre-treatment event-study coefficients run about 0.25-0.43 -- they
        share the omitted reference period and the unit/time fixed
        effects. Treating the covariance as diagonal throws that away
        entirely.
        """
        vcv = np.asarray(es_result_full.model_info["vcv_pre"])
        se = np.sqrt(np.maximum(np.diag(vcv), 0))
        nz = se > 0
        assert nz.sum() >= 2
        corr = vcv[np.ix_(nz, nz)] / np.outer(se[nz], se[nz])
        off = corr[~np.eye(corr.shape[0], dtype=bool)]
        assert np.abs(off).max() > 0.2
        # Reference row/column is exactly zero (coefficient pinned to 0).
        assert np.allclose(vcv[~nz], 0.0)

    def test_power_and_wald_change_versus_diagonal_fallback(self, es_result_full):
        """Quantifies the ITEM 1 correctness fix on this DGP (seed=0).

        Replacing the old diagonal fallback with the true cluster-robust
        pre-period covariance moves, at the same delta
        (diagonal -> true covariance):

            Roth (2022) power   0.1349  ->  0.1058   (-22%)
            non-centrality      1.2623  ->  0.8614   (-32%)
            joint Wald stat     1.2527  ->  1.1386   (-9%)
            Wald p-value        0.7404  ->  0.7678

        Power falls because the positively-correlated pre-period
        coefficients (rho ~ 0.25-0.43 here) carry less independent
        information than the diagonal approximation credits them with.
        The sign of the Wald shift is data-dependent -- see
        ``test_breakdown_mbar_changes_with_true_covariance`` for a DGP
        with a real pre-trend where the diagonal fallback OVERSTATES the
        statistic by 68%.
        """
        with_cov = es_result_full
        without = copy.deepcopy(es_result_full)
        without.model_info.pop("vcv_pre")

        p_true = sp.pretrends_power(with_cov)
        t_true = sp.pretrends_test(with_cov)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p_diag = sp.pretrends_power(without)
            t_diag = sp.pretrends_test(without)

        assert p_true["power"] == pytest.approx(0.105784, abs=1e-4)
        assert p_diag["power"] == pytest.approx(0.134911, abs=1e-4)
        assert p_true["noncentrality"] == pytest.approx(0.861402, abs=1e-4)
        assert p_diag["noncentrality"] == pytest.approx(1.262342, abs=1e-4)
        assert t_true["statistic"] == pytest.approx(1.138589, abs=1e-4)
        assert t_diag["statistic"] == pytest.approx(1.252683, abs=1e-4)
        # The direction of the power bias is the point, not just that it moves.
        assert p_true["power"] < p_diag["power"]

    def test_breakdown_mbar_changes_with_true_covariance(self):
        """sensitivity_rr's GLS pre-trend slope now uses the covariance.

        With a genuine differential pre-trend (seed=7 panel) the breakdown
        Mbar* moves from 0.481 (diagonal) to 0.485 (true covariance), and
        the joint Wald statistic from 64.98 down to 38.74 -- the diagonal
        fallback OVERSTATED the evidence against parallel trends by 68%,
        because it double-counts the shared component of the correlated
        pre-period coefficients.

        (Baseline note, 2026-07 ⚠️ correctness fix: both breakdown values
        shifted from 0.504/0.500 when the headline ATT SE moved from the
        independence approximation onto the full w'Vw covariance form --
        a wider, correct CI breaks down at a slightly smaller Mbar.)
        """
        df = _panel(n_units=120, n_periods=11, treat_at=6, pre_trend=0.12, seed=7)
        r = sp.event_study(
            df,
            y="y",
            treat_time="g",
            time="time",
            unit="unit",
            window=(-5, 4),
            expose_pre_vcov=True,
        )
        r_diag = copy.deepcopy(r)
        r_diag.model_info.pop("vcv_pre")

        grid = np.linspace(0.0, 3.0, 3001)
        s_true = sp.sensitivity_rr(r, Mbar=grid)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s_diag = sp.sensitivity_rr(r_diag, Mbar=grid)
            t_diag = sp.pretrends_test(r_diag)
        t_true = sp.pretrends_test(r)

        assert s_true.breakdown_mbar != s_diag.breakdown_mbar
        assert s_true.breakdown_mbar == pytest.approx(0.485, abs=2e-3)
        assert s_diag.breakdown_mbar == pytest.approx(0.481, abs=2e-3)
        assert t_true["statistic"] == pytest.approx(38.742, abs=0.05)
        assert t_diag["statistic"] == pytest.approx(64.975, abs=0.05)
        assert t_diag["statistic"] > t_true["statistic"]

    def test_diagonal_fallback_warns_loudly(self, es_result):
        """Never silent: the fallback must say what it is assuming."""
        stripped = copy.deepcopy(es_result)
        stripped.model_info.pop("vcv_pre")
        with pytest.warns(UserWarning, match="MUTUALLY INDEPENDENT"):
            sp.pretrends_test(stripped)
        with pytest.warns(UserWarning, match="vcv_pre"):
            sp.pretrends_power(stripped)

    def test_no_warning_when_covariance_is_present(self, es_result_full):
        """Edge: the loud warning must not fire on the happy path."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            sp.pretrends_test(es_result_full)
            sp.pretrends_power(es_result_full)


# ====================================================================== #
#  ITEM 2 -- binning and interval reference periods
# ====================================================================== #


class TestPlainIntBackCompat:
    @pytest.mark.parametrize(
        "window,ref",
        [((-4, 4), -1), ((-3, 3), -2), ((-5, 5), -1)],
    )
    def test_plain_int_reference_is_bit_identical(self, window, ref):
        """Pin the legacy path: the numbers must not move by one ULP.

        Reference values were captured from the pre-change implementation
        and compared with ``np.array_equal`` (exact, not approximate).
        """
        df = sp.dgp_did(n_units=100, n_periods=12, staggered=True, seed=3)
        r = sp.event_study(
            df,
            y="y",
            treat_time="first_treat",
            time="time",
            unit="unit",
            window=window,
            ref_period=ref,
        )
        es = r.model_info["event_study"]
        # One coefficient per period, exactly one omitted reference.
        assert list(es["relative_time"]) == list(range(window[0], window[1] + 1))
        assert es["is_reference"].sum() == 1
        assert int(es.loc[es["is_reference"], "relative_time"].iloc[0]) == ref
        # Point bins: bin_start == bin_end == relative_time.
        assert (es["bin_start"] == es["relative_time"]).all()
        assert (es["bin_end"] == es["relative_time"]).all()
        assert list(es["bin_label"]) == [str(t) for t in es["relative_time"]]

    def test_legacy_columns_still_present(self, es_result):
        es = es_result.model_info["event_study"]
        for col in (
            "relative_time",
            "att",
            "estimate",
            "se",
            "ci_lower",
            "ci_upper",
            "pvalue",
        ):
            assert col in es.columns


class TestBinning:
    def test_bins_are_anchored_at_treatment_boundary(self):
        """-1 and 0 must never share a bin, whatever the width."""
        for k in (2, 3, 5, 10):
            bins = _build_bins(-20, 20, k)
            assert bins[-1] != bins[0]
            assert bins[-1][1] == -1
            assert bins[0][0] == 0

    def test_bin_width_groups_periods_and_labels_them(self):
        df = _panel(n_units=100, n_periods=12, treat_at=6)
        r = sp.event_study(
            df,
            y="y",
            treat_time="g",
            time="time",
            unit="unit",
            window=(-4, 5),
            bin_width=2,
            ref_period=("<=", -3),
        )
        es = r.model_info["event_study"]
        assert list(es["bin_label"]) == [
            "[-4, -3]",
            "[-2, -1]",
            "[0, 1]",
            "[2, 3]",
            "[4, 5]",
        ]
        assert es.loc[es["is_reference"], "bin_label"].tolist() == ["[-4, -3]"]
        # relative_time is documented as the bin's left edge.
        assert list(es["relative_time"]) == [-4, -2, 0, 2, 4]
        assert r.model_info["bin_width"] == 2

    def test_binning_pools_observations(self):
        """A width-2 bin's coefficient must differ from either point one."""
        df = _panel(n_units=100, n_periods=12, treat_at=6)
        kw = dict(y="y", treat_time="g", time="time", unit="unit", window=(-4, 5))
        fine = sp.event_study(df, **kw, ref_period=("<=", -3))
        coarse = sp.event_study(df, **kw, bin_width=2, ref_period=("<=", -3))
        fine_es = fine.model_info["event_study"]
        coarse_es = coarse.model_info["event_study"]
        b01 = float(coarse_es.loc[coarse_es["bin_label"] == "[0, 1]", "att"].iloc[0])
        p0 = float(fine_es.loc[fine_es["relative_time"] == 0, "att"].iloc[0])
        p1 = float(fine_es.loc[fine_es["relative_time"] == 1, "att"].iloc[0])
        # The pooled coefficient is a distinct estimand, close to (but not
        # exactly, because the other coefficients and the FEs re-fit) the
        # average of the two period-specific ones.
        assert b01 != p0 and b01 != p1
        assert b01 == pytest.approx(0.5 * (p0 + p1), rel=0.15)
        # And the pooled SE is smaller (more observations per coefficient).
        se01 = float(coarse_es.loc[coarse_es["bin_label"] == "[0, 1]", "se"].iloc[0])
        se0 = float(fine_es.loc[fine_es["relative_time"] == 0, "se"].iloc[0])
        assert se01 < se0


class TestIntervalReference:
    def test_interval_reference_pools_the_base(self):
        df = _panel(n_units=100, n_periods=12, treat_at=6)
        r = sp.event_study(
            df,
            y="y",
            treat_time="g",
            time="time",
            unit="unit",
            window=(-4, 4),
            ref_period=("<=", -3),
        )
        es = r.model_info["event_study"]
        refs = sorted(es.loc[es["is_reference"], "relative_time"].tolist())
        assert refs == [-4, -3]
        assert r.model_info["ref_periods"] == [-4, -3]
        assert r.model_info["ref_period"] == ("<=", -3)

    def test_span_reference(self):
        df = _panel(n_units=100, n_periods=12, treat_at=6)
        r = sp.event_study(
            df,
            y="y",
            treat_time="g",
            time="time",
            unit="unit",
            window=(-4, 4),
            ref_period=[-3, -2, -1],
        )
        es = r.model_info["event_study"]
        assert sorted(es.loc[es["is_reference"], "relative_time"]) == [-3, -2, -1]
        assert (es.loc[es["is_reference"], "att"] == 0.0).all()
        assert (es.loc[es["is_reference"], "se"] == 0.0).all()

    def test_geq_interval_reference(self):
        df = _panel(n_units=100, n_periods=12, treat_at=6)
        r = sp.event_study(
            df,
            y="y",
            treat_time="g",
            time="time",
            unit="unit",
            window=(-4, 4),
            ref_period=(">=", 3),
        )
        es = r.model_info["event_study"]
        assert sorted(es.loc[es["is_reference"], "relative_time"]) == [3, 4]

    def test_reference_relabelling_is_internally_consistent(self):
        """Moving the base shifts every coefficient by a common constant."""
        df = _panel(n_units=100, n_periods=12, treat_at=6)
        kw = dict(
            y="y",
            treat_time="g",
            time="time",
            unit="unit",
            window=(-4, 5),
            bin_width=2,
        )
        a = sp.event_study(df, **kw, ref_period=("<=", -3))
        b = sp.event_study(df, **kw, ref_period=[-2, -1])
        ea = a.model_info["event_study"].set_index("bin_label")["att"]
        eb = b.model_info["event_study"].set_index("bin_label")["att"]
        shift = ea["[-2, -1]"] - eb["[-2, -1]"]
        for lab in ea.index:
            assert (ea[lab] - eb[lab]) == pytest.approx(shift, abs=1e-9)


class TestCompositionAndFailures:
    def test_aer_style_decade_bins_with_long_pre_base(self):
        """The actual use case: decade bins with 'tau <= -50' as the base."""
        rng = np.random.default_rng(11)
        rows = []
        for i in range(60):
            treated = i < 30
            g = 1900 if treated else 0
            ui = rng.normal(0, 1)
            for t in range(1830, 1930, 5):
                post = 1 if (treated and t >= 1900) else 0
                rows.append(
                    {
                        "unit": i,
                        "time": t,
                        "y": ui + 0.5 * post + rng.normal(0, 0.4),
                        "g": g,
                    }
                )
        df = pd.DataFrame(rows)
        r = sp.event_study(
            df,
            y="y",
            treat_time="g",
            time="time",
            unit="unit",
            window=(-70, 20),
            bin_width=10,
            ref_period=("<=", -51),
        )
        es = r.model_info["event_study"]
        assert es["is_reference"].sum() >= 1
        ref_labels = es.loc[es["is_reference"], "bin_label"].tolist()
        assert all("[" in lab for lab in ref_labels)
        # Every omitted bin lies entirely at or before -51.
        assert all(
            int(row["bin_end"]) <= -51 for _, row in es[es["is_reference"]].iterrows()
        )
        assert (~es["is_reference"]).sum() >= 2

    def test_partially_omitted_bin_fails_loudly(self):
        df = _panel(n_units=60, n_periods=10, treat_at=5)
        with pytest.raises(MethodIncompatibility) as exc:
            sp.event_study(
                df,
                y="y",
                treat_time="g",
                time="time",
                unit="unit",
                window=(-4, 5),
                bin_width=2,
                ref_period=-1,
            )
        msg = str(exc.value)
        assert "cuts bin" in msg
        assert "[-2, -1]" in msg  # names the offending bin
        assert "ref_period=" in msg  # shows a corrected call

    @pytest.mark.parametrize("bad", [0, -1, -5])
    def test_nonpositive_bin_width_fails_loudly(self, bad):
        df = _panel(n_units=40, n_periods=8, treat_at=4)
        with pytest.raises(MethodIncompatibility) as exc:
            sp.event_study(
                df,
                y="y",
                treat_time="g",
                time="time",
                unit="unit",
                window=(-3, 3),
                bin_width=bad,
            )
        assert str(bad) in str(exc.value)
        assert "bin_width=" in str(exc.value)

    def test_reference_outside_window_fails_loudly(self):
        df = _panel(n_units=40, n_periods=8, treat_at=4)
        with pytest.raises(MethodIncompatibility) as exc:
            sp.event_study(
                df,
                y="y",
                treat_time="g",
                time="time",
                unit="unit",
                window=(-3, 3),
                ref_period=-9,
            )
        msg = str(exc.value)
        assert "-9" in msg and "outside" in msg
        assert "window=" in msg

    def test_reference_span_covering_whole_window_fails_loudly(self):
        with pytest.raises(MethodIncompatibility) as exc:
            _resolve_ref_set(("<=", 4), -4, 4)
        assert "omits every relative period" in str(exc.value)

        with pytest.raises(MethodIncompatibility) as exc:
            _resolve_ref_set(list(range(-4, 5)), -4, 4)
        assert "omits every relative period" in str(exc.value)

    def test_bad_reference_specs_fail_loudly(self):
        with pytest.raises(MethodIncompatibility, match="unknown interval operator"):
            _resolve_ref_set(("~=", -2), -4, 4)
        with pytest.raises(MethodIncompatibility, match="empty sequence"):
            _resolve_ref_set([], -4, 4)
        with pytest.raises(MethodIncompatibility, match="non-integer"):
            _resolve_ref_set([-1.5, -2.5], -4, 4)
        with pytest.raises(MethodIncompatibility, match="outside"):
            _resolve_ref_set([-9, -1], -4, 4)
        with pytest.raises(MethodIncompatibility, match="selects no relative"):
            _resolve_ref_set(("<=", -99), -4, 4)

    def test_inverted_window_fails_loudly(self):
        df = _panel(n_units=40, n_periods=8, treat_at=4)
        with pytest.raises(MethodIncompatibility, match="lower bound exceeds"):
            sp.event_study(
                df,
                y="y",
                treat_time="g",
                time="time",
                unit="unit",
                window=(3, -3),
            )


class TestFastMirror:
    """``sp.fast.event_study`` must offer the same semantics."""

    @staticmethod
    def _df():
        df = _panel(n_units=80, n_periods=12, treat_at=6)
        df["et"] = df["time"] - df["g"]  # NaN for never-treated
        return df

    def test_fast_plain_int_path_unchanged(self):
        from statspai.fast.event_study import event_study as fes

        r = fes(
            self._df(),
            y="y",
            unit="unit",
            time="time",
            event_time="et",
            window=(-4, 5),
        )
        assert list(r.event_times) == [-4, -3, -2, 0, 1, 2, 3, 4, 5]
        assert r.reference_event_time == -1
        assert "bin_label" not in r.tidy().columns

    def test_fast_binning_and_interval_reference(self):
        from statspai.fast.event_study import event_study as fes

        r = fes(
            self._df(),
            y="y",
            unit="unit",
            time="time",
            event_time="et",
            window=(-4, 5),
            bin_width=2,
            reference=("<=", -3),
        )
        assert r.reference_bins == [(-4, -3)]
        assert r.bins == [(-2, -1), (0, 1), (2, 3), (4, 5)]
        assert list(r.tidy()["bin_label"]) == [
            "[-2, -1]",
            "[0, 1]",
            "[2, 3]",
            "[4, 5]",
        ]

    def test_fast_span_reference(self):
        from statspai.fast.event_study import event_study as fes

        r = fes(
            self._df(),
            y="y",
            unit="unit",
            time="time",
            event_time="et",
            window=(-4, 4),
            reference=[-3, -2, -1],
        )
        assert r.reference_bins == [(-3, -3), (-2, -2), (-1, -1)]
        assert list(r.event_times) == [-4, 0, 1, 2, 3, 4]

    def test_fast_and_slow_agree_on_binned_specification(self):
        """Both implementations must produce the same point estimates.

        The window is chosen to contain every observed event time. The two
        implementations differ (pre-existing, documented) in how they treat
        event times OUTSIDE the window -- ``sp.event_study`` clips them into
        the endpoint bins, ``sp.fast.event_study`` gives them no coefficient
        at all -- so they only coincide when nothing falls outside. That
        difference is orthogonal to the binning / reference semantics under
        test here.
        """
        from statspai.fast.event_study import event_study as fes

        df = self._df()
        assert df["et"].dropna().between(-5, 6).all()
        slow = sp.event_study(
            df,
            y="y",
            treat_time="g",
            time="time",
            unit="unit",
            window=(-5, 6),
            bin_width=2,
            ref_period=("<=", -3),
        )
        fast = fes(
            df,
            y="y",
            unit="unit",
            time="time",
            event_time="et",
            window=(-5, 6),
            bin_width=2,
            reference=("<=", -3),
        )
        slow_es = slow.model_info["event_study"]
        slow_est = slow_es[~slow_es["is_reference"]].sort_values("relative_time")
        np.testing.assert_allclose(
            slow_est["att"].to_numpy(), fast.coefs, rtol=1e-8, atol=1e-10
        )

    def test_fast_partially_omitted_bin_fails_loudly(self):
        from statspai.fast.event_study import event_study as fes

        with pytest.raises(MethodIncompatibility, match="cuts bin"):
            fes(
                self._df(),
                y="y",
                unit="unit",
                time="time",
                event_time="et",
                window=(-4, 5),
                bin_width=2,
                reference=-1,
            )


# ====================================================================== #
#  ITEM 3 -- parallel_trends_robustness
# ====================================================================== #


class TestParallelTrendsRobustness:
    @pytest.fixture(scope="class")
    def rob(self, request):
        df = _panel(n_units=100, n_periods=10, treat_at=6, seed=1)
        es = sp.event_study(
            df, y="y", treat_time="g", time="time", unit="unit", window=(-4, 4)
        )
        return parallel_trends_robustness(es), es

    def test_shape_and_contents(self, rob):
        out, _ = rob
        assert isinstance(out, ParallelTrendsRobustnessResult)
        assert set(out.breakdown) == {"SD", "RM"}
        assert set(out.ci_grid["family"]) == {"SD", "RM"}
        assert list(out.ci_grid.columns) == [
            "family",
            "M",
            "ci_lower",
            "ci_upper",
            "rejects_zero",
        ]
        assert not out.power_table.empty
        assert 0.0 <= float(out.pretrend_power["power"]) <= 1.0

    def test_result_protocol_surface(self, rob):
        out, _ = rob
        assert isinstance(out.summary(), str)
        assert "Parallel-Trends Robustness" in out.summary()
        assert "Verdict" in out.summary()
        assert isinstance(out._repr_html_(), str)
        latex = out.to_latex()
        assert "\\begin{table}" in latex and "\\bottomrule" in latex
        assert "Breakdown" in latex
        assert isinstance(out.to_dict(), dict)
        # Citations resolve to keys that exist in paper.bib.
        assert "rambachan2023more" in out.cite()
        assert "roth2022pretest" in out.cite()

    def test_plot(self, rob):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        out, _ = rob
        ax = out.plot()
        assert ax is not None
        assert ax.get_xlabel()

    def test_breakdown_is_consistent_with_its_own_ci_grid(self, rob):
        """Mbar* must agree with the grid the same object reports.

        This is the reason the pipeline inverts ``honest_did`` rather than
        calling ``did.honest_did.breakdown_m``: that helper accepts a
        ``method`` argument, validates it, and then ignores it, always
        returning the smoothness answer.
        """
        out, es = rob
        for fam in out.families:
            mstar = out.breakdown[fam]
            sub = out.ci_grid[out.ci_grid["family"] == fam]
            # Every grid point strictly below Mbar* must exclude zero.
            below = sub[sub["M"] < mstar - 1e-9]
            assert below["rejects_zero"].all()
            # Every grid point strictly above must include zero.
            above = sub[sub["M"] > mstar + 1e-9]
            assert not above["rejects_zero"].any()

    def test_sd_breakdown_matches_breakdown_m(self, rob):
        """Cross-check the smoothness family against the existing helper."""
        out, es = rob
        assert out.breakdown["SD"] == pytest.approx(
            float(sp.breakdown_m(es, e=0, method="smoothness")), rel=1e-6
        )

    def test_sd_and_rm_breakdowns_differ(self, rob):
        """The two families bound the violation differently."""
        out, _ = rob
        assert out.breakdown["SD"] != out.breakdown["RM"]

    def test_verdict_is_one_line_and_quantified(self, rob):
        out, _ = rob
        assert "\n" not in out.verdict
        assert "survives up to Mbar" in out.verdict or "NOT robust" in out.verdict

    def test_verdict_inf_breakdown_is_reported_as_robust(self):
        """⚠️ correctness fix (2026-07): ``inf`` breakdown = maximal robustness.

        ``_breakdown_for_family`` returns ``inf`` when the honest CI still
        excludes zero at the upper search bound.  The historical guard
        ``not np.isfinite(weakest)`` routed that (and NaN) into the
        "NOT robust ... at M = 0" message — the exact opposite verdict.
        """
        from statspai.did.robustness_pipeline import _build_verdict

        inf = float("inf")
        v = _build_verdict(
            {"SD": inf, "RM": inf}, pd.DataFrame(), {"power": 0.9}, att=5.0, e=0
        )
        assert "NOT robust" not in v
        assert "robust over the entire searched range" in v

    def test_verdict_nan_families_excluded_not_inverted(self):
        """NaN families are excluded (with a note), never treated as binding."""
        from statspai.did.robustness_pipeline import _build_verdict

        v = _build_verdict(
            {"SD": float("nan"), "RM": 0.7}, pd.DataFrame(), {}, att=1.0, e=0
        )
        assert "Mbar = 0.7" in v
        assert "NOT robust" not in v
        assert "failed to evaluate" in v and "SD" in v
        # All-NaN: no verdict, not a fabricated one.
        v_all = _build_verdict({"SD": float("nan")}, pd.DataFrame(), {}, att=1.0, e=0)
        assert "no verdict available" in v_all

    def test_verdict_mixed_inf_and_finite_binds_on_finite(self):
        from statspai.did.robustness_pipeline import _build_verdict

        v = _build_verdict(
            {"SD": float("inf"), "RM": 0.4}, pd.DataFrame(), {}, att=1.0, e=0
        )
        assert "Mbar = 0.4" in v and "(RM)" in v
        assert "NOT robust" not in v

    def test_verdict_reports_non_robustness(self):
        """Edge: a null effect must be reported as not robust at M = 0."""
        df = _panel(n_units=40, n_periods=8, treat_at=5, seed=4)
        # Strip the treatment effect entirely.
        df["y"] = df["y"] - 2.0 * ((df["g"] > 0) & (df["time"] >= 5))
        es = sp.event_study(
            df, y="y", treat_time="g", time="time", unit="unit", window=(-3, 3)
        )
        out = parallel_trends_robustness(es)
        assert all(v == 0.0 for v in out.breakdown.values())
        assert "NOT robust" in out.verdict

    def test_custom_m_grid_and_single_family(self, rob):
        _, es = rob
        out = parallel_trends_robustness(es, m_grid=[0.0, 0.25, 0.5], families=("RM",))
        assert out.families == ["RM"]
        assert set(out.breakdown) == {"RM"}
        assert list(out.ci_grid["M"]) == [0.0, 0.25, 0.5]

    @pytest.mark.parametrize("families", [("nope",), (), 42, (1, 2), "bogus"])
    def test_bad_families_fail_loudly(self, rob, families):
        _, es = rob
        with pytest.raises(MethodIncompatibility):
            parallel_trends_robustness(es, families=families)

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, "x", True])
    def test_bad_alpha_fails_loudly(self, rob, alpha):
        _, es = rob
        with pytest.raises(MethodIncompatibility):
            parallel_trends_robustness(es, alpha=alpha)

    def test_family_aliases_are_case_insensitive(self, rob):
        _, es = rob
        out = parallel_trends_robustness(es, families=("sd", "rm", "SD"))
        assert out.families == ["SD", "RM"]  # de-duplicated, order preserved

    def test_order_of_magnitude_sanity_anchor(self):
        """Loose sanity check against hand-worked reading notes.

        Reading notes for a long-horizon DiD record a baseline ATT of
        0.0380 with SE 0.0166 and a maximum pre-trend deviation of about
        0.02, which the note-taker read as an RM breakdown Mbar* of
        roughly 1.9. This is NOT a published figure and NOT a parity
        target -- it is used only to confirm we land in a plausible
        range rather than at 0 or 1e6.

        StatsPAI's native RM bound gives
        ``(|theta| - z*SE) / max|delta_pre|``
        = (0.0380 - 1.96*0.0166) / 0.02 = 0.27, i.e. the same order of
        magnitude as the note but not equal to it; the native analytic
        interval is more conservative than the note's arithmetic. The
        assertion below is deliberately wide enough to cover both.
        """
        from statspai.core.results import CausalResult

        es = pd.DataFrame(
            {
                "relative_time": [-3, -2, -1, 0],
                "att": [0.02, -0.01, 0.0, 0.0380],
                "estimate": [0.02, -0.01, 0.0, 0.0380],
                "se": [0.015, 0.014, 0.0, 0.0166],
                "ci_lower": [0.0, 0.0, 0.0, 0.0],
                "ci_upper": [0.0, 0.0, 0.0, 0.0],
                "pvalue": [1.0, 1.0, 1.0, 0.02],
            }
        )
        result = CausalResult(
            method="stub",
            estimand="ATT",
            estimate=0.0380,
            se=0.0166,
            pvalue=0.02,
            ci=(0.0055, 0.0705),
            alpha=0.05,
            n_obs=1000,
            detail=es,
            model_info={"event_study": es},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = parallel_trends_robustness(result, families=("RM",))
        mstar = out.breakdown["RM"]
        assert np.isfinite(mstar)
        # Order-of-magnitude band spanning both 0.27 and the note's 1.9.
        assert 0.05 < mstar < 20.0
        # And the closed form the native backend actually implements.
        expected = (0.0380 - 1.959963985 * 0.0166) / 0.02
        assert mstar == pytest.approx(expected, rel=1e-3)
