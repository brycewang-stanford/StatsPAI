"""Options and degenerate inputs of ``sp.hierarchy`` and ``sp.reconcile``.

A reconciled forecast is ``S G y_hat``. The tests check the properties
that define each method: coherence, idempotence on coherent forecasts,
the generalised least-squares formula for ``G`` written out with the
stated weight matrix, and the proportions of the top-down variants.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility

SPEC = [["all"], ["all", "region"], ["all", "region", "state"]]
LEAVES = [("N", "a"), ("N", "b"), ("S", "c"), ("S", "d"), ("S", "e")]


def _long(periods: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(periods):
        for region, state in LEAVES:
            rows.append(
                ("T", region, state, t, 10 + 3 * (region == "S") + rng.normal())
            )
    return pd.DataFrame(rows, columns=["all", "region", "state", "t", "v"])


def _setup(seed: int = 0):
    rng = np.random.default_rng(seed + 100)
    h = sp.hierarchy(_long(seed=seed), SPEC, time="t", value="v")
    base = h.Y.iloc[-2:] + rng.normal(size=(2, 8))
    res = pd.DataFrame(rng.normal(size=(40, 8)), columns=h.S.index)
    return h, base, res


LINEAR = ["ols", "wls_struct", "wls_var", "mint_shrink", "mint_cov"]


class TestHierarchy:
    def test_aggregates_and_summing_matrix(self):
        long = _long()
        h = sp.hierarchy(long, SPEC, time="t", value="v")
        assert h.bottom == ["T/N/a", "T/N/b", "T/S/c", "T/S/d", "T/S/e"]
        assert list(h.S.index) == ["T", "T/N", "T/S"] + h.bottom
        assert h.tags == {
            "all": ["T"],
            "all/region": ["T/N", "T/S"],
            "all/region/state": h.bottom,
        }
        np.testing.assert_array_equal(
            h.S.to_numpy()[:3],
            [[1, 1, 1, 1, 1], [1, 1, 0, 0, 0], [0, 0, 1, 1, 1]],
        )
        np.testing.assert_array_equal(h.S.to_numpy()[3:], np.eye(5))
        # Every series is its row of S times the bottom series.
        np.testing.assert_allclose(
            h.Y.to_numpy(), h.Y[h.bottom].to_numpy() @ h.S.to_numpy().T
        )
        np.testing.assert_allclose(h.Y["T"], long.groupby("t")["v"].sum())
        np.testing.assert_allclose(
            h.Y["T/S"], long[long.region == "S"].groupby("t")["v"].sum()
        )
        assert "8 series over 12 periods, 5 at the bottom" in repr(h)

    def test_rows_may_come_in_any_order_and_sep_names_the_series(self):
        long = _long()
        a = sp.hierarchy(long, SPEC, time="t", value="v")
        b = sp.hierarchy(
            long.sample(frac=1.0, random_state=1), SPEC, time="t", value="v", sep="|"
        )
        np.testing.assert_allclose(b.Y.to_numpy(), a.Y.to_numpy())
        assert b.bottom[0] == "T|N|a"

    def test_grouped_structure_crosses_two_keys(self):
        long = _long()
        long = long.assign(kind=np.where(long["state"].isin(["a", "c"]), "x", "y"))
        spec = [
            ["all"],
            ["all", "region"],
            ["all", "kind"],
            ["all", "region", "state", "kind"],
        ]
        g = sp.hierarchy(long, spec, time="t", value="v")
        assert g.S.shape == (10, 5)
        np.testing.assert_allclose(
            g.Y["T/x"], long[long.kind == "x"].groupby("t")["v"].sum()
        )
        np.testing.assert_allclose(g.Y["T/N"] + g.Y["T/S"], g.Y["T/x"] + g.Y["T/y"])

    @pytest.mark.parametrize(
        "mutate, spec, match",
        [
            (lambda d: d, [], "non-empty"),
            (lambda d: d, [["all"], []], "non-empty"),
            (lambda d: d, [["all"], ["zone"]], "not in data"),
            (lambda d: d, [["all", "region", "state"], ["all"]], "does not contain"),
            (
                lambda d: d.assign(v=d["v"].where(d.index != 2)),
                SPEC,
                "1 missing values",
            ),
            (lambda d: pd.concat([d, d.iloc[:1]]), SPEC, "1 duplicated"),
            (lambda d: d.iloc[1:], SPEC, "unbalanced"),
            (
                lambda d: d.assign(r2=d["region"]),
                [["region"], ["r2"], ["region", "r2", "state"]],
                "same series identifiers",
            ),
        ],
    )
    def test_refusals(self, mutate, spec, match):
        with pytest.raises(MethodIncompatibility, match=match):
            sp.hierarchy(mutate(_long()), spec, time="t", value="v")


class TestCoherence:
    @pytest.mark.parametrize("method", ["bottom_up", "top_down", "middle_out"] + LINEAR)
    def test_reconciled_forecasts_add_up_and_coherent_ones_are_left_alone(self, method):
        h, base, res = _setup()
        kw = dict(method=method, residuals=res, history=h.Y, middle="all/region")
        rec = sp.reconcile(base, h, **kw)
        S = h.S.to_numpy()
        f = rec.forecasts.to_numpy()
        # Coherent: every series is the sum of its bottom series.
        np.testing.assert_allclose(f, f[:, 3:] @ S.T, atol=1e-10)
        again = sp.reconcile(rec.forecasts, h, **kw)
        np.testing.assert_allclose(again.forecasts.to_numpy(), f, atol=1e-10)
        np.testing.assert_allclose(again.adjustment.to_numpy(), 0.0, atol=1e-10)
        # The map is linear with the reported G.
        np.testing.assert_allclose(
            f, base.to_numpy() @ (S @ rec.G.to_numpy()).T, atol=1e-10
        )
        np.testing.assert_allclose(
            rec.adjustment.to_numpy(), f - base.to_numpy(), atol=1e-12
        )
        assert rec.method == method

    @pytest.mark.parametrize("method", LINEAR)
    def test_projection_reproduces_any_coherent_vector(self, method):
        h, base, res = _setup()
        rec = sp.reconcile(base, h, method=method, residuals=res)
        # G S = I: unbiased base forecasts stay unbiased.
        np.testing.assert_allclose(
            rec.G.to_numpy() @ h.S.to_numpy(), np.eye(5), atol=1e-10
        )


class TestWeights:
    def _gls(self, S, W):
        Wi = np.linalg.inv(W)
        return np.linalg.solve(S.T @ Wi @ S, S.T @ Wi)

    def test_ols_is_the_orthogonal_projection(self):
        h, base, _ = _setup()
        S = h.S.to_numpy()
        rec = sp.reconcile(base, h, method="ols")
        proj = S @ np.linalg.solve(S.T @ S, S.T)
        np.testing.assert_allclose(
            rec.forecasts.to_numpy(), base.to_numpy() @ proj, atol=1e-10
        )
        # No other coherent vector is closer to the base forecasts.
        other = sp.reconcile(base, h, method="wls_struct").forecasts.to_numpy()
        d_ols = np.sum((rec.forecasts.to_numpy() - base.to_numpy()) ** 2)
        assert d_ols < np.sum((other - base.to_numpy()) ** 2)

    def test_structural_weights_count_the_bottom_series(self):
        h, base, _ = _setup()
        S = h.S.to_numpy()
        rec = sp.reconcile(base, h, method="structural")
        np.testing.assert_allclose(np.diag(rec.weights), [5, 2, 3, 1, 1, 1, 1, 1])
        np.testing.assert_allclose(
            rec.G.to_numpy(), self._gls(S, np.diag(S.sum(axis=1))), atol=1e-10
        )

    def test_variance_weights_are_uncentred_mean_squares(self):
        h, base, res = _setup()
        S = h.S.to_numpy()
        r = res.to_numpy() + 0.5  # a non-zero mean, to tell the conventions apart
        cov = r.T @ r / len(r)
        rec = sp.reconcile(base, h, method="wls_var", residuals=r)
        np.testing.assert_allclose(rec.weights, np.diag(np.diag(cov)))
        np.testing.assert_allclose(
            rec.G.to_numpy(), self._gls(S, np.diag(np.diag(cov))), atol=1e-10
        )
        full = sp.reconcile(base, h, method="mint_cov", residuals=r)
        np.testing.assert_allclose(full.weights, cov)
        np.testing.assert_allclose(full.G.to_numpy(), self._gls(S, cov), atol=1e-9)
        assert full.shrinkage is None

    def test_shrinkage_moves_the_covariance_towards_its_diagonal(self):
        h, base, res = _setup()
        # A common factor, so that there are correlations worth keeping.
        r = res.to_numpy() + 1.5 * res.to_numpy()[:, [0]]
        cov = r.T @ r / len(r)
        rec = sp.reconcile(base, h, residuals=r)  # default: mint_shrink
        lam = rec.shrinkage
        assert rec.method == "mint_shrink" and 0.0 < lam < 1.0
        np.testing.assert_allclose(
            rec.weights, lam * np.diag(np.diag(cov)) + (1 - lam) * cov
        )
        np.testing.assert_allclose(
            rec.G.to_numpy(), self._gls(h.S.to_numpy(), rec.weights), atol=1e-9
        )
        assert "shrinkage" in rec.summary()

    def test_incomplete_residual_rows_are_dropped(self):
        h, base, res = _setup()
        holed = res.copy()
        holed.iloc[3, 2] = np.nan
        a = sp.reconcile(base, h, method="mint_shrink", residuals=holed)
        b = sp.reconcile(base, h, method="mint_shrink", residuals=res.drop(index=3))
        np.testing.assert_allclose(a.forecasts.to_numpy(), b.forecasts.to_numpy())

    def test_rescaling_one_series_residuals_rescales_its_weight(self):
        h, base, res = _setup()
        a = sp.reconcile(base, h, method="wls_var", residuals=res)
        b = sp.reconcile(
            base, h, method="wls_var", residuals=res.assign(T=3.0 * res["T"])
        )
        assert b.weights[0, 0] == pytest.approx(9.0 * a.weights[0, 0])
        # A noisier total is trusted less: the reconciled total moves
        # further from its base forecast.
        move_a = abs(a.adjustment["T"].iloc[0])
        move_b = abs(b.adjustment["T"].iloc[0])
        assert move_b > move_a


class TestSingleLevelMethods:
    def test_bottom_up_sums_the_bottom_forecasts(self):
        h, base, _ = _setup()
        rec = sp.reconcile(base, h, method="bu")
        S = h.S.to_numpy()
        np.testing.assert_allclose(
            rec.forecasts.to_numpy(), base[h.bottom].to_numpy() @ S.T
        )
        np.testing.assert_allclose(rec.forecasts[h.bottom], base[h.bottom])

    def test_top_down_average_proportions(self):
        h, base, _ = _setup()
        rec = sp.reconcile(base, h, method="top_down", history=h.Y)
        shares = (h.Y[h.bottom].div(h.Y["T"], axis=0)).mean()
        np.testing.assert_allclose(
            rec.forecasts[h.bottom].to_numpy(),
            np.outer(base["T"].to_numpy(), shares.to_numpy()),
        )
        # The total is the base forecast of the total.
        np.testing.assert_allclose(rec.forecasts["T"], base["T"])

    def test_top_down_proportions_of_averages(self):
        h, base, _ = _setup()
        rec = sp.reconcile(base, h, method="td", history=h.Y, proportions="of_averages")
        shares = h.Y[h.bottom].mean() / h.Y["T"].mean()
        np.testing.assert_allclose(
            rec.forecasts[h.bottom].to_numpy(),
            np.outer(base["T"].to_numpy(), shares.to_numpy()),
        )

    def test_top_down_forecast_proportions_split_level_by_level(self):
        h, base, _ = _setup()
        rec = sp.reconcile(base, h, method="top_down", proportions="forecast")
        b = base.iloc[0]
        north = b["T"] * b["T/N"] / (b["T/N"] + b["T/S"])
        a = north * b["T/N/a"] / (b["T/N/a"] + b["T/N/b"])
        assert rec.forecasts["T/N"].iloc[0] == pytest.approx(north)
        assert rec.forecasts["T/N/a"].iloc[0] == pytest.approx(a)
        np.testing.assert_allclose(rec.forecasts["T"], base["T"])
        # Not a linear map of the base forecasts: no G, hence no sd.
        assert rec.G.isna().all().all()
        with pytest.raises(MethodIncompatibility, match="not a linear map"):
            sp.reconcile(
                base,
                h,
                method="top_down",
                proportions="forecast",
                sd=np.ones((2, 8)),
            )

    def test_middle_out_keeps_the_middle_level(self):
        h, base, _ = _setup()
        for prop in ("average", "of_averages", "forecast"):
            rec = sp.reconcile(
                base,
                h,
                method="middle_out",
                middle="all/region",
                history=h.Y,
                proportions=prop,
            )
            np.testing.assert_allclose(
                rec.forecasts[["T/N", "T/S"]], base[["T/N", "T/S"]]
            )
            np.testing.assert_allclose(rec.forecasts["T"], base["T/N"] + base["T/S"])
        by_list = sp.reconcile(
            base, h.S, method="mo", middle=["T/N", "T/S"], history=h.Y
        )
        by_name = sp.reconcile(
            base, h, method="middle_out", middle="all/region", history=h.Y
        )
        pd.testing.assert_frame_equal(by_list.forecasts, by_name.forecasts)
        shares = h.Y["T/S/c"].div(h.Y["T/S"]).mean()
        np.testing.assert_allclose(by_name.forecasts["T/S/c"], base["T/S"] * shares)

    def test_middle_out_refusals(self):
        h, base, _ = _setup()
        with pytest.raises(MethodIncompatibility, match="needs middle="):
            sp.reconcile(base, h, method="middle_out", history=h.Y)
        with pytest.raises(MethodIncompatibility, match="not a level"):
            sp.reconcile(base, h, method="middle_out", history=h.Y, middle="zone")
        with pytest.raises(MethodIncompatibility, match="not rows of S"):
            sp.reconcile(base, h, method="middle_out", history=h.Y, middle=["T/W"])
        with pytest.raises(MethodIncompatibility, match="disjoint groups"):
            sp.reconcile(base, h, method="middle_out", history=h.Y, middle=["T/N"])

    def test_top_down_needs_a_strict_hierarchy(self):
        long = _long()
        long = long.assign(kind=np.where(long["state"].isin(["a", "c"]), "x", "y"))
        spec = [
            ["all"],
            ["all", "region"],
            ["all", "kind"],
            ["all", "region", "state", "kind"],
        ]
        g = sp.hierarchy(long, spec, time="t", value="v")
        with pytest.raises(MethodIncompatibility, match="strict hierarchy"):
            sp.reconcile(g.Y.iloc[-1:], g, method="top_down", history=g.Y)
        rec = sp.reconcile(g.Y.iloc[-1:] + 1.0, g, method="ols")
        f = rec.forecasts.iloc[0]
        assert f["T/N"] + f["T/S"] == pytest.approx(f["T/x"] + f["T/y"])

    def test_history_that_does_not_add_up_is_refused(self):
        h, base, _ = _setup()
        bad = h.Y.assign(T=2.0 * h.Y["T"])
        with pytest.raises(MethodIncompatibility, match="history"):
            sp.reconcile(base, h, method="top_down", history=bad)
        with pytest.raises(MethodIncompatibility, match="does not add up"):
            sp.reconcile(
                base,
                h,
                method="middle_out",
                middle="all/region",
                history=h.Y.assign(**{"T/N": 1.01 * h.Y["T/N"]}),
            )
        # Rounding-size incoherence (below 1e-6 of the largest value) is
        # accepted, and top-down then returns the base total.
        nearly = h.Y.assign(T=h.Y["T"] * (1 + 1e-9))
        rec = sp.reconcile(base, h, method="top_down", history=nearly)
        np.testing.assert_allclose(rec.forecasts["T"], base["T"], rtol=1e-7)


class TestInputsAndIntervals:
    def test_array_and_frame_inputs_agree(self):
        h, base, res = _setup()
        a = sp.reconcile(base, h, method="wls_var", residuals=res)
        b = sp.reconcile(base.to_numpy(), h.S, method="wls", residuals=res.to_numpy())
        np.testing.assert_allclose(b.forecasts.to_numpy(), a.forecasts.to_numpy())
        # Columns are matched by name: order and extras do not matter.
        shuffled = base[base.columns[::-1]].assign(extra=1.0)
        c = sp.reconcile(shuffled, h, method="wls_var", residuals=res)
        pd.testing.assert_frame_equal(c.forecasts, a.forecasts)
        np.testing.assert_allclose(c.adjustment.to_numpy(), a.adjustment.to_numpy())
        one = sp.reconcile(base.to_numpy()[0], h, method="ols")
        assert one.forecasts.shape == (1, 8)

    def test_standard_deviations_of_the_reconciled_forecasts(self):
        h, base, res = _setup()
        sd = np.abs(np.random.default_rng(3).normal(1.0, 0.2, size=(2, 8)))
        S = h.S.to_numpy()
        # Uncorrelated base errors for the methods that estimate no
        # correlations.
        rec = sp.reconcile(base, h, method="ols", sd=sd)
        SG = S @ rec.G.to_numpy()
        for r in range(2):
            cov = SG @ np.diag(sd[r] ** 2) @ SG.T
            np.testing.assert_allclose(rec.sd.iloc[r], np.sqrt(np.diag(cov)))
        # The correlations of the estimated covariance for MinT.
        mint = sp.reconcile(base, h, method="mint_cov", residuals=res, sd=sd)
        d = np.sqrt(np.diag(mint.weights))
        corr = mint.weights / np.outer(d, d)
        SG = S @ mint.G.to_numpy()
        cov = SG @ (corr * np.outer(sd[1], sd[1])) @ SG.T
        np.testing.assert_allclose(mint.sd.iloc[1], np.sqrt(np.diag(cov)))
        lo, hi = mint.intervals(90)
        z = stats.norm.ppf(0.95)
        np.testing.assert_allclose(hi - mint.forecasts, z * mint.sd)
        np.testing.assert_allclose(mint.forecasts - lo, z * mint.sd)
        pd.testing.assert_frame_equal(mint.intervals(0.9)[0], lo)

    def test_sd_refusals(self):
        h, base, _ = _setup()
        with pytest.raises(MethodIncompatibility, match="missing or negative"):
            sp.reconcile(base, h, method="ols", sd=-np.ones((2, 8)))
        with pytest.raises(MethodIncompatibility, match="sd has shape"):
            sp.reconcile(base, h, method="ols", sd=np.ones((3, 8)))
        rec = sp.reconcile(base, h, method="ols")
        with pytest.raises(MethodIncompatibility, match="no forecast standard"):
            rec.intervals()
        with_sd = sp.reconcile(base, h, method="ols", sd=np.ones((2, 8)))
        with pytest.raises(MethodIncompatibility, match="not a coverage"):
            with_sd.intervals(120)

    @pytest.mark.parametrize(
        "kwargs, exc, match",
        [
            (dict(method="erm"), MethodIncompatibility, "method='erm'"),
            (dict(method="wls_var"), MethodIncompatibility, "needs residuals="),
            (dict(method="top_down"), MethodIncompatibility, "needs history="),
            (
                dict(method="top_down", proportions="x", history=1),
                MethodIncompatibility,
                "proportions='x'",
            ),
        ],
    )
    def test_method_refusals(self, kwargs, exc, match):
        h, base, _ = _setup()
        with pytest.raises(exc, match=match):
            sp.reconcile(base, h, **kwargs)

    def test_data_refusals(self):
        h, base, res = _setup()
        with pytest.raises(MethodIncompatibility, match="missing values"):
            sp.reconcile(base.assign(T=np.nan), h, method="ols")
        with pytest.raises(MethodIncompatibility, match="base lacks the series"):
            sp.reconcile(base.drop(columns="T"), h, method="ols")
        with pytest.raises(MethodIncompatibility, match="7 columns, S has 8"):
            sp.reconcile(np.ones((1, 7)), h, method="ols")
        with pytest.raises(DataInsufficient, match="8 series from 6 periods"):
            sp.reconcile(base, h, method="mint_cov", residuals=res.iloc[:6])
        with pytest.raises(DataInsufficient, match="2 complete residual periods"):
            sp.reconcile(base, h, method="wls_var", residuals=res.iloc[:2])
        with pytest.raises(DataInsufficient, match="zero"):
            sp.reconcile(base, h, method="wls_var", residuals=res.assign(T=0.0))
        with pytest.raises(DataInsufficient, match="identically zero"):
            sp.reconcile(base, h, method="mint_shrink", residuals=res.assign(T=0.0))

    def test_summing_matrix_refusals(self):
        with pytest.raises(MethodIncompatibility, match="more rows than columns"):
            sp.reconcile(np.ones(3), pd.DataFrame(np.eye(3)), method="ols")
        with pytest.raises(MethodIncompatibility, match="S must be a DataFrame"):
            sp.reconcile(np.ones(3), np.eye(3), method="ols")
        rank_deficient = pd.DataFrame(
            [[1.0, 1.0], [1.0, 1.0], [2.0, 2.0]], index=list("abc"), columns=list("xy")
        )
        with pytest.raises(MethodIncompatibility, match="full column rank"):
            sp.reconcile(np.ones(3), rank_deficient, method="ols")

    def test_bottom_rows_found_without_matching_labels(self):
        # Row labels of S need not repeat the column labels: a bottom
        # series is then the row that loads on it alone.
        S = pd.DataFrame(
            [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
            index=["total", "first", "second"],
            columns=["x", "y"],
        )
        rec = sp.reconcile(np.array([[10.0, 3.0, 4.0]]), S, method="bottom_up")
        assert rec.forecasts.iloc[0].tolist() == [7.0, 3.0, 4.0]
        no_leaf = pd.DataFrame(
            [[1.0, 1.0], [1.0, 0.0], [1.0, 2.0]],
            index=["total", "first", "odd"],
            columns=["x", "y"],
        )
        with pytest.raises(MethodIncompatibility, match="no row for the bottom"):
            sp.reconcile(np.array([[10.0, 3.0, 4.0]]), no_leaf, method="bottom_up")
