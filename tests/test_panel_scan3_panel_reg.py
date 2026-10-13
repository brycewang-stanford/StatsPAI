"""Option and degenerate-input tests for ``sp.panel`` (scan round 3).

Every numerical assertion compares with an independent computation written
here (least squares on explicit dummies, a cluster sandwich with a stated
small-sample factor) or with an identity.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility, StatsPAIError

# Both sides are a few float64 matrix products on well-conditioned designs
# with at most ~40 columns; 1e-8 leaves room for the within sweep against a
# dummy-variable least squares and for a different BLAS summation order.
RTOL = 1e-8


def _panel(seed: int = 0, n_units: int = 30, n_periods: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(n_units), n_periods)
    t = np.tile(np.arange(n_periods), n_units)
    a = rng.normal(size=n_units)[ids]
    n = n_units * n_periods
    df = pd.DataFrame(
        {
            "id": ids,
            "t": t,
            "x1": rng.normal(size=n) + a,
            "x2": rng.normal(size=n),
            "w": rng.uniform(0.5, 3.0, n),
            "g": ids % 7,  # units nested in g
            "z": rng.normal(size=n_units)[ids],  # constant within unit
        }
    )
    df["y"] = 1 + 2 * df.x1 - df.x2 + a + rng.normal(size=n) * np.sqrt(1 + df.x1**2)
    return df


def _unbalanced(seed: int = 0) -> pd.DataFrame:
    df = _panel(seed)
    return df.drop(index=[0, 1, 2, 3, 4, 17, 40, 41, 100]).reset_index(drop=True)


def _fit(df, method="fe", formula="y ~ x1 + x2", **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.panel(df, formula, entity="id", time="t", method=method, **kw)


def _dummy_fit(df: pd.DataFrame, effects, weights=None):
    """Slopes of x1, x2 from least squares on explicit dummies, plus the
    Frisch-Waugh pieces the sandwich formulas need."""
    D = np.column_stack(
        [pd.get_dummies(df[c], dtype=float).to_numpy() for c in effects]
    )
    w = np.ones(len(df)) if weights is None else df[weights].to_numpy()
    s = np.sqrt(w)
    Ds = D * s[:, None]

    def sweep(v):
        vs = v * (s if v.ndim == 1 else s[:, None])
        return vs - Ds @ np.linalg.lstsq(Ds, vs, rcond=None)[0]

    X = sweep(df[["x1", "x2"]].to_numpy())
    y = sweep(df["y"].to_numpy())
    bread = np.linalg.inv(X.T @ X)
    beta = bread @ X.T @ y
    return {
        "beta": beta,
        "resid": y - X @ beta,
        "X": X,
        "bread": bread,
        "rank_d": int(np.linalg.matrix_rank(D)),
        "n": len(df),
    }


def _cluster_cov(piece, groups):
    scores = (
        pd.DataFrame(piece["X"] * piece["resid"][:, None])
        .groupby(np.asarray(groups))
        .sum()
        .to_numpy()
    )
    return piece["bread"] @ (scores.T @ scores) @ piece["bread"]


class TestWithinEstimatorAgainstDummies:
    @pytest.mark.parametrize(
        "method,effects", [("fe", ["id"]), ("twoway", ["id", "t"])]
    )
    @pytest.mark.parametrize("weights", [None, "w"])
    @pytest.mark.parametrize("balanced", [True, False])
    def test_coefficients_and_classical_se(self, method, effects, weights, balanced):
        df = _panel() if balanced else _unbalanced()
        piece = _dummy_fit(df, effects, weights)
        kw = {} if weights is None else {"weights": weights}
        res = _fit(df, method, **kw)
        np.testing.assert_allclose(res.params.to_numpy(), piece["beta"], rtol=RTOL)
        # classical: s^2 (X'MX)^-1 with s^2 = rss / (N - K - rank of dummies)
        s2 = piece["resid"] @ piece["resid"] / (piece["n"] - 2 - piece["rank_d"])
        np.testing.assert_allclose(
            res.std_errors.to_numpy(), np.sqrt(np.diag(piece["bread"]) * s2), rtol=RTOL
        )

    @pytest.mark.parametrize("weights", [None, "w"])
    def test_stata_cluster_factor_fe(self, weights):
        """``ssc='stata'`` is ``xtreg, fe vce(cluster id)``: the sandwich on
        the within design times G/(G-1) * (N-1)/(N-K), K the slopes and
        the constant xtreg reports (the unit effects, nested in the cluster,
        are not charged)."""
        df = _unbalanced()
        piece = _dummy_fit(df, ["id"], weights)
        kw = {} if weights is None else {"weights": weights}
        G, n = df.id.nunique(), len(df)
        factor = G / (G - 1) * (n - 1) / (n - 3)
        res = _fit(df, "fe", cluster="entity", ssc="stata", **kw)
        np.testing.assert_allclose(
            res.std_errors.to_numpy() ** 2,
            np.diag(_cluster_cov(piece, df.id)) * factor,
            rtol=RTOL,
        )

    def test_named_cluster_column_is_the_one_used(self):
        """``cluster='g'`` clusters on g (7 groups), not on the unit: the
        sandwich is rebuilt here on g with Stata's factor."""
        df = _unbalanced()
        piece = _dummy_fit(df, ["id"])
        n = len(df)
        factor = 7 / 6 * (n - 1) / (n - 3)
        res = _fit(df, "fe", cluster="g", ssc="stata")
        np.testing.assert_allclose(
            res.std_errors.to_numpy() ** 2,
            np.diag(_cluster_cov(piece, df.g)) * factor,
            rtol=RTOL,
        )

    def test_twoway_stata_cluster_counts_period_dummies(self):
        """``xtreg y x i.t, fe vce(cluster id)``: K counts the slopes, the
        T - 1 period dummies and the constant."""
        df = _unbalanced()
        piece = _dummy_fit(df, ["id", "t"])
        G, n, T = df.id.nunique(), len(df), df.t.nunique()
        factor = G / (G - 1) * (n - 1) / (n - (2 + (T - 1) + 1))
        res = _fit(df, "twoway", cluster="entity", ssc="stata")
        np.testing.assert_allclose(
            res.std_errors.to_numpy() ** 2,
            np.diag(_cluster_cov(piece, df.id)) * factor,
            rtol=RTOL,
        )

    def test_twoway_balanced_is_double_demeaning(self):
        df = _panel()
        cols = ["y", "x1", "x2"]
        dd = (
            df[cols]
            - df.groupby("id")[cols].transform("mean")
            - df.groupby("t")[cols].transform("mean")
            + df[cols].mean()
        )
        beta = np.linalg.lstsq(
            dd[["x1", "x2"]].to_numpy(), dd["y"].to_numpy(), rcond=None
        )[0]
        np.testing.assert_allclose(
            _fit(df, "twoway").params.to_numpy(), beta, rtol=RTOL
        )


class TestIdentities:
    def test_first_difference_equals_fe_with_two_periods(self):
        df = _panel(n_periods=2)
        fd, fe = _fit(df, "fd"), _fit(df, "fe")
        np.testing.assert_allclose(
            fd.params.to_numpy(), fe.params.to_numpy(), rtol=RTOL
        )
        # with T = 2 the two residual vectors differ by a factor sqrt(2) and
        # the degrees of freedom coincide (N - K), so the classical SEs agree
        np.testing.assert_allclose(
            fd.std_errors.to_numpy(), fe.std_errors.to_numpy(), rtol=RTOL
        )

    def test_pooled_weight_two_equals_duplicated_rows(self):
        df = _panel()
        df = df.assign(w2=np.where(df.id < 10, 2.0, 1.0))
        weighted = _fit(df, "pooled", weights="w2")
        stacked = _fit(pd.concat([df, df[df.id < 10]], ignore_index=True), "pooled")
        np.testing.assert_allclose(
            weighted.params.to_numpy(), stacked.params.to_numpy(), rtol=RTOL
        )

    def test_singleton_units_do_not_move_fe_coefficients(self):
        df = _panel()
        extra = df[df.id == 0].iloc[[0]].assign(id=999, y=50.0, x1=-7.0)
        with_singleton = _fit(pd.concat([df, extra], ignore_index=True), "fe")
        np.testing.assert_allclose(
            with_singleton.params.to_numpy(),
            _fit(df, "fe").params.to_numpy(),
            rtol=RTOL,
        )

    def test_row_order_is_irrelevant(self):
        df = _unbalanced()
        shuffled = df.sample(frac=1.0, random_state=3)
        for method in ("fe", "re", "be", "twoway", "mundlak"):
            a, b = _fit(df, method), _fit(shuffled, method)
            np.testing.assert_allclose(
                a.params.to_numpy(), b.params.to_numpy(), rtol=1e-7
            )
            np.testing.assert_allclose(
                a.std_errors.to_numpy(), b.std_errors.to_numpy(), rtol=1e-7
            )

    def test_time_invariant_regressor_is_omitted_and_named(self):
        df = _panel()
        with pytest.warns(UserWarning, match="omitted"):
            res = sp.panel(df, "y ~ x1 + z + x2", entity="id", time="t", method="fe")
        assert res.model_info["omitted"] == ["z"]
        np.testing.assert_allclose(
            res.params.to_numpy(), _fit(df, "fe").params.to_numpy(), rtol=RTOL
        )


def _fd_by_hand(df: pd.DataFrame) -> np.ndarray:
    """OLS of D.y on D.x without a constant, differences taken between
    consecutive periods of the same unit (Stata ``regress D.y D.x, nocons``
    on an ``xtset`` panel)."""
    d = df.sort_values(["id", "t"])
    g = d.groupby("id")
    diff = g[["y", "x1", "x2"]].diff()
    ok = (g["t"].diff() == 1).to_numpy()
    return np.linalg.lstsq(
        diff.loc[ok, ["x1", "x2"]].to_numpy(), diff.loc[ok, "y"].to_numpy(), rcond=None
    )[0]


class TestFirstDifferences:
    def test_balanced_sorted_panel_matches_differences_by_hand(self):
        df = _panel()
        np.testing.assert_allclose(
            _fit(df, "fd").params.to_numpy(), _fd_by_hand(df), rtol=RTOL
        )

    def test_row_order_is_irrelevant(self):
        df = _panel()
        shuffled = df.sample(frac=1.0, random_state=3)
        np.testing.assert_allclose(
            _fit(shuffled, "fd").params.to_numpy(), _fd_by_hand(df), rtol=RTOL
        )

    def test_first_unit_missing_its_first_period(self):
        df = _panel().drop(index=[0])
        np.testing.assert_allclose(
            _fit(df, "fd").params.to_numpy(), _fd_by_hand(df), rtol=RTOL
        )

    def test_period_sorted_rows_give_the_consecutive_differences(self):
        """Sorted by (period, unit) the time axis is in order and the
        estimate is the by-hand one, gaps included: the reference the two
        row-order tests above compare with."""
        df = _unbalanced().sort_values(["t", "id"])
        np.testing.assert_allclose(
            _fit(df, "fd").params.to_numpy(), _fd_by_hand(df), rtol=RTOL
        )


class TestRefusals:
    def test_missing_cluster_value_is_refused(self):
        df = _unbalanced()
        gm = df.g.astype(float).where(df.index > 4)
        with pytest.raises(MethodIncompatibility, match="missing values"):
            sp.panel(
                df.assign(gm=gm), "y ~ x1 + x2", entity="id", time="t", cluster="gm"
            )

    @pytest.mark.parametrize("kw", [{"vce": "robust"}, {"vce": "nonsense"}])
    def test_unknown_vce_is_refused(self, kw):
        with pytest.raises(MethodIncompatibility, match="vce="):
            sp.panel(_panel(), "y ~ x1 + x2", entity="id", time="t", **kw)

    def test_unknown_robust_is_refused(self):
        with pytest.raises(MethodIncompatibility, match="robust="):
            sp.panel(_panel(), "y ~ x1 + x2", entity="id", time="t", robust="hc1")

    @pytest.mark.parametrize("method", ["re", "be", "fd", "mundlak", "mle"])
    def test_weights_refused_where_no_reference_exists(self, method):
        with pytest.raises(MethodIncompatibility, match="weights"):
            sp.panel(
                _panel(),
                "y ~ x1 + x2",
                entity="id",
                time="t",
                method=method,
                weights="w",
            )

    @pytest.mark.parametrize("bad", [0.0, -1.0, np.nan])
    def test_bad_weight_is_refused(self, bad):
        df = _panel()
        w = df["w"].to_numpy().copy()
        w[3] = bad
        with pytest.raises(MethodIncompatibility, match="strictly positive"):
            sp.panel(df.assign(w=w), "y ~ x1 + x2", entity="id", time="t", weights="w")

    def test_fd_cannot_cluster_on_time_through_default_path(self):
        with pytest.raises(MethodIncompatibility, match="constant within each unit"):
            sp.panel(
                _panel(),
                "y ~ x1 + x2",
                entity="id",
                time="t",
                method="fd",
                cluster="time",
            )

    def test_ssc_refused_with_extended_menu(self):
        with pytest.raises(MethodIncompatibility, match="ssc="):
            sp.panel(
                _panel(),
                "y ~ x1 + x2",
                entity="id",
                time="t",
                cluster="g",
                vce="CR2",
                ssc="stata",
            )

    def test_extended_menu_drops_rows_with_missing_cluster(self):
        """The CR2 path uses listwise deletion on the cluster column: the fit
        equals the fit on the rows with a cluster id."""
        df = _unbalanced()
        gm = df.g.astype(float).where(df.index > 4)
        d = df.assign(gm=gm)
        full = _fit(d, "fe", cluster="gm", vce="CR2")
        pre = _fit(d[d.gm.notna()], "fe", cluster="gm", vce="CR2")
        np.testing.assert_allclose(
            full.params.to_numpy(), pre.params.to_numpy(), rtol=RTOL
        )
        np.testing.assert_allclose(
            full.std_errors.to_numpy(), pre.std_errors.to_numpy(), rtol=RTOL
        )


class TestDegenerateInputsDefects:
    @pytest.mark.parametrize("method", ["fe", "re", "pooled", "twoway"])
    def test_ssc_with_missing_cluster_value_is_refused_cleanly(self, method):
        df = _unbalanced()
        gm = df.g.astype(float).where(df.index > 4)
        with pytest.raises(StatsPAIError, match="missing"):
            sp.panel(
                df.assign(gm=gm),
                "y ~ x1 + x2",
                entity="id",
                time="t",
                method=method,
                cluster="gm",
                ssc="stata",
            )

    @pytest.mark.parametrize("method", ["fe", "re", "pooled", "fd", "be", "twoway"])
    def test_missing_entity_id_is_refused_or_dropped(self, method):
        df = _unbalanced().astype({"id": float})
        ids = df["id"].to_numpy().copy()
        ids[[5, 6]] = np.nan
        d = df.assign(id=ids)
        try:
            res = _fit(d, method)
        except StatsPAIError:
            return
        pre = _fit(d[d.id.notna()], method)
        np.testing.assert_allclose(
            res.params.to_numpy(), pre.params.to_numpy(), rtol=RTOL
        )

    @pytest.mark.parametrize("method", ["fe", "be", "twoway"])
    def test_single_unit_is_refused_with_a_typed_error(self, method):
        df = _panel()
        with pytest.raises(StatsPAIError):
            _fit(df[df.id == 0], method)

    def test_mle_with_one_period_is_refused(self):
        df = _panel()
        with pytest.raises(StatsPAIError):
            _fit(df[df.t == 0], "mle")


class TestFirstDifferenceConventions:
    def test_gap_inside_a_unit_warns_and_uses_consecutive_periods(self):
        """Unit 16 of the unbalanced panel misses t = 4 between t = 3 and
        t = 5. Stata's ``D.`` forms no difference there (140 differences);
        R ``plm(model='fd')`` differences the adjacent rows (141) and gets
        other coefficients, which is what the warning says."""
        df = _unbalanced()
        with pytest.warns(UserWarning, match="gap"):
            res = sp.panel(df, "y ~ x1 + x2", entity="id", time="t", method="fd")
        np.testing.assert_allclose(res.params.to_numpy(), _fd_by_hand(df), rtol=RTOL)
        assert res.data_info["nobs"] == 140

    def test_no_gap_no_warning(self):
        df = _panel().drop(index=[0, 5])  # a late start and an early end
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sp.panel(df, "y ~ x1 + x2", entity="id", time="t", method="fd")

    @pytest.mark.parametrize("cluster", [None, "entity", "t"])
    def test_ssc_stata_does_not_depend_on_row_order(self, cluster):
        df = _unbalanced()
        kw = dict(method="fd", ssc="stata", cluster=cluster)
        a = _fit(df, **kw)
        b = _fit(df.sample(frac=1.0, random_state=5), **kw)
        np.testing.assert_allclose(b.params.to_numpy(), a.params.to_numpy(), rtol=RTOL)
        np.testing.assert_allclose(
            b.std_errors.to_numpy(), a.std_errors.to_numpy(), rtol=1e-7
        )

    def test_missing_time_id_is_refused(self):
        df = _unbalanced().astype({"t": float})
        t = df["t"].to_numpy().copy()
        t[7] = np.nan
        with pytest.raises(MethodIncompatibility, match="missing"):
            _fit(df.assign(t=t), "fd")
