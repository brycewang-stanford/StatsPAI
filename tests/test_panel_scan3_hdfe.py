"""Option and degenerate-input tests for ``sp.hdfe_ols`` (``panel/feols.py``,
``panel/hdfe.py``, ``panel/_hdfe_iv.py``), scan round 3.

The reference in every numerical test is least squares / 2SLS on explicit
dummies written here, with the small-sample factor of the convention the
docstring names (reghdfe / ivreg2).
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility, StatsPAIError

# The absorber iterates to 1e-8 on the largest change of a sweep, relative
# to the variable's scale; coefficients and variances inherit roughly that,
# so comparisons with an exact dummy-variable projection use 1e-6.
RTOL = 1e-6
N = 400
FORMULA = "y ~ x1 + x2 | f1 + f2"
IV_FORMULA = "yiv ~ x1 | f1 + f2 | d ~ z1 + z2"


def _data(seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "f1": rng.integers(0, 25, N),
            "f2": rng.integers(0, 8, N),
            "x1": rng.normal(size=N),
            "x2": rng.normal(size=N),
            "w": rng.uniform(0.5, 3.0, N),
            "z1": rng.normal(size=N),
            "z2": rng.normal(size=N),
            "cl": rng.integers(0, 12, N),
            "cl2": rng.integers(0, 10, N),
        }
    )
    u = rng.normal(size=N)
    df["d"] = 0.5 * df.z1 + 0.4 * df.z2 + rng.normal(size=N) + 0.1 * df.f1 + 0.5 * u
    df["y"] = (
        1
        + 2 * df.x1
        - df.x2
        + 0.3 * df.f1
        + 0.2 * df.f2
        + rng.normal(size=N) * np.sqrt(1 + df.x1**2)
    )
    df["yiv"] = 1 + 1.5 * df.d + df.x1 + 0.3 * df.f1 + u * np.sqrt(1 + df.z1**2)
    return df


def _fit(formula, df, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.hdfe_ols(formula, df, **kw)


def _dummies(df):
    return np.column_stack(
        [pd.get_dummies(df.f1, dtype=float), pd.get_dummies(df.f2, dtype=float)]
    )


def _sweeper(df, weights=None):
    D = _dummies(df)
    s = np.ones(len(df)) if weights is None else np.sqrt(df[weights].to_numpy())
    Ds = D * s[:, None]

    def sweep(v):
        vs = v * (s if v.ndim == 1 else s[:, None])
        return vs - Ds @ np.linalg.lstsq(Ds, vs, rcond=None)[0]

    return sweep, int(np.linalg.matrix_rank(D))


def _meat(scores, groups):
    sums = pd.DataFrame(scores).groupby(np.asarray(groups)).sum().to_numpy()
    return sums.T @ sums


def _ols_pieces(df, weights=None):
    sweep, rank_d = _sweeper(df, weights)
    X = sweep(df[["x1", "x2"]].to_numpy())
    y = sweep(df["y"].to_numpy())
    bread = np.linalg.inv(X.T @ X)
    beta = bread @ X.T @ y
    resid = y - X @ beta
    return beta, bread, X * resid[:, None], resid, rank_d


class TestOLSAgainstDummies:
    @pytest.mark.parametrize("weights", [None, "w"])
    def test_coefficients_and_classical_se(self, weights):
        df = _data()
        beta, bread, _, resid, rank_d = _ols_pieces(df, weights)
        res = _fit(FORMULA, df, weights=weights)
        np.testing.assert_allclose(res.coef.to_numpy(), beta, rtol=RTOL)
        s2 = resid @ resid / (N - 2 - rank_d)
        np.testing.assert_allclose(
            res.se.to_numpy(), np.sqrt(np.diag(bread) * s2), rtol=RTOL
        )
        assert res.df_resid == N - 2 - rank_d

    @pytest.mark.parametrize("weights", [None, "w"])
    def test_one_way_cluster_reghdfe_factor(self, weights):
        """Neither effect is nested in ``cl``: reghdfe charges every absorbed
        level, G/(G-1) * (N-1)/(N-K-df_a)."""
        df = _data()
        _, bread, scores, _, rank_d = _ols_pieces(df, weights)
        G = df.cl.nunique()
        factor = G / (G - 1) * (N - 1) / (N - 2 - rank_d)
        res = _fit(FORMULA, df, cluster="cl", weights=weights)
        np.testing.assert_allclose(
            np.diag(res.cov_params()),
            np.diag(bread @ _meat(scores, df.cl) @ bread) * factor,
            rtol=RTOL,
        )
        assert res.df_inference == G - 1

    def test_two_way_cluster_is_inclusion_exclusion_per_term(self):
        """Two-way clustering: each of the three sandwiches carries its own
        G/(G-1) (first way, second way, their intersection, subtracted)."""
        df = _data()
        _, bread, scores, _, rank_d = _ols_pieces(df)
        inter = df.cl.astype(str) + "_" + df.cl2.astype(str)
        g1, g2, g12 = df.cl.nunique(), df.cl2.nunique(), inter.nunique()
        meat = (
            g1 / (g1 - 1) * _meat(scores, df.cl)
            + g2 / (g2 - 1) * _meat(scores, df.cl2)
            - g12 / (g12 - 1) * _meat(scores, inter)
        )
        cov = bread @ meat @ bread * (N - 1) / (N - 2 - rank_d)
        res = _fit(FORMULA, df, cluster=["cl", "cl2"])
        np.testing.assert_allclose(np.diag(res.cov_params()), np.diag(cov), rtol=RTOL)

    @pytest.mark.parametrize("vce,scaled", [("robust", True), ("hc0", False)])
    def test_heteroskedasticity_robust(self, vce, scaled):
        df = _data()
        _, bread, scores, _, rank_d = _ols_pieces(df)
        cov = bread @ (scores.T @ scores) @ bread
        if scaled:  # reghdfe: N / (N - K - df_a)
            cov = cov * N / (N - 2 - rank_d)
        res = _fit(FORMULA, df, vce=vce)
        np.testing.assert_allclose(res.se.to_numpy(), np.sqrt(np.diag(cov)), rtol=RTOL)

    def test_redundant_fixed_effect_changes_nothing(self):
        df = _data()
        df = df.assign(f1_copy=df.f1 + 100)
        a = _fit(FORMULA, df)
        b = _fit("y ~ x1 + x2 | f1 + f2 + f1_copy", df)
        np.testing.assert_allclose(a.coef.to_numpy(), b.coef.to_numpy(), rtol=RTOL)

    def test_weight_two_equals_duplicated_rows(self):
        df = _data()
        df = df.assign(w2=np.where(df.cl < 4, 2.0, 1.0))
        stacked = pd.concat([df, df[df.cl < 4]], ignore_index=True)
        np.testing.assert_allclose(
            _fit(FORMULA, df, weights="w2").coef.to_numpy(),
            _fit(FORMULA, stacked).coef.to_numpy(),
            rtol=RTOL,
        )

    def test_singletons_are_counted_but_do_not_move_coefficients(self):
        df = _data()
        extra = df.iloc[[0, 1]].assign(f1=[900, 901], y=[40.0, -40.0])
        both = pd.concat([df, extra], ignore_index=True)
        dropped = _fit(FORMULA, both)
        assert dropped.n_singletons_dropped == 2
        assert dropped.n_obs == N
        np.testing.assert_allclose(
            dropped.coef.to_numpy(), _fit(FORMULA, df).coef.to_numpy(), rtol=RTOL
        )
        kept = _fit(FORMULA, both, drop_singletons=False)
        assert kept.n_obs == N + 2
        np.testing.assert_allclose(
            kept.coef.to_numpy(), dropped.coef.to_numpy(), rtol=RTOL
        )

    def test_absorbed_regressor_is_nan_and_named(self):
        df = _data()
        df = df.assign(xa=2.0 * df.f1)
        with pytest.warns(UserWarning, match="xa"):
            res = sp.hdfe_ols("y ~ x1 + x2 + xa | f1 + f2", df)
        assert np.isnan(res.coef["xa"]) and np.isnan(res.se["xa"])
        np.testing.assert_allclose(
            res.coef[["x1", "x2"]].to_numpy(),
            _fit(FORMULA, df).coef.to_numpy(),
            rtol=RTOL,
        )


class TestMissingValues:
    """A missing cluster id, fixed-effect label or weight drops the row
    (listwise deletion): the fit equals the fit on the complete rows."""

    @staticmethod
    def _same(a, b):
        assert a.n_obs == b.n_obs
        np.testing.assert_allclose(a.coef.to_numpy(), b.coef.to_numpy(), rtol=1e-10)
        np.testing.assert_allclose(a.se.to_numpy(), b.se.to_numpy(), rtol=1e-10)

    def _with_missing(self, column, as_string=False):
        df = _data()
        col = df[column].astype(float).where(df.index > 4)
        if as_string:
            col = col.map(lambda v: None if pd.isna(v) else f"L{int(v)}")
        d = df.assign(**{column: col})
        return d, d[d[column].notna()]

    @pytest.mark.parametrize(
        "kw",
        [
            {"cluster": "cl"},
            {"cluster": ["cl", "cl2"]},
            {"cluster": "cl", "vce": "CR2"},
        ],
    )
    def test_missing_cluster(self, kw):
        full, complete = self._with_missing("cl")
        self._same(_fit(FORMULA, full, **kw), _fit(FORMULA, complete, **kw))

    def test_missing_cluster_iv(self):
        full, complete = self._with_missing("cl")
        self._same(
            _fit(IV_FORMULA, full, cluster="cl"),
            _fit(IV_FORMULA, complete, cluster="cl"),
        )

    @pytest.mark.parametrize("as_string", [False, True])
    def test_missing_fixed_effect_label(self, as_string):
        full, complete = self._with_missing("f1", as_string)
        self._same(_fit(FORMULA, full), _fit(FORMULA, complete))

    def test_missing_weight(self):
        full, complete = self._with_missing("w")
        self._same(
            _fit(FORMULA, full, weights="w"), _fit(FORMULA, complete, weights="w")
        )


class TestIVAgainstDummies:
    @staticmethod
    def _pieces(df):
        sweep, rank_d = _sweeper(df)
        Z = sweep(df[["x1", "z1", "z2"]].to_numpy())
        X = sweep(df[["d", "x1"]].to_numpy())  # reported order: endogenous first
        y = sweep(df["yiv"].to_numpy())
        Xh = Z @ np.linalg.solve(Z.T @ Z, Z.T @ X)
        bread = np.linalg.inv(Xh.T @ X)
        beta = bread @ Xh.T @ y
        resid = y - X @ beta
        return beta, bread, Xh * resid[:, None], resid, rank_d

    def test_coefficients_and_three_variances(self):
        df = _data()
        beta, bread, scores, resid, rank_d = self._pieces(df)
        dof = N - 2 - rank_d
        res = _fit(IV_FORMULA, df)
        assert list(res.coef.index) == ["d", "x1"]
        np.testing.assert_allclose(res.coef.to_numpy(), beta, rtol=RTOL)
        np.testing.assert_allclose(
            res.se.to_numpy() ** 2, np.diag(bread) * (resid @ resid) / dof, rtol=RTOL
        )
        robust = bread @ (scores.T @ scores) @ bread.T * N / dof
        np.testing.assert_allclose(
            _fit(IV_FORMULA, df, vce="robust").se.to_numpy() ** 2,
            np.diag(robust),
            rtol=RTOL,
        )
        G = df.cl.nunique()
        clustered = bread @ _meat(scores, df.cl) @ bread.T * G / (G - 1) * (N - 1) / dof
        np.testing.assert_allclose(
            _fit(IV_FORMULA, df, cluster="cl").se.to_numpy() ** 2,
            np.diag(clustered),
            rtol=RTOL,
        )

    def test_just_identified_is_the_ratio_form(self):
        """One instrument, no exogenous regressor: beta = z'My / z'Md."""
        df = _data()
        sweep, _ = _sweeper(df)
        z, d, y = (sweep(df[c].to_numpy()) for c in ("z1", "d", "yiv"))
        res = _fit("yiv ~ 1 | f1 + f2 | d ~ z1", df)
        np.testing.assert_allclose(res.coef["d"], (z @ y) / (z @ d), rtol=RTOL)

    def test_first_stage_f_and_cragg_donald_coincide_when_iid(self):
        """One endogenous regressor, classical errors: the first-stage F of
        the excluded instruments, computed here from two residual sums of
        squares, is the Cragg-Donald F."""
        df = _data()
        sweep, rank_d = _sweeper(df)
        d = sweep(df["d"].to_numpy())
        x = sweep(df[["x1"]].to_numpy())
        xz = sweep(df[["x1", "z1", "z2"]].to_numpy())

        def rss(A):
            return float(np.sum((d - A @ np.linalg.lstsq(A, d, rcond=None)[0]) ** 2))

        f_hand = ((rss(x) - rss(xz)) / 2) / (rss(xz) / (N - 3 - rank_d))
        diag = _fit(IV_FORMULA, df).iv_diagnostics
        np.testing.assert_allclose(diag["cragg_donald_F"], f_hand, rtol=RTOL)
        np.testing.assert_allclose(diag["first_stage"]["d"]["F"], f_hand, rtol=RTOL)

    @pytest.mark.parametrize(
        "kw,match",
        [
            ({"weights": "w"}, "weights"),
            ({"cluster": ["cl", "cl2"]}, "one-way"),
            ({"cluster": "cl", "vce": "CR2"}, "cluster= or vce='robust'"),
            ({"vce": "hc0"}, "cluster= or vce='robust'"),
            ({"cluster": "cl", "wild": True}, "cluster= or vce='robust'"),
            ({"cluster": "cl", "absorb_dof": "areg"}, "areg"),
        ],
    )
    def test_unsupported_options_are_refused_not_dropped(self, kw, match):
        with pytest.raises(MethodIncompatibility, match=match):
            sp.hdfe_ols(IV_FORMULA, _data(), **kw)

    def test_underidentified_and_collinear_instruments_are_refused(self):
        df = _data()
        with pytest.raises(MethodIncompatibility, match="Underidentified"):
            sp.hdfe_ols("yiv ~ x1 | f1 | d + x2 ~ z1", df)
        with pytest.raises(MethodIncompatibility, match="collinear"):
            sp.hdfe_ols("yiv ~ x1 | f1 | d ~ z1 + zb", df.assign(zb=2.0 * df.z1))
        with pytest.raises(MethodIncompatibility, match="collinear"):
            sp.hdfe_ols("yiv ~ x1 | f1 | d ~ za", df.assign(za=1.0 * df.f1))


class TestOptionStrings:
    @pytest.mark.parametrize(
        "kw,exc",
        [
            ({"vce": "nonsense"}, ValueError),
            ({"cluster": "cl", "absorb_dof": "nonsense"}, MethodIncompatibility),
            ({"df_inference": "nonsense"}, MethodIncompatibility),
            ({"vce": "CR2"}, ValueError),
            ({"cluster": "cl", "wild": True, "wild_weight_type": "zzz"}, ValueError),
            ({"weights": np.ones(10)}, ValueError),
        ],
    )
    def test_invalid_values_are_refused(self, kw, exc):
        with pytest.raises(exc):
            sp.hdfe_ols(FORMULA, _data(), **kw)

    def test_df_inference_leaves_standard_errors_alone(self):
        df = _data()
        base = _fit(FORMULA, df, cluster="cl")
        for choice in ("resid", "normal", 5):
            other = _fit(FORMULA, df, cluster="cl", df_inference=choice)
            np.testing.assert_allclose(
                other.se.to_numpy(), base.se.to_numpy(), rtol=1e-12
            )

    def test_all_regressors_absorbed_is_refused(self):
        df = _data()
        with pytest.raises(MethodIncompatibility, match="collinear"):
            sp.hdfe_ols("y ~ xa | f1 + f2", df.assign(xa=2.0 * df.f1))


class TestDefects:
    @pytest.mark.parametrize("formula", [FORMULA, "y ~ x1 + x2"])
    def test_single_cluster_is_refused_or_reported_missing(self, formula):
        df = _data().assign(one=1)
        try:
            res = _fit(formula, df, cluster="one")
        except StatsPAIError:
            return
        assert np.all(np.isnan(res.se.to_numpy()))

    @pytest.mark.parametrize("vce", ["robust", "hc0"])
    def test_robust_and_cluster_together_are_refused(self, vce):
        with pytest.raises((StatsPAIError, ValueError)):
            sp.hdfe_ols(FORMULA, _data(), vce=vce, cluster="cl")

    def test_se_type_iid_overrides_cluster_or_is_refused(self):
        df = _data()
        try:
            res = _fit(FORMULA, df, cluster="cl", se_type="iid")
        except (StatsPAIError, ValueError):
            return
        np.testing.assert_allclose(
            res.se.to_numpy(), _fit(FORMULA, df).se.to_numpy(), rtol=1e-10
        )

    def test_unknown_se_type_is_refused(self):
        with pytest.raises((StatsPAIError, ValueError)):
            sp.hdfe_ols(FORMULA, _data(), se_type="nonsense")

    @pytest.mark.parametrize("kw", [{"wild": True}, {"vce": "wild"}])
    def test_wild_without_cluster_is_refused(self, kw):
        with pytest.raises((StatsPAIError, ValueError)):
            sp.hdfe_ols(FORMULA, _data(), **kw)

    def test_negative_weight_error_names_the_weights(self):
        df = _data()
        w = df["w"].to_numpy().copy()
        w[2] = -1.0
        with pytest.raises((StatsPAIError, ValueError), match="weight"):
            _fit(FORMULA, df.assign(w=w), weights="w")


class TestSeTypeIsALabel:
    @pytest.mark.parametrize(
        "kw,label",
        [
            ({}, "iid"),
            ({"cluster": "cl"}, "cluster"),
            ({"cluster": ["cl", "cl2"]}, "multiway_cluster"),
            ({"vce": "robust"}, "hc_robust"),
            ({"vce": "robust"}, "robust"),
            ({"vce": "hc0"}, "HC0"),
            ({"vce": "CR2", "cluster": "cl"}, "cluster"),
            ({"vce": "CR2", "cluster": "cl"}, "cr2"),
        ],
    )
    def test_a_label_that_agrees_changes_nothing(self, kw, label):
        df = _data()
        ref = _fit(FORMULA, df, **kw)
        got = _fit(FORMULA, df, se_type=label, **kw)
        np.testing.assert_array_equal(got.se.to_numpy(), ref.se.to_numpy())
        np.testing.assert_array_equal(got.coef.to_numpy(), ref.coef.to_numpy())
        if "vce" in kw:
            assert got.se_type == ref.se_type

    @pytest.mark.parametrize(
        "kw,label",
        [
            ({"vce": "robust"}, "iid"),
            ({"vce": "robust"}, "cluster"),
            ({"vce": "CR2", "cluster": "cl"}, "iid"),
            ({"vce": "CR2", "cluster": "cl"}, "cr3"),
            ({"cluster": "cl"}, "multiway_cluster"),
        ],
    )
    def test_a_label_that_disagrees_is_refused(self, kw, label):
        with pytest.raises(MethodIncompatibility, match="se_type"):
            sp.hdfe_ols(FORMULA, _data(), se_type=label, **kw)

    def test_feols_names_the_options_the_user_passed(self):
        df = _data()
        with pytest.raises(MethodIncompatibility) as err:
            sp.feols("y ~ x1 | f1[x2]", df, vcov="hetero", cluster="cl")
        text = str(err.value)
        assert "vcov='hetero'" in text and "cluster='cl'" in text
        assert "hdfe_ols" not in text and "vce=" not in text
