"""Unit tests for statspai.did._core — shared DiD primitives."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statspai.did import _core as dc

# ----------------------------------------------------------------------
# cluster_bootstrap_draw
# ----------------------------------------------------------------------


class TestClusterBootstrapDraw:
    def _panel(self, n_clust=8, n_periods=4, seed=0):
        rng = np.random.default_rng(seed)
        rows = []
        for g in range(n_clust):
            for t in range(n_periods):
                rows.append({"g": g, "t": t, "y": rng.normal()})
        return pd.DataFrame(rows)

    def test_preserves_row_count(self):
        df = self._panel()
        rng = np.random.default_rng(42)
        draw = dc.cluster_bootstrap_draw(df, cluster_col="g", rng=rng)
        assert len(draw) == len(df)

    def test_relabel_prevents_collisions(self):
        """When a cluster is drawn twice, its two copies must have distinct
        relabelled values, so downstream groupby doesn't collapse them."""
        df = self._panel(n_clust=4)
        rng = np.random.default_rng(0)
        # Force collisions by sampling many times
        for _ in range(10):
            draw = dc.cluster_bootstrap_draw(df, cluster_col="g", rng=rng)
            # Each chunk has n_periods rows with identical relabelled id
            per_id = draw.groupby("g").size()
            # All chunk sizes should equal the periods-per-cluster constant
            assert (per_id == 4).all(), "Bootstrap relabel merged independent draws"

    def test_raises_on_missing_cluster_col(self):
        df = self._panel()
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="cluster_col"):
            dc.cluster_bootstrap_draw(df, cluster_col="nonexistent", rng=rng)


# ----------------------------------------------------------------------
# event_study_frame
# ----------------------------------------------------------------------


class TestEventStudyFrame:
    def test_empty_input_returns_empty_frame(self):
        df = dc.event_study_frame([])
        assert list(df.columns) == list(dc.EVENT_STUDY_COLUMNS)
        assert len(df) == 0

    def test_fills_missing_optional_keys(self):
        rows = [
            {"relative_time": -1, "att": 0.1, "se": 0.05},
            {"relative_time": 0, "att": 0.3, "se": 0.06, "pvalue": 0.01},
        ]
        df = dc.event_study_frame(rows)
        assert set(dc.EVENT_STUDY_COLUMNS) <= set(df.columns)
        assert df["type"].tolist() == ["", ""]
        # First row lacks pvalue → should be NaN
        assert pd.isna(df.iloc[0]["pvalue"])
        assert df.iloc[1]["pvalue"] == 0.01

    def test_preserves_extra_columns(self):
        rows = [{"relative_time": 0, "att": 1.0, "se": 0.1, "n_switchers": 50}]
        df = dc.event_study_frame(rows)
        assert "n_switchers" in df.columns


# ----------------------------------------------------------------------
# influence_function_se
# ----------------------------------------------------------------------


class TestInfluenceFunctionSE:
    def test_scalar_case_matches_manual_var(self):
        rng = np.random.default_rng(1)
        ifv = rng.normal(size=500)
        se = dc.influence_function_se(ifv)
        expected = np.sqrt(np.var(ifv, ddof=1) / len(ifv))
        assert se == pytest.approx(expected, rel=1e-10)

    def test_vector_case_shape(self):
        rng = np.random.default_rng(2)
        mat = rng.normal(size=(500, 3))
        se = dc.influence_function_se(mat)
        assert se.shape == (3,)
        assert (se > 0).all()

    def test_clustered_se_uses_cluster_sums(self):
        """With identical within-cluster IFs, cluster-robust SE should
        equal the single-observation SE times sqrt(n_per_cluster)."""
        rng = np.random.default_rng(3)
        n_clust, per = 40, 5
        # Per-cluster draw, replicated across rows within cluster.
        cluster_val = rng.normal(size=n_clust)
        if_vec = np.repeat(cluster_val, per)
        cluster_ids = np.repeat(np.arange(n_clust), per)

        se_cluster = dc.influence_function_se(if_vec, cluster_ids=cluster_ids)
        # Cluster sum IF = cluster_val * per; Var across clusters / n_clust.
        expected_var = (per**2) * np.var(cluster_val, ddof=1) / n_clust
        assert se_cluster == pytest.approx(np.sqrt(expected_var), rel=1e-10)

    def test_raises_on_mismatched_cluster_length(self):
        with pytest.raises(ValueError, match="cluster_ids"):
            dc.influence_function_se(np.zeros(5), cluster_ids=np.zeros(4))


# ----------------------------------------------------------------------
# joint_wald
# ----------------------------------------------------------------------


class TestJointWald:
    def test_scalar_case(self):
        r = dc.joint_wald(np.array([2.0]), np.array([[1.0]]))
        assert r["df"] == 1
        assert r["statistic"] == pytest.approx(4.0)
        # chi2(1) at 4 → p ≈ 0.0455
        assert 0.04 < r["pvalue"] < 0.05

    def test_vector_case(self):
        est = np.array([1.0, -0.5])
        cov = np.array([[0.25, 0.0], [0.0, 0.25]])
        r = dc.joint_wald(est, cov)
        # W = 1^2/0.25 + 0.25/0.25 = 4 + 1 = 5. Looser tolerance because
        # joint_wald adds a 1e-10 ridge before inversion.
        assert r["df"] == 2
        assert r["statistic"] == pytest.approx(5.0, rel=1e-6)

    def test_singular_covariance_regularises(self):
        est = np.array([1.0, 1.0])
        cov = np.array([[1.0, 1.0], [1.0, 1.0]])  # rank 1
        r = dc.joint_wald(est, cov)
        assert np.isfinite(r["statistic"])
        assert 0 <= r["pvalue"] <= 1


# ----------------------------------------------------------------------
# long_difference + sorted_periods
# ----------------------------------------------------------------------


class TestLongDifference:
    def test_long_difference_matches_manual(self):
        df = pd.DataFrame(
            {
                "id": [1, 1, 1, 2, 2, 2],
                "t": [1, 2, 3, 1, 2, 3],
                "y": [10, 12, 15, 20, 22, 25],
            }
        )
        out = dc.long_difference(
            df,
            id_col="id",
            time_col="t",
            y_col="y",
            t_base=1,
            t_future=3,
        )
        assert set(out.columns) == {"id", "ldy"}
        assert out.set_index("id")["ldy"].to_dict() == {1: 5, 2: 5}

    def test_missing_unit_is_dropped(self):
        df = pd.DataFrame(
            {
                "id": [1, 1, 2],  # id=2 only at t=1
                "t": [1, 2, 1],
                "y": [10.0, 11.0, 20.0],
            }
        )
        out = dc.long_difference(
            df,
            id_col="id",
            time_col="t",
            y_col="y",
            t_base=1,
            t_future=2,
        )
        assert out["id"].tolist() == [1]


def test_sorted_periods_is_ascending_unique():
    s = pd.Series([3, 1, 2, 2, np.nan, 3])
    assert dc.sorted_periods(s) == [1, 2, 3]
