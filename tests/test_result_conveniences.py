"""Tests for result-object ergonomics added in the notebook-driven pass:

- ``EconometricResults.coef(term)`` / ``CausalResult.coef()``
- uniform CATE access (``result.cate`` / ``result.effect``)
- panel counterfactual accessors (``weights`` / ``counterfactual`` / ``gaps``)
- labeled matrix-completion diagnostics
- DML summary label polish
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


@pytest.fixture(scope="module")
def regdata():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    return pd.DataFrame({"y": 2.0 * x + rng.normal(size=200), "x": x})


@pytest.fixture(scope="module")
def paneldata():
    rng = np.random.default_rng(1)
    rows = []
    for u in range(8):
        fe = rng.normal()
        for t in range(12):
            rows.append(
                {
                    "st": f"u{u}",
                    "yr": 2000 + t,
                    "sales": fe
                    - 0.1 * t
                    + rng.normal(0, 0.2)
                    + (-3 if (u == 0 and t >= 8) else 0),
                    "d": int(u == 0 and t >= 8),
                }
            )
    return pd.DataFrame(rows)


class TestCoef:
    def test_regress_coef_matches_tidy(self, regdata):
        res = sp.regress("y ~ x", data=regdata)
        row = res.coef("x")
        tidy_row = res.tidy().set_index("term").loc["x"]
        assert row["estimate"] == pytest.approx(tidy_row["estimate"])
        assert row["p_value"] == pytest.approx(tidy_row["p_value"])
        assert set(row.index) >= {
            "estimate",
            "std_error",
            "statistic",
            "p_value",
            "conf_low",
            "conf_high",
        }

    def test_regress_coef_unknown_term(self, regdata):
        res = sp.regress("y ~ x", data=regdata)
        with pytest.raises(MethodIncompatibility, match="Available terms"):
            res.coef("nope")

    def test_causal_coef(self, regdata):
        df = regdata.copy()
        rng = np.random.default_rng(2)
        df["d"] = rng.binomial(1, 0.5, len(df))
        df["post"] = rng.binomial(1, 0.5, len(df))
        r = sp.did_2x2(df, y="y", d="d", time="post")
        c = r.coef
        # Float-compatible: duck-typing probes keep working.
        assert float(c) == pytest.approx(r.estimate)
        assert float(getattr(r, "coef", r.estimate)) == pytest.approx(r.estimate)
        assert c.se == pytest.approx(r.se)
        assert c.estimand == r.estimand
        assert "SE = " in str(c)


class TestCATEAccess:
    def test_metalearner_cate_property(self, regdata):
        df = regdata.copy()
        rng = np.random.default_rng(3)
        df["d"] = rng.binomial(1, 0.5, len(df))
        df["x2"] = rng.normal(size=len(df))
        r = sp.metalearner(df, y="y", d="d", x=["x", "x2"], learner="t", n_bootstrap=20)
        assert r.cate.shape == (len(df),)
        assert np.allclose(r.effect(), r.cate)
        with pytest.raises(MethodIncompatibility, match="no fitted model"):
            r.effect(np.zeros((3, 2)))

    def test_non_cate_result_raises(self, regdata):
        df = regdata.copy()
        rng = np.random.default_rng(4)
        df["d"] = rng.binomial(1, 0.5, len(df))
        df["post"] = rng.binomial(1, 0.5, len(df))
        r = sp.did_2x2(df, y="y", d="d", time="post")
        with pytest.raises(MethodIncompatibility, match="per-unit effects"):
            _ = r.cate

    def test_cate_by_group_accepts_forest_and_array(self, regdata):
        df = regdata.copy()
        rng = np.random.default_rng(5)
        df["d"] = rng.binomial(1, 0.5, len(df))
        df["x2"] = rng.normal(size=len(df))
        cf = sp.causal_forest(
            data=df,
            y="y",
            d="d",
            x=["x", "x2"],
            n_estimators=30,
            random_state=0,
        )
        by_model = sp.cate_by_group(cf, df, by="cate", n_groups=4)
        assert len(by_model) == 4
        tau = cf.effect(df[["x", "x2"]].to_numpy())
        by_array = sp.cate_by_group(tau, df, by="cate", n_groups=4)
        pd.testing.assert_frame_equal(by_model, by_array)


class TestPanelCounterfactuals:
    def test_synth_accessors(self, paneldata):
        r = sp.synth(
            paneldata,
            y="sales",
            unit="st",
            time="yr",
            treated_unit="u0",
            treatment_time=2008,
            placebo=False,
        )
        w = r.weights
        assert isinstance(w, pd.DataFrame)
        assert "weight" in w.columns
        cf = r.counterfactual()
        assert isinstance(cf, pd.Series)
        assert list(cf.index) == sorted(paneldata["yr"].unique())
        gaps = r.gaps
        # Post-treatment gap should be clearly negative (true effect -3).
        assert gaps.loc[2008:].mean() < -1.5
        # Identity: observed - counterfactual == gap.
        y_treated = np.asarray(r.model_info["Y_treated"], dtype=float)
        assert np.allclose(y_treated - cf.to_numpy(), gaps.to_numpy())

    def test_mc_labeled_diagnostics(self, paneldata):
        r = sp.matrix_completion(
            paneldata,
            y="sales",
            d="d",
            unit="st",
            time="yr",
            n_bootstrap=50,
        )
        mi = r.model_info
        assert mi["treated_units"] == ["u0"]
        assert list(mi["completed_df"].index) == mi["units"]
        assert np.allclose(mi["completed_df"].to_numpy(), mi["completed_matrix"])
        cf = r.counterfactual()
        assert isinstance(cf, pd.Series)  # single treated unit squeezes
        assert cf.name == "u0"
        gaps = r.gaps
        assert gaps.loc[2008:].mean() < -1.5

    def test_mc_row_order_is_labeled_not_positional(self):
        """Unit labels that sort lexicographically differently from
        their numeric order (u1, u10, u2, ...) must still map to the
        right counterfactual rows — the failure mode of positional
        guessing."""
        rng = np.random.default_rng(7)
        rows = []
        for u in range(12):
            fe = rng.normal()
            for t in range(10):
                rows.append(
                    {
                        "st": f"u{u + 1}",
                        "yr": 2000 + t,
                        "sales": fe
                        + 0.1 * t
                        + rng.normal(0, 0.1)
                        + (-5 if (u == 4 and t >= 6) else 0),
                        "d": int(u == 4 and t >= 6),
                    }
                )
        r = sp.matrix_completion(
            pd.DataFrame(rows),
            y="sales",
            d="d",
            unit="st",
            time="yr",
            n_bootstrap=50,
        )
        assert r.model_info["treated_units"] == ["u5"]
        assert r.counterfactual().name == "u5"

    def test_no_counterfactual_raises(self, regdata):
        df = regdata.copy()
        rng = np.random.default_rng(8)
        df["d"] = rng.binomial(1, 0.5, len(df))
        df["post"] = rng.binomial(1, 0.5, len(df))
        r = sp.did_2x2(df, y="y", d="d", time="post")
        with pytest.raises(MethodIncompatibility, match="counterfactual"):
            r.counterfactual()
        with pytest.raises(MethodIncompatibility, match="donor weights"):
            _ = r.weights


class TestDMLSummaryLabels:
    def test_labels_readable(self):
        rng = np.random.default_rng(1)
        n = 400
        X = rng.uniform(-1, 1, (n, 3))
        D = rng.binomial(1, 1 / (1 + np.exp(-X[:, 0])))
        Y = 2 * D + X[:, 0] ** 2 + rng.normal(0, 1, n)
        df = pd.DataFrame(X, columns=["a", "b", "c"])
        df["d"] = D
        df["y"] = Y
        text = sp.dml(df, y="y", d="d", X=["a", "b", "c"]).summary()
        assert "DML model" in text
        assert "ML model for g(X)" in text
        assert "Ml G" not in text
        # Private keys must not leak into the footer.
        assert "Pscore" not in text
