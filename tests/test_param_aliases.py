"""Tests for canonical y/d/x parameter aliases across estimators.

The alias contract: additive only (native spellings keep working),
alias and native produce identical numbers, and supplying both with
different values fails loudly.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

torch = pytest.importorskip  # noqa: E731  (used selectively below)


@pytest.fixture(scope="module")
def xdata():
    rng = np.random.default_rng(0)
    n = 300
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    d = rng.binomial(1, 0.5, n)
    y = 1 + 2 * d + x1 + rng.normal(size=n)
    return pd.DataFrame(
        {
            "y": y,
            "d": d,
            "x1": x1,
            "x2": x2,
            "post": rng.binomial(1, 0.5, n),
        }
    )


@pytest.fixture(scope="module")
def paneldata():
    rng = np.random.default_rng(1)
    rows = []
    for u in range(8):
        fe = rng.normal()
        for t in range(10):
            rows.append(
                {
                    "st": f"u{u}",
                    "yr": 2000 + t,
                    "sales": fe
                    - 0.1 * t
                    + rng.normal(0, 0.2)
                    + (-3 if (u == 0 and t >= 6) else 0),
                }
            )
    return pd.DataFrame(rows)


class TestAliasEquivalence:
    def test_metalearner_d_x(self, xdata):
        native = sp.metalearner(
            xdata,
            y="y",
            treat="d",
            covariates=["x1", "x2"],
            learner="t",
            n_bootstrap=20,
        )
        alias = sp.metalearner(
            xdata, y="y", d="d", x=["x1", "x2"], learner="t", n_bootstrap=20
        )
        assert alias.estimate == pytest.approx(native.estimate)

    def test_did_2x2_d(self, xdata):
        native = sp.did_2x2(xdata, y="y", treat="d", time="post")
        alias = sp.did_2x2(xdata, y="y", d="d", time="post")
        assert alias.estimate == pytest.approx(native.estimate)

    def test_lasso_iv_d(self, xdata):
        df = xdata.copy()
        rng = np.random.default_rng(2)
        df["z1"] = rng.normal(size=len(df))
        df["z2"] = df["z1"] ** 2
        df["p"] = df["z1"] + rng.normal(size=len(df))
        df["q"] = 10 - 2 * df["p"] + rng.normal(size=len(df))
        native = sp.lasso_iv(df, y="q", x_endog=["p"], z=["z1", "z2"])
        alias = sp.lasso_iv(df, y="q", d="p", z=["z1", "z2"])
        assert alias.params["p"] == pytest.approx(native.params["p"])

    def test_synth_y(self, paneldata):
        native = sp.synth(
            paneldata,
            outcome="sales",
            unit="st",
            time="yr",
            treated_unit="u0",
            treatment_time=2006,
            placebo=False,
        )
        alias = sp.synth(
            paneldata,
            y="sales",
            unit="st",
            time="yr",
            treated_unit="u0",
            treatment_time=2006,
            placebo=False,
        )
        assert alias.estimate == pytest.approx(native.estimate)

    def test_causal_forest_column_interface(self, xdata):
        by_cols = sp.causal_forest(
            data=xdata,
            y="y",
            d="d",
            x=["x1", "x2"],
            n_estimators=30,
            random_state=0,
        )
        by_arrays = sp.causal_forest(
            Y=xdata["y"].to_numpy(),
            T=xdata["d"].to_numpy(),
            X=xdata[["x1", "x2"]].to_numpy(),
            n_estimators=30,
            random_state=0,
        )
        assert float(by_cols.ate()) == pytest.approx(float(by_arrays.ate()))

    def test_tarnet_d_x(self, xdata):
        pytest.importorskip("torch")
        native = sp.tarnet(
            xdata,
            y="y",
            treat="d",
            covariates=["x1", "x2"],
            epochs=3,
            n_bootstrap=0,
            random_state=0,
        )
        alias = sp.tarnet(
            xdata,
            y="y",
            d="d",
            x=["x1", "x2"],
            epochs=3,
            n_bootstrap=0,
            random_state=0,
        )
        assert alias.estimate == pytest.approx(native.estimate)


class TestAliasConflicts:
    def test_conflicting_values_raise(self, xdata):
        with pytest.raises(MethodIncompatibility, match="alias"):
            sp.did_2x2(xdata, y="y", treat="d", d="post", time="post")

    def test_same_value_in_both_is_fine(self, xdata):
        r = sp.did_2x2(xdata, y="y", treat="d", d="d", time="post")
        assert np.isfinite(r.estimate)

    def test_missing_required_raises_typeerror(self, xdata):
        with pytest.raises(TypeError, match="treat"):
            sp.did_2x2(xdata, y="y", time="post")
        with pytest.raises(TypeError, match="outcome"):
            sp.synth(xdata, unit="st", time="yr")

    def test_causal_forest_mixed_interfaces_raise(self, xdata):
        with pytest.raises(MethodIncompatibility, match="not a mixture"):
            sp.causal_forest(
                data=xdata,
                y="y",
                d="d",
                x=["x1"],
                Y=xdata["y"].to_numpy(),
            )
        with pytest.raises(MethodIncompatibility, match="require data"):
            sp.causal_forest(y="y", d="d", x=["x1"])
        with pytest.raises(MethodIncompatibility, match="not in data"):
            sp.causal_forest(data=xdata, y="y", d="d", x=["nope"])
