"""Tests for call sites migrated to the typed exception taxonomy.

Each test verifies that:
1. The typed subclass is raised (so agents can branch on it).
2. The parent class (ValueError / RuntimeError) still catches it
   (so existing `except` blocks are untouched).
3. The exception carries a useful ``recovery_hint``.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import (
    MethodIncompatibility,
    DataInsufficient,
    NumericalInstability,
    IdentificationFailure,
)


# ====================================================================== #
#  DID 2x2 — MethodIncompatibility on wrong treat/time cardinality
# ====================================================================== #


class TestDid2x2Validation:
    def _df_multi_treat(self):
        return pd.DataFrame({
            "y": np.arange(12, dtype=float),
            "treat": [0, 1, 2, 0, 1, 2] * 2,
            "time": [0, 0, 0, 1, 1, 1] * 2,
        })

    def _df_multi_time(self):
        return pd.DataFrame({
            "y": np.arange(12, dtype=float),
            "treat": [0, 1] * 6,
            "time": [0, 1, 2] * 4,
        })

    def test_multi_treat_raises_method_incompatibility(self):
        with pytest.raises(MethodIncompatibility) as excinfo:
            sp.did_2x2(self._df_multi_treat(), y="y", treat="treat", time="time")
        err = excinfo.value
        assert err.recovery_hint
        assert err.diagnostics["n_unique_values"] == 3
        assert "sp.callaway_santanna" in err.alternative_functions

    def test_multi_treat_still_catches_as_value_error(self):
        with pytest.raises(ValueError):  # bw-compat: MethodIncompatibility IS ValueError
            sp.did_2x2(self._df_multi_treat(), y="y", treat="treat", time="time")

    def test_multi_time_raises_method_incompatibility(self):
        with pytest.raises(MethodIncompatibility) as excinfo:
            sp.did_2x2(self._df_multi_time(), y="y", treat="treat", time="time")
        err = excinfo.value
        assert "sp.callaway_santanna" in err.alternative_functions


# ====================================================================== #
#  IV under-identification — MethodIncompatibility
# ====================================================================== #


class TestIVUnderIdentified:
    def test_raises_method_incompatibility(self):
        # Two endogenous regressors but only one instrument.
        rng = np.random.default_rng(0)
        n = 200
        z = rng.normal(size=n)
        d1 = 0.5 * z + rng.normal(size=n)
        d2 = rng.normal(size=n)
        y = d1 + d2 + rng.normal(size=n)
        df = pd.DataFrame({"y": y, "d1": d1, "d2": d2, "z": z})

        with pytest.raises(MethodIncompatibility) as excinfo:
            sp.ivreg("y ~ (d1 + d2 ~ z)", data=df)
        err = excinfo.value
        assert err.diagnostics["n_instruments"] < err.diagnostics["n_endogenous"]
        assert "sp.bounds" in err.alternative_functions

    def test_still_catches_as_value_error(self):
        rng = np.random.default_rng(0)
        n = 200
        z = rng.normal(size=n)
        d1 = 0.5 * z + rng.normal(size=n)
        d2 = rng.normal(size=n)
        y = d1 + d2 + rng.normal(size=n)
        df = pd.DataFrame({"y": y, "d1": d1, "d2": d2, "z": z})
        with pytest.raises(ValueError):
            sp.ivreg("y ~ (d1 + d2 ~ z)", data=df)


# ====================================================================== #
#  DID analysis() — MethodIncompatibility when id missing for staggered
# ====================================================================== #


class TestDidAnalysisIdRequired:
    def _staggered_df(self):
        rng = np.random.default_rng(0)
        n_units, n_periods = 20, 5
        df = pd.DataFrame({
            "unit": np.repeat(range(n_units), n_periods),
            "year": np.tile(range(2018, 2023), n_units),
            "treated": 0,
            "y": rng.normal(size=n_units * n_periods),
        })
        # staggered
        df.loc[df["unit"] < 10, "treated"] = (df["year"] >= 2020).astype(int)
        df.loc[(df["unit"] >= 10) & (df["unit"] < 15), "treated"] = (df["year"] >= 2021).astype(int)
        return df

    def test_cs_requires_id(self):
        with pytest.raises(MethodIncompatibility) as excinfo:
            sp.did_analysis(
                self._staggered_df(),
                y="y", treat="treated", time="year",
                method="cs",
            )
        err = excinfo.value
        assert "id" in err.diagnostics.get("missing", "")
        assert "sp.did" in err.alternative_functions

    def test_sa_requires_id(self):
        with pytest.raises(MethodIncompatibility):
            sp.did_analysis(
                self._staggered_df(),
                y="y", treat="treated", time="year",
                method="sa",
            )


# ====================================================================== #
#  Matching migrations
# ====================================================================== #


class TestMatchValidation:
    def _multi_treat_df(self):
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            "y": rng.normal(size=200),
            "treat": np.random.choice([0, 1, 2], size=200),
            "x1": rng.normal(size=200),
        })

    def test_non_binary_treatment_raises_method_incompatibility(self):
        with pytest.raises(MethodIncompatibility) as excinfo:
            sp.match(
                self._multi_treat_df(),
                y="y", treat="treat", covariates=["x1"],
            )
        err = excinfo.value
        assert "sp.multi_treatment" in err.alternative_functions

    def test_non_binary_still_value_error(self):
        with pytest.raises(ValueError):
            sp.match(
                self._multi_treat_df(),
                y="y", treat="treat", covariates=["x1"],
            )


class TestEbalanceInsufficient:
    def test_raises_data_insufficient(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "y": rng.normal(size=10),
            "treat": [1] + [0] * 9,  # only one treated unit
            "x1": rng.normal(size=10),
        })
        with pytest.raises(DataInsufficient) as excinfo:
            sp.ebalance(df, y="y", treat="treat", covariates=["x1"])
        err = excinfo.value
        assert err.diagnostics["n_treated"] == 1


# ====================================================================== #
#  DML IRM migrations
# ====================================================================== #


class TestDmlIrmValidation:
    def _cont_treat_df(self):
        rng = np.random.default_rng(0)
        n = 300
        return pd.DataFrame({
            "y": rng.normal(size=n),
            "d": rng.normal(size=n),  # continuous treatment
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        })

    def test_continuous_treatment_raises_method_incompatibility(self):
        with pytest.raises(MethodIncompatibility) as excinfo:
            sp.dml(
                self._cont_treat_df(),
                y="y", treat="d",
                covariates=["x1", "x2"],
                model="irm",
            )
        err = excinfo.value
        assert "sp.dml" in err.alternative_functions


# ====================================================================== #
#  Synth migrations
# ====================================================================== #


class TestSynthInsufficientPeriods:
    def _tiny_panel(self, n_pre=1, n_post=1):
        rng = np.random.default_rng(0)
        n_units = 5
        times = list(range(n_pre)) + list(range(100, 100 + n_post))
        rows = []
        for i in range(n_units):
            for t in times:
                rows.append({
                    "unit": f"u{i}",
                    "time": t,
                    "y": rng.normal(),
                })
        return pd.DataFrame(rows)

    def test_conformal_synth_insufficient_pre(self):
        df = self._tiny_panel(n_pre=1, n_post=2)
        with pytest.raises(DataInsufficient):
            sp.conformal_synth(
                df, outcome="y", unit="unit", time="time",
                treated_unit="u0", treatment_time=100,
            )

    def test_conformal_synth_insufficient_post(self):
        df = self._tiny_panel(n_pre=5, n_post=0)
        with pytest.raises(DataInsufficient):
            sp.conformal_synth(
                df, outcome="y", unit="unit", time="time",
                treated_unit="u0", treatment_time=100,
            )

    def test_gsynth_insufficient_pre(self):
        # 2 pre-periods, 2 post-periods — GSynth needs 3 pre
        rng = np.random.default_rng(0)
        n_units, n_pre, n_post = 10, 2, 2
        times = list(range(n_pre)) + list(range(100, 100 + n_post))
        rows = []
        for i in range(n_units):
            for t in times:
                rows.append({
                    "unit": f"u{i}", "time": t, "y": rng.normal(),
                })
        df = pd.DataFrame(rows)
        with pytest.raises(DataInsufficient) as excinfo:
            sp.gsynth(
                df, outcome="y", unit="unit", time="time",
                treated_unit="u0", treatment_time=100,
            )
        assert excinfo.value.diagnostics["n_pre_periods"] == 2
        assert "sp.synth" in excinfo.value.alternative_functions


# ====================================================================== #
#  SBW non-binary treatment
# ====================================================================== #


class TestSbwBinary:
    def test_non_binary_raises_method_incompatibility(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "y": rng.normal(size=100),
            "treat": np.random.choice([0, 1, 2], size=100),
            "x1": rng.normal(size=100),
        })
        with pytest.raises(MethodIncompatibility) as excinfo:
            sp.sbw(df, y="y", treat="treat", covariates=["x1"])
        assert "sp.multi_treatment" in excinfo.value.alternative_functions


# ====================================================================== #
#  optimal_match needs both arms
# ====================================================================== #


class TestOptimalMatchInsufficient:
    def test_all_treated_raises(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "y": rng.normal(size=20),
            "treat": [1] * 20,  # all treated
            "x1": rng.normal(size=20),
        })
        with pytest.raises(DataInsufficient) as excinfo:
            sp.optimal_match(
                df, treatment="treat", outcome="y", covariates=["x1"],
            )
        assert excinfo.value.diagnostics["n_control"] == 0
