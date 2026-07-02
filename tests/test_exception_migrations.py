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
    DataInsufficient,
    IdentificationFailure,
    MethodIncompatibility,
    NumericalInstability,
)

# ====================================================================== #
#  DID 2x2 — MethodIncompatibility on wrong treat/time cardinality
# ====================================================================== #


class TestDid2x2Validation:
    def _df_multi_treat(self):
        return pd.DataFrame(
            {
                "y": np.arange(12, dtype=float),
                "treat": [0, 1, 2, 0, 1, 2] * 2,
                "time": [0, 0, 0, 1, 1, 1] * 2,
            }
        )

    def _df_multi_time(self):
        return pd.DataFrame(
            {
                "y": np.arange(12, dtype=float),
                "treat": [0, 1] * 6,
                "time": [0, 1, 2] * 4,
            }
        )

    def test_multi_treat_raises_method_incompatibility(self):
        with pytest.raises(MethodIncompatibility) as excinfo:
            sp.did_2x2(self._df_multi_treat(), y="y", treat="treat", time="time")
        err = excinfo.value
        assert err.recovery_hint
        assert err.diagnostics["n_unique_values"] == 3
        assert "sp.callaway_santanna" in err.alternative_functions

    def test_multi_treat_still_catches_as_value_error(self):
        with pytest.raises(
            ValueError
        ):  # bw-compat: MethodIncompatibility IS ValueError
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
        assert "sp.iv_bounds" in err.alternative_functions

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
        df = pd.DataFrame(
            {
                "unit": np.repeat(range(n_units), n_periods),
                "year": np.tile(range(2018, 2023), n_units),
                "treated": 0,
                "y": rng.normal(size=n_units * n_periods),
            }
        )
        # staggered
        df.loc[df["unit"] < 10, "treated"] = (df["year"] >= 2020).astype(int)
        df.loc[(df["unit"] >= 10) & (df["unit"] < 15), "treated"] = (
            df["year"] >= 2021
        ).astype(int)
        return df

    def test_cs_requires_id(self):
        with pytest.raises(MethodIncompatibility) as excinfo:
            sp.did_analysis(
                self._staggered_df(),
                y="y",
                treat="treated",
                time="year",
                method="cs",
            )
        err = excinfo.value
        assert "id" in err.diagnostics.get("missing", "")
        assert "sp.did" in err.alternative_functions

    def test_sa_requires_id(self):
        with pytest.raises(MethodIncompatibility):
            sp.did_analysis(
                self._staggered_df(),
                y="y",
                treat="treated",
                time="year",
                method="sa",
            )


# ====================================================================== #
#  Matching migrations
# ====================================================================== #


class TestMatchValidation:
    def _multi_treat_df(self):
        rng = np.random.default_rng(0)
        return pd.DataFrame(
            {
                "y": rng.normal(size=200),
                "treat": np.random.choice([0, 1, 2], size=200),
                "x1": rng.normal(size=200),
            }
        )

    def test_non_binary_treatment_raises_method_incompatibility(self):
        with pytest.raises(MethodIncompatibility) as excinfo:
            sp.match(
                self._multi_treat_df(),
                y="y",
                treat="treat",
                covariates=["x1"],
            )
        err = excinfo.value
        assert "sp.multi_treatment" in err.alternative_functions

    def test_non_binary_still_value_error(self):
        with pytest.raises(ValueError):
            sp.match(
                self._multi_treat_df(),
                y="y",
                treat="treat",
                covariates=["x1"],
            )


class TestEbalanceInsufficient:
    def test_raises_data_insufficient(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "y": rng.normal(size=10),
                "treat": [1] + [0] * 9,  # only one treated unit
                "x1": rng.normal(size=10),
            }
        )
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
        return pd.DataFrame(
            {
                "y": rng.normal(size=n),
                "d": rng.normal(size=n),  # continuous treatment
                "x1": rng.normal(size=n),
                "x2": rng.normal(size=n),
            }
        )

    def test_continuous_treatment_raises_method_incompatibility(self):
        with pytest.raises(MethodIncompatibility) as excinfo:
            sp.dml(
                self._cont_treat_df(),
                y="y",
                treat="d",
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
                rows.append(
                    {
                        "unit": f"u{i}",
                        "time": t,
                        "y": rng.normal(),
                    }
                )
        return pd.DataFrame(rows)

    def test_conformal_synth_insufficient_pre(self):
        df = self._tiny_panel(n_pre=1, n_post=2)
        with pytest.raises(DataInsufficient):
            sp.conformal_synth(
                df,
                outcome="y",
                unit="unit",
                time="time",
                treated_unit="u0",
                treatment_time=100,
            )

    def test_conformal_synth_insufficient_post(self):
        df = self._tiny_panel(n_pre=5, n_post=0)
        with pytest.raises(DataInsufficient):
            sp.conformal_synth(
                df,
                outcome="y",
                unit="unit",
                time="time",
                treated_unit="u0",
                treatment_time=100,
            )

    def test_gsynth_insufficient_pre(self):
        # 2 pre-periods, 2 post-periods — GSynth needs 3 pre
        rng = np.random.default_rng(0)
        n_units, n_pre, n_post = 10, 2, 2
        times = list(range(n_pre)) + list(range(100, 100 + n_post))
        rows = []
        for i in range(n_units):
            for t in times:
                rows.append(
                    {
                        "unit": f"u{i}",
                        "time": t,
                        "y": rng.normal(),
                    }
                )
        df = pd.DataFrame(rows)
        with pytest.raises(DataInsufficient) as excinfo:
            sp.gsynth(
                df,
                outcome="y",
                unit="unit",
                time="time",
                treated_unit="u0",
                treatment_time=100,
            )
        assert excinfo.value.diagnostics["n_pre_periods"] == 2
        assert "sp.synth" in excinfo.value.alternative_functions


# ====================================================================== #
#  SBW non-binary treatment
# ====================================================================== #


class TestSbwBinary:
    def test_non_binary_raises_method_incompatibility(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "y": rng.normal(size=100),
                "treat": np.random.choice([0, 1, 2], size=100),
                "x1": rng.normal(size=100),
            }
        )
        with pytest.raises(MethodIncompatibility) as excinfo:
            sp.sbw(df, y="y", treat="treat", covariates=["x1"])
        assert "sp.multi_treatment" in excinfo.value.alternative_functions


# ====================================================================== #
#  optimal_match needs both arms
# ====================================================================== #


class TestOptimalMatchInsufficient:
    def test_all_treated_raises(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "y": rng.normal(size=20),
                "treat": [1] * 20,  # all treated
                "x1": rng.normal(size=20),
            }
        )
        with pytest.raises(DataInsufficient) as excinfo:
            sp.optimal_match(
                df,
                treatment="treat",
                outcome="y",
                covariates=["x1"],
            )
        assert excinfo.value.diagnostics["n_control"] == 0


# ====================================================================== #
#  Reachability contract — every alternative_functions pointer must resolve
# ====================================================================== #
#
# A StatsPAIError's ``alternative_functions`` is a list of ``sp.xxx`` names an
# agent is told to try when the primary call fails. If one of those names does
# not resolve to a callable on the ``sp`` namespace, the agent follows the
# recovery straight into an AttributeError — the recovery channel becomes a
# trap. This was real: ``sp.iv_bounds`` was shipped as ``sp.bounds`` (a module,
# not a callable) at three IV under-identification raise sites. This test scans
# every ``alternative_functions=[...]`` literal in the source tree and asserts
# each pointer resolves, so no future raise site can ship a dead recovery link.


def _resolve_sp_pointer(ref: str):
    """Walk the dotted ``sp.xxx[.yyy]`` path on the statspai namespace exactly
    as an agent would when calling it. Returns True if it lands on a callable,
    False if any attribute is missing / not callable, None if ``ref`` is not an
    ``sp.`` pointer (those are reported separately, never silently passed)."""
    if not ref or not ref.startswith("sp."):
        return None
    obj = sp
    for part in ref[3:].split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return False
    return callable(obj)


def _alternative_function_pointers():
    """Every string literal appearing in an ``alternative_functions=[...]``
    keyword across ``src/statspai`` — one entry per (pointer, file) so a dead
    link's origin is reportable. Parsed from source (not by triggering every
    raise site) so all ~150 call sites are covered in microseconds."""
    import ast
    import importlib
    import pathlib

    pkg_root = pathlib.Path(importlib.import_module("statspai").__file__).parent
    refs: dict = {}
    for path in pkg_root.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.keyword)
                and node.arg == "alternative_functions"
                and isinstance(node.value, ast.List)
            ):
                for elt in node.value.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        refs.setdefault(elt.value, set()).add(
                            str(path.relative_to(pkg_root))
                        )
    return refs


def test_exception_alternative_functions_resolve():
    """Every ``alternative_functions`` recovery pointer names a callable on the
    ``sp`` namespace — a dead pointer turns the recovery channel into a trap."""
    refs = _alternative_function_pointers()
    # Sanity: the scan found the population (guards against a rename that would
    # silently empty this contract).
    assert len(refs) >= 30, (
        f"expected >=30 distinct alternative_functions pointers, found "
        f"{len(refs)} — did the keyword get renamed?"
    )

    dead, non_sp = [], []
    for ref in sorted(refs):
        status = _resolve_sp_pointer(ref)
        if status is False:
            dead.append(f"{ref} <- {sorted(refs[ref])}")
        elif status is None:
            non_sp.append(f"{ref} <- {sorted(refs[ref])}")

    assert not dead, (
        "alternative_functions name unresolvable pointers — an agent following "
        "the recovery hits AttributeError:\n  " + "\n  ".join(dead)
    )
    assert not non_sp, (
        "alternative_functions contains non-sp. pointers (agents expect "
        "callable sp.xxx names):\n  " + "\n  ".join(non_sp)
    )
