"""Tests for the agent-native exception taxonomy (statspai.exceptions)."""

import warnings

import pytest

import statspai as sp
from statspai import exceptions as spx


class TestHierarchy:
    def test_all_errors_inherit_from_base(self):
        for cls in (
            spx.AssumptionViolation,
            spx.IdentificationFailure,
            spx.DataInsufficient,
            spx.ConvergenceFailure,
            spx.NumericalInstability,
            spx.MethodIncompatibility,
        ):
            assert issubclass(cls, spx.StatsPAIError)
            assert issubclass(cls, Exception)

    def test_identification_failure_is_assumption_violation(self):
        assert issubclass(spx.IdentificationFailure, spx.AssumptionViolation)

    def test_assumption_violation_is_value_error_for_bwcompat(self):
        assert issubclass(spx.AssumptionViolation, ValueError)

    def test_convergence_failure_is_runtime_error_for_bwcompat(self):
        assert issubclass(spx.ConvergenceFailure, RuntimeError)

    def test_warnings_inherit_user_warning(self):
        for cls in (spx.ConvergenceWarning, spx.AssumptionWarning):
            assert issubclass(cls, spx.StatsPAIWarning)
            assert issubclass(cls, UserWarning)


class TestPayload:
    def test_basic_recovery_payload(self):
        err = spx.AssumptionViolation(
            "Parallel trends rejected",
            recovery_hint="Try sp.callaway_santanna(...)",
            diagnostics={"pvalue": 0.003, "test": "pretrends"},
            alternative_functions=["sp.callaway_santanna", "sp.did_imputation"],
        )
        assert err.recovery_hint.startswith("Try")
        assert err.diagnostics["pvalue"] == 0.003
        assert err.alternative_functions == [
            "sp.callaway_santanna",
            "sp.did_imputation",
        ]

    def test_defaults_are_empty_not_none(self):
        err = spx.DataInsufficient("n too small")
        assert err.recovery_hint == ""
        assert err.diagnostics == {}
        assert err.alternative_functions == []

    def test_str_includes_recovery_hint(self):
        err = spx.ConvergenceFailure(
            "MCMC did not converge",
            recovery_hint="Increase tune to 4000",
        )
        s = str(err)
        assert "MCMC did not converge" in s
        assert "Increase tune to 4000" in s

    def test_to_dict_is_json_ready(self):
        err = spx.NumericalInstability(
            "Singular design",
            recovery_hint="Drop collinear regressor",
            diagnostics={"condition_number": 1e20},
        )
        d = err.to_dict()
        assert d["kind"] == "numerical_instability"
        assert d["class"] == "NumericalInstability"
        assert d["message"] == "Singular design"
        assert d["diagnostics"] == {"condition_number": 1e20}
        # to_dict mutation must not affect the original
        d["diagnostics"]["extra"] = 1
        assert "extra" not in err.diagnostics


class TestRaiseAndCatch:
    def test_can_catch_specific_and_base(self):
        with pytest.raises(spx.AssumptionViolation):
            raise spx.IdentificationFailure("not identified")
        with pytest.raises(spx.StatsPAIError):
            raise spx.IdentificationFailure("not identified")
        with pytest.raises(ValueError):
            # backwards-compat: AssumptionViolation is a ValueError
            raise spx.AssumptionViolation("assumption failed")


class TestWarn:
    def test_warn_emits_rich_instance(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            spx.warn(
                spx.ConvergenceWarning,
                "rhat=1.05",
                recovery_hint="Increase tune",
                diagnostics={"rhat": 1.05},
            )
        assert len(caught) == 1
        msg = caught[0].message
        assert isinstance(msg, spx.ConvergenceWarning)
        assert msg.recovery_hint == "Increase tune"
        assert msg.diagnostics == {"rhat": 1.05}

    def test_warn_rejects_non_subclass(self):
        with pytest.raises(TypeError):
            spx.warn(UserWarning, "nope")


class TestTopLevelExposure:
    def test_exports_on_sp(self):
        for name in (
            "StatsPAIError",
            "AssumptionViolation",
            "IdentificationFailure",
            "DataInsufficient",
            "ConvergenceFailure",
            "NumericalInstability",
            "MethodIncompatibility",
            "StatsPAIWarning",
            "ConvergenceWarning",
            "AssumptionWarning",
        ):
            assert hasattr(sp, name), f"sp.{name} missing"
            assert name in sp.__all__

    def test_sp_exceptions_module(self):
        assert hasattr(sp, "exceptions")
        assert sp.exceptions.AssumptionViolation is spx.AssumptionViolation
