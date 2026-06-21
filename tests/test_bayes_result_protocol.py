"""Protocol tests for Bayesian result containers without requiring PyMC."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from statspai.bayes._base import (
    BayesianCausalResult,
    BayesianDIDResult,
    BayesianHTEIVResult,
    BayesianIVResult,
    BayesianMTEResult,
    _sample_model,
)
from statspai.exceptions import (
    MethodIncompatibility,
    NumericalInstability,
)


def _bayes_stub(**overrides):
    base = dict(
        method="Bayesian DID",
        estimand="ATT",
        posterior_mean=1.2,
        posterior_median=1.1,
        posterior_sd=0.3,
        hdi_lower=0.5,
        hdi_upper=1.8,
        prob_positive=0.97,
        rhat=1.02,
        ess=250.0,
        n_obs=120,
        model_info={
            "inference": "nuts",
            "chains": np.int64(2),
            "draws": np.int64(500),
            "bad_scalar": float("nan"),
            "array": np.array([1.0, 2.0]),
            "series": pd.Series({"a": 1.0}),
        },
        trace=object(),
    )
    base.update(overrides)
    return BayesianCausalResult(**base)


def test_bayesian_result_to_dict_is_json_safe_and_excludes_trace():
    result = _bayes_stub()

    payload = result.to_dict()
    encoded = json.dumps(payload)

    assert payload["kind"] == "bayesian_causal_result"
    assert payload["posterior"]["hdi"]["lower"] == 0.5
    assert payload["diagnostics"]["rhat"] == 1.02
    assert payload["model_info"]["bad_scalar"] is None
    assert payload["model_info"]["array"] == [1.0, 2.0]
    assert payload["tidy"][0]["term"] == "att"
    assert "trace" not in encoded


def test_bayesian_result_to_agent_summary_flags_convergence_risk():
    result = _bayes_stub()

    payload = result.to_agent_summary()

    assert payload["kind"] == "bayesian_causal_agent_summary"
    assert payload["posterior"]["prob_positive"] == 0.97
    assert any("R-hat" in warning for warning in payload["warnings"])
    assert any("ESS" in warning for warning in payload["warnings"])
    json.dumps(payload)


def test_bayesian_subclass_inherits_agent_protocol():
    result = BayesianDIDResult(
        method="Bayesian DID",
        estimand="ATT",
        posterior_mean=0.4,
        posterior_median=0.4,
        posterior_sd=0.2,
        hdi_lower=0.1,
        hdi_upper=0.8,
        prob_positive=0.95,
        rhat=1.0,
        ess=800.0,
        n_obs=80,
        cohort_summaries={"2019": {"posterior_mean": 0.3}},
        cohort_labels=["2019"],
    )

    assert result.to_dict()["kind"] == "bayesian_causal_result"
    assert result.to_agent_summary()["warnings"] == []


def test_sample_model_validates_controls_before_pymc_import():
    with pytest.raises(MethodIncompatibility, match="inference"):
        _sample_model(None, inference="bad")
    with pytest.raises(MethodIncompatibility, match="draws"):
        _sample_model(None, draws=0)
    with pytest.raises(MethodIncompatibility, match="tune"):
        _sample_model(None, tune=-1)
    with pytest.raises(MethodIncompatibility, match="target_accept"):
        _sample_model(None, target_accept=np.nan)


def test_bayesian_did_tidy_contract_errors_are_taxonomy():
    result = BayesianDIDResult(
        method="Bayesian DID",
        estimand="ATT",
        posterior_mean=0.4,
        posterior_median=0.4,
        posterior_sd=0.2,
        hdi_lower=0.1,
        hdi_upper=0.8,
        prob_positive=0.95,
        rhat=1.0,
        ess=800.0,
        n_obs=80,
    )

    with pytest.raises(MethodIncompatibility, match="cohort_summaries"):
        result.tidy(terms="per_cohort")
    with pytest.raises(MethodIncompatibility, match="Unknown term"):
        result.tidy(terms=["cohort:2099"])
    with pytest.raises(MethodIncompatibility, match="terms"):
        result.tidy(terms=object())


def test_bayesian_iv_tidy_contract_errors_are_taxonomy():
    result = BayesianIVResult(
        method="Bayesian IV",
        estimand="LATE",
        posterior_mean=0.4,
        posterior_median=0.4,
        posterior_sd=0.2,
        hdi_lower=0.1,
        hdi_upper=0.8,
        prob_positive=0.95,
        rhat=1.0,
        ess=800.0,
        n_obs=80,
    )

    with pytest.raises(MethodIncompatibility, match="instrument_summaries"):
        result.tidy(terms="per_instrument")
    with pytest.raises(MethodIncompatibility, match="Unknown term"):
        result.tidy(terms=["instrument:z99"])


def test_bayesian_mte_tidy_contract_errors_are_taxonomy():
    result = BayesianMTEResult(
        method="Bayesian MTE",
        estimand="ATE",
        posterior_mean=0.4,
        posterior_median=0.4,
        posterior_sd=0.2,
        hdi_lower=0.1,
        hdi_upper=0.8,
        prob_positive=0.95,
        rhat=1.0,
        ess=800.0,
        n_obs=80,
    )

    with pytest.raises(MethodIncompatibility, match="Unknown term"):
        result.tidy(terms=["ate", "bogus"])
    with pytest.raises(MethodIncompatibility, match="terms"):
        result.tidy(terms=object())


def test_bayesian_hte_predict_cate_missing_state_is_taxonomy():
    result = BayesianHTEIVResult(
        method="Bayesian HTE-IV",
        estimand="LATE",
        posterior_mean=0.4,
        posterior_median=0.4,
        posterior_sd=0.2,
        hdi_lower=0.1,
        hdi_upper=0.8,
        prob_positive=0.95,
        rhat=1.0,
        ess=800.0,
        n_obs=80,
        effect_modifiers=["x"],
    )

    with pytest.raises(MethodIncompatibility, match="posterior trace"):
        result.predict_cate({"x": 1.0})


class _PosteriorArray:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)


class _Trace:
    def __init__(self, values):
        self.posterior = {"b_mte": _PosteriorArray(values)}


def _mte_policy_stub() -> BayesianMTEResult:
    return BayesianMTEResult(
        method="Bayesian MTE",
        estimand="ATE",
        posterior_mean=0.4,
        posterior_median=0.4,
        posterior_sd=0.2,
        hdi_lower=0.1,
        hdi_upper=0.8,
        prob_positive=0.95,
        rhat=1.0,
        ess=800.0,
        n_obs=80,
        trace=_Trace(np.ones((1, 2, 2))),
        u_grid=np.array([0.0, 1.0]),
    )


def test_bayesian_mte_policy_effect_contract_errors_are_taxonomy():
    result = BayesianMTEResult(
        method="Bayesian MTE",
        estimand="ATE",
        posterior_mean=0.4,
        posterior_median=0.4,
        posterior_sd=0.2,
        hdi_lower=0.1,
        hdi_upper=0.8,
        prob_positive=0.95,
        rhat=1.0,
        ess=800.0,
        n_obs=80,
    )
    with pytest.raises(MethodIncompatibility, match="trace"):
        result.policy_effect(lambda u: np.ones_like(u))

    fitted = _mte_policy_stub()
    with pytest.raises(MethodIncompatibility, match="callable"):
        fitted.policy_effect(None)
    with pytest.raises(MethodIncompatibility, match="shape"):
        fitted.policy_effect(lambda u: np.ones(len(u) + 1))
    with pytest.raises(MethodIncompatibility, match="all-zero"):
        fitted.policy_effect(lambda u: np.zeros_like(u))
    with pytest.raises(NumericalInstability, match="NaN or infinite"):
        fitted.policy_effect(lambda u: np.array([1.0, np.inf]))
