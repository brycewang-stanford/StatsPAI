"""Discovery and wording items of the 2026-11 gap list (experiments / ML).

* ``sp.search_functions`` finds the sequential tests, the off-policy value
  estimators, the GATES / BLP / CLAN tests, the uplift curve and the new
  sample-ratio-mismatch test by the words a practitioner types.
* ``sp.metalearner`` no longer reports the ignored ``n_bootstrap``.
* ``sp.rate`` and the ``sp.ope`` estimators name callable things.
"""

from __future__ import annotations

import inspect
import re

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _names(query: str, k: int) -> list:
    return [hit["name"] for hit in sp.search_functions(query)][:k]


@pytest.mark.parametrize(
    "query, wanted, k",
    [
        ("sequential test", {"msprt", "group_sequential"}, 2),
        ("sequential testing", {"msprt", "group_sequential"}, 2),
        ("off-policy evaluation", {"ips", "snips", "policy_value"}, 4),
        ("off policy evaluation", {"ips", "snips", "policy_value"}, 4),
        ("GATES BLP CLAN", {"gate_test", "blp_test"}, 2),
        ("uplift qini auuc", {"cate_gain_curve"}, 2),
        ("sample ratio mismatch", {"srm_test"}, 1),
        ("SRM", {"srm_test"}, 1),
    ],
)
def test_search_surfaces_the_expected_functions(query, wanted, k):
    top = _names(query, k)
    assert wanted <= set(top), (query, top)


def test_search_keeps_the_exact_name_first():
    """The new concept words do not displace a function named by the query."""
    assert _names("blp", 1) == ["blp"]
    assert _names("rate", 1) == ["rate"]
    assert _names("sample size", 1) == ["power"]


def test_srm_test_is_registered_with_its_parameters():
    card = sp.describe_function("srm_test")
    assert [p["name"] for p in card["params"]] == list(
        inspect.signature(sp.srm_test).parameters
    )
    schema = sp.function_schema("srm_test")
    assert set(schema["parameters"]["properties"]) >= {"data", "treat", "expected"}


def test_metalearner_does_not_report_the_ignored_bootstrap_count():
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=(n, 2))
    d = rng.integers(0, 2, size=n)
    y = x[:, 0] + d * (1 + x[:, 1]) + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "d": d, "x1": x[:, 0], "x2": x[:, 1]})
    fit = sp.metalearner(df, y="y", treat="d", covariates=["x1", "x2"], learner="t")
    assert "n_bootstrap" not in fit.model_info
    assert not re.search(r"bootstrap", str(fit.summary()), flags=re.IGNORECASE)


def test_rate_docstring_names_callable_things():
    doc = inspect.getdoc(sp.rate)
    assert "sp.get_scores" in doc and callable(sp.get_scores)
    assert "sp.rate_split" in doc and callable(sp.rate_split)
    from statspai.forest.forest_inference import rate_from_scores

    assert "statspai.forest.forest_inference.rate_from_scores" in doc
    assert callable(rate_from_scores)
    # the two names that are not top-level functions are no longer cross-refs
    assert ":func:`aipw_scores`" not in doc
    assert ":func:`rate_from_scores`" not in doc


@pytest.mark.parametrize("name", ["direct_method", "doubly_robust", "switch_dr"])
def test_ope_docstrings_state_the_reward_model_contract(name):
    doc = inspect.getdoc(getattr(sp.ope, name))
    assert "reward_model(X, a)" in doc
    assert "length-``n``" in doc and "per row" in doc


def test_ope_reward_model_contract_is_what_the_code_calls():
    """The documented contract (int action and per-row array) is the real one."""
    rng = np.random.default_rng(1)
    n, k = 200, 3
    X = rng.normal(size=(n, 2))
    pi_b = np.full((n, k), 1.0 / k)
    pi_e = np.tile([0.5, 0.3, 0.2], (n, 1))
    a = rng.integers(0, k, size=n)
    r = X[:, 0] + a + rng.normal(size=n)
    seen = []

    def reward_model(X_, act):
        seen.append(np.ndim(act))
        return X_[:, 0] + np.broadcast_to(act, len(X_))

    res = sp.ope.doubly_robust(X, a, r, pi_b, pi_e, reward_model)
    assert seen == [0, 0, 0, 1]
    # with the true reward model the DR value is the DM value plus a
    # mean-zero correction: check the arithmetic it documents
    dm = sp.ope.direct_method(reward_model, X, pi_e).value
    rho = pi_e[np.arange(n), a] / pi_b[np.arange(n), a]
    assert res.value == pytest.approx(
        dm + np.mean(rho * (r - (X[:, 0] + a))), rel=1e-12
    )
