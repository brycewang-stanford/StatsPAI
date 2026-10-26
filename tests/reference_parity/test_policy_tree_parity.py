"""Analytical parity: sp.policy_tree oracle recovery + exact value accounting.

Doubly-robust policy tree (Athey & Wager 2021 framework). On a randomized DGP
with tau(x) = x, the oracle rule is 1{x > 0} with population value
E[x 1{x > 0}] = 0.25 for x ~ Uniform(-1, 1). The learned depth-2 tree recovers
the oracle rule and its value. The reported value accounting also satisfies
exact identities against the returned scores and policy:

    value_policy     == policy_value(scores, policy) == mean(scores * policy)
    value_treat_all  == mean(scores)
    value_treat_none == 0
    value_gain       == value_policy - max(value_treat_all, value_treat_none)
    fraction_treated == mean(policy)

Analytical evidence tier (known-truth recovery; the tree search itself is
greedy, so structure recovery is approximate while the accounting is exact).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

ORACLE_VALUE = 0.25  # E[x 1{x>0}] for x ~ U(-1, 1)


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.uniform(-1, 1, n)
    d = rng.integers(0, 2, n)
    y = x * d + 0.5 * x + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"y": y, "d": d, "x": x})
    # The identities below are those of the in-sample objective.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return x, sp.policy_tree(
            df, y="y", d="d", X=["x"], depth=2, value_evaluation="in_sample"
        )


def test_value_accounting_identities_exact(fitted):
    _, r = fitted
    s = np.asarray(r["scores"], dtype=float)
    p = np.asarray(r["policy"], dtype=float)
    assert r["value_policy"] == pytest.approx(sp.policy_value(s, p), abs=1e-12)
    assert r["value_policy"] == pytest.approx(float(np.mean(s * p)), abs=1e-12)
    assert r["value_treat_all"] == pytest.approx(float(s.mean()), abs=1e-12)
    assert r["value_treat_none"] == 0.0
    expected_gain = r["value_policy"] - max(r["value_treat_all"], r["value_treat_none"])
    assert r["value_gain"] == pytest.approx(expected_gain, abs=1e-12)
    assert r["fraction_treated"] == pytest.approx(float(p.mean()), abs=1e-12)


def test_recovers_oracle_threshold_rule(fitted):
    x, r = fitted
    p = np.asarray(r["policy"], dtype=int)
    oracle = (x > 0).astype(int)
    assert float((p == oracle).mean()) > 0.95
    assert r["value_policy"] == pytest.approx(ORACLE_VALUE, abs=0.05)
    assert r["fraction_treated"] == pytest.approx(0.5, abs=0.05)
