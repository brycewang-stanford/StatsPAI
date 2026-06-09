"""``EconometricResults.cite()`` — citation parity with ``CausalResult``.

Agent-native uniformity: a fitted *regression* result should answer ``.cite()``
just like a causal result, and ``sp.bib_for(result)`` (which duck-types on
``.cite``) should work for it. Zero-hallucination (CLAUDE.md §10): textbook
estimators with no canonical paper return a placeholder, never a fabricated
entry; registered methods resolve to the verified ``CausalResult._CITATIONS``
table.

(Note: ``sp.cite(result, "x")`` — the *inline coefficient reporter* — is a
different, unrelated function; this module exercises the bound ``.cite()``
citation method.)
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core.results import EconometricResults, CausalResult


@pytest.fixture
def ols_result():
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=n)
    y = 1.0 + 0.5 * x + rng.normal(size=n)
    return sp.regress("y ~ x", data=pd.DataFrame({"y": y, "x": x}))


def test_regression_returns_econometric_results(ols_result):
    assert isinstance(ols_result, EconometricResults)
    assert hasattr(ols_result, "cite") and callable(ols_result.cite)


def test_textbook_estimator_returns_placeholder_not_fabrication(ols_result):
    """OLS has no canonical paper — must NOT invent one (§10)."""
    bib = ols_result.cite()
    assert isinstance(bib, str)
    assert bib.startswith("% No citation registered")
    # json form is structured and explicitly flags the absence
    payload = ols_result.cite(format="json")
    assert payload["key"] is None
    assert payload["note"] == "no citation registered"


def test_registered_citation_key_resolves_to_verified_entry(ols_result):
    """A model carrying a registered citation_key resolves from _CITATIONS."""
    ols_result.model_info["citation_key"] = "tobit"
    bibtex = ols_result.cite()
    assert bibtex == CausalResult._CITATIONS["tobit"]
    # APA / JSON are *derived* from that single source, never generated.
    apa = ols_result.cite(format="apa")
    assert "Tobin" in apa and "1958" in apa
    assert ols_result.cite(format="json")["key"] == "tobin1958estimation"


def test_invalid_format_raises(ols_result):
    with pytest.raises(ValueError, match="format must be"):
        ols_result.cite(format="xml")


def test_bib_for_now_works_for_regression(ols_result):
    """sp.bib_for duck-types on .cite — previously raised for regressions."""
    payload = sp.bib_for(ols_result)
    assert isinstance(payload, dict)
    assert "key" in payload  # None for unregistered OLS, but no exception


def test_unknown_model_type_does_not_false_match(ols_result):
    """Exact-only resolution: an arbitrary model_type never grabs a wrong paper."""
    ols_result.model_info["citation_key"] = None
    ols_result.model_info["model_type"] = "SomeNovelEstimator"
    assert ols_result.cite().startswith("% No citation registered")
