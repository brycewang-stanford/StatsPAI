"""Did-you-mean for keyword arguments points at the parameter meant.

String similarity alone offered ``se_method`` for ``seed`` and nothing for
``clustr`` behind ``reject_unknown_kwargs``.
"""

from __future__ import annotations

import warnings

import pytest

import statspai as sp
from statspai._kwarg_suggest import signature_keywords, suggest_keywords


def test_synonyms_come_before_string_similarity():
    assert suggest_keywords("seed", ["se_method", "random_state", "alpha"]) == [
        "random_state",
        "se_method",
    ]
    assert suggest_keywords("cluster_var", ["cluster", "alpha"]) == ["cluster"]
    # Only names the function takes are ever offered.
    assert suggest_keywords("seed", ["alpha", "n_folds"]) == []


def test_signature_keywords_reads_call_shaped_names():
    names = signature_keywords("synth(method='classic')")
    assert names is not None and "outcome" in names
    assert signature_keywords("no_such_function") is None


def test_regress_suggests_cluster_for_clustr():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nsw = sp.datasets.nsw_dw()
    with pytest.raises(TypeError, match=r"'clustr' \(did you mean 'cluster'\?\)"):
        sp.regress("re78 ~ treat", data=nsw, clustr="age")


def test_aliases_path_puts_the_synonym_first():
    with pytest.raises(TypeError, match=r"'seed' \(did you mean 'random_state'"):
        sp.rate_split(None, seed=1)


def test_call_shaped_function_names_are_not_doubled():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ca = sp.datasets.california_prop99()
    with pytest.raises(TypeError) as ei:
        sp.synth(
            ca,
            outcome="cigsale",
            unit="state",
            time="year",
            treated_unit="California",
            treatment_time=1989,
            bogus=1,
        )
    assert ")()" not in str(ei.value)
