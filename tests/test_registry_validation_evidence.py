"""Guard: the registry's test-evidence ledger must point at real evidence.

``_VALIDATED_TEST_SEED_FUNCTIONS`` is surfaced verbatim through
``sp.describe_function(name)['validation_notes']`` as
``"API/unit contract evidence: <path>"``.  An agent (or a referee) reading
that note treats the path as the place the estimator is exercised, so a
stale path is worse than no note at all -- it asserts coverage that does
not exist.

Ten entries had rotted this way: ``stacked_did`` cited
``tests/test_did.py``, which contains no stacked-DiD test at all, and
``cic`` / ``ddd`` / ``did_analysis`` / ``pretrends_test`` /
``breslow_day_test`` / ``mr_clust`` / ``target_trial_protocol`` cited
files that had been split or renamed out from under them.
"""

from __future__ import annotations

import pathlib

import pytest

from statspai.registry import _VALIDATED_TEST_SEED_FUNCTIONS

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_ledger_is_populated():
    """An empty ledger would satisfy every per-entry check vacuously."""
    assert len(_VALIDATED_TEST_SEED_FUNCTIONS) > 100


@pytest.mark.parametrize(
    "function, test_path",
    sorted(
        (fn, path)
        for fn, paths in _VALIDATED_TEST_SEED_FUNCTIONS.items()
        for path in paths
    ),
)
def test_cited_test_file_exists_and_exercises_the_function(function, test_path):
    path = _ROOT / test_path
    assert path.exists(), f"{function}: cited evidence {test_path} does not exist"
    text = path.read_text(encoding="utf-8", errors="replace")
    assert function in text, (
        f"{function}: {test_path} never mentions {function!r}; "
        "the citation is stale -- point it at a file that exercises it"
    )
