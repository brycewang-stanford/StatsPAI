"""Unit tests for the keyword-alias decorator (grammar convergence plumbing)."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from statspai._aliases import accepts_aliases  # noqa: E402


def test_canonical_alias_forwarded() -> None:
    @accepts_aliases(vce="robust")
    def fit(formula, data=None, robust="nonrobust"):
        return robust

    assert fit("y ~ x", vce="hc1") == "hc1"
    assert fit("y ~ x", robust="hc2") == "hc2"
    assert fit("y ~ x") == "nonrobust"


def test_multiple_aliases() -> None:
    @accepts_aliases(vce="robust", outcome="y", treatment="treat")
    def fit(y=None, treat=None, robust=None):
        return (y, treat, robust)

    assert fit(outcome="wage", treatment="union", vce="cluster") == (
        "wage",
        "union",
        "cluster",
    )


def test_conflict_raises() -> None:
    @accepts_aliases(vce="robust")
    def fit(robust=None):
        return robust

    with pytest.raises(TypeError, match="both 'vce' and its canonical target"):
        fit(vce="hc1", robust="hc2")


def test_signature_preserved_for_introspection() -> None:
    @accepts_aliases(vce="robust")
    def fit(formula, data=None, robust="nonrobust"):
        return robust

    import inspect

    params = list(inspect.signature(fit).parameters)
    assert params == ["formula", "data", "robust"]
    assert fit.__statspai_aliases__ == {"vce": "robust"}


def test_warn_off_by_default() -> None:
    @accepts_aliases(vce="robust")
    def fit(robust=None):
        return robust

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would raise
        assert fit(vce="hc1") == "hc1"


def test_warn_opt_in() -> None:
    @accepts_aliases(_warn=True, vce="robust")
    def fit(robust=None):
        return robust

    with pytest.warns(DeprecationWarning):
        fit(vce="hc1")


def test_stacked_decorators_merge_alias_record() -> None:
    @accepts_aliases(vce="robust")
    @accepts_aliases(outcome="y")
    def fit(y=None, robust=None):
        return (y, robust)

    assert fit.__statspai_aliases__ == {"vce": "robust", "outcome": "y"}
    assert fit(outcome="wage", vce="hc1") == ("wage", "hc1")


def test_empty_map_rejected() -> None:
    with pytest.raises(ValueError):
        accepts_aliases()
