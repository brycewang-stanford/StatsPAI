"""The registry's ``example=`` one-liners must not hand agents a dead call.

``sp.describe_function(name)["example"]`` is frequently the only call
syntax an agent sees before dispatching. Unlike docstring ``Examples``
blocks — executed by ``scripts/check_example_execution.py`` — these
reference undefined names by design and can only be checked structurally.

The budget is zero: the sweep is clean today, so any new finding is a
regression introduced with the change that caused it.
"""

import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from registry_example_audit import audit_one, collect  # noqa: E402


def test_no_registry_example_is_structurally_broken():
    findings = collect()
    assert not findings, (
        "registry example(s) that cannot work as written:\n"
        + "\n".join(
            f"  {name}: [{kind}] {detail}"
            for name, items in sorted(findings.items())
            for kind, detail in items
        )
        + "\nFix the example= string in registry.py — an agent reading "
        "describe_function() will dispatch it verbatim."
    )


class TestAuditActuallyDetects:
    """A gate that cannot fail is not a gate."""

    def test_catches_syntax_error(self):
        assert audit_one("t", "sp.regress('y ~ x', data=df")[0][0] == "syntax"

    def test_catches_unknown_callee(self):
        assert audit_one("t", "sp.no_such_function(df)")[0][0] == "unknown-callee"

    def test_catches_unknown_keyword_on_closed_signature(self):
        # sp.ipw takes treat=/covariates=, not sp.psm's d=/X=.
        findings = audit_one("t", "sp.ipw(df, y='y', d='d', covariates=['x'])")
        assert findings and findings[0][0] == "unknown-keyword"

    def test_catches_unknown_keyword_behind_accepts_aliases(self):
        # sp.regress has **kwargs, but only to accept vce= for robust=.
        findings = audit_one("t", "sp.regress('y ~ x', data=df, vcetype='hc1')")
        assert findings and findings[0][0] == "unknown-keyword"

    @pytest.mark.parametrize(
        "example",
        [
            "sp.ipw(df, y='y', treat='d', covariates=['x'])",
            "sp.regress('y ~ x', data=df, vce='hc1')",  # registered alias
            "sp.psm(df, y='y', d='d', X=['x'], caliper=0.2)",  # kwargs delegate
            "sp.regress('y ~ x', data=df).summary()",  # method call, out of scope
        ],
    )
    def test_accepts_valid_examples(self, example):
        assert audit_one("t", example) == []
