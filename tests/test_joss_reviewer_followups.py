"""Backward-compatible JOSS reviewer follow-up test path.

The focused tests were renamed to ``test_external_reviewer_followups.py`` when
the JSS source-snapshot work separated journal-specific file names from the
package release gate.  The JOSS review thread still references this original
path, so keep it importable and runnable for reviewers copying that command.
"""

import importlib.util
from pathlib import Path

_MODULE_PATH = Path(__file__).with_name("test_external_reviewer_followups.py")
_SPEC = importlib.util.spec_from_file_location(
    "_statspai_external_reviewer_followups",
    _MODULE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

TestAdvancedPostEstimationMargins = _MODULE.TestAdvancedPostEstimationMargins
TestDAGReasoningHelpers = _MODULE.TestDAGReasoningHelpers
TestEconometricResultsPredict = _MODULE.TestEconometricResultsPredict

__all__ = [
    "TestAdvancedPostEstimationMargins",
    "TestDAGReasoningHelpers",
    "TestEconometricResultsPredict",
]
