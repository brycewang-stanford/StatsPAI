"""Execute the runnable ``Examples`` doctests on high-traffic dispatchers.

These guarantee the docstring examples actually run (and produce the documented
result) — a "doctest-backed" example, so the agent/human copy-paste path stays
green. The examples are written with a stable final check (``isinstance`` /
``bool(...)``) so the output does not drift with estimates or numpy repr.
"""

import doctest
import warnings

import pytest

from statspai._article_aliases import rdd, xlearner
from statspai.matching.ps_diagnostics import propensity_score
from statspai.regression.iv import ivreg


@pytest.mark.parametrize(
    "fn", [rdd, xlearner, propensity_score, ivreg],
    ids=lambda f: f.__name__,
)
def test_docstring_example_runs(fn):
    finder = doctest.DocTestFinder(verbose=False)
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS, verbose=False)
    tests = finder.find(fn, name=fn.__name__, globs={})
    doctests = [t for t in tests if t.examples]
    assert doctests, f"{fn.__name__} has no runnable doctest example"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in doctests:
            runner.run(t)
    result = runner.summarize(verbose=False)
    assert result.failed == 0, (
        f"{fn.__name__}: {result.failed} doctest failure(s)"
    )
