"""``RecommendationResult.run_all`` must not fail silently.

``run_all`` deliberately keeps going when one recommended estimator
raises — losing the whole comparison because one specification was
inapplicable would be worse. But the failure used to leave *only* an
``"Error: ..."`` string in the returned dict: no warning, nothing
machine-readable. A run in which every recommendation failed was
indistinguishable from a clean one until the caller tried to use a
result and got an ``AttributeError`` on a ``str``.

CLAUDE.md §7 requires these best-effort paths in ``smart/`` to record the
degradation, which warns and appends a structured entry.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.workflow._degradation import WorkflowDegradedWarning


@pytest.fixture(scope="module")
def absorbed_panel():
    """Treatment is time-invariant within unit, so unit FE absorbs it.

    A legitimate specification failure — the point here is how it is
    reported, not that it happens.
    """
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame(
        {"id": np.repeat(np.arange(n // 2), 2), "time": np.tile([0, 1], n // 2)}
    )
    df["treat"] = (df["id"] % 2 == 0).astype(int)
    df["post"] = df["time"]
    df["y"] = 1.0 + 0.5 * df["treat"] * df["post"] + rng.normal(0, 1, len(df))
    return df


def _recommend(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.recommend(df, y="y", treatment="treat", time="post", id="id")


def test_failed_estimator_warns_and_is_recorded(absorbed_panel):
    rec = _recommend(absorbed_panel)
    with pytest.warns(WorkflowDegradedWarning):
        results = rec.run_all()

    failed = [k for k, v in results.items() if isinstance(v, str)]
    assert failed, "this fixture is meant to produce at least one failure"

    assert rec.degradations, "a failed estimator must reach .degradations"
    assert len(rec.degradations) == len(failed)
    for entry in rec.degradations:
        assert entry["section"].startswith("run_all: ")
        assert entry["error_type"]
        assert entry["message"]


def test_every_failure_is_individually_accounted_for(absorbed_panel):
    """The count must match, so a swallowed failure cannot hide."""
    rec = _recommend(absorbed_panel)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = rec.run_all()
    degraded = [w for w in caught if issubclass(w.category, WorkflowDegradedWarning)]
    failed = [k for k, v in results.items() if isinstance(v, str)]
    assert len(degraded) == len(failed)
    recorded = {d["section"].removeprefix("run_all: ") for d in rec.degradations}
    assert recorded == set(failed)


def test_return_shape_is_unchanged(absorbed_panel):
    """Failures still map to 'Error: ...' strings — this adds a channel."""
    rec = _recommend(absorbed_panel)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = rec.run_all()
    assert isinstance(results, dict)
    for value in results.values():
        if isinstance(value, str):
            assert value.startswith("Error: ")


def test_clean_run_records_nothing(absorbed_panel):
    """A recommendation set that runs must leave degradations empty."""
    rng = np.random.default_rng(1)
    n = 400
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "treat": rng.integers(0, 2, n).astype(float),
        }
    )
    df["y"] = 1.0 + 0.5 * df["treat"] + 0.3 * df["x"] + rng.normal(size=n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = sp.recommend(df, y="y", treatment="treat", covariates=["x"])
        results = rec.run_all()
    if any(isinstance(v, str) for v in results.values()):
        pytest.skip("this design also fails here; covered by the tests above")
    assert rec.degradations == []
