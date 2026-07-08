"""Smoke test for ``sp.recommend_benchmark`` (added 2026-07-08).

Locks in:
* the public call is importable from the package root,
* the bundled corpus is discoverable from a source checkout,
* the recommend-only pass (``fit=False``) returns the documented top-level
  keys and a non-empty scorecard with hit-rate metrics in [0, 1].

Full audit-dynamic and citation-benchmarks paths are intentionally NOT
exercised here — those run end-to-end in
``tests/benchmarks/test_recommend_hit_rate.py`` and the workflow's
benchmarks job. This file exists only to satisfy the public-API-has-tests
ratchet on the new ``sp.recommend_benchmark`` symbol.
"""

from __future__ import annotations

import statspai as sp


def test_recommend_benchmark_is_public_callable():
    """sp.recommend_benchmark must be a registered public callable."""
    assert callable(sp.recommend_benchmark)


def test_recommend_benchmark_recommend_only_returns_documented_shape():
    """``fit=False`` should return a JSON-safe dict with the documented keys."""
    card = sp.recommend_benchmark(fit=False)
    assert isinstance(card, dict)
    assert "summary" in card
    assert "recommend" in card
    assert isinstance(card["summary"], dict)
    assert isinstance(card["recommend"], list)

    summary = card["summary"]
    # hit_rate_top1, hit_rate_topk and hard_miss_rate are the headline metrics
    # documented in the scorecard; all must lie in [0, 1] when present.
    for key in ("hit_rate_top1", "hit_rate_topk", "hard_miss_rate"):
        if key in summary and summary[key] is not None:
            assert 0.0 <= summary[key] <= 1.0, key
    assert summary["n_errors"] == 0

    rows = card["recommend"]
    assert len(rows) > 0
    for row in rows:
        # Each row carries at minimum the entry id, the design label, and
        # the hit/miss status of the top-1 recommendation.
        assert "id" in row
        assert "design" in row
        assert "status" in row
        assert "hit_top1" in row
        assert isinstance(row["hit_top1"], bool)
