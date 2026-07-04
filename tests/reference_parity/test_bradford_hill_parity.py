"""Analytical parity: sp.bradford_hill total-score + verdict boundaries.

Bradford-Hill viewpoints are scored in [0, 1] and summed; the verdict
thresholds (STRONG / MODERATE / WEAK / INSUFFICIENT) partition the [0, 1]
range of the average score. Analytical evidence tier (deterministic
arithmetic; the qualitative verdict is invariant to non-vanishing noise).
"""

from __future__ import annotations

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

# All 9 viewpoints at the maximum (1.0) -> strongest possible support.
ALL_MAX = dict.fromkeys(
    (
        "strength",
        "consistency",
        "specificity",
        "temporality",
        "biological_gradient",
        "plausibility",
        "coherence",
        "experiment",
        "analogy",
    ),
    1.0,
)


def test_maximum_evidence_total_equals_nine():
    r = sp.bradford_hill(evidence=ALL_MAX)
    assert float(r.total) == 9.0
    assert float(r.max_total) == 9.0
    assert r.total == r.max_total


def test_zero_evidence_is_insufficient():
    # Zero scores are insufficient *and* the temporality prerequisite fails
    # (so the verdict mentions both reasons).
    r = sp.bradford_hill(evidence={k: 0.0 for k in ALL_MAX})
    assert float(r.total) == 0.0
    assert "INSUFFICIENT" in r.verdict
    assert "temporality" in r.missing_prerequisites


def test_homogeneous_evidence_total_equals_score_times_nine():
    # With every viewpoint at the same score s, the total must be 9s.
    for s in (0.0, 0.25, 0.5, 0.75, 1.0):
        r = sp.bradford_hill(evidence={k: s for k in ALL_MAX})
        assert float(r.total) == 9.0 * s
        assert 0.0 <= r.total <= r.max_total


def test_missing_viewpoints_decrement_total():
    # Omit the dropped viewpoint entirely; it is counted as unassessed.
    full = sp.bradford_hill(evidence=ALL_MAX).total
    partial = {k: v for k, v in ALL_MAX.items() if k != "specificity"}
    r = sp.bradford_hill(evidence=partial)
    assert r.total == full - 1.0
    # Dropped viewpoint is recorded with NaN score; assessed set size = 8.
    assert r.max_total == full - 1.0
