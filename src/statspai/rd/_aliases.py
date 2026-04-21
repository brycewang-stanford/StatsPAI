"""User-friendly RD aliases (Chinese-market & agent discoverability).

These thin wrappers re-export the canonical estimators under names the v3
methodology document uses, so that ``sp.geographic_rd(...)`` and
``sp.multi_cutoff_rd(...)`` work out-of-the-box alongside the R/Stata-style
``sp.rdms`` and ``sp.rdmc``.
"""

from __future__ import annotations

from .rdmulti import rdmc, rdms, RDMultiResult
from .rd2d import rd2d
from .multi_score import rd_multi_score, MultiScoreRDResult


__all__ = [
    "multi_cutoff_rd",
    "geographic_rd",
    "boundary_rd",
    "multi_score_rd",
]


def multi_cutoff_rd(*args, **kwargs):
    """User-friendly alias for :func:`sp.rdmc` (multi-cutoff RD).

    See :func:`statspai.rd.rdmulti.rdmc` for full documentation.
    """
    return rdmc(*args, **kwargs)


def geographic_rd(*args, **kwargs):
    """User-friendly alias for :func:`sp.rdms` (multi-score RD).

    Geographic RD is the most common multi-score special case: the running
    score is a signed distance to a political boundary.  Cattaneo et al.
    (2024) dispatch it via ``rdms``; this alias makes the intent explicit.
    """
    return rdms(*args, **kwargs)


def boundary_rd(*args, **kwargs):
    """User-friendly alias for :func:`sp.rd2d` (boundary discontinuity design).

    Cattaneo, Titiunik & Yu (2025) boundary discontinuity design for 2D
    running variables (lat/long, for example).
    """
    return rd2d(*args, **kwargs)


def multi_score_rd(*args, **kwargs):
    """User-friendly alias for :func:`sp.rd_multi_score`.

    Multi-score RD when eligibility depends on more than one discontinuous
    rule (e.g. income AND age thresholds).  Separate from boundary RDD in
    that the rules are axis-aligned.
    """
    return rd_multi_score(*args, **kwargs)
