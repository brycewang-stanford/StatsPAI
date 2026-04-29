"""Aggregate every per-family spec list into the single ``TOOL_REGISTRY``.

The hand-curated tools are intentionally split by causal-inference
family (regression / DID / IV / RD / matching / diagnostics /
orchestrators) so each file stays small and reviewable. Adding a new
hand-curated tool: drop a dict into the matching family file (or
create a new family file + add it here).
"""
from __future__ import annotations

from typing import Any, Dict, List

from . import (
    _regression,
    _did,
    _iv,
    _rd,
    _matching,
    _diag,
    _orchestrate,
)


# Order matters only insofar as ``tool_manifest`` returns this list
# verbatim and a few tests depend on the regression entry being first.
# Family ordering otherwise mirrors the canonical estimator categories
# in CLAUDE.md §2.
TOOL_REGISTRY: List[Dict[str, Any]] = (
    _regression.SPECS
    + _did.SPECS
    + _iv.SPECS
    + _rd.SPECS
    + _matching.SPECS
    + _diag.SPECS
    + _orchestrate.SPECS
)


__all__ = ["TOOL_REGISTRY"]
