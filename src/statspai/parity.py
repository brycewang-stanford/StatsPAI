"""Queryable cross-language parity status for every StatsPAI function.

This module turns StatsPAI's parity evidence from prose scattered across
three test subsystems into a **first-class, queryable property**: for any
public function you can ask *what was it aligned against, to what
tolerance, on which test, and how closely did it match?*

    >>> import statspai as sp
    >>> sp.parity_status("feols")["status"]
    'bit-exact'

The data is a frozen snapshot (``_parity_index.json``) regenerated from
committed parity artifacts by ``scripts/build_parity_index.py`` and
drift-checked in CI, so it works identically from a source checkout and an
installed wheel, and can never silently drift from the underlying tests.

Status taxonomy
---------------
``bit-exact``
    Matches a named R/Stata reference to the machine tolerance tier
    (headline relative error ``<= 1e-6``).
``aligned``
    Matches a named R/Stata reference within a documented, pre-registered
    looser tolerance (iterative / moderate / methodological tier — e.g. a
    cross-fit estimator, or a documented convention disagreement).
``analytical-only``
    Recovers a known population parameter on a deterministic DGP, or a
    closed-form identity — verified, but with no cross-package reference.
``external-replication``
    Reproduces published paper numbers on a calibrated replica.
``unverified``
    Registered public API with no qualifying numerical-parity evidence
    attached yet. Honestly surfaced — *this is the auditable gap map.*
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._parity_taxonomy import (
    CROSS_LANGUAGE_STATUSES,
    INFRASTRUCTURE_CATEGORIES,
    INTERNAL_EVIDENCE_STATUSES,
)

_SNAPSHOT = Path(__file__).resolve().parent / "_parity_index.json"

TAXONOMY = (
    "bit-exact",
    "aligned",
    "analytical-only",
    "external-replication",
    "unverified",
)

#: Ranking from strongest to weakest evidence (for sorting / matrix order).
_GRADE_RANK = {name: i for i, name in enumerate(TAXONOMY)}


@lru_cache(maxsize=1)
def _load_index() -> Dict[str, Any]:
    """Load the frozen parity snapshot (empty skeleton if absent)."""
    if not _SNAPSHOT.exists():  # pragma: no cover - packaging guard
        return {"schema_version": 1, "taxonomy": list(TAXONOMY), "records": []}
    try:
        return json.loads(_SNAPSHOT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive
        return {"schema_version": 1, "taxonomy": list(TAXONOMY), "records": []}


@lru_cache(maxsize=1)
def _records_by_function() -> Dict[str, Dict[str, Any]]:
    return {rec["function"]: rec for rec in _load_index().get("records", [])}


def _registered_functions() -> List[str]:
    """Full public surface; falls back to the index keys off-tree."""
    try:
        import statspai as sp

        return list(sp.list_functions())
    except (AttributeError, ImportError):  # pragma: no cover - registry unavailable
        return list(_records_by_function())


class ParityStatus(dict):
    """Dict-compatible parity record with a human-readable rendering.

    Behaves as an ordinary ``dict`` for agents/JSON, but prints a compact
    summary for humans at the REPL and in notebooks.

    Examples
    --------
    >>> import statspai as sp
    >>> status = sp.parity_status("regress")
    >>> isinstance(status, dict)          # plain dict for agents / JSON
    True
    >>> "function" in status
    True
    >>> isinstance(status.summary(), str)  # one-line human rendering
    True
    """

    def summary(self) -> str:
        """One line a reader cannot mistake for a stronger claim.

        Cross-language rows name the reference implementation and the
        observed deviation. Rows with no external software reference say so
        in words instead of rendering an empty ``vs`` and a headline of
        ``n/a``, which read like a comparison that had been attempted and
        failed rather than one that was never possible.
        """
        fn = self.get("function", "?")
        status = self.get("status", "unverified")
        if status == "unverified":
            return f"{fn}: unverified — no numerical evidence attached yet."

        ref = str(self.get("reference", "") or "").strip()
        tol = str(self.get("tolerance", "") or "").strip()
        sides = "/".join(self.get("sides", []) or [])

        if status not in CROSS_LANGUAGE_STATUSES:
            kind = (
                "known-truth recovery on a deterministic DGP"
                if status == "analytical-only"
                else "published-reference replication"
            )
            detail = f" vs {ref}" if ref else ""
            tail = f"; tolerance {tol}" if tol else ""
            return (
                f"{fn}: {status} — {kind}, no external software reference"
                f"{detail}{tail}."
            )

        head = self.get("headline", {}) or {}
        rels = [head.get("rel_vs_R"), head.get("rel_vs_Stata")]
        worst = max([r for r in rels if isinstance(r, (int, float))], default=None)
        worst_s = f"{worst:.1e}" if isinstance(worst, (int, float)) else "n/a"
        return (
            f"{fn}: {status} vs {ref} [{sides}] "
            f"(headline {head.get('metric', 'rel')} {worst_s} within {tol})"
        )

    def __repr__(self) -> str:
        return self.summary()

    def _repr_html_(self) -> str:  # pragma: no cover - notebook nicety
        rows = "".join(
            f"<tr><td><b>{k}</b></td><td><code>{v}</code></td></tr>"
            for k, v in self.items()
        )
        return f"<table>{rows}</table>"


def parity_status(name: str) -> ParityStatus:
    """Return the cross-language parity record for one function.

    Parameters
    ----------
    name : str
        A registered ``sp.*`` function name (e.g. ``"feols"``).

    Returns
    -------
    ParityStatus
        Dict-compatible record. For a verified function it carries
        ``status`` / ``reference`` / ``reference_versions`` / ``tolerance``
        / ``headline`` (statistic, metric, relative error vs R & Stata) /
        ``test`` / ``module_id``. For a registered-but-unverified function
        it returns ``{"function": name, "status": "unverified", ...}``.

    Raises
    ------
    KeyError
        If ``name`` is not a registered StatsPAI function.

    Examples
    --------
    >>> import statspai as sp
    >>> rec = sp.parity_status("regress")
    >>> rec["status"]
    'bit-exact'
    >>> rec["reference"]
    'lm + sandwich::vcovHC'
    >>> sp.parity_status("nonexistent_fn_xyz")  # doctest: +SKIP
    KeyError: ...
    """
    records = _records_by_function()
    if name in records:
        return ParityStatus(records[name])

    registered = set(_registered_functions())
    if name not in registered and registered:
        raise KeyError(
            f"'{name}' is not a registered StatsPAI function "
            f"(see sp.list_functions())."
        )
    return ParityStatus(
        {
            "function": name,
            "status": "unverified",
            "source": "none",
            "reference": "",
            "reference_versions": {},
            "tolerance": "",
            "sides": [],
            "headline": {},
            "test": [],
            "notes": [
                "No cross-language (R/Stata) or published-reference parity "
                "evidence is attached to this function yet."
            ],
        }
    )


def parity_matrix(
    *,
    status: Optional[str] = None,
    source: Optional[str] = None,
    fmt: str = "records",
) -> Any:
    """Return the full parity matrix over every registered function.

    This is the auditable asset map: one row per public function, each
    carrying its parity ``status`` (including ``unverified`` rows, which are
    the honest coverage gaps).

    Parameters
    ----------
    status : str, optional
        Filter to one taxonomy grade (see :data:`TAXONOMY`).
    source : str, optional
        Filter by evidence source (``"track_a"``, ``"reference_parity"``,
        ``"external_parity"``, or ``"none"``).
    fmt : {"records", "dataframe", "markdown"}, default "records"
        Output format. ``"dataframe"`` / ``"markdown"`` need pandas.

    Examples
    --------
    >>> import statspai as sp
    >>> rows = sp.parity_matrix(status="bit-exact")
    >>> all(r["status"] == "bit-exact" for r in rows)
    True
    >>> isinstance(rows, list)
    True
    """
    if status is not None and status not in TAXONOMY:
        raise ValueError(f"status={status!r} must be one of {list(TAXONOMY)} or None")

    records = _records_by_function()
    rows: List[Dict[str, Any]] = []
    for fn in sorted(_registered_functions()):
        rec = records.get(fn)
        if rec is None:
            rec = {"function": fn, "status": "unverified", "source": "none"}
        rows.append(rec)

    if status is not None:
        rows = [r for r in rows if r.get("status") == status]
    if source is not None:
        rows = [r for r in rows if r.get("source") == source]

    rows.sort(key=lambda r: (_GRADE_RANK.get(r.get("status"), 99), r["function"]))

    if fmt == "records":
        return rows
    import pandas as pd

    flat = [
        {
            "function": r["function"],
            "status": r.get("status", "unverified"),
            "source": r.get("source", "none"),
            "reference": r.get("reference", ""),
            "tolerance": r.get("tolerance", ""),
            "module_id": r.get("module_id", ""),
        }
        for r in rows
    ]
    df = pd.DataFrame(flat)
    if fmt == "dataframe":
        return df
    if fmt == "markdown":
        return df.to_markdown(index=False)
    raise ValueError("fmt must be one of 'records', 'dataframe', or 'markdown'")


def _denominator_strata(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """Split the registered surface into parity-applicable strata.

    ``verified / total_functions`` over the whole registry is not a coverage
    metric anyone should quote: about a quarter of the registered surface is
    result and exception classes, which can never carry a parity grade, and
    another sixth is infrastructure that renders tables, draws plots, builds
    agent schemas or loads data. Reporting the strata keeps the estimator
    fraction — the number worth driving release over release — visible next
    to the diluted one instead of leaving readers to recompute it.
    """
    import inspect

    try:
        import statspai as sp
        from statspai import registry as _registry
    except ImportError:  # pragma: no cover - defensive
        return {}

    infra_categories = INFRASTRUCTURE_CATEGORIES
    strata = {
        key: {"cross_language": 0, "verified": 0, "total": 0}
        for key in ("estimator", "infrastructure", "classes", "all")
    }
    for row in rows:
        name = row.get("function", "")
        status = row.get("status", "unverified")
        spec = _registry._REGISTRY.get(name)
        obj = getattr(sp, name, None)
        if inspect.isclass(obj):
            key = "classes"
        elif spec is not None and spec.category in infra_categories:
            key = "infrastructure"
        else:
            key = "estimator"
        for target in (key, "all"):
            strata[target]["total"] += 1
            if status != "unverified":
                strata[target]["verified"] += 1
            if status in CROSS_LANGUAGE_STATUSES:
                strata[target]["cross_language"] += 1
    for bucket in strata.values():
        total = bucket["total"]
        bucket["cross_language_fraction"] = (
            round(bucket["cross_language"] / total, 4) if total else 0.0
        )
    return strata


def parity_summary() -> Dict[str, Any]:
    """Headline counts of the parity matrix — the honest coverage snapshot.

    ``verified`` sums two evidence kinds that answer different questions, so
    it is reported alongside ``by_evidence_kind`` rather than on its own:
    ``cross_language`` functions were compared against a named R or Stata
    implementation, while ``internal_evidence`` functions recover a known
    population parameter or reproduce published numbers, which is the only
    evidence available for a method with no Stata/R sibling. ``denominators``
    splits the registered surface into estimators, infrastructure and result
    classes, because the all-registered fraction dilutes coverage with
    symbols that can never carry a parity grade.

    Examples
    --------
    >>> import statspai as sp
    >>> s = sp.parity_summary()
    >>> s["total_functions"] > 1000
    True
    >>> sorted(s["by_status"]) == sorted(set(s["by_status"]))
    True
    >>> s["by_status"]["bit-exact"] >= 50
    True
    >>> s["by_evidence_kind"]["cross_language"] >= 100
    True
    >>> s["denominators"]["estimator"]["total"] < s["total_functions"]
    True
    """
    rows = parity_matrix(fmt="records")
    by_status: Dict[str, int] = {}
    by_source: Dict[str, int] = {}
    for r in rows:
        by_status[r.get("status", "unverified")] = (
            by_status.get(r.get("status", "unverified"), 0) + 1
        )
        src = r.get("source", "none")
        by_source[src] = by_source.get(src, 0) + 1
    cross = sum(by_status.get(g, 0) for g in CROSS_LANGUAGE_STATUSES)
    internal = sum(by_status.get(g, 0) for g in INTERNAL_EVIDENCE_STATUSES)
    verified = cross + internal
    total = len(rows)
    return {
        "total_functions": total,
        "verified": verified,
        "unverified": by_status.get("unverified", 0),
        "verified_fraction": round(verified / total, 4) if total else 0.0,
        "by_evidence_kind": {
            "cross_language": cross,
            "internal_evidence": internal,
        },
        "denominators": _denominator_strata(rows),
        "by_status": dict(sorted(by_status.items())),
        "by_source": dict(sorted(by_source.items())),
    }
