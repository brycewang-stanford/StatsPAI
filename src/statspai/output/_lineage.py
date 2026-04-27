"""Numerical lineage / provenance for causal-inference estimates.

Every number a paper reports is the output of *some* function call on
*some* data. Stata, R, and most Python libraries lose that link the
moment the estimate hits a docx/xlsx/tex file: the reader has the value
but no machine-readable trail back to the call that produced it.

This module attaches a small ``Provenance`` record to estimator results
so downstream exporters (`sp.paper`, `sp.replication_pack`,
`sp.regtable`) can stamp each number with:

- The function/method name that produced it.
- The arguments (filtered for non-serialisable objects).
- A short SHA-256 fingerprint of the input DataFrame.
- A run id (uuid4) — disambiguates two identical calls in the same
  paper.
- StatsPAI + Python versions and a wall-clock timestamp.

The record is **additive**: we attach it as ``result._provenance``
without touching the result's public surface. Estimators that don't
opt in still work; estimators that do gain free traceability into
every downstream export.

Design choices
--------------
- **Hash the DataFrame, not serialise the params verbatim.** Hashes
  are stable, small, and safe to print. Params are filtered to
  JSON-able types before being captured (lists/dicts/scalars/strings);
  arrays/frames/Series get a fingerprint summary (shape + dtype +
  hash) instead of being serialised verbatim.
- **No external deps.** Lives in ``output/`` because it's a sibling
  of ``_repro.py`` (the version/data-hash footer); the two modules
  share the same fingerprint primitive so ``replication_pack`` and
  ``regtable``'s footer agree on what "the same data" means.
- **Lazy by default.** ``attach_provenance()`` does nothing if the
  caller passes ``enabled=False``. Free for users who don't care.

This module is the foundation that ``sp.replication_pack`` and the
Quarto emitter both build on — every downstream artifact can read
``result._provenance`` to traceably wire a number back to a call.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import sys
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Mapping, Optional


__all__ = [
    "Provenance",
    "attach_provenance",
    "get_provenance",
    "compute_data_hash",
    "format_provenance",
    "lineage_summary",
]


# Reuse the same hashing semantics as _repro.py so a paper-tables footer
# and a provenance record always agree on "same data".
MAX_HASH_ROWS = 1_000_000


def _statspai_version() -> str:
    try:
        from .. import __version__
        return str(__version__)
    except Exception:  # pragma: no cover — defensive
        return "unknown"


def _python_version() -> str:
    v = sys.version_info
    return f"{v.major}.{v.minor}.{v.micro}"


def compute_data_hash(data: Any, length: int = 12) -> Optional[str]:
    """Return a short SHA-256 fingerprint of *data*, or ``None``.

    Accepts:
    - ``pandas.DataFrame`` — order- and column-name-sensitive hash.
    - ``pandas.Series`` — hashed via ``hash_pandas_object``.
    - ``numpy.ndarray`` — bytes-hashed (shape-included).
    - bytes / bytearray — direct SHA-256.

    Anything else returns ``None`` rather than raising — provenance
    must never break the calling estimator.
    """
    if data is None:
        return None
    try:
        import pandas as pd
        import numpy as np

        if isinstance(data, pd.DataFrame):
            if len(data) > MAX_HASH_ROWS:
                return None
            # ``hash_pandas_object`` is row-value sensitive but does NOT
            # mix column names into the per-row hash. Column renames
            # change "what the dataset means" — fold names + dtypes into
            # the digest explicitly so two frames with identical values
            # but different column names hash differently.
            row_h = pd.util.hash_pandas_object(data, index=True).values.tobytes()
            schema = (
                "|".join(map(str, data.columns)) + "::"
                + "|".join(str(dt) for dt in data.dtypes)
            ).encode()
            digest = hashlib.sha256()
            digest.update(schema)
            digest.update(row_h)
            return digest.hexdigest()[:length]
        if isinstance(data, pd.Series):
            if len(data) > MAX_HASH_ROWS:
                return None
            row_h = pd.util.hash_pandas_object(data, index=True).values.tobytes()
            meta = (str(data.name) + "::" + str(data.dtype)).encode()
            digest = hashlib.sha256()
            digest.update(meta)
            digest.update(row_h)
            return digest.hexdigest()[:length]
        if isinstance(data, np.ndarray):
            if data.size > MAX_HASH_ROWS:
                return None
            # Include shape so two arrays with the same bytes but
            # different layouts hash differently.
            h = hashlib.sha256()
            h.update(str(data.shape).encode())
            h.update(str(data.dtype).encode())
            h.update(np.ascontiguousarray(data).tobytes())
            return h.hexdigest()[:length]
        if isinstance(data, (bytes, bytearray)):
            return hashlib.sha256(bytes(data)).hexdigest()[:length]
    except Exception:
        return None
    return None


def _summarise_value(v: Any) -> Any:
    """Reduce *v* to a JSON-serialisable summary.

    Strings/scalars/bools/None pass through. Lists/tuples/dicts recurse
    (capped at 50 entries). Arrays/frames/Series become a fingerprint
    dict ``{"_kind": "DataFrame", "shape": [n, k], "hash": "abc..."}``.
    Anything else becomes its repr (truncated to 200 chars).
    """
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        if len(v) > 50:
            return [_summarise_value(x) for x in list(v)[:50]] + [
                f"...(+{len(v) - 50} more)"
            ]
        return [_summarise_value(x) for x in v]
    if isinstance(v, Mapping):
        items = list(v.items())
        if len(items) > 50:
            items = items[:50] + [("...", f"(+{len(v) - 50} more)")]
        return {str(k): _summarise_value(val) for k, val in items}
    try:
        import pandas as pd
        import numpy as np

        if isinstance(v, pd.DataFrame):
            return {
                "_kind": "DataFrame",
                "shape": list(v.shape),
                "columns": list(map(str, v.columns))[:20],
                "hash": compute_data_hash(v),
            }
        if isinstance(v, pd.Series):
            return {
                "_kind": "Series",
                "name": str(v.name) if v.name is not None else None,
                "shape": [len(v)],
                "hash": compute_data_hash(v),
            }
        if isinstance(v, np.ndarray):
            return {
                "_kind": "ndarray",
                "shape": list(v.shape),
                "dtype": str(v.dtype),
                "hash": compute_data_hash(v),
            }
    except Exception:
        pass
    r = repr(v)
    return r if len(r) <= 200 else r[:197] + "..."


def _summarise_params(params: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not params:
        return {}
    return {str(k): _summarise_value(v) for k, v in params.items()}


@dataclass
class Provenance:
    """A traceable record of how a single estimate was produced.

    Attributes
    ----------
    function : str
        Fully qualified function name (e.g. ``"statspai.did.callaway_santanna"``).
    params : dict
        JSON-serialisable summary of the call arguments.
    data_hash : str or None
        12-char SHA-256 prefix of the input data, or None when the
        data was too large to hash (>1M rows) or wasn't a recognised
        type.
    data_shape : list[int] or None
        ``[n_rows, n_cols]`` of the input frame, when known.
    run_id : str
        Per-call uuid4 — disambiguates two structurally identical
        calls in the same session.
    statspai_version : str
        Package version at the time of the call.
    python_version : str
        ``"3.11.5"``-style.
    timestamp : str
        ISO-8601 wall-clock of when the call returned.
    """

    function: str
    params: Dict[str, Any] = field(default_factory=dict)
    data_hash: Optional[str] = None
    data_shape: Optional[list] = None
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    statspai_version: str = field(default_factory=_statspai_version)
    python_version: str = field(default_factory=_python_version)
    timestamp: str = field(
        default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds")
    )

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict view, suitable for JSON dumping."""
        return asdict(self)

    def short(self) -> str:
        """One-line human summary."""
        bits = [self.function]
        if self.data_hash:
            bits.append(f"data:{self.data_hash}")
        if self.run_id:
            bits.append(f"run:{self.run_id}")
        return " · ".join(bits)


def attach_provenance(
    result: Any,
    *,
    function: str,
    params: Optional[Mapping[str, Any]] = None,
    data: Optional[Any] = None,
    enabled: bool = True,
    overwrite: bool = False,
) -> Any:
    """Attach a :class:`Provenance` record as ``result._provenance``.

    Parameters
    ----------
    result : object
        The estimator result. Must accept attribute assignment;
        ``CausalResult`` / ``ResultBase`` / dataclasses / SimpleNamespace
        all work. Tuples / dicts / immutable types do not — for those
        the call is a silent no-op.
    function : str
        Logical name of the producing call, e.g.
        ``"statspai.did.callaway_santanna"``.
    params : mapping, optional
        Call arguments. Will be summarised (frames hashed; long
        sequences truncated; non-serialisable values reduced to repr).
    data : DataFrame / Series / ndarray, optional
        The estimator's input data. Used to compute a 12-char
        SHA-256 fingerprint.
    enabled : bool, default True
        Set to False to skip provenance entirely (zero-overhead path).
    overwrite : bool, default False
        If False (default) and ``result._provenance`` already exists,
        do nothing — preserves the *first* (most-specific) record set
        by an inner estimator.

    Returns
    -------
    result : same object
        Returned for chaining: ``return attach_provenance(res, ...)``.

    Notes
    -----
    Failures are swallowed. Provenance must never break the caller —
    if attribute assignment isn't possible, we no-op and move on.
    """
    if not enabled or result is None:
        return result
    if not overwrite:
        try:
            existing = getattr(result, "_provenance", None)
            if existing is not None:
                return result
        except Exception:
            return result

    try:
        data_hash = compute_data_hash(data) if data is not None else None
        data_shape = None
        if data is not None:
            try:
                shape = getattr(data, "shape", None)
                if shape is not None:
                    data_shape = list(shape)
            except Exception:
                data_shape = None
        prov = Provenance(
            function=function,
            params=_summarise_params(params),
            data_hash=data_hash,
            data_shape=data_shape,
        )
        try:
            setattr(result, "_provenance", prov)
        except Exception:
            return result
    except Exception:
        return result
    return result


def get_provenance(result: Any) -> Optional[Provenance]:
    """Return ``result._provenance`` if present, else ``None``.

    Walks one level of common containers (``dict``, ``list``,
    ``tuple``) — useful when an estimator returns a tuple
    ``(result, diagnostics)``.
    """
    if result is None:
        return None
    direct = getattr(result, "_provenance", None)
    if isinstance(direct, Provenance):
        return direct
    if isinstance(result, Mapping):
        cand = result.get("_provenance")
        if isinstance(cand, Provenance):
            return cand
    if isinstance(result, (list, tuple)):
        for item in result:
            cand = getattr(item, "_provenance", None)
            if isinstance(cand, Provenance):
                return cand
    return None


def format_provenance(prov: Provenance, *, indent: int = 2) -> str:
    """Pretty multi-line rendering of a :class:`Provenance` record."""
    pad = " " * indent
    lines = [
        "Provenance",
        f"{pad}function   : {prov.function}",
        f"{pad}run_id     : {prov.run_id}",
        f"{pad}timestamp  : {prov.timestamp}",
        f"{pad}StatsPAI v{prov.statspai_version} · Python {prov.python_version}",
    ]
    if prov.data_hash:
        shape = (f" {prov.data_shape[0]}×{prov.data_shape[1]}"
                 if prov.data_shape and len(prov.data_shape) == 2 else "")
        lines.append(f"{pad}data       : SHA256:{prov.data_hash}{shape}")
    if prov.params:
        lines.append(f"{pad}params     :")
        for k, v in prov.params.items():
            r = repr(v)
            if len(r) > 100:
                r = r[:97] + "..."
            lines.append(f"{pad}  - {k} = {r}")
    return "\n".join(lines)


def lineage_summary(*results: Any) -> Dict[str, Any]:
    """Aggregate a lineage report across multiple results.

    Useful for ``sp.replication_pack`` / Quarto appendix generation:
    pass every fitted result the paper depends on and get back a
    ``{run_id: provenance_dict}`` map plus a deduped list of input
    data hashes.
    """
    runs: Dict[str, Dict[str, Any]] = {}
    data_hashes: Dict[str, list] = {}
    for r in results:
        prov = get_provenance(r)
        if prov is None:
            continue
        runs[prov.run_id] = prov.to_dict()
        if prov.data_hash:
            data_hashes.setdefault(prov.data_hash, []).append(
                {"function": prov.function, "run_id": prov.run_id}
            )
    return {
        "n_runs": len(runs),
        "runs": runs,
        "data_inputs": [
            {"hash": h, "consumers": v} for h, v in data_hashes.items()
        ],
        "statspai_version": _statspai_version(),
        "python_version": _python_version(),
    }
