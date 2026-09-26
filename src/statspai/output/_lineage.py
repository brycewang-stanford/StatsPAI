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
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional

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

    Examples
    --------
    >>> import pandas as pd
    >>> import statspai as sp
    >>> df = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [0.1, 0.2, 0.3]})
    >>> h = sp.compute_data_hash(df)
    >>> len(h)
    12
    >>> bool(sp.compute_data_hash(df) == h)  # deterministic
    True
    """
    if data is None:
        return None
    try:
        import numpy as np
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            if len(data) > MAX_HASH_ROWS:
                return None
            # ``hash_pandas_object`` is row-value sensitive but does NOT
            # mix column names into the per-row hash. Column renames
            # change "what the dataset means" — fold names + dtypes into
            # the digest explicitly so two frames with identical values
            # but different column names hash differently.
            schema = (
                "|".join(map(str, data.columns))
                + "::"
                + "|".join(str(dt) for dt in data.dtypes)
            ).encode()
            if isinstance(data.index, pd.RangeIndex) and all(
                pd.api.types.is_numeric_dtype(dt) for dt in data.dtypes
            ):
                arr = data.to_numpy(copy=False)
                if arr.dtype.kind in "biufc":
                    # Hot path for estimator inputs: all-numeric frames with
                    # a default RangeIndex. This preserves schema, row order,
                    # dtype, shape, and index metadata while avoiding pandas'
                    # per-column object hashing overhead on every fit.
                    digest = hashlib.sha256()
                    digest.update(b"numeric-dataframe-v1")
                    digest.update(schema)
                    digest.update(
                        (
                            f"RangeIndex:{data.index.start}:"
                            f"{data.index.stop}:{data.index.step}:"
                            f"{data.index.name}"
                        ).encode()
                    )
                    arr = np.ascontiguousarray(arr)
                    digest.update(str(arr.shape).encode())
                    digest.update(str(arr.dtype).encode())
                    digest.update(arr.tobytes())
                    return digest.hexdigest()[:length]

            row_h = pd.util.hash_pandas_object(data, index=True).values.tobytes()
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
        import numpy as np
        import pandas as pd

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

    Examples
    --------
    >>> import statspai as sp
    >>> prov = sp.Provenance(function="sp.did.callaway_santanna",
    ...                      params={"method": "dr"})
    >>> prov.function
    'sp.did.callaway_santanna'
    >>> sorted(prov.to_dict())  # JSON-able view
    ['data_hash', 'data_shape', 'function', 'params', 'python_version', 'run_id', 'statspai_version', 'timestamp']
    >>> prov.short().startswith("sp.did.callaway_santanna")
    True
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

    Examples
    --------
    >>> import pandas as pd
    >>> import statspai as sp
    >>> from types import SimpleNamespace
    >>> df = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [0.1, 0.2, 0.3]})
    >>> res = SimpleNamespace(estimate=1.23)
    >>> _ = sp.attach_provenance(res, function="sp.did.callaway_santanna",
    ...                          params={"method": "dr"}, data=df)
    >>> res._provenance.function
    'sp.did.callaway_santanna'
    >>> res._provenance.data_shape
    [3, 2]
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
    _note_listwise_deletion(result, function, params, data)
    return result


def _reported_nobs(result: Any) -> Optional[int]:
    """The estimation-sample size a result reports, if it reports one."""
    import numpy as np

    for owner, keys in (
        (result, ("n_obs", "nobs")),
        (getattr(result, "data_info", None), ("nobs", "n_obs")),
        (getattr(result, "model_info", None), ("n_obs", "nobs")),
    ):
        if owner is None:
            continue
        for key in keys:
            value = (
                owner.get(key)
                if isinstance(owner, Mapping)
                else getattr(owner, key, None)
            )
            if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
                return int(value)
            if isinstance(value, (float, np.floating)) and float(value).is_integer():
                return int(value)
    return None


def _referenced_columns(params: Optional[Mapping[str, Any]], columns: Any) -> List[str]:
    """Data columns named by the call's arguments (strings, lists, formulas)."""
    import re

    cols = set(map(str, columns))
    found: List[str] = []

    def add(name: Any) -> None:
        if isinstance(name, str) and name in cols and name not in found:
            found.append(name)

    for value in (params or {}).values():
        if isinstance(value, str):
            if value in cols:
                add(value)
            elif "~" in value:
                for token in re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", value):
                    add(token)
        elif isinstance(value, (list, tuple)):
            for item in value:
                add(item)
    return found


#: Designs in which a missing value is information, not a dropped row:
#: attrition / truncation-by-death bounds use the missing outcomes to set
#: the trimming share, censoring weights model them, a surrogate index or
#: a transport weight is built precisely where the outcome is unobserved,
#: and imputation fills them. The listwise-deletion note would be false.
_MISSING_BY_DESIGN = frozenset(
    {
        "lee_bounds",
        "manski_bounds",
        "horowitz_manski",
        "selection_bounds",
        "survivor_average_causal_effect",
        "ipcw",
        "clone_censor_weight",
        "surrogate_index",
        "transport_weights",
        "mice",
        "heckman",
    }
)


def _note_listwise_deletion(
    result: Any,
    function: str,
    params: Optional[Mapping[str, Any]],
    data: Any,
) -> None:
    """Warn when rows with missing values were dropped from the estimation.

    Estimators drop rows with a missing value in any variable they use
    (listwise deletion), as Stata and R do -- but R prints how many rows
    went and Stata shows the smaller N, while StatsPAI said nothing. The
    note fires only when the result's reported N equals the input rows
    minus the incomplete ones, i.e. when listwise deletion demonstrably
    happened; estimators that impute, trim to a bandwidth or subset the
    sample for other reasons do not match and stay silent. Never raises.
    """
    try:
        import pandas as pd

        if function.split(".")[-1] in _MISSING_BY_DESIGN:
            return
        if not isinstance(data, pd.DataFrame) or len(data) == 0:
            return
        used = _referenced_columns(params, data.columns)
        if not used:
            return
        incomplete = data[used].isna()
        n_dropped = int(incomplete.any(axis=1).sum())
        if n_dropped == 0:
            return
        nobs = _reported_nobs(result)
        if nobs is None or nobs != len(data) - n_dropped:
            return
        by_column = {c: int(n) for c, n in incomplete.sum().items() if n}
        info = {"n_rows_dropped_missing": n_dropped, "missing_by_column": by_column}
        model_info = getattr(result, "model_info", None)
        if isinstance(model_info, dict):
            model_info.setdefault("listwise_deletion", info)
    except Exception:
        return
    from ..exceptions import AssumptionWarning

    name = function.split(".")[-1]
    warnings.warn(
        AssumptionWarning(
            f"{name}: {n_dropped} of {len(data)} rows dropped for missing "
            f"values (listwise deletion) -- by column: {by_column}.",
            recovery_hint=(
                "Estimates use complete cases only, which is unbiased only if "
                "missingness is unrelated to the outcome given the model's "
                "variables. Impute with sp.mice to check, or drop the rows "
                "explicitly to silence this note."
            ),
            diagnostics=info,
        ),
        stacklevel=3,
    )


def get_provenance(result: Any) -> Optional[Provenance]:
    """Return ``result._provenance`` if present, else ``None``.

    Walks one level of common containers (``dict``, ``list``,
    ``tuple``) — useful when an estimator returns a tuple
    ``(result, diagnostics)``.

    Examples
    --------
    >>> import statspai as sp
    >>> from types import SimpleNamespace
    >>> res = SimpleNamespace(estimate=1.23)
    >>> _ = sp.attach_provenance(res, function="sp.iv.ivreg")
    >>> prov = sp.get_provenance(res)
    >>> isinstance(prov, sp.Provenance)
    True
    >>> prov.function
    'sp.iv.ivreg'
    >>> sp.get_provenance(SimpleNamespace()) is None  # no record attached
    True
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
    """Pretty multi-line rendering of a :class:`Provenance` record.

    Examples
    --------
    >>> import statspai as sp
    >>> prov = sp.Provenance(function="sp.rd.rdrobust",
    ...                      params={"kernel": "triangular"})
    >>> text = sp.format_provenance(prov)
    >>> text.splitlines()[0]
    'Provenance'
    >>> "sp.rd.rdrobust" in text
    True
    """
    pad = " " * indent
    lines = [
        "Provenance",
        f"{pad}function   : {prov.function}",
        f"{pad}run_id     : {prov.run_id}",
        f"{pad}timestamp  : {prov.timestamp}",
        f"{pad}StatsPAI v{prov.statspai_version} · Python {prov.python_version}",
    ]
    if prov.data_hash:
        shape = (
            f" {prov.data_shape[0]}×{prov.data_shape[1]}"
            if prov.data_shape and len(prov.data_shape) == 2
            else ""
        )
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

    Examples
    --------
    >>> import statspai as sp
    >>> from types import SimpleNamespace
    >>> r1, r2 = SimpleNamespace(), SimpleNamespace()
    >>> _ = sp.attach_provenance(r1, function="sp.did.callaway_santanna")
    >>> _ = sp.attach_provenance(r2, function="sp.iv.ivreg")
    >>> report = sp.lineage_summary(r1, r2)
    >>> report["n_runs"]
    2
    >>> sorted(report)
    ['data_inputs', 'n_runs', 'python_version', 'runs', 'statspai_version']
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
        "data_inputs": [{"hash": h, "consumers": v} for h, v in data_hashes.items()],
        "statspai_version": _statspai_version(),
        "python_version": _python_version(),
    }
