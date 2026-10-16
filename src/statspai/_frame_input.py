"""Accept polars, Arrow and other DataFrame libraries at every ``sp.*`` entry.

The estimators are written against pandas. Rather than teach each of them
another library, the public functions are wrapped once (``statspai/__init__``
calls :func:`install`): an argument that is a polars ``DataFrame`` /
``LazyFrame`` / ``Series``, a pyarrow ``Table`` / ``RecordBatch`` /
``ChunkedArray``, or any object speaking the Arrow PyCapsule stream
protocol (``__arrow_c_stream__``) or the DataFrame interchange protocol
(``__dataframe__``) is converted to pandas before the call. Results are
the usual pandas-based objects.

pandas inputs, and everything else, are passed through untouched: the
wrapper hands the function the very same objects it was given (a test pins
this). That is why the two provenance tracers
(``scripts/trace_parity_provenance.py``, ``scripts/trace_perf_path.py``)
leave this file out of the recorded estimation path, as the perf tracer
does for ``statspai/__init__.py``: for the pandas inputs the parity and
timing runs use, it does no computation.

polars and pyarrow are never imported here unless such an object arrives.
"""

from __future__ import annotations

import functools
import sys
from typing import Any, Callable, Dict, Optional, Tuple, cast

import pandas as pd

#: Marks a wrapped function (and points at the original).
ATTR = "__statspai_frame_input__"


def bare(obj: Any) -> Any:
    """The function beneath the frame-input wrapper (``obj`` if unwrapped)."""
    return getattr(obj, ATTR, None) or obj


def _polars(value: Any) -> Any:
    kind = type(value).__name__
    if kind == "LazyFrame":
        value = value.collect()
        kind = "DataFrame"
    try:
        return value.to_pandas()
    except ImportError as exc:  # polars needs pyarrow for to_pandas
        raise ImportError(
            f"Converting a polars {kind} to pandas needs pyarrow: "
            "pip install pyarrow (or pass df.to_pandas())."
        ) from exc


def convert(value: Any) -> Any:
    """``value`` as pandas if it is a foreign DataFrame-like, else itself."""
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return value
    cls = type(value)
    root = cls.__module__.split(".", 1)[0]
    if root == "polars" and cls.__name__ in ("DataFrame", "LazyFrame", "Series"):
        return _polars(value)
    if root == "pyarrow" and cls.__name__ in (
        "Table",
        "RecordBatch",
        "ChunkedArray",
        "Array",
    ):
        return value.to_pandas()
    if hasattr(cls, "__arrow_c_stream__") and not isinstance(value, type):
        try:
            import pyarrow as pa
        except ImportError as exc:
            raise ImportError(
                f"Reading a {cls.__module__}.{cls.__name__} through the Arrow "
                "stream protocol needs pyarrow: pip install pyarrow."
            ) from exc
        return pa.table(value).to_pandas()
    if hasattr(cls, "__dataframe__") and not isinstance(value, type):
        return pd.api.interchange.from_dataframe(value)
    return value


def _convert_all(
    args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> Optional[Tuple[Tuple[Any, ...], Dict[str, Any]]]:
    """Converted ``(args, kwargs)``, or None when nothing needed converting."""
    changed = False
    new_args = list(args)
    for i, v in enumerate(args):
        c = convert(v)
        if c is not v:
            new_args[i] = c
            changed = True
    new_kwargs = dict(kwargs)
    for k, v in kwargs.items():
        c = convert(v)
        if c is not v:
            new_kwargs[k] = c
            changed = True
    return (tuple(new_args), new_kwargs) if changed else None


def _is_plain(v: Any) -> bool:
    # Cheap test that lets the common call (pandas, numbers, strings, None)
    # skip convert() entirely.
    return v is None or isinstance(
        v, (pd.DataFrame, pd.Series, str, int, float, bool, list, tuple, dict)
    )


def _passthrough(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
    """None when the call needs no conversion, else the converted arguments."""
    if all(_is_plain(v) for v in args) and all(_is_plain(v) for v in kwargs.values()):
        return None
    return _convert_all(args, kwargs)


# The wrapper body is compiled under a file name containing "importlib" and
# "_bootstrap". CPython's warnings machinery (C and pure-Python alike) skips
# such frames when it counts ``stacklevel`` -- the convention that keeps
# import machinery out of warning locations -- so a warning an estimator
# issues with ``stacklevel=2`` still points at the user's line, not here.
# Without it every warning of the 850 wrapped functions would be reported
# from this file. Checked on Python 3.10, 3.12 and 3.13
# (tests/test_frame_input.py::test_warnings_still_point_at_the_caller).
_SHIM_SOURCE = """
def _make(fn):
    def accepting_frames(*args, **kwargs):
        converted = _passthrough(args, kwargs)
        if converted is None:
            return fn(*args, **kwargs)
        return fn(*converted[0], **converted[1])
    return accepting_frames
"""
_shim_ns: Dict[str, Any] = {"_passthrough": _passthrough}
exec(  # noqa: S102  # nosec B102 - the fixed literal above, no user input
    compile(
        _SHIM_SOURCE,
        "<statspai frame-input shim; importlib._bootstrap convention>",
        "exec",
    ),
    _shim_ns,
)
_SHIM_FACTORY: Callable[[Callable[..., Any]], Callable[..., Any]] = _shim_ns["_make"]


#: One wrapper per original function, so aliases (``sp.bjs`` and
#: ``sp.did_imputation``) stay the same object however they are reached.
_WRAPPERS: Dict[int, Tuple[Any, Any]] = {}


def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
    """``fn`` taking foreign DataFrames; pandas arguments pass by identity."""
    if getattr(fn, ATTR, None) is not None:
        return fn
    cached = _WRAPPERS.get(id(fn))
    if cached is not None and cached[0] is fn:
        return cast(Callable[..., Any], cached[1])

    accepting_frames = functools.wraps(fn)(_SHIM_FACTORY(fn))
    setattr(accepting_frames, ATTR, fn)
    _WRAPPERS[id(fn)] = (fn, accepting_frames)
    return accepting_frames


def _eligible(obj: Any) -> bool:
    import inspect

    return inspect.isfunction(obj) and getattr(obj, ATTR, None) is None


def _rebind(mapping: Dict[int, Tuple[Any, Any]]) -> None:
    """Point every loaded ``statspai.*`` name that holds an original at its wrapper.

    ``statspai.survival.cox``, ``statspai.survival.models.cox`` and
    ``sp.cox`` must stay one object: the parity index checks that identity,
    and pickle finds a function as ``module.qualname``, which must resolve
    to the object being pickled (joblib and process pools rely on it).
    """
    if not mapping:
        return
    for modname, module in list(sys.modules.items()):
        if module is None or not (
            modname == "statspai" or modname.startswith("statspai.")
        ):
            continue
        namespace = getattr(module, "__dict__", None)
        if not isinstance(namespace, dict):
            continue
        for key, value in list(namespace.items()):
            hit = mapping.get(id(value))
            if hit is not None and hit[0] is value:
                namespace[key] = hit[1]


def wrap_public(name: str, obj: Any) -> Any:
    """Wrap one public ``sp.<name>`` and rebind it wherever it is loaded."""
    if not _eligible(obj):
        return obj
    wrapped = wrap(obj)
    _rebind({id(obj): (obj, wrapped)})
    return wrapped


def install(namespace: Dict[str, Any], names: Any) -> None:
    """Wrap every eligible public function already bound in ``namespace``."""
    mapping: Dict[int, Tuple[Any, Any]] = {}
    for name in names:
        obj = namespace.get(name)
        if obj is not None and _eligible(obj):
            mapping[id(obj)] = (obj, wrap(obj))
    _rebind(mapping)
    for name in names:
        hit = mapping.get(id(namespace.get(name)))
        if hit is not None:
            namespace[name] = hit[1]
