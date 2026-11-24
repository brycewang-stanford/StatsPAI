"""Accept polars, Arrow and other DataFrame libraries at every ``sp.*`` entry.

The estimators are written against pandas. Rather than teach each of them
another library, the public functions are wrapped once (``statspai/__init__``
calls :func:`install`): an argument that is a polars ``DataFrame`` /
``LazyFrame`` / ``Series``, a pyarrow ``Table`` / ``RecordBatch`` /
``ChunkedArray``, or any object speaking the Arrow PyCapsule stream
protocol (``__arrow_c_stream__``) or the DataFrame interchange protocol
(``__dataframe__``) is converted to pandas before the call. Results are
the usual pandas-based objects. The exception is a function that returns
the data itself with columns added or replaced (``sp.winsor``): decorated
with :func:`returns_input_frame`, it gives a polars caller polars back.

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
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

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

def _make_same(fn, needs):
    def same_frame_out(*args, **kwargs):
        narrow = _narrow(fn, needs, args, kwargs)
        if narrow is None:
            converted = _passthrough(args, kwargs)
        else:
            converted = _passthrough(narrow[0], narrow[1])
        if converted is None:
            return fn(*args, **kwargs)
        out = fn(*converted[0], **converted[1])
        return _restore(args, kwargs, converted, out, narrow)
    return same_frame_out

def _make_part(fn, part):
    def reading_part(*args, **kwargs):
        cut = _cut(fn, part, args, kwargs)
        if cut is not None:
            args, kwargs = cut
        converted = _passthrough(args, kwargs)
        if converted is None:
            return fn(*args, **kwargs)
        return fn(*converted[0], **converted[1])
    return reading_part
"""


def _first_frame(
    args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> Optional[Tuple[Any, Any]]:
    """``(position or name, value)`` of the data argument, if it is polars."""
    candidates: List[Tuple[Any, Any]] = [(0, args[0])] if args else []
    if "data" in kwargs:
        candidates.append(("data", kwargs["data"]))
    for where, value in candidates:
        cls = type(value)
        if cls.__module__.split(".", 1)[0] == "polars" and cls.__name__ in (
            "DataFrame",
            "LazyFrame",
        ):
            return where, value
    return None


def _narrow(
    fn: Callable[..., Any],
    needs: Optional[Callable[[Dict[str, Any]], Optional[List[str]]]],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Optional[Tuple[Tuple[Any, ...], Dict[str, Any], Any]]:
    """The call with its polars frame cut down to the columns ``fn`` reads.

    Converting a wide frame to pandas to change five of its columns costs
    several times the frame in memory. ``needs`` (see
    :func:`returns_input_frame`) names the columns the call reads; only
    those go to pandas, and :func:`_restore` puts the result back beside
    the caller's other columns. Returns ``(args, kwargs, frame)`` with
    ``frame`` the caller's (collected) frame, or None when there is nothing
    to cut: pandas input (decided by a type check alone), a call ``needs``
    cannot resolve, or a name that is not a column, which is left for
    ``fn`` to report.
    """
    if needs is None:
        return None
    hit = _first_frame(args, kwargs)
    if hit is None:
        return None
    where, original = hit
    import inspect

    try:
        bound = inspect.signature(fn).bind(*args, **kwargs)
    except TypeError:
        return None
    bound.apply_defaults()
    wanted = needs(dict(bound.arguments))
    if wanted is None or not all(isinstance(c, str) for c in wanted):
        return None
    lazy = type(original).__name__ == "LazyFrame"
    names = original.collect_schema().names() if lazy else original.columns
    keep = [c for c in names if c in set(wanted)]
    if len(keep) != len(set(wanted)) or len(keep) == len(names):
        return None
    frame = original.collect() if lazy else original
    part = frame.select(keep)
    if where == 0:
        return (part,) + tuple(args[1:]), kwargs, frame
    return args, {**kwargs, "data": part}, frame


def _key_values(frame: Any, column: str) -> Any:
    """The values of ``column`` in a pandas or polars frame, as a list."""
    if isinstance(frame, pd.DataFrame):
        if column not in frame.columns:
            return None
        return frame[column].dropna().unique().tolist()
    cls = type(frame)
    if cls.__module__.split(".", 1)[0] == "polars" and cls.__name__ in (
        "DataFrame",
        "LazyFrame",
    ):
        lazy = cls.__name__ == "LazyFrame"
        names = frame.collect_schema().names() if lazy else frame.columns
        if column not in names:
            return None
        keys = frame.select(column).unique().drop_nulls()
        return (keys.collect() if lazy else keys).get_column(column).to_list()
    return None


def _cut(
    fn: Callable[..., Any],
    part: Callable[[Dict[str, Any]], Any],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Optional[Tuple[Tuple[Any, ...], Dict[str, Any]]]:
    """The call with its polars frame cut down to the part ``fn`` reads.

    ``part`` (see :func:`reads_part`) names the columns the call reads
    and, optionally, the rows: those whose ``column`` holds a value found
    in a column of another frame of the call. The selection and the filter
    run in polars, so a ten-million-row return file of which the call
    reads forty securities is not copied into pandas first. Returns None
    when there is nothing to cut: pandas input (decided by a type check
    alone), a call ``part`` cannot resolve, or a name that is not a
    column, which is left for ``fn`` to report. The row filter is skipped
    when the keys are of another type than the column; ``fn`` then meets
    the rows it would have met without this step.
    """
    hit = _first_frame(args, kwargs)
    if hit is None:
        return None
    where, original = hit
    import inspect

    try:
        bound = inspect.signature(fn).bind(*args, **kwargs)
    except TypeError:
        return None
    bound.apply_defaults()
    spec = part(dict(bound.arguments))
    if spec is None:
        return None
    columns, rows = spec
    if not all(isinstance(c, str) for c in columns):
        return None
    lazy = type(original).__name__ == "LazyFrame"
    names = original.collect_schema().names() if lazy else original.columns
    keep = [c for c in names if c in set(columns)]
    if len(keep) != len(set(columns)):
        return None
    import importlib

    pl: Any = importlib.import_module("polars")
    query = original.lazy()
    if rows is not None:
        column, other, other_column = rows
        keys = _key_values(other, other_column) if column in names else None
        if keys is not None:
            have = original.collect_schema()[column]
            try:
                wanted = pl.Series(keys)
            except (TypeError, ValueError, pl.exceptions.PolarsError):
                wanted = None  # keys of mixed types: read every row
            # The filter runs only when the two columns hold one kind of
            # value; comparing integers with strings is left to ``fn``.
            if wanted is not None and (
                wanted.dtype == have
                or (wanted.dtype.is_integer() and have.is_integer())
            ):
                query = query.filter(pl.col(column).is_in(wanted.to_list()))
    cut = query.select(keep).collect()
    if where == 0:
        return (cut,) + tuple(args[1:]), kwargs
    return args, {**kwargs, "data": cut}


def _restore(
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    converted: Tuple[Tuple[Any, ...], Dict[str, Any]],
    out: Any,
    narrow: Optional[Tuple[Tuple[Any, ...], Dict[str, Any], Any]] = None,
) -> Any:
    """Give a polars caller a polars frame back.

    For functions that return their data with columns added or replaced
    (``sp.winsor``). The columns the function left alone are taken from the
    caller's own frame, so their dtypes survive (a ``Date`` would come back
    from pandas as a ``Datetime``); only new and changed columns are built
    from the pandas result, with NaN as null. A ``LazyFrame`` gets a
    ``LazyFrame``. Anything else (pandas in, or a result that is not a
    frame of the same length) is returned as the function produced it.
    """
    hit = _first_frame(args, kwargs)
    if hit is None or not isinstance(out, pd.DataFrame):
        return out
    where, original = hit
    as_pandas = converted[0][0] if where == 0 else converted[1]["data"]
    import importlib

    pl: Any = importlib.import_module("polars")
    lazy = type(original).__name__ == "LazyFrame"
    if narrow is not None:
        frame = narrow[2]
    else:
        frame = original.collect() if lazy else original
    if len(out) != frame.height or not isinstance(out.index, pd.RangeIndex):
        if narrow is not None:  # pragma: no cover - a needs= function keeps rows
            raise RuntimeError(
                "a function decorated with returns_input_frame(needs=...) "
                "must return one row per input row"
            )
        result = pl.from_pandas(out, nan_to_null=True)
        return result.lazy() if lazy else result
    if narrow is not None:
        # Only the columns the function read went to pandas: the caller's
        # frame keeps every other column, changed ones are replaced in
        # place and new ones are appended, as the full round trip does.
        built = [
            pl.from_pandas(out[name], nan_to_null=True).alias(str(name))
            for name in out.columns
            if not (name in as_pandas.columns and out[name].equals(as_pandas[name]))
        ]
        result = frame.with_columns(built)
        return result.lazy() if lazy else result
    columns = []
    for name in out.columns:
        key = str(name)
        untouched = (
            key in frame.columns
            and name in as_pandas.columns
            and out[name].equals(as_pandas[name])
        )
        if untouched:
            columns.append(frame.get_column(key))
        else:
            columns.append(pl.from_pandas(out[name], nan_to_null=True).alias(key))
    result = pl.DataFrame(columns)
    return result.lazy() if lazy else result


_shim_ns: Dict[str, Any] = {
    "_passthrough": _passthrough,
    "_restore": _restore,
    "_narrow": _narrow,
    "_cut": _cut,
}
exec(  # noqa: S102  # nosec B102 - the fixed literal above, no user input
    compile(
        _SHIM_SOURCE,
        "<statspai frame-input shim; importlib._bootstrap convention>",
        "exec",
    ),
    _shim_ns,
)
_SHIM_FACTORY: Callable[[Callable[..., Any]], Callable[..., Any]] = _shim_ns["_make"]
_SAME_FACTORY: Callable[..., Callable[..., Any]] = _shim_ns["_make_same"]
_PART_FACTORY: Callable[..., Callable[..., Any]] = _shim_ns["_make_part"]


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


def accepts_frames(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator form of :func:`wrap` for methods that take new data.

    Result methods are not reached by :func:`install` (which wraps the
    public ``sp.*`` functions), so ``result.predict(polars_frame)`` would
    otherwise meet a pandas-only check. Applied to the ``predict`` methods
    of result classes: ``self`` and pandas arguments pass by identity, a
    polars / Arrow frame is converted to pandas first.
    """
    return wrap(fn)


def returns_input_frame(
    fn: Optional[Callable[..., Any]] = None,
    *,
    needs: Optional[Callable[[Dict[str, Any]], Optional[List[str]]]] = None,
) -> Any:
    """Decorator for functions that hand the data back with columns changed.

    ``sp.winsor(polars_frame, ...)`` then returns a polars frame (see
    :func:`_restore`), so it can sit in a ``.pipe()`` chain. Estimators do
    not use this: their results are pandas-based whatever came in. pandas
    arguments still pass by identity. :func:`install` leaves a function
    decorated here alone.

    ``needs`` is optional: a function of the call's bound arguments (a
    dict, defaults applied) returning the names of the columns the call
    reads, or None when it cannot tell. With it, a polars frame sends only
    those columns through pandas (see :func:`_narrow`); the function must
    then return one row per input row and leave the row order alone.
    """

    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        same_frame_out = functools.wraps(func)(_SAME_FACTORY(func, needs))
        setattr(same_frame_out, ATTR, func)
        _WRAPPERS[id(func)] = (func, same_frame_out)
        return cast(Callable[..., Any], same_frame_out)

    return decorate if fn is None else decorate(fn)


def reads_part(
    part: Callable[[Dict[str, Any]], Any],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for functions that read a small part of a large frame.

    ``part`` is a function of the call's bound arguments (a dict, defaults
    applied). It returns ``(columns, rows)``: the names of the columns of
    ``data`` the call reads, and either None or ``(column, frame,
    frame_column)``, meaning that only the rows of ``data`` whose
    ``column`` holds a value of ``frame[frame_column]`` are read. It
    returns None when it cannot tell. A polars ``data`` is then selected
    and filtered in polars before it is converted (see :func:`_cut`). The
    function must give the same result on the cut frame as on the whole
    one. pandas arguments pass by identity, and :func:`install` leaves a
    function decorated here alone.
    """

    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        reading_part = functools.wraps(func)(_PART_FACTORY(func, part))
        setattr(reading_part, ATTR, func)
        _WRAPPERS[id(func)] = (func, reading_part)
        return cast(Callable[..., Any], reading_part)

    return decorate


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
