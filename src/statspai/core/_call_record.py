"""Exact record of the public call that produced a result.

``result._provenance`` is a *summary* of the call, written for humans and
replication packs: arrays are reduced to fingerprints, and several
estimators record only a subset of their arguments (``sp.regress`` leaves
out ``weights=`` and ``hac_lags=``). That is fine for a footer and wrong for
refitting: replaying a summary would silently drop the weights.

``@records_call`` keeps the arguments themselves, so that
:func:`statspai.vcov` can re-run the user's own call with another ``vce=``.
Two rules keep it cheap and honest:

* the input DataFrame is held through a weak reference, so a result never
  keeps a large frame alive on its own; when the frame is gone the caller
  passes ``data=`` again;
* nothing here decides whether a replay is valid. :func:`statspai.vcov`
  re-runs the recorded call unchanged first and refuses unless it
  reproduces the original estimates, which catches edited data, a different
  frame passed as ``data=``, and unseeded randomness.
"""

from __future__ import annotations

import functools
import weakref
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, cast

import pandas as pd

_F = TypeVar("_F", bound=Callable[..., Any])

__all__ = ["CallRecord", "records_call", "get_call_record"]

#: Attribute the record lives under on a result object.
ATTR = "_statspai_call"


class _FrameRef:
    """A weak reference to a DataFrame argument, with its shape for messages."""

    __slots__ = ("ref", "shape")

    def __init__(self, frame: pd.DataFrame) -> None:
        self.ref = weakref.ref(frame)
        self.shape = tuple(frame.shape)

    def get(self) -> Optional[pd.DataFrame]:
        return self.ref() if self.ref is not None else None

    def __reduce__(self):
        # A weak reference cannot be pickled, and a result that crosses a
        # process (the MCP state store, isolated fits) no longer shares the
        # caller's frame anyway: it arrives as a reference to a frame that is
        # gone, and sp.vcov asks for data= again.
        return (_gone, (self.shape,))


def _gone(shape: Tuple[int, ...]) -> "_FrameRef":
    ref = _FrameRef.__new__(_FrameRef)
    ref.ref = None
    ref.shape = tuple(shape)
    return ref


def _freeze(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        try:
            return _FrameRef(value)
        except TypeError:  # pragma: no cover - DataFrames are weak-referenceable
            return value
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    return value


@dataclass
class CallRecord:
    """The public call that produced a result.

    Attributes
    ----------
    function : str
        Public name, resolved as ``getattr(statspai, function)`` on replay.
    args, kwargs
        The arguments as the caller passed them (before alias resolution).
        DataFrames are held weakly.
    """

    function: str
    args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def frames(self) -> Dict[str, _FrameRef]:
        """Positions (``"0"``, ``"1"`` ...) and keywords that hold a frame."""
        out: Dict[str, _FrameRef] = {}
        for i, value in enumerate(self.args):
            if isinstance(value, _FrameRef):
                out[str(i)] = value
        for key, value in self.kwargs.items():
            if isinstance(value, _FrameRef):
                out[key] = value
        return out

    def restore(
        self, data: Optional[pd.DataFrame] = None
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """``(args, kwargs)`` ready to call again.

        ``data`` replaces the recorded frame (there must be exactly one).
        Without it the recorded frame is used; if it has been garbage
        collected, :class:`LookupError` names what to pass.
        """
        frames = self.frames()
        if data is not None and len(frames) != 1:
            raise LookupError(
                f"sp.{self.function} was called with {len(frames)} data "
                "frames; data= cannot say which one to replace."
            )

        def thaw(value: Any) -> Any:
            if not isinstance(value, _FrameRef):
                return value
            if data is not None:
                return data
            frame = value.get()
            if frame is None:
                raise LookupError(
                    f"the {value.shape[0]}x{value.shape[1]} DataFrame passed to "
                    f"sp.{self.function} no longer exists; pass it again as data=."
                )
            return frame

        args = tuple(thaw(v) for v in self.args)
        kwargs = {k: thaw(v) for k, v in self.kwargs.items()}
        return args, kwargs


def get_call_record(result: Any) -> Optional[CallRecord]:
    """The :class:`CallRecord` on ``result``, or None."""
    record = getattr(result, ATTR, None)
    return record if isinstance(record, CallRecord) else None


def records_call(function: str) -> Callable[[_F], _F]:
    """Decorator: attach a :class:`CallRecord` of this call to the result.

    Applied as the outermost decorator of a public estimator. When one
    recorded estimator calls another, the outer call's record wins: it is
    the call the user made, and the one whose ``vce=`` grammar applies.
    Results that do not take attributes (tuples, frames) are returned
    untouched.
    """

    def decorator(func: _F) -> _F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)
            if result is None or isinstance(
                result, (tuple, list, dict, pd.DataFrame, pd.Series)
            ):
                return result
            try:
                record = CallRecord(
                    function=function,
                    args=tuple(_freeze(v) for v in args),
                    kwargs={k: _freeze(v) for k, v in kwargs.items()},
                )
                object.__setattr__(result, ATTR, record)
            except (AttributeError, TypeError):
                # Slotted or immutable result: no record, and sp.vcov says so.
                pass
            return result

        return cast(_F, wrapper)

    return decorator
