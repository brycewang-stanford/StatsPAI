"""Refuse category-label columns in a regressor / covariate list.

For estimators that take regressors as a list of column names and build
their design with ``astype(float)``: a column of digit labels (an industry
code kept as text, ``"1"`` ... ``"12"``) would be fitted as one number, and
a ``category`` column as its values. :func:`refuses_label_columns` checks
the named arguments before the call and raises ``MethodIncompatibility``
(see ``core.utils.refuse_label_columns``). Columns are read as labels by
``core.utils.label_columns``: ``category`` dtype, or a text / object column
holding strings. Numeric and boolean columns pass.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, List, TypeVar

import pandas as pd

F = TypeVar("F", bound=Callable[..., Any])


def _names(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        return list(value)
    return []


def refuses_label_columns(*params: str, data: str = "data") -> Callable[[F], F]:
    """Decorator: refuse label columns named by ``params`` in ``data``."""

    def deco(fn: F) -> F:
        sig = inspect.signature(fn)
        name = f"sp.{fn.__name__}"

        @functools.wraps(fn)
        def guarded(*args: Any, **kwargs: Any) -> Any:
            try:
                bound = sig.bind_partial(*args, **kwargs).arguments
            except TypeError:
                return fn(*args, **kwargs)  # the function reports the misuse
            frame = bound.get(data)
            if isinstance(frame, pd.DataFrame):
                cols = [
                    c
                    for p in params
                    for c in _names(bound.get(p))
                    if c in frame.columns
                ]
                if cols:
                    from .utils import refuse_label_columns

                    refuse_label_columns(frame, cols, function=name)
            return fn(*args, **kwargs)

        return guarded  # type: ignore[return-value]

    return deco
