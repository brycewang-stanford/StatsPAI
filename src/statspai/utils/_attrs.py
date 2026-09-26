"""Arrays that can live in ``DataFrame.attrs``.

pandas propagates ``attrs`` through ``concat`` only when every piece has the
same attrs, and decides that with a plain ``dict ==``.  A NumPy array compares
elementwise, so ``bool(a == b)`` raises and the ``concat`` fails -- including
the one pandas runs itself to truncate a wide or long frame for display, so a
result table carrying a covariance matrix could not even be printed.

``attrs_array`` returns a read-only copy whose ``==`` / ``!=`` answer with a
single bool (the arrays have the same shape and values, NaN matching NaN).
Everything else -- indexing, ``@``, ufuncs, ``np.diag`` -- behaves as for a
plain array, and the results of arithmetic are plain arrays again, so the
changed comparison does not spread past the stored matrix.  Use
``np.equal(a, b)`` for an elementwise comparison.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["attrs_array"]


class _AttrsArray(np.ndarray):
    """ndarray whose equality is a single bool; see the module docstring."""

    def __array_wrap__(
        self, obj: Any, context: Any = None, return_scalar: bool = False
    ) -> Any:
        # Results of ufuncs and arithmetic drop the subclass.
        out = np.asarray(obj).view(np.ndarray)
        return out[()] if return_scalar else out

    def __eq__(self, other: Any) -> bool:  # type: ignore[override]
        try:
            other_arr = np.asarray(other)
        except Exception:
            return False
        if self.shape != other_arr.shape:
            return False
        return bool(
            np.array_equal(
                self.view(np.ndarray),
                other_arr,
                equal_nan=self.dtype.kind in "fc" and other_arr.dtype.kind in "fc",
            )
        )

    def __ne__(self, other: Any) -> bool:  # type: ignore[override]
        return not self.__eq__(other)

    __hash__ = None  # type: ignore[assignment]


def attrs_array(values: Any) -> np.ndarray:
    """Read-only copy of ``values`` that is safe to store in ``DataFrame.attrs``."""
    out = np.array(values, copy=True).view(_AttrsArray)
    out.flags.writeable = False
    return out
