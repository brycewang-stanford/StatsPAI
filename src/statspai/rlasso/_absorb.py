"""Within transformation for the ``absorb=`` option of the rigorous Lasso.

Stata's ``rlasso, fe`` and ``pdslasso, fe`` (lassopack / pdslasso) demean
every variable by the panel identifier and then run the Lasso without a
constant. Nothing else changes: the penalty level uses the full number of
observations, and the post-selection variances carry no correction for the
absorbed levels (checked against both commands on the fixture in
``tests/reference_parity/test_rlasso_fe_stata_parity.py``, where ``fe`` and
``nocons`` on hand-demeaned data give identical numbers).

With one factor the transform is exact group demeaning. With several
factors it is the orthogonal projection onto the complement of all the
dummies, computed by alternating projections; Stata reaches the same
quantity with ``fe partial(i.<second factor>)``.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..exceptions import ConvergenceFailure, DataInsufficient, MethodIncompatibility

_MAX_SWEEPS = 10_000
_TOL = 1e-14


def absorb_codes(
    absorb: Any, n: int, data: Optional[pd.DataFrame] = None
) -> List[np.ndarray]:
    """Integer codes ``0..G-1`` for each factor in ``absorb``.

    ``absorb`` may be a column name or a list of column names (with
    ``data``), a single array of length ``n``, a 2-D array with one column per
    factor, or a list of arrays.
    """
    if isinstance(absorb, str):
        absorb = [absorb]
    if isinstance(absorb, pd.DataFrame):
        factors: List[Any] = [absorb[c].to_numpy() for c in absorb.columns]
    elif isinstance(absorb, (list, tuple)):
        factors = []
        for f in absorb:
            if isinstance(f, str):
                if data is None:
                    raise MethodIncompatibility(
                        f"absorb={f!r} is a column name, which needs `data`.",
                        recovery_hint="Pass data=, or pass the factor as an array.",
                    )
                if f not in data.columns:
                    raise MethodIncompatibility(
                        f"absorb column {f!r} is not in data.",
                        recovery_hint="Check the spelling of the absorb column.",
                    )
                factors.append(data[f].to_numpy())
            else:
                factors.append(np.asarray(f))
    else:
        arr = np.asarray(absorb)
        if arr.ndim == 2:
            factors = [arr[:, j] for j in range(arr.shape[1])]
        else:
            factors = [arr]
    if not factors:
        raise MethodIncompatibility(
            "absorb= is empty.", recovery_hint="Pass at least one factor or None."
        )
    out: List[np.ndarray] = []
    for k, f in enumerate(factors):
        f = np.asarray(f)
        if f.ndim != 1 or f.shape[0] != n:
            raise MethodIncompatibility(
                f"absorb factor {k} must have one entry per row ({n}); "
                f"got shape {f.shape}.",
                recovery_hint="Pass one identifier per observation.",
            )
        if pd.isna(pd.Series(f)).any():
            raise DataInsufficient(
                f"absorb factor {k} contains missing values.",
                recovery_hint="Drop the rows with a missing identifier first.",
            )
        codes, _ = pd.factorize(pd.Series(f), sort=True)
        out.append(codes.astype(np.intp))
    return out


def _demean_once(M: np.ndarray, codes: np.ndarray) -> np.ndarray:
    G = int(codes.max()) + 1
    counts = np.bincount(codes, minlength=G).astype(float)
    sums = np.zeros((G, M.shape[1]))
    np.add.at(sums, codes, M)
    out: np.ndarray = M - (sums / counts[:, None])[codes]
    return out


def within(M: np.ndarray, codes: Sequence[np.ndarray]) -> np.ndarray:
    """Project the columns of ``M`` off the dummies of every factor."""
    M = np.asarray(M, dtype=float)
    squeeze = M.ndim == 1
    if squeeze:
        M = M.reshape(-1, 1)
    out = _demean_once(M, codes[0])
    if len(codes) > 1:
        scale = max(float(np.abs(M).max()), 1.0)
        for _ in range(_MAX_SWEEPS):
            prev = out
            for c in codes:
                out = _demean_once(out, c)
            if float(np.abs(out - prev).max()) <= _TOL * scale:
                break
        else:
            raise ConvergenceFailure(
                f"absorb: alternating projections did not converge in "
                f"{_MAX_SWEEPS} sweeps.",
                recovery_hint=(
                    "The factors are nearly collinear; absorb fewer factors or "
                    "add their interaction as a single factor."
                ),
            )
    return out[:, 0] if squeeze else out


def absorbed_levels(codes: Sequence[np.ndarray]) -> Tuple[int, ...]:
    """Number of levels of each absorbed factor."""
    return tuple(int(c.max()) + 1 for c in codes)
