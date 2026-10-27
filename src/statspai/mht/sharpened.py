"""Benjamini-Krieger-Yekutieli (2006) sharpened two-stage FDR q-values.

The default reproduces Michael Anderson's Stata do-file
``fdr_sharpened_qvalues.do`` (last modified 11/20/07, distributed with
Anderson 2008), which applied economists use to report "sharpened
q-values". That file does not solve for the q-value. It runs the two-stage
procedure at q = 1, 0.999, ..., 0.001 and records the smallest grid level
at which each hypothesis is rejected, so its q-values are multiples of
0.001. This module repeats that grid search and the arithmetic of the
do-file, including what Stata stores in single precision: the p-values
(the ``pval`` variable is created as ``float``) and the two threshold
variables (``gen`` under the default ``set type float``). The q levels
themselves follow the do-file's local macro, which accumulates the
decrement ``- .001`` and is stored as a 17-character number.

``exact=True`` returns the infimum of the levels at which each hypothesis
is rejected, computed in double precision without the grid.
"""

from __future__ import annotations

import math
from typing import Literal, Sequence, Union

import numpy as np

from ..exceptions import MethodIncompatibility

PValueInput = Union[Sequence[float], np.ndarray]

#: Number of q levels the do-file visits (1.000 down to 0.001).
_N_GRID = 1000


def _stata_local(x: float) -> float:
    """Value that ``local m = x`` keeps in Stata.

    Stata writes the number into the macro as at most 17 characters: fixed
    notation with as many decimals as fit when ``1e-5 <= |x| < 1e16``
    (leading zero dropped), otherwise ``d.ddddddddddde+XX``. Checked in
    Stata 18 on 1/3 ... 1/3e18 and on the 1,000 levels of the do-file's
    loop. The values this module feeds it lie in [1e-4, 1e5].
    """
    if not math.isfinite(x) or x == 0.0:
        return x
    ax = abs(x)
    if 1e-5 <= ax < 1e16:
        int_digits = len(str(int(ax))) if ax >= 1.0 else 0
        decimals = 16 - int_digits - (1 if x < 0 else 0)
        return float(f"{x:.{max(decimals, 0)}f}")
    return float(f"{x:.11e}")


def _dofile_levels() -> np.ndarray:
    """The q levels of the do-file's ``while`` loop, as Stata holds them."""
    levels = np.empty(_N_GRID)
    q = 1.0
    for k in range(_N_GRID):
        levels[k] = q
        q = _stata_local(q - 0.001)
    return levels


_LEVELS = _dofile_levels()
#: ``local qval_adj = `qval' / ( 1 + `qval' )`` at each level.
_LEVELS_ADJ = np.array([_stata_local(q / (1.0 + q)) for q in _LEVELS])


def _max_rank_rejected(thresh32: np.ndarray, p32: np.ndarray) -> int:
    """Largest rank r with ``p_(r) <= threshold_r`` (0 when none)."""
    hit = np.flatnonzero(thresh32 >= p32)
    return int(hit[-1]) + 1 if hit.size else 0


def _dofile_qvalues(sorted_p: np.ndarray) -> np.ndarray:
    """Grid q-values for ascending p-values, as the do-file computes them."""
    M = sorted_p.size
    p32 = sorted_p.astype(np.float32)
    rank = np.arange(1, M + 1, dtype=float)
    out = np.ones(M)
    # The loop runs from q = 1 downwards and overwrites, so each hypothesis
    # ends with the smallest grid level at which it was rejected.
    for k, q_adj in enumerate(_LEVELS_ADJ):
        # gen fdr_temp1 = `qval_adj' * rank / `totalpvals'  (float storage)
        fdr1 = ((q_adj * rank) / M).astype(np.float32)
        r1 = _max_rank_rejected(fdr1, p32)
        if r1 == M:
            # M / (M - R1) is missing in Stata, so fdr_temp2 is missing and
            # `missing >= pval` is true: every hypothesis is rejected.
            r2 = M
        else:
            q_2st = _stata_local(q_adj * (M / (M - r1)))
            fdr2 = ((q_2st * rank) / M).astype(np.float32)
            r2 = _max_rank_rejected(fdr2, p32)
        if r2:
            # bky06_qval = `qval'; reported as the decimal grid level.
            out[:r2] = (_N_GRID - k) / _N_GRID
    return out


def _exact_qvalues(sorted_p: np.ndarray) -> np.ndarray:
    """Infimum q at which each ascending p-value is rejected (double)."""
    M = sorted_p.size
    rank = np.arange(1, M + 1, dtype=float)
    # Stage 1 rejects rank r once q' >= M p_(r) / r; R1(q') is the largest
    # such r, a step function of q' = q / (1 + q).
    t1 = M * sorted_p / rank
    # Within a stretch where R1 = k < M, stage 2 rejects rank i once
    # q' >= (M - k) * min_{r >= i} p_(r) / r.
    suffix_min = np.minimum.accumulate((sorted_p / rank)[::-1])[::-1]
    breaks = np.unique(t1)
    starts = np.concatenate(([0.0], breaks))
    ends = np.concatenate((breaks, [np.inf]))
    qprime = np.full(M, np.inf)
    todo = np.ones(M, dtype=bool)
    for a, b in zip(starts, ends):
        if not todo.any() or a > 0.5:
            break
        below = np.flatnonzero(t1 <= a)
        k = int(below[-1]) + 1 if below.size else 0
        if k == M:
            qprime[todo] = a
            todo[:] = False
            break
        need = np.maximum(a, (M - k) * suffix_min)
        hit = todo & (need < b)
        qprime[hit] = need[hit]
        todo &= ~hit
    q = np.ones(M)
    ok = qprime <= 0.5
    q[ok] = qprime[ok] / (1.0 - qprime[ok])
    return np.minimum(q, 1.0)


def sharpened_qvalues(
    pvalues: PValueInput,
    *,
    exact: bool = False,
    missing: Literal["count", "drop"] = "count",
) -> np.ndarray:
    """
    Sharpened FDR q-values of Benjamini, Krieger and Yekutieli (2006).

    The two-stage procedure first runs the Benjamini-Hochberg step-up at
    level q' = q / (1 + q), estimates the number of true nulls as M minus
    the number rejected, and runs the step-up again at q' M / m0. The
    q-value of a hypothesis is the smallest q at which it is rejected.
    Anderson (2008) brought these "sharpened q-values" to economics, and
    his Stata do-file ``fdr_sharpened_qvalues.do`` is how most papers
    compute them. The default reproduces that file.

    Parameters
    ----------
    pvalues : array-like
        Unadjusted p-values in [0, 1].
    exact : bool, default False
        False: search q = 1, 0.999, ..., 0.001 as the do-file does, so
        every q-value is a multiple of 0.001 and never below 0.001; a
        hypothesis not rejected at q = 1 gets 1. The p-values and the two
        threshold variables are rounded to single precision, as Stata
        stores them in the do-file, so values on a grid boundary land
        where Stata puts them. True: the infimum of the levels at which
        the hypothesis is rejected, in double precision (0 for p = 0,
        capped at 1). The grid value is this number rounded up to the
        next multiple of 0.001.
    missing : {'count', 'drop'}, default 'count'
        'count': a missing p-value stays missing and still counts towards
        the size of the family (as p = 1), as in :func:`adjust_pvalues`.
        'drop': the family is the non-missing p-values only, which is what
        the do-file does (it takes M from ``summarize``).

    Returns
    -------
    np.ndarray
        Sharpened q-values in the original order.

    Notes
    -----
    Sharpened q-values can be smaller than the unadjusted p-values when
    many hypotheses are rejected, as the do-file itself remarks. Tied
    p-values get the same q-value. The do-file stores ranks as Stata
    ``int``, so it cannot take more than 32,740 p-values; this function
    has no such limit.

    Examples
    --------
    >>> import statspai as sp
    >>> sp.sharpened_qvalues([0.001, 0.01, 0.03, 0.04, 0.2]).tolist()
    [0.006, 0.021, 0.031, 0.031, 0.053]
    >>> q = sp.sharpened_qvalues([0.001, 0.01, 0.03, 0.04, 0.2], exact=True)
    >>> [round(float(v), 4) for v in q]
    [0.005, 0.0204, 0.0309, 0.0309, 0.0526]

    References
    ----------
    [@benjamini2006adaptive], [@anderson2008multiple]
    """
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1:
        raise MethodIncompatibility(
            f"pvalues must be one-dimensional; got shape {p.shape}.",
            recovery_hint="Pass a flat list or 1-D array of p-values.",
        )
    if missing not in ("count", "drop"):
        raise MethodIncompatibility(
            f"missing must be 'count' or 'drop'; got {missing!r}.",
            recovery_hint="Use missing='count' (as sp.adjust_pvalues) or "
            "missing='drop' (as Anderson's do-file).",
        )
    nan = np.isnan(p)
    finite = p[~nan]
    if finite.size and (finite.min() < 0.0 or finite.max() > 1.0):
        raise MethodIncompatibility(
            "p-values must lie in [0, 1].",
            recovery_hint="Check that the input holds p-values, not test "
            "statistics or log p-values.",
            diagnostics={"min": float(finite.min()), "max": float(finite.max())},
        )
    out = np.full(p.size, np.nan)
    if missing == "drop":
        keep = np.flatnonzero(~nan)
        work = p[keep]
    else:
        keep = np.arange(p.size)
        work = np.where(nan, 1.0, p)
    if work.size == 0:
        return out
    order = np.argsort(work, kind="stable")
    sorted_p = work[order]
    q_sorted = _exact_qvalues(sorted_p) if exact else _dofile_qvalues(sorted_p)
    q = np.empty(work.size)
    q[order] = q_sorted
    out[keep] = q
    out[nan] = np.nan
    return out
