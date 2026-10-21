"""Covariate-tightened Lee (2009) bounds (``sp.lee_bounds(covariates=...)``).

Clean-room rewrite (2026-10-20); Stata ``leebounds, tight()`` was used only
as a black box. Specification, sources and experiments:
``docs/dev/cleanroom/leebounds.md``.

Lee (2009, Section III) tightens the bounds by trimming within cells of
discrete pre-treatment covariates and averaging the cell bounds over the
covariate distribution of the always-observed, estimated by the share of
each cell among the selected units of the less-retained arm (Tauchmann's
documentation of ``leebounds``). The arm trimmed in every cell is the one
more retained overall.

With ``trimming='leebounds'`` a cell whose trimmed arm is the less retained
one keeps its negative trimming share, as ``leebounds`` does (the tail mean
then over-weights the extreme observation); the other trimmings leave such a
cell untrimmed. Analytic variances exist for ``trimming='leebounds'`` only;
they reproduce ``leebounds``' approximation::

    Var = sum_c w_c^2 V_c + sum_c w_c (m_c - m)^2 / K + V_other

with ``V_c`` the trimmed-arm part of Lee's variance within cell ``c``,
``m_c`` the cell's trimmed mean, ``m`` their weighted average, ``V_other``
the untrimmed arm's term on the whole sample and ``K`` the number of
selected trimmed-arm units -- overall, unless a cell's threshold had tied
observations, in which case ``K`` is that count in the last such cell.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..exceptions import MethodIncompatibility


def tightened_lee_bounds(
    y: np.ndarray,
    D: np.ndarray,
    S: np.ndarray,
    cells: np.ndarray,
    trimming: str,
    analytic: bool = False,
) -> Dict[str, Any]:
    """Tightened bounds (and, for ``trimming='leebounds'``, Stata's variance).

    ``y`` / ``D`` / ``S`` / ``cells`` cover the whole estimation sample
    (outcome NaN where not selected). Returns a dict with ``lower``,
    ``upper``, ``hetero`` (some cell's trimmed arm is the less retained),
    ``trimmed_arm``, a per-cell table ``cells`` and, with ``analytic=True``,
    ``var_lower``, ``var_upper`` and ``between_cell_divisor`` (``K`` in the
    module docstring).
    """
    from .lee_manski import _lee_trimmed, _trimmed_arm_variance

    y = np.asarray(y, dtype=float)
    D = np.asarray(D, dtype=float)
    S = np.asarray(S, dtype=float)
    cells = np.asarray(cells)
    if analytic and trimming != "leebounds":
        raise MethodIncompatibility(
            "lee_bounds: analytic SEs of covariate-tightened bounds are "
            "available as Stata's leebounds approximates them "
            "(trimming='leebounds').",
            recovery_hint="Use se_method='bootstrap', or trimming='leebounds'.",
        )
    p1 = float(np.mean(S[D == 1]))
    p0 = float(np.mean(S[D == 0]))
    if p1 == p0:
        raise MethodIncompatibility(
            "lee_bounds: retention is identical in both arms; the effect is "
            "point identified and there is nothing to tighten.",
            recovery_hint="Drop covariates=; the bounds collapse to the "
            "difference in means.",
        )
    treated_trimmed = p1 > p0
    T = D if treated_trimmed else 1.0 - D
    codes = np.unique(cells)
    for c in codes:
        m = cells == c
        if not (np.any(m & (T == 1)) and np.any(m & (T == 0))):
            raise MethodIncompatibility(
                "lee_bounds: some covariate cells lack treated or control "
                "units; use coarser covariates (Stata: 'cells without "
                "variation in treatment').",
                recovery_hint="Merge sparse covariate levels.",
            )
    for c in codes:
        m = cells == c
        if not (np.any(m & (S == 1) & (T == 1)) and np.any(m & (S == 1) & (T == 0))):
            raise MethodIncompatibility(
                "lee_bounds: some covariate cells have no selected outcome in "
                "one arm; use coarser covariates.",
                recovery_hint="Merge sparse covariate levels.",
            )

    selected_other = (S == 1) & (T == 0)
    n_selected_other = int(np.sum(selected_other))
    mean_other = float(np.mean(y[selected_other]))
    divisor = float(np.sum((S == 1) & (T == 1)))
    rows: List[Dict[str, Any]] = []
    hetero = False
    for c in codes:
        m = cells == c
        S_c, T_c, y_c = S[m], T[m], y[m]
        keep_t = float(np.mean(S_c[T_c == 1]))
        keep_o = float(np.mean(S_c[T_c == 0]))
        share = (keep_t - keep_o) / keep_t
        if share < 0:
            hetero = True
            if trimming != "leebounds":
                share = 0.0
        y_t = y_c[(S_c == 1) & (T_c == 1)]
        row = {
            "cell": c,
            "n": int(np.sum(m)),
            "weight": float(np.sum(m & selected_other)) / n_selected_other,
            "trim": share,
            "mean_top": _lee_trimmed(y_t, share, True, trimming)[0],
            "mean_bottom": _lee_trimmed(y_t, share, False, trimming)[0],
            "retention_trimmed_arm": keep_t,
            "retention_other_arm": keep_o,
        }
        if analytic:
            tied = False
            for top, key in ((True, "var_top"), (False, "var_bottom")):
                v, _ = _trimmed_arm_variance(y_t, share, top, S_c, T_c, trimming)
                row[key] = v / len(S_c)
                tied |= _threshold_tied(y_t, share, top)
            if tied:
                divisor = float(len(y_t))
        rows.append(row)
    table = pd.DataFrame(rows)
    w = table["weight"].to_numpy()
    top_mean = float(np.sum(w * table["mean_top"].to_numpy()))
    bottom_mean = float(np.sum(w * table["mean_bottom"].to_numpy()))
    if treated_trimmed:
        lower, upper = bottom_mean - mean_other, top_mean - mean_other
    else:
        lower, upper = mean_other - top_mean, mean_other - bottom_mean
    out: Dict[str, Any] = {
        "cells": table,
        "hetero": hetero,
        "trimmed_arm": "treatment" if treated_trimmed else "control",
        "lower": float(lower),
        "upper": float(upper),
    }
    if analytic:
        n = len(S)
        v_other = (
            float(np.var(y[selected_other], ddof=1)) / float(np.mean(S * (1.0 - T))) / n
        )

        def total(mean_col: str, var_col: str) -> float:
            m_c = table[mean_col].to_numpy()
            between = float(np.sum(w * (m_c - np.sum(w * m_c)) ** 2))
            return (
                float(np.sum(w**2 * table[var_col].to_numpy()))
                + between / divisor
                + v_other
            )

        v_top, v_bottom = total("mean_top", "var_top"), total(
            "mean_bottom", "var_bottom"
        )
        out["var_lower"], out["var_upper"] = (
            (v_bottom, v_top) if treated_trimmed else (v_top, v_bottom)
        )
        out["between_cell_divisor"] = divisor
    return out


def _threshold_tied(y: np.ndarray, share: float, top: bool) -> bool:
    from .lee_manski import _leebounds_trimmed

    return bool(_leebounds_trimmed(y, share, top)[4])


def cell_codes(data: pd.DataFrame, covariates: List[str]) -> np.ndarray:
    """Integer codes 0, 1, ... of the covariate combinations, in sorted order."""
    _ret: np.ndarray = data.groupby(list(covariates), sort=True).ngroup().to_numpy()
    return _ret
