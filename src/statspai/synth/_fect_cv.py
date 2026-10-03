"""
Cross-validation for :func:`statspai.fect` (port of ``fect``'s selectors).

Ports the tuning-parameter selection of Liu, Wang and Xu's **fect** R
package (2.4.x) for the two outcome models that need it:

- the number of factors ``r`` under ``method="ife"``;
- the nuclear-norm penalty ``lam`` under ``method="mc"``.

Each function mirrors one R function of the package, named in its
docstring: ``fect:::cv.sample`` (block holdout draws),
``fect:::.build_cv_mask_rolling`` (rolling holdout), the fold loop and
grid construction of ``fect:::fect_cv``, ``fect:::.score_residuals``
(loss functions), ``fect:::.fect_cv_aggregate_folds`` (pooled score and
fold-level standard error) and ``fect:::.fect_apply_cv_rule`` (the
``cv.rule`` re-pick). Cell indices are column-major (unit-major) like
R's ``c(matrix)`` so the two code paths can be read side by side.

Fold draws are random on both sides and R's stream cannot be reproduced
from numpy, so the port is checked statistically -- a stochastic screen,
not a seed-replicated equivalence test: given identical folds the per-fold
scores equal R's; on a fixed panel the StatsPAI CV curve must lie inside
the across-seed spread of the R curves and the selection must fall in
R's modal choice (fect) or flat-loss plateau (interflex) (see
``tests/reference_parity/test_fect_interflex_cv_parity.py``).

References
----------
[@liu2024practical]; [@xu2017generalized]; [@athey2021matrix].
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..exceptions import DataInsufficient, MethodIncompatibility

CRITERIA = ("mspe", "gmspe", "moment", "pc")
CV_METHODS = ("block", "treated_units", "rolling")
CV_RULES = ("1se", "min", "1pct")
# fect.default's ``proportion`` (pre-treatment periods whose cell count is
# below this share of the largest one get zero weight in the ``moment``
# criterion); only that criterion depends on it.
_MOMENT_PROPORTION = 0.3
# fect initialises every CV score to 1e20 before the scan.
_UNSET = 1e20


# ----------------------------------------------------------------------
# Holdout construction
# ----------------------------------------------------------------------


def _cv_sample(
    II: np.ndarray,
    D: np.ndarray,
    count: int,
    cv_count: int,
    cv_treat: bool,
    cv_donut: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Port of ``fect:::cv.sample``: one draw of ``count`` untreated
    cells in ``cv_count``-period blocks within units.

    Returns ``(cv_id, est_id)``: the cells hidden from the fit and the
    subset scored (block interior after removing ``cv_donut`` cells at
    each end), as 0-based column-major cell indices.
    """
    T, N = II.shape
    flat_I = II.ravel(order="F")
    tr_pos = np.where(D.sum(axis=0) >= 1)[0]
    prop = (T * N) / float(II.sum())
    if not cv_treat:
        oci = np.where(flat_I == 1)[0]
    else:
        fake = np.zeros((T, N))
        fake[:, tr_pos] = 1.0
        oci = np.where((flat_I == 1) & (fake.ravel(order="F") == 1))[0]
        if len(oci) <= count:
            raise DataInsufficient(
                "Too few untreated cells of treated units for cross-validation.",
                recovery_hint="Use cv_method='block' or a smaller cv_prop.",
                diagnostics={"n_cells": int(len(oci)), "count": int(count)},
            )
    if cv_count == 1 or count <= 2:
        cv_id = np.sort(rng.choice(oci, size=count, replace=False))
        return cv_id, cv_id.copy()
    res = T % cv_count
    n_int = T // cv_count
    subcount = int(np.floor(count / cv_count * prop))
    if subcount == 0:
        subcount = 1
    units = np.arange(N) if not cv_treat else tr_pos
    starts = res + np.arange(n_int) * cv_count
    rm_pos = (
        np.concatenate([T * j + starts for j in units])
        if len(units)
        else np.zeros(0, int)
    )
    rm_id = rng.choice(rm_pos, size=min(subcount, len(rm_pos)), replace=False)
    if cv_count == 2:
        rm_all = np.concatenate([rm_id, rm_id + 1])
        rm_use = rm_all.copy()
    else:
        all_parts = [rm_id]
        use_parts: List[np.ndarray] = []
        for i in range(cv_count):
            all_parts.append(rm_id + i)
            if cv_donut <= i <= cv_count - cv_donut - 1:
                use_parts.append(rm_id + i)
        rm_all = np.concatenate(all_parts)
        rm_use = np.concatenate(use_parts) if use_parts else np.zeros(0, int)
    oci_set = set(oci.tolist())
    rm_all = np.array(sorted({int(x) for x in rm_all if int(x) in oci_set}), dtype=int)
    rm_use = np.array(sorted({int(x) for x in rm_use if int(x) in oci_set}), dtype=int)
    if len(rm_all) >= count:
        cv_id = rm_all[:count]
        rm_use = rm_use[(rm_use >= cv_id.min()) & (rm_use <= cv_id.max())]
    else:
        pool = np.array(sorted(oci_set - set(rm_all.tolist())), dtype=int)
        cv_id2 = rng.choice(pool, size=count - len(rm_all), replace=False)
        cv_id = np.sort(np.concatenate([rm_all, cv_id2]))
    return cv_id.astype(int), rm_use.astype(int)


def _block_folds(
    II: np.ndarray,
    D: np.ndarray,
    k: int,
    cv_prop: float,
    cv_nobs: int,
    cv_treat: bool,
    cv_donut: int,
    min_t0: int,
    rng: np.random.Generator,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """The fold loop of ``fect:::fect_cv`` for ``cv.method`` ``"block"``
    (alias ``"all_units"``) and ``"treated_units"``:
    redraw ``cv.sample`` until every period keeps at least one untreated
    cell and every unit keeps at least ``min.T0``; after 200 failed
    draws restore the offending rows / columns instead."""
    T, N = II.shape
    count = int(np.floor(II.sum() * cv_prop))
    if count < 1:
        raise DataInsufficient(
            "cv_prop leaves no cell to hold out.",
            recovery_hint="Increase cv_prop.",
            diagnostics={"n_untreated_cells": int(II.sum()), "cv_prop": cv_prop},
        )
    flat_II = II.ravel(order="F")
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    flagged = False
    for _ in range(k):
        n_try = 0
        while True:
            n_try += 1
            cv_id, est_id = _cv_sample(II, D, count, cv_nobs, cv_treat, cv_donut, rng)
            flat_cv = flat_II.copy()
            flat_cv[cv_id] = 0
            II_cv = flat_cv.reshape((T, N), order="F")
            con1 = bool(np.all(II_cv.sum(axis=1) >= 1))
            con2 = bool(np.all(II_cv.sum(axis=0) >= min_t0))
            if con1 and con2:
                break
            if n_try >= 200:
                flagged = True
                keep_rows = np.where(II_cv.sum(axis=1) < 1)[0]
                keep_cols = np.where(II_cv.sum(axis=0) < min_t0)[0]
                valid_flat = flat_II.astype(float).copy()
                valid_flat[cv_id] = -1.0
                valid = valid_flat.reshape((T, N), order="F").copy()
                valid[keep_rows, :] = II[keep_rows, :]
                valid[:, keep_cols] = II[:, keep_cols]
                new_cv = np.where(valid.ravel(order="F") != flat_II)[0]
                dropped = set(cv_id.tolist()) - set(new_cv.tolist())
                est_id = np.array(sorted(set(est_id.tolist()) - dropped), dtype=int)
                cv_id = new_cv
                break
        if len(cv_id) == 0:
            raise DataInsufficient(
                "Some units have too few pre-treatment observations for "
                "cross-validation.",
                recovery_hint="Set a larger cv_prop or cv_method='block'.",
            )
        folds.append((np.asarray(cv_id, dtype=int), np.asarray(est_id, dtype=int)))
    if flagged:
        warnings.warn(
            "fect cv: some units have too few pre-treatment observations; "
            "they were kept out of the holdout, as in fect.",
            UserWarning,
            stacklevel=3,
        )
    return folds


def _rolling_folds(
    II: np.ndarray,
    D: np.ndarray,
    k: int,
    cv_nobs: int,
    cv_buffer: int,
    cv_prop: float,
    min_t0: int,
    rng: np.random.Generator,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Port of ``fect:::.build_cv_mask_rolling`` (``cv.method="rolling"``,
    fect's default): per fold, a ``cv.prop`` share of the eligible units (at least
    ``min.T0 + cv.nobs`` untreated pre-onset periods) each gets one
    random anchor; the ``cv.nobs`` periods from the anchor are scored,
    the ``cv.buffer`` periods before it and everything after are hidden."""
    T, N = II.shape
    elig: Dict[int, np.ndarray] = {}
    for j in range(N):
        obs_t = np.where(II[:, j] == 1)[0]
        treated_t = np.where(D[:, j] >= 1)[0]
        if len(treated_t):
            obs_t = obs_t[obs_t < treated_t.min()]
        if len(obs_t) >= min_t0 + cv_nobs:
            elig[j] = obs_t
    if not elig:
        raise DataInsufficient(
            "No unit has enough untreated periods for rolling cross-validation.",
            recovery_hint=f"Need min_t0 + cv_nobs = {min_t0 + cv_nobs} periods.",
        )
    units = np.array(sorted(elig))
    n_sample = max(1, int(round(cv_prop * len(units))))
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(k):
        sampled = (
            units
            if n_sample >= len(units)
            else np.sort(rng.choice(units, size=n_sample, replace=False))
        )
        cv_acc: List[int] = []
        est_acc: List[int] = []
        for j in sampled:
            obs_t = elig[int(j)]
            n_obs = len(obs_t)
            valid = np.arange(min_t0, n_obs - cv_nobs + 1)
            if len(valid) == 0:
                continue
            a = int(rng.choice(valid))
            base = int(j) * T
            holdout = obs_t[a : a + cv_nobs]
            buf = obs_t[max(0, a - cv_buffer) : a] if cv_buffer > 0 else obs_t[:0]
            drop = obs_t[a + cv_nobs :]
            cv_acc += (base + np.concatenate([holdout, buf, drop])).tolist()
            est_acc += (base + holdout).tolist()
        folds.append(
            (
                np.array(sorted(set(cv_acc)), dtype=int),
                np.array(sorted(set(est_acc)), dtype=int),
            )
        )
    return folds


# ----------------------------------------------------------------------
# Scores and selection rules
# ----------------------------------------------------------------------


def _count_weights(T_on: np.ndarray, proportion: float) -> Dict[str, float]:
    """``count.T.cv`` of ``fect:::fect_cv``: relative-period cell counts
    (periods <= 0) normalised by their mean, ``"Control"`` set to their
    median, and periods with fewer than ``proportion`` x the largest
    count zeroed. Used by the ``moment`` criterion only."""
    vals = T_on[~np.isnan(T_on)]
    vals = vals[vals <= 0]
    keys, counts = np.unique(vals.astype(int), return_counts=True)
    counts = counts.astype(float)
    cut = counts.max() * proportion
    drop = counts <= cut
    norm = counts / counts.mean()
    out = {str(int(kk)): float(v) for kk, v in zip(keys, norm)}
    out["Control"] = float(np.median(norm))
    for kk, d in zip(keys, drop):
        if d:
            out[str(int(kk))] = 0.0
    return out


def _score(
    resid: np.ndarray, time_idx: List[str], count_w: Dict[str, float]
) -> Dict[str, float]:
    """Port of ``fect:::.score_residuals`` for the criteria fect.default
    accepts: MSPE, GMSPE (geometric mean of squared errors) and Moment
    (count-weighted mean over relative periods of |mean residual|)."""
    n = resid.size
    if n == 0:
        raise DataInsufficient("No residuals collected from the CV folds.")
    e2 = resid**2
    mspe = float(e2.sum() / n)
    with np.errstate(divide="ignore"):
        gmspe = float(np.exp(np.sum(np.log(e2)) / n))
    groups: Dict[str, List[float]] = {}
    for r_, t_ in zip(resid.tolist(), time_idx):
        groups.setdefault(t_, []).append(r_)
    names = sorted(groups)
    means = np.array([abs(float(np.mean(groups[g]))) for g in names])
    w = np.array([count_w.get(g, 0.0) for g in names])
    moment = float(np.sum(w * means) / np.sum(w)) if np.sum(w) > 0 else float("nan")
    return {"MSPE": mspe, "GMSPE": gmspe, "Moment": moment}


def _aggregate(fold_results: List[Tuple[np.ndarray, List[str]]], count_w: Any) -> Any:
    """Port of ``fect:::.fect_cv_aggregate_folds``: pooled score over all
    holdout residuals plus the fold-level standard error
    ``sd(fold scores) / sqrt(#finite folds)``."""
    all_resid = np.concatenate([r for r, _ in fold_results])
    all_idx = [t for _, ts in fold_results for t in ts]
    pooled = _score(all_resid, all_idx, count_w)
    per_fold = [_score(r, ts, count_w) for r, ts in fold_results if r.size]
    ses = {}
    for key in pooled:
        x = np.array([f[key] for f in per_fold], dtype=float)
        x = x[np.isfinite(x)]
        ses[key] = float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
    folds = pd.DataFrame(per_fold)
    return pooled, ses, folds


def _apply_cv_rule(means: np.ndarray, ses: np.ndarray, rule: str) -> Optional[int]:
    """Port of ``fect:::.fect_apply_cv_rule``: the smallest grid index
    (smallest ``r`` / largest ``lambda``) whose mean score is within one
    fold-level SE of the minimum (``"1se"``, fect's default), within 1 %
    (``"1pct"``), or the minimum itself (``"min"``)."""
    valid = np.isfinite(means)
    if not valid.any():
        return None
    m_min = np.min(means[valid])
    i_min = int(np.min(np.where(valid & (means == m_min))[0]))
    if rule == "min":
        return i_min
    if rule == "1pct":
        thr = m_min * 1.01
    else:
        se_min = ses[i_min] if np.isfinite(ses[i_min]) else 0.0
        thr = means[i_min] + se_min
    return int(np.min(np.where(valid & (means <= thr))[0]))


def _improves(col: np.ndarray, val: float) -> bool:
    """fect's scan rule: a candidate replaces the running best only if it
    lowers the current minimum (initialised at 1e20) by more than 1 %."""
    m = float(col.min())
    return (m - val) > 0.01 * m


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


def _relative_period_index(T_on: np.ndarray, cells: np.ndarray) -> List[str]:
    flat = T_on.ravel(order="F")[cells]
    return ["Control" if np.isnan(v) else str(int(v)) for v in flat]


def run_fect_cv(
    Yz: np.ndarray,
    D: np.ndarray,
    I: np.ndarray,
    X: Optional[np.ndarray],
    method: str,
    force: int,
    tol: float,
    max_iter: int,
    *,
    r_range: Optional[Sequence[int]],
    lambda_grid: Optional[Sequence[float]],
    nlambda: int,
    k: int,
    cv_prop: float,
    cv_method: str,
    cv_nobs: int,
    cv_donut: int,
    cv_buffer: int,
    min_t0: int,
    criterion: str,
    cv_rule: str,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Cross-validate ``r`` (``method="ife"``) or ``lam`` (``method="mc"``)
    exactly as ``fect:::fect_cv`` does, returning the selected value and
    the full CV table. See the module docstring for the R functions each
    step mirrors."""
    from .fect import _get_term, _initial_fit, _inter_fe_core

    if criterion not in CRITERIA:
        raise ValueError(f"criterion must be one of {CRITERIA}")
    if cv_method not in CV_METHODS:
        raise ValueError(f"cv_method must be one of {CV_METHODS}")
    if cv_rule not in CV_RULES:
        raise ValueError(f"cv_rule must be one of {CV_RULES}")
    if criterion == "pc" and method != "ife":
        raise MethodIncompatibility("criterion='pc' is defined for method='ife' only")
    if int(k) < 1:
        raise ValueError("k must be >= 1")
    T, N = Yz.shape
    p = 0 if X is None else X.shape[2]
    II = ((I == 1) & (D == 0)).astype(int)
    cv_tol = max(float(tol), 1e-3)  # fect: cv_tol <- max(tol, 0.001)
    Y0, beta0 = _initial_fit(Yz, X, II, force)
    T_on = np.full((T, N), np.nan)
    for j in range(N):
        if I[:, j].sum() > 0:
            T_on[:, j] = _get_term(D[:, j], I[:, j])
    count_w = _count_weights(T_on, _MOMENT_PROPORTION)

    # ---- grid -----------------------------------------------------------
    if method == "ife":
        if r_range is None:
            r_lo, r_hi = 0, 5  # fect's recommended r = c(0, 5)
        else:
            rr = [int(v) for v in r_range]
            if len(rr) == 2:
                r_lo, r_hi = rr
            else:
                r_lo, r_hi = min(rr), max(rr)
            if r_lo < 0 or r_hi < r_lo:
                raise ValueError(
                    "r_range must be (r_min, r_max) with 0 <= r_min <= r_max"
                )
        # fect caps r.end by the data: enough cells to estimate the factors
        # and fewer factors than the shortest untreated history.
        r_end = r_hi
        colsum = II.sum(axis=0)
        t0_min = int(colsum[colsum > 0].min())
        while r_end > r_lo and (II.sum() - r_end * (N + T) + r_end**2 - p) <= 0:
            r_end -= 1
        if r_end >= t0_min:
            r_end = max(r_lo, t0_min - 1)
        if r_end != r_hi:
            warnings.warn(
                f"fect cv: the factor-number grid was capped at r={r_end} "
                "(fewer than min untreated periods and enough cells), as in fect.",
                UserWarning,
                stacklevel=3,
            )
        grid = [float(v) for v in range(r_lo, r_end + 1)]
        eigen_max = None
    else:
        Y_lambda = np.where(II == 1, Yz - Y0, 0.0)
        eigen_max = float(np.linalg.svd(Y_lambda / (T * N), compute_uv=False).max())
        if lambda_grid is not None:
            grid = [float(v) for v in lambda_grid]
            if any(v < 0 for v in grid):
                raise ValueError("lambda_grid must be non-negative")
        else:
            nl = int(nlambda)
            if nl < 3:
                raise ValueError(
                    "nlambda must be >= 3 (fect's grid needs nlambda - 2 steps)"
                )
            lmax = np.log10(eigen_max)
            by = 3.0 / (nl - 2)
            grid = [10.0 ** (lmax - i * by) for i in range(nl - 1)] + [0.0]
    n_grid = len(grid)

    # ---- folds (skipped for the PC criterion, which uses no holdout) ----
    need_folds = criterion != "pc"
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    if need_folds:
        if cv_method == "rolling":
            folds = _rolling_folds(
                II, D, int(k), cv_nobs, cv_buffer, cv_prop, min_t0, rng
            )
        else:
            folds = _block_folds(
                II,
                D,
                int(k),
                cv_prop,
                cv_nobs,
                cv_method == "treated_units",
                cv_donut,
                min_t0,
                rng,
            )
        for cv_id, est_id in folds:
            if len(est_id) == 0:
                raise DataInsufficient(
                    "A CV fold has no scored cell; with cv_nobs=1 or a larger "
                    "cv_donut nothing is left inside the holdout block.",
                    recovery_hint="Lower cv_donut or raise cv_nobs.",
                )
    fold_init: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    flat_II = II.ravel(order="F")
    for cv_id, _ in folds:
        f = flat_II.copy()
        f[cv_id] = 0
        II_cv = f.reshape((T, N), order="F")
        Y0cv, b0cv = _initial_fit(Yz, X, II_cv, force)
        fold_init.append((II_cv, Y0cv, b0cv))

    # ---- scan -----------------------------------------------------------
    crit_col = {"mspe": "MSPE", "gmspe": "GMSPE", "moment": "Moment", "pc": "PC"}[
        criterion
    ]
    rows: List[Dict[str, float]] = []
    fold_tables: List[pd.DataFrame] = []
    col = np.full(n_grid, _UNSET)
    scan_best: Optional[int] = None
    break_check = 0
    break_count = 0
    evaluated = 0
    flat_Y = Yz.ravel(order="F")
    for gi, gval in enumerate(grid):
        row: Dict[str, float] = {}
        if method == "ife":
            r_i = int(gval)
            full = _inter_fe_core(Yz, Y0, X, II, beta0, r_i, force, cv_tol, max_iter)
            row.update(
                {
                    "r": r_i,
                    "sigma2": float(full["sigma2"]),
                    "IC": float(full["IC"]),
                    "PC": float(full["PC"]),
                }
            )
        else:
            row.update({"lambda": gval, "lambda_norm": gval / eigen_max})
        if need_folds:
            fold_res = []
            for (cv_id, est_id), (II_cv, Y0cv, b0cv) in zip(folds, fold_init):
                if method == "ife":
                    fit = _inter_fe_core(
                        Yz, Y0cv, X, II_cv, b0cv, int(gval), force, cv_tol, max_iter
                    )["fit"]
                else:
                    fit = _inter_fe_core(
                        Yz,
                        Y0cv,
                        X,
                        II_cv,
                        b0cv,
                        1,
                        force,
                        cv_tol,
                        max_iter,
                        mc=1,
                        lam=gval,
                    )["fit"]
                resid = flat_Y[est_id] - fit.ravel(order="F")[est_id]
                fold_res.append((resid, _relative_period_index(T_on, est_id)))
            pooled, ses, per_fold = _aggregate(fold_res, count_w)
            row.update(pooled)
            row.update({f"{kk}_se": v for kk, v in ses.items()})
            row.update({f"{kk}_fold_mean": float(per_fold[kk].mean()) for kk in pooled})
            fold_tables.append(per_fold[crit_col].rename(gi))
            val = pooled[crit_col]
            if _improves(col, val):
                scan_best = gi
                break_check = 0
                break_count = 0
            elif method == "mc" and gi > 0 and scan_best == gi - 1:
                break_check = 1
                break_count = 0
            if break_check == 1:
                break_count += 1
            col[gi] = val
        else:
            val = row["PC"]
            if val < col.min():
                scan_best = gi
            col[gi] = val
        rows.append(row)
        evaluated += 1
        if method == "mc" and break_count == 3:
            break  # fect stops the lambda scan after 3 non-improving steps

    # ---- cv.rule re-pick (fect's default "1se") -------------------------
    means = col.copy()
    means[~np.isfinite(means) | (means >= 1e19)] = np.nan
    if criterion == "pc":
        pick = scan_best
    else:
        ses_arr = np.full(n_grid, np.nan)
        for gi, row in enumerate(rows):
            ses_arr[gi] = row.get(f"{crit_col}_se", np.nan)
        pick = _apply_cv_rule(means, ses_arr, cv_rule)
        if pick is None:
            pick = scan_best
    if pick is None:
        raise DataInsufficient("Cross-validation produced no finite score.")
    table = pd.DataFrame(rows)
    fold_scores = (
        pd.concat(fold_tables, axis=1).T.reset_index(drop=True)
        if fold_tables
        else pd.DataFrame()
    )
    if not fold_scores.empty:
        fold_scores.columns = [f"fold_{i + 1}" for i in range(fold_scores.shape[1])]
        fold_scores.insert(
            0, "r" if method == "ife" else "lambda", grid[: len(fold_scores)]
        )
    return {
        "method": method,
        "criterion": criterion,
        "cv_rule": cv_rule,
        "cv_method": cv_method,
        "k": int(k),
        "cv_prop": float(cv_prop),
        "cv_nobs": int(cv_nobs),
        "cv_donut": int(cv_donut),
        "cv_buffer": int(cv_buffer),
        "min_t0": int(min_t0),
        "cv_tol": cv_tol,
        "grid": grid,
        "n_evaluated": evaluated,
        "table": table,
        "fold_scores": fold_scores,
        "n_holdout_per_fold": [int(len(c)) for c, _ in folds],
        "n_scored_per_fold": [int(len(e)) for _, e in folds],
        "scan_selected": grid[scan_best] if scan_best is not None else None,
        "selected": grid[pick],
        "selected_index": int(pick),
        "lambda_max": eigen_max,
    }
