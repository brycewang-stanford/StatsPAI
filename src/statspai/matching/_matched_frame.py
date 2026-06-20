"""Assemble Stata ``psmatch2``-style post-matching variables.

After propensity-score matching, Stata's ``psmatch2`` writes a handful of
per-observation variables back into the dataset so the analyst can run
post-matching balance tests, plot the matched propensity score distribution,
and estimate frequency-weighted PSM-DID regressions.  The discrete-neighbour
columns (``_n1`` … ``_nk``, ``_nn``, ``_pdif``) are nearest-neighbour only;
kernel and radius matching emit ``_weight`` / ``_y`` without those columns:

================  ====================================================
Variable          Meaning
================  ====================================================
``_id``           Running observation id over the estimation sample.
``_treated``      Treatment indicator (1 treated, 0 control).
``_pscore``       Estimated propensity score.
``_support``      Common-support indicator (1 on support, 0 off).
``_weight``       Frequency weight of the observation in the matched
                  sample.  Treated-on-support = 1; a control used as a
                  match accumulates the share(s) assigned to it; rows
                  outside the matched sample are missing (``NaN``).
``_n1`` … ``_nk`` ``_id`` of the 1st … k-th matched control (treated
                  rows only).
``_nn``           Number of matched controls (0 on control rows, like
                  psmatch2).
``_pdif``         |Δ propensity score| between a treated unit and its
                  *nearest* matched control (treated rows only).
``_y``            Mean outcome of the matched control(s) (treated rows
                  only); emitted only when an outcome is supplied.
================  ====================================================

The per-row semantics were verified against Stata 18 ``psmatch2`` (Leuven &
Sianesi 2003): treated rows carry ``_weight = 1``; a control's ``_weight``
is the sum of the ``1/k`` shares it receives across the treated units that
matched it; ``_pdif`` is the propensity-score gap to the *nearest* match
only (it is identical under ``neighbor(1)`` and ``neighbor(2)``); and the
ATT equals the weighted mean of ``y - _y`` over treated rows.  Stata's
``_id`` is an internal sort key, so absolute ``_id`` / ``_n{j}`` *labels*
need not coincide with psmatch2's — but they reference the same physical
observations and every ``_weight`` / ``_pscore`` / ``_pdif`` value matches
row-for-row.

This module is intentionally *pure bookkeeping*: it takes the match
assignments that the estimator already computed for its point estimate and
turns them into the columns above.  It never re-runs matching and never
touches the ATT, so attaching a matched frame to an existing estimator is
numerically inert.

References
----------
Leuven, E. and Sianesi, B. (2003). PSMATCH2: Stata module to perform full
    Mahalanobis and propensity score matching, common support graphing, and
    covariate imbalance testing.  Statistical Software Components S432001,
    Boston College Department of Economics.
Rosenbaum, P.R. and Rubin, D.B. (1983). Biometrika, 70(1), 41-55.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# Stata-faithful column names (psmatch2).  Kept in one place so the public
# functions, the result object and the tests all agree on the spelling.
COL_ID = "_id"
COL_TREATED = "_treated"
COL_PSCORE = "_pscore"
COL_SUPPORT = "_support"
COL_WEIGHT = "_weight"
COL_NN = "_nn"
COL_PDIF = "_pdif"
COL_Y = "_y"


def neighbor_col(j: int) -> str:
    """Name of the ``_n{j}`` neighbour column (1-based)."""
    return f"_n{j}"


def common_support_mask(
    pscore: np.ndarray,
    treated: np.ndarray,
    *,
    rule: str = "minmax",
) -> np.ndarray:
    """Common-support indicator over a propensity-score vector.

    Parameters
    ----------
    pscore : ndarray
        Estimated propensity scores (positional, length ``n``).
    treated : ndarray
        0/1 treatment indicator aligned with ``pscore``.
    rule : {'minmax', 'none'}, default 'minmax'
        ``'minmax'`` follows psmatch2's ``comsup``: a *treated* unit is on
        support iff its propensity score lies within the [min, max] range of
        the *control* scores; controls are always on support.  ``'none'``
        marks every observation on support.

    Returns
    -------
    ndarray of bool
    """
    pscore = np.asarray(pscore, dtype=float)
    treated = np.asarray(treated)
    if rule == "none":
        return np.ones(len(pscore), dtype=bool)
    if rule != "minmax":
        raise ValueError(f"rule must be 'minmax' or 'none', got {rule!r}")

    ctrl = treated == 0
    if not np.any(ctrl):
        return np.ones(len(pscore), dtype=bool)
    lo = float(np.min(pscore[ctrl]))
    hi = float(np.max(pscore[ctrl]))
    on = np.ones(len(pscore), dtype=bool)
    trt = treated == 1
    on[trt] = (pscore[trt] >= lo) & (pscore[trt] <= hi)
    return on


def psmatch2_se(
    outcome: np.ndarray,
    treated: np.ndarray,
    support: np.ndarray,
    weight: np.ndarray,
) -> float:
    """Stata ``psmatch2`` default analytic ATT standard error.

    Reproduces, digit for digit, the formula in ``psmatch2.ado``
    (Leuven & Sianesi 2003)::

        seatt = sqrt(var1 / N1  +  var0 * wtot / N1^2)

    where

    * ``var1`` is the (ddof=1) outcome variance among the **treated on
      support**,
    * ``var0`` is the (ddof=1) outcome variance among the **controls that
      were actually used as matches** (``_weight`` not missing),
    * ``wtot`` is the sum of squared control ``_weight``\\ s, and
    * ``N1`` is the number of treated units on support.

    This treats the matching weights as fixed and assumes homoskedastic,
    independent outcomes within group (Lechner 2001).  It does **not**
    account for the estimation of the propensity score — exactly matching
    Stata's note.  It applies identically to nearest-neighbour, kernel and
    radius matching because all three feed it the same ``_weight`` column.

    Parameters
    ----------
    outcome : ndarray
        Outcome values (positional, length ``n``).
    treated : ndarray
        0/1 treatment indicator.
    support : ndarray
        Common-support flag (1/0 or bool).
    weight : ndarray
        The ``_weight`` column (``NaN`` outside the matched sample).

    Returns
    -------
    float
        The analytic SE, or ``nan`` if it cannot be formed (e.g. fewer than
        two treated, or no used controls).
    """
    y = np.asarray(outcome, dtype=float)
    t = np.asarray(treated)
    s = np.asarray(support).astype(bool)
    w = np.asarray(weight, dtype=float)

    treated_on = (t == 1) & s
    n1 = int(np.sum(treated_on))
    used_c = (t == 0) & np.isfinite(w)
    if n1 < 2 or np.sum(used_c) < 2:
        return float("nan")

    var1 = float(np.var(y[treated_on], ddof=1))
    var0 = float(np.var(y[used_c], ddof=1))
    wtot = float(np.sum(w[used_c] ** 2))
    return float(np.sqrt(var1 / n1 + var0 * wtot / n1**2))


def _within_group_self_outcome(
    outcome: np.ndarray,
    treated: np.ndarray,
    pscore: np.ndarray,
    n_ai_matches: int,
) -> np.ndarray:
    """psmatch2's ``_self_y``: mean outcome of the J nearest *same-group* units.

    For every unit, the ``n_ai_matches`` nearest neighbours **within its own
    treatment arm** (by propensity-score distance, ties broken by index) are
    found and their outcomes averaged.  This is the conditional-mean estimate
    used by the Abadie-Imbens (2006) variance to gauge the within-cell outcome
    noise ``σ²(X)``.
    """
    y = np.asarray(outcome, dtype=float)
    t = np.asarray(treated)
    ps = np.asarray(pscore, dtype=float)
    n = len(y)
    self_y = np.full(n, np.nan)
    j = max(int(n_ai_matches), 1)
    for g in (0, 1):
        idx = np.where(t == g)[0]
        if len(idx) < 2:
            continue
        for i in idx:
            others = idx[idx != i]
            order = np.argsort(np.abs(ps[i] - ps[others]), kind="stable")
            knn = others[order[: min(j, len(others))]]
            self_y[i] = float(np.mean(y[knn]))
    return self_y


def abadie_imbens_se(
    outcome: np.ndarray,
    treated: np.ndarray,
    pscore: np.ndarray,
    support: np.ndarray,
    weight: np.ndarray,
    n_ai_matches: int = 1,
) -> float:
    """Abadie-Imbens (2006) heteroskedasticity-robust ATT standard error.

    Reproduces, digit for digit, Stata ``psmatch2 , ai(J)`` (Leuven &
    Sianesi 2003; Abadie & Imbens 2006, eq. 14, p. 250 for the sample ATT)::

        shat_i  = (J / (J+1)) · (Y_i − Ȳ_self_i)²          # σ²(X_i) estimate
        VhatEt_i = shat_i · (D_i − (1 − D_i)·w_i)²          # on support
        seatt    = sqrt(Σ_i VhatEt_i) / N1

    where ``Ȳ_self_i`` is the mean outcome of the ``J`` nearest *same-arm*
    neighbours (:func:`_within_group_self_outcome`), ``w_i = max(_weight_i,
    0)`` is the matching weight, ``D_i`` the treatment indicator and ``N1``
    the number of treated on support.  Unlike :func:`psmatch2_se` (which
    assumes homoskedastic outcomes within arm), this allows ``σ²(X)`` to vary
    with the covariates.

    Parameters
    ----------
    outcome, treated, pscore, support, weight : ndarray
        Positional arrays over the estimation sample.  ``pscore`` drives the
        within-arm neighbour search; ``weight`` is the matched-sample
        ``_weight`` column (``NaN`` outside the matched sample).
    n_ai_matches : int, default 1
        Number of within-arm matches ``J`` (Stata's ``ai(J)``).

    Returns
    -------
    float
        The robust SE, or ``nan`` if it cannot be formed.

    References
    ----------
    Abadie, A. and Imbens, G.W. (2006). Large Sample Properties of Matching
        Estimators for Average Treatment Effects. *Econometrica*, 74(1),
        235-267.
    """
    y = np.asarray(outcome, dtype=float)
    t = np.asarray(treated)
    s = np.asarray(support).astype(bool)
    w = np.asarray(weight, dtype=float)
    j = max(int(n_ai_matches), 1)

    treated_on = (t == 1) & s
    n1 = int(np.sum(treated_on))
    if n1 < 1:
        return float("nan")

    self_y = _within_group_self_outcome(y, t, pscore, j)
    shat = (j / (j + 1.0)) * (y - self_y) ** 2
    w_pos = np.where(np.isfinite(w), np.maximum(w, 0.0), 0.0)
    vhat = shat * (t - (1 - t) * w_pos) ** 2
    total = float(np.nansum(vhat[s]))
    if not np.isfinite(total) or total < 0:
        return float("nan")
    return float(np.sqrt(total) / n1)


def build_matched_frame(
    *,
    index: pd.Index,
    treated: np.ndarray,
    pscore: np.ndarray,
    idx_t: np.ndarray,
    idx_c: np.ndarray,
    matches: Sequence[np.ndarray],
    weights: Sequence[np.ndarray],
    n_matches: int,
    support: Optional[np.ndarray] = None,
    outcome: Optional[np.ndarray] = None,
    neighbors: bool = True,
) -> pd.DataFrame:
    """Build the psmatch2-style per-observation frame.

    All array arguments are *positional* over the estimation sample (the
    complete-case rows), with ``index`` giving the corresponding labels from
    the source ``DataFrame``.

    Parameters
    ----------
    index : pd.Index
        Row labels of the estimation sample (length ``n``).
    treated : ndarray
        0/1 treatment indicator (length ``n``).
    pscore : ndarray
        Estimated propensity scores (length ``n``).
    idx_t, idx_c : ndarray
        Positions (into ``0..n-1``) of treated and control units.
    matches : sequence of ndarray
        ``matches[i]`` holds positions *into ``idx_c``* of the controls
        matched to the i-th treated unit (``idx_t[i]``), nearest first.
    weights : sequence of ndarray
        ``weights[i]`` holds the share each matched control receives; each
        array sums to 1 (or is empty when the treated unit found no match).
    n_matches : int
        Requested number of neighbours ``k`` — fixes how many ``_n{j}``
        columns are emitted.
    support : ndarray of bool, optional
        Common-support flag (length ``n``).  Defaults to all-on-support.
        This only fills the ``_support`` column; it does **not** gate the
        weights — the caller is responsible for having excluded off-support
        treated units from ``matches`` when it wants them trimmed.
    outcome : ndarray, optional
        Outcome values (length ``n``).  When supplied, the matched-control
        mean outcome is written to the ``_y`` column on treated rows
        (psmatch2's ``_y``).
    neighbors : bool, default True
        Emit the discrete-neighbour columns ``_n1`` … ``_nk`` / ``_nn`` /
        ``_pdif``.  Set ``False`` for kernel / radius matching, where every
        treated unit matches many controls with fractional weights and
        Stata's ``psmatch2`` does not create those columns.

    Returns
    -------
    pd.DataFrame
        Indexed like ``index`` with the columns documented at module level.
    """
    n = len(treated)
    treated = np.asarray(treated)
    pscore = np.asarray(pscore, dtype=float)
    if support is None:
        support = np.ones(n, dtype=bool)
    support = np.asarray(support, dtype=bool)
    outcome_arr: Optional[np.ndarray] = None
    if outcome is not None:
        outcome_arr = np.asarray(outcome, dtype=float)
    has_outcome = outcome_arr is not None

    obs_id = np.arange(1, n + 1, dtype=float)  # _id = 1..n (estimation order)

    weight = np.full(n, np.nan, dtype=float)
    # psmatch2 reports _nn = 0 on control rows (not missing).
    nn = np.zeros(n, dtype=float)
    pdif = np.full(n, np.nan, dtype=float)
    matched_y = np.full(n, np.nan, dtype=float)
    k = max(int(n_matches), 1)
    neighbor = np.full((n, k), np.nan, dtype=float)

    for i, (m, w) in enumerate(zip(matches, weights)):
        t_pos = int(idx_t[i])
        if len(m) == 0:
            # Treated unit found no match (caliper bound, or trimmed off
            # support by the caller): leave _weight missing so it drops out
            # of the matched sample.
            continue
        # Treated unit enters the matched sample with frequency weight 1.
        weight[t_pos] = 1.0

        ctrl_pos = idx_c[np.asarray(m, dtype=int)]
        w_arr = np.asarray(w, dtype=float)

        if neighbors:
            nn[t_pos] = float(len(m))
            # Neighbour ids (nearest first), padded/truncated to k columns.
            ids = obs_id[ctrl_pos]
            neighbor[t_pos, : min(len(ids), k)] = ids[:k]
            # _pdif: propensity-score gap to the *nearest* match (matches are
            # ordered nearest-first), matching Stata's definition.
            pdif[t_pos] = float(abs(pscore[t_pos] - pscore[ctrl_pos[0]]))

        if outcome_arr is not None:
            matched_y[t_pos] = float(np.average(outcome_arr[ctrl_pos], weights=w_arr))

        # Accumulate each matched control's frequency weight.
        for pos, share in zip(ctrl_pos, w_arr):
            pos = int(pos)
            weight[pos] = (0.0 if np.isnan(weight[pos]) else weight[pos]) + share

    data: Dict[str, np.ndarray] = {
        COL_ID: obs_id,
        COL_TREATED: treated.astype(float),
        COL_PSCORE: pscore,
        COL_SUPPORT: support.astype(float),
        COL_WEIGHT: weight,
    }
    if neighbors:
        for j in range(k):
            data[neighbor_col(j + 1)] = neighbor[:, j]
        data[COL_NN] = nn
        data[COL_PDIF] = pdif
    if has_outcome:
        data[COL_Y] = matched_y

    return pd.DataFrame(data, index=index)


def matched_columns(
    n_matches: int, *, with_outcome: bool = False, neighbors: bool = True
) -> List[str]:
    """Ordered list of the psmatch2 columns for ``k = n_matches`` neighbours."""
    k = max(int(n_matches), 1)
    cols = [COL_ID, COL_TREATED, COL_PSCORE, COL_SUPPORT, COL_WEIGHT]
    if neighbors:
        cols += [neighbor_col(j + 1) for j in range(k)]
        cols += [COL_NN, COL_PDIF]
    if with_outcome:
        cols.append(COL_Y)
    return cols


def attach_matched_frame(
    data: pd.DataFrame,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Return a copy of ``data`` with the psmatch2 columns merged in.

    Rows of ``data`` that were dropped from the estimation sample (missing
    covariates / outcome) receive ``NaN`` in every appended column, mirroring
    how Stata leaves ``psmatch2`` variables missing outside ``e(sample)``.
    """
    out = data.copy()
    aligned = frame.reindex(out.index)
    for col in frame.columns:
        out[col] = aligned[col].values
    return out
