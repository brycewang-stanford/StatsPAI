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
#: Cell label for stratification / CEM frames (no psmatch2 analogue; Stata's
#: ``pscore``/``cem`` packages call it ``block`` and ``cem_strata``).
COL_STRATUM = "_stratum"


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


def _within_group_self_outcome_reference(
    outcome: np.ndarray,
    treated: np.ndarray,
    pscore: np.ndarray,
    n_ai_matches: int,
) -> np.ndarray:
    """Literal O(n^2 log n) definition of ``_self_y``, kept as a test oracle.

    :func:`_within_group_self_outcome` is the fast implementation actually
    used.  This one states the rule directly -- for every unit, stably sort
    the same-arm units by propensity distance and average the first ``J``
    outcomes -- so the equivalence test has something to compare against
    that is obviously correct rather than merely fast.
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


def _within_group_self_outcome(
    outcome: np.ndarray,
    treated: np.ndarray,
    pscore: np.ndarray,
    n_ai_matches: int,
) -> np.ndarray:
    """psmatch2's ``_self_y``: mean outcome of the J nearest *same-group* units.

    For every unit, the ``n_ai_matches`` nearest neighbours **within its own
    treatment arm** (by propensity-score distance) are found and their
    outcomes averaged.  This is the conditional-mean estimate used by the
    Abadie-Imbens (2006) variance to gauge the within-cell outcome noise
    :math:`\\sigma^2(X)`.

    The selection rule is ``(distance ascending, original index ascending)``
    -- a stable sort of the same-arm units by ``|p_i - p_k|``, ties broken by
    position in the arm, which is ascending original index.

    Implementation.  Sorting each arm by propensity score once turns the
    neighbour search into an outward walk from each unit's own rank, because
    in one dimension sorted order *is* distance order.  The walk proceeds
    over **groups of equal propensity score**, not individual units: within
    one distance the rule takes units in ascending original index, and the
    units tied at a given distance can sit on *both* sides of the origin, so
    a left/right frontier comparison is not enough.  (Walking outward visits
    the left-hand ties in *descending* index order, which is what an earlier
    version of this function got wrong.)  Only the *set* of ``J`` neighbours
    matters, since their outcomes are averaged.

    This is roughly ``O(m log m + m·J)`` per arm rather than the
    ``O(m² log m)`` of the literal definition in
    :func:`_within_group_self_outcome_reference`, against which it is
    verified bit-identical (``tests/test_matching_self_outcome_equivalence.py``).
    That matters because it runs on every default ``sp.match`` call: at
    n = 16,000 the literal version took 12.5 s.
    """
    y = np.asarray(outcome, dtype=float)
    t = np.asarray(treated)
    ps = np.asarray(pscore, dtype=float)
    n = len(y)
    self_y = np.full(n, np.nan)
    j = max(int(n_ai_matches), 1)

    for g in (0, 1):
        idx = np.where(t == g)[0]
        m = len(idx)
        if m < 2:
            continue
        # Stable sort keeps equal propensity scores in ascending original
        # index order -- the tie-break the rule needs, for free.
        order = np.argsort(ps[idx], kind="stable")
        s_idx = idx[order]
        s_ps = ps[s_idx]
        s_y = y[s_idx]

        # Collapse to groups of identical propensity score.
        is_new = np.empty(m, dtype=bool)
        is_new[0] = True
        np.not_equal(s_ps[1:], s_ps[:-1], out=is_new[1:])
        starts = np.flatnonzero(is_new)
        ends = np.append(starts[1:], m)
        vals = s_ps[starts]
        grp_of = np.cumsum(is_new) - 1
        n_grp = len(starts)
        take = min(j, m - 1)

        for r in range(m):
            gi = int(grp_of[r])
            need = take
            # Collected in (distance ascending, original index ascending)
            # order -- the reference's order -- so that the final np.mean
            # sums in the same sequence and agrees bit for bit.
            picked: List[np.ndarray] = []

            # Distance 0: the unit's own group, minus itself, already in
            # ascending original index order.
            lo_g, hi_g = int(starts[gi]), int(ends[gi])
            if hi_g - lo_g > 1:
                own = np.concatenate([s_y[lo_g:r], s_y[r + 1 : hi_g]])
                use = min(need, own.size)
                if use:
                    picked.append(own[:use])
                    need -= use

            left, right = gi - 1, gi + 1
            while need > 0:
                d_left = (vals[gi] - vals[left]) if left >= 0 else np.inf
                d_right = (vals[right] - vals[gi]) if right < n_grp else np.inf
                if d_left < d_right:
                    blocks = [(starts[left], ends[left])]
                    left -= 1
                elif d_right < d_left:
                    blocks = [(starts[right], ends[right])]
                    right += 1
                else:
                    # Tied distance on both sides: the two blocks are ranked
                    # together by original index, not by side.
                    blocks = [
                        (starts[left], ends[left]),
                        (starts[right], ends[right]),
                    ]
                    left -= 1
                    right += 1
                if len(blocks) == 1:
                    a0, b0 = blocks[0]
                    chunk = slice(int(a0), int(b0))
                    cand_y = s_y[chunk]
                else:
                    (a0, b0), (a1, b1) = blocks
                    cand_i = np.concatenate(
                        [s_idx[int(a0) : int(b0)], s_idx[int(a1) : int(b1)]]
                    )
                    cand_y = np.concatenate(
                        [s_y[int(a0) : int(b0)], s_y[int(a1) : int(b1)]]
                    )
                    cand_y = cand_y[np.argsort(cand_i, kind="stable")]
                use = min(need, cand_y.size)
                picked.append(cand_y[:use])
                need -= use
            chosen = picked[0] if len(picked) == 1 else np.concatenate(picked)
            self_y[s_idx[r]] = float(np.mean(chosen))
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
    n_matches: int,
    *,
    with_outcome: bool = False,
    neighbors: bool = True,
    stratum: bool = False,
) -> List[str]:
    """Ordered list of the psmatch2 columns for ``k = n_matches`` neighbours.

    ``stratum=True`` describes a :func:`build_stratum_matched_frame` result,
    which carries ``_stratum`` and ``_nn`` but none of the ordered
    nearest-neighbour columns.
    """
    k = max(int(n_matches), 1)
    cols = [COL_ID, COL_TREATED, COL_PSCORE, COL_SUPPORT, COL_WEIGHT]
    if stratum:
        cols += [COL_STRATUM, COL_NN]
        if with_outcome:
            cols.append(COL_Y)
        return cols
    if neighbors:
        cols += [neighbor_col(j + 1) for j in range(k)]
        cols += [COL_NN, COL_PDIF]
    if with_outcome:
        cols.append(COL_Y)
    return cols


def build_ate_matched_frame(
    *,
    index: pd.Index,
    treated: np.ndarray,
    pscore: np.ndarray,
    idx_t: np.ndarray,
    idx_c: np.ndarray,
    matches_tc: Sequence[np.ndarray],
    weights_tc: Sequence[np.ndarray],
    matches_ct: Sequence[np.ndarray],
    weights_ct: Sequence[np.ndarray],
    n_matches: int,
    support: Optional[np.ndarray] = None,
    outcome: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Matched frame for ``estimand='ATE'`` (both arms matched).

    The ATT frame's ``_weight`` is a *frequency*: how many times a control
    stands in for a treated unit.  The ATE estimator has no such asymmetry —
    every unit both contributes its own outcome and serves as a counterfactual
    for the other arm — so the natural per-unit weight is the Abadie-Imbens
    (2006) matching weight

    .. math:: w_i = 1 + K_M(i)

    where :math:`K_M(i)` is the total share unit *i* receives across the
    opposite-arm units matched to it.  The estimator is then the signed sum

    .. math:: \\widehat{ATE} = \\frac{1}{N} \\sum_i (2 D_i - 1)\\, w_i\\, Y_i

    which this function's output reproduces exactly — the same "pure
    bookkeeping" contract the ATT frame satisfies.

    Because the sign matters, ``_weight`` here is **not** a frequency weight
    and must not be fed to ``[fweight=]``.  Callers record the distinction in
    ``model_info['matched_frame_weight_kind']``.

    ``_n{j}`` / ``_nn`` / ``_pdif`` are filled for *both* arms: for a treated
    row they point at the matched controls, for a control row at the matched
    treated units.  ``_y`` is the matched counterfactual mean outcome, again
    for both arms.

    Parameters
    ----------
    matches_tc, weights_tc : sequence of ndarray
        Treated-to-control assignment: ``matches_tc[i]`` holds positions into
        ``idx_c`` for the i-th treated unit.
    matches_ct, weights_ct : sequence of ndarray
        Control-to-treated assignment: ``matches_ct[j]`` holds positions into
        ``idx_t`` for the j-th control unit.

    Other parameters are as in :func:`build_matched_frame`.

    References
    ----------
    Abadie, A. and Imbens, G.W. (2006). Large Sample Properties of Matching
        Estimators for Average Treatment Effects. *Econometrica*, 74(1),
        235-267.
    """
    n = len(treated)
    treated = np.asarray(treated)
    pscore = np.asarray(pscore, dtype=float)
    if support is None:
        support = np.ones(n, dtype=bool)
    support = np.asarray(support, dtype=bool)
    outcome_arr = None if outcome is None else np.asarray(outcome, dtype=float)

    obs_id = np.arange(1, n + 1, dtype=float)
    k = max(int(n_matches), 1)

    # K_M(i): the share unit i receives as somebody else's match.
    k_share = np.zeros(n, dtype=float)
    nn = np.zeros(n, dtype=float)
    pdif = np.full(n, np.nan, dtype=float)
    matched_y = np.full(n, np.nan, dtype=float)
    neighbor = np.full((n, k), np.nan, dtype=float)

    for target_idx, pool_idx, matches, weights in (
        (idx_t, idx_c, matches_tc, weights_tc),
        (idx_c, idx_t, matches_ct, weights_ct),
    ):
        for i, (m, w) in enumerate(zip(matches, weights)):
            pos = int(target_idx[i])
            if len(m) == 0:
                continue
            partner_pos = pool_idx[np.asarray(m, dtype=int)]
            w_arr = np.asarray(w, dtype=float)

            nn[pos] = float(len(m))
            ids = obs_id[partner_pos]
            neighbor[pos, : min(len(ids), k)] = ids[:k]
            pdif[pos] = float(abs(pscore[pos] - pscore[partner_pos[0]]))
            if outcome_arr is not None:
                matched_y[pos] = float(
                    np.average(outcome_arr[partner_pos], weights=w_arr)
                )
            for p, share in zip(partner_pos, w_arr):
                k_share[int(p)] += share

    weight = 1.0 + k_share
    # A unit that never matched and was never matched to contributes nothing.
    unmatched = (nn == 0) & (k_share == 0)
    weight[unmatched] = np.nan

    data: Dict[str, np.ndarray] = {
        COL_ID: obs_id,
        COL_TREATED: treated.astype(float),
        COL_PSCORE: pscore,
        COL_SUPPORT: support.astype(float),
        COL_WEIGHT: weight,
    }
    for j in range(k):
        data[neighbor_col(j + 1)] = neighbor[:, j]
    data[COL_NN] = nn
    data[COL_PDIF] = pdif
    if outcome_arr is not None:
        data[COL_Y] = matched_y

    return pd.DataFrame(data, index=index)


def build_stratum_matched_frame(
    *,
    index: pd.Index,
    treated: np.ndarray,
    pscore: np.ndarray,
    stratum: np.ndarray,
    keep: np.ndarray,
    outcome: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Matched frame for stratification / coarsened exact matching.

    Both estimators partition the sample into cells and compare arm means
    within each retained cell.  With :math:`n^s_1` treated and :math:`n^s_0`
    controls in cell *s*, the ATT is

    .. math::

        \\widehat{ATT} = \\sum_s \\frac{n^s_1}{N_1}
            \\left( \\bar{Y}^s_1 - \\bar{Y}^s_0 \\right)

    which is exactly the psmatch2 weighting scheme written cell-wise: give
    every retained treated unit ``_weight = 1`` and every retained control
    ``_weight = n^s_1 / n^s_0``.  The control weights then sum to the number
    of matched treated units, just as they do after nearest-neighbour
    matching, and the weighted difference of arm means reproduces the
    estimator's own point estimate.

    Units in a cell that lacks one of the two arms fall out of the matched
    sample: ``_weight`` is missing and ``_support`` is 0.

    Parameters
    ----------
    stratum : ndarray
        Cell label per unit (any hashable dtype).
    keep : ndarray of bool
        Whether the unit's cell contains both arms, i.e. whether the unit
        entered the estimator's comparison.

    Returns
    -------
    pd.DataFrame
        ``_id``, ``_treated``, ``_pscore``, ``_support``, ``_weight``,
        ``_stratum``, ``_nn`` and (when an outcome is supplied) ``_y``.  The
        discrete-neighbour columns ``_n{j}`` / ``_pdif`` are omitted: a cell
        comparison has no ordered nearest neighbour to name, exactly as
        kernel matching has none.
    """
    n = len(treated)
    treated = np.asarray(treated)
    pscore = np.asarray(pscore, dtype=float)
    keep = np.asarray(keep, dtype=bool)
    outcome_arr = None if outcome is None else np.asarray(outcome, dtype=float)

    obs_id = np.arange(1, n + 1, dtype=float)
    weight = np.full(n, np.nan, dtype=float)
    nn = np.zeros(n, dtype=float)
    matched_y = np.full(n, np.nan, dtype=float)

    for s in pd.unique(stratum[keep]) if keep.any() else []:
        in_s = keep & (stratum == s)
        t_in = in_s & (treated == 1)
        c_in = in_s & (treated == 0)
        n_t = int(t_in.sum())
        n_c = int(c_in.sum())
        if n_t == 0 or n_c == 0:  # pragma: no cover - keep already excludes
            continue
        weight[t_in] = 1.0
        weight[c_in] = n_t / n_c
        # _nn: how many opposite-arm units the row is compared against.
        nn[t_in] = float(n_c)
        nn[c_in] = float(n_t)
        if outcome_arr is not None:
            matched_y[t_in] = float(np.mean(outcome_arr[c_in]))
            matched_y[c_in] = float(np.mean(outcome_arr[t_in]))

    data: Dict[str, np.ndarray] = {
        COL_ID: obs_id,
        COL_TREATED: treated.astype(float),
        COL_PSCORE: pscore,
        COL_SUPPORT: keep.astype(float),
        COL_WEIGHT: weight,
        COL_STRATUM: np.asarray(stratum, dtype=object),
        COL_NN: nn,
    }
    if outcome_arr is not None:
        data[COL_Y] = matched_y

    return pd.DataFrame(data, index=index)


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
