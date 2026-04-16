"""
Forward Difference-in-Differences (FDID).

Implements the Forward DID estimator from Li (2024) and Greathouse (2024).
Instead of using *all* control units (standard DID) or finding optimal
*weights* (synthetic control), FDID uses **forward selection** to find the
optimal *subset* of control units that best approximates the treated unit
in the pre-treatment period.

Three selection strategies are provided:

* **forward** — greedy forward selection (default, fast)
* **forward_cv** — forward selection with rolling-window cross-validation
* **best_subset** — exhaustive 2^J search (feasible when J <= 15)

Inference is performed via **placebo permutation**: each control unit is
assigned treatment status in turn and the FDID procedure is re-run; the
resulting distribution of placebo effects produces p-values and standard
errors.

References
----------
Li, K.T. (2024). "Forward Difference-in-Differences."
Greathouse, J. (2024). "Forward Difference-in-Differences: fdid command
for Stata."
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult


# ======================================================================
# Internal helpers
# ======================================================================


def _pre_rmse(
    Y1_pre: np.ndarray,
    Y0_pre: np.ndarray,
    selected_idx: np.ndarray,
) -> float:
    """Compute pre-treatment RMSE of DID counterfactual.

    The counterfactual is the simple average of selected control units.
    RMSE measures how well the average of selected donors tracks the
    treated unit in pre-treatment periods.

    Parameters
    ----------
    Y1_pre : (T_pre,)
        Treated unit outcomes in the pre-treatment period.
    Y0_pre : (J, T_pre)
        Donor outcomes in the pre-treatment period.
    selected_idx : (K,)
        Integer indices into the first axis of *Y0_pre*.

    Returns
    -------
    float
        Root mean squared error.
    """
    if len(selected_idx) == 0:
        return np.inf
    synth_pre = Y0_pre[selected_idx].mean(axis=0)
    return float(np.sqrt(np.mean((Y1_pre - synth_pre) ** 2)))


def _did_estimate(
    Y1_pre: np.ndarray,
    Y1_post: np.ndarray,
    Y0_pre: np.ndarray,
    Y0_post: np.ndarray,
    selected_idx: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Compute the DID estimate using the selected control subset.

    ATT = (mean(Y1_post) - mean(Y1_pre))
        - (mean(Y0_sel_post) - mean(Y0_sel_pre))

    Also returns per-period treatment effects.

    Parameters
    ----------
    Y1_pre, Y1_post : (T_pre,), (T_post,)
    Y0_pre, Y0_post : (J, T_pre), (J, T_post)
    selected_idx : (K,)

    Returns
    -------
    att : float
        Average treatment effect on the treated.
    effects : (T_post,)
        Period-by-period treatment effects.
    """
    synth_pre = Y0_pre[selected_idx].mean(axis=0)
    synth_post = Y0_post[selected_idx].mean(axis=0)

    # DID gap = level shift
    bias = Y1_pre.mean() - synth_pre.mean()

    # Per-period effects: Y1_post_t - (synth_post_t + bias)
    effects = Y1_post - (synth_post + bias)
    att = float(effects.mean())
    return att, effects


# ======================================================================
# Selection algorithms
# ======================================================================


def _forward_select(
    Y1_pre: np.ndarray,
    Y0_pre: np.ndarray,
    donor_names: List[Any],
    max_donors: Optional[int] = None,
) -> Tuple[List[int], List[Tuple[Any, float]]]:
    """Greedy forward selection of donor units.

    At each step, add the donor that produces the largest reduction in
    pre-treatment RMSE.  Stop when RMSE no longer improves or
    *max_donors* is reached.

    Parameters
    ----------
    Y1_pre : (T_pre,)
    Y0_pre : (J, T_pre)
    donor_names : list of length J
    max_donors : int, optional
        Maximum number of donors to select.  Defaults to J.

    Returns
    -------
    selected : list of int
        Indices of selected donors (order = selection order).
    path : list of (donor_name, rmse)
        Selection path showing the donor added and RMSE at each step.
    """
    J = Y0_pre.shape[0]
    if max_donors is None:
        max_donors = J

    remaining = set(range(J))
    selected: List[int] = []
    path: List[Tuple[Any, float]] = []
    best_rmse = np.inf

    for _ in range(min(max_donors, J)):
        best_j: Optional[int] = None
        best_candidate_rmse = np.inf

        for j in remaining:
            candidate = np.array(selected + [j], dtype=int)
            rmse = _pre_rmse(Y1_pre, Y0_pre, candidate)
            if rmse < best_candidate_rmse:
                best_candidate_rmse = rmse
                best_j = j

        # Stop if no improvement
        if best_j is None or best_candidate_rmse >= best_rmse:
            break

        selected.append(best_j)
        remaining.discard(best_j)
        best_rmse = best_candidate_rmse
        path.append((donor_names[best_j], float(best_rmse)))

    return selected, path


def _forward_select_cv(
    Y1_pre: np.ndarray,
    Y0_pre: np.ndarray,
    donor_names: List[Any],
    max_donors: Optional[int] = None,
    min_train: int = 3,
) -> Tuple[List[int], List[Tuple[Any, float]]]:
    """Forward selection with rolling-window cross-validation.

    Uses an expanding-window scheme: train on periods [0..t], validate on
    period t+1, averaged over all feasible splits.  This guards against
    over-fitting the pre-treatment period.

    Parameters
    ----------
    Y1_pre : (T_pre,)
    Y0_pre : (J, T_pre)
    donor_names : list of length J
    max_donors : int, optional
    min_train : int, default 3
        Minimum training window length.

    Returns
    -------
    selected, path : same as :func:`_forward_select`.
    """
    T_pre = Y1_pre.shape[0]
    J = Y0_pre.shape[0]
    if max_donors is None:
        max_donors = J

    # If not enough pre-periods for CV, fall back to plain forward
    if T_pre <= min_train + 1:
        return _forward_select(Y1_pre, Y0_pre, donor_names, max_donors)

    remaining = set(range(J))
    selected: List[int] = []
    path: List[Tuple[Any, float]] = []
    best_cv_rmse = np.inf

    for _ in range(min(max_donors, J)):
        best_j: Optional[int] = None
        best_candidate_cv = np.inf

        for j in remaining:
            candidate = np.array(selected + [j], dtype=int)
            # Expanding-window CV
            cv_errors: List[float] = []
            for split in range(min_train, T_pre):
                train_synth = Y0_pre[np.ix_(candidate, np.arange(split))].mean(axis=0)
                train_y = Y1_pre[:split]
                bias = train_y.mean() - train_synth.mean()
                pred = Y0_pre[np.ix_(candidate, np.array([split]))].mean(axis=0) + bias
                actual = Y1_pre[split]
                cv_errors.append(float((actual - pred) ** 2))
            cv_rmse = float(np.sqrt(np.mean(cv_errors)))
            if cv_rmse < best_candidate_cv:
                best_candidate_cv = cv_rmse
                best_j = j

        if best_j is None or best_candidate_cv >= best_cv_rmse:
            break

        selected.append(best_j)
        remaining.discard(best_j)
        best_cv_rmse = best_candidate_cv
        path.append((donor_names[best_j], float(best_cv_rmse)))

    return selected, path


def _best_subset(
    Y1_pre: np.ndarray,
    Y0_pre: np.ndarray,
    donor_names: List[Any],
    max_donors: Optional[int] = None,
) -> Tuple[List[int], List[Tuple[Any, float]]]:
    """Exhaustive best-subset search over all 2^J donor combinations.

    Only feasible when J <= 15 (32 768 subsets).  For larger donor pools,
    raises ``ValueError``.

    Parameters
    ----------
    Y1_pre : (T_pre,)
    Y0_pre : (J, T_pre)
    donor_names : list of length J
    max_donors : int, optional

    Returns
    -------
    selected, path : same as :func:`_forward_select`.
        *path* contains only a single entry — the winning subset.
    """
    J = Y0_pre.shape[0]
    if J > 15:
        raise ValueError(
            f"best_subset requires J <= 15 donors, got J = {J}. "
            "Use method='forward' for larger donor pools."
        )
    if max_donors is None:
        max_donors = J

    best_rmse = np.inf
    best_combo: Tuple[int, ...] = ()

    for size in range(1, min(max_donors, J) + 1):
        for combo in combinations(range(J), size):
            idx = np.array(combo, dtype=int)
            rmse = _pre_rmse(Y1_pre, Y0_pre, idx)
            if rmse < best_rmse:
                best_rmse = rmse
                best_combo = combo

    selected = list(best_combo)
    names_str = ", ".join(str(donor_names[i]) for i in selected)
    path = [(names_str, float(best_rmse))]
    return selected, path


# ======================================================================
# Placebo inference
# ======================================================================


def _placebo_inference(
    panel: pd.DataFrame,
    pre_times: List[Any],
    post_times: List[Any],
    treated_unit: Any,
    donors: List[Any],
    method: str,
    max_donors: Optional[int],
    att: float,
) -> Tuple[float, float, pd.DataFrame]:
    """Run placebo permutation to obtain SE and p-value.

    Each control unit is treated as a pseudo-treated unit.  The full
    FDID procedure (selection + DID) is re-run, producing a distribution
    of placebo ATTs.

    Parameters
    ----------
    panel : pd.DataFrame
        Pivoted panel (units x times).
    pre_times, post_times : lists
    treated_unit : scalar
    donors : list
    method : str
    max_donors : int or None
    att : float
        The actual (non-placebo) ATT.

    Returns
    -------
    se : float
        Permutation-based standard error.
    pvalue : float
        Two-sided permutation p-value.
    placebo_df : pd.DataFrame
        Columns: unit, att, pre_rmspe, post_rmspe.
    """
    selector = {
        "forward": _forward_select,
        "forward_cv": _forward_select_cv,
        "best_subset": _best_subset,
    }[method]

    records: List[Dict[str, Any]] = []

    for placebo_unit in donors:
        p_donors = [u for u in panel.index if u != placebo_unit]
        Y1p_pre = panel.loc[placebo_unit, pre_times].values.astype(np.float64)
        Y1p_post = panel.loc[placebo_unit, post_times].values.astype(np.float64)
        Y0p_pre = panel.loc[p_donors, pre_times].values.astype(np.float64)
        Y0p_post = panel.loc[p_donors, post_times].values.astype(np.float64)

        try:
            sel_idx, _ = selector(Y1p_pre, Y0p_pre, p_donors, max_donors)
        except (ValueError, IndexError):
            continue

        if len(sel_idx) == 0:
            continue

        p_att, _ = _did_estimate(Y1p_pre, Y1p_post, Y0p_pre, Y0p_post,
                                 np.array(sel_idx, dtype=int))
        pre_rmspe = _pre_rmse(Y1p_pre, Y0p_pre, np.array(sel_idx, dtype=int))
        # Post RMSPE (using counterfactual)
        synth_post = Y0p_post[sel_idx].mean(axis=0)
        bias = Y1p_pre.mean() - Y0p_pre[sel_idx].mean(axis=0).mean()
        post_rmspe = float(np.sqrt(np.mean((Y1p_post - (synth_post + bias)) ** 2)))

        records.append({
            "unit": placebo_unit,
            "att": p_att,
            "pre_rmspe": pre_rmspe,
            "post_rmspe": post_rmspe,
        })

    placebo_df = pd.DataFrame(records)

    if len(placebo_df) < 2:
        return np.nan, np.nan, placebo_df

    placebo_atts = placebo_df["att"].values
    se = float(np.std(placebo_atts, ddof=1))

    # Two-sided p-value: fraction of |placebo ATT| >= |actual ATT|
    n_extreme = np.sum(np.abs(placebo_atts) >= np.abs(att))
    pvalue = float((n_extreme + 1) / (len(placebo_atts) + 1))

    return se, pvalue, placebo_df


# ======================================================================
# Main public function
# ======================================================================


def fdid(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    treatment_time: Any,
    method: str = "forward",
    max_donors: Optional[int] = None,
    placebo: bool = True,
    alpha: float = 0.05,
) -> CausalResult:
    """Forward Difference-in-Differences estimator.

    Selects the *optimal subset* of control (donor) units via forward
    selection, then applies a standard DID estimator using only the
    selected donors.  This combines the flexibility of synthetic control
    methods with the simplicity of DID.

    Parameters
    ----------
    data : pd.DataFrame
        Balanced panel data in long format.
    outcome : str
        Column name of the outcome variable.
    unit : str
        Column name of the unit identifier.
    time : str
        Column name of the time period variable.
    treated_unit : scalar
        Identifier of the treated unit.
    treatment_time : scalar
        First period of treatment (inclusive).
    method : {'forward', 'forward_cv', 'best_subset'}, default 'forward'
        Donor selection strategy:

        * ``'forward'`` — greedy forward selection minimising pre-RMSE.
        * ``'forward_cv'`` — forward selection with expanding-window CV.
        * ``'best_subset'`` — exhaustive search (J <= 15 only).
    max_donors : int, optional
        Maximum number of donors to select.  By default, the algorithm
        selects as many as improve pre-period fit.
    placebo : bool, default True
        Whether to run placebo permutation for inference.
    alpha : float, default 0.05
        Significance level for confidence interval.

    Returns
    -------
    CausalResult
        Unified result object.  Key ``model_info`` entries:

        - ``selected_donors`` : list of donor unit names
        - ``selection_path`` : list of (donor_added, rmse) tuples
        - ``n_selected`` : int
        - ``pre_rmspe`` : float
        - ``post_rmspe`` : float
        - ``effects_by_period`` : pd.DataFrame
        - ``weights`` : dict mapping donor names to 1/K
        - ``method`` : selection strategy used

    Raises
    ------
    ValueError
        If ``method='best_subset'`` and there are more than 15 donors,
        or if the treated unit is not found in the data.

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.synth.fdid(
    ...     data=panel_df,
    ...     outcome='gdp',
    ...     unit='country',
    ...     time='year',
    ...     treated_unit='West Germany',
    ...     treatment_time=1990,
    ... )
    >>> result.summary()

    References
    ----------
    Li, K.T. (2024). "Forward Difference-in-Differences."
    Greathouse, J. (2024). "Forward Difference-in-Differences: fdid
    command for Stata."
    """
    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if method not in ("forward", "forward_cv", "best_subset"):
        raise ValueError(
            f"method must be 'forward', 'forward_cv', or 'best_subset', "
            f"got '{method}'."
        )

    for col in (outcome, unit, time):
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data.")

    if treated_unit not in data[unit].values:
        raise ValueError(
            f"treated_unit '{treated_unit}' not found in column '{unit}'."
        )

    if treatment_time not in data[time].values:
        raise ValueError(
            f"treatment_time '{treatment_time}' not found in column '{time}'."
        )

    # ------------------------------------------------------------------
    # Reshape to panel (units x times)
    # ------------------------------------------------------------------
    panel = data.pivot_table(index=unit, columns=time, values=outcome)
    all_times = sorted(panel.columns)
    pre_times = [t for t in all_times if t < treatment_time]
    post_times = [t for t in all_times if t >= treatment_time]

    if len(pre_times) < 2:
        raise ValueError(
            f"Need at least 2 pre-treatment periods, got {len(pre_times)}."
        )
    if len(post_times) == 0:
        raise ValueError("No post-treatment periods found.")

    Y1_pre = panel.loc[treated_unit, pre_times].values.astype(np.float64)
    Y1_post = panel.loc[treated_unit, post_times].values.astype(np.float64)
    donors = [u for u in panel.index if u != treated_unit]
    donor_names = list(donors)

    if len(donors) < 1:
        raise ValueError("Need at least 1 donor unit.")

    Y0_pre = panel.loc[donors, pre_times].values.astype(np.float64)
    Y0_post = panel.loc[donors, post_times].values.astype(np.float64)

    # ------------------------------------------------------------------
    # Donor selection
    # ------------------------------------------------------------------
    selector = {
        "forward": _forward_select,
        "forward_cv": _forward_select_cv,
        "best_subset": _best_subset,
    }[method]

    selected_idx, selection_path = selector(
        Y1_pre, Y0_pre, donor_names, max_donors,
    )

    if len(selected_idx) == 0:
        raise ValueError(
            "Forward selection could not find any improving donor. "
            "Check that the data has sufficient variation."
        )

    sel_arr = np.array(selected_idx, dtype=int)

    # ------------------------------------------------------------------
    # Point estimate
    # ------------------------------------------------------------------
    att, effects = _did_estimate(Y1_pre, Y1_post, Y0_pre, Y0_post, sel_arr)

    pre_rmspe = _pre_rmse(Y1_pre, Y0_pre, sel_arr)

    # Post-treatment RMSPE (how far outcomes deviate from counterfactual)
    synth_post = Y0_post[sel_arr].mean(axis=0)
    bias = Y1_pre.mean() - Y0_pre[sel_arr].mean(axis=0).mean()
    post_rmspe = float(np.sqrt(np.mean((Y1_post - (synth_post + bias)) ** 2)))

    # ------------------------------------------------------------------
    # Period-by-period effects
    # ------------------------------------------------------------------
    effects_df = pd.DataFrame({
        "time": post_times,
        "treated": Y1_post,
        "counterfactual": synth_post + bias,
        "effect": effects,
    })

    # ------------------------------------------------------------------
    # Inference (placebo permutation)
    # ------------------------------------------------------------------
    se: float = np.nan
    pvalue: float = np.nan
    ci: Tuple[float, float] = (np.nan, np.nan)
    placebo_df: Optional[pd.DataFrame] = None

    if placebo and len(donors) >= 2:
        se, pvalue, placebo_df = _placebo_inference(
            panel, pre_times, post_times,
            treated_unit, donors, method, max_donors, att,
        )
        if not np.isnan(se) and se > 0:
            z = stats.norm.ppf(1 - alpha / 2)
            ci = (att - z * se, att + z * se)
    elif not placebo:
        # Without placebo: use Neyman-style SE from donor variation
        if len(selected_idx) > 1:
            donor_atts = []
            for j in selected_idx:
                j_pre = Y0_pre[j].mean()
                j_post = Y0_post[j].mean()
                donor_atts.append((Y1_post.mean() - Y1_pre.mean()) - (j_post - j_pre))
            se = float(np.std(donor_atts, ddof=1) / np.sqrt(len(donor_atts)))
            z = stats.norm.ppf(1 - alpha / 2)
            ci = (att - z * se, att + z * se)
            pvalue = float(2 * (1 - stats.norm.cdf(abs(att / se)))) if se > 0 else np.nan

    # ------------------------------------------------------------------
    # Selected donor info
    # ------------------------------------------------------------------
    selected_donor_names = [donor_names[i] for i in selected_idx]
    n_selected = len(selected_donor_names)
    equal_weight = 1.0 / n_selected
    weights = {name: equal_weight for name in selected_donor_names}

    # ------------------------------------------------------------------
    # Build model_info
    # ------------------------------------------------------------------
    model_info: Dict[str, Any] = {
        "selected_donors": selected_donor_names,
        "selection_path": selection_path,
        "n_selected": n_selected,
        "n_donors_total": len(donors),
        "pre_rmspe": pre_rmspe,
        "post_rmspe": post_rmspe,
        "rmspe_ratio": post_rmspe / pre_rmspe if pre_rmspe > 0 else np.inf,
        "effects_by_period": effects_df,
        "weights": weights,
        "method": method,
        "treated_unit": treated_unit,
        "treatment_time": treatment_time,
        "n_pre": len(pre_times),
        "n_post": len(post_times),
    }
    if placebo_df is not None:
        model_info["placebo_results"] = placebo_df

    # ------------------------------------------------------------------
    # Return CausalResult
    # ------------------------------------------------------------------
    return CausalResult(
        method="Forward Difference-in-Differences (FDID)",
        estimand="ATT",
        estimate=float(att),
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=len(data),
        detail=effects_df,
        model_info=model_info,
    )
