"""
Multiple Outcomes Synthetic Control Method.

When multiple outcome variables are available, find a single set of
donor weights that simultaneously matches ALL outcomes in the
pre-treatment period.  Under a low-rank factor structure, using K
outcomes reduces bias by a factor of 1/sqrt(K) compared to a single
outcome.

Two approaches
--------------
- **concatenated**: Stack standardised pre-treatment outcomes vertically
  and solve a single quadratic programme for shared weights.
- **averaged**: Standardise each outcome, average across K outcomes,
  then run standard SCM on the averaged series.

References
----------
Sun, L. (2023).
"Using Multiple Outcomes to Improve the Synthetic Control Method."
*Review of Economics and Statistics*, forthcoming.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.optimize import minimize

from ..core.results import CausalResult


# ====================================================================== #
#  Public API
# ====================================================================== #

def multi_outcome_synth(
    data: pd.DataFrame,
    outcomes: List[str],
    unit: str,
    time: str,
    treated_unit: Any,
    treatment_time: Any,
    method: str = "concatenated",
    standardize: bool = True,
    penalization: float = 0.0,
    placebo: bool = True,
    alpha: float = 0.05,
) -> CausalResult:
    """
    Multiple Outcomes Synthetic Control Method (Sun 2023).

    Finds a *single* set of donor weights that simultaneously matches
    the treated unit across all K outcomes in the pre-treatment period.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel data containing all outcome columns.
    outcomes : list of str
        Column names for the K outcome variables.
    unit : str
        Unit identifier column.
    time : str
        Time period column.
    treated_unit : any
        Value identifying the treated unit.
    treatment_time : any
        First treatment period (inclusive).
    method : {'concatenated', 'averaged'}, default 'concatenated'
        Weight-estimation strategy.

        * ``'concatenated'`` -- stack all K standardised outcome panels
          vertically and solve one quadratic programme.
        * ``'averaged'`` -- standardise each outcome, average across K,
          then solve SCM on the mean series.
    standardize : bool, default True
        Standardise each outcome to zero mean / unit variance before
        stacking or averaging (strongly recommended when outcome
        scales differ).
    penalization : float, default 0.0
        Ridge-type penalty added to the diagonal of the donor
        cross-product matrix (``penalization * I``).  Helps when donors
        are collinear.
    placebo : bool, default True
        Run in-space placebo permutations for inference (each donor is
        pretended to be treated in turn).
    alpha : float, default 0.05
        Significance level for confidence intervals and joint test.

    Returns
    -------
    CausalResult
        Unified result object with:

        - ``estimate`` : average treatment effect across outcomes
          (mean of per-outcome ATTs).
        - ``model_info['per_outcome_effects']`` : DataFrame with
          columns ``outcome``, ``att``, ``se``, ``pvalue``.
        - ``model_info['weights']`` : dict mapping donor names to
          shared SCM weights.
        - ``model_info['gap_tables']`` : dict of DataFrames (one per
          outcome) with time-level gaps.
        - ``model_info['joint_pvalue']`` : joint p-value across all K
          outcomes (Fisher combination of placebo p-values).
        - ``model_info['Y_synth']`` : dict mapping outcome name to
          full synthetic series.
        - ``model_info['Y_treated']`` : dict mapping outcome name to
          observed treated series.
        - ``model_info['times']`` : sorted list of all time periods.

    Examples
    --------
    >>> result = sp.multi_outcome_synth(
    ...     df,
    ...     outcomes=['gdp', 'employment', 'wages'],
    ...     unit='state', time='year',
    ...     treated_unit='California', treatment_time=1989,
    ... )
    >>> print(result.summary())
    >>> result.model_info['per_outcome_effects']

    Notes
    -----
    Sun (2023) shows that under a low-rank factor model the bias of
    the concatenated estimator shrinks as O(1/sqrt(K)), where K is
    the number of outcomes.  The key requirement is that the outcomes
    share a *common* latent-factor structure.
    """
    # ------------------------------------------------------------------
    #  Validate inputs
    # ------------------------------------------------------------------
    if not outcomes or len(outcomes) < 2:
        raise ValueError("Need at least 2 outcome columns for "
                         "multi-outcome SCM.")
    method = method.lower()
    if method not in ("concatenated", "averaged"):
        raise ValueError(
            f"method must be 'concatenated' or 'averaged', got '{method}'"
        )
    for col in [unit, time] + outcomes:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data.")

    K = len(outcomes)

    # ------------------------------------------------------------------
    #  Build per-outcome wide panels
    # ------------------------------------------------------------------
    panels: Dict[str, pd.DataFrame] = {}
    for oc in outcomes:
        panels[oc] = data.pivot_table(index=unit, columns=time, values=oc)

    all_times = sorted(panels[outcomes[0]].columns.tolist())
    pre_times = [t for t in all_times if t < treatment_time]
    post_times = [t for t in all_times if t >= treatment_time]

    if len(pre_times) < 2:
        raise ValueError("Need at least 2 pre-treatment periods.")
    if len(post_times) < 1:
        raise ValueError("Need at least 1 post-treatment period.")

    donors = [u for u in panels[outcomes[0]].index if u != treated_unit]
    J = len(donors)
    T0 = len(pre_times)
    T1 = len(post_times)

    if J < 2:
        raise ValueError("Need at least 2 donor units.")

    # Extract matrices per outcome: Y1_pre(T0,), Y0_pre(J,T0), etc.
    Y1_pres: Dict[str, np.ndarray] = {}
    Y1_posts: Dict[str, np.ndarray] = {}
    Y0_pres: Dict[str, np.ndarray] = {}
    Y0_posts: Dict[str, np.ndarray] = {}

    # Standardisation parameters (per outcome)
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}

    for oc in outcomes:
        p = panels[oc]
        y1_pre = p.loc[treated_unit, pre_times].values.astype(np.float64)
        y1_post = p.loc[treated_unit, post_times].values.astype(np.float64)
        y0_pre = p.loc[donors, pre_times].values.astype(np.float64)
        y0_post = p.loc[donors, post_times].values.astype(np.float64)

        if standardize:
            # Standardise using pre-treatment control data
            mu = np.mean(y0_pre)
            sigma = np.std(y0_pre)
            sigma = sigma if sigma > 1e-12 else 1.0
            means[oc] = mu
            stds[oc] = sigma
            y1_pre = (y1_pre - mu) / sigma
            y1_post = (y1_post - mu) / sigma
            y0_pre = (y0_pre - mu) / sigma
            y0_post = (y0_post - mu) / sigma

        Y1_pres[oc] = y1_pre
        Y1_posts[oc] = y1_post
        Y0_pres[oc] = y0_pre
        Y0_posts[oc] = y0_post

    # ------------------------------------------------------------------
    #  Compute shared weights
    # ------------------------------------------------------------------
    if method == "concatenated":
        omega = _concatenated_weights(
            Y1_pres, Y0_pres, outcomes, penalization,
        )
    else:
        omega = _averaged_weights(
            Y1_pres, Y0_pres, outcomes, penalization,
        )

    # ------------------------------------------------------------------
    #  Counterfactuals & treatment effects per outcome
    # ------------------------------------------------------------------
    per_outcome_att: Dict[str, float] = {}
    gap_tables: Dict[str, pd.DataFrame] = {}
    Y_synth_dict: Dict[str, np.ndarray] = {}
    Y_treated_dict: Dict[str, np.ndarray] = {}

    for oc in outcomes:
        # Need raw (un-standardised) panels for final effects
        p = panels[oc]
        y1_pre_raw = p.loc[treated_unit, pre_times].values.astype(np.float64)
        y1_post_raw = p.loc[treated_unit, post_times].values.astype(np.float64)
        y0_pre_raw = p.loc[donors, pre_times].values.astype(np.float64)
        y0_post_raw = p.loc[donors, post_times].values.astype(np.float64)

        synth_pre = y0_pre_raw.T @ omega   # (T0,)
        synth_post = y0_post_raw.T @ omega  # (T1,)

        gap_pre = y1_pre_raw - synth_pre
        gap_post = y1_post_raw - synth_post
        att_k = float(np.mean(gap_post))
        per_outcome_att[oc] = att_k

        gap_df = pd.DataFrame({
            "time": all_times,
            "treated": np.concatenate([y1_pre_raw, y1_post_raw]),
            "synthetic": np.concatenate([synth_pre, synth_post]),
            "gap": np.concatenate([gap_pre, gap_post]),
        })
        gap_tables[oc] = gap_df
        Y_synth_dict[oc] = np.concatenate([synth_pre, synth_post])
        Y_treated_dict[oc] = np.concatenate([y1_pre_raw, y1_post_raw])

    # Average ATT across outcomes
    overall_att = float(np.mean(list(per_outcome_att.values())))

    # ------------------------------------------------------------------
    #  Placebo inference
    # ------------------------------------------------------------------
    # For each donor j, pretend it is treated, compute per-outcome ATTs
    placebo_atts_per_outcome: Dict[str, List[float]] = {oc: [] for oc in outcomes}
    placebo_atts_overall: List[float] = []

    if placebo and J >= 2:
        for j_idx in range(J):
            donor_j = donors[j_idx]
            other_donors = [donors[i] for i in range(J) if i != j_idx]

            # Build placebo panels (standardised)
            p_Y1_pres: Dict[str, np.ndarray] = {}
            p_Y0_pres: Dict[str, np.ndarray] = {}
            p_Y0_posts_raw: Dict[str, np.ndarray] = {}
            p_Y1_posts_raw: Dict[str, np.ndarray] = {}

            try:
                for oc in outcomes:
                    p = panels[oc]
                    y1p = p.loc[donor_j, pre_times].values.astype(np.float64)
                    y1pp = p.loc[donor_j, post_times].values.astype(np.float64)
                    y0p = p.loc[other_donors, pre_times].values.astype(np.float64)
                    y0pp = p.loc[other_donors, post_times].values.astype(np.float64)

                    if standardize:
                        mu = np.mean(y0p)
                        sigma = np.std(y0p)
                        sigma = sigma if sigma > 1e-12 else 1.0
                        p_Y1_pres[oc] = (y1p - mu) / sigma
                        p_Y0_pres[oc] = (y0p - mu) / sigma
                    else:
                        p_Y1_pres[oc] = y1p
                        p_Y0_pres[oc] = y0p

                    p_Y0_posts_raw[oc] = y0pp
                    p_Y1_posts_raw[oc] = y1pp

                # Weights for this placebo
                if method == "concatenated":
                    w_plac = _concatenated_weights(
                        p_Y1_pres, p_Y0_pres, outcomes, penalization,
                    )
                else:
                    w_plac = _averaged_weights(
                        p_Y1_pres, p_Y0_pres, outcomes, penalization,
                    )

                # Per-outcome placebo ATTs
                plac_atts_k: List[float] = []
                for oc in outcomes:
                    synth_post_plac = p_Y0_posts_raw[oc].T @ w_plac
                    att_plac_k = float(
                        np.mean(p_Y1_posts_raw[oc] - synth_post_plac)
                    )
                    placebo_atts_per_outcome[oc].append(att_plac_k)
                    plac_atts_k.append(att_plac_k)

                placebo_atts_overall.append(float(np.mean(plac_atts_k)))

            except Exception:
                continue

    # ------------------------------------------------------------------
    #  Per-outcome SE, p-value from placebos
    # ------------------------------------------------------------------
    per_outcome_se: Dict[str, float] = {}
    per_outcome_pval: Dict[str, float] = {}

    for oc in outcomes:
        plac = np.array(placebo_atts_per_outcome[oc])
        if len(plac) > 1:
            per_outcome_se[oc] = float(np.std(plac, ddof=1))
            pv = float(np.mean(np.abs(plac) >= abs(per_outcome_att[oc])))
            per_outcome_pval[oc] = max(pv, 1 / (len(plac) + 1))
        else:
            per_outcome_se[oc] = np.nan
            per_outcome_pval[oc] = np.nan

    # Overall SE and p-value
    if len(placebo_atts_overall) > 1:
        overall_se = float(np.std(placebo_atts_overall, ddof=1))
        overall_pvalue = float(
            np.mean(np.abs(placebo_atts_overall) >= abs(overall_att))
        )
        overall_pvalue = max(overall_pvalue, 1 / (len(placebo_atts_overall) + 1))
    else:
        overall_se = np.nan
        overall_pvalue = np.nan

    # ------------------------------------------------------------------
    #  Joint p-value: Fisher combination of per-outcome p-values
    # ------------------------------------------------------------------
    valid_pvals = [per_outcome_pval[oc] for oc in outcomes
                   if np.isfinite(per_outcome_pval[oc])
                   and per_outcome_pval[oc] > 0]
    if len(valid_pvals) >= 2:
        # Fisher's method: -2 * sum(log(p_k)) ~ chi2(2K)
        fisher_stat = -2.0 * np.sum(np.log(valid_pvals))
        joint_pvalue = float(
            1 - sp_stats.chi2.cdf(fisher_stat, df=2 * len(valid_pvals))
        )
    elif len(valid_pvals) == 1:
        joint_pvalue = valid_pvals[0]
    else:
        joint_pvalue = np.nan

    # Confidence interval
    z_crit = sp_stats.norm.ppf(1 - alpha / 2)
    ci = (overall_att - z_crit * overall_se,
          overall_att + z_crit * overall_se)

    # ------------------------------------------------------------------
    #  Assemble results
    # ------------------------------------------------------------------
    per_outcome_df = pd.DataFrame({
        "outcome": outcomes,
        "att": [per_outcome_att[oc] for oc in outcomes],
        "se": [per_outcome_se[oc] for oc in outcomes],
        "pvalue": [per_outcome_pval[oc] for oc in outcomes],
    })

    weights_dict = dict(zip(donors, omega))

    model_info: Dict[str, Any] = {
        "method": method,
        "per_outcome_effects": per_outcome_df,
        "weights": weights_dict,
        "n_outcomes": K,
        "n_donors": J,
        "n_pre_periods": T0,
        "n_post_periods": T1,
        "gap_tables": gap_tables,
        "joint_pvalue": joint_pvalue,
        "treatment_time": treatment_time,
        "treated_unit": treated_unit,
        "Y_synth": Y_synth_dict,
        "Y_treated": Y_treated_dict,
        "times": all_times,
        "standardize": standardize,
        "penalization": penalization,
    }

    if placebo_atts_overall:
        model_info["placebo_atts_overall"] = placebo_atts_overall
        model_info["placebo_atts_per_outcome"] = {
            oc: placebo_atts_per_outcome[oc] for oc in outcomes
        }
        model_info["n_placebos"] = len(placebo_atts_overall)

    return CausalResult(
        method="Multiple Outcomes SCM (Sun 2023)",
        estimand="ATT (multi-outcome)",
        estimate=overall_att,
        se=overall_se,
        pvalue=overall_pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=len(data),
        detail=per_outcome_df,
        model_info=model_info,
        _citation_key="multi_outcome_synth",
    )


# ====================================================================== #
#  Internal helpers
# ====================================================================== #

def _scm_weights(
    y_treated: np.ndarray,
    Y_donors: np.ndarray,
    penalization: float = 0.0,
) -> np.ndarray:
    """
    Simplex-constrained QP: min ||y - Y' w||^2 + pen * ||w||^2
    subject to w >= 0, sum(w) = 1.

    Parameters
    ----------
    y_treated : ndarray, shape (D,)
        Treated unit's (possibly stacked) pre-treatment vector.
    Y_donors : ndarray, shape (J, D)
        Donor matrix (rows = donors, cols = stacked time / outcomes).
    penalization : float
        Ridge penalty on weights.

    Returns
    -------
    w : ndarray, shape (J,)
    """
    J = Y_donors.shape[0]

    def objective(w):
        resid = y_treated - Y_donors.T @ w
        return float(np.sum(resid ** 2) + penalization * np.sum(w ** 2))

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0)] * J
    w0 = np.ones(J) / J

    result = minimize(
        objective, w0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    return result.x


def _concatenated_weights(
    Y1_pres: Dict[str, np.ndarray],
    Y0_pres: Dict[str, np.ndarray],
    outcomes: List[str],
    penalization: float,
) -> np.ndarray:
    """
    Concatenated approach: stack K outcomes vertically.

    Z_treated = [Y^(1)_pre; Y^(2)_pre; ...; Y^(K)_pre]   shape (K*T0,)
    Z_donors  = [Y0^(1)_pre; Y0^(2)_pre; ...; Y0^(K)_pre] shape (J, K*T0)

    Then solve SCM on the stacked vectors.
    """
    z_treated_parts = []
    z_donors_parts = []
    for oc in outcomes:
        z_treated_parts.append(Y1_pres[oc])        # (T0,)
        z_donors_parts.append(Y0_pres[oc])          # (J, T0)

    z_treated = np.concatenate(z_treated_parts)           # (K*T0,)
    z_donors = np.hstack(z_donors_parts)                  # (J, K*T0)

    return _scm_weights(z_treated, z_donors, penalization)


def _averaged_weights(
    Y1_pres: Dict[str, np.ndarray],
    Y0_pres: Dict[str, np.ndarray],
    outcomes: List[str],
    penalization: float,
) -> np.ndarray:
    """
    Averaged approach: take element-wise mean across K standardised
    outcomes, then solve SCM on the average series.

    This is simpler and faster but may discard information when outcome
    factor loadings differ substantially.
    """
    K = len(outcomes)
    y1_avg = np.mean([Y1_pres[oc] for oc in outcomes], axis=0)   # (T0,)
    Y0_avg = np.mean([Y0_pres[oc] for oc in outcomes], axis=0)   # (J, T0)

    return _scm_weights(y1_avg, Y0_avg, penalization)


# ====================================================================== #
#  Citation
# ====================================================================== #

CausalResult._CITATIONS["multi_outcome_synth"] = (
    "@article{sun2023multiple,\n"
    "  title={Using Multiple Outcomes to Improve the Synthetic "
    "Control Method},\n"
    "  author={Sun, Liyang},\n"
    "  journal={Review of Economics and Statistics},\n"
    "  year={2023},\n"
    "  publisher={MIT Press}\n"
    "}"
)
