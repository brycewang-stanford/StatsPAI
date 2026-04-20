"""
Power Analysis and Sample Size Planning for Synthetic Control Designs.

A **novel** feature with no existing package equivalent.  Helps researchers
plan their SCM study by answering:

    *"Given my donor pool and pre-treatment periods, what is the minimum
    detectable effect size?"*

Functions
---------
- ``synth_power``   — full power curve across a grid of effect sizes
- ``synth_mde``     — quick MDE at a target power level
- ``synth_power_plot`` — visualise the power curve with MDE annotation

Algorithm
---------
1. Fit SCM on actual data to obtain baseline weights and the null
   (placebo) distribution of RMSPE ratios.
2. For each hypothetical effect size *delta*:
   a. Inject *delta* into the treated unit's post-treatment outcomes.
   b. Re-compute the treated RMSPE ratio under the augmented data.
   c. Compare against the pre-computed null distribution.
   d. Record whether H0 is rejected at level *alpha*.
3. Repeat over *n_simulations* with optional noise perturbation.
4. Power = fraction of simulations where H0 is rejected.
5. MDE = smallest *delta* where power >= 0.80.

References
----------
Abadie, A., Diamond, A. and Hainmueller, J. (2010).
"Synthetic Control Methods for Comparative Case Studies."
*JASA*, 105(490), 493-505.

Firpo, S. and Possebom, V. (2018).
"Synthetic Control Method: Inference, Sensitivity Analysis and
Confidence Sets."
*Journal of Causal Inference*, 6(2).
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats

from .scm import synth, SyntheticControl
from ..core.results import CausalResult


# ======================================================================
# Internal helpers
# ======================================================================

def _build_null_distribution(
    model: SyntheticControl,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Compute the placebo RMSPE-ratio distribution (null) from donor units.

    Parameters
    ----------
    model : SyntheticControl
        Fitted model with matrices already prepared.
    weights : np.ndarray
        Donor weights from the baseline fit.

    Returns
    -------
    np.ndarray
        Array of RMSPE ratios for each valid placebo unit.
    """
    all_units = np.column_stack([
        model.Y_treated[:, np.newaxis], model.Y_donors,
    ])

    ratios: list[float] = []

    for i in range(len(model.donor_units)):
        idx_placebo = i + 1  # treated is at index 0
        Y_placebo = all_units[:, idx_placebo]
        donor_idx = [j for j in range(all_units.shape[1]) if j != idx_placebo]
        Y_placebo_donors = all_units[:, donor_idx]

        Y_pre_p = Y_placebo[model.pre_mask]
        Y_pre_d = Y_placebo_donors[model.pre_mask]

        try:
            # Placebo unit isn't the model's treated unit, so its
            # predictor matrices aren't pre-computed.  Use pre-Y as
            # both the outer target and the inner predictor matrix
            # (equal-V SCM, the standard placebo setup in Abadie et al.).
            solver_out = model._solve_weights(
                Y_pre_p, Y_pre_d,
                Y_pre_p, Y_pre_d,
                run_nested=False,
            )
            w = solver_out["w"]
            synth_p = Y_placebo_donors @ w
            gap_p = Y_placebo - synth_p

            pre_mspe = float(np.mean(gap_p[model.pre_mask] ** 2))
            post_mspe = float(np.mean(gap_p[model.post_mask] ** 2))

            if pre_mspe > 1e-10:
                ratios.append(np.sqrt(post_mspe) / np.sqrt(pre_mspe))
        except Exception:
            continue

    return np.asarray(ratios, dtype=float)


def _treated_ratio_with_effect(
    model: SyntheticControl,
    weights: np.ndarray,
    delta: float,
    rng: np.random.Generator,
    noise_scale: float,
) -> float:
    """
    Compute the treated RMSPE ratio after injecting an effect *delta*.

    The simulation adds *delta* to the **actual** post-treatment outcomes
    (capturing real noise), plus a small Gaussian perturbation scaled to
    the pre-treatment residual standard deviation so that repeated
    simulations are not identical.

    Parameters
    ----------
    model : SyntheticControl
        Fitted model.
    weights : np.ndarray
        Baseline donor weights.
    delta : float
        Hypothetical additive treatment effect.
    rng : np.random.Generator
        Random number generator for noise.
    noise_scale : float
        Standard deviation of the pre-treatment residual (used to
        scale the perturbation).

    Returns
    -------
    float
        Post-RMSPE / pre-RMSPE ratio for the treated unit under *delta*.
    """
    Y_synth = model.Y_donors @ weights
    gap = model.Y_treated.copy() - Y_synth

    # Inject effect into post-treatment gap
    gap_sim = gap.copy()
    n_post = int(model.post_mask.sum())
    gap_sim[model.post_mask] += delta

    # Add small noise so repeated simulations are not deterministic
    if noise_scale > 0:
        perturbation = rng.normal(0, noise_scale * 0.1, size=n_post)
        gap_sim[model.post_mask] += perturbation

    pre_mspe = float(np.mean(gap_sim[model.pre_mask] ** 2))
    post_mspe = float(np.mean(gap_sim[model.post_mask] ** 2))

    if pre_mspe < 1e-10:
        return np.inf
    return np.sqrt(post_mspe) / np.sqrt(pre_mspe)


# ======================================================================
# 1.  synth_power — full power curve
# ======================================================================

def synth_power(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    treatment_time: Any,
    effect_sizes: Optional[Sequence[float]] = None,
    n_simulations: int = 200,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Power analysis for Synthetic Control designs.

    Estimates statistical power across a grid of hypothetical effect
    sizes using placebo-based inference.  Identifies the Minimum
    Detectable Effect (MDE) — the smallest effect where power >= 0.80.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel data.
    outcome : str
        Outcome variable name.
    unit : str
        Unit identifier column.
    time : str
        Time period column.
    treated_unit : any
        Identifier of the treated unit.
    treatment_time : any
        First treatment period (inclusive).
    effect_sizes : array-like of float, optional
        Grid of hypothetical additive effect sizes to evaluate.
        If ``None``, auto-generates 10 steps from 0 to
        3 * pre-treatment SD of the outcome.
    n_simulations : int, default 200
        Number of Monte-Carlo simulations per effect size.
    alpha : float, default 0.05
        Significance level for the placebo test.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: ``effect_size``, ``power``, ``n_rejections``,
        ``n_simulations``, ``mde_flag``.

        The ``mde_flag`` column is ``True`` for the row corresponding
        to the Minimum Detectable Effect (first row with power >= 0.80).

    Notes
    -----
    The null distribution is the set of RMSPE ratios from in-space
    placebos (computed once on the original data).  For each effect
    size, the simulation adds *delta* to the treated unit's
    post-treatment outcomes and re-computes the RMSPE ratio.  A small
    noise perturbation (10 % of pre-treatment residual SD) is added so
    that each simulation draw is unique.

    This is a **novel** diagnostic — no existing SCM package provides
    an equivalent power-planning tool.

    Examples
    --------
    >>> import statspai as sp
    >>> power_df = sp.synth_power(
    ...     df, outcome='gdp', unit='state', time='year',
    ...     treated_unit='California', treatment_time=1989,
    ...     n_simulations=500, seed=42,
    ... )
    >>> power_df
       effect_size  power  n_rejections  n_simulations  mde_flag
    0     0.000000   0.04            20            500     False
    1     1.234567   0.23           115            500     False
    ...
    8     9.876543   0.82           410            500      True
    9    11.111111   0.95           475            500     False

    >>> mde_row = power_df[power_df['mde_flag']]
    >>> print(f"MDE = {mde_row['effect_size'].values[0]:.2f}")

    See Also
    --------
    synth_mde : Quick MDE extraction.
    synth_power_plot : Visualise the power curve.
    """
    rng = np.random.default_rng(seed)

    # --- Step 1: baseline SCM fit ---
    model = SyntheticControl(
        data=data, outcome=outcome, unit=unit, time=time,
        treated_unit=treated_unit, treatment_time=treatment_time,
    )

    Y_pre_treated = model.Y_treated[model.pre_mask]
    Y_pre_donors = model.Y_donors[model.pre_mask]
    solver_out = model._solve_weights(
        Y_pre_treated, Y_pre_donors,
        model.X_treated, model.X_donors,
        run_nested=model._should_run_nested(),
    )
    weights = solver_out["w"]

    # Pre-treatment residual SD (noise scale for perturbation)
    Y_synth_pre = (model.Y_donors @ weights)[model.pre_mask]
    residual_sd = float(np.std(Y_pre_treated - Y_synth_pre))

    # --- Step 2: null distribution (compute once) ---
    null_ratios = _build_null_distribution(model, weights)
    if len(null_ratios) < 2:
        raise ValueError(
            "Need at least 2 valid placebo units to build the null "
            "distribution.  Consider adding more donors."
        )

    # Critical value: (1 - alpha) quantile of null distribution
    critical_value = float(np.quantile(null_ratios, 1 - alpha))

    # --- Step 3: auto-generate effect grid if needed ---
    if effect_sizes is None:
        pre_sd = float(np.std(Y_pre_treated))
        if pre_sd < 1e-10:
            pre_sd = 1.0
        effect_sizes = np.linspace(0, 3 * pre_sd, 10)
    else:
        effect_sizes = np.asarray(effect_sizes, dtype=float)

    # --- Step 4: simulate power for each effect size ---
    records: list[dict[str, Any]] = []

    for delta in effect_sizes:
        n_reject = 0
        for _ in range(n_simulations):
            ratio = _treated_ratio_with_effect(
                model, weights, delta, rng, residual_sd,
            )
            if ratio >= critical_value:
                n_reject += 1

        power = n_reject / n_simulations
        records.append({
            "effect_size": float(delta),
            "power": power,
            "n_rejections": n_reject,
            "n_simulations": n_simulations,
        })

    result_df = pd.DataFrame(records)

    # --- Step 5: flag MDE row ---
    mde_mask = result_df["power"] >= 0.80
    result_df["mde_flag"] = False
    if mde_mask.any():
        mde_idx = result_df.loc[mde_mask, "power"].index[0]
        result_df.loc[mde_idx, "mde_flag"] = True

    return result_df


# ======================================================================
# 2.  synth_mde — quick MDE extraction
# ======================================================================

def synth_mde(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    treatment_time: Any,
    power_target: float = 0.80,
    alpha: float = 0.05,
    n_simulations: int = 200,
    seed: Optional[int] = None,
) -> float:
    """
    Minimum Detectable Effect for a Synthetic Control design.

    Convenience wrapper around :func:`synth_power` that returns only
    the MDE (the smallest effect size achieving the target power).

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel data.
    outcome : str
        Outcome variable name.
    unit : str
        Unit identifier column.
    time : str
        Time period column.
    treated_unit : any
        Identifier of the treated unit.
    treatment_time : any
        First treatment period.
    power_target : float, default 0.80
        Desired power level.
    alpha : float, default 0.05
        Significance level for the placebo test.
    n_simulations : int, default 200
        Number of simulations per effect size.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    float
        Minimum detectable effect size.  Returns ``np.inf`` if no
        effect size in the default grid achieves the target power.

    Examples
    --------
    >>> import statspai as sp
    >>> mde = sp.synth_mde(
    ...     df, outcome='gdp', unit='state', time='year',
    ...     treated_unit='California', treatment_time=1989,
    ...     seed=42,
    ... )
    >>> print(f"MDE at 80%% power: {mde:.2f}")

    See Also
    --------
    synth_power : Full power curve with details.
    """
    power_df = synth_power(
        data=data,
        outcome=outcome,
        unit=unit,
        time=time,
        treated_unit=treated_unit,
        treatment_time=treatment_time,
        n_simulations=n_simulations,
        alpha=alpha,
        seed=seed,
    )

    mde_rows = power_df[power_df["power"] >= power_target]
    if mde_rows.empty:
        return np.inf

    return float(mde_rows["effect_size"].iloc[0])


# ======================================================================
# 3.  synth_power_plot — power curve visualisation
# ======================================================================

def synth_power_plot(
    power_result: pd.DataFrame,
    ax: Any = None,
    figsize: tuple = (9, 6),
    title: Optional[str] = None,
) -> Any:
    """
    Plot the power curve from :func:`synth_power`.

    Displays power (y-axis) against effect size (x-axis) with
    reference lines at power = 0.80 and the MDE.

    Parameters
    ----------
    power_result : pd.DataFrame
        Output of :func:`synth_power`.  Must contain columns
        ``effect_size``, ``power``, and ``mde_flag``.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.  If ``None``, a new figure is created.
    figsize : tuple, default (9, 6)
        Figure size (width, height) in inches.
    title : str, optional
        Custom plot title.  Defaults to
        ``"SCM Power Curve — Minimum Detectable Effect"``.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> import statspai as sp
    >>> power_df = sp.synth_power(
    ...     df, outcome='gdp', unit='state', time='year',
    ...     treated_unit='California', treatment_time=1989, seed=42,
    ... )
    >>> sp.synth_power_plot(power_df)

    See Also
    --------
    synth_power : Compute the power curve.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    effect = power_result["effect_size"]
    power = power_result["power"]

    # Main power curve
    ax.plot(effect, power, "o-", color="#2c7bb6", linewidth=2,
            markersize=6, label="Power", zorder=3)

    # 80 % reference line
    ax.axhline(0.80, color="#d7191c", linestyle="--", linewidth=1,
               alpha=0.7, label="Power = 0.80")

    # MDE vertical line
    mde_rows = power_result[power_result["mde_flag"]]
    if not mde_rows.empty:
        mde_val = float(mde_rows["effect_size"].iloc[0])
        mde_power = float(mde_rows["power"].iloc[0])
        ax.axvline(mde_val, color="#fdae61", linestyle="--", linewidth=1,
                   alpha=0.8, label=f"MDE = {mde_val:.2f}")
        ax.plot(mde_val, mde_power, "D", color="#fdae61", markersize=10,
                zorder=4)

    ax.set_xlabel("Effect Size (additive)")
    ax.set_ylabel("Power")
    ax.set_title(title or "SCM Power Curve \u2014 Minimum Detectable Effect")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return ax
