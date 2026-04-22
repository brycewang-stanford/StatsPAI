r"""Synthetic Survival Control (Agarwal-Shah 2025, arXiv:2511.14133).

Estimates the **survival difference** caused by treatment for a single treated
unit (or a small set thereof) by constructing a synthetic control on the
log-cumulative-hazard scale.  This is the survival-data analogue of the
classical Abadie-Diamond-Hainmueller SCM:

1. Transform each control unit's Kaplan-Meier survival curve :math:`S_i(t)` to
   :math:`L_i(t) = \log(-\log S_i(t))` (the "complementary log-log" or
   "log-cumulative-hazard" scale) so that time-varying covariate adjustments
   act linearly.
2. Solve a nonnegative-weight, unit-simplex least-squares fit that matches the
   treated unit's pre-treatment :math:`L_1(t)` by a convex combination of the
   donor :math:`L_j(t)`.
3. Project the weighted donor hazard to the post-treatment window and invert
   the link to obtain the counterfactual survival curve
   :math:`\hat S_1^{(0)}(t)`.
4. Report the gap :math:`S_1(t) - \hat S_1^{(0)}(t)` with a placebo-test
   uniform band constructed by permuting the treated-vs-donor label.

This operates on panel data where each row describes one unit's survival at a
time point — typically derived from Kaplan-Meier estimates per unit using
:func:`sp.km_estimate` or from clinical trial data aggregated at the group
level.

References
----------
Agarwal, A. and Shah, D. (2025).  "Synthetic Survival Control."
    arXiv:2511.14133.
Abadie, A., Diamond, A. and Hainmueller, J. (2010).  "Synthetic Control
    Methods for Comparative Case Studies: Estimating the Effect of
    California's Tobacco Control Program."  *JASA*, 105(490), 493-505.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


__all__ = ["synth_survival", "SyntheticSurvivalResult"]


@dataclass
class SyntheticSurvivalResult:
    """Output of :func:`synth_survival`."""
    treated_unit: str
    time_grid: np.ndarray
    s_treated: np.ndarray
    s_synth: np.ndarray
    gap: np.ndarray
    weights: Dict[str, float]
    treat_time: float
    alpha: float
    ci_low: Optional[np.ndarray] = None
    ci_high: Optional[np.ndarray] = None
    pre_rmse: Optional[float] = None
    placebo_gaps: Optional[np.ndarray] = None

    def summary(self) -> str:
        post_mask = self.time_grid >= self.treat_time
        avg_gap = float(np.mean(self.gap[post_mask]))
        rows = [
            "Synthetic Survival Control",
            "=" * 42,
            f"  Treated unit      : {self.treated_unit}",
            f"  Treatment time    : {self.treat_time}",
            f"  N grid points     : {len(self.time_grid)}",
            f"  Pre-treat RMSE    : {self.pre_rmse:.4f}" if self.pre_rmse is not None else "",
            f"  Mean post-gap S(t): {avg_gap:+.4f}",
            "  Top-5 donor weights:",
        ]
        top = sorted(self.weights.items(), key=lambda kv: -abs(kv[1]))[:5]
        for name, w in top:
            rows.append(f"    {name:<20s} {w:.4f}")
        return "\n".join([r for r in rows if r])


def _cloglog(S: np.ndarray) -> np.ndarray:
    """Complementary log-log of a survival curve."""
    S = np.clip(S, 1e-6, 1 - 1e-6)
    return np.log(-np.log(S))


def _inv_cloglog(L: np.ndarray) -> np.ndarray:
    """Inverse: ``exp(-exp(L))``."""
    return np.exp(-np.exp(np.clip(L, -50, 50)))


def _simplex_ls(Y: np.ndarray, X: np.ndarray, n_iter: int = 2000) -> np.ndarray:
    """Minimise ||Y - X w||^2 over w in the unit simplex (w_j >= 0, sum w_j = 1).

    Projected-gradient with an exponentiated-gradient-style step — robust
    enough for small donor pools without pulling in ``cvxpy``.
    """
    n_donors = X.shape[1]
    w = np.ones(n_donors) / n_donors
    lr = 0.3
    for _ in range(n_iter):
        grad = X.T @ (X @ w - Y)
        # Multiplicative EG step
        w = w * np.exp(-lr * grad)
        w = np.clip(w, 1e-12, None)
        w /= w.sum()
        lr *= 0.9995
    return w


def synth_survival(
    data: pd.DataFrame,
    unit: str,
    time: str,
    survival: str,
    treated: str,
    treat_time: float,
    alpha: float = 0.05,
    n_placebos: int = 100,
    seed: int = 0,
) -> SyntheticSurvivalResult:
    """Synthetic Survival Control estimator.

    Parameters
    ----------
    data : pd.DataFrame
        Long panel: one row per (unit, time) with a precomputed Kaplan-Meier
        survival probability in column ``survival``.  Each unit should have
        the *same* time grid (or be padded by forward/back-fill before
        calling — ragged grids are not accepted).
    unit : str
        Unit (panel-id) column.
    time : str
        Time grid column.
    survival : str
        Column containing the survival probability :math:`S_i(t)`
        (in :math:`(0,1)`).
    treated : str
        Column containing the name of the single treated unit.  Accepts
        either a boolean column or a dedicated string/int identifier.
    treat_time : float
        Time at which treatment starts (times >= ``treat_time`` are the
        post-treatment window).
    alpha : float, default 0.05
        Uniform placebo CI level.
    n_placebos : int, default 100
        Number of placebo permutations used to bootstrap the uniform band.
    seed : int, default 0

    Returns
    -------
    SyntheticSurvivalResult
        Fitted counterfactual survival curve, gap trajectory, donor
        weights, and a placebo-based uniform confidence band.

    Examples
    --------
    >>> import statspai as sp
    >>> r = sp.synth_survival(
    ...     df, unit="trial_arm", time="month",
    ...     survival="km_est", treated="treated_arm", treat_time=6,
    ... )
    >>> r.summary()
    """
    for col in [unit, time, survival]:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data")

    df = data.copy()

    # Detect treated unit
    if treated in df.columns:
        if df[treated].dtype == bool:
            treated_units = df.loc[df[treated], unit].unique()
        else:
            treated_units = df.loc[df[treated].astype(bool), unit].unique()
        if len(treated_units) != 1:
            raise ValueError(
                "Exactly one treated unit expected; got "
                f"{len(treated_units)}."
            )
        treated_unit = str(treated_units[0])
    else:
        # Interpret ``treated`` as the explicit unit name
        treated_unit = str(treated)

    wide = df.pivot(index=time, columns=unit, values=survival).sort_index()
    time_grid = wide.index.to_numpy(dtype=float)
    if treated_unit not in wide.columns:
        raise ValueError(
            f"Treated unit '{treated_unit}' not found among units: "
            f"{list(wide.columns)}"
        )

    # Transform to complementary log-log scale
    L_wide = wide.apply(_cloglog)
    donors = [c for c in wide.columns if c != treated_unit]
    if not donors:
        from statspai.exceptions import DataInsufficient
        raise DataInsufficient(
            "At least one donor unit required",
            recovery_hint=(
                "synth_survival needs at least one untreated (donor) unit. "
                "Check the unit / treatment_unit columns."
            ),
            diagnostics={"n_donors": 0},
            alternative_functions=[],
        )

    pre_mask = time_grid < treat_time
    if pre_mask.sum() < 2:
        raise ValueError(
            "At least two pre-treatment time points required "
            f"(got {int(pre_mask.sum())})"
        )

    Y_pre = L_wide[treated_unit].to_numpy()[pre_mask]
    X_pre = L_wide[donors].to_numpy()[pre_mask]
    weights = _simplex_ls(Y_pre, X_pre)

    L_synth = L_wide[donors].to_numpy() @ weights
    s_synth = _inv_cloglog(L_synth)
    s_treated = wide[treated_unit].to_numpy()
    gap = s_treated - s_synth
    pre_rmse = float(np.sqrt(np.mean((Y_pre - X_pre @ weights) ** 2)))

    # --- Placebo test: uniform CI via in-space permutation -------- #
    rng = np.random.default_rng(seed)
    placebo_gaps = []
    candidate_donors = donors.copy()
    n_sample = min(n_placebos, len(candidate_donors))
    if n_sample > 0 and len(candidate_donors) > 1:
        chosen = rng.choice(candidate_donors, size=n_sample, replace=False)
        for placebo in chosen:
            others = [d for d in candidate_donors if d != placebo]
            Y_pre_p = L_wide[placebo].to_numpy()[pre_mask]
            X_pre_p = L_wide[others].to_numpy()[pre_mask]
            try:
                w_p = _simplex_ls(Y_pre_p, X_pre_p)
            except Exception:
                continue
            L_synth_p = L_wide[others].to_numpy() @ w_p
            s_synth_p = _inv_cloglog(L_synth_p)
            placebo_gaps.append(wide[placebo].to_numpy() - s_synth_p)
    placebo_gaps_arr = np.asarray(placebo_gaps) if placebo_gaps else None

    if placebo_gaps_arr is not None and len(placebo_gaps_arr) >= 2:
        q_low = np.quantile(placebo_gaps_arr, alpha / 2, axis=0)
        q_high = np.quantile(placebo_gaps_arr, 1 - alpha / 2, axis=0)
        ci_low = gap + q_low  # re-centred
        ci_high = gap + q_high
    else:
        ci_low = ci_high = None

    return SyntheticSurvivalResult(
        treated_unit=treated_unit,
        time_grid=time_grid,
        s_treated=s_treated,
        s_synth=s_synth,
        gap=gap,
        weights=dict(zip(donors, weights.tolist())),
        treat_time=float(treat_time),
        alpha=alpha,
        ci_low=ci_low,
        ci_high=ci_high,
        pre_rmse=pre_rmse,
        placebo_gaps=placebo_gaps_arr,
    )
