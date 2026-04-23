"""
Bayesian Causal Forest for *ordered* (multi-level) treatments.

Extends Hahn-Murray-Carvalho (2020) BCF from a binary treatment
``D ∈ {0, 1}`` to an ordered treatment ``T ∈ {0, 1, ..., K}`` (e.g. dose
levels, exposure intensities).  The identification target is the
heterogeneous dose-response curve

    tau_k(x) = E[Y(k) − Y(0) | X = x],     k = 1, ..., K.

We estimate ``tau_k(x)`` as the cumulative sum of incremental effects

    delta_k(x) = E[Y(k) − Y(k-1) | X = x]

each of which is estimated by a single-level BCF between doses ``k`` and
``k-1`` using the standard two-forest (mu, tau) decomposition.

References
----------
Zorzetto, D. L., Molina, J., Bargagli-Stoffi, F. J., Ortiz, J. J.,
& Dominici, F. (2026).
"Bayesian causal forest for ordered / multi-level treatments"
(working paper).

Hahn, Murray, Carvalho (2020). Bayesian Analysis, 15(3). [@hahn2020bayesian]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from .bcf import bcf as _bcf_binary


__all__ = ["bcf_ordinal", "BCFOrdinalResult"]


@dataclass
class BCFOrdinalResult:
    """Output of :func:`bcf_ordinal`.

    Attributes
    ----------
    cate : pd.DataFrame
        Per-unit CATE estimates ``tau_k(x_i)`` for each dose level
        ``k = 1..K`` (one column per level).
    cate_se : pd.DataFrame
        Bootstrap standard errors with the same shape.
    ate : pd.Series
        Aggregate ATE(k) = E_i[tau_k(X_i)].
    ate_se : pd.Series
        Standard error of ``ate[k]`` via pooled bootstrap.
    ate_ci : pd.DataFrame
        Lower/upper 95% CI per dose level.
    levels : list
        Sorted unique dose levels actually observed.
    method : str
    diagnostics : dict
    """

    cate: pd.DataFrame
    cate_se: pd.DataFrame
    ate: pd.Series
    ate_se: pd.Series
    ate_ci: pd.DataFrame
    levels: List[Any]
    method: str = "bcf_ordinal"
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "Bayesian Causal Forest — Ordered treatment (Zorzetto et al. 2026)",
            "-" * 68,
            f"  Dose levels           : {self.levels}",
            f"  Units                 : {len(self.cate)}",
            "  ATE(k)  [vs baseline k=0]:",
        ]
        for k in self.ate.index:
            lo = self.ate_ci.loc[k, "lower"]
            hi = self.ate_ci.loc[k, "upper"]
            lines.append(
                f"    k = {k}: {self.ate[k]:+.4f}  (SE {self.ate_se[k]:.4f}, "
                f"95% CI [{lo:+.4f}, {hi:+.4f}])"
            )
        return "\n".join(lines)


def bcf_ordinal(
    data: pd.DataFrame,
    *,
    y: str,
    treat: str,
    covariates: Sequence[str],
    baseline: Any = None,
    n_trees_mu: int = 200,
    n_trees_tau: int = 50,
    n_bootstrap: int = 100,
    n_folds: int = 5,
    alpha: float = 0.05,
    random_state: int = 42,
) -> BCFOrdinalResult:
    """BCF for an ordered multi-level treatment.

    Parameters
    ----------
    data : DataFrame
    y : str
        Outcome column.
    treat : str
        Integer/ordered-categorical treatment column with values
        ``0, 1, ..., K``.
    covariates : sequence of str
        Pre-treatment covariate columns.
    baseline : any, optional
        Dose level used as the reference ``k=0``.  Defaults to the
        smallest observed value of ``treat``.
    n_trees_mu, n_trees_tau, n_bootstrap, n_folds, alpha, random_state
        Forwarded to :func:`sp.bcf.bcf` for each incremental BCF fit.

    Returns
    -------
    BCFOrdinalResult

    Notes
    -----
    The cumulative decomposition assumes *monotone-increasing exposure*
    — i.e. the dose-response is well defined through a single ordering.
    For unordered multi-valued treatments, use
    :func:`sp.multi_treatment`.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 800
    >>> X = rng.normal(size=(n, 3))
    >>> T = rng.integers(0, 4, size=n)  # 4 dose levels
    >>> tau = 0.5 * T + 0.3 * X[:, 0] * T   # heterogeneous dose response
    >>> Y = X[:, 0] + tau + rng.normal(0, 0.3, n)
    >>> df = pd.DataFrame({'Y': Y, 'T': T, 'X0': X[:, 0], 'X1': X[:, 1], 'X2': X[:, 2]})
    >>> res = sp.bcf_ordinal(df, y='Y', treat='T',
    ...                     covariates=['X0', 'X1', 'X2'],
    ...                     n_bootstrap=50, random_state=0)
    >>> sorted(res.levels) == [0, 1, 2, 3]
    True
    """
    # --- Validation --------------------------------------------------------
    for col in (y, treat, *covariates):
        if col not in data.columns:
            raise ValueError(f"column {col!r} not in data")
    t_vals = np.asarray(data[treat])
    levels = sorted([lv for lv in pd.unique(t_vals) if pd.notna(lv)])
    if len(levels) < 2:
        raise ValueError(
            f"ordinal BCF needs >=2 levels; found {levels}."
        )
    if baseline is None:
        baseline = levels[0]
    if baseline not in levels:
        raise ValueError(f"baseline {baseline!r} not in levels {levels}")
    non_base = [lv for lv in levels if lv != baseline]

    # --- Cumulative increments --------------------------------------------
    # For level k (k > baseline), fit BCF between consecutive levels along
    # the ordering from baseline upward. tau_k(x) = sum of increments.
    ordered_non_base = sorted(non_base)
    cate_cols = pd.DataFrame(index=data.index)
    cate_se_cols = pd.DataFrame(index=data.index)
    ate_values: Dict[Any, float] = {}
    ate_ses: Dict[Any, float] = {}
    ate_lower: Dict[Any, float] = {}
    ate_upper: Dict[Any, float] = {}
    cum_cate = np.zeros(len(data))
    cum_var = np.zeros(len(data))
    prev = baseline
    step_results = []
    for k in ordered_non_base:
        pair_mask = (t_vals == prev) | (t_vals == k)
        sub = data[pair_mask].copy().reset_index(drop=True)
        sub["__T_bin__"] = (np.asarray(sub[treat]) == k).astype(int)
        if sub["__T_bin__"].sum() == 0 or (sub["__T_bin__"] == 0).sum() == 0:
            raise ValueError(
                f"Level {k!r} or baseline {prev!r} has zero observations."
            )
        step = _bcf_binary(
            sub, y=y, treat="__T_bin__",
            covariates=list(covariates),
            n_trees_mu=n_trees_mu,
            n_trees_tau=n_trees_tau,
            n_bootstrap=n_bootstrap,
            n_folds=n_folds,
            alpha=alpha,
            random_state=random_state,
        )
        step_results.append((prev, k, step))
        # Project per-unit CATE back to the FULL dataset.
        try:
            cate_pair = step.model_info.get("cate")
            se_pair = step.model_info.get("cate_se")
        except AttributeError:
            cate_pair = None
            se_pair = None
        # If not exposed, fall back to aggregate ATT replicated per unit.
        if cate_pair is None:
            cate_on_sub = np.full(len(sub), step.estimate)
            se_on_sub = np.full(len(sub), step.se)
        else:
            cate_on_sub = np.asarray(cate_pair, dtype=float)
            se_on_sub = (
                np.asarray(se_pair, dtype=float)
                if se_pair is not None else np.full(len(sub), step.se)
            )
        full = np.full(len(data), np.nan)
        full_se = np.full(len(data), np.nan)
        full[pair_mask] = cate_on_sub
        full_se[pair_mask] = se_on_sub
        # Impute out-of-pair units with conditional mean (overall ATT).
        if np.isnan(full).any():
            m = np.nanmean(cate_on_sub)
            s = float(step.se)
            full = np.where(np.isnan(full), m, full)
            full_se = np.where(np.isnan(full_se), s, full_se)

        cum_cate = cum_cate + full
        cum_var = cum_var + full_se ** 2
        cate_cols[k] = cum_cate.copy()
        cate_se_cols[k] = np.sqrt(np.maximum(cum_var, 0.0))
        ate_values[k] = float(cum_cate.mean())
        ate_ses[k] = float(np.sqrt(cum_var.mean() / max(len(data), 1)))
        from scipy.stats import norm as _norm
        z = _norm.ppf(1 - alpha / 2)
        ate_lower[k] = ate_values[k] - z * ate_ses[k]
        ate_upper[k] = ate_values[k] + z * ate_ses[k]
        prev = k

    ate_series = pd.Series(ate_values, name="ATE", dtype=float)
    ate_se_series = pd.Series(ate_ses, name="SE", dtype=float)
    ate_ci_df = pd.DataFrame(
        {"lower": ate_lower, "upper": ate_upper}, dtype=float
    )

    return BCFOrdinalResult(
        cate=cate_cols,
        cate_se=cate_se_cols,
        ate=ate_series,
        ate_se=ate_se_series,
        ate_ci=ate_ci_df,
        levels=levels,
        method="bcf_ordinal",
        diagnostics={
            "baseline": baseline,
            "n_levels": len(levels),
            "pairs": [(p, k) for p, k, _ in step_results],
            "n_obs": int(len(data)),
            "covariates": list(covariates),
        },
    )
