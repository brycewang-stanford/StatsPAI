"""
Transportability via density-ratio weighting (Stuart et al. 2011;
Dahabreh et al. 2020).

Given an RCT or observational study on a source population S and a
target population T (with only baseline covariates observed in T),
transport the causal effect via weights

    w_i = P(T) / P(S | X_i)   for i in S.

Also known as inverse odds of sampling weighting.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pandas as pd


@dataclass
class TransportWeightResult:
    weights: np.ndarray
    ess: float
    max_weight: float
    effect_source: float
    effect_transported: float
    se_transported: float

    def summary(self) -> str:
        return (
            f"Transport via density-ratio weighting\n"
            f"  Source effect: {self.effect_source:.4f}\n"
            f"  Transported effect: {self.effect_transported:.4f} "
            f"(SE {self.se_transported:.4f})\n"
            f"  Effective sample size: {self.ess:.1f}  "
            f"Max weight: {self.max_weight:.2f}"
        )


def transport_weights(
    source: pd.DataFrame,
    target: pd.DataFrame,
    features: Sequence[str],
    treatment: str,
    outcome: str,
    truncate: tuple[float, float] | None = (0.01, 0.99),
) -> TransportWeightResult:
    """Compute transport weights and a transported ATE.

    Parameters
    ----------
    source, target : pd.DataFrame
        Source (must contain ``features``, ``treatment``, ``outcome``)
        and target (must contain ``features``).
    features : list[str]
        Baseline covariates shared by both populations that identify
        target-conditional exchangeability.
    treatment, outcome : str
        Column names in ``source``.
    truncate : tuple[float, float] | None
        Weight quantiles for truncation.

    Returns
    -------
    TransportWeightResult
    """
    features = list(features)
    missing = [c for c in features if c not in source.columns or c not in target.columns]
    if missing:
        raise KeyError(f"Missing features in source or target: {missing}")

    source = source.reset_index(drop=True).copy()
    target = target.reset_index(drop=True).copy()
    source["_pop"] = 1
    target["_pop"] = 0
    pooled = pd.concat([source[features + ["_pop"]], target[features + ["_pop"]]], ignore_index=True)

    X = pooled[features].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(X)), X])
    y = pooled["_pop"].to_numpy(dtype=float)

    from statspai.censoring.ipcw import _fit_logit, _sigmoid
    beta = _fit_logit(y, X)
    prob_source = np.clip(_sigmoid(X @ beta), 1e-6, 1 - 1e-6)

    prob_source_in_source = prob_source[: len(source)]
    pi_target = (pooled["_pop"] == 0).mean()
    pi_source = 1.0 - pi_target
    # Inverse odds weight: w_i = pi_target / pi_source * P(S=0|X) / P(S=1|X)
    w = (pi_target / pi_source) * (1 - prob_source_in_source) / prob_source_in_source

    if truncate is not None:
        lo, hi = np.quantile(w[np.isfinite(w) & (w > 0)], list(truncate))
        w = np.clip(w, lo, hi)

    a = source[treatment].to_numpy(dtype=float)
    yobs = source[outcome].to_numpy(dtype=float)

    # Source effect (unweighted difference-in-means)
    src_effect = _wmean(yobs, a) - _wmean(yobs, 1 - a)

    # Transported effect (weighted difference)
    w1 = w * a
    w0 = w * (1 - a)
    mt1 = (w1 * yobs).sum() / max(w1.sum(), 1e-12)
    mt0 = (w0 * yobs).sum() / max(w0.sum(), 1e-12)
    transported = mt1 - mt0

    v1 = (w1 * (yobs - mt1) ** 2).sum() / max(w1.sum(), 1e-12) ** 2
    v0 = (w0 * (yobs - mt0) ** 2).sum() / max(w0.sum(), 1e-12) ** 2
    se = float(np.sqrt(v1 + v0))
    ess = float(w.sum() ** 2 / (w ** 2).sum())

    return TransportWeightResult(
        weights=w,
        ess=ess,
        max_weight=float(w.max()),
        effect_source=float(src_effect),
        effect_transported=float(transported),
        se_transported=se,
    )


def _wmean(y: np.ndarray, a: np.ndarray) -> float:
    if a.sum() == 0:
        return float("nan")
    return float((y * a).sum() / a.sum())
