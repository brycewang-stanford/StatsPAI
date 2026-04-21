"""
Iterative Conditional Expectation (ICE) parametric g-formula.

Given K time points with treatments A_0, ..., A_{K-1}, time-varying
confounders L_0, ..., L_{K-1}, and a scalar outcome Y, the ICE
estimator sequentially regresses Y on the history (A_k, L_k) under
the observed data distribution and recursively plugs in the
intervention of interest to obtain

    E[Y(a_0, ..., a_{K-1})] = E_{L_0} E_{L_1 | A_0 = a_0, L_0}
                              ... E[Y | hist, A_{K-1} = a_{K-1}].

Reference
---------
Bang, H. & Robins, J.M. (2005). "Doubly Robust Estimation in Missing
Data and Causal Inference Models." *Biometrics*, 61(4).
Hernan & Robins. *Causal Inference: What If*, ch. 21.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pandas as pd


@dataclass
class ICEResult:
    strategy: list
    value: float
    se: float
    ci: tuple[float, float]
    method: str = "parametric-g-formula-ICE"
    per_timepoint_means: list[float] = None

    def summary(self) -> str:
        return (
            f"g-formula ICE (strategy={self.strategy})\n"
            f"  E[Y({self.strategy})] = {self.value:.4f} "
            f"(SE {self.se:.4f}, 95% CI [{self.ci[0]:.4f}, {self.ci[1]:.4f}])"
        )


def ice(
    data: pd.DataFrame,
    id_col: str,
    time_col: str,
    treatment_cols: Sequence[str],
    confounder_cols: Sequence[Sequence[str]] | Sequence[str],
    outcome_col: str,
    treatment_strategy,
    bootstrap: int = 0,
    seed: int | None = None,
) -> ICEResult:
    """Parametric g-formula estimate of E[Y(a_0, ..., a_{K-1})].

    Parameters
    ----------
    data : pd.DataFrame
        Wide-format dataframe: one row per subject, with columns
        listed in ``treatment_cols``, ``confounder_cols``, and the
        scalar ``outcome_col``.
    id_col : str
    time_col : str
        Present for documentation / downstream hooks; ICE itself
        works on the wide-format table.
    treatment_cols : list[str]
        Treatment column at each time point, in order.
    confounder_cols : list[list[str]] | list[str]
        Confounders available at each time point. May be a flat list
        (same confounders at every time point) or a nested list
        (time-specific confounders).
    outcome_col : str
        Terminal outcome measured at the end of follow-up.
    treatment_strategy : list | callable
        Either a static sequence of treatment values (e.g. ``[1, 1, 1]``
        = always-treat) or a callable taking history and returning
        the intervention value.
    bootstrap : int, default 0
        Number of nonparametric bootstrap replicates for SE. 0 uses
        analytic linear-regression SE of the terminal mean.
    seed : int, optional
        Random seed for bootstrap.

    Returns
    -------
    ICEResult
    """
    K = len(treatment_cols)
    if not isinstance(confounder_cols[0], (list, tuple)):
        confounder_cols = [list(confounder_cols)] * K
    else:
        confounder_cols = [list(c) for c in confounder_cols]

    strategy = _resolve_strategy(treatment_strategy, K)

    val = _ice_once(data, treatment_cols, confounder_cols, outcome_col, strategy)

    if bootstrap > 0:
        rng = np.random.default_rng(seed)
        n = len(data)
        vals = []
        for _ in range(bootstrap):
            idx = rng.integers(0, n, n)
            bd = data.iloc[idx].reset_index(drop=True)
            vals.append(
                _ice_once(bd, treatment_cols, confounder_cols, outcome_col, strategy)
            )
        vals = np.array(vals)
        se = float(vals.std(ddof=1))
        ci = (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)))
    else:
        se = float(
            data[outcome_col].std(ddof=1) / np.sqrt(max(len(data), 1))
        )
        ci = (val - 1.96 * se, val + 1.96 * se)

    return ICEResult(
        strategy=list(strategy),
        value=float(val),
        se=se,
        ci=(float(ci[0]), float(ci[1])),
    )


def gformula_ice(*args, **kwargs) -> ICEResult:
    """Alias for :func:`ice` to match Stata's gformula naming."""
    return ice(*args, **kwargs)


# --------------------------------------------------------------------------- #
#  Internal sequential regression
# --------------------------------------------------------------------------- #

def _ice_once(
    data: pd.DataFrame,
    treatment_cols: list[str],
    confounder_cols: list[list[str]],
    outcome_col: str,
    strategy: list,
) -> float:
    df = data.copy()
    # pseudo outcome: start with Y_K = Y
    pseudo = df[outcome_col].to_numpy(dtype=float)
    K = len(treatment_cols)

    # Walk backwards from t=K-1 ... 0
    for t in reversed(range(K)):
        hist = []
        for s in range(t + 1):
            hist.extend(confounder_cols[s])
            hist.append(treatment_cols[s])
        # Fit linear regression of pseudo on hist (+ intercept)
        X = df[hist].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(X)), X])
        beta, *_ = np.linalg.lstsq(X, pseudo, rcond=None)

        # Plug-in the strategy's A_t value
        X_intervened = X.copy()
        # Column index of A_t in hist is (t-th A slot): count L's before + t A's
        a_offset = 1  # intercept
        col_idx = None
        for s in range(t):
            a_offset += len(confounder_cols[s]) + 1
        # after loop, a_offset points to L_t's start; add len(L_t) for A_t
        col_idx = a_offset + len(confounder_cols[t])
        a_t_val = strategy[t]
        X_intervened[:, col_idx] = a_t_val

        pseudo = X_intervened @ beta

    return float(pseudo.mean())


def _resolve_strategy(strategy, K: int) -> list:
    if callable(strategy):
        return [strategy(t) for t in range(K)]
    if isinstance(strategy, (list, tuple, np.ndarray)):
        if len(strategy) != K:
            raise ValueError(f"strategy length {len(strategy)} != K={K}")
        return list(strategy)
    if isinstance(strategy, (int, float)):
        return [float(strategy)] * K
    raise TypeError(
        "treatment_strategy must be list[int|float], callable, or scalar"
    )
