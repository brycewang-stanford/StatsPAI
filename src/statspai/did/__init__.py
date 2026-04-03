"""
Difference-in-Differences (DID) module for StatsPAI.

Provides estimators for:
- Classic 2x2 DID (two groups, two periods)
- Staggered DID with heterogeneous treatment effects (Callaway & Sant'Anna, 2021)

Planned:
- Sun & Abraham (2021) interaction-weighted event study
- Goodman-Bacon (2021) decomposition
- Doubly Robust DID (Sant'Anna & Zhao, 2020) — standalone
"""

from typing import Optional, List

import pandas as pd

from ..core.results import CausalResult
from .did_2x2 import did_2x2
from .callaway_santanna import callaway_santanna


def did(
    data: pd.DataFrame,
    y: str,
    treat: str,
    time: str,
    id: Optional[str] = None,
    covariates: Optional[List[str]] = None,
    method: str = 'auto',
    estimator: str = 'dr',
    control_group: str = 'nevertreated',
    base_period: str = 'universal',
    cluster: Optional[str] = None,
    robust: bool = True,
    alpha: float = 0.05,
    **kwargs,
) -> CausalResult:
    """
    Difference-in-Differences estimation.

    Unified entry point that auto-detects 2×2 vs. staggered designs
    and dispatches to the appropriate estimator.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    y : str
        Outcome variable name.
    treat : str
        Treatment variable. For 2×2: binary group indicator (0/1).
        For staggered: first treatment period (0 = never treated).
    time : str
        Time period variable.
    id : str, optional
        Unit identifier. Required for staggered DID (Callaway-Sant'Anna).
    covariates : list of str, optional
        Covariate names for conditional parallel trends / controls.
    method : str, default 'auto'
        - ``'auto'`` — 2×2 if ``id`` is None and treatment is binary,
          else Callaway-Sant'Anna.
        - ``'2x2'`` — classic two-period, two-group DID.
        - ``'callaway_santanna'`` or ``'cs'`` — staggered DID.
    estimator : str, default 'dr'
        For staggered DID: ``'dr'`` (doubly robust), ``'ipw'``, ``'reg'``.
    control_group : str, default 'nevertreated'
        For staggered DID: ``'nevertreated'`` or ``'notyettreated'``.
    base_period : str, default 'universal'
        For staggered DID: ``'universal'`` (always g−1) or ``'varying'``.
    cluster : str, optional
        Cluster variable for standard errors (2×2 only).
    robust : bool, default True
        HC1 robust standard errors (2×2 only).
    alpha : float, default 0.05
        Significance level for confidence intervals.

    Returns
    -------
    CausalResult
        Estimation results with ``.summary()``, ``.plot()``,
        ``.to_latex()``, ``.cite()`` methods.

    Examples
    --------
    Classic 2×2 DID:

    >>> result = did(df, y='wage', treat='treated', time='post')
    >>> print(result.summary())

    Staggered DID (Callaway & Sant'Anna):

    >>> result = did(df, y='earnings', treat='first_treat',
    ...             time='year', id='worker_id')
    >>> result.event_study_plot()
    """
    # Auto-detect method
    if method == 'auto':
        if id is not None:
            method = 'callaway_santanna'
        else:
            treat_vals = set(data[treat].dropna().unique())
            if treat_vals <= {0, 1, True, False}:
                method = '2x2'
            else:
                raise ValueError(
                    f"Cannot auto-detect DID type. Treatment '{treat}' has "
                    f"values {sorted(treat_vals)}. Provide 'id' for staggered "
                    "DID, or set method='2x2' or method='callaway_santanna'."
                )

    if method == '2x2':
        return did_2x2(
            data, y=y, treat=treat, time=time,
            covariates=covariates, cluster=cluster,
            robust=robust, alpha=alpha,
        )

    if method in ('callaway_santanna', 'cs'):
        if id is None:
            raise ValueError(
                "'id' (unit identifier) is required for staggered DID."
            )
        return callaway_santanna(
            data, y=y, g=treat, t=time, i=id,
            x=covariates, estimator=estimator,
            control_group=control_group,
            base_period=base_period, alpha=alpha,
        )

    raise ValueError(
        f"Unknown DID method: '{method}'. "
        "Available: '2x2', 'callaway_santanna' (or 'cs')."
    )


__all__ = [
    'did',
    'did_2x2',
    'callaway_santanna',
]
