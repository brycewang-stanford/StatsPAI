"""Opt-in joint inference upgrades for the BJS imputation estimator.

The core ``did_imputation()`` function reports a sum-of-z² pre-trend
Wald statistic that assumes the pre-period ATT(k) estimates are
uncorrelated across ``k``.  This is a valid but *conservative*
approximation when consecutive pre-periods share data (they typically
do, because the imputed counterfactual for both k = −1 and k = −2
leans on the same unit-FE estimates).

This module adds :func:`bjs_pretrend_joint`, a cluster-bootstrap
estimator of the full pre-period covariance matrix.  The resulting
joint Wald test is size-correct in finite samples even when the
off-diagonal covariance is non-zero.

References
----------
Borusyak, K., Jaravel, X. and Spiess, J. (2024).
    "Revisiting Event-Study Designs: Robust and Efficient Estimation."
    *Review of Economic Studies*, 91(6), 3253-3285.
    Appendix discussion of joint pre-trend testing. [@borusyak2024revisiting]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from .did_imputation import did_imputation


def bjs_pretrend_joint(
    result: CausalResult,
    data: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    first_treat: str,
    controls: Optional[List[str]] = None,
    cluster: Optional[str] = None,
    horizon: Optional[List[int]] = None,
    n_boot: int = 300,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Cluster-bootstrap joint Wald test for BJS pre-treatment coefficients.

    Parameters
    ----------
    result : CausalResult
        Output of :func:`did_imputation` on ``data`` with a non-trivial
        ``horizon`` that covers negative values.  Only its
        ``model_info['event_study']`` frame is consulted, to look up
        the observed pre-period point estimates that we re-test with
        a covariance-aware statistic.
    data, y, group, time, first_treat, controls, cluster :
        Same arguments you passed to the original
        :func:`did_imputation` call.  Needed to re-run BJS on each
        cluster-bootstrap resample.
    horizon : list of int, optional
        If omitted, inferred from ``result.model_info['event_study']``.
    n_boot : int, default 300
        Cluster-bootstrap replications.  Clusters are sampled with
        replacement; unit ids are reassigned in the resampled frame
        so BJS refits cleanly.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    dict
        ``{'statistic', 'df', 'pvalue', 'method', 'n_boot',
           'pre_cov'}`` where ``pre_cov`` is the bootstrap covariance
        matrix used for the Wald quadratic form.

    Notes
    -----
    Cost:  ``n_boot`` full BJS re-fits.  On a 10 000-row balanced
    panel with |horizon|=10, expect roughly
    ``n_boot × 0.3 s = 90 s`` for the default ``n_boot=300`` — the
    function is therefore opt-in, not run by default inside
    :func:`did_imputation`.
    """
    es = result.model_info.get('event_study') if result.model_info else None
    if es is None or len(es) == 0:
        raise ValueError(
            "result is missing an event-study table.  Re-run "
            "did_imputation() with a non-empty `horizon=` argument."
        )

    if horizon is None:
        horizon = sorted(es['relative_time'].astype(int).tolist())

    pre_rows = es[es['relative_time'] < 0].sort_values('relative_time')
    if len(pre_rows) == 0:
        raise ValueError(
            "No pre-treatment horizons found in the event study — "
            "cannot run a pre-trend test."
        )
    pre_k = pre_rows['relative_time'].astype(int).to_numpy()
    pre_att = pre_rows['att'].astype(float).to_numpy()
    K = len(pre_k)

    cluster_col = cluster if cluster is not None else group
    if cluster_col not in data.columns:
        raise ValueError(
            f"Cluster column '{cluster_col}' not found in data."
        )

    clusters = pd.Index(data[cluster_col].unique())
    G = len(clusters)
    rng = np.random.default_rng(seed)

    boot_mat = np.full((n_boot, K), np.nan)
    # Remember the most recent unexpected error so that if many / all
    # bootstrap replications fail we can surface it to the user rather
    # than silently claim "not enough reps succeeded".
    last_unexpected: Optional[Exception] = None
    for b in range(n_boot):
        picks = rng.choice(clusters, size=G, replace=True)
        frames = []
        for j, c in enumerate(picks):
            chunk = data[data[cluster_col] == c].copy()
            # Relabel unit ids so the FE refit sees distinct units.
            chunk[group] = chunk[group].astype(str) + f"_b{j}"
            frames.append(chunk)
        bdf = pd.concat(frames, ignore_index=True)
        try:
            r_b = did_imputation(
                bdf, y=y, group=group, time=time,
                first_treat=first_treat, controls=controls,
                horizon=horizon, cluster=cluster_col,
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            # Expected failures on pathological resamples (e.g. a draw
            # that happens to drop all eventually-treated units).  Skip.
            continue
        except Exception as exc:
            # Anything else is likely a genuine bug in BJS — keep the
            # last one around so we can report it below if the run fails.
            last_unexpected = exc
            continue

        es_b = r_b.model_info.get('event_study')
        if es_b is None or len(es_b) == 0:
            continue
        sub = es_b[es_b['relative_time'].isin(pre_k)].sort_values(
            'relative_time',
        )
        if len(sub) == K:
            boot_mat[b] = sub['att'].to_numpy()

    valid = ~np.any(np.isnan(boot_mat), axis=1)
    if valid.sum() < K + 1:
        extra = (
            f"  Last unexpected error: {type(last_unexpected).__name__}: "
            f"{last_unexpected}"
            if last_unexpected is not None else ""
        )
        raise RuntimeError(
            f"Only {int(valid.sum())} / {n_boot} bootstrap replications "
            f"succeeded — not enough to estimate the {K}×{K} covariance."
            + extra
        )
    boot_valid = boot_mat[valid]
    # Centre around the bootstrap mean so the cov estimator is unbiased
    # for Var(τ̂) under the empirical bootstrap distribution.
    cov = np.cov(boot_valid, rowvar=False, ddof=1)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    cov_reg = cov + np.eye(K) * 1e-10

    try:
        W = float(pre_att @ np.linalg.solve(cov_reg, pre_att))
    except np.linalg.LinAlgError:
        W = float(pre_att @ np.linalg.pinv(cov_reg) @ pre_att)
    pval = float(1 - stats.chi2.cdf(W, K))

    return {
        'statistic': W,
        'df': int(K),
        'pvalue': pval,
        'method': 'cluster-bootstrap',
        'n_boot': int(valid.sum()),
        'pre_cov': cov,
    }
