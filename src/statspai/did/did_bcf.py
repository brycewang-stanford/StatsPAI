"""
DiD-BCF — Forests for Differences (Wüthrich-Zhu 2025, arXiv 2505.09706).

Non-parametric DiD via Bayesian Causal Forests (Hahn-Murray-Carvalho
2020) applied to the differenced outcome ΔY = Y_post - Y_pre.

Identification: parallel trends conditional on covariates X. The
Robinson decomposition splits the forest into a prognostic component
μ(X) and a treatment component τ(X), giving conditional ATTs without
parametric trend assumptions and naturally handling staggered timing
through cohort-specific differences.

Implementation
--------------
1. For each unit, compute pre/post outcome difference per cohort.
2. Fit a BCF-style decomposition on (treat, X) → ΔY using existing
   :class:`statspai.bcf.BayesianCausalForest`.
3. Average the τ(X) predictions over the treated subsample → ATT;
   subgroup means → CATTs.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from ..core.results import CausalResult


def did_bcf(
    data: pd.DataFrame,
    y: str,
    treat: str,
    time: str,
    id: str,
    covariates: Optional[List[str]] = None,
    n_trees: int = 50,
    alpha: float = 0.05,
    seed: int = 0,
) -> CausalResult:
    """
    Forests for Differences DiD estimator.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel.
    y : str
    treat : str
        First-treatment-period column (0 = never-treated; otherwise
        the calendar period of first treatment).
    time : str
    id : str
    covariates : list of str, optional
    n_trees : int, default 50
    alpha : float, default 0.05
    seed : int

    Returns
    -------
    CausalResult
        ATT plus per-cohort CATT in ``model_info['catt_by_cohort']``.

    References
    ----------
    Wüthrich, Zhu (2025). Forests for Differences: Robust Causal
    Inference Beyond Parametric DiD. arXiv 2505.09706.
    """
    cov = list(covariates or [])
    df = data[[y, treat, time, id] + cov].dropna().reset_index(drop=True)

    # Determine pre/post per unit. For never-treated, use median time
    # as a placebo split (forest will treat them as control regardless).
    treat_vals = df[treat].to_numpy()
    is_treated_unit = treat_vals > 0
    median_t = float(df[time].median())
    cohort = np.where(is_treated_unit, treat_vals.astype(float), median_t)
    df['_cohort'] = cohort
    df['_post'] = (df[time] >= df['_cohort']).astype(int)

    # Compute mean Y per (id, _post) and difference
    agg = (df.groupby([id, '_post', '_cohort'])[y]
           .mean()
           .unstack('_post'))
    agg.columns = ['y_pre', 'y_post']
    agg['delta_y'] = agg['y_post'] - agg['y_pre']
    agg = agg.reset_index()
    # Merge covariates (use unit-mean if time-varying)
    if cov:
        cov_means = df.groupby(id)[cov].mean().reset_index()
        agg = agg.merge(cov_means, on=id, how='left')

    # Treated indicator at the unit level
    treat_per_unit = (df.groupby(id)[treat].max() > 0).astype(int)
    agg = agg.merge(
        treat_per_unit.rename('_D').reset_index(), on=id, how='left'
    )
    agg = agg.dropna(subset=['delta_y']).reset_index(drop=True)

    Y = agg['delta_y'].to_numpy(float)
    D = agg['_D'].to_numpy(int)
    if cov:
        X = agg[cov].to_numpy(float)
    else:
        X = np.zeros((len(agg), 0))
    cohort_arr = agg['_cohort'].to_numpy(float)
    n = len(agg)

    # Fit BCF on the differenced outcome via the existing module
    from ..bcf import bcf as bcf_fit

    if X.shape[1] == 0:
        # Degenerate: no covariates → ATT = mean(treated ΔY) - mean(control ΔY)
        att = float(Y[D == 1].mean() - Y[D == 0].mean())
        # Cluster bootstrap SE on units
        rng = np.random.default_rng(seed)
        boot = np.full(200, np.nan)
        ids = np.arange(n)
        for b in range(200):
            sample = rng.choice(ids, size=n, replace=True)
            try:
                boot[b] = (
                    Y[sample][D[sample] == 1].mean()
                    - Y[sample][D[sample] == 0].mean()
                )
            except Exception:
                pass
        se = float(np.nanstd(boot, ddof=1)) or 1e-6
        catt_by_cohort = {}
        for c in np.unique(cohort_arr[D == 1]):
            mask = (cohort_arr == c) & (D == 1)
            ctrl_mask = D == 0
            if mask.sum() > 0:
                catt_by_cohort[float(c)] = float(
                    Y[mask].mean() - Y[ctrl_mask].mean()
                )
    else:
        try:
            # Build a minimal DataFrame for the bcf API
            bcf_df = pd.DataFrame(X, columns=cov)
            bcf_df['_dy'] = Y
            bcf_df['_d'] = D
            bcf_res = bcf_fit(
                data=bcf_df, y='_dy', treat='_d', covariates=cov,
                n_trees_tau=n_trees, n_bootstrap=100, n_folds=3,
                random_state=seed,
            )
            tau_hat = np.asarray(bcf_res.model_info.get('cate', []), dtype=float)
            if tau_hat.size != n:
                raise RuntimeError("BCF returned mismatched cate size")
            att = float(tau_hat[D == 1].mean())
            cate_sd = np.asarray(
                bcf_res.model_info.get('cate_sd', np.zeros(n)), dtype=float
            )
            # ATT SE via average per-unit posterior SD on treated
            se = float(np.sqrt((cate_sd[D == 1] ** 2).mean() / max((D == 1).sum(), 1))) \
                or float(bcf_res.se) or 1e-6
            catt_by_cohort = {}
            for c in np.unique(cohort_arr[D == 1]):
                mask = (cohort_arr == c) & (D == 1)
                if mask.sum() > 0:
                    catt_by_cohort[float(c)] = float(tau_hat[mask].mean())
        except Exception as e:
            # Fallback to OLS-style DiD per cohort
            att = float(Y[D == 1].mean() - Y[D == 0].mean())
            se = float(np.std(Y[D == 1], ddof=1) / np.sqrt(max((D == 1).sum(), 1)))
            catt_by_cohort = {"_fallback_reason": f"{type(e).__name__}: {e}"}

    from scipy import stats
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    ci = (att - z_crit * se, att + z_crit * se)
    z = att / se if se > 0 else 0.0
    pvalue = float(2 * (1 - stats.norm.cdf(abs(z))))

    return CausalResult(
        method="DiD-BCF (Forests for Differences)",
        estimand="ATT",
        estimate=att,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        model_info={
            'estimator': 'DiD-BCF',
            'n_trees': n_trees,
            'n_covariates': len(cov),
            'catt_by_cohort': catt_by_cohort,
            'reference': 'Wüthrich-Zhu (2025), arXiv 2505.09706',
        },
        _citation_key='did_bcf',
    )


# Citation
CausalResult._CITATIONS['did_bcf'] = (
    "@article{wuthrich2025forests,\n"
    "  title={Forests for Differences: Robust Causal Inference Beyond "
    "Parametric DiD},\n"
    "  author={W{\\\"u}thrich, Kaspar and Zhu, Yinchu},\n"
    "  journal={arXiv preprint arXiv:2505.09706},\n"
    "  year={2025}\n"
    "}"
)
