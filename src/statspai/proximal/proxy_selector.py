"""
Demystifying PCI: proxy selection guide (Ringlein et al. 2026,
arXiv 2512.24413).

PCI requires two proxy variables Z, W satisfying specific
independence assumptions involving the unmeasured confounder U.
Picking valid proxies is the practical bottleneck. This module
provides a heuristic selector that scores candidate proxies on:

1. **Correlation with treatment** (proxy_z should correlate with D
   marginally — strong signal).
2. **Correlation with outcome** (proxy_w should correlate with Y
   marginally — strong signal).
3. **Conditional independence** (Z ⊥ Y | D, X and W ⊥ D | X) tested
   via partial-correlation Fisher z-test.

Returns ranked candidates so the user can pick the strongest pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ProxyScoreResult:
    """Per-candidate proxy score for PCI."""
    z_candidates: pd.DataFrame   # cols: name, score_z, p_indep
    w_candidates: pd.DataFrame   # cols: name, score_w, p_indep
    recommended_z: List[str]
    recommended_w: List[str]

    def summary(self) -> str:
        rows = [
            "PCI Proxy Selector",
            "=" * 42,
            "Z (treatment-side) candidates:",
            self.z_candidates.to_string(index=False),
            "",
            "W (outcome-side) candidates:",
            self.w_candidates.to_string(index=False),
            "",
            f"Recommended Z: {self.recommended_z}",
            f"Recommended W: {self.recommended_w}",
        ]
        return "\n".join(rows)


def select_pci_proxies(
    data: pd.DataFrame,
    y: str,
    treat: str,
    candidates: List[str],
    covariates: Optional[List[str]] = None,
    top_k: int = 2,
) -> ProxyScoreResult:
    """
    Score and rank candidate proxies for PCI.

    Parameters
    ----------
    data : pd.DataFrame
    y, treat : str
    candidates : list of str
        All variables that could plausibly serve as proxies.
    covariates : list of str, optional
    top_k : int, default 2
        Number of top candidates to recommend per side.

    Returns
    -------
    ProxyScoreResult
    """
    cov = list(covariates or [])
    df = data[[y, treat] + list(candidates) + cov].dropna()
    Y = df[y].to_numpy(float)
    D = df[treat].to_numpy(float)

    z_rows = []
    w_rows = []
    for c in candidates:
        v = df[c].to_numpy(float)
        # Marginal correlations
        rho_d = float(np.corrcoef(v, D)[0, 1]) if v.std() > 0 else 0.0
        rho_y = float(np.corrcoef(v, Y)[0, 1]) if v.std() > 0 else 0.0
        # Partial correlation for conditional independence test
        # (Z ⊥ Y | D, X) — score_z = |rho_d| - |rho_partial(Z, Y | D, X)|
        controls = np.column_stack([D] + ([df[c2].to_numpy(float)
                                            for c2 in cov] if cov else []))
        try:
            # Residualise v and Y on controls
            from sklearn.linear_model import LinearRegression
            r_v = v - LinearRegression().fit(controls, v).predict(controls)
            r_y = Y - LinearRegression().fit(controls, Y).predict(controls)
            partial_zy = float(np.corrcoef(r_v, r_y)[0, 1]) if r_v.std() > 0 else 0.0
            # Fisher z-test on partial correlation
            n = len(df) - controls.shape[1] - 2
            if n > 5:
                z_stat = 0.5 * np.log((1 + partial_zy) / (1 - partial_zy)) * np.sqrt(n)
                p_indep = float(2 * (1 - stats.norm.cdf(abs(z_stat))))
            else:
                p_indep = 1.0
        except Exception:
            partial_zy = rho_y
            p_indep = 1.0

        # Score for Z: strong correlation with D, weak partial corr with Y
        score_z = abs(rho_d) - abs(partial_zy)
        # Score for W: strong correlation with Y, weak partial corr with D
        # (compute symmetric quantity)
        try:
            r_v_w = v - LinearRegression().fit(
                np.column_stack([df[c2].to_numpy(float) for c2 in cov]) if cov
                else np.zeros((len(df), 1)), v
            ).predict(np.column_stack([df[c2].to_numpy(float) for c2 in cov])
                       if cov else np.zeros((len(df), 1)))
            r_d = D - LinearRegression().fit(
                np.column_stack([df[c2].to_numpy(float) for c2 in cov]) if cov
                else np.zeros((len(df), 1)), D
            ).predict(np.column_stack([df[c2].to_numpy(float) for c2 in cov])
                       if cov else np.zeros((len(df), 1)))
            partial_wd = float(np.corrcoef(r_v_w, r_d)[0, 1]) if r_v_w.std() > 0 else 0.0
        except Exception:
            partial_wd = rho_d
        score_w = abs(rho_y) - abs(partial_wd)

        z_rows.append({'name': c, 'score_z': score_z, 'p_indep': p_indep})
        w_rows.append({'name': c, 'score_w': score_w, 'p_indep': p_indep})

    z_df = pd.DataFrame(z_rows).sort_values('score_z', ascending=False) \
        .reset_index(drop=True)
    w_df = pd.DataFrame(w_rows).sort_values('score_w', ascending=False) \
        .reset_index(drop=True)
    rec_z = z_df.head(top_k)['name'].tolist()
    rec_w = w_df.head(top_k)['name'].tolist()
    # Avoid recommending same variable for both roles
    rec_w = [c for c in rec_w if c not in rec_z][:top_k]

    return ProxyScoreResult(
        z_candidates=z_df,
        w_candidates=w_df,
        recommended_z=rec_z,
        recommended_w=rec_w,
    )
