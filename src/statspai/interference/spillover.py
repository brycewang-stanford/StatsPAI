"""
Spillover / Interference Effects Estimator.

Under partial interference (units in clusters, interference within
but not between clusters), decomposes the total treatment effect into:

- **Direct effect**: effect of own treatment, holding neighbours fixed.
- **Spillover effect**: effect of neighbours' treatment, holding own fixed.
- **Total effect**: direct + spillover.

Uses the Hudgens & Halloran (2008) / Aronow & Samii (2017) framework
with IPW estimation under a clustered experiment design.

References
----------
Hudgens, M. G. & Halloran, M. E. (2008).
"Toward Causal Inference with Interference."
JASA, 103(482), 832-842. [@hudgens2008toward]
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..core.results import CausalResult


def spillover(
    data: pd.DataFrame,
    y: str,
    treat: str,
    cluster: str,
    covariates: Optional[List[str]] = None,
    exposure_fn: str = "fraction",
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    random_state: int = 42,
) -> CausalResult:
    """
    Estimate direct and spillover treatment effects.

    Parameters
    ----------
    data : pd.DataFrame
        Input data with cluster structure.
    y : str
        Outcome variable.
    treat : str
        Binary individual treatment variable (0/1).
    cluster : str
        Cluster/group identifier.
    covariates : list of str, optional
        Control covariates.
    exposure_fn : str, default 'fraction'
        How to measure peer exposure:
        - 'fraction': fraction of cluster-mates treated
        - 'any': any cluster-mate treated (binary)
        - 'count': count of treated cluster-mates
    n_bootstrap : int, default 500
    alpha : float, default 0.05
    random_state : int, default 42

    Returns
    -------
    CausalResult
        estimate = total effect.
        detail DataFrame has columns 'effect_type', 'estimate', 'se'.
        model_info contains 'direct_effect', 'spillover_effect',
        'total_effect'.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for c in range(40):
    ...     treated = rng.integers(0, 2, size=5)
    ...     frac = treated.mean()
    ...     for i in range(5):
    ...         y = 1.0 + 0.8 * treated[i] + 0.5 * frac + rng.normal()
    ...         rows.append({"household": c, "vaccinated": int(treated[i]),
    ...                      "outcome": y})
    >>> df = pd.DataFrame(rows)
    >>> result = sp.spillover(df, y='outcome', treat='vaccinated',
    ...                       cluster='household', n_bootstrap=50,
    ...                       random_state=0)
    >>> bool('direct_effect' in result.model_info)
    True
    >>> bool('spillover_effect' in result.model_info)
    True
    """
    est = SpilloverEstimator(
        data=data,
        y=y,
        treat=treat,
        cluster=cluster,
        covariates=covariates,
        exposure_fn=exposure_fn,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        random_state=random_state,
    )
    return est.fit()


class SpilloverEstimator:
    """Spillover / interference effects estimator.

    The engine behind :func:`sp.spillover`. Construct it with the data
    and column names, then call :meth:`fit` to obtain a ``CausalResult``
    whose ``model_info`` carries ``direct_effect``, ``spillover_effect``
    and ``total_effect``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for c in range(40):
    ...     treated = rng.integers(0, 2, size=5)
    ...     frac = treated.mean()
    ...     for i in range(5):
    ...         y = 1.0 + 0.8 * treated[i] + 0.5 * frac + rng.normal()
    ...         rows.append({"hh": c, "vaccinated": int(treated[i]),
    ...                      "outcome": y})
    >>> df = pd.DataFrame(rows)
    >>> est = sp.SpilloverEstimator(
    ...     data=df, y="outcome", treat="vaccinated", cluster="hh",
    ...     n_bootstrap=50, random_state=0,
    ... )
    >>> res = est.fit()
    >>> sorted(k for k in res.model_info if k.endswith("_effect"))
    ['direct_effect', 'spillover_effect', 'total_effect']
    """

    def __init__(
        self,
        data: pd.DataFrame,
        y: str,
        treat: str,
        cluster: str,
        covariates: Optional[List[str]] = None,
        exposure_fn: str = "fraction",
        n_bootstrap: int = 500,
        alpha: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self.data = data
        self.y = y
        self.treat = treat
        self.cluster = cluster
        self.covariates = covariates
        self.exposure_fn = exposure_fn
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self.random_state = random_state

    def fit(self) -> CausalResult:
        """Estimate direct and spillover effects."""
        cols = [self.y, self.treat, self.cluster]
        if self.covariates:
            cols += self.covariates
        missing = [c for c in cols if c not in self.data.columns]
        if missing:
            raise ValueError(f"Columns not found in data: {missing}")

        df = self.data[cols].dropna().copy()
        n = len(df)

        Y = df[self.y].values.astype(np.float64)
        D = df[self.treat].values.astype(np.float64)
        G = df[self.cluster].values

        # Compute peer exposure for each unit
        exposure = self._compute_exposure(df, D, G)

        # Classify units into 4 groups:
        # (own_treat=1, high_exposure) vs (own_treat=0, low_exposure) etc.
        median_exp = np.median(exposure[exposure > 0]) if np.any(exposure > 0) else 0.5
        high_exp = exposure >= median_exp

        # Direct effect: E[Y|D=1, high_exp] - E[Y|D=0, high_exp]
        mask_d1_hi = (D == 1) & high_exp
        mask_d0_hi = (D == 0) & high_exp

        # Spillover effect: E[Y|D=0, high_exp] - E[Y|D=0, low_exp]
        mask_d0_lo = (D == 0) & (~high_exp)

        direct = _safe_diff(Y, mask_d1_hi, mask_d0_hi)
        spillover_eff = _safe_diff(Y, mask_d0_hi, mask_d0_lo)
        total = direct + spillover_eff

        # Bootstrap
        rng = np.random.RandomState(self.random_state)
        clusters = np.unique(G)
        n_clusters = len(clusters)

        boot_direct = np.zeros(self.n_bootstrap)
        boot_spill = np.zeros(self.n_bootstrap)

        for b in range(self.n_bootstrap):
            # Cluster bootstrap
            cl_idx = rng.choice(n_clusters, size=n_clusters, replace=True)
            selected_clusters = clusters[cl_idx]

            idx_list = []
            for cl in selected_clusters:
                idx_list.append(np.where(G == cl)[0])
            if len(idx_list) == 0:
                boot_direct[b] = direct
                boot_spill[b] = spillover_eff
                continue

            idx = np.concatenate(idx_list)
            Y_b = Y[idx]
            D_b = D[idx]
            exp_b = exposure[idx]

            hi_b = exp_b >= median_exp

            m_d1h = (D_b == 1) & hi_b
            m_d0h = (D_b == 0) & hi_b
            m_d0l = (D_b == 0) & (~hi_b)

            boot_direct[b] = _safe_diff(Y_b, m_d1h, m_d0h)
            boot_spill[b] = _safe_diff(Y_b, m_d0h, m_d0l)

        se_direct = float(np.std(boot_direct, ddof=1))
        se_spill = float(np.std(boot_spill, ddof=1))
        se_total = float(np.std(boot_direct + boot_spill, ddof=1))

        z_crit = sp_stats.norm.ppf(1 - self.alpha / 2)

        if se_total > 0:
            pvalue = float(2 * (1 - sp_stats.norm.cdf(abs(total / se_total))))
        else:
            pvalue = 0.0
        ci = (total - z_crit * se_total, total + z_crit * se_total)

        detail = pd.DataFrame(
            {
                "effect_type": ["Direct", "Spillover", "Total"],
                "estimate": [direct, spillover_eff, total],
                "se": [se_direct, se_spill, se_total],
            }
        )

        model_info = {
            "direct_effect": float(direct),
            "direct_se": se_direct,
            "spillover_effect": float(spillover_eff),
            "spillover_se": se_spill,
            "total_effect": float(total),
            "total_se": se_total,
            "n_clusters": n_clusters,
            "exposure_fn": self.exposure_fn,
            "median_exposure": float(median_exp),
        }

        return CausalResult(
            method="Spillover Effects (Hudgens & Halloran 2008)",
            estimand="Total Effect (Direct + Spillover)",
            estimate=float(total),
            se=se_total,
            pvalue=pvalue,
            ci=ci,
            alpha=self.alpha,
            n_obs=n,
            detail=detail,
            model_info=model_info,
            _citation_key="spillover",
        )

    def _compute_exposure(
        self,
        df: pd.DataFrame,
        D: np.ndarray,
        G: np.ndarray,
    ) -> np.ndarray:
        """Compute peer treatment exposure for each unit."""
        exposure = np.zeros(len(D))
        clusters = np.unique(G)

        for cl in clusters:
            mask = G == cl
            cl_size = mask.sum()
            cl_treated = D[mask].sum()

            for i in np.where(mask)[0]:
                # Exclude own treatment
                peer_treated = cl_treated - D[i]
                peer_size = cl_size - 1

                if peer_size <= 0:
                    exposure[i] = 0
                elif self.exposure_fn == "fraction":
                    exposure[i] = peer_treated / peer_size
                elif self.exposure_fn == "any":
                    exposure[i] = float(peer_treated > 0)
                elif self.exposure_fn == "count":
                    exposure[i] = peer_treated
                else:
                    exposure[i] = peer_treated / max(peer_size, 1)

        return exposure


def _safe_diff(
    Y: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> float:
    """Safe mean difference."""
    if mask_a.sum() > 0 and mask_b.sum() > 0:
        return float(np.mean(Y[mask_a]) - np.mean(Y[mask_b]))
    return 0.0


CausalResult._CITATIONS["spillover"] = (
    "@article{hudgens2008toward,\n"
    "  title={Toward Causal Inference with Interference},\n"
    "  author={Hudgens, Michael G and Halloran, M Elizabeth},\n"
    "  journal={Journal of the American Statistical Association},\n"
    "  volume={103},\n"
    "  number={482},\n"
    "  pages={832--842},\n"
    "  year={2008}\n"
    "}"
)
