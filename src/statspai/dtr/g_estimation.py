"""
G-estimation for Dynamic Treatment Regimes.

Estimates the optimal treatment rule in a two-stage setting using
the structural nested mean model (SNMM):

    E[Y(a1, a2) - Y(0, a2) | H1] = psi1 * a1 * f1(H1)
    E[Y(a1, a2) - Y(a1, 0) | H2] = psi2 * a2 * f2(H2)

where H_t is the history up to stage t.

The algorithm works backwards from the last stage:
1. Stage 2: estimate psi2 by regressing Y - psi2*A2*f2(H2) on A2
   (find psi2 that makes residuals uncorrelated with A2 given H2).
2. Stage 1: using the "de-blipped" outcome Y_tilde = Y - psi2*A2*f2(H2),
   estimate psi1 similarly.

References
----------
Robins, J. M. (2004). "Optimal Structural Nested Models."
Murphy, S. A. (2003). "Optimal Dynamic Treatment Regimes." [@robins2004optimal]
"""

from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.linear_model import LinearRegression

from ..core.results import CausalResult


def g_estimation(
    data: pd.DataFrame,
    y: str,
    treatments: List[str],
    covariates_by_stage: List[List[str]],
    propensity_covariates: Optional[List[List[str]]] = None,
    alpha: float = 0.05,
    n_bootstrap: int = 500,
    random_state: int = 42,
) -> CausalResult:
    """
    G-estimation for a multi-stage dynamic treatment regime.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    y : str
        Final outcome variable.
    treatments : list of str
        Treatment variables at each stage, in temporal order.
        E.g., ['A1', 'A2'] for a two-stage DTR.
    covariates_by_stage : list of list of str
        Covariates (tailoring variables) available at each stage.
        covariates_by_stage[k] are the variables available when
        deciding treatment k.
    propensity_covariates : list of list of str, optional
        Covariates for propensity model at each stage.
        If None, uses covariates_by_stage.
    alpha : float, default 0.05
    n_bootstrap : int, default 500
    random_state : int, default 42

    Returns
    -------
    CausalResult
        detail has stage-level blip function estimates.
        model_info contains optimal treatment rules.

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.g_estimation(
    ...     df, y='outcome',
    ...     treatments=['treatment_stage1', 'treatment_stage2'],
    ...     covariates_by_stage=[['x1', 'x2'], ['x1', 'x2', 'x3']])
    >>> print(result.summary())
    """
    est = GEstimation(
        data=data, y=y, treatments=treatments,
        covariates_by_stage=covariates_by_stage,
        propensity_covariates=propensity_covariates,
        alpha=alpha, n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    return est.fit()


class GEstimation:
    """
    G-estimation for dynamic treatment regimes via SNMM.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
    treatments : list of str
    covariates_by_stage : list of list of str
    propensity_covariates : list of list of str, optional
    alpha : float
    n_bootstrap : int
    random_state : int
    """

    def __init__(
        self,
        data: pd.DataFrame,
        y: str,
        treatments: List[str],
        covariates_by_stage: List[List[str]],
        propensity_covariates: Optional[List[List[str]]] = None,
        alpha: float = 0.05,
        n_bootstrap: int = 500,
        random_state: int = 42,
    ):
        self.data = data
        self.y = y
        self.treatments = treatments
        self.covariates_by_stage = covariates_by_stage
        self.propensity_covariates = propensity_covariates or covariates_by_stage
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.n_stages = len(treatments)

        if len(covariates_by_stage) != self.n_stages:
            raise ValueError(
                f"covariates_by_stage has {len(covariates_by_stage)} entries "
                f"but {self.n_stages} treatments were specified"
            )

    def fit(self) -> CausalResult:
        """Run G-estimation backward induction."""
        all_cols = [self.y] + self.treatments
        for stage_covs in self.covariates_by_stage:
            all_cols.extend(stage_covs)
        all_cols = list(set(all_cols))

        missing = [c for c in all_cols if c not in self.data.columns]
        if missing:
            raise ValueError(f"Columns not found in data: {missing}")

        clean = self.data[all_cols].dropna()
        n = len(clean)

        Y = clean[self.y].values.astype(np.float64)
        A = [clean[t].values.astype(np.float64) for t in self.treatments]
        X_stages = [clean[covs].values.astype(np.float64)
                    for covs in self.covariates_by_stage]

        # Backward induction
        psi_estimates, optimal_rules = self._backward_induction(Y, A, X_stages, n)

        # Bootstrap
        rng = np.random.RandomState(self.random_state)
        boot_psis = np.zeros((self.n_bootstrap, self.n_stages))

        for b in range(self.n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            Y_b = Y[idx]
            A_b = [a[idx] for a in A]
            X_b = [x[idx] for x in X_stages]
            psis_b, _ = self._backward_induction(Y_b, A_b, X_b, n)
            boot_psis[b] = psis_b

        se_psis = np.std(boot_psis, axis=0, ddof=1)

        # Overall value: sum of stage effects (average blip)
        total_value = sum(psi_estimates)
        se_total = float(np.std(np.sum(boot_psis, axis=1), ddof=1))

        z_crit = sp_stats.norm.ppf(1 - self.alpha / 2)

        if se_total > 0:
            pvalue = float(2 * (1 - sp_stats.norm.cdf(abs(total_value / se_total))))
        else:
            pvalue = 0.0

        ci = (total_value - z_crit * se_total, total_value + z_crit * se_total)

        detail_rows = []
        for k in range(self.n_stages):
            detail_rows.append({
                'stage': k + 1,
                'treatment': self.treatments[k],
                'blip_estimate': psi_estimates[k],
                'se': se_psis[k],
                'optimal_rule': optimal_rules[k],
            })
        detail = pd.DataFrame(detail_rows)

        model_info = {
            'n_stages': self.n_stages,
            'psi_estimates': list(psi_estimates),
            'optimal_rules': optimal_rules,
            'total_value_optimal': float(total_value),
        }

        return CausalResult(
            method='G-Estimation (Robins 2004)',
            estimand='Optimal DTR Value',
            estimate=float(total_value),
            se=se_total,
            pvalue=pvalue,
            ci=ci,
            alpha=self.alpha,
            n_obs=n,
            detail=detail,
            model_info=model_info,
            _citation_key='g_estimation',
        )

    def _backward_induction(self, Y, A, X_stages, n):
        """Backward induction for G-estimation."""
        psi_estimates = np.zeros(self.n_stages)
        optimal_rules = []
        Y_tilde = Y.copy()

        for k in range(self.n_stages - 1, -1, -1):
            A_k = A[k]
            X_k = X_stages[k]

            # Simple blip model: gamma(H_k, a_k) = psi_k * a_k
            # G-estimation: find psi_k such that Y_tilde - psi_k * A_k
            # is uncorrelated with A_k given X_k

            # Residualise both Y_tilde and A_k on X_k
            lr_y = LinearRegression()
            lr_y.fit(X_k, Y_tilde)
            Y_res = Y_tilde - lr_y.predict(X_k)

            lr_a = LinearRegression()
            lr_a.fit(X_k, A_k)
            A_res = A_k - lr_a.predict(X_k)

            # psi_k = Cov(Y_res, A_res) / Var(A_res)
            denom = np.sum(A_res ** 2)
            if denom > 1e-10:
                psi_k = float(np.sum(Y_res * A_res) / denom)
            else:
                psi_k = 0.0

            psi_estimates[k] = psi_k

            # Optimal rule: treat if blip > 0
            # For simple model: treat if psi_k > 0
            if psi_k > 0:
                optimal_rules.append("Always treat")
            elif psi_k < 0:
                optimal_rules.append("Never treat")
            else:
                optimal_rules.append("Indifferent")

            # De-blip: remove stage-k treatment effect
            Y_tilde = Y_tilde - psi_k * A_k

        optimal_rules.reverse()  # Back to forward order
        return psi_estimates, optimal_rules


CausalResult._CITATIONS['g_estimation'] = (
    "@incollection{robins2004optimal,\n"
    "  title={Optimal Structural Nested Models for Optimal Sequential "
    "Decisions},\n"
    "  author={Robins, James M},\n"
    "  booktitle={Proceedings of the Second Seattle Symposium in "
    "Biostatistics},\n"
    "  pages={189--326},\n"
    "  year={2004},\n"
    "  publisher={Springer}\n"
    "}"
)
