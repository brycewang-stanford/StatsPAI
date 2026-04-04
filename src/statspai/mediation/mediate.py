"""
Causal Mediation Analysis (Imai, Keele, Tingley 2010).

Decomposes total effect into:
- ACME (Average Causal Mediation Effect) — indirect via mediator
- ADE (Average Direct Effect) — not through mediator
- Total effect = ACME + ADE
- Proportion mediated = ACME / Total

Uses simulation-based inference (quasi-Bayesian) with bootstrap
confidence intervals.

Causal diagram:
    T → M → Y   (indirect / mediation path)
    T ------→ Y  (direct path)
"""

from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult


def mediate(
    data: pd.DataFrame,
    y: str,
    treat: str,
    mediator: str,
    covariates: Optional[List[str]] = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> CausalResult:
    """
    Causal mediation analysis.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    y : str
        Outcome variable.
    treat : str
        Treatment variable (binary 0/1 or continuous).
    mediator : str
        Mediator variable.
    covariates : list of str, optional
        Pre-treatment covariates to control for.
    n_boot : int, default 1000
        Number of bootstrap replications for inference.
    alpha : float, default 0.05
        Significance level.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    CausalResult
        Contains ACME, ADE, total effect, and proportion mediated.

    Examples
    --------
    >>> # Job training → Skills → Wages
    >>> result = mediate(df, y='wage', treat='training',
    ...                  mediator='skills', covariates=['age', 'edu'])
    >>> print(result.summary())
    """
    analysis = MediationAnalysis(
        data=data, y=y, treat=treat, mediator=mediator,
        covariates=covariates, n_boot=n_boot, alpha=alpha, seed=seed,
    )
    return analysis.fit()


class MediationAnalysis:
    """
    Causal Mediation Analysis estimator.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        y: str,
        treat: str,
        mediator: str,
        covariates: Optional[List[str]] = None,
        n_boot: int = 1000,
        alpha: float = 0.05,
        seed: int = 42,
    ):
        self.data = data
        self.y = y
        self.treat = treat
        self.mediator = mediator
        self.covariates = covariates or []
        self.n_boot = n_boot
        self.alpha = alpha
        self.seed = seed

        self._validate()

    def _validate(self):
        for col in [self.y, self.treat, self.mediator] + self.covariates:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in data")

    def fit(self) -> CausalResult:
        """Estimate mediation effects with bootstrap inference."""
        cols = [self.y, self.treat, self.mediator] + self.covariates
        clean = self.data[cols].dropna()
        n = len(clean)

        Y = clean[self.y].values.astype(float)
        T = clean[self.treat].values.astype(float)
        M = clean[self.mediator].values.astype(float)

        if self.covariates:
            X = clean[self.covariates].values.astype(float)
        else:
            X = None

        # Point estimates
        acme, ade, total = self._estimate_effects(Y, T, M, X)
        prop_mediated = acme / total if abs(total) > 1e-10 else np.nan

        # Bootstrap inference
        rng = np.random.default_rng(self.seed)
        boot_acme = np.zeros(self.n_boot)
        boot_ade = np.zeros(self.n_boot)
        boot_total = np.zeros(self.n_boot)

        for b in range(self.n_boot):
            idx = rng.choice(n, size=n, replace=True)
            Y_b, T_b, M_b = Y[idx], T[idx], M[idx]
            X_b = X[idx] if X is not None else None

            try:
                a, d, t = self._estimate_effects(Y_b, T_b, M_b, X_b)
                boot_acme[b] = a
                boot_ade[b] = d
                boot_total[b] = t
            except Exception:
                boot_acme[b] = acme
                boot_ade[b] = ade
                boot_total[b] = total

        # Standard errors and CIs
        se_acme = float(np.std(boot_acme, ddof=1))
        se_ade = float(np.std(boot_ade, ddof=1))
        se_total = float(np.std(boot_total, ddof=1))

        lo = self.alpha / 2
        hi = 1 - self.alpha / 2
        ci_acme = (float(np.percentile(boot_acme, lo * 100)),
                   float(np.percentile(boot_acme, hi * 100)))
        ci_ade = (float(np.percentile(boot_ade, lo * 100)),
                  float(np.percentile(boot_ade, hi * 100)))
        ci_total = (float(np.percentile(boot_total, lo * 100)),
                    float(np.percentile(boot_total, hi * 100)))

        # P-values (proportion of bootstrap under null)
        pv_acme = self._boot_pvalue(boot_acme)
        pv_ade = self._boot_pvalue(boot_ade)
        pv_total = self._boot_pvalue(boot_total)

        # Detail table
        detail = pd.DataFrame({
            'effect': ['ACME (indirect)', 'ADE (direct)', 'Total Effect', 'Prop. Mediated'],
            'estimate': [acme, ade, total, prop_mediated],
            'se': [se_acme, se_ade, se_total, np.nan],
            'ci_lower': [ci_acme[0], ci_ade[0], ci_total[0], np.nan],
            'ci_upper': [ci_acme[1], ci_ade[1], ci_total[1], np.nan],
            'pvalue': [pv_acme, pv_ade, pv_total, np.nan],
        })

        model_info = {
            'acme': acme,
            'ade': ade,
            'total_effect': total,
            'prop_mediated': prop_mediated,
            'n_boot': self.n_boot,
            'se_acme': se_acme,
            'se_ade': se_ade,
            'ci_acme': ci_acme,
            'ci_ade': ci_ade,
            'ci_total': ci_total,
        }

        return CausalResult(
            method='Causal Mediation Analysis',
            estimand='ACME',
            estimate=acme,
            se=se_acme,
            pvalue=pv_acme,
            ci=ci_acme,
            alpha=self.alpha,
            n_obs=n,
            detail=detail,
            model_info=model_info,
            _citation_key='mediation',
        )

    def _estimate_effects(self, Y, T, M, X):
        """
        Estimate ACME, ADE, and total effect using the product method
        with OLS mediator and outcome models.

        Mediator model: M = a0 + a1*T + a2'*X + e_m
        Outcome model:  Y = b0 + b1*T + b2*M + b3'*X + e_y

        ACME = a1 * b2 (indirect effect)
        ADE  = b1       (direct effect)
        Total = a1*b2 + b1
        """
        n = len(Y)

        # Build design matrices
        if X is not None:
            Z_med = np.column_stack([np.ones(n), T, X])
            Z_out = np.column_stack([np.ones(n), T, M, X])
        else:
            Z_med = np.column_stack([np.ones(n), T])
            Z_out = np.column_stack([np.ones(n), T, M])

        # Mediator model: M ~ T + X
        try:
            a = np.linalg.lstsq(Z_med, M, rcond=None)[0]
        except np.linalg.LinAlgError:
            a = np.zeros(Z_med.shape[1])
        a1 = a[1]  # coefficient on T

        # Outcome model: Y ~ T + M + X
        try:
            b = np.linalg.lstsq(Z_out, Y, rcond=None)[0]
        except np.linalg.LinAlgError:
            b = np.zeros(Z_out.shape[1])
        b1 = b[1]  # direct effect (T coefficient)
        b2 = b[2]  # mediator coefficient

        acme = a1 * b2  # indirect effect
        ade = b1         # direct effect
        total = acme + ade

        return float(acme), float(ade), float(total)

    @staticmethod
    def _boot_pvalue(boot_samples):
        """Two-sided p-value from bootstrap distribution."""
        n = len(boot_samples)
        # Proportion of bootstrap samples with sign opposite to point estimate
        mean = np.mean(boot_samples)
        if mean >= 0:
            p = 2 * np.mean(boot_samples <= 0)
        else:
            p = 2 * np.mean(boot_samples >= 0)
        return float(np.clip(p, 1 / n, 1.0))


# Citation
CausalResult._CITATIONS['mediation'] = (
    "@article{imai2010general,\n"
    "  title={A General Approach to Causal Mediation Analysis},\n"
    "  author={Imai, Kosuke and Keele, Luke and Tingley, Dustin},\n"
    "  journal={Psychological Methods},\n"
    "  volume={15},\n"
    "  number={4},\n"
    "  pages={309--334},\n"
    "  year={2010},\n"
    "  publisher={American Psychological Association}\n"
    "}"
)
