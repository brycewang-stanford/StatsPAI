"""
Generalized Propensity Score (GPS) for continuous treatment effects.

Estimates the dose-response function E[Y(t)] and the average
dose-response derivative (marginal effect) for continuous treatments.

Approach:
1. Estimate GPS: f(T | X) using a normal density model.
   GPS_i = phi((T_i - m(X_i)) / sigma)
   where m(X) = E[T|X] is estimated by regression.

2. Estimate conditional outcome: E[Y | T, GPS] via flexible regression
   (GBM or kernel smoothing).

3. Average over GPS at each dose level to get E[Y(t)]:
   E[Y(t)] = E_X[ E[Y | T=t, GPS(t, X)] ]

References
----------
Hirano, K. & Imbens, G. W. (2004).
"The Propensity Score with Continuous Treatments." [@hirano2004propensity]

Kennedy, E. H., Ma, Z., McHugh, M. D., & Small, D. S. (2017).
"Non-parametric methods for doubly robust estimation of continuous
treatment effects." JRSS-B, 79(4), 1229-1245. [@kennedy2017parametric]
"""

from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# sklearn is imported lazily inside the methods that need it so that
# ``import statspai`` doesn't pull ~245 sklearn submodules through this
# file when the user never touches dose_response. ``BaseEstimator`` only
# appears in type annotations here and is gated behind ``TYPE_CHECKING``.
if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

from ..core.results import CausalResult


# ======================================================================
# Public API
# ======================================================================

def dose_response(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    n_dose_points: int = 20,
    dose_range: Optional[Tuple[float, float]] = None,
    treatment_model: 'Optional[BaseEstimator]' = None,
    outcome_model: 'Optional[BaseEstimator]' = None,
    n_bootstrap: int = 200,
    alpha: float = 0.05,
    random_state: int = 42,
) -> CausalResult:
    """
    Estimate the dose-response function for a continuous treatment.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    y : str
        Outcome variable.
    treat : str
        Continuous treatment variable.
    covariates : list of str
        Covariate names.
    n_dose_points : int, default 20
        Number of dose levels to evaluate.
    dose_range : tuple of (float, float), optional
        (min, max) dose range. If None, uses 5th-95th percentile.
    treatment_model : sklearn estimator, optional
        Model for E[T|X]. Default: GBM.
    outcome_model : sklearn estimator, optional
        Model for E[Y|T, GPS]. Default: GBM.
    n_bootstrap : int, default 200
        Bootstrap iterations for confidence bands.
    alpha : float, default 0.05
        Significance level.
    random_state : int, default 42

    Returns
    -------
    CausalResult
        detail contains the dose-response curve:
        columns 'dose', 'response', 'se', 'ci_lower', 'ci_upper'.
        model_info contains 'avg_marginal_effect' (derivative).

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.dose_response(df, y='outcome', treat='dosage',
    ...                           covariates=['age', 'weight'])
    >>> print(result.detail)  # dose-response curve
    """
    est = DoseResponse(
        data=data, y=y, treat=treat, covariates=covariates,
        n_dose_points=n_dose_points, dose_range=dose_range,
        treatment_model=treatment_model, outcome_model=outcome_model,
        n_bootstrap=n_bootstrap, alpha=alpha,
        random_state=random_state,
    )
    return est.fit()


# ======================================================================
# DoseResponse class
# ======================================================================

class DoseResponse:
    """
    Generalized Propensity Score dose-response estimator.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
    treat : str
    covariates : list of str
    n_dose_points : int
    dose_range : tuple, optional
    treatment_model : sklearn estimator, optional
    outcome_model : sklearn estimator, optional
    n_bootstrap : int
    alpha : float
    random_state : int
    """

    def __init__(
        self,
        data: pd.DataFrame,
        y: str,
        treat: str,
        covariates: List[str],
        n_dose_points: int = 20,
        dose_range: Optional[Tuple[float, float]] = None,
        treatment_model: 'Optional[BaseEstimator]' = None,
        outcome_model: 'Optional[BaseEstimator]' = None,
        n_bootstrap: int = 200,
        alpha: float = 0.05,
        random_state: int = 42,
    ):
        from sklearn.ensemble import GradientBoostingRegressor
        self.data = data
        self.y = y
        self.treat = treat
        self.covariates = covariates
        self.n_dose_points = n_dose_points
        self.dose_range = dose_range
        self.treatment_model = treatment_model or GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=random_state,
        )
        self.outcome_model = outcome_model or GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=random_state,
        )
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self.random_state = random_state

    def fit(self) -> CausalResult:
        """Estimate the dose-response function."""
        cols = [self.y, self.treat] + self.covariates
        missing = [c for c in cols if c not in self.data.columns]
        if missing:
            raise ValueError(f"Columns not found in data: {missing}")

        clean = self.data[cols].dropna()
        Y = clean[self.y].values.astype(np.float64)
        T = clean[self.treat].values.astype(np.float64)
        X = clean[self.covariates].values.astype(np.float64)
        n = len(Y)

        # Dose grid
        if self.dose_range is None:
            t_lo, t_hi = np.percentile(T, [5, 95])
        else:
            t_lo, t_hi = self.dose_range
        dose_grid = np.linspace(t_lo, t_hi, self.n_dose_points)

        # Estimate dose-response curve
        dr_curve = self._estimate_curve(Y, T, X, dose_grid, n)

        # Bootstrap for confidence bands
        rng = np.random.RandomState(self.random_state)
        boot_curves = np.zeros((self.n_bootstrap, self.n_dose_points))

        for b in range(self.n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            boot_curves[b] = self._estimate_curve(
                Y[idx], T[idx], X[idx], dose_grid, n
            )

        dr_se = np.std(boot_curves, axis=0, ddof=1)
        z_crit = sp_stats.norm.ppf(1 - self.alpha / 2)
        ci_lower = dr_curve - z_crit * dr_se
        ci_upper = dr_curve + z_crit * dr_se

        # Average marginal effect (numerical derivative)
        if len(dose_grid) >= 2:
            dt = dose_grid[1] - dose_grid[0]
            marginal_effects = np.gradient(dr_curve, dt)
            avg_marginal_effect = float(np.mean(marginal_effects))
        else:
            avg_marginal_effect = 0.0
            marginal_effects = np.zeros_like(dr_curve)

        # ATE-like summary: effect of going from 25th to 75th percentile
        idx_25 = max(0, int(0.25 * self.n_dose_points))
        idx_75 = min(self.n_dose_points - 1, int(0.75 * self.n_dose_points))
        effect_iqr = float(dr_curve[idx_75] - dr_curve[idx_25])

        # SE of the IQR effect
        boot_iqr = boot_curves[:, idx_75] - boot_curves[:, idx_25]
        se_iqr = float(np.std(boot_iqr, ddof=1))

        if se_iqr > 0:
            z_stat = effect_iqr / se_iqr
            pvalue = float(2 * (1 - sp_stats.norm.cdf(abs(z_stat))))
        else:
            pvalue = 0.0
        ci = (effect_iqr - z_crit * se_iqr, effect_iqr + z_crit * se_iqr)

        detail = pd.DataFrame({
            'dose': dose_grid,
            'response': dr_curve,
            'se': dr_se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'marginal_effect': marginal_effects,
        })

        model_info = {
            'dose_range': (float(t_lo), float(t_hi)),
            'n_dose_points': self.n_dose_points,
            'avg_marginal_effect': avg_marginal_effect,
            'effect_25_to_75': effect_iqr,
            'dose_25': float(dose_grid[idx_25]),
            'dose_75': float(dose_grid[idx_75]),
        }

        return CausalResult(
            method='Dose-Response (GPS, Hirano & Imbens 2004)',
            estimand='E[Y(t75)] - E[Y(t25)]',
            estimate=effect_iqr,
            se=se_iqr,
            pvalue=pvalue,
            ci=ci,
            alpha=self.alpha,
            n_obs=n,
            detail=detail,
            model_info=model_info,
            _citation_key='dose_response',
        )

    def _estimate_curve(self, Y, T, X, dose_grid, n):
        """Estimate E[Y(t)] at each dose level."""
        from sklearn.base import clone
        # Step 1: Treatment model E[T|X]
        t_model = clone(self.treatment_model)
        t_model.fit(X, T)
        T_hat = t_model.predict(X)
        residuals = T - T_hat
        sigma = max(np.std(residuals), 1e-8)

        # Step 2: GPS for observed treatments
        gps = sp_stats.norm.pdf(T, loc=T_hat, scale=sigma)

        # Step 3: Outcome model E[Y | T, GPS]
        TX = np.column_stack([T, gps])
        y_model = clone(self.outcome_model)
        y_model.fit(TX, Y)

        # Step 4: For each dose t, compute E_X[E[Y | T=t, GPS(t,X)]]
        curve = np.zeros(len(dose_grid))
        for k, t in enumerate(dose_grid):
            gps_t = sp_stats.norm.pdf(t, loc=T_hat, scale=sigma)
            tx_t = np.column_stack([np.full(n, t), gps_t])
            curve[k] = np.mean(y_model.predict(tx_t))

        return curve


# ======================================================================
# Citation
# ======================================================================

CausalResult._CITATIONS['dose_response'] = (
    "@incollection{hirano2004propensity,\n"
    "  title={The Propensity Score with Continuous Treatments},\n"
    "  author={Hirano, Keisuke and Imbens, Guido W},\n"
    "  booktitle={Applied Bayesian Modeling and Causal Inference from "
    "Incomplete-Data Perspectives},\n"
    "  pages={73--84},\n"
    "  year={2004},\n"
    "  publisher={Wiley}\n"
    "}"
)
