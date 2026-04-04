"""
Causal Impact: Bayesian structural time-series intervention analysis.

Given a time series that experienced an intervention, estimates the causal
effect by constructing a counterfactual from covariates (control series)
using a regression + AR(1) model fit on the pre-intervention period.

This is a frequentist approximation of the full Bayesian approach in
Brodersen et al. (2015), designed to work without MCMC dependencies
(no PyMC/Stan required). For the full Bayesian version, users can
install optional dependencies.

The model:
    Y_t = beta' * X_t + mu_t + eps_t     (observation equation)
    mu_t = mu_{t-1} + eta_t               (local level / random walk)

Fit on pre-period, then forecast into post-period for counterfactual.
"""

from typing import Optional, List, Dict, Any, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from ..core.results import CausalResult


def causal_impact(
    data: pd.DataFrame,
    y: str,
    time: str,
    intervention_time: Any,
    covariates: Optional[List[str]] = None,
    alpha: float = 0.05,
    n_seasons: Optional[int] = None,
) -> CausalResult:
    """
    Estimate the causal impact of an intervention on a time series.

    Parameters
    ----------
    data : pd.DataFrame
        Time-series data (one row per time period).
    y : str
        Outcome variable (the series that was intervened upon).
    time : str
        Time column (used for ordering; can be int, date, etc.).
    intervention_time : any
        First period of the intervention (inclusive).
    covariates : list of str, optional
        Control time series (not affected by the intervention).
        If None, uses a local-level model without covariates.
    alpha : float, default 0.05
        Significance level for credible intervals.
    n_seasons : int, optional
        Seasonal period (e.g., 7 for weekly, 12 for monthly).

    Returns
    -------
    CausalResult

    Examples
    --------
    >>> # Effect of a marketing campaign starting week 50
    >>> result = causal_impact(df, y='sales', time='week',
    ...                        intervention_time=50,
    ...                        covariates=['competitor_sales', 'ad_spend'])
    >>> print(result.summary())
    >>> result.plot()
    """
    estimator = CausalImpactEstimator(
        data=data, y=y, time=time,
        intervention_time=intervention_time,
        covariates=covariates, alpha=alpha,
        n_seasons=n_seasons,
    )
    return estimator.fit()


class CausalImpactEstimator:
    """
    Causal Impact estimator using structural time-series model.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        y: str,
        time: str,
        intervention_time: Any,
        covariates: Optional[List[str]] = None,
        alpha: float = 0.05,
        n_seasons: Optional[int] = None,
    ):
        self.data = data.sort_values(time).copy()
        self.y = y
        self.time = time
        self.intervention_time = intervention_time
        self.covariates = covariates or []
        self.alpha = alpha
        self.n_seasons = n_seasons

        self._validate()
        self._prepare()

    def _validate(self):
        for col in [self.y, self.time] + self.covariates:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in data")

        times = self.data[self.time].values
        if self.intervention_time not in times:
            # Check if intervention_time is between min and max
            if not (times.min() < self.intervention_time <= times.max()):
                raise ValueError(
                    f"intervention_time={self.intervention_time} outside data range "
                    f"[{times.min()}, {times.max()}]"
                )

    def _prepare(self):
        """Split data into pre and post periods."""
        self.times = self.data[self.time].values
        self.Y = self.data[self.y].values.astype(float)

        self.pre_mask = self.times < self.intervention_time
        self.post_mask = self.times >= self.intervention_time

        self.n_pre = int(self.pre_mask.sum())
        self.n_post = int(self.post_mask.sum())

        if self.n_pre < 3:
            raise ValueError(f"Need at least 3 pre-intervention periods, got {self.n_pre}")
        if self.n_post < 1:
            raise ValueError("Need at least 1 post-intervention period")

        if self.covariates:
            self.X = self.data[self.covariates].values.astype(float)
        else:
            self.X = None

    def fit(self) -> CausalResult:
        """Fit the structural time-series model and estimate causal impact."""
        Y_pre = self.Y[self.pre_mask]
        X_pre = self.X[self.pre_mask] if self.X is not None else None
        X_post = self.X[self.post_mask] if self.X is not None else None

        # Fit model on pre-intervention period
        model = self._fit_structural_model(Y_pre, X_pre)

        # Generate counterfactual predictions for full period
        Y_pred, Y_pred_se = self._predict(model, self.X)

        # Point estimates
        Y_actual = self.Y
        pointwise_effect = Y_actual - Y_pred
        pointwise_lower = Y_actual - (Y_pred + stats.norm.ppf(1 - self.alpha / 2) * Y_pred_se)
        pointwise_upper = Y_actual - (Y_pred - stats.norm.ppf(1 - self.alpha / 2) * Y_pred_se)

        # Cumulative effect (post-period only)
        post_effect = pointwise_effect[self.post_mask]
        cumulative_effect = np.cumsum(post_effect)

        # Summary statistics
        avg_effect = float(np.mean(post_effect))
        total_effect = float(np.sum(post_effect))

        # Standard error from prediction uncertainty
        post_se = Y_pred_se[self.post_mask]
        se_avg = float(np.sqrt(np.mean(post_se**2)) / np.sqrt(self.n_post))
        se_total = float(np.sqrt(np.sum(post_se**2)))

        # Relative effect
        post_pred_mean = float(np.mean(Y_pred[self.post_mask]))
        relative_effect = avg_effect / post_pred_mean if abs(post_pred_mean) > 1e-10 else np.nan

        # P-value (two-sided test against zero effect)
        z_stat = avg_effect / se_avg if se_avg > 0 else 0
        pvalue = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

        z_crit = stats.norm.ppf(1 - self.alpha / 2)
        ci = (avg_effect - z_crit * se_avg, avg_effect + z_crit * se_avg)

        # Build detail table (time series)
        detail = pd.DataFrame({
            'time': self.times,
            'actual': Y_actual,
            'predicted': Y_pred,
            'predicted_se': Y_pred_se,
            'effect': pointwise_effect,
            'effect_lower': pointwise_lower,
            'effect_upper': pointwise_upper,
            'post_intervention': self.post_mask,
        })

        model_info: Dict[str, Any] = {
            'intervention_time': self.intervention_time,
            'n_pre': self.n_pre,
            'n_post': self.n_post,
            'avg_effect': avg_effect,
            'total_effect': total_effect,
            'se_avg': se_avg,
            'se_total': se_total,
            'relative_effect': relative_effect,
            'relative_effect_pct': relative_effect * 100 if not np.isnan(relative_effect) else np.nan,
            'n_covariates': len(self.covariates),
            'model_params': model,
            'Y_pred': Y_pred,
            'Y_actual': Y_actual,
            'times': self.times,
            'pre_mask': self.pre_mask,
            'post_mask': self.post_mask,
            'cumulative_effect': cumulative_effect,
        }

        return CausalResult(
            method='Causal Impact (Structural Time Series)',
            estimand='Average Causal Effect',
            estimate=avg_effect,
            se=se_avg,
            pvalue=pvalue,
            ci=ci,
            alpha=self.alpha,
            n_obs=len(self.Y),
            detail=detail,
            model_info=model_info,
            _citation_key='causal_impact',
        )

    # ------------------------------------------------------------------
    # Structural time-series model
    # ------------------------------------------------------------------

    def _fit_structural_model(self, Y_pre, X_pre):
        """
        Fit a local-level + regression model on pre-intervention data.

        Model: Y_t = beta' * X_t + mu_t + eps_t
               mu_t = mu_{t-1} + eta_t

        Returns dict with fitted parameters.
        """
        n = len(Y_pre)

        if X_pre is not None and X_pre.shape[1] > 0:
            # Add intercept
            X_aug = np.column_stack([np.ones(n), X_pre])
            # OLS for regression coefficients
            try:
                beta = np.linalg.lstsq(X_aug, Y_pre, rcond=None)[0]
            except np.linalg.LinAlgError:
                beta = np.zeros(X_aug.shape[1])
            residuals = Y_pre - X_aug @ beta
        else:
            X_aug = np.ones((n, 1))
            beta = np.array([np.mean(Y_pre)])
            residuals = Y_pre - beta[0]

        # Estimate AR(1) on residuals for local level dynamics
        if n > 2:
            r_lag = residuals[:-1]
            r_lead = residuals[1:]
            if np.var(r_lag) > 1e-10:
                rho = float(np.corrcoef(r_lag, r_lead)[0, 1])
                rho = np.clip(rho, -0.99, 0.99)
            else:
                rho = 0.0
        else:
            rho = 0.0

        # Variance decomposition
        sigma_obs = float(np.std(residuals, ddof=1)) if n > 1 else 1.0
        sigma_state = sigma_obs * np.sqrt(1 - rho**2) if abs(rho) < 1 else sigma_obs * 0.1

        return {
            'beta': beta,
            'rho': rho,
            'sigma_obs': max(sigma_obs, 1e-10),
            'sigma_state': max(sigma_state, 1e-10),
            'last_residual': residuals[-1] if len(residuals) > 0 else 0.0,
            'n_covariates': X_pre.shape[1] if X_pre is not None else 0,
        }

    def _predict(self, model, X_full):
        """
        Generate counterfactual predictions for all time periods.

        Returns (Y_pred, Y_pred_se) arrays.
        """
        n = len(self.Y)
        beta = model['beta']
        rho = model['rho']
        sigma_obs = model['sigma_obs']
        sigma_state = model['sigma_state']

        if X_full is not None and X_full.shape[1] > 0:
            X_aug = np.column_stack([np.ones(n), X_full])
        else:
            X_aug = np.ones((n, 1))

        # Regression component
        Y_reg = X_aug @ beta

        # Local level (state) component: AR(1) on residuals
        Y_pred = np.zeros(n)
        Y_pred_se = np.zeros(n)

        state = 0.0
        state_var = sigma_state**2

        for t in range(n):
            Y_pred[t] = Y_reg[t] + state
            Y_pred_se[t] = np.sqrt(sigma_obs**2 + state_var)

            if self.pre_mask[t]:
                # Update state with observed data (Kalman-like)
                innovation = self.Y[t] - Y_pred[t]
                K = state_var / (state_var + sigma_obs**2)  # Kalman gain
                state = rho * (state + K * innovation)
                state_var = rho**2 * (1 - K) * state_var + sigma_state**2
            else:
                # Post-period: propagate uncertainty (no update)
                state = rho * state
                state_var = rho**2 * state_var + sigma_state**2

        return Y_pred, Y_pred_se


# Citation
CausalResult._CITATIONS['causal_impact'] = (
    "@article{brodersen2015inferring,\n"
    "  title={Inferring Causal Impact Using Bayesian Structural "
    "Time-Series Models},\n"
    "  author={Brodersen, Kay H and Gallusser, Fabian and "
    "Koehler, Jim and Remy, Nicolas and Scott, Steven L},\n"
    "  journal={The Annals of Applied Statistics},\n"
    "  volume={9},\n"
    "  number={1},\n"
    "  pages={247--274},\n"
    "  year={2015},\n"
    "  publisher={Institute of Mathematical Statistics}\n"
    "}"
)
