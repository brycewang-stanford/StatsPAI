"""
Targeted Maximum Likelihood Estimation (TMLE) for causal inference.

TMLE is a two-step semiparametric estimator:

1. **Initial estimate**: Fit outcome model Q(Y | A, W) and propensity
   model g(A | W) using flexible ML (Super Learner).

2. **Targeting step**: Update the initial outcome estimate along the
   least-favourable submodel using the clever covariate:
       H(A, W) = A/g(W) - (1-A)/(1-g(W))
   Fit epsilon by regressing Y on H with offset logit(Q_bar).

3. **Plug-in estimate**: Compute ATE as mean(Q*(1,W) - Q*(0,W))
   using the targeted (updated) outcome predictions.

The resulting estimator is:
- Doubly robust: consistent if either Q or g is correct
- Semiparametrically efficient: achieves the efficiency bound
- Regular and asymptotically linear with known influence function

References
----------
van der Laan, M. J. & Rose, S. (2011).
Targeted Learning. Springer Series in Statistics.

van der Laan, M. J. & Rubin, D. (2006).
Targeted Maximum Likelihood Learning.
International Journal of Biostatistics, 2(1).
"""

from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.special import logit, expit
from sklearn.base import BaseEstimator, clone

from ..core.results import CausalResult
from .super_learner import SuperLearner


# ======================================================================
# Public API
# ======================================================================

def tmle(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    outcome_library: Optional[List[BaseEstimator]] = None,
    propensity_library: Optional[List[BaseEstimator]] = None,
    n_folds: int = 5,
    estimand: str = 'ATE',
    alpha: float = 0.05,
    propensity_bounds: Tuple[float, float] = (0.025, 0.975),
    random_state: int = 42,
) -> CausalResult:
    """
    Estimate causal effects using TMLE with Super Learner.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    y : str
        Outcome variable (binary or continuous).
    treat : str
        Binary treatment variable (0/1).
    covariates : list of str
        Covariate names.
    outcome_library : list of sklearn estimators, optional
        Candidate learners for the outcome model Q(Y|A,W).
        If None, uses a default diverse library.
    propensity_library : list of sklearn estimators, optional
        Candidate learners for propensity model g(A|W).
        If None, uses a default diverse library.
    n_folds : int, default 5
        Cross-validation folds for Super Learner.
    estimand : str, default 'ATE'
        'ATE' or 'ATT'.
    alpha : float, default 0.05
        Significance level.
    propensity_bounds : tuple, default (0.025, 0.975)
        Bounds for propensity score truncation.
    random_state : int, default 42

    Returns
    -------
    CausalResult

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.tmle(df, y='outcome', treat='treatment',
    ...                  covariates=['x1', 'x2', 'x3'])
    >>> print(result.summary())

    >>> # Custom learner library
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> result = sp.tmle(df, y='outcome', treat='treatment',
    ...                  covariates=['x1', 'x2'],
    ...                  outcome_library=[RandomForestRegressor()])
    """
    est = TMLE(
        data=data, y=y, treat=treat, covariates=covariates,
        outcome_library=outcome_library,
        propensity_library=propensity_library,
        n_folds=n_folds, estimand=estimand, alpha=alpha,
        propensity_bounds=propensity_bounds,
        random_state=random_state,
    )
    return est.fit()


# ======================================================================
# TMLE Estimator
# ======================================================================

class TMLE:
    """
    Targeted Maximum Likelihood Estimation.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
    treat : str
    covariates : list of str
    outcome_library : list of sklearn estimators, optional
    propensity_library : list of sklearn estimators, optional
    n_folds : int
    estimand : str
    alpha : float
    propensity_bounds : tuple
    random_state : int
    """

    def __init__(
        self,
        data: pd.DataFrame,
        y: str,
        treat: str,
        covariates: List[str],
        outcome_library: Optional[List[BaseEstimator]] = None,
        propensity_library: Optional[List[BaseEstimator]] = None,
        n_folds: int = 5,
        estimand: str = 'ATE',
        alpha: float = 0.05,
        propensity_bounds: Tuple[float, float] = (0.025, 0.975),
        random_state: int = 42,
    ):
        self.data = data
        self.y = y
        self.treat = treat
        self.covariates = covariates
        self.outcome_library = outcome_library
        self.propensity_library = propensity_library
        self.n_folds = n_folds
        self.estimand = estimand
        self.alpha = alpha
        self.propensity_bounds = propensity_bounds
        self.random_state = random_state

    def fit(self) -> CausalResult:
        """Run TMLE and return causal effect estimates."""
        # Prepare data
        cols = [self.y, self.treat] + self.covariates
        missing = [c for c in cols if c not in self.data.columns]
        if missing:
            raise ValueError(f"Columns not found in data: {missing}")

        clean = self.data[cols].dropna()
        Y = clean[self.y].values.astype(np.float64)
        A = clean[self.treat].values.astype(np.float64)
        W = clean[self.covariates].values.astype(np.float64)
        n = len(Y)

        unique_a = np.unique(A)
        if not (len(unique_a) == 2 and set(unique_a.astype(int)) == {0, 1}):
            raise ValueError(
                f"Treatment must be binary (0/1), got: {unique_a}"
            )

        # Detect if outcome is binary
        is_binary_outcome = set(np.unique(Y)) <= {0.0, 1.0}

        # For continuous outcomes, bound Y to [0,1] for logistic fluctuation
        if not is_binary_outcome:
            y_min, y_max = Y.min(), Y.max()
            y_range = y_max - y_min
            Y_scaled = (Y - y_min) / (y_range + 1e-10)
        else:
            Y_scaled = Y
            y_min, y_max, y_range = 0.0, 1.0, 1.0

        # ---------------------------------------------------------------
        # Step 1: Initial estimates via Super Learner
        # ---------------------------------------------------------------

        # Outcome model: Q(Y | A, W)
        AW = np.column_stack([A, W])
        sl_Q = SuperLearner(
            library=self.outcome_library,
            n_folds=self.n_folds,
            task='classification' if is_binary_outcome else 'regression',
            random_state=self.random_state,
        )
        sl_Q.fit(AW, Y_scaled)

        # Initial predictions Q_bar(a, W) for a=0 and a=1
        W1 = np.column_stack([np.ones(n), W])
        W0 = np.column_stack([np.zeros(n), W])

        Q_bar_A = sl_Q.predict(AW)  # Q(A_i, W_i) for observed A
        Q_bar_1 = sl_Q.predict(W1)  # Q(1, W_i)
        Q_bar_0 = sl_Q.predict(W0)  # Q(0, W_i)

        # Bound predictions
        eps_bound = 1e-5
        Q_bar_A = np.clip(Q_bar_A, eps_bound, 1 - eps_bound)
        Q_bar_1 = np.clip(Q_bar_1, eps_bound, 1 - eps_bound)
        Q_bar_0 = np.clip(Q_bar_0, eps_bound, 1 - eps_bound)

        # Propensity model: g(A | W)
        sl_g = SuperLearner(
            library=self.propensity_library,
            n_folds=self.n_folds,
            task='classification',
            random_state=self.random_state,
        )
        sl_g.fit(W, A)
        g_hat = sl_g.predict(W)
        g_hat = np.clip(g_hat, self.propensity_bounds[0],
                        self.propensity_bounds[1])

        # ---------------------------------------------------------------
        # Step 2: Targeting step (fluctuation parameter epsilon)
        # ---------------------------------------------------------------

        # Clever covariate H(A, W)
        if self.estimand == 'ATE':
            H_A = A / g_hat - (1 - A) / (1 - g_hat)
            H_1 = 1.0 / g_hat
            H_0 = -1.0 / (1 - g_hat)
        else:  # ATT
            H_A = A - (1 - A) * g_hat / (1 - g_hat)
            H_1 = np.ones(n)
            H_0 = -g_hat / (1 - g_hat)

        # Logistic fluctuation model:
        # logit(Q*(A,W)) = logit(Q_bar(A,W)) + epsilon * H(A,W)
        # Fit epsilon by MLE (logistic regression with offset)
        logit_Q_A = logit(Q_bar_A)

        # Simple Newton-Raphson for epsilon (1D optimisation)
        epsilon = self._fit_epsilon(Y_scaled, logit_Q_A, H_A)

        # Update Q predictions
        Q_star_A = expit(logit_Q_A + epsilon * H_A)
        Q_star_1 = expit(logit(Q_bar_1) + epsilon * H_1)
        Q_star_0 = expit(logit(Q_bar_0) + epsilon * H_0)

        # ---------------------------------------------------------------
        # Step 3: Plug-in estimate
        # ---------------------------------------------------------------

        if not is_binary_outcome:
            # Rescale back to original Y scale
            Q_star_1_orig = Q_star_1 * (y_range + 1e-10) + y_min
            Q_star_0_orig = Q_star_0 * (y_range + 1e-10) + y_min
            Q_star_A_orig = Q_star_A * (y_range + 1e-10) + y_min
        else:
            Q_star_1_orig = Q_star_1
            Q_star_0_orig = Q_star_0
            Q_star_A_orig = Q_star_A

        if self.estimand == 'ATE':
            psi = float(np.mean(Q_star_1_orig - Q_star_0_orig))

            # Efficient influence function
            EIF = (
                (Q_star_1_orig - Q_star_0_orig)
                + A * (Y - Q_star_A_orig) / g_hat
                - (1 - A) * (Y - Q_star_A_orig) / (1 - g_hat)
                - psi
            )
        else:  # ATT
            p_treat = np.mean(A)
            psi = float(np.mean(
                A * (Y - Q_star_0_orig) / p_treat
                - (1 - A) * g_hat * (Y - Q_star_0_orig) / ((1 - g_hat) * p_treat)
            ))

            EIF = (
                A * (Y - Q_star_0_orig) / p_treat
                - (1 - A) * g_hat * (Y - Q_star_0_orig) / ((1 - g_hat) * p_treat)
                - psi * A / p_treat
            )

        # Standard error from influence function
        se = float(np.std(EIF, ddof=1) / np.sqrt(n))

        if se > 0:
            z_stat = psi / se
            pvalue = float(2 * (1 - sp_stats.norm.cdf(abs(z_stat))))
        else:
            pvalue = 0.0

        z_crit = sp_stats.norm.ppf(1 - self.alpha / 2)
        ci = (psi - z_crit * se, psi + z_crit * se)

        # Model info
        model_info = {
            'estimand': self.estimand,
            'epsilon': float(epsilon),
            'se_method': 'efficient_influence_function',
            'propensity_mean': float(np.mean(g_hat)),
            'propensity_std': float(np.std(g_hat)),
            'propensity_bounds': self.propensity_bounds,
            'outcome_type': 'binary' if is_binary_outcome else 'continuous',
            'n_folds': self.n_folds,
            'Q_star_1_mean': float(np.mean(Q_star_1_orig)),
            'Q_star_0_mean': float(np.mean(Q_star_0_orig)),
            'n_treated': int(np.sum(A == 1)),
            'n_control': int(np.sum(A == 0)),
            'sl_outcome_weights': sl_Q.weights_.tolist(),
            'sl_propensity_weights': sl_g.weights_.tolist(),
        }

        self._sl_Q = sl_Q
        self._sl_g = sl_g
        self._epsilon = epsilon

        return CausalResult(
            method='TMLE (van der Laan & Rose 2011)',
            estimand=self.estimand,
            estimate=psi,
            se=se,
            pvalue=pvalue,
            ci=ci,
            alpha=self.alpha,
            n_obs=n,
            detail=None,
            model_info=model_info,
            _citation_key='tmle',
        )

    def _fit_epsilon(self, Y, logit_Q, H, max_iter=50, tol=1e-8):
        """
        Fit the fluctuation parameter epsilon via Newton-Raphson.

        Logistic model: P(Y=1 | H) = expit(logit_Q + epsilon * H)
        MLE for epsilon using iteratively reweighted least squares.
        """
        epsilon = 0.0

        for _ in range(max_iter):
            p = expit(logit_Q + epsilon * H)
            # Score: sum(H * (Y - p))
            score = np.sum(H * (Y - p))
            # Hessian: -sum(H^2 * p * (1-p))
            hessian = -np.sum(H ** 2 * p * (1 - p))

            if abs(hessian) < 1e-15:
                break

            delta = -score / hessian
            epsilon += delta

            if abs(delta) < tol:
                break

        return epsilon


# ======================================================================
# Citation
# ======================================================================

CausalResult._CITATIONS['tmle'] = (
    "@book{vanderlaan2011targeted,\n"
    "  title={Targeted Learning: Causal Inference for Observational "
    "and Experimental Data},\n"
    "  author={van der Laan, Mark J and Rose, Sherri},\n"
    "  year={2011},\n"
    "  publisher={Springer}\n"
    "}\n\n"
    "@article{vanderlaan2007super,\n"
    "  title={Super learner},\n"
    "  author={van der Laan, Mark J and Polley, Eric C and Hubbard, Alan E},\n"
    "  journal={Statistical Applications in Genetics and Molecular Biology},\n"
    "  volume={6},\n"
    "  number={1},\n"
    "  year={2007}\n"
    "}"
)
