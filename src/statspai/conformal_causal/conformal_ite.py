"""
Conformal prediction intervals for individual treatment effects.

Uses a split-conformal approach:
1. Split data into training and calibration sets.
2. Fit outcome models mu_0, mu_1 on training set.
3. Compute conformity scores on calibration set.
4. Construct prediction intervals using quantiles of scores.

The ITE interval for a new unit x is:
    [mu_1(x) - mu_0(x) - q_{1-alpha}, mu_1(x) - mu_0(x) + q_{1-alpha}]

where q is the (1-alpha) quantile of calibration residuals adjusted
for the counterfactual nature of the problem.

This implementation follows the "inexact nested" approach from
Lei & Candes (2021) adapted for the split-conformal setting.

References
----------
Lei, L. & Candes, E. J. (2021).
"Conformal Inference of Counterfactuals and Individual Treatment Effects."
JRSS-B, 83(5), 911-938.
"""

from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingRegressor

from ..core.results import CausalResult


# ======================================================================
# Public API
# ======================================================================

def conformal_cate(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    model: Optional[BaseEstimator] = None,
    alpha: float = 0.05,
    calib_fraction: float = 0.25,
    random_state: int = 42,
) -> CausalResult:
    """
    Compute conformal prediction intervals for CATE.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    y : str
        Outcome variable.
    treat : str
        Binary treatment variable (0/1).
    covariates : list of str
        Covariate names.
    model : sklearn estimator, optional
        Outcome model for mu_d(X). If None, uses GBM.
    alpha : float, default 0.05
        Miscoverage level. Intervals have (1-alpha) coverage.
    calib_fraction : float, default 0.25
        Fraction of data used for calibration.
    random_state : int, default 42

    Returns
    -------
    CausalResult
        Includes CATE point estimates and prediction intervals.
        model_info contains:
        - 'cate': point estimates
        - 'cate_lower': lower bounds of prediction intervals
        - 'cate_upper': upper bounds of prediction intervals
        - 'interval_width': average width of prediction intervals

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.conformal_cate(df, y='outcome', treat='treatment',
    ...                            covariates=['x1', 'x2'])
    >>> cate_lower = result.model_info['cate_lower']
    >>> cate_upper = result.model_info['cate_upper']
    """
    est = ConformalCATE(
        data=data, y=y, treat=treat, covariates=covariates,
        model=model, alpha=alpha, calib_fraction=calib_fraction,
        random_state=random_state,
    )
    return est.fit()


# ======================================================================
# ConformalCATE class
# ======================================================================

class ConformalCATE:
    """
    Conformal prediction intervals for individual treatment effects.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
    treat : str
    covariates : list of str
    model : sklearn estimator, optional
    alpha : float
    calib_fraction : float
    random_state : int
    """

    def __init__(
        self,
        data: pd.DataFrame,
        y: str,
        treat: str,
        covariates: List[str],
        model: Optional[BaseEstimator] = None,
        alpha: float = 0.05,
        calib_fraction: float = 0.25,
        random_state: int = 42,
    ):
        self.data = data
        self.y = y
        self.treat = treat
        self.covariates = covariates
        self.model = model or GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=random_state,
        )
        self.alpha = alpha
        self.calib_fraction = calib_fraction
        self.random_state = random_state

    def fit(self) -> CausalResult:
        """Compute conformal CATE intervals."""
        # Prepare data
        cols = [self.y, self.treat] + self.covariates
        missing = [c for c in cols if c not in self.data.columns]
        if missing:
            raise ValueError(f"Columns not found in data: {missing}")

        clean = self.data[cols].dropna()
        Y = clean[self.y].values.astype(np.float64)
        D = clean[self.treat].values.astype(np.float64)
        X = clean[self.covariates].values.astype(np.float64)
        n = len(Y)

        unique_d = np.unique(D)
        if not (len(unique_d) == 2 and set(unique_d.astype(int)) == {0, 1}):
            raise ValueError(f"Treatment must be binary (0/1)")

        rng = np.random.RandomState(self.random_state)

        # Split into train and calibration
        n_calib = max(int(n * self.calib_fraction), 10)
        indices = rng.permutation(n)
        calib_idx = indices[:n_calib]
        train_idx = indices[n_calib:]

        X_tr, Y_tr, D_tr = X[train_idx], Y[train_idx], D[train_idx]
        X_cal, Y_cal, D_cal = X[calib_idx], Y[calib_idx], D[calib_idx]

        # Fit outcome models on training set
        mask1_tr = D_tr == 1
        mask0_tr = D_tr == 0

        mu1 = clone(self.model)
        mu0 = clone(self.model)
        mu1.fit(X_tr[mask1_tr], Y_tr[mask1_tr])
        mu0.fit(X_tr[mask0_tr], Y_tr[mask0_tr])

        # Calibration: compute conformity scores
        # For treated calibration units: |Y_i - mu_1(X_i)| (observed arm)
        # For control calibration units: |Y_i - mu_0(X_i)| (observed arm)
        mask1_cal = D_cal == 1
        mask0_cal = D_cal == 0

        scores_1 = np.abs(Y_cal[mask1_cal] - mu1.predict(X_cal[mask1_cal]))
        scores_0 = np.abs(Y_cal[mask0_cal] - mu0.predict(X_cal[mask0_cal]))

        # Quantile for prediction intervals
        # For ITE interval, we need to account for uncertainty in both arms
        # Use the max of the two arm quantiles (conservative)
        def _conformal_quantile(scores, alpha):
            n_s = len(scores)
            if n_s == 0:
                return np.inf
            level = np.ceil((1 - alpha) * (n_s + 1)) / n_s
            level = min(level, 1.0)
            return float(np.quantile(scores, level))

        q1 = _conformal_quantile(scores_1, self.alpha / 2)
        q0 = _conformal_quantile(scores_0, self.alpha / 2)

        # Predict on full dataset
        mu1_all = mu1.predict(X)
        mu0_all = mu0.predict(X)

        cate = mu1_all - mu0_all
        cate_lower = cate - (q1 + q0)
        cate_upper = cate + (q1 + q0)

        ate = float(np.mean(cate))
        interval_width = float(np.mean(cate_upper - cate_lower))

        # Bootstrap SE for ATE
        boot_means = np.array([
            rng.choice(cate, size=n, replace=True).mean()
            for _ in range(500)
        ])
        se = float(np.std(boot_means, ddof=1))

        if se > 0:
            z_stat = ate / se
            pvalue = float(2 * (1 - sp_stats.norm.cdf(abs(z_stat))))
        else:
            pvalue = 0.0

        z_crit = sp_stats.norm.ppf(1 - self.alpha / 2)
        ci = (ate - z_crit * se, ate + z_crit * se)

        model_info = {
            'cate': cate,
            'cate_lower': cate_lower,
            'cate_upper': cate_upper,
            'interval_width': interval_width,
            'q_treated': q1,
            'q_control': q0,
            'calib_fraction': self.calib_fraction,
            'n_calib': n_calib,
            'n_train': len(train_idx),
            'coverage_level': 1 - self.alpha,
            'cate_mean': float(np.mean(cate)),
            'cate_std': float(np.std(cate)),
            'n_treated': int(np.sum(D == 1)),
            'n_control': int(np.sum(D == 0)),
        }

        self._mu1 = mu1
        self._mu0 = mu0
        self._q1 = q1
        self._q0 = q0

        return CausalResult(
            method='Conformal Causal Inference (Lei & Candes 2021)',
            estimand='ATE',
            estimate=ate,
            se=se,
            pvalue=pvalue,
            ci=ci,
            alpha=self.alpha,
            n_obs=n,
            detail=None,
            model_info=model_info,
            _citation_key='conformal_cate',
        )

    def predict(self, X_new: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict CATE with conformal intervals for new data.

        Returns
        -------
        dict with 'cate', 'lower', 'upper'
        """
        if not hasattr(self, '_mu1'):
            raise ValueError("Model must be fitted first.")

        X_new = np.asarray(X_new, dtype=np.float64)
        cate = self._mu1.predict(X_new) - self._mu0.predict(X_new)
        lower = cate - (self._q1 + self._q0)
        upper = cate + (self._q1 + self._q0)

        return {'cate': cate, 'lower': lower, 'upper': upper}


# ======================================================================
# Citation
# ======================================================================

CausalResult._CITATIONS['conformal_cate'] = (
    "@article{lei2021conformal,\n"
    "  title={Conformal Inference of Counterfactuals and Individual "
    "Treatment Effects},\n"
    "  author={Lei, Lihua and Cand{\\`e}s, Emmanuel J},\n"
    "  journal={Journal of the Royal Statistical Society: "
    "Series B (Statistical Methodology)},\n"
    "  volume={83},\n"
    "  number={5},\n"
    "  pages={911--938},\n"
    "  year={2021},\n"
    "  publisher={Wiley}\n"
    "}"
)
