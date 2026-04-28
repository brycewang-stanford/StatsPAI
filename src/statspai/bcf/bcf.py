"""
Bayesian Causal Forest (BCF): Warm-start BART for causal inference.

Since full BART MCMC is complex and typically requires dedicated C/C++
backends, this implementation provides a practical approximation using
ensemble methods with Bayesian-inspired regularisation:

1. Estimate propensity score e(X) = P(D=1|X).
2. Fit prognostic model: mu(X, e(X)) via ensemble (random forest).
3. Fit treatment effect model: tau(X) by regressing residuals
   Y - mu(X) on covariates, for treated units only, with shrinkage.
4. Bootstrap for posterior-like uncertainty quantification.

This captures the key BCF insights (propensity inclusion, separate
mu/tau, regularization toward homogeneity) without requiring MCMC.

References
----------
Hahn, P. R., Murray, J. S., & Carvalho, C. M. (2020).
"Bayesian Regression Tree Models for Causal Inference."
Bayesian Analysis, 15(3), 965-1056. [@hahn2020bayesian]
"""

from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn.model_selection import KFold

from ..core.results import CausalResult


# ======================================================================
# Public API
# ======================================================================

def bcf(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    n_trees_mu: int = 200,
    n_trees_tau: int = 50,
    n_bootstrap: int = 200,
    n_folds: int = 5,
    alpha: float = 0.05,
    random_state: int = 42,
) -> CausalResult:
    """
    Estimate heterogeneous treatment effects using Bayesian Causal Forest.

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
    n_trees_mu : int, default 200
        Number of trees for the prognostic function mu(X).
    n_trees_tau : int, default 50
        Number of trees for the treatment effect function tau(X).
        Fewer trees = stronger shrinkage toward homogeneous effects.
    n_bootstrap : int, default 200
        Bootstrap iterations for uncertainty quantification.
    n_folds : int, default 5
        Cross-fitting folds for propensity estimation.
    alpha : float, default 0.05
        Significance level.
    random_state : int, default 42

    Returns
    -------
    CausalResult

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.bcf(df, y='outcome', treat='treatment',
    ...                 covariates=['x1', 'x2', 'x3'])
    >>> print(result.summary())
    >>> cate = result.model_info['cate']  # individual effects
    >>> cate_sd = result.model_info['cate_sd']  # posterior SD
    """
    est = BayesianCausalForest(
        data=data, y=y, treat=treat, covariates=covariates,
        n_trees_mu=n_trees_mu, n_trees_tau=n_trees_tau,
        n_bootstrap=n_bootstrap, n_folds=n_folds,
        alpha=alpha, random_state=random_state,
    )
    _result = est.fit()
    try:
        from ..output._lineage import attach_provenance as _attach_prov
        _attach_prov(
            _result,
            function="sp.bcf",
            params={
                "y": y, "treat": treat,
                "covariates": list(covariates),
                "n_trees_mu": n_trees_mu,
                "n_trees_tau": n_trees_tau,
                "n_bootstrap": n_bootstrap,
                "n_folds": n_folds,
                "alpha": alpha,
                "random_state": random_state,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


# ======================================================================
# BayesianCausalForest class
# ======================================================================

class BayesianCausalForest:
    """
    Bayesian Causal Forest estimator.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
    treat : str
    covariates : list of str
    n_trees_mu : int
    n_trees_tau : int
    n_bootstrap : int
    n_folds : int
    alpha : float
    random_state : int
    """

    def __init__(
        self,
        data: pd.DataFrame,
        y: str,
        treat: str,
        covariates: List[str],
        n_trees_mu: int = 200,
        n_trees_tau: int = 50,
        n_bootstrap: int = 200,
        n_folds: int = 5,
        alpha: float = 0.05,
        random_state: int = 42,
    ):
        self.data = data
        self.y = y
        self.treat = treat
        self.covariates = covariates
        self.n_trees_mu = n_trees_mu
        self.n_trees_tau = n_trees_tau
        self.n_bootstrap = n_bootstrap
        self.n_folds = n_folds
        self.alpha = alpha
        self.random_state = random_state

    def fit(self) -> CausalResult:
        """Fit BCF and return treatment effect estimates."""
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

        # Step 1: Estimate propensity scores via cross-fitting
        e_hat = np.zeros(n)
        kf = KFold(n_splits=self.n_folds, shuffle=True,
                    random_state=self.random_state)

        for train_idx, test_idx in kf.split(X):
            prop = GradientBoostingClassifier(
                n_estimators=100, max_depth=3,
                random_state=self.random_state
            )
            prop.fit(X[train_idx], D[train_idx])
            e_hat[test_idx] = np.clip(
                prop.predict_proba(X[test_idx])[:, 1], 0.025, 0.975
            )

        # Step 2: Fit prognostic model mu(X, e_hat)
        X_aug = np.column_stack([X, e_hat])

        mu_model = RandomForestRegressor(
            n_estimators=self.n_trees_mu, max_depth=6,
            min_samples_leaf=5, random_state=self.random_state,
        )
        # Fit mu on control observations
        mask0 = D == 0
        mu_model.fit(X_aug[mask0], Y[mask0])
        mu_hat = mu_model.predict(X_aug)

        # Step 3: Fit treatment effect model tau(X)
        # Residuals for treated: Y_i - mu_hat(X_i, e_hat_i)
        mask1 = D == 1
        residuals_1 = Y[mask1] - mu_hat[mask1]

        tau_model = GradientBoostingRegressor(
            n_estimators=self.n_trees_tau, max_depth=3,
            learning_rate=0.05, subsample=0.8,
            min_samples_leaf=10,
            random_state=self.random_state,
        )
        tau_model.fit(X[mask1], residuals_1)
        cate = tau_model.predict(X)

        ate = float(np.mean(cate))

        # Step 4: Bootstrap for uncertainty
        boot_cate = np.zeros((self.n_bootstrap, n))
        boot_ates = np.zeros(self.n_bootstrap)

        for b in range(self.n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            Y_b, D_b, X_b, e_b = Y[idx], D[idx], X[idx], e_hat[idx]

            X_aug_b = np.column_stack([X_b, e_b])
            mask0_b = D_b == 0
            mask1_b = D_b == 1

            if mask0_b.sum() < 5 or mask1_b.sum() < 5:
                boot_cate[b] = cate
                boot_ates[b] = ate
                continue

            mu_b = clone(mu_model)
            mu_b.fit(X_aug_b[mask0_b], Y_b[mask0_b])
            mu_pred_b = mu_b.predict(X_aug_b)

            resid_b = Y_b[mask1_b] - mu_pred_b[mask1_b]
            tau_b = clone(tau_model)
            tau_b.fit(X_b[mask1_b], resid_b)

            boot_cate[b] = tau_b.predict(X)
            boot_ates[b] = np.mean(boot_cate[b])

        se = float(np.std(boot_ates, ddof=1))
        cate_sd = np.std(boot_cate, axis=0, ddof=1)

        # Credible intervals for CATE
        cate_lower = np.percentile(boot_cate, 100 * self.alpha / 2, axis=0)
        cate_upper = np.percentile(boot_cate, 100 * (1 - self.alpha / 2), axis=0)

        if se > 0:
            z_stat = ate / se
            pvalue = float(2 * (1 - sp_stats.norm.cdf(abs(z_stat))))
        else:
            pvalue = 0.0

        z_crit = sp_stats.norm.ppf(1 - self.alpha / 2)
        ci = (ate - z_crit * se, ate + z_crit * se)

        model_info = {
            'architecture': 'BCF',
            'n_trees_mu': self.n_trees_mu,
            'n_trees_tau': self.n_trees_tau,
            'n_bootstrap': self.n_bootstrap,
            'propensity_mean': float(np.mean(e_hat)),
            'cate': cate,
            'cate_sd': cate_sd,
            'cate_lower': cate_lower,
            'cate_upper': cate_upper,
            'cate_mean': float(np.mean(cate)),
            'cate_median': float(np.median(cate)),
            'cate_std': float(np.std(cate)),
            'n_treated': int(np.sum(D == 1)),
            'n_control': int(np.sum(D == 0)),
        }

        self._mu_model = mu_model
        self._tau_model = tau_model
        self._cate = cate

        return CausalResult(
            method='BCF (Hahn, Murray, Carvalho 2020)',
            estimand='ATE',
            estimate=ate,
            se=se,
            pvalue=pvalue,
            ci=ci,
            alpha=self.alpha,
            n_obs=n,
            detail=None,
            model_info=model_info,
            _citation_key='bcf',
        )

    def effect(self, X_new: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict CATE for new observations."""
        if not hasattr(self, '_tau_model'):
            raise ValueError("Model must be fitted first.")
        if X_new is None:
            return self._cate.copy()
        X_new = np.asarray(X_new, dtype=np.float64)
        return self._tau_model.predict(X_new)


# ======================================================================
# Citation
# ======================================================================

CausalResult._CITATIONS['bcf'] = (
    "@article{hahn2020bayesian,\n"
    "  title={Bayesian Regression Tree Models for Causal Inference: "
    "Regularization, Confounding, and Heterogeneous Effects},\n"
    "  author={Hahn, P Richard and Murray, Jared S and Carvalho, "
    "Carlos M},\n"
    "  journal={Bayesian Analysis},\n"
    "  volume={15},\n"
    "  number={3},\n"
    "  pages={965--1056},\n"
    "  year={2020}\n"
    "}"
)
