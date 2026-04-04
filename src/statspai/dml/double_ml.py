"""
Double/Debiased Machine Learning (Chernozhukov et al. 2018).

Implements the partially linear model:
    Y = theta * D + g(X) + eps
    D = m(X) + v

where g(.) and m(.) are estimated via cross-fitting with any sklearn-
compatible learner, and theta is the causal parameter of interest.

Supports:
- Partially linear regression (PLR)
- Interactive regression model (IRM) for binary treatment
- Multiple cross-fitting splits for median aggregation
"""

from typing import Optional, List, Dict, Any, Union
import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult


def dml(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    model: str = 'plr',
    ml_g: Optional[Any] = None,
    ml_m: Optional[Any] = None,
    n_folds: int = 5,
    n_rep: int = 1,
    alpha: float = 0.05,
) -> CausalResult:
    """
    Estimate causal effect using Double/Debiased Machine Learning.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    y : str
        Outcome variable.
    treat : str
        Treatment variable.
    covariates : list of str
        Control variables (high-dimensional confounders).
    model : str, default 'plr'
        DML model: 'plr' (partially linear) or 'irm' (interactive).
    ml_g : sklearn estimator, optional
        ML model for outcome nuisance (default: GradientBoostingRegressor).
    ml_m : sklearn estimator, optional
        ML model for treatment nuisance (default: same as ml_g, or
        GradientBoostingClassifier for binary treatment in IRM).
    n_folds : int, default 5
        Number of cross-fitting folds.
    n_rep : int, default 1
        Number of repeated cross-fitting splits (median aggregation).
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    CausalResult

    Examples
    --------
    >>> result = dml(df, y='wage', treat='training',
    ...             covariates=['age', 'edu', 'exp'])
    >>> print(result.summary())

    >>> # Custom ML models
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> result = dml(df, y='wage', treat='training',
    ...             covariates=['age', 'edu'],
    ...             ml_g=RandomForestRegressor(n_estimators=200))
    """
    estimator = DoubleML(
        data=data, y=y, treat=treat, covariates=covariates,
        model=model, ml_g=ml_g, ml_m=ml_m,
        n_folds=n_folds, n_rep=n_rep, alpha=alpha,
    )
    return estimator.fit()


class DoubleML:
    """
    Double Machine Learning estimator.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        y: str,
        treat: str,
        covariates: List[str],
        model: str = 'plr',
        ml_g: Optional[Any] = None,
        ml_m: Optional[Any] = None,
        n_folds: int = 5,
        n_rep: int = 1,
        alpha: float = 0.05,
    ):
        self.data = data
        self.y = y
        self.treat = treat
        self.covariates = covariates
        self.model = model.lower()
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.alpha = alpha

        self._validate()

        # Set default ML models
        self.ml_g = ml_g if ml_g is not None else self._default_ml_g()
        self.ml_m = ml_m if ml_m is not None else self._default_ml_m()

    def _validate(self):
        for col in [self.y, self.treat] + self.covariates:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in data")
        if self.model not in ('plr', 'irm'):
            raise ValueError(f"model must be 'plr' or 'irm', got '{self.model}'")
        if self.n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {self.n_folds}")

    def _default_ml_g(self):
        """Default ML model for outcome equation."""
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=42,
        )

    def _default_ml_m(self):
        """Default ML model for treatment equation."""
        if self.model == 'irm':
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                random_state=42,
            )
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=42,
        )

    def fit(self) -> CausalResult:
        """Fit the DML model and return results."""
        # Prepare data
        cols = [self.y, self.treat] + self.covariates
        clean = self.data[cols].dropna()
        Y = clean[self.y].values.astype(float)
        D = clean[self.treat].values.astype(float)
        X = clean[self.covariates].values.astype(float)
        n = len(Y)

        if self.model == 'plr':
            thetas, ses = self._fit_plr(Y, D, X, n)
        else:
            thetas, ses = self._fit_irm(Y, D, X, n)

        # Aggregate across repetitions (median for robustness)
        if len(thetas) == 1:
            theta = thetas[0]
            se = ses[0]
        else:
            theta = float(np.median(thetas))
            se = float(np.median(ses))

        # Inference
        t_stat = theta / se if se > 0 else 0
        pvalue = float(2 * (1 - stats.norm.cdf(abs(t_stat))))
        z_crit = stats.norm.ppf(1 - self.alpha / 2)
        ci = (theta - z_crit * se, theta + z_crit * se)

        model_info = {
            'dml_model': self.model.upper(),
            'n_folds': self.n_folds,
            'n_rep': self.n_rep,
            'ml_g': type(self.ml_g).__name__,
            'ml_m': type(self.ml_m).__name__,
            'n_covariates': len(self.covariates),
        }

        if self.n_rep > 1:
            model_info['theta_all_reps'] = thetas
            model_info['se_all_reps'] = ses

        return CausalResult(
            method=f'Double ML ({self.model.upper()})',
            estimand='ATE',
            estimate=theta,
            se=se,
            pvalue=pvalue,
            ci=ci,
            alpha=self.alpha,
            n_obs=n,
            detail=None,
            model_info=model_info,
            _citation_key='dml',
        )

    # ------------------------------------------------------------------
    # Partially Linear Regression (PLR)
    # ------------------------------------------------------------------

    def _fit_plr(self, Y, D, X, n):
        """
        PLR: Y = theta*D + g(X) + eps, D = m(X) + v

        Orthogonal score: psi = (Y - g_hat) - theta*(D - m_hat)
        theta = E[v_hat * (Y - g_hat)] / E[v_hat * (D - m_hat)]
              = E[v_hat * y_tilde] / E[v_hat * d_tilde]
        """
        from sklearn.base import clone
        from sklearn.model_selection import KFold

        thetas = []
        ses = []

        for rep in range(self.n_rep):
            kf = KFold(n_splits=self.n_folds, shuffle=True,
                        random_state=42 + rep)

            y_residuals = np.zeros(n)
            d_residuals = np.zeros(n)

            for train_idx, test_idx in kf.split(X):
                # Fit g: E[Y|X]
                ml_g = clone(self.ml_g)
                ml_g.fit(X[train_idx], Y[train_idx])
                y_residuals[test_idx] = Y[test_idx] - ml_g.predict(X[test_idx])

                # Fit m: E[D|X]
                ml_m = clone(self.ml_m)
                ml_m.fit(X[train_idx], D[train_idx])
                d_residuals[test_idx] = D[test_idx] - ml_m.predict(X[test_idx])

            # DML1: theta = sum(v_hat * y_tilde) / sum(v_hat * d_tilde)
            theta = float(
                np.sum(d_residuals * y_residuals) /
                np.sum(d_residuals * d_residuals)
            )

            # Standard error
            psi = y_residuals - theta * d_residuals  # orthogonal score
            J = -np.mean(d_residuals**2)
            sigma2 = np.mean((d_residuals * psi)**2)
            se = float(np.sqrt(sigma2 / (J**2 * n))) if abs(J) > 1e-10 else 0.0

            thetas.append(theta)
            ses.append(se)

        return thetas, ses

    # ------------------------------------------------------------------
    # Interactive Regression Model (IRM) — binary treatment
    # ------------------------------------------------------------------

    def _fit_irm(self, Y, D, X, n):
        """
        IRM for binary treatment:
        theta = E[g1(X) - g0(X)]  (ATE via AIPW)

        AIPW score:
        psi = g1(X) - g0(X)
              + D*(Y - g1(X))/m(X)
              - (1-D)*(Y - g0(X))/(1-m(X))
              - theta
        """
        from sklearn.base import clone
        from sklearn.model_selection import KFold

        thetas = []
        ses = []

        for rep in range(self.n_rep):
            kf = KFold(n_splits=self.n_folds, shuffle=True,
                        random_state=42 + rep)

            psi_scores = np.zeros(n)

            for train_idx, test_idx in kf.split(X):
                D_train, D_test = D[train_idx], D[test_idx]
                Y_train, Y_test = Y[train_idx], Y[test_idx]
                X_train, X_test = X[train_idx], X[test_idx]

                # g1: E[Y|X, D=1]
                ml_g1 = clone(self.ml_g)
                mask1 = D_train == 1
                if mask1.sum() > 0:
                    ml_g1.fit(X_train[mask1], Y_train[mask1])
                    g1_hat = ml_g1.predict(X_test)
                else:
                    g1_hat = np.zeros(len(test_idx))

                # g0: E[Y|X, D=0]
                ml_g0 = clone(self.ml_g)
                mask0 = D_train == 0
                if mask0.sum() > 0:
                    ml_g0.fit(X_train[mask0], Y_train[mask0])
                    g0_hat = ml_g0.predict(X_test)
                else:
                    g0_hat = np.zeros(len(test_idx))

                # m: P(D=1|X) — propensity score
                ml_m = clone(self.ml_m)
                ml_m.fit(X_train, D_train)
                if hasattr(ml_m, 'predict_proba'):
                    m_hat = ml_m.predict_proba(X_test)[:, 1]
                else:
                    m_hat = ml_m.predict(X_test)

                m_hat = np.clip(m_hat, 0.01, 0.99)

                # AIPW score
                D_t = D_test
                Y_t = Y_test
                psi_scores[test_idx] = (
                    g1_hat - g0_hat
                    + D_t * (Y_t - g1_hat) / m_hat
                    - (1 - D_t) * (Y_t - g0_hat) / (1 - m_hat)
                )

            theta = float(np.mean(psi_scores))
            se = float(np.std(psi_scores, ddof=1) / np.sqrt(n))

            thetas.append(theta)
            ses.append(se)

        return thetas, ses


# Citation
CausalResult._CITATIONS['dml'] = (
    "@article{chernozhukov2018double,\n"
    "  title={Double/Debiased Machine Learning for Treatment and "
    "Structural Parameters},\n"
    "  author={Chernozhukov, Victor and Chetverikov, Denis and "
    "Demirer, Mert and Duflo, Esther and Hansen, Christian and "
    "Newey, Whitney and Robins, James},\n"
    "  journal={The Econometrics Journal},\n"
    "  volume={21},\n"
    "  number={1},\n"
    "  pages={C1--C68},\n"
    "  year={2018},\n"
    "  publisher={Oxford University Press}\n"
    "}"
)
