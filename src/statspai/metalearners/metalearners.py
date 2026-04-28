"""
Meta-Learners for heterogeneous treatment effect (CATE) estimation.

Implements five canonical meta-learners that decompose CATE estimation into
standard supervised-learning sub-problems, following:

- Kunzel, S. R., Seetharam, J. S., Liang, P., & Athey, S. (2019).
  Metalearners for estimating heterogeneous treatment effects using
  machine learning. *PNAS*, 116(10), 4156-4165.
- Kennedy, E. H. (2023). Towards optimal doubly robust estimation of
  heterogeneous causal effects. *Electronic Journal of Statistics*,
  17(2), 3008-3049.
- Nie, X., & Wager, S. (2021). Quasi-oracle estimation of heterogeneous
  treatment effects. *Biometrika*, 108(2), 299-319.

All learners accept any scikit-learn compatible estimators for the
nuisance and CATE stages.

Supported learners
------------------
- **S-Learner** : single model with treatment as feature
- **T-Learner** : separate models per treatment arm
- **X-Learner** : two-stage imputed treatment effect (Kunzel et al.)
- **R-Learner** : Robinson decomposition + loss minimisation (Nie & Wager)
- **DR-Learner**: doubly robust pseudo-outcome regression (Kennedy)
"""

from typing import Optional, List, Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.ensemble import (
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    RandomForestRegressor,
)

from ..core.results import CausalResult


# ======================================================================
# Helpers
# ======================================================================

def _default_outcome_model():
    return GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )


def _default_propensity_model():
    return GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )


def _default_cate_model():
    return GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )


def _get_propensity(model, X, clip=(0.01, 0.99)):
    """Return P(D=1|X), clipped for stability."""
    if hasattr(model, 'predict_proba'):
        p = model.predict_proba(X)[:, 1]
    else:
        p = model.predict(X)
    return np.clip(p, clip[0], clip[1])


def _cross_fit_predict(model, X, y, n_folds, method='predict'):
    """Out-of-fold predictions via cross-fitting."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    preds = np.zeros(len(y))
    for train_idx, test_idx in kf.split(X):
        m = clone(model)
        m.fit(X[train_idx], y[train_idx])
        if method == 'predict_proba':
            preds[test_idx] = m.predict_proba(X[test_idx])[:, 1]
        else:
            preds[test_idx] = m.predict(X[test_idx])
    return preds


def _prepare_data(data, y, treat, covariates):
    """Extract and validate arrays from DataFrame."""
    cols = [y, treat] + covariates
    for c in cols:
        if c not in data.columns:
            raise ValueError(f"Column '{c}' not found in data")
    clean = data[cols].dropna()
    Y = clean[y].values.astype(float)
    D = clean[treat].values.astype(float)
    X = clean[covariates].values.astype(float)
    return Y, D, X, len(Y)


def _bootstrap_se(cate_values, n_bootstrap=200, rng=None):
    """Bootstrap standard error of the ATE from individual CATE estimates."""
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(cate_values)
    boot_means = np.array([
        rng.choice(cate_values, size=n, replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    return float(np.std(boot_means, ddof=1))


# ======================================================================
# S-Learner
# ======================================================================

class SLearner:
    """
    S-Learner: single model, treatment as a feature.

    Fits one model mu(X, D) and estimates CATE as:
        tau(x) = mu(x, 1) - mu(x, 0)

    Simple but may under-regularise the treatment effect when
    the treatment variable is just one of many features.

    Parameters
    ----------
    model : sklearn estimator, optional
        Outcome model mu(X, D). Default: GradientBoostingRegressor.
    """

    def __init__(self, model=None):
        self.model = model if model is not None else _default_outcome_model()
        self._fitted = False

    def fit(self, X, Y, D):
        """Fit mu(X, D)."""
        X, Y, D = np.asarray(X), np.asarray(Y).ravel(), np.asarray(D).ravel()
        XD = np.column_stack([X, D])
        self._model = clone(self.model)
        self._model.fit(XD, Y)
        self._n_features = X.shape[1]
        self._fitted = True
        return self

    def effect(self, X):
        """Estimate CATE: mu(X,1) - mu(X,0)."""
        X = np.asarray(X)
        X1 = np.column_stack([X, np.ones(len(X))])
        X0 = np.column_stack([X, np.zeros(len(X))])
        return self._model.predict(X1) - self._model.predict(X0)


# ======================================================================
# T-Learner
# ======================================================================

class TLearner:
    """
    T-Learner: separate models for each treatment arm.

    Fits mu_1(X) on treated units and mu_0(X) on controls:
        tau(x) = mu_1(x) - mu_0(x)

    Simple and flexible but can suffer from regularisation imbalance
    when treatment/control group sizes differ substantially.

    Parameters
    ----------
    model_0 : sklearn estimator, optional
        Control outcome model. Default: GradientBoostingRegressor.
    model_1 : sklearn estimator, optional
        Treated outcome model. Default: same type as model_0.
    """

    def __init__(self, model_0=None, model_1=None):
        self.model_0 = model_0 if model_0 is not None else _default_outcome_model()
        self.model_1 = model_1 if model_1 is not None else clone(self.model_0)
        self._fitted = False

    def fit(self, X, Y, D):
        X, Y, D = np.asarray(X), np.asarray(Y).ravel(), np.asarray(D).ravel()
        mask1 = D == 1
        mask0 = D == 0

        self._mu0 = clone(self.model_0)
        self._mu1 = clone(self.model_1)
        self._mu0.fit(X[mask0], Y[mask0])
        self._mu1.fit(X[mask1], Y[mask1])
        self._fitted = True
        return self

    def effect(self, X):
        X = np.asarray(X)
        return self._mu1.predict(X) - self._mu0.predict(X)


# ======================================================================
# X-Learner
# ======================================================================

class XLearner:
    """
    X-Learner (Kunzel et al. 2019).

    Two-stage procedure:
    1. Fit mu_0, mu_1 (T-Learner first stage).
    2. Impute individual treatment effects:
       - For treated: D1_i = Y_i - mu_0(X_i)
       - For controls: D0_i = mu_1(X_i) - Y_i
    3. Fit CATE models tau_1(X) on D1 and tau_0(X) on D0.
    4. Combine: tau(x) = e(x)*tau_0(x) + (1-e(x))*tau_1(x)
       where e(x) is the propensity score.

    Particularly effective when treatment/control groups are
    very unbalanced.

    Parameters
    ----------
    model_0 : sklearn estimator, optional
        Control outcome model.
    model_1 : sklearn estimator, optional
        Treated outcome model.
    cate_model_0 : sklearn estimator, optional
        CATE model for control-side imputed effects.
    cate_model_1 : sklearn estimator, optional
        CATE model for treated-side imputed effects.
    propensity_model : sklearn estimator, optional
        Model for e(x) = P(D=1|X).
    """

    def __init__(
        self,
        model_0=None,
        model_1=None,
        cate_model_0=None,
        cate_model_1=None,
        propensity_model=None,
    ):
        self.model_0 = model_0 if model_0 is not None else _default_outcome_model()
        self.model_1 = model_1 if model_1 is not None else clone(self.model_0)
        self.cate_model_0 = cate_model_0 if cate_model_0 is not None else _default_cate_model()
        self.cate_model_1 = cate_model_1 if cate_model_1 is not None else clone(self.cate_model_0)
        self.propensity_model = propensity_model if propensity_model is not None else _default_propensity_model()
        self._fitted = False

    def fit(self, X, Y, D):
        X, Y, D = np.asarray(X), np.asarray(Y).ravel(), np.asarray(D).ravel()
        mask1 = D == 1
        mask0 = D == 0

        # Stage 1: outcome models
        self._mu0 = clone(self.model_0)
        self._mu1 = clone(self.model_1)
        self._mu0.fit(X[mask0], Y[mask0])
        self._mu1.fit(X[mask1], Y[mask1])

        # Stage 2: imputed treatment effects
        D1 = Y[mask1] - self._mu0.predict(X[mask1])  # treated imputation
        D0 = self._mu1.predict(X[mask0]) - Y[mask0]   # control imputation

        self._tau1 = clone(self.cate_model_1)
        self._tau0 = clone(self.cate_model_0)
        self._tau1.fit(X[mask1], D1)
        self._tau0.fit(X[mask0], D0)

        # Propensity score
        self._prop = clone(self.propensity_model)
        self._prop.fit(X, D)

        self._fitted = True
        return self

    def effect(self, X):
        X = np.asarray(X)
        e = _get_propensity(self._prop, X)
        tau0 = self._tau0.predict(X)
        tau1 = self._tau1.predict(X)
        return e * tau0 + (1 - e) * tau1


# ======================================================================
# R-Learner
# ======================================================================

class RLearner:
    """
    R-Learner (Nie & Wager 2021).

    Based on the Robinson (1988) decomposition:
        Y - m(X) = tau(X) * (D - e(X)) + epsilon

    Estimates nuisance functions m(X) = E[Y|X] and e(X) = E[D|X]
    via cross-fitting, then minimises the R-loss:
        L(tau) = sum_i [ (Y_i - m_hat(X_i)) - tau(X_i)*(D_i - e_hat(X_i)) ]^2

    Achieves quasi-oracle rates under mild conditions.

    Parameters
    ----------
    outcome_model : sklearn estimator, optional
        Model for m(X) = E[Y|X].
    propensity_model : sklearn estimator, optional
        Model for e(X) = P(D=1|X).
    cate_model : sklearn estimator, optional
        Model for tau(X). Fit on pseudo-outcome.
    n_folds : int, default 5
        Cross-fitting folds for nuisance estimation.
    """

    def __init__(
        self,
        outcome_model=None,
        propensity_model=None,
        cate_model=None,
        n_folds=5,
    ):
        self.outcome_model = outcome_model if outcome_model is not None else _default_outcome_model()
        self.propensity_model = propensity_model if propensity_model is not None else _default_propensity_model()
        self.cate_model = cate_model if cate_model is not None else _default_cate_model()
        self.n_folds = n_folds
        self._fitted = False

    def fit(self, X, Y, D):
        X, Y, D = np.asarray(X), np.asarray(Y).ravel(), np.asarray(D).ravel()

        # Cross-fit nuisance
        m_hat = _cross_fit_predict(self.outcome_model, X, Y, self.n_folds)
        e_hat = _cross_fit_predict(
            self.propensity_model, X, D, self.n_folds, method='predict_proba'
        )
        e_hat = np.clip(e_hat, 0.01, 0.99)

        # Residuals
        Y_res = Y - m_hat
        D_res = D - e_hat

        # R-learner pseudo-outcome: Y_res / D_res
        # Weighted regression: minimise sum (Y_res - tau(X)*D_res)^2
        # Equivalent to fitting tau(X) on pseudo_Y = Y_res / D_res
        # with sample weights w = D_res^2
        weights = D_res ** 2
        pseudo_Y = np.where(np.abs(D_res) > 1e-6, Y_res / D_res, 0.0)

        self._cate = clone(self.cate_model)
        self._cate.fit(X, pseudo_Y, sample_weight=weights)

        self._fitted = True
        return self

    def effect(self, X):
        X = np.asarray(X)
        return self._cate.predict(X)


# ======================================================================
# DR-Learner
# ======================================================================

class DRLearner:
    """
    DR-Learner (Kennedy 2023): doubly robust CATE estimation.

    Constructs the doubly robust pseudo-outcome:
        phi(X) = mu_1(X) - mu_0(X)
                 + D*(Y - mu_1(X)) / e(X)
                 - (1-D)*(Y - mu_0(X)) / (1-e(X))

    Then regresses phi on X to obtain tau(X).

    Achieves oracle rates and is robust to mis-specification
    of either the outcome or propensity model (but not both).

    Parameters
    ----------
    outcome_model : sklearn estimator, optional
        Model for mu_d(X) = E[Y|X, D=d].
    propensity_model : sklearn estimator, optional
        Model for e(X) = P(D=1|X).
    cate_model : sklearn estimator, optional
        Final-stage model for tau(X).
    n_folds : int, default 5
        Cross-fitting folds for nuisance estimation.
    """

    def __init__(
        self,
        outcome_model=None,
        propensity_model=None,
        cate_model=None,
        n_folds=5,
    ):
        self.outcome_model = outcome_model if outcome_model is not None else _default_outcome_model()
        self.propensity_model = propensity_model if propensity_model is not None else _default_propensity_model()
        self.cate_model = cate_model if cate_model is not None else _default_cate_model()
        self.n_folds = n_folds
        self._fitted = False

    def fit(self, X, Y, D):
        X, Y, D = np.asarray(X), np.asarray(Y).ravel(), np.asarray(D).ravel()
        n = len(Y)
        mask1 = D == 1
        mask0 = D == 0

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        mu1_hat = np.zeros(n)
        mu0_hat = np.zeros(n)
        e_hat = np.zeros(n)

        for train_idx, test_idx in kf.split(X):
            X_tr, D_tr, Y_tr = X[train_idx], D[train_idx], Y[train_idx]
            X_te = X[test_idx]

            # Outcome models (fit on respective treatment arm)
            m1 = clone(self.outcome_model)
            m0 = clone(self.outcome_model)

            tr_mask1 = D_tr == 1
            tr_mask0 = D_tr == 0

            if tr_mask1.sum() > 0:
                m1.fit(X_tr[tr_mask1], Y_tr[tr_mask1])
                mu1_hat[test_idx] = m1.predict(X_te)
            if tr_mask0.sum() > 0:
                m0.fit(X_tr[tr_mask0], Y_tr[tr_mask0])
                mu0_hat[test_idx] = m0.predict(X_te)

            # Propensity
            prop = clone(self.propensity_model)
            prop.fit(X_tr, D_tr)
            e_hat[test_idx] = _get_propensity(prop, X_te)

        # DR pseudo-outcome
        phi = (
            mu1_hat - mu0_hat
            + D * (Y - mu1_hat) / e_hat
            - (1 - D) * (Y - mu0_hat) / (1 - e_hat)
        )

        # Final CATE model
        self._cate = clone(self.cate_model)
        self._cate.fit(X, phi)

        # Store pseudo-outcomes for diagnostics
        self._pseudo_outcomes = phi

        self._fitted = True
        return self

    def effect(self, X):
        X = np.asarray(X)
        return self._cate.predict(X)


# ======================================================================
# High-level API
# ======================================================================

def metalearner(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    learner: str = 'dr',
    outcome_model: Optional[Any] = None,
    propensity_model: Optional[Any] = None,
    cate_model: Optional[Any] = None,
    n_folds: int = 5,
    n_bootstrap: int = 200,
    alpha: float = 0.05,
) -> CausalResult:
    """
    Estimate heterogeneous treatment effects using meta-learners.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    y : str
        Outcome variable.
    treat : str
        Binary treatment variable (0/1).
    covariates : list of str
        Covariate / effect modifier variables.
    learner : str, default 'dr'
        Meta-learner type: 's', 't', 'x', 'r', or 'dr'.
    outcome_model : sklearn estimator, optional
        Custom ML model for outcome nuisance.
    propensity_model : sklearn estimator, optional
        Custom propensity score model (used by X/R/DR learners).
    cate_model : sklearn estimator, optional
        Custom model for final CATE stage (R/DR learners).
    n_folds : int, default 5
        Cross-fitting folds for nuisance estimation (R/DR learners).
    n_bootstrap : int, default 200
        Bootstrap iterations for ATE standard error.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    CausalResult
        Result with ATE estimate, SE, CI, p-value, and individual
        CATE predictions accessible via ``result.model_info['cate']``.

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.metalearner(df, y='wage', treat='training',
    ...                         covariates=['age', 'edu', 'exp'])
    >>> print(result.summary())

    >>> # Use X-Learner with custom models
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> result = sp.metalearner(df, y='wage', treat='training',
    ...                         covariates=['age', 'edu'],
    ...                         learner='x',
    ...                         outcome_model=RandomForestRegressor())

    >>> # Access individual CATE predictions
    >>> cate = result.model_info['cate']  # array of per-unit effects
    """
    Y, D, X, n = _prepare_data(data, y, treat, covariates)

    # Validate binary treatment
    unique_d = np.unique(D)
    if not (len(unique_d) == 2 and set(unique_d) == {0.0, 1.0}):
        raise ValueError(
            f"Treatment must be binary (0/1), got unique values: {unique_d}"
        )

    learner = learner.lower()
    valid = {'s', 't', 'x', 'r', 'dr'}
    if learner not in valid:
        raise ValueError(
            f"learner must be one of {valid}, got '{learner}'"
        )

    # Build and fit the learner
    if learner == 's':
        est = SLearner(model=outcome_model)
    elif learner == 't':
        est = TLearner(
            model_0=outcome_model,
            model_1=clone(outcome_model) if outcome_model is not None else None,
        )
    elif learner == 'x':
        est = XLearner(
            model_0=outcome_model,
            model_1=clone(outcome_model) if outcome_model is not None else None,
            cate_model_0=cate_model,
            cate_model_1=clone(cate_model) if cate_model is not None else None,
            propensity_model=propensity_model,
        )
    elif learner == 'r':
        est = RLearner(
            outcome_model=outcome_model,
            propensity_model=propensity_model,
            cate_model=cate_model,
            n_folds=n_folds,
        )
    else:  # 'dr'
        est = DRLearner(
            outcome_model=outcome_model,
            propensity_model=propensity_model,
            cate_model=cate_model,
            n_folds=n_folds,
        )

    est.fit(X, Y, D)
    cate = est.effect(X)

    # Aggregate: ATE = mean(CATE)
    ate = float(np.mean(cate))

    # Standard error for ATE
    # DR-Learner: use analytic SE from influence function (semiparametric efficient)
    # Others: bootstrap SE
    if learner == 'dr' and hasattr(est, '_pseudo_outcomes'):
        phi = est._pseudo_outcomes
        se = float(np.std(phi, ddof=1) / np.sqrt(n))
    else:
        rng = np.random.default_rng(42)
        se = _bootstrap_se(cate, n_bootstrap=n_bootstrap, rng=rng)

    # Inference
    if se > 0:
        t_stat = ate / se
        pvalue = float(2 * (1 - stats.norm.cdf(abs(t_stat))))
    else:
        pvalue = 0.0

    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = (ate - z_crit * se, ate + z_crit * se)

    learner_names = {
        's': 'S-Learner', 't': 'T-Learner', 'x': 'X-Learner',
        'r': 'R-Learner', 'dr': 'DR-Learner',
    }

    model_info = {
        'learner': learner_names[learner],
        'n_covariates': len(covariates),
        'n_folds': n_folds if learner in ('r', 'dr') else None,
        'n_bootstrap': n_bootstrap,
        'se_method': 'influence_function' if (learner == 'dr' and hasattr(est, '_pseudo_outcomes')) else 'bootstrap',
        'covariates': covariates,
        '_estimator': est,
        'cate': cate,
        'cate_mean': float(np.mean(cate)),
        'cate_median': float(np.median(cate)),
        'cate_std': float(np.std(cate)),
        'cate_q25': float(np.percentile(cate, 25)),
        'cate_q75': float(np.percentile(cate, 75)),
        'n_treated': int(np.sum(D == 1)),
        'n_control': int(np.sum(D == 0)),
    }

    _result = CausalResult(
        method=f'Meta-Learner ({learner_names[learner]})',
        estimand='ATE',
        estimate=ate,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        detail=None,
        model_info=model_info,
        _citation_key='metalearner',
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov
        _attach_prov(
            _result,
            function="sp.metalearner",
            params={
                "y": y, "treat": treat,
                "covariates": list(covariates),
                "learner": learner,
                "n_folds": n_folds, "n_bootstrap": n_bootstrap,
                "alpha": alpha,
                "outcome_model": type(outcome_model).__name__
                                  if outcome_model is not None else None,
                "propensity_model": type(propensity_model).__name__
                                     if propensity_model is not None else None,
                "cate_model": type(cate_model).__name__
                               if cate_model is not None else None,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


# Register citation
CausalResult._CITATIONS['metalearner'] = (
    "@article{kunzel2019metalearners,\n"
    "  title={Metalearners for Estimating Heterogeneous Treatment Effects "
    "using Machine Learning},\n"
    "  author={K{\\\"u}nzel, S{\\\"o}ren R and Seetharam, Jasjeet S and "
    "Liang, Peter and Athey, Susan},\n"
    "  journal={Proceedings of the National Academy of Sciences},\n"
    "  volume={116},\n"
    "  number={10},\n"
    "  pages={4156--4165},\n"
    "  year={2019},\n"
    "  publisher={National Academy of Sciences}\n"
    "}\n\n"
    "@article{nie2021quasi,\n"
    "  title={Quasi-oracle estimation of heterogeneous treatment effects},\n"
    "  author={Nie, Xinkun and Wager, Stefan},\n"
    "  journal={Biometrika},\n"
    "  volume={108},\n"
    "  number={2},\n"
    "  pages={299--319},\n"
    "  year={2021},\n"
    "  publisher={Oxford University Press}\n"
    "}\n\n"
    "@article{kennedy2023towards,\n"
    "  title={Towards optimal doubly robust estimation of heterogeneous "
    "causal effects},\n"
    "  author={Kennedy, Edward H},\n"
    "  journal={Electronic Journal of Statistics},\n"
    "  volume={17},\n"
    "  number={2},\n"
    "  pages={3008--3049},\n"
    "  year={2023}\n"
    "}"
)
