"""
Meta-Learners for heterogeneous treatment effect (CATE) estimation.

Implements five canonical meta-learners that decompose CATE estimation
into standard supervised-learning sub-problems. Citations live in
``paper.bib``; refer to the bib keys below for the canonical record:

- ``@kunzel2019metalearners`` — Künzel, Sekhon, Bickel & Yu (2019),
  *PNAS* 116(10), 4156-4165.
- ``@kennedy2023towards``     — Kennedy (2023), *EJS* 17(2).
- ``@nie2021quasi``           — Nie & Wager (2021), *Biometrika* 108(2).

All learners accept any scikit-learn compatible estimators for the
nuisance and CATE stages.

Supported learners
------------------
- **S-Learner** : single model with treatment as feature
- **T-Learner** : separate models per treatment arm
- **X-Learner** : two-stage imputed treatment effect (Künzel et al.)
- **R-Learner** : Robinson decomposition + loss minimisation (Nie & Wager)
- **DR-Learner**: doubly robust pseudo-outcome regression (Kennedy)
"""

from typing import Optional, List, Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from scipy import stats

# sklearn is imported lazily inside the functions/methods that need it so
# that ``import statspai`` doesn't pull ~245 sklearn submodules through
# this file when the user never touches metalearners.

from ..core.results import CausalResult


# ======================================================================
# Helpers
# ======================================================================

def _default_outcome_model():
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )


def _default_propensity_model():
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )


def _default_cate_model():
    from sklearn.ensemble import GradientBoostingRegressor
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
    from sklearn.base import clone
    from sklearn.model_selection import KFold
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


def _cross_fit_aipw_phi(
    X, Y, D, outcome_model, propensity_model, n_folds=5, clip=(0.01, 0.99),
    seed=42,
):
    """Cross-fit AIPW (DR) pseudo-outcome :math:`\\varphi_i`.

    Returns
    -------
    phi : ndarray of shape (n,)
        :math:`\\varphi_i = \\hat\\mu_1(X_i) - \\hat\\mu_0(X_i) +
        D_i (Y_i - \\hat\\mu_1(X_i)) / \\hat e(X_i) -
        (1-D_i)(Y_i - \\hat\\mu_0(X_i)) / (1 - \\hat e(X_i))`.
    diagnostics : dict
        Underlying nuisance arrays and clip counts so the caller can
        surface overlap warnings.

    Notes
    -----
    This is the **semiparametric efficient** estimating function for
    :math:`E[Y(1) - Y(0)]` (van der Laan & Robins 2003, Kennedy 2023).
    :math:`\\hat{\\rm ATE} = \\bar\\varphi` and its asymptotic SE is
    :math:`\\sigma(\\varphi)/\\sqrt{n}` regardless of which CATE
    estimator (S/T/X/R/DR) the user has chosen for heterogeneity.
    """
    from sklearn.base import clone
    from sklearn.model_selection import KFold
    n = len(Y)
    mu1_hat = np.zeros(n)
    mu0_hat = np.zeros(n)
    e_hat = np.zeros(n)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        X_tr, Y_tr, D_tr = X[tr], Y[tr], D[tr]
        X_te = X[te]
        m1 = clone(outcome_model)
        m0 = clone(outcome_model)
        tr_mask1 = D_tr == 1
        tr_mask0 = D_tr == 0
        # Fit each arm if both arms present in the training fold; fall
        # back to the arm mean otherwise. With pre-validated data and
        # n_folds=5 this fallback essentially never triggers, but the
        # branch keeps the helper safe on tiny / lopsided samples.
        if tr_mask1.sum() >= 2:
            m1.fit(X_tr[tr_mask1], Y_tr[tr_mask1])
            mu1_hat[te] = m1.predict(X_te)
        else:
            mu1_hat[te] = float(np.mean(Y_tr[tr_mask1])) if tr_mask1.any() else 0.0
        if tr_mask0.sum() >= 2:
            m0.fit(X_tr[tr_mask0], Y_tr[tr_mask0])
            mu0_hat[te] = m0.predict(X_te)
        else:
            mu0_hat[te] = float(np.mean(Y_tr[tr_mask0])) if tr_mask0.any() else 0.0
        prop = clone(propensity_model)
        prop.fit(X_tr, D_tr)
        e_hat[te] = _get_propensity(prop, X_te, clip=clip)
    e_clip = np.clip(e_hat, clip[0], clip[1])
    n_clip_lo = int(np.sum(e_hat < clip[0]))
    n_clip_hi = int(np.sum(e_hat > clip[1]))
    phi = (
        mu1_hat - mu0_hat
        + D * (Y - mu1_hat) / e_clip
        - (1 - D) * (Y - mu0_hat) / (1 - e_clip)
    )
    diag = {
        "mu1_hat": mu1_hat,
        "mu0_hat": mu0_hat,
        "e_hat": e_hat,
        "n_clipped_below": n_clip_lo,
        "n_clipped_above": n_clip_hi,
        "clip": clip,
    }
    return phi, diag


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
        from sklearn.base import clone
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
        from sklearn.base import clone
        self.model_0 = model_0 if model_0 is not None else _default_outcome_model()
        self.model_1 = model_1 if model_1 is not None else clone(self.model_0)
        self._fitted = False

    def fit(self, X, Y, D):
        from sklearn.base import clone
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
    X-Learner (Künzel et al. 2019).

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
        from sklearn.base import clone
        self.model_0 = model_0 if model_0 is not None else _default_outcome_model()
        self.model_1 = model_1 if model_1 is not None else clone(self.model_0)
        self.cate_model_0 = cate_model_0 if cate_model_0 is not None else _default_cate_model()
        self.cate_model_1 = cate_model_1 if cate_model_1 is not None else clone(self.cate_model_0)
        self.propensity_model = propensity_model if propensity_model is not None else _default_propensity_model()
        self._fitted = False

    def fit(self, X, Y, D):
        from sklearn.base import clone
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
        from sklearn.base import clone
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
        from sklearn.base import clone
        from sklearn.model_selection import KFold
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

        # DR pseudo-outcome — note that ``e_hat`` is already passed
        # through ``_get_propensity`` which clips to (0.01, 0.99).  We
        # also surface raw clip counts so the wrapper can warn on poor
        # overlap.
        n_clip_lo = int(np.sum(e_hat <= 0.01 + 1e-12))
        n_clip_hi = int(np.sum(e_hat >= 0.99 - 1e-12))
        phi = (
            mu1_hat - mu0_hat
            + D * (Y - mu1_hat) / e_hat
            - (1 - D) * (Y - mu0_hat) / (1 - e_hat)
        )

        # Final CATE model
        self._cate = clone(self.cate_model)
        self._cate.fit(X, phi)

        # Store pseudo-outcomes + diagnostics so the high-level
        # ``metalearner`` wrapper can reuse them for ATE / SE without
        # re-running cross-fitting.
        self._pseudo_outcomes = phi
        self._pseudo_diag = {
            "mu1_hat": mu1_hat,
            "mu0_hat": mu0_hat,
            "e_hat": e_hat,
            "n_clipped_below": n_clip_lo,
            "n_clipped_above": n_clip_hi,
            "clip": (0.01, 0.99),
        }

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
        Cross-fitting folds for nuisance estimation (used by R/DR
        learners and the unified AIPW SE path; see Notes).
    n_bootstrap : int, default 200
        **Deprecated and ignored** as of v1.11.4. Previously the SE for
        S/T/X/R-Learner came from a re-sampling bootstrap of the fitted
        CATE values, which treats τ̂ as fixed and severely
        under-estimates uncertainty. The function now uses the AIPW
        influence function for SE regardless of ``learner=``. The
        argument is kept for backward compatibility and will be removed
        in a future minor release.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    CausalResult
        Result with ATE estimate, SE, CI, p-value, and individual
        CATE predictions accessible via ``result.model_info['cate']``.

    Notes
    -----
    **ATE / SE convention (v1.11.4+).** Regardless of which CATE
    estimator the user selects via ``learner=``, the population ATE and
    its SE are computed via the AIPW (DR) pseudo-outcome:

    .. math::

       \\varphi_i = \\hat\\mu_1(X_i) - \\hat\\mu_0(X_i)
                  + \\frac{D_i (Y_i - \\hat\\mu_1(X_i))}{\\hat e(X_i)}
                  - \\frac{(1-D_i)(Y_i - \\hat\\mu_0(X_i))}{1 - \\hat e(X_i)}

    with :math:`\\hat{\\rm ATE} = \\bar\\varphi`,
    :math:`\\widehat{\\rm SE} = \\sigma(\\varphi)/\\sqrt n`. AIPW is
    the semiparametric-efficient estimating function for
    :math:`E[Y(1) - Y(0)]` (van der Laan & Robins 2003; Kennedy 2023),
    so the SE is valid for *any* CATE estimator. The chosen
    ``learner=`` determines τ̂(X) (heterogeneity); ATE inference is
    learner-independent.

    Prior to v1.11.4, S/T/X/R-Learner used ``mean(τ̂)`` as ATE and a
    re-sampling bootstrap of τ̂ as SE. The bootstrap silently treated
    τ̂ as fixed → systematically too small SEs and severe under-
    coverage. ⚠️ This is a correctness fix; numerical results will
    change for non-DR learners.

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
    from sklearn.base import clone
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

    # ATE estimation + SE — unified AIPW (DR pseudo-outcome) path for ALL
    # learners.  Rationale: the chosen learner determines τ̂(X) (CATE),
    # but the semiparametric-efficient estimating function for the
    # population ATE is always the AIPW score (van der Laan & Robins
    # 2003; Kennedy 2023).  Using mean(τ̂(X)) as ATE plus a re-sampling
    # bootstrap of those τ̂ values — as v1.11.3 and earlier did for
    # S/T/X/R-Learner — silently treats τ̂ as fixed and severely
    # under-estimates the SE.  We now reuse DR-Learner's own pseudo
    # outcomes when available (avoids a second cross-fit) and otherwise
    # build them via :func:`_cross_fit_aipw_phi`.
    if learner == 'dr' and hasattr(est, '_pseudo_outcomes'):
        phi = est._pseudo_outcomes
        aipw_diag = getattr(est, '_pseudo_diag', None)
    else:
        # Use the user-supplied or default outcome / propensity models
        # for a clean, learner-independent AIPW fit.  Without explicit
        # user models this matches the DR-Learner default exactly so
        # results are reproducible across learner= choices.
        _outcome = outcome_model if outcome_model is not None else _default_outcome_model()
        _prop = propensity_model if propensity_model is not None else _default_propensity_model()
        phi, aipw_diag = _cross_fit_aipw_phi(
            X, Y, D, _outcome, _prop, n_folds=n_folds,
        )
    ate = float(np.mean(phi))
    se = float(np.std(phi, ddof=1) / np.sqrt(n))

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

    # Overlap diagnostics + warning when many propensities were clipped
    # (the AIPW score blows up at e≈0 or e≈1, so a heavy clip share is
    # a red flag for identification, not a noise issue).
    n_clip_lo = int(aipw_diag.get("n_clipped_below", 0)) if aipw_diag else 0
    n_clip_hi = int(aipw_diag.get("n_clipped_above", 0)) if aipw_diag else 0
    clip_share = (n_clip_lo + n_clip_hi) / n if n > 0 else 0.0
    if clip_share > 0.05:
        import warnings
        warnings.warn(
            f"sp.metalearner: {n_clip_lo + n_clip_hi}/{n} "
            f"({100 * clip_share:.1f}%) propensity scores hit the "
            f"{aipw_diag.get('clip', (0.01, 0.99))} clip in the AIPW "
            f"score — overlap is poor and the ATE / SE may be biased "
            f"toward the trimmed sample. Inspect "
            f"result.model_info['aipw_diagnostics'] and consider "
            f"sp.overlap_plot() / a more flexible propensity model.",
            UserWarning,
            stacklevel=2,
        )

    model_info = {
        'learner': learner_names[learner],
        'n_covariates': len(covariates),
        'n_folds': n_folds if learner in ('r', 'dr') else None,
        'n_bootstrap': n_bootstrap,
        # All learners now use AIPW (DR pseudo-outcome) for ATE + SE —
        # the chosen learner only governs CATE prediction. See the
        # ``ate_method`` note in the docstring + the v1.11.x migration
        # guide.
        'se_method': 'aipw_influence_function',
        'ate_method': 'aipw_dr_pseudo_outcome',
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
        'aipw_diagnostics': {
            'n_clipped_below': n_clip_lo,
            'n_clipped_above': n_clip_hi,
            'clip_share': float(clip_share),
            'clip': aipw_diag.get('clip', (0.01, 0.99)) if aipw_diag else (0.01, 0.99),
        },
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


# Register citation — kept verbatim in sync with paper.bib (the single source
# of truth per CLAUDE.md §10).  Touching this block requires touching
# paper.bib too; the BibTeX strings below are byte-identical to the entries
# under bib keys @kunzel2019metalearners, @nie2021quasi, @kennedy2023towards.
CausalResult._CITATIONS['metalearner'] = (
    "@article{kunzel2019metalearners,\n"
    "  title={Metalearners for estimating heterogeneous treatment effects "
    "using machine learning},\n"
    "  author={K{\\\"u}nzel, S{\\\"o}ren R. and Sekhon, Jasjeet S. and "
    "Bickel, Peter J. and Yu, Bin},\n"
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
    "  author={Kennedy, Edward H.},\n"
    "  journal={Electronic Journal of Statistics},\n"
    "  volume={17},\n"
    "  number={2},\n"
    "  pages={3008--3049},\n"
    "  year={2023}\n"
    "}"
)
