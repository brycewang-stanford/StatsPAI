"""
Double/Debiased Machine Learning (Chernozhukov et al. 2018).

Implements the partially linear model family:
    Y = theta * D + g(X) + eps
    D = m(X) + v

where g(.) and m(.) are estimated via cross-fitting with any sklearn-
compatible learner, and theta is the causal parameter of interest.

Supports:
- Partially linear regression (PLR)               — continuous/binary D, no IV
- Interactive regression model (IRM)              — binary D, ATE via AIPW
- Partially linear IV (PLIV)                      — endogenous D with instrument Z
- Interactive IV model (IIVM)                     — binary D, binary Z, LATE via Wald
- Multiple cross-fitting splits for median aggregation

References
----------
Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
Newey, W., and Robins, J. (2018). "Double/Debiased Machine Learning for
Treatment and Structural Parameters." *Econometrics Journal*, 21(1), C1-C68.
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
    instrument: Optional[Union[str, List[str]]] = None,
    ml_g: Optional[Any] = None,
    ml_m: Optional[Any] = None,
    ml_r: Optional[Any] = None,
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
        Treatment variable (continuous or binary).
    covariates : list of str
        High-dimensional controls (nuisance covariates X).
    model : str, default 'plr'
        DML model:

        - ``'plr'`` : partially linear (continuous or binary D)
        - ``'irm'`` : interactive regression model (binary D; ATE via AIPW)
        - ``'pliv'`` : partially linear IV (endogenous D with instrument(s) Z)
        - ``'iivm'`` : interactive IV (binary D, binary Z; LATE via Wald ratio)

    instrument : str, optional
        Scalar instrument variable Z. **Required when**
        ``model in {'pliv', 'iivm'}``.
        For many-instrument settings, project Z onto a scalar index
        (e.g. first-stage linear combination) before passing.
        A list with ``len>1`` is rejected to avoid silently dropping
        instruments.
    ml_g : sklearn estimator, optional
        ML model for outcome nuisance ``E[Y|X]``
        (default: ``GradientBoostingRegressor``).
    ml_m : sklearn estimator, optional
        ML model for treatment nuisance ``E[D|X]``. For IRM, a classifier
        is used.
    ml_r : sklearn estimator, optional
        ML model for instrument reduced form ``E[Z|X]`` (PLIV only).
        Defaults to ``ml_g`` type.
    n_folds : int, default 5
        K-fold cross-fitting.
    n_rep : int, default 1
        Repeated splits (median aggregation — robustness against
        split randomness, Chernozhukov et al. 2018 §3.5).
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    CausalResult

    Examples
    --------
    >>> # Partially Linear Regression
    >>> result = sp.dml(df, y='wage', treat='training',
    ...                 covariates=['age', 'edu', 'exp'])

    >>> # Interactive Regression (binary treatment, ATE)
    >>> result = sp.dml(df, y='wage', treat='D', covariates=X_cols,
    ...                 model='irm')

    >>> # Partially Linear IV — endogenous D, instrument Z
    >>> result = sp.dml(df, y='earnings', treat='schooling',
    ...                 covariates=['age', 'father_edu', 'mother_edu'],
    ...                 model='pliv', instrument='quarter_of_birth')

    >>> # Interactive IV — binary D, binary Z (LATE)
    >>> result = sp.dml(df, y='earnings', treat='college',
    ...                 covariates=['age', 'ability', 'parent_edu'],
    ...                 model='iivm', instrument='lottery_win')
    """
    estimator = DoubleML(
        data=data, y=y, treat=treat, covariates=covariates,
        model=model, instrument=instrument,
        ml_g=ml_g, ml_m=ml_m, ml_r=ml_r,
        n_folds=n_folds, n_rep=n_rep, alpha=alpha,
    )
    return estimator.fit()


class DoubleML:
    """Double Machine Learning estimator (PLR / IRM / PLIV)."""

    _VALID_MODELS = ('plr', 'irm', 'pliv', 'iivm')

    def __init__(
        self,
        data: pd.DataFrame,
        y: str,
        treat: str,
        covariates: List[str],
        model: str = 'plr',
        instrument: Optional[Union[str, List[str]]] = None,
        ml_g: Optional[Any] = None,
        ml_m: Optional[Any] = None,
        ml_r: Optional[Any] = None,
        n_folds: int = 5,
        n_rep: int = 1,
        alpha: float = 0.05,
    ):
        self.data = data
        self.y = y
        self.treat = treat
        self.covariates = covariates
        self.model = model.lower()
        # Normalize instrument to list (or None)
        if instrument is None:
            self.instrument = None
        elif isinstance(instrument, str):
            self.instrument = [instrument]
        else:
            self.instrument = list(instrument)
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.alpha = alpha

        self._validate()

        self.ml_g = ml_g if ml_g is not None else self._default_ml_g()
        self.ml_m = ml_m if ml_m is not None else self._default_ml_m()
        self.ml_r = ml_r if ml_r is not None else self._default_ml_r()

    def _validate(self):
        required = [self.y, self.treat] + self.covariates
        if self.instrument is not None:
            required = required + self.instrument
        for col in required:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in data")
        if self.model not in self._VALID_MODELS:
            raise ValueError(
                f"model must be one of {self._VALID_MODELS}, got '{self.model}'"
            )
        iv_models = ('pliv', 'iivm')
        if self.model in iv_models and not self.instrument:
            raise ValueError(
                f"model='{self.model}' requires an 'instrument' argument"
            )
        if self.model not in iv_models and self.instrument is not None:
            raise ValueError(
                f"'instrument' is only valid when model in {iv_models} "
                f"(got model='{self.model}')"
            )
        if self.model in iv_models and self.instrument is not None and len(self.instrument) > 1:
            raise ValueError(
                f"model='{self.model}' currently accepts a single scalar "
                f"instrument; got {len(self.instrument)}: {self.instrument}. "
                f"For multiple instruments, project them onto a scalar index "
                f"(e.g. the OLS first-stage fitted value) and pass that column "
                f"name."
            )
        if self.n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {self.n_folds}")

    def _default_ml_g(self):
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=42,
        )

    def _default_ml_m(self):
        # IRM + IIVM both expect binary-treatment classifier first stage
        if self.model in ('irm', 'iivm'):
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

    def _default_ml_r(self):
        """Instrument reduced-form learner. IIVM needs a classifier for binary Z."""
        if self.model == 'iivm':
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                random_state=42,
            )
        return self._default_ml_g()

    def fit(self) -> CausalResult:
        """Fit the DML model and return results."""
        cols = [self.y, self.treat] + self.covariates
        if self.instrument is not None:
            cols = cols + self.instrument
        clean = self.data[cols].dropna()
        Y = clean[self.y].values.astype(float)
        D = clean[self.treat].values.astype(float)
        X = clean[self.covariates].values.astype(float)
        if self.instrument is not None:
            # First instrument column (scalar projection assumed)
            Z = clean[self.instrument[0]].values.astype(float)
        else:
            Z = None
        n = len(Y)

        if self.model == 'plr':
            thetas, ses = self._fit_plr(Y, D, X, n)
        elif self.model == 'irm':
            thetas, ses = self._fit_irm(Y, D, X, n)
        elif self.model == 'pliv':
            thetas, ses = self._fit_pliv(Y, D, X, Z, n)
        elif self.model == 'iivm':
            thetas, ses = self._fit_iivm(Y, D, X, Z, n)
        else:  # pragma: no cover
            raise RuntimeError(f"Unknown model: {self.model}")

        if len(thetas) == 1:
            theta = thetas[0]
            se = ses[0]
        else:
            theta = float(np.median(thetas))
            se = float(np.median(ses))

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
        if self.model in ('pliv', 'iivm'):
            model_info['ml_r'] = type(self.ml_r).__name__
            model_info['instrument'] = self.instrument[0]

        if self.n_rep > 1:
            model_info['theta_all_reps'] = thetas
            model_info['se_all_reps'] = ses

        return CausalResult(
            method=f'Double ML ({self.model.upper()})',
            estimand='LATE' if self.model in ('pliv', 'iivm') else 'ATE',
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
        """PLR — Y = theta*D + g(X) + eps, D = m(X) + v."""
        from sklearn.base import clone
        from sklearn.model_selection import KFold

        thetas, ses = [], []

        for rep in range(self.n_rep):
            kf = KFold(n_splits=self.n_folds, shuffle=True,
                       random_state=42 + rep)

            y_resid = np.zeros(n)
            d_resid = np.zeros(n)

            for train_idx, test_idx in kf.split(X):
                ml_g = clone(self.ml_g)
                ml_g.fit(X[train_idx], Y[train_idx])
                y_resid[test_idx] = Y[test_idx] - ml_g.predict(X[test_idx])

                ml_m = clone(self.ml_m)
                ml_m.fit(X[train_idx], D[train_idx])
                d_resid[test_idx] = D[test_idx] - ml_m.predict(X[test_idx])

            theta = float(
                np.sum(d_resid * y_resid) / np.sum(d_resid * d_resid)
            )
            psi = y_resid - theta * d_resid
            J = -np.mean(d_resid**2)
            sigma2 = np.mean((d_resid * psi)**2)
            se = float(np.sqrt(sigma2 / (J**2 * n))) if abs(J) > 1e-10 else 0.0
            thetas.append(theta)
            ses.append(se)

        return thetas, ses

    # ------------------------------------------------------------------
    # Interactive Regression Model (IRM)
    # ------------------------------------------------------------------

    def _fit_irm(self, Y, D, X, n):
        """IRM — AIPW for binary D."""
        from sklearn.base import clone
        from sklearn.model_selection import KFold

        thetas, ses = [], []

        for rep in range(self.n_rep):
            kf = KFold(n_splits=self.n_folds, shuffle=True,
                       random_state=42 + rep)

            psi_scores = np.zeros(n)

            for train_idx, test_idx in kf.split(X):
                D_train, D_test = D[train_idx], D[test_idx]
                Y_train, Y_test = Y[train_idx], Y[test_idx]
                X_train, X_test = X[train_idx], X[test_idx]

                ml_g1 = clone(self.ml_g)
                mask1 = D_train == 1
                if mask1.sum() > 0:
                    ml_g1.fit(X_train[mask1], Y_train[mask1])
                    g1_hat = ml_g1.predict(X_test)
                else:
                    g1_hat = np.zeros(len(test_idx))

                ml_g0 = clone(self.ml_g)
                mask0 = D_train == 0
                if mask0.sum() > 0:
                    ml_g0.fit(X_train[mask0], Y_train[mask0])
                    g0_hat = ml_g0.predict(X_test)
                else:
                    g0_hat = np.zeros(len(test_idx))

                ml_m = clone(self.ml_m)
                ml_m.fit(X_train, D_train)
                if hasattr(ml_m, 'predict_proba'):
                    m_hat = ml_m.predict_proba(X_test)[:, 1]
                else:
                    m_hat = ml_m.predict(X_test)
                m_hat = np.clip(m_hat, 0.01, 0.99)

                psi_scores[test_idx] = (
                    g1_hat - g0_hat
                    + D_test * (Y_test - g1_hat) / m_hat
                    - (1 - D_test) * (Y_test - g0_hat) / (1 - m_hat)
                )

            theta = float(np.mean(psi_scores))
            se = float(np.std(psi_scores, ddof=1) / np.sqrt(n))
            thetas.append(theta)
            ses.append(se)

        return thetas, ses

    # ------------------------------------------------------------------
    # Partially Linear IV (PLIV)  —  Chernozhukov et al. 2018, §4.2
    # ------------------------------------------------------------------

    def _fit_pliv(self, Y, D, X, Z, n):
        """
        PLIV — Y = theta*D + g(X) + eps,   E[eps | Z,X] = 0.

        Neyman-orthogonal score:
            psi = ( Y - g(X) - theta*(D - m(X)) ) * ( Z - r(X) )
        where
            g(X) = E[Y|X], m(X) = E[D|X], r(X) = E[Z|X].

        DML1 estimator:
            theta = sum( y_tilde * z_tilde ) / sum( d_tilde * z_tilde )
        with y_tilde = Y-g_hat, d_tilde = D-m_hat, z_tilde = Z-r_hat.

        Asymptotic variance (Chernozhukov 2018, Eq. 4.12):
            sqrt(n) (θ̂ - θ₀)  →  N(0, σ² / J²)
        with σ² = E[ψ²],  J = E[(Z - r(X))(D - m(X))] = E[z̃ · d̃],
        i.e.  Var(θ̂) ≈ σ² / (J² · n)   and   SE = √(σ² / (J² · n)).
        """
        from sklearn.base import clone
        from sklearn.model_selection import KFold

        thetas, ses = [], []

        for rep in range(self.n_rep):
            kf = KFold(n_splits=self.n_folds, shuffle=True,
                       random_state=42 + rep)

            y_resid = np.zeros(n)
            d_resid = np.zeros(n)
            z_resid = np.zeros(n)

            for train_idx, test_idx in kf.split(X):
                # g(X) = E[Y|X]
                ml_g = clone(self.ml_g)
                ml_g.fit(X[train_idx], Y[train_idx])
                y_resid[test_idx] = Y[test_idx] - ml_g.predict(X[test_idx])

                # m(X) = E[D|X]
                ml_m = clone(self.ml_m)
                ml_m.fit(X[train_idx], D[train_idx])
                d_resid[test_idx] = D[test_idx] - ml_m.predict(X[test_idx])

                # r(X) = E[Z|X]  — instrument reduced form
                ml_r = clone(self.ml_r)
                ml_r.fit(X[train_idx], Z[train_idx])
                z_resid[test_idx] = Z[test_idx] - ml_r.predict(X[test_idx])

            denom = float(np.sum(z_resid * d_resid))
            # Scale-aware weak-instrument guard: compare the covariance
            # sum to the geometric mean of z̃ and d̃ scales. A truly weak
            # instrument satisfies |E[z̃·d̃]| ≪ sd(z̃)·sd(d̃).
            scale = (float(np.sqrt(np.sum(z_resid**2) *
                                   np.sum(d_resid**2))))
            if abs(denom) < 1e-6 * max(scale, 1.0):
                raise RuntimeError(
                    f"Degenerate PLIV first stage: residualized "
                    f"instrument is (near-)orthogonal to residualized "
                    f"treatment. |E[z̃·d̃]| = {abs(denom):.2e}, "
                    f"scale = {scale:.2e}. Instrument likely weak or "
                    f"irrelevant conditional on X."
                )
            theta = float(np.sum(z_resid * y_resid) / denom)

            # Variance via Neyman-orthogonal influence function
            psi = (y_resid - theta * d_resid) * z_resid
            J = -np.mean(z_resid * d_resid)  # -E[z_tilde * d_tilde]
            sigma2 = np.mean(psi**2)
            se = float(np.sqrt(sigma2 / (J**2 * n))) if abs(J) > 1e-10 else 0.0

            thetas.append(theta)
            ses.append(se)

        return thetas, ses

    # ------------------------------------------------------------------
    # Interactive IV Model (IIVM) — Chernozhukov et al. 2018, §5
    # ------------------------------------------------------------------

    def _fit_iivm(self, Y, D, X, Z, n):
        """
        IIVM — binary D, binary Z. Estimates LATE (compliers).

        Neyman-orthogonal score (DR / efficient influence function):

            psi_a(W; g, m) =
                g(1, X) - g(0, X)
                + Z*(Y - g(1, X))/m(X)
                - (1-Z)*(Y - g(0, X))/(1 - m(X))

            psi_b(W; r, m) =
                r(1, X) - r(0, X)
                + Z*(D - r(1, X))/m(X)
                - (1-Z)*(D - r(0, X))/(1 - m(X))

            theta_LATE = E[psi_a] / E[psi_b]

        where g(z, X) = E[Y|Z=z, X], r(z, X) = E[D|Z=z, X], m(X) = P(Z=1|X).
        Variance via delta method on the ratio.
        """
        from sklearn.base import clone
        from sklearn.model_selection import KFold

        if not set(np.unique(Z)).issubset({0, 1}):
            raise ValueError(
                "model='iivm' requires a binary (0/1) instrument Z. "
                "For continuous instruments use model='pliv'."
            )
        if not set(np.unique(D)).issubset({0, 1}):
            raise ValueError(
                "model='iivm' requires a binary (0/1) treatment D. "
                "For continuous treatments use model='pliv'."
            )

        thetas, ses = [], []

        for rep in range(self.n_rep):
            kf = KFold(n_splits=self.n_folds, shuffle=True,
                       random_state=42 + rep)

            g1 = np.zeros(n)   # E[Y|Z=1, X]
            g0 = np.zeros(n)   # E[Y|Z=0, X]
            r1 = np.zeros(n)   # E[D|Z=1, X]
            r0 = np.zeros(n)   # E[D|Z=0, X]
            m_hat = np.zeros(n)  # P(Z=1|X)

            for train_idx, test_idx in kf.split(X):
                X_tr, X_te = X[train_idx], X[test_idx]
                Y_tr, Z_tr, D_tr = Y[train_idx], Z[train_idx], D[train_idx]

                # g(1, X) = E[Y | Z=1, X]
                mask_z1 = Z_tr == 1
                ml_g1 = clone(self.ml_g)
                if mask_z1.sum() > 1:
                    ml_g1.fit(X_tr[mask_z1], Y_tr[mask_z1])
                    g1[test_idx] = ml_g1.predict(X_te)
                else:
                    g1[test_idx] = float(np.mean(Y_tr[mask_z1])) if mask_z1.any() else 0.0

                # g(0, X) = E[Y | Z=0, X]
                mask_z0 = Z_tr == 0
                ml_g0 = clone(self.ml_g)
                if mask_z0.sum() > 1:
                    ml_g0.fit(X_tr[mask_z0], Y_tr[mask_z0])
                    g0[test_idx] = ml_g0.predict(X_te)
                else:
                    g0[test_idx] = float(np.mean(Y_tr[mask_z0])) if mask_z0.any() else 0.0

                # r(1, X) = E[D | Z=1, X] — first-stage compliance among Z=1
                ml_r1 = clone(self.ml_r)
                if mask_z1.sum() > 1 and len(np.unique(D_tr[mask_z1])) > 1:
                    ml_r1.fit(X_tr[mask_z1], D_tr[mask_z1])
                    if hasattr(ml_r1, 'predict_proba'):
                        r1[test_idx] = ml_r1.predict_proba(X_te)[:, 1]
                    else:
                        r1[test_idx] = ml_r1.predict(X_te)
                else:
                    r1[test_idx] = float(np.mean(D_tr[mask_z1])) if mask_z1.any() else 0.0

                # r(0, X) = E[D | Z=0, X]
                ml_r0 = clone(self.ml_r)
                if mask_z0.sum() > 1 and len(np.unique(D_tr[mask_z0])) > 1:
                    ml_r0.fit(X_tr[mask_z0], D_tr[mask_z0])
                    if hasattr(ml_r0, 'predict_proba'):
                        r0[test_idx] = ml_r0.predict_proba(X_te)[:, 1]
                    else:
                        r0[test_idx] = ml_r0.predict(X_te)
                else:
                    r0[test_idx] = float(np.mean(D_tr[mask_z0])) if mask_z0.any() else 0.0

                # m(X) = P(Z=1 | X) — instrument propensity
                ml_m = clone(self.ml_m)
                ml_m.fit(X_tr, Z_tr)
                if hasattr(ml_m, 'predict_proba'):
                    m_hat[test_idx] = ml_m.predict_proba(X_te)[:, 1]
                else:
                    m_hat[test_idx] = ml_m.predict(X_te)

            m_hat = np.clip(m_hat, 0.01, 0.99)
            # First-stage probabilities must stay in (0,1) for stability
            r1 = np.clip(r1, 1e-4, 1 - 1e-4)
            r0 = np.clip(r0, 1e-4, 1 - 1e-4)

            psi_a = (
                g1 - g0
                + Z * (Y - g1) / m_hat
                - (1 - Z) * (Y - g0) / (1 - m_hat)
            )
            psi_b = (
                r1 - r0
                + Z * (D - r1) / m_hat
                - (1 - Z) * (D - r0) / (1 - m_hat)
            )

            num = float(np.mean(psi_a))
            den = float(np.mean(psi_b))
            # Weak-instrument guard
            if abs(den) < 1e-6:
                raise RuntimeError(
                    f"Degenerate IIVM first stage: E[psi_b] ≈ {den:.2e}. "
                    "Compliance (first-stage effect of Z on D) is near zero; "
                    "LATE is not identified."
                )
            theta = num / den

            # Delta-method variance on the ratio
            # Var(theta) ≈ Var(psi_a - theta * psi_b) / (n * den^2)
            influence = (psi_a - theta * psi_b) / den
            sigma2 = float(np.var(influence, ddof=1))
            se = float(np.sqrt(sigma2 / n))

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
