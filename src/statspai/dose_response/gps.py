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

from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .._aliases import accepts_aliases

# sklearn is imported lazily inside the methods that need it so that
# ``import statspai`` doesn't pull ~245 sklearn submodules through this
# file when the user never touches dose_response. ``BaseEstimator`` only
# appears in type annotations here and is gated behind ``TYPE_CHECKING``.
if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

from .._input_validation import missing_named_columns
from ..core._covariates import expands_categorical_covariates as _expands_categorical
from ..core.results import CausalResult
from ..exceptions import ColumnNotFound, MethodIncompatibility

# ======================================================================
# Public API
# ======================================================================


@accepts_aliases(seed="random_state")
@_expands_categorical("covariates")
def dose_response(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    n_dose_points: int = 20,
    dose_range: Optional[Tuple[float, float]] = None,
    treatment_model: "Optional[BaseEstimator]" = None,
    outcome_model: "Optional[BaseEstimator]" = None,
    n_bootstrap: int = 200,
    alpha: float = 0.05,
    random_state: int = 42,
    method: str = "gps",
    bandwidth: Optional[float] = None,
    n_folds: int = 5,
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
        Model for E[Y|T, GPS]. Default: GBM. The outcome model sees the two
        columns ``[T, GPS]``, so
        ``make_pipeline(PolynomialFeatures(2), LinearRegression())`` with
        ``treatment_model=LinearRegression()`` is the quadratic
        specification of Hirano and Imbens (2004).

        Neither choice is reliable in general. On a linear confounded
        design (``T = 1 + 0.8x + 0.3x2 + e``, ``Y = 0.8T + x + 0.5x2 + u``,
        n = 1000, 30 draws) the default GBM pair is nearly unbiased for the
        25th-75th percentile effect (bias -0.13 against a truth of about
        1.9) but noisy (SD 0.43) and its curve was never monotone, while the
        Hirano-Imbens quadratic is precise (SD 0.07) and badly biased
        (+1.27, close to the unadjusted slope), because ``E[Y | T, GPS]`` is
        not quadratic there. The default is therefore kept, and a call
        with both default models warns. For an average slope with valid
        inference use ``sp.dml(model='plr')``.
    n_bootstrap : int, default 200
        Bootstrap iterations for confidence bands.
    alpha : float, default 0.05
        Significance level.
    random_state : int, default 42
    method : {'gps', 'dr'}, default 'gps'
        ``'gps'``: the generalized-propensity-score regression of Hirano
        and Imbens (2004) with bootstrap bands (the default, unchanged).
        ``'dr'``: the doubly robust curve of Kennedy, Ma, McHugh and Small
        (2017): a local linear regression of the pseudo-outcome
        ``(Y - mu(X, T)) / pi(T | X) * mean_j pi(T | X_j) + mean_j
        mu(X_j, T)`` on ``T``, consistent if either the outcome regression
        ``mu`` (``outcome_model``, on ``[X, T]``; default gradient
        boosting) or the normal treatment density ``pi``
        (``treatment_model`` for its mean, default linear, and a linear
        regression of squared residuals for its variance) is right.
        Nuisances are cross-fitted (``n_folds``); the bandwidth minimises
        the smoother's leave-one-out risk unless ``bandwidth`` is given;
        bands are the robust local-linear sandwich on the pseudo-outcomes
        (no bootstrap), for the smoothed curve. R ``npcausal::ctseff``
        computes the same estimator with other nuisance and smoothing
        choices; the curves agree in shape, not to parity.
    bandwidth : float, optional
        ``method='dr'``: kernel bandwidth in units of the treatment.
    n_folds : int, default 5
        ``method='dr'``: cross-fitting folds (1 = no cross-fitting).

    Returns
    -------
    CausalResult
        detail contains the dose-response curve:
        columns 'dose', 'response', 'se', 'ci_lower', 'ci_upper'.
        model_info contains 'avg_marginal_effect' (derivative).

    Examples
    --------
    >>> import statspai as sp, numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> age = rng.normal(40, 10, n)
    >>> weight = rng.normal(70, 12, n)
    >>> dosage = 0.5 * age + rng.normal(0, 5, n)      # confounded dose
    >>> outcome = 0.2 * dosage + 0.1 * weight + rng.normal(0, 1, n)
    >>> df = pd.DataFrame({"outcome": outcome, "dosage": dosage,
    ...                    "age": age, "weight": weight})
    >>> result = sp.dose_response(df, y="outcome", treat="dosage",
    ...                           covariates=["age", "weight"],
    ...                           n_dose_points=10, n_bootstrap=20)
    >>> curve = result.detail            # dose-response curve
    >>> curve.shape
    (10, 6)
    >>> list(curve.columns)
    ['dose', 'response', 'se', 'ci_lower', 'ci_upper', 'marginal_effect']
    """
    if method not in ("gps", "dr"):
        raise MethodIncompatibility(
            f"dose_response: method must be 'gps' or 'dr'; got {method!r}.",
            recovery_hint="Use method='dr' for the doubly robust curve.",
        )
    if method == "gps" and treatment_model is None and outcome_model is None:
        import warnings

        from ..exceptions import StatsPAIWarning

        warnings.warn(
            StatsPAIWarning(
                "sp.dose_response with its default gradient-boosting models "
                "gives a noisy, often non-monotone curve (on a linear "
                "confounded design with n = 1000 the 25th-75th percentile "
                "effect had SD 0.43 against a truth of 1.9); treat the curve "
                "shape with caution.",
                recovery_hint=(
                    "Pass treatment_model= / outcome_model= suited to the "
                    "design, check result.model_info['curve_monotone'], or "
                    "use sp.dml(model='plr') for an average slope."
                ),
                alternative_functions=["sp.dml", "sp.continuous_did"],
            ),
            stacklevel=2,
        )
    est = DoseResponse(
        data=data,
        y=y,
        treat=treat,
        covariates=covariates,
        n_dose_points=n_dose_points,
        dose_range=dose_range,
        treatment_model=treatment_model,
        outcome_model=outcome_model,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        random_state=random_state,
    )
    est.method = method
    est.bandwidth = bandwidth
    est.n_folds = int(n_folds)
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

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 150
    >>> age = rng.normal(50, 10, n)
    >>> weight = rng.normal(70, 12, n)
    >>> dosage = 0.5 * age + 0.3 * weight + rng.normal(0, 5, n)
    >>> outcome = 0.8 * dosage + 0.2 * age + rng.normal(0, 3, n)
    >>> df = pd.DataFrame({"outcome": outcome, "dosage": dosage,
    ...                    "age": age, "weight": weight})
    >>> est = sp.DoseResponse(df, y="outcome", treat="dosage",
    ...                       covariates=["age", "weight"],
    ...                       n_dose_points=6, n_bootstrap=8, random_state=0)
    >>> res = est.fit()
    >>> isinstance(res, sp.CausalResult)
    True
    >>> list(res.detail.columns)
    ['dose', 'response', 'se', 'ci_lower', 'ci_upper', 'marginal_effect']

    References
    ----------
    [@hirano2004propensity], [@kennedy2017parametric]
    """

    def __init__(
        self,
        data: pd.DataFrame,
        y: str,
        treat: str,
        covariates: List[str],
        n_dose_points: int = 20,
        dose_range: Optional[Tuple[float, float]] = None,
        treatment_model: "Optional[BaseEstimator]" = None,
        outcome_model: "Optional[BaseEstimator]" = None,
        n_bootstrap: int = 200,
        alpha: float = 0.05,
        random_state: int = 42,
    ) -> None:
        from sklearn.ensemble import GradientBoostingRegressor

        self.data = data
        self.y = y
        self.treat = treat
        self.covariates = covariates
        self.n_dose_points = n_dose_points
        self.dose_range = dose_range
        # method='dr' has its own defaults (linear treatment mean, boosted
        # outcome); it must see what the user passed, not the GPS defaults.
        self._user_models = (treatment_model, outcome_model)
        self.treatment_model = (
            treatment_model
            if treatment_model is not None
            else GradientBoostingRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=random_state,
            )
        )
        self.outcome_model = (
            outcome_model
            if outcome_model is not None
            else GradientBoostingRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=random_state,
            )
        )
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self.random_state = random_state

    def fit(self) -> CausalResult:
        """Estimate the dose-response function."""
        cols = [self.y, self.treat] + self.covariates
        missing = [c for c in cols if c not in self.data.columns]
        if missing:
            raise ColumnNotFound.from_missing(
                missing_named_columns(
                    {"y": self.y, "treat": self.treat, "covariates": self.covariates},
                    self.data.columns,
                ),
                self.data.columns,
                message=f"Columns not found in data: {missing}",
            )

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

        if getattr(self, "method", "gps") == "dr":
            return self._fit_dr(Y, T, X, dose_grid, t_lo, t_hi)

        # Estimate dose-response curve
        dr_curve = self._estimate_curve(Y, T, X, dose_grid, n)

        # Bootstrap for confidence bands
        rng = np.random.RandomState(self.random_state)
        boot_curves = np.zeros((self.n_bootstrap, self.n_dose_points))

        for b in range(self.n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            boot_curves[b] = self._estimate_curve(Y[idx], T[idx], X[idx], dose_grid, n)

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
            pvalue = float(2 * sp_stats.norm.sf(abs(z_stat)))
        else:
            pvalue = 0.0
        ci = (effect_iqr - z_crit * se_iqr, effect_iqr + z_crit * se_iqr)

        detail = pd.DataFrame(
            {
                "dose": dose_grid,
                "response": dr_curve,
                "se": dr_se,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "marginal_effect": marginal_effects,
            }
        )

        model_info = {
            "dose_range": (float(t_lo), float(t_hi)),
            "n_dose_points": self.n_dose_points,
            "avg_marginal_effect": avg_marginal_effect,
            "effect_25_to_75": effect_iqr,
            "dose_25": float(dose_grid[idx_25]),
            "dose_75": float(dose_grid[idx_75]),
            "curve_monotone": bool(
                np.all(np.diff(dr_curve) > 0) or np.all(np.diff(dr_curve) < 0)
            ),
        }

        return CausalResult(
            method="Dose-Response (GPS, Hirano & Imbens 2004)",
            estimand="E[Y(t75)] - E[Y(t25)]",
            estimate=effect_iqr,
            se=se_iqr,
            pvalue=pvalue,
            ci=ci,
            alpha=self.alpha,
            n_obs=n,
            detail=detail,
            model_info=model_info,
            _citation_key="dose_response",
        )

    def _fit_dr(
        self,
        Y: np.ndarray,
        T: np.ndarray,
        X: np.ndarray,
        dose_grid: np.ndarray,
        t_lo: float,
        t_hi: float,
    ) -> CausalResult:
        from .dr import dr_curve

        out = dr_curve(
            Y,
            T,
            X,
            dose_grid,
            treatment_model=self._user_models[0],
            outcome_model=self._user_models[1],
            bandwidth=self.bandwidth,
            n_folds=self.n_folds,
            random_state=self.random_state,
            alpha=self.alpha,
        )
        est = out["est"]
        k = len(dose_grid)
        i25 = max(0, int(0.25 * k))
        i75 = min(k - 1, int(0.75 * k))
        effect = float(est[i75] - est[i25])
        se_eff = float(
            np.sqrt(np.sum((out["influence"][i75] - out["influence"][i25]) ** 2))
        )
        z = sp_stats.norm.ppf(1 - self.alpha / 2)
        marg = np.gradient(est, dose_grid) if k >= 2 else np.zeros_like(est)
        detail = pd.DataFrame(
            {
                "dose": dose_grid,
                "response": est,
                "se": out["se"],
                "ci_lower": out["ci_lower"],
                "ci_upper": out["ci_upper"],
                "marginal_effect": marg,
            }
        )
        info = {
            "method": "dr",
            "dose_range": (float(t_lo), float(t_hi)),
            "n_dose_points": k,
            "bandwidth": out["bandwidth"],
            "bandwidth_risk": out["bw_risk"],
            "avg_marginal_effect": float(np.mean(marg)),
            "effect_25_to_75": effect,
            "dose_25": float(dose_grid[i25]),
            "dose_75": float(dose_grid[i75]),
            "curve_monotone": bool(
                np.all(np.diff(est) > 0) or np.all(np.diff(est) < 0)
            ),
            "pseudo_outcomes": out["pseudo_outcomes"],
        }
        info.update(out["info"])
        return CausalResult(
            method="Dose-Response (doubly robust, Kennedy et al. 2017)",
            estimand="E[Y(t75)] - E[Y(t25)]",
            estimate=effect,
            se=se_eff,
            pvalue=(
                float(2 * sp_stats.norm.sf(abs(effect / se_eff)))
                if se_eff > 0
                else float("nan")
            ),
            ci=(effect - z * se_eff, effect + z * se_eff),
            alpha=self.alpha,
            n_obs=len(Y),
            detail=detail,
            model_info=info,
            _citation_key="dose_response",
        )

    def _estimate_curve(
        self,
        Y: np.ndarray,
        T: np.ndarray,
        X: np.ndarray,
        dose_grid: np.ndarray,
        n: int,
    ) -> np.ndarray:
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

CausalResult._CITATIONS["dose_response"] = (
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
