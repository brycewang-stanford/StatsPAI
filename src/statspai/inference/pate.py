"""
Population Average Treatment Effect (PATE) Estimation for External Validity.

When an experiment is conducted on a non-representative sample, PATE
estimators reweight (or model) the experimental data to match a target
population, recovering the treatment effect that would have been
observed had the experiment been run on that population.

Methods
-------
- **IPW**: Inverse-probability weighting on the participation probability
  P(S=1|X), following Stuart et al. (2011) and Buchanan et al. (2018).
- **AIPW**: Augmented IPW (doubly robust) combining outcome modelling
  with IPW reweighting.
- **Calibration**: Entropy balancing to equate covariate moments between
  the experimental sample and target population.

References
----------
Stuart, E. A., Cole, S. R., Bradshaw, C. P., & Leaf, P. J. (2011).
"The use of propensity scores to assess the generalizability of results
from randomized trials." *JRSS-A*, 174(2), 369-386.

Buchanan, A. L., Hudgens, M. G., Cole, S. R., Mollan, K. R.,
Sax, P. E., Daar, E. S., ... & Mugavero, M. J. (2018).
"Generalizing evidence from randomized trials using inverse probability
of sampling weights." *JRSS-A*, 181(4), 1193-1209.

Dahabreh, I. J., Robertson, S. E., Tchetgen, E. J. T., Stuart, E. A.,
& Hernán, M. A. (2019). "Generalizing causal inferences from
randomized trials: counterfactual and graphical identification."
*Biometrics*, 75(3), 685-694.
"""

from typing import Optional, List

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from ..core.results import CausalResult


def pate(
    data_experiment: pd.DataFrame,
    data_target: pd.DataFrame,
    y: str,
    treatment: str,
    covariates: List[str],
    method: str = "ipw",
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: Optional[int] = None,
    trim: float = 0.01,
) -> CausalResult:
    """
    Estimate the Population Average Treatment Effect (PATE).

    Adjusts experimental estimates for external validity by reweighting
    the study sample to match a target population.

    Parameters
    ----------
    data_experiment : pd.DataFrame
        Experimental/study sample.  Must contain *y*, *treatment*, and
        all *covariates*.
    data_target : pd.DataFrame
        Target population sample.  Must contain all *covariates*.
        Need not contain *y* or *treatment*.
    y : str
        Outcome variable (only used from data_experiment).
    treatment : str
        Binary treatment indicator (only in data_experiment).
    covariates : list of str
        Shared covariates present in both datasets.
    method : {'ipw', 'aipw', 'calibration'}
        Estimation strategy:

        * ``'ipw'`` -- Inverse probability of sampling weights.
        * ``'aipw'`` -- Augmented IPW (doubly robust).
        * ``'calibration'`` -- Entropy balancing on covariate moments.
    n_boot : int, default 500
        Number of bootstrap replications for standard-error estimation.
    alpha : float, default 0.05
        Significance level for the confidence interval.
    seed : int, optional
        Random seed for reproducibility.
    trim : float, default 0.01
        Trimming threshold for participation propensities (values
        below *trim* or above 1 - *trim* are clipped).

    Returns
    -------
    CausalResult
        With ``estimate`` = PATE, ``se`` from the bootstrap, and
        ``ci`` at the requested level.

    Examples
    --------
    >>> result = sp.pate(
    ...     data_experiment=df_rct,
    ...     data_target=df_pop,
    ...     y="outcome",
    ...     treatment="treated",
    ...     covariates=["age", "edu", "income"],
    ...     method="aipw",
    ... )
    >>> result.summary()
    """
    estimator = PATEEstimator(
        data_experiment=data_experiment,
        data_target=data_target,
        y=y,
        treatment=treatment,
        covariates=covariates,
        method=method,
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
        trim=trim,
    )
    return estimator.fit()


class PATEEstimator:
    """Population Average Treatment Effect estimator."""

    _METHODS = ("ipw", "aipw", "calibration")

    def __init__(
        self,
        data_experiment: pd.DataFrame,
        data_target: pd.DataFrame,
        y: str,
        treatment: str,
        covariates: List[str],
        method: str = "ipw",
        n_boot: int = 500,
        alpha: float = 0.05,
        seed: Optional[int] = None,
        trim: float = 0.01,
    ):
        self.data_exp = data_experiment.copy()
        self.data_tgt = data_target.copy()
        self.y = y
        self.treatment = treatment
        self.covariates = list(covariates)
        self.method = method.lower()
        self.n_boot = n_boot
        self.alpha = alpha
        self.trim = trim
        self.rng = np.random.default_rng(seed)

        self._validate()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate(self):
        if self.method not in self._METHODS:
            raise ValueError(
                f"method must be one of {self._METHODS}, got '{self.method}'"
            )
        for col in [self.y, self.treatment] + self.covariates:
            if col not in self.data_exp.columns:
                raise ValueError(
                    f"Column '{col}' not found in data_experiment"
                )
        for col in self.covariates:
            if col not in self.data_tgt.columns:
                raise ValueError(
                    f"Column '{col}' not found in data_target"
                )
        # Treatment must be binary
        vals = self.data_exp[self.treatment].dropna().unique()
        if not set(vals).issubset({0, 1, 0.0, 1.0, True, False}):
            raise ValueError("Treatment must be binary (0/1)")

    # ------------------------------------------------------------------
    # Participation propensity  P(S=1 | X)
    # ------------------------------------------------------------------
    def _estimate_participation(self, X_exp: np.ndarray, X_tgt: np.ndarray):
        """Logistic regression for P(S=1|X) on the pooled sample."""
        n_exp = X_exp.shape[0]
        n_tgt = X_tgt.shape[0]
        X_pool = np.vstack([X_exp, X_tgt])
        S = np.concatenate([np.ones(n_exp), np.zeros(n_tgt)])

        # Add intercept
        X_pool_c = np.column_stack([np.ones(len(S)), X_pool])

        # Fit via scipy minimize (logistic NLL)
        p = X_pool_c.shape[1]
        beta0 = np.zeros(p)

        def neg_loglik(beta):
            z = X_pool_c @ beta
            z = np.clip(z, -30, 30)
            ll = S * z - np.log1p(np.exp(z))
            return -np.sum(ll)

        def grad(beta):
            z = X_pool_c @ beta
            z = np.clip(z, -30, 30)
            prob = 1 / (1 + np.exp(-z))
            return -X_pool_c.T @ (S - prob)

        res = minimize(neg_loglik, beta0, jac=grad, method="L-BFGS-B")
        beta_hat = res.x

        # Predicted P(S=1|X) for the experimental sample
        z_exp = np.column_stack([np.ones(n_exp), X_exp]) @ beta_hat
        z_exp = np.clip(z_exp, -30, 30)
        p_exp = 1 / (1 + np.exp(-z_exp))
        p_exp = np.clip(p_exp, self.trim, 1 - self.trim)
        return p_exp, beta_hat

    # ------------------------------------------------------------------
    # Outcome model  E[Y|X, D]  (linear)
    # ------------------------------------------------------------------
    @staticmethod
    def _fit_outcome_model(Y, D, X):
        """Linear regression of Y on (1, X, D, D*X)."""
        n = len(Y)
        X_c = np.column_stack([np.ones(n), X, D.reshape(-1, 1),
                                D.reshape(-1, 1) * X])
        beta = np.linalg.lstsq(X_c, Y, rcond=None)[0]
        return beta

    @staticmethod
    def _predict_outcome(beta, X, d_val):
        """Predict E[Y|X, D=d_val] from the linear model."""
        n = X.shape[0]
        d = np.full(n, d_val)
        X_c = np.column_stack([np.ones(n), X, d.reshape(-1, 1),
                                d.reshape(-1, 1) * X])
        return X_c @ beta

    # ------------------------------------------------------------------
    # Entropy balancing weights
    # ------------------------------------------------------------------
    def _entropy_balance(self, X_exp: np.ndarray, X_tgt: np.ndarray):
        """
        Compute entropy-balancing weights so that reweighted experimental
        covariate means match target population means.
        """
        target_means = np.mean(X_tgt, axis=0)
        n_exp = X_exp.shape[0]
        p = X_exp.shape[1]

        # Solve dual: min sum exp(lambda' x_i) s.t. sum w_i x_i / sum w_i = target_means
        lam0 = np.zeros(p)

        def objective(lam):
            raw = X_exp @ lam
            raw = np.clip(raw, -30, 30)
            w = np.exp(raw)
            w_sum = np.sum(w)
            loss = np.log(w_sum) - lam @ target_means
            return loss

        def obj_grad(lam):
            raw = X_exp @ lam
            raw = np.clip(raw, -30, 30)
            w = np.exp(raw)
            w_sum = np.sum(w)
            weighted_mean = (w @ X_exp) / w_sum
            return weighted_mean - target_means

        res = minimize(objective, lam0, jac=obj_grad, method="L-BFGS-B",
                       options={"maxiter": 2000})
        raw = X_exp @ res.x
        raw = np.clip(raw, -30, 30)
        weights = np.exp(raw)
        weights = weights / np.sum(weights)  # normalise to sum to 1
        return weights * n_exp  # scale so mean weight = 1

    # ------------------------------------------------------------------
    # Point estimators
    # ------------------------------------------------------------------
    def _pate_ipw(self, Y, D, X_exp, X_tgt):
        """IPW estimator for PATE."""
        p_s, _ = self._estimate_participation(X_exp, X_tgt)
        w = (1 - p_s) / p_s  # odds weight

        # Weighted difference in means
        w1 = w * D
        w0 = w * (1 - D)
        if np.sum(w1) == 0 or np.sum(w0) == 0:
            return np.nan
        mu1 = np.sum(w1 * Y) / np.sum(w1)
        mu0 = np.sum(w0 * Y) / np.sum(w0)
        return mu1 - mu0

    def _pate_aipw(self, Y, D, X_exp, X_tgt):
        """Augmented IPW (doubly robust) estimator for PATE."""
        p_s, _ = self._estimate_participation(X_exp, X_tgt)
        w = (1 - p_s) / p_s

        # Outcome model on experimental data
        beta = self._fit_outcome_model(Y, D, X_exp)
        mu1_exp = self._predict_outcome(beta, X_exp, 1.0)
        mu0_exp = self._predict_outcome(beta, X_exp, 0.0)

        # Outcome predictions for target population
        mu1_tgt = self._predict_outcome(beta, X_tgt, 1.0)
        mu0_tgt = self._predict_outcome(beta, X_tgt, 0.0)

        # AIPW: augment with outcome-model predictions on target
        n_tgt = X_tgt.shape[0]
        # Outcome-model component (on target pop)
        om_component = np.mean(mu1_tgt - mu0_tgt)

        # IPW augmentation component (on experimental sample)
        w_sum = np.sum(w)
        if w_sum == 0:
            return om_component

        aug1 = np.sum(w * D * (Y - mu1_exp)) / w_sum
        aug0 = np.sum(w * (1 - D) * (Y - mu0_exp)) / w_sum
        n_exp = len(Y)
        # Scale augmentation by relative sample size
        scale = n_exp / (n_exp + n_tgt)

        return om_component + (aug1 - aug0) * (n_exp / n_tgt)

    def _pate_calibration(self, Y, D, X_exp, X_tgt):
        """Entropy-balancing (calibration) estimator for PATE."""
        w = self._entropy_balance(X_exp, X_tgt)

        w1 = w * D
        w0 = w * (1 - D)
        if np.sum(w1) == 0 or np.sum(w0) == 0:
            return np.nan
        mu1 = np.sum(w1 * Y) / np.sum(w1)
        mu0 = np.sum(w0 * Y) / np.sum(w0)
        return mu1 - mu0

    def _point_estimate(self, Y, D, X_exp, X_tgt):
        """Dispatch to the chosen method."""
        if self.method == "ipw":
            return self._pate_ipw(Y, D, X_exp, X_tgt)
        elif self.method == "aipw":
            return self._pate_aipw(Y, D, X_exp, X_tgt)
        elif self.method == "calibration":
            return self._pate_calibration(Y, D, X_exp, X_tgt)

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------
    def _bootstrap(self, Y, D, X_exp, X_tgt):
        """Paired bootstrap over both datasets."""
        n_exp = len(Y)
        n_tgt = X_tgt.shape[0]
        estimates = np.empty(self.n_boot)

        for b in range(self.n_boot):
            idx_exp = self.rng.choice(n_exp, size=n_exp, replace=True)
            idx_tgt = self.rng.choice(n_tgt, size=n_tgt, replace=True)
            try:
                estimates[b] = self._point_estimate(
                    Y[idx_exp], D[idx_exp],
                    X_exp[idx_exp], X_tgt[idx_tgt],
                )
            except Exception:
                estimates[b] = np.nan

        return estimates[~np.isnan(estimates)]

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self) -> CausalResult:
        """Estimate PATE and return a CausalResult."""
        Y = self.data_exp[self.y].values.astype(float)
        D = self.data_exp[self.treatment].values.astype(float)
        X_exp = self.data_exp[self.covariates].values.astype(float)
        X_tgt = self.data_tgt[self.covariates].values.astype(float)

        # Point estimate
        estimate = self._point_estimate(Y, D, X_exp, X_tgt)

        # Bootstrap SE and CI
        boot_ests = self._bootstrap(Y, D, X_exp, X_tgt)
        se = float(np.std(boot_ests, ddof=1)) if len(boot_ests) > 1 else np.nan
        ci_lo = float(np.percentile(boot_ests, 100 * self.alpha / 2)) if len(boot_ests) > 1 else np.nan
        ci_hi = float(np.percentile(boot_ests, 100 * (1 - self.alpha / 2))) if len(boot_ests) > 1 else np.nan

        # p-value (Wald)
        if se > 0:
            z = estimate / se
            pvalue = float(2 * (1 - stats.norm.cdf(abs(z))))
        else:
            pvalue = np.nan

        method_labels = {
            "ipw": "PATE (IPW)",
            "aipw": "PATE (AIPW, doubly robust)",
            "calibration": "PATE (Entropy Balancing)",
        }

        result = CausalResult(
            method=method_labels[self.method],
            estimand="PATE",
            estimate=float(estimate),
            se=se,
            pvalue=pvalue,
            ci=(ci_lo, ci_hi),
            alpha=self.alpha,
            n_obs=len(Y),
            detail=pd.DataFrame({
                "bootstrap_mean": [float(np.mean(boot_ests))] if len(boot_ests) else [np.nan],
                "bootstrap_median": [float(np.median(boot_ests))] if len(boot_ests) else [np.nan],
                "n_boot_valid": [len(boot_ests)],
                "n_experiment": [len(Y)],
                "n_target": [X_tgt.shape[0]],
            }),
            model_info={
                "method": self.method,
                "covariates": self.covariates,
                "n_boot": self.n_boot,
                "trim": self.trim,
                "n_experiment": len(Y),
                "n_target": X_tgt.shape[0],
            },
            _citation_key="pate",
        )
        return result


# Register citation
CausalResult._CITATIONS["pate"] = (
    "@article{stuart2011propensity,\n"
    "  title={The use of propensity scores to assess the generalizability\n"
    "         of results from randomized trials},\n"
    "  author={Stuart, Elizabeth A and Cole, Stephen R and\n"
    "          Bradshaw, Catherine P and Leaf, Philip J},\n"
    "  journal={Journal of the Royal Statistical Society: Series A},\n"
    "  volume={174},\n"
    "  number={2},\n"
    "  pages={369--386},\n"
    "  year={2011}\n"
    "}\n"
    "@article{buchanan2018generalizing,\n"
    "  title={Generalizing evidence from randomized trials using inverse\n"
    "         probability of sampling weights},\n"
    "  author={Buchanan, Ashley L and Hudgens, Michael G and Cole, Stephen R\n"
    "          and others},\n"
    "  journal={Journal of the Royal Statistical Society: Series A},\n"
    "  volume={181},\n"
    "  number={4},\n"
    "  pages={1193--1209},\n"
    "  year={2018}\n"
    "}"
)
