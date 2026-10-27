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
from randomized trials." *JRSS-A*, 174(2), 369-386. [@stuart2011propensity]

Buchanan, A. L., Hudgens, M. G., Cole, S. R., Mollan, K. R.,
Sax, P. E., Daar, E. S., ... & Mugavero, M. J. (2018).
"Generalizing evidence from randomized trials using inverse probability
of sampling weights." *JRSS-A*, 181(4), 1193-1209.

Dahabreh, I. J., Robertson, S. E., Tchetgen, E. J. T., Stuart, E. A.,
& Hernán, M. A. (2019). "Generalizing causal inferences from
randomized trials: counterfactual and graphical identification."
*Biometrics*, 75(3), 685-694. [@dahabreh2019generalizing]
"""

import numbers
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from .._aliases import accepts_aliases
from .._input_validation import check_columns
from ..core.results import CausalResult
from ..exceptions import (
    AssumptionWarning,
    ConvergenceWarning,
    DataInsufficient,
    MethodIncompatibility,
)
from ..exceptions import warn as _warn
from .basic_tests import _check_alpha

#: Largest covariate imbalance, in standard deviations of the experiment's
#: covariates, that counts as a converged entropy-balancing solution.
_BALANCE_TOL = 1e-4
#: Fitted participation probability treated as numerically 0 or 1 (the
#: quasi-complete separation signature, as in ``sp.ipw``).
_SEPARATION_BOUND = 1e-8


@accepts_aliases(treat="treatment")
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
        ``ci`` at the requested level. ``model_info['converged']`` says
        whether the participation logit (``'ipw'`` / ``'aipw'``) or the
        entropy-balancing solve (``'calibration'``, judged by the largest
        remaining covariate imbalance, at most 1e-4 standard deviations)
        converged on the full sample, with the optimiser records in
        ``model_info['convergence']``; ``model_info['n_boot_not_converged']``
        counts bootstrap replicates that did not. Either failure issues a
        ``ConvergenceWarning``. ``model_info['separation']`` is
        ``'complete'`` / ``'quasi-complete'`` when the covariates separate
        the experiment from the target sample in the participation logit
        (``None`` otherwise, and for ``'calibration'``), with an
        ``AssumptionWarning``; ``model_info['n_participation_clipped']``
        counts experimental units whose participation probability
        ``trim`` clipped. When every unit is clipped the odds weights are
        constant and the estimate is the unweighted experiment contrast.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> age = rng.normal(45, 8, 120)      # experiment skews older
    >>> edu = rng.normal(13, 2, 120)
    >>> treated = rng.binomial(1, 0.5, 120)
    >>> outcome = 2.0 * treated + 0.05 * age + 0.1 * edu + rng.normal(0, 1, 120)
    >>> df_rct = pd.DataFrame({"outcome": outcome, "treated": treated,
    ...                        "age": age, "edu": edu})
    >>> df_pop = pd.DataFrame({"age": rng.normal(38, 8, 200),   # younger target
    ...                        "edu": rng.normal(14, 2, 200)})
    >>> result = sp.pate(
    ...     data_experiment=df_rct, data_target=df_pop,
    ...     y="outcome", treatment="treated", covariates=["age", "edu"],
    ...     method="aipw", n_boot=30, seed=0,
    ... )
    >>> bool(np.isfinite(result.estimate))
    True
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
    _result = estimator.fit()
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.inference.pate",
            params={
                "y": y,
                "treatment": treatment,
                "covariates": list(covariates) if covariates else None,
                "method": method,
                "n_boot": n_boot,
                "alpha": alpha,
                "seed": seed,
                "trim": trim,
            },
            data=data_experiment,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


class PATEEstimator:
    """Population Average Treatment Effect estimator.

    Reweights an experimental (RCT) sample to a target population to recover
    the PATE, correcting the SATE for external validity. The convenience
    wrapper :func:`pate` builds this object and calls :meth:`fit`; construct
    it directly when you want to reuse the estimator.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n_exp = 300
    >>> x_exp = rng.normal(0.0, 1.0, n_exp)
    >>> d = rng.integers(0, 2, n_exp)
    >>> y = 1.0 + 0.5 * x_exp + 2.0 * d + rng.normal(0, 1.0, n_exp)
    >>> df_rct = pd.DataFrame({"x": x_exp, "treated": d, "outcome": y})
    >>> df_pop = pd.DataFrame({"x": rng.normal(0.5, 1.0, 400)})  # target pop
    >>> est = sp.PATEEstimator(
    ...     data_experiment=df_rct,
    ...     data_target=df_pop,
    ...     y="outcome",
    ...     treatment="treated",
    ...     covariates=["x"],
    ...     method="ipw",
    ...     n_boot=50,
    ...     seed=0,
    ... )
    >>> result = est.fit()
    >>> bool(np.isfinite(result.estimate))
    True
    """

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
    ) -> None:
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
        # One record per optimiser run (participation logit or entropy
        # balancing), read by fit() to report convergence.
        self._fit_log: List[Dict[str, Any]] = []
        self._boot_not_converged = 0

        self._validate()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate(self) -> None:
        if self.method not in self._METHODS:
            raise ValueError(
                f"method must be one of {self._METHODS}, got '{self.method}'"
            )
        check_columns(
            self.data_exp,
            {"y": self.y, "treatment": self.treatment, "covariates": self.covariates},
            function="pate (data_experiment)",
        )
        check_columns(
            self.data_tgt,
            {"covariates": self.covariates},
            function="pate (data_target)",
            extra_hint="data_target needs every covariate.",
        )
        # Treatment must be binary
        vals = self.data_exp[self.treatment].dropna().unique()
        if not set(vals).issubset({0, 1, 0.0, 1.0, True, False}):
            raise ValueError("Treatment must be binary (0/1)")
        _check_alpha(self.alpha, "pate")
        trim_is_number = isinstance(self.trim, numbers.Real) and not isinstance(
            self.trim, bool
        )
        if not (trim_is_number and 0.0 <= float(self.trim) < 0.5):
            # at trim >= 0.5 the interval [trim, 1 - trim] is empty or a point
            raise MethodIncompatibility(
                f"pate: trim must lie in [0, 0.5), got {self.trim!r}.",
                recovery_hint="Use trim=0.01 (the default) or trim=0.",
            )
        # Complete cases only, on each sample, with a note.
        exp_cols = list(dict.fromkeys([self.y, self.treatment] + self.covariates))
        keep_exp = self.data_exp[exp_cols].notna().all(axis=1).to_numpy()
        keep_tgt = self.data_tgt[self.covariates].notna().all(axis=1).to_numpy()
        n_drop_exp = int((~keep_exp).sum())
        n_drop_tgt = int((~keep_tgt).sum())
        if n_drop_exp or n_drop_tgt:
            warnings.warn(
                f"pate: dropped {n_drop_exp} of {len(keep_exp)} experiment "
                f"rows and {n_drop_tgt} of {len(keep_tgt)} target rows with "
                "missing values in the outcome, treatment or covariates.",
                UserWarning,
                stacklevel=4,
            )
            self.data_exp = self.data_exp.loc[keep_exp]
            self.data_tgt = self.data_tgt.loc[keep_tgt]
        d = self.data_exp[self.treatment].to_numpy(dtype=float)
        n1 = int(d.sum())
        if n1 == 0 or n1 == len(d) or len(self.data_tgt) == 0:
            raise DataInsufficient(
                f"pate: the experiment has {n1} treated and {len(d) - n1} "
                f"control units and the target sample {len(self.data_tgt)} "
                "rows on the complete cases; the estimator needs both arms "
                "and a target sample.",
                diagnostics={"n_treated": n1, "n_control": len(d) - n1},
            )

    # ------------------------------------------------------------------
    # Participation propensity  P(S=1 | X)
    # ------------------------------------------------------------------
    def _estimate_participation(
        self,
        X_exp: np.ndarray,
        X_tgt: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
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

        def neg_loglik(beta: np.ndarray) -> float:
            z = X_pool_c @ beta
            z = np.clip(z, -30, 30)
            ll = S * z - np.log1p(np.exp(z))
            return float(-np.sum(ll))

        def grad(beta: np.ndarray) -> np.ndarray:
            z = X_pool_c @ beta
            z = np.clip(z, -30, 30)
            prob = 1 / (1 + np.exp(-z))
            return np.asarray(-X_pool_c.T @ (S - prob), dtype=float)

        res = minimize(neg_loglik, beta0, jac=grad, method="L-BFGS-B")
        beta_hat = res.x

        # Predicted P(S=1|X) for the experimental sample
        z_exp = np.column_stack([np.ones(n_exp), X_exp]) @ beta_hat
        z_exp = np.clip(z_exp, -30, 30)
        p_exp = 1 / (1 + np.exp(-z_exp))
        n_clipped = int(np.sum((p_exp < self.trim) | (p_exp > 1 - self.trim)))

        # Separation of the two samples by the covariates. The trimming
        # below turns it into constant odds weights (the unweighted
        # experiment contrast), so it is detected on the fitted index:
        # complete when the index classifies every unit correctly (no
        # finite maximum-likelihood fit, no overlap to reweight on),
        # quasi-complete when some fitted probabilities are numerically 0
        # or 1 although the samples are not separated.
        z_pool = np.clip(X_pool_c @ beta_hat, -30, 30)
        p_pool = 1 / (1 + np.exp(-z_pool))
        separation: Optional[str] = None
        if np.array_equal(p_pool >= 0.5, S == 1):
            separation = "complete"
        elif np.any((p_pool <= _SEPARATION_BOUND) | (p_pool >= 1 - _SEPARATION_BOUND)):
            separation = "quasi-complete"
        self._fit_log.append(
            {
                "step": "participation_logit",
                "converged": bool(res.success),
                "message": str(res.message),
                "n_iter": int(res.nit),
                "max_abs_gradient": float(np.max(np.abs(grad(beta_hat)))),
                "separation": separation,
                "n_clipped": n_clipped,
            }
        )
        p_exp = np.clip(p_exp, self.trim, 1 - self.trim)
        return p_exp, beta_hat

    # ------------------------------------------------------------------
    # Outcome model  E[Y|X, D]  (linear)
    # ------------------------------------------------------------------
    @staticmethod
    def _fit_outcome_model(
        Y: np.ndarray,
        D: np.ndarray,
        X: np.ndarray,
    ) -> np.ndarray:
        """Linear regression of Y on (1, X, D, D*X)."""
        n = len(Y)
        X_c = np.column_stack(
            [
                np.ones(n),
                X,
                D.reshape(-1, 1),
                D.reshape(-1, 1) * X,
            ]
        )
        beta = np.asarray(np.linalg.lstsq(X_c, Y, rcond=None)[0], dtype=float)
        return beta

    @staticmethod
    def _predict_outcome(
        beta: np.ndarray,
        X: np.ndarray,
        d_val: float,
    ) -> np.ndarray:
        """Predict E[Y|X, D=d_val] from the linear model."""
        n = X.shape[0]
        d = np.full(n, d_val)
        X_c = np.column_stack(
            [
                np.ones(n),
                X,
                d.reshape(-1, 1),
                d.reshape(-1, 1) * X,
            ]
        )
        return np.asarray(X_c @ beta, dtype=float)

    # ------------------------------------------------------------------
    # Entropy balancing weights
    # ------------------------------------------------------------------
    def _entropy_balance(
        self,
        X_exp: np.ndarray,
        X_tgt: np.ndarray,
    ) -> np.ndarray:
        """
        Compute entropy-balancing weights so that reweighted experimental
        covariate means match target population means.
        """
        target_means = np.mean(X_tgt, axis=0)
        n_exp = X_exp.shape[0]
        p = X_exp.shape[1]

        # Solve dual so reweighted experimental means match target means.
        lam0 = np.zeros(p)

        def objective(lam: np.ndarray) -> float:
            raw = X_exp @ lam
            raw = np.clip(raw, -30, 30)
            w = np.exp(raw)
            w_sum = np.sum(w)
            loss = np.log(w_sum) - lam @ target_means
            return float(loss)

        def obj_grad(lam: np.ndarray) -> np.ndarray:
            raw = X_exp @ lam
            raw = np.clip(raw, -30, 30)
            w = np.exp(raw)
            w_sum = np.sum(w)
            weighted_mean = (w @ X_exp) / w_sum
            return np.asarray(weighted_mean - target_means, dtype=float)

        res = minimize(
            objective,
            lam0,
            jac=obj_grad,
            method="L-BFGS-B",
            options={"maxiter": 2000},
        )
        raw = X_exp @ res.x
        raw = np.clip(raw, -30, 30)
        weights = np.exp(raw)
        weights = weights / np.sum(weights)  # normalise to sum to 1
        # Converged means the weights balance the covariates: the
        # optimiser's own flag can fail on a line search at a solution, and
        # pass at a point where the target means lie outside the hull of
        # the experiment's covariates and cannot be reached.
        sd = np.std(X_exp, axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        imbalance = float(np.max(np.abs(weights @ X_exp - target_means) / sd))
        self._fit_log.append(
            {
                "step": "entropy_balancing",
                "converged": imbalance <= _BALANCE_TOL,
                "message": str(res.message),
                "n_iter": int(res.nit),
                "max_std_imbalance": imbalance,
            }
        )
        return np.asarray(weights * n_exp, dtype=float)  # scale so mean weight = 1

    # ------------------------------------------------------------------
    # Point estimators
    # ------------------------------------------------------------------
    def _pate_ipw(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        X_exp: np.ndarray,
        X_tgt: np.ndarray,
    ) -> float:
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
        return float(mu1 - mu0)

    def _pate_aipw(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        X_exp: np.ndarray,
        X_tgt: np.ndarray,
    ) -> float:
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

        # AIPW: outcome-model prediction averaged over the target sample,
        # plus the odds-weighted residual correction from the experiment,
        # normalised within each arm (the Hajek form of the IPW estimator
        # above). If either the participation model or the outcome model is
        # correct the sum is consistent for the target-population ATE.
        om_component = np.mean(mu1_tgt - mu0_tgt)

        w1 = w * D
        w0 = w * (1 - D)
        if np.sum(w1) == 0 or np.sum(w0) == 0:
            return np.nan
        aug1 = np.sum(w1 * (Y - mu1_exp)) / np.sum(w1)
        aug0 = np.sum(w0 * (Y - mu0_exp)) / np.sum(w0)

        return float(om_component + aug1 - aug0)

    def _pate_calibration(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        X_exp: np.ndarray,
        X_tgt: np.ndarray,
    ) -> float:
        """Entropy-balancing (calibration) estimator for PATE."""
        w = self._entropy_balance(X_exp, X_tgt)

        w1 = w * D
        w0 = w * (1 - D)
        if np.sum(w1) == 0 or np.sum(w0) == 0:
            return np.nan
        mu1 = np.sum(w1 * Y) / np.sum(w1)
        mu0 = np.sum(w0 * Y) / np.sum(w0)
        return float(mu1 - mu0)

    def _point_estimate(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        X_exp: np.ndarray,
        X_tgt: np.ndarray,
    ) -> float:
        """Dispatch to the chosen method."""
        if self.method == "ipw":
            return self._pate_ipw(Y, D, X_exp, X_tgt)
        elif self.method == "aipw":
            return self._pate_aipw(Y, D, X_exp, X_tgt)
        elif self.method == "calibration":
            return self._pate_calibration(Y, D, X_exp, X_tgt)
        raise ValueError(f"Unsupported PATE method '{self.method}'")

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------
    def _bootstrap(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        X_exp: np.ndarray,
        X_tgt: np.ndarray,
    ) -> np.ndarray:
        """Paired bootstrap over both datasets."""
        n_exp = len(Y)
        n_tgt = X_tgt.shape[0]
        estimates = np.empty(self.n_boot)
        last_exc: Optional[BaseException] = None

        self._boot_not_converged = 0
        for b in range(self.n_boot):
            idx_exp = self.rng.choice(n_exp, size=n_exp, replace=True)
            idx_tgt = self.rng.choice(n_tgt, size=n_tgt, replace=True)
            start = len(self._fit_log)
            try:
                estimates[b] = self._point_estimate(
                    Y[idx_exp],
                    D[idx_exp],
                    X_exp[idx_exp],
                    X_tgt[idx_tgt],
                )
            except Exception as exc:
                estimates[b] = np.nan
                last_exc = exc
            if not all(e["converged"] for e in self._fit_log[start:]):
                self._boot_not_converged += 1

        n_failed = int(np.isnan(estimates).sum())
        if n_failed:
            from ..core._fallback import warn_fallback

            warn_fallback(
                f"{n_failed} of {self.n_boot} PATE bootstrap replicates",
                last_exc,
                "the bootstrap standard error uses the survivors only",
            )
        return np.asarray(estimates[~np.isnan(estimates)], dtype=float)

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
        self._fit_log = []
        estimate = self._point_estimate(Y, D, X_exp, X_tgt)
        point_log = list(self._fit_log)
        converged = all(e["converged"] for e in point_log)
        if not converged:
            bad = [e for e in point_log if not e["converged"]]
            _warn(
                ConvergenceWarning,
                f"pate(method={self.method!r}): the {bad[0]['step']} "
                f"optimiser did not converge ({bad[0]['message']}); the "
                "estimate is reported from the last iterate. With "
                "entropy balancing this usually means the target "
                "covariate means lie outside the range the experiment's "
                "covariates can reweight to.",
                stacklevel=3,
                recovery_hint="Check overlap between the experiment and the "
                "target sample on the covariates, drop or coarsen the "
                "covariate that cannot be balanced, or use method='aipw'.",
                diagnostics={"convergence": bad},
            )

        participation = [e for e in point_log if e["step"] == "participation_logit"]
        separation = participation[0]["separation"] if participation else None
        n_clipped = participation[0]["n_clipped"] if participation else None
        if separation is not None:
            _warn(
                AssumptionWarning,
                f"pate(method={self.method!r}): {separation} separation of "
                "the experiment from the target sample by the covariates. "
                f"{n_clipped} of {len(Y)} experimental units have a "
                "participation probability set by trim= rather than by the "
                "data, so their odds weights are the trimming bound"
                + (
                    " (all of them: the estimate is the unweighted "
                    "experiment contrast, not a population effect)."
                    if n_clipped == len(Y)
                    else "."
                ),
                stacklevel=3,
                recovery_hint="Check covariate overlap between the two "
                "samples (sp.overlap_plot); restrict the target population "
                "to the covariate range the experiment covers, or drop the "
                "separating covariate.",
                diagnostics={"separation": separation, "n_clipped": n_clipped},
            )

        # Bootstrap SE and CI
        self._fit_log = []
        boot_ests = self._bootstrap(Y, D, X_exp, X_tgt)
        n_boot_nc = self._boot_not_converged
        if n_boot_nc:
            _warn(
                ConvergenceWarning,
                f"pate(method={self.method!r}): the optimiser did not "
                f"converge in {n_boot_nc} of {self.n_boot} bootstrap "
                "replicates; they are kept in the standard error and the "
                "interval.",
                stacklevel=3,
                recovery_hint="See model_info['n_boot_not_converged']; check "
                "covariate overlap between the two samples.",
                diagnostics={"n_boot_not_converged": n_boot_nc},
            )
        se = float(np.std(boot_ests, ddof=1)) if len(boot_ests) > 1 else np.nan
        ci_lo = (
            float(np.percentile(boot_ests, 100 * self.alpha / 2))
            if len(boot_ests) > 1
            else np.nan
        )
        ci_hi = (
            float(np.percentile(boot_ests, 100 * (1 - self.alpha / 2)))
            if len(boot_ests) > 1
            else np.nan
        )

        # p-value (Wald)
        if se > 0:
            z = estimate / se
            pvalue = float(2 * stats.norm.sf(abs(z)))
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
            detail=pd.DataFrame(
                {
                    "bootstrap_mean": (
                        [float(np.mean(boot_ests))] if len(boot_ests) else [np.nan]
                    ),
                    "bootstrap_median": (
                        [float(np.median(boot_ests))] if len(boot_ests) else [np.nan]
                    ),
                    "n_boot_valid": [len(boot_ests)],
                    "n_experiment": [len(Y)],
                    "n_target": [X_tgt.shape[0]],
                }
            ),
            model_info={
                "method": self.method,
                "covariates": self.covariates,
                "n_boot": self.n_boot,
                "trim": self.trim,
                "n_experiment": len(Y),
                "n_target": X_tgt.shape[0],
                "converged": converged,
                "convergence": point_log,
                "n_boot_not_converged": n_boot_nc,
                # Participation-logit separation ('complete' /
                # 'quasi-complete' / None; None also for 'calibration') and
                # the experimental units whose probability trim= clipped.
                "separation": separation,
                "n_participation_clipped": n_clipped,
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
