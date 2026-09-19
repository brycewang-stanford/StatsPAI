"""
Multi-valued Treatment Effects via IPW and AIPW.

For treatment D in {0, 1, ..., K}, estimates:
- ATE(k, 0): effect of treatment k vs. control (0) for each k.
- Pairwise ATEs: effect of treatment k vs. j.
- Generalized propensity scores via multinomial logit.

Uses AIPW (augmented IPW) for doubly robust estimation:
    tau(k, j) = E[Y(k)] - E[Y(j)]
             = E[ mu_k(X) - mu_j(X) + D_k*(Y - mu_k(X))/e_k(X)
                  - D_j*(Y - mu_j(X))/e_j(X) ]

References
----------
Cattaneo, M. D. (2010).
"Efficient semiparametric estimation of multi-valued treatment effects."
Journal of Econometrics, 155(2), 138-154. [@cattaneo2010efficient]
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..core.results import CausalResult

# sklearn is imported lazily inside the methods that need it so that
# ``import statspai`` doesn't pull ~245 sklearn submodules through this
# file when the user never touches multi_treatment.


def multi_treatment(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    reference: Optional[int] = None,
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    random_state: int = 42,
    outcome_model: str = "gbm",
    se_method: str = "bootstrap",
) -> CausalResult:
    """
    Estimate effects of multi-valued treatments via AIPW.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    y : str
        Outcome variable.
    treat : str
        Treatment variable with K+1 levels (0, 1, ..., K).
    covariates : list of str
        Covariate names.
    reference : int, optional
        Reference treatment level (control). Default: minimum value.
    n_bootstrap : int, default 500
    alpha : float, default 0.05
    random_state : int, default 42
    outcome_model : {'gbm', 'linear'}, default 'gbm'
        Per-arm outcome regression ``mu_k(X)``. ``'gbm'`` is a gradient
        boosting regressor (100 trees, depth 3); ``'linear'`` is OLS with
        an intercept fitted separately in each arm, which is the outcome
        model of Stata's ``teffects aipw`` (``linear by ML``).
    se_method : {'bootstrap', 'influence', 'sandwich'}, default 'bootstrap'
        ``'bootstrap'`` re-fits the whole estimator (same outcome model)
        on ``n_bootstrap`` resamples. ``'influence'`` reports
        ``sd(phi_k - phi_ref) / sqrt(n)`` from the AIPW influence
        function with the nuisance fits treated as known.
        ``'sandwich'`` stacks the multinomial-logit score, the per-arm
        OLS normal equations and the AIPW moments and reports the
        M-estimation sandwich (divisor ``n``) -- the robust standard
        error of Stata's ``teffects aipw`` with a multivalued
        treatment. Both analytic options require
        ``outcome_model='linear'``.

    Returns
    -------
    CausalResult
        detail DataFrame has pairwise effects vs reference.
        model_info contains all pairwise contrasts.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 120
    >>> age = rng.normal(50, 10, n)
    >>> weight = rng.normal(70, 12, n)
    >>> # Treatment with 3 levels: 0 (control), 1 (low), 2 (high)
    >>> dose = rng.integers(0, 3, n)
    >>> y = 2.0 * dose + 0.05 * age + rng.normal(0, 1, n)
    >>> df = pd.DataFrame({'outcome': y, 'dose_level': dose,
    ...                    'age': age, 'weight': weight})
    >>> result = sp.multi_treatment(df, y='outcome', treat='dose_level',
    ...                             covariates=['age', 'weight'],
    ...                             n_bootstrap=15)
    >>> list(result.detail['treatment'])  # effects vs control
    [1, 2]
    """
    est = MultiTreatment(
        data=data,
        y=y,
        treat=treat,
        covariates=covariates,
        reference=reference,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        random_state=random_state,
        outcome_model=outcome_model,
        se_method=se_method,
    )
    return est.fit()


class MultiTreatment:
    """Multi-valued treatment effects estimator.

    Estimates AIPW (doubly robust) contrasts of each treatment arm
    against a reference level for a discrete treatment ``D in
    {0, 1, ..., K}``. Construct, then call :meth:`fit`.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> from statspai.multi_treatment.multi_ipw import MultiTreatment
    >>> rng = np.random.default_rng(0)
    >>> n = 120
    >>> age = rng.normal(50, 10, n)
    >>> weight = rng.normal(70, 12, n)
    >>> dose = rng.integers(0, 3, n)
    >>> y = 2.0 * dose + 0.05 * age + rng.normal(0, 1, n)
    >>> df = pd.DataFrame({'outcome': y, 'dose_level': dose,
    ...                    'age': age, 'weight': weight})
    >>> est = MultiTreatment(data=df, y='outcome', treat='dose_level',
    ...                      covariates=['age', 'weight'], n_bootstrap=15)
    >>> res = est.fit()
    >>> list(res.detail['treatment'])
    [1, 2]

    References
    ----------
    [@cattaneo2010efficient]
    """

    def __init__(
        self,
        data: pd.DataFrame,
        y: str,
        treat: str,
        covariates: List[str],
        reference: Optional[int] = None,
        n_bootstrap: int = 500,
        alpha: float = 0.05,
        random_state: int = 42,
        outcome_model: str = "gbm",
        se_method: str = "bootstrap",
    ):
        if outcome_model not in ("gbm", "linear"):
            raise ValueError(
                f"outcome_model must be 'gbm' or 'linear', got {outcome_model!r}"
            )
        if se_method not in ("bootstrap", "influence", "sandwich"):
            raise ValueError(
                "se_method must be 'bootstrap', 'influence' or 'sandwich', "
                f"got {se_method!r}"
            )
        if se_method != "bootstrap" and outcome_model != "linear":
            raise ValueError(
                f"se_method={se_method!r} is the analytic variance of the "
                "parametric AIPW estimator and requires outcome_model='linear'."
            )
        self.outcome_model = outcome_model
        self.se_method = se_method
        self.data = data
        self.y = y
        self.treat = treat
        self.covariates = covariates
        self.reference = reference
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self.random_state = random_state

    def fit(self) -> CausalResult:
        """Estimate multi-valued treatment effects."""
        cols = [self.y, self.treat] + self.covariates
        missing = [c for c in cols if c not in self.data.columns]
        if missing:
            raise ValueError(f"Columns not found in data: {missing}")

        clean = self.data[cols].dropna()
        Y = clean[self.y].values.astype(np.float64)
        D = clean[self.treat].values.astype(int)
        X = clean[self.covariates].values.astype(np.float64)
        n = len(Y)

        levels = np.sort(np.unique(D))
        K = len(levels)
        if K < 2:
            raise ValueError("Treatment must have at least 2 levels")

        if self.reference is None:
            self.reference = int(levels[0])

        if self.reference not in levels:
            raise ValueError(
                f"Reference level {self.reference} not found in treatment "
                f"levels: {levels}"
            )

        ey, phi, parts = self._point(Y, D, X, levels)

        # Contrasts vs reference
        ref = self.reference
        detail_rows = []
        for k in levels:
            if k == ref:
                continue
            ate_k = ey[k] - ey[ref]
            detail_rows.append(
                {
                    "treatment": int(k),
                    "reference": int(ref),
                    "estimate": float(ate_k),
                }
            )
        n_contrasts = len(detail_rows)

        if self.se_method == "bootstrap":
            rng = np.random.RandomState(self.random_state)
            boot_ates = np.zeros((self.n_bootstrap, n_contrasts))
            for b in range(self.n_bootstrap):
                idx = rng.choice(n, size=n, replace=True)
                ey_b, _, _ = self._point(Y[idx], D[idx], X[idx], levels)
                for j, row in enumerate(detail_rows):
                    boot_ates[b, j] = ey_b[row["treatment"]] - ey_b[ref]
            ses = [float(np.std(boot_ates[:, j], ddof=1)) for j in range(n_contrasts)]
            po_se = None
        else:
            if self.se_method == "sandwich":
                ifs = _stacked_if(X, D, Y, levels, parts, phi, ey)
                ses = [
                    float(np.sqrt(np.mean((ifs[r["treatment"]] - ifs[ref]) ** 2) / n))
                    for r in detail_rows
                ]
                po_se = {
                    int(k): float(np.sqrt(np.mean(ifs[k] ** 2) / n)) for k in levels
                }
            else:
                ses = [
                    float(np.sqrt(np.var(phi[r["treatment"]] - phi[ref], ddof=1) / n))
                    for r in detail_rows
                ]
                po_se = {
                    int(k): float(np.sqrt(np.var(phi[k], ddof=1) / n)) for k in levels
                }

        # Add SE and CI to detail
        z_crit = sp_stats.norm.ppf(1 - self.alpha / 2)
        for j, row in enumerate(detail_rows):
            se = ses[j]
            row["se"] = se
            z_stat = row["estimate"] / max(se, 1e-10)
            pv = float(2 * sp_stats.norm.sf(abs(z_stat)))
            row["pvalue"] = pv
            row["ci_lower"] = row["estimate"] - z_crit * se
            row["ci_upper"] = row["estimate"] + z_crit * se

        detail = pd.DataFrame(detail_rows)

        # Overall F-test-like: is any treatment different from reference?
        if n_contrasts > 0:
            main_estimate = detail_rows[0]["estimate"]
            main_se = detail_rows[0]["se"]
            main_pvalue = detail_rows[0]["pvalue"]
            main_ci = (detail_rows[0]["ci_lower"], detail_rows[0]["ci_upper"])
        else:
            main_estimate, main_se, main_pvalue = 0.0, 0.0, 1.0
            main_ci = (0.0, 0.0)

        model_info = {
            "treatment_levels": levels.tolist(),
            "reference": int(ref),
            "n_levels": K,
            "potential_outcomes": {int(k): v for k, v in ey.items()},
            "potential_outcomes_se": po_se,
            "outcome_model": self.outcome_model,
            "se_method": self.se_method,
            "propensity_model": "multinomial logit (unpenalized MLE)",
        }

        return CausalResult(
            method="Multi-valued Treatment (AIPW, Cattaneo 2010)",
            estimand="ATE vs reference",
            estimate=main_estimate,
            se=main_se,
            pvalue=main_pvalue,
            ci=main_ci,
            alpha=self.alpha,
            n_obs=n,
            detail=detail,
            model_info=model_info,
            _citation_key="multi_treatment",
        )

    def _point(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        X: np.ndarray,
        levels: np.ndarray,
    ):
        """AIPW potential-outcome means; the same estimator for the point
        estimate and every bootstrap replicate."""
        gps, gamma, Xc = _mlogit_gps(X, D, levels)
        mu_hats = {}
        betas = {}
        for k in levels:
            mask_k = D == k
            if self.outcome_model == "linear":
                if mask_k.sum() < Xc.shape[1]:
                    raise ValueError(
                        f"Treatment arm {k} has {int(mask_k.sum())} units, "
                        f"fewer than the {Xc.shape[1]} outcome-model parameters."
                    )
                beta = np.linalg.lstsq(Xc[mask_k], Y[mask_k], rcond=None)[0]
                betas[k] = beta
                mu_hats[k] = Xc @ beta
            elif mask_k.sum() > 2:
                from sklearn.ensemble import GradientBoostingRegressor

                m = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=self.random_state,
                )
                m.fit(X[mask_k], Y[mask_k])
                mu_hats[k] = m.predict(X)
            else:
                raise ValueError(
                    f"Treatment arm {k} has {int(mask_k.sum())} units; the "
                    "outcome model needs at least 3."
                )

        ey = {}
        phi = {}
        e_clip = {}
        for j, k in enumerate(levels):
            d_k = (D == k).astype(float)
            e_k = np.clip(gps[:, j], 0.01, 0.99)
            e_clip[k] = e_k
            phi[k] = mu_hats[k] + d_k * (Y - mu_hats[k]) / e_k
            ey[k] = float(np.mean(phi[k]))
        parts = {"gps": gps, "gamma": gamma, "Xc": Xc, "mu": mu_hats, "beta": betas}
        return ey, phi, parts


def _mlogit_gps(X: np.ndarray, D: np.ndarray, levels: np.ndarray):
    """Unpenalised multinomial-logit generalized propensity scores.

    Newton-Raphson on the full multinomial log-likelihood with the first
    level as base (the fitted probabilities do not depend on the base).
    Before 1.29 this was ``sklearn.linear_model.LogisticRegression()`` at
    its default ``C=1.0``: an L2-penalised fit whose shrinkage depends on
    the covariates' scale, so the "multinomial logit" GPS was not the
    MLE that Cattaneo (2010) and Stata's ``teffects`` use.

    Returns ``(gps, gamma, Xc)`` with ``gps`` of shape ``(n, K)`` aligned
    with ``levels``, ``gamma`` of shape ``(K-1, p)`` for the non-base
    levels and the design ``Xc`` with a leading intercept.
    """
    n = len(D)
    Xc = np.column_stack([np.ones(n), X])
    p = Xc.shape[1]
    K = len(levels)
    Dk = np.column_stack([(D == k).astype(float) for k in levels[1:]])
    gamma = np.zeros((K - 1, p))
    for _ in range(200):
        eta = Xc @ gamma.T
        eta_max = np.maximum(eta.max(axis=1), 0.0)
        ex = np.exp(eta - eta_max[:, None])
        denom = np.exp(-eta_max) + ex.sum(axis=1)
        P = ex / denom[:, None]
        score = ((Dk - P).T @ Xc).ravel()
        H = np.zeros(((K - 1) * p, (K - 1) * p))
        for a in range(K - 1):
            for b in range(K - 1):
                w = P[:, a] * ((a == b) - P[:, b])
                H[a * p : (a + 1) * p, b * p : (b + 1) * p] = (Xc * w[:, None]).T @ Xc
        step = np.linalg.solve(H, score)
        gamma = gamma + step.reshape(K - 1, p)
        if np.max(np.abs(step)) < 1e-12:
            break
    else:
        import warnings

        from ..exceptions import ConvergenceWarning

        warnings.warn(
            "multi_treatment: the multinomial-logit propensity model did not "
            "converge in 200 Newton iterations (separation?).",
            ConvergenceWarning,
            stacklevel=3,
        )
    eta = Xc @ gamma.T
    eta_max = np.maximum(eta.max(axis=1), 0.0)
    ex = np.exp(eta - eta_max[:, None])
    denom = np.exp(-eta_max) + ex.sum(axis=1)
    gps = np.column_stack([np.exp(-eta_max) / denom, ex / denom[:, None]])
    return gps, gamma, Xc


def _stacked_if(X, D, Y, levels, parts, phi, ey):
    """Influence functions of the AIPW means under the stacked M-estimator.

    Parameters: multinomial-logit ``gamma`` (non-base levels), per-arm OLS
    ``beta_k`` and the means ``m_k``. Row ``i`` of the returned arrays is
    ``-A^{-1} psi_i`` for ``m_k``: the centred AIPW moment plus
    ``E[d psi_m / d gamma] IF_gamma + E[d psi_m / d beta_k] IF_beta_k``.
    Assumes no propensity clipping bound (the derivative of a clipped
    score is zero, not the logit derivative).
    """
    gps, Xc = parts["gps"], parts["Xc"]
    n, p = Xc.shape
    K = len(levels)
    if np.any((gps < 0.01) | (gps > 0.99)):
        import warnings

        warnings.warn(
            "multi_treatment: some generalized propensity scores were "
            "clipped to [0.01, 0.99]; the stacked sandwich is approximate "
            "for those rows.",
            RuntimeWarning,
            stacklevel=3,
        )
    P = gps[:, 1:]
    Dk = np.column_stack([(D == k).astype(float) for k in levels[1:]])
    H = np.zeros(((K - 1) * p, (K - 1) * p))
    for a in range(K - 1):
        for b in range(K - 1):
            w = P[:, a] * ((a == b) - P[:, b])
            H[a * p : (a + 1) * p, b * p : (b + 1) * p] = (Xc * w[:, None]).T @ Xc / n
    scores = np.column_stack([Xc * (Dk[:, a] - P[:, a])[:, None] for a in range(K - 1)])
    if_gamma = np.linalg.solve(H, scores.T).T  # (n, (K-1) p)

    out = {}
    for j, k in enumerate(levels):
        d_k = (D == k).astype(float)
        e_k = gps[:, j]
        r_k = Y - parts["mu"][k]
        # d e_k / d gamma_a = e_k (1{k = a} - e_a) x, a over non-base levels
        grad_g = np.concatenate(
            [
                (Xc * (-d_k * r_k * ((j == a + 1) - P[:, a]) / e_k)[:, None]).mean(
                    axis=0
                )
                for a in range(K - 1)
            ]
        )
        grad_b = (Xc * (1 - d_k / e_k)[:, None]).mean(axis=0)
        bread_b = (Xc * d_k[:, None]).T @ Xc / n
        if_b = np.linalg.solve(bread_b, (Xc * (d_k * r_k)[:, None]).T).T
        out[k] = phi[k] - ey[k] + if_gamma @ grad_g + if_b @ grad_b
    return out


CausalResult._CITATIONS["multi_treatment"] = (
    "@article{cattaneo2010efficient,\n"
    "  title={Efficient Semiparametric Estimation of Multi-valued "
    "Treatment Effects under Ignorability},\n"
    "  author={Cattaneo, Matias D},\n"
    "  journal={Journal of Econometrics},\n"
    "  volume={155},\n"
    "  number={2},\n"
    "  pages={138--154},\n"
    "  year={2010},\n"
    "  publisher={Elsevier}\n"
    "}"
)
