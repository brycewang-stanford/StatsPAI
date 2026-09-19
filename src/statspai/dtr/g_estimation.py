"""
G-estimation for Dynamic Treatment Regimes.

Estimates the optimal treatment rule in a two-stage setting using
the structural nested mean model (SNMM):

    E[Y(a1, a2) - Y(0, a2) | H1] = psi1 * a1 * f1(H1)
    E[Y(a1, a2) - Y(a1, 0) | H2] = psi2 * a2 * f2(H2)

where H_t is the history up to stage t.

The algorithm works backwards from the last stage:
1. Stage 2: estimate psi2 by regressing Y - psi2*A2*f2(H2) on A2
   (find psi2 that makes residuals uncorrelated with A2 given H2).
2. Stage 1: using the "de-blipped" outcome Y_tilde = Y - psi2*A2*f2(H2),
   estimate psi1 similarly.

References
----------
Robins, J. M. (2004). "Optimal Structural Nested Models."
Murphy, S. A. (2003). "Optimal Dynamic Treatment Regimes." [@robins2004optimal]
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..core.results import CausalResult


def g_estimation(
    data: pd.DataFrame,
    y: str,
    treatments: List[str],
    covariates_by_stage: List[List[str]],
    propensity_covariates: Optional[List[List[str]]] = None,
    alpha: float = 0.05,
    n_bootstrap: int = 500,
    random_state: int = 42,
    propensity_model: str = "logit",
) -> CausalResult:
    """
    G-estimation for a multi-stage dynamic treatment regime.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    y : str
        Final outcome variable.
    treatments : list of str
        Treatment variables at each stage, in temporal order.
        E.g., ['A1', 'A2'] for a two-stage DTR.
    covariates_by_stage : list of list of str
        Covariates (tailoring variables) available at each stage.
        covariates_by_stage[k] are the variables available when
        deciding treatment k.
    propensity_covariates : list of list of str, optional
        Covariates for propensity model at each stage.
        If None, uses covariates_by_stage.
    alpha : float, default 0.05
    n_bootstrap : int, default 500
    random_state : int, default 42
    propensity_model : {'logit', 'linear'}, default 'logit'
        Model for ``E[A_k | propensity covariates]`` in the g-estimating
        equation ``sum (A_k - A_hat_k) (Y~ - X_k beta - psi_k A_k) = 0``
        (solved jointly with ``sum X_k (Y~ - X_k beta - psi_k A_k) = 0``,
        ``X_k`` = ``[1, covariates_by_stage[k]]``). ``'logit'`` is the
        logistic MLE that R ``DTRreg(method = "gest")`` fits for a binary
        treatment. ``'linear'`` is a linear-probability OLS fit; with
        propensity covariates equal to the stage covariates it reduces
        ``psi_k`` to the OLS coefficient on ``A_k`` in ``Y~ ~ X_k + A_k``.

    Notes
    -----
    Before 1.29 ``propensity_covariates`` was accepted and silently
    ignored and the propensity was always the linear-probability fit on
    ``covariates_by_stage``; ``propensity_model='linear'`` without
    ``propensity_covariates`` reproduces those numbers. The blip at every
    stage is the constant ``psi_k * a_k``, so ``estimand`` is the sum of
    the stage blips (the effect of always- versus never-treating), which
    is also the optimal regime's value gain only when every ``psi_k > 0``.

    Returns
    -------
    CausalResult
        detail has stage-level blip function estimates.
        model_info contains optimal treatment rules.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> x1 = rng.normal(size=n)
    >>> x2 = rng.normal(size=n)
    >>> x3 = rng.normal(size=n)
    >>> a1 = rng.integers(0, 2, size=n)
    >>> a2 = rng.integers(0, 2, size=n)
    >>> y = 1.0 * a1 + 0.5 * a2 + x1 + 0.5 * x2 + rng.normal(size=n)
    >>> df = pd.DataFrame({
    ...     "outcome": y, "A1": a1, "A2": a2,
    ...     "x1": x1, "x2": x2, "x3": x3,
    ... })
    >>> result = sp.g_estimation(
    ...     df, y='outcome',
    ...     treatments=['A1', 'A2'],
    ...     covariates_by_stage=[['x1', 'x2'], ['x1', 'x2', 'x3']],
    ...     n_bootstrap=50, random_state=0,
    ... )
    >>> result.model_info['n_stages']
    2
    """
    est = GEstimation(
        data=data,
        y=y,
        treatments=treatments,
        covariates_by_stage=covariates_by_stage,
        propensity_covariates=propensity_covariates,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
        propensity_model=propensity_model,
    )
    return est.fit()


class GEstimation:
    """
    G-estimation for dynamic treatment regimes via SNMM.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
    treatments : list of str
    covariates_by_stage : list of list of str
    propensity_covariates : list of list of str, optional
    alpha : float
    n_bootstrap : int
    random_state : int

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> x1 = rng.normal(size=n)
    >>> x2 = rng.normal(size=n)
    >>> x3 = rng.normal(size=n)
    >>> a1 = rng.integers(0, 2, size=n)
    >>> a2 = rng.integers(0, 2, size=n)
    >>> y = 1.0 * a1 + 0.5 * a2 + x1 + 0.5 * x2 + rng.normal(size=n)
    >>> df = pd.DataFrame({
    ...     "outcome": y, "A1": a1, "A2": a2,
    ...     "x1": x1, "x2": x2, "x3": x3,
    ... })
    >>> est = sp.GEstimation(
    ...     df, y="outcome",
    ...     treatments=["A1", "A2"],
    ...     covariates_by_stage=[["x1", "x2"], ["x1", "x2", "x3"]],
    ...     n_bootstrap=50, random_state=0,
    ... )
    >>> res = est.fit()
    >>> res.model_info["n_stages"]
    2
    """

    def __init__(
        self,
        data: pd.DataFrame,
        y: str,
        treatments: List[str],
        covariates_by_stage: List[List[str]],
        propensity_covariates: Optional[List[List[str]]] = None,
        alpha: float = 0.05,
        n_bootstrap: int = 500,
        random_state: int = 42,
        propensity_model: str = "logit",
    ) -> None:
        if propensity_model not in ("logit", "linear"):
            raise ValueError(
                f"propensity_model must be 'logit' or 'linear', got {propensity_model!r}"
            )
        self.propensity_model = propensity_model
        if propensity_covariates is not None and len(propensity_covariates) != len(
            treatments
        ):
            raise ValueError(
                f"propensity_covariates has {len(propensity_covariates)} entries "
                f"but {len(treatments)} treatments were specified"
            )
        self.data = data
        self.y = y
        self.treatments = treatments
        self.covariates_by_stage = covariates_by_stage
        self.propensity_covariates = propensity_covariates or covariates_by_stage
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.n_stages = len(treatments)

        if len(covariates_by_stage) != self.n_stages:
            raise ValueError(
                f"covariates_by_stage has {len(covariates_by_stage)} entries "
                f"but {self.n_stages} treatments were specified"
            )

    def fit(self) -> CausalResult:
        """Run G-estimation backward induction."""
        all_cols = [self.y] + self.treatments
        for stage_covs in self.covariates_by_stage:
            all_cols.extend(stage_covs)
        for stage_covs in self.propensity_covariates:
            all_cols.extend(stage_covs)
        all_cols = list(dict.fromkeys(all_cols))

        missing = [c for c in all_cols if c not in self.data.columns]
        if missing:
            raise ValueError(f"Columns not found in data: {missing}")

        clean = self.data[all_cols].dropna()
        n = len(clean)

        Y = clean[self.y].values.astype(np.float64)
        A = [clean[t].values.astype(np.float64) for t in self.treatments]
        X_stages = [
            clean[covs].values.astype(np.float64) for covs in self.covariates_by_stage
        ]
        P_stages = [
            clean[covs].values.astype(np.float64) for covs in self.propensity_covariates
        ]

        # Backward induction
        psi_estimates, optimal_rules = self._backward_induction(
            Y, A, X_stages, n, P_stages
        )

        # Bootstrap
        rng = np.random.RandomState(self.random_state)
        boot_psis = np.zeros((self.n_bootstrap, self.n_stages))

        for b in range(self.n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            Y_b = Y[idx]
            A_b = [a[idx] for a in A]
            X_b = [x[idx] for x in X_stages]
            P_b = [x[idx] for x in P_stages]
            psis_b, _ = self._backward_induction(Y_b, A_b, X_b, n, P_b)
            boot_psis[b] = psis_b

        se_psis = np.std(boot_psis, axis=0, ddof=1)

        # Overall value: sum of stage effects (average blip)
        total_value = sum(psi_estimates)
        se_total = float(np.std(np.sum(boot_psis, axis=1), ddof=1))

        z_crit = sp_stats.norm.ppf(1 - self.alpha / 2)

        if se_total > 0:
            pvalue = float(2 * sp_stats.norm.sf(abs(total_value / se_total)))
        else:
            pvalue = 0.0

        ci = (total_value - z_crit * se_total, total_value + z_crit * se_total)

        detail_rows = []
        for k in range(self.n_stages):
            detail_rows.append(
                {
                    "stage": k + 1,
                    "treatment": self.treatments[k],
                    "blip_estimate": psi_estimates[k],
                    "se": se_psis[k],
                    "optimal_rule": optimal_rules[k],
                }
            )
        detail = pd.DataFrame(detail_rows)

        model_info = {
            "n_stages": self.n_stages,
            "psi_estimates": list(psi_estimates),
            "optimal_rules": optimal_rules,
            "total_value_optimal": float(total_value),
            "propensity_model": self.propensity_model,
            "propensity_covariates": [list(c) for c in self.propensity_covariates],
        }

        return CausalResult(
            method="G-Estimation (Robins 2004)",
            estimand="Optimal DTR Value",
            estimate=float(total_value),
            se=se_total,
            pvalue=pvalue,
            ci=ci,
            alpha=self.alpha,
            n_obs=n,
            detail=detail,
            model_info=model_info,
            _citation_key="g_estimation",
        )

    def _backward_induction(
        self,
        Y: np.ndarray,
        A: List[np.ndarray],
        X_stages: List[np.ndarray],
        n: int,
        P_stages: Optional[List[np.ndarray]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Backward induction for G-estimation."""
        if P_stages is None:
            P_stages = X_stages
        psi_estimates = np.zeros(self.n_stages)
        optimal_rules = []
        Y_tilde = Y.copy()

        for k in range(self.n_stages - 1, -1, -1):
            A_k = A[k]
            m = len(A_k)
            X_k = np.column_stack([np.ones(m), X_stages[k]])
            P_k = np.column_stack([np.ones(m), P_stages[k]])

            # Propensity E[A_k | P_k]
            if self.propensity_model == "logit":
                A_hat = _logit_fit_predict(P_k, A_k)
            else:
                A_hat = P_k @ np.linalg.lstsq(P_k, A_k, rcond=None)[0]

            # Blip gamma(H_k, a_k) = psi_k * a_k. G-estimation solves
            #   sum [X_k ; A_k - A_hat] (Y~ - X_k beta - psi_k A_k) = 0,
            # the estimating equations of R DTRreg(method = "gest")
            # (instrument matrix Hw, design Hd).
            Hd = np.column_stack([X_k, A_k])
            Hw = np.column_stack([X_k, A_k - A_hat])
            try:
                est = np.linalg.solve(Hw.T @ Hd, Hw.T @ Y_tilde)
                psi_k = float(est[-1])
            except np.linalg.LinAlgError:
                psi_k = float("nan")

            psi_estimates[k] = psi_k

            # Optimal rule: treat if blip > 0
            # For simple model: treat if psi_k > 0
            if psi_k > 0:
                optimal_rules.append("Always treat")
            elif psi_k < 0:
                optimal_rules.append("Never treat")
            else:
                optimal_rules.append("Indifferent")

            # De-blip: remove stage-k treatment effect. (DTRreg adds the
            # regret psi_k (a_opt - A_k); with a constant blip the two
            # pseudo-outcomes differ by a constant, which the intercept in
            # the earlier stage's treatment-free model absorbs.)
            Y_tilde = Y_tilde - psi_k * A_k

        optimal_rules.reverse()  # Back to forward order
        return psi_estimates, optimal_rules


def _logit_fit_predict(X: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Unpenalised logistic MLE by Newton-Raphson; returns fitted P(A=1|X)."""
    beta = np.zeros(X.shape[1])
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-(X @ beta)))
        W = p * (1 - p)
        step = np.linalg.solve((X * W[:, None]).T @ X, X.T @ (a - p))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-12:
            break
    return 1.0 / (1.0 + np.exp(-(X @ beta)))


CausalResult._CITATIONS["g_estimation"] = (
    "@incollection{robins2004optimal,\n"
    "  title={Optimal Structural Nested Models for Optimal Sequential "
    "Decisions},\n"
    "  author={Robins, James M},\n"
    "  booktitle={Proceedings of the Second Seattle Symposium in "
    "Biostatistics},\n"
    "  pages={189--326},\n"
    "  year={2004},\n"
    "  publisher={Springer}\n"
    "}"
)
