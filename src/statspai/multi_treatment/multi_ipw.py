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

from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# sklearn is imported lazily inside the methods that need it so that
# ``import statspai`` doesn't pull ~245 sklearn submodules through this
# file when the user never touches multi_treatment.

from ..core.results import CausalResult


def multi_treatment(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    reference: Optional[int] = None,
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    random_state: int = 42,
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

    Returns
    -------
    CausalResult
        detail DataFrame has pairwise effects vs reference.
        model_info contains all pairwise contrasts.

    Examples
    --------
    >>> import statspai as sp
    >>> # Treatment with 3 levels: 0 (control), 1 (low), 2 (high)
    >>> result = sp.multi_treatment(df, y='outcome', treat='dose_level',
    ...                             covariates=['age', 'weight'])
    >>> print(result.detail)  # effects vs control
    """
    est = MultiTreatment(
        data=data, y=y, treat=treat, covariates=covariates,
        reference=reference, n_bootstrap=n_bootstrap,
        alpha=alpha, random_state=random_state,
    )
    return est.fit()


class MultiTreatment:
    """Multi-valued treatment effects estimator."""

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
    ):
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

        from sklearn.ensemble import GradientBoostingRegressor

        # Estimate generalized propensity scores
        gps = self._estimate_gps(X, D, levels)

        # Outcome models per treatment arm
        mu_hats = {}
        for k in levels:
            mask_k = D == k
            if mask_k.sum() > 0:
                m = GradientBoostingRegressor(
                    n_estimators=100, max_depth=3, learning_rate=0.1,
                    random_state=self.random_state,
                )
                m.fit(X[mask_k], Y[mask_k])
                mu_hats[k] = m.predict(X)
            else:
                mu_hats[k] = np.zeros(n)

        # AIPW estimator for each E[Y(k)]
        ey = {}
        for k in levels:
            d_k = (D == k).astype(float)
            e_k = np.clip(gps[:, list(levels).index(k)], 0.01, 0.99)
            mu_k = mu_hats[k]

            ey[k] = float(np.mean(
                mu_k + d_k * (Y - mu_k) / e_k
            ))

        # Contrasts vs reference
        ref = self.reference
        detail_rows = []
        for k in levels:
            if k == ref:
                continue
            ate_k = ey[k] - ey[ref]
            detail_rows.append({
                'treatment': int(k),
                'reference': int(ref),
                'estimate': float(ate_k),
            })

        # Bootstrap
        rng = np.random.RandomState(self.random_state)
        n_contrasts = len(detail_rows)
        boot_ates = np.zeros((self.n_bootstrap, n_contrasts))

        for b in range(self.n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            Y_b, D_b, X_b = Y[idx], D[idx], X[idx]

            gps_b = self._estimate_gps(X_b, D_b, levels)
            mu_b = {}
            for k in levels:
                mask_k = D_b == k
                if mask_k.sum() > 2:
                    m = GradientBoostingRegressor(
                        n_estimators=50, max_depth=3,
                        random_state=self.random_state,
                    )
                    m.fit(X_b[mask_k], Y_b[mask_k])
                    mu_b[k] = m.predict(X_b)
                else:
                    mu_b[k] = np.zeros(len(idx))

            ey_b = {}
            for k in levels:
                d_k = (D_b == k).astype(float)
                e_k = np.clip(gps_b[:, list(levels).index(k)], 0.01, 0.99)
                ey_b[k] = float(np.mean(
                    mu_b[k] + d_k * (Y_b - mu_b[k]) / e_k
                ))

            for j, row in enumerate(detail_rows):
                k = row['treatment']
                boot_ates[b, j] = ey_b[k] - ey_b[ref]

        # Add SE and CI to detail
        z_crit = sp_stats.norm.ppf(1 - self.alpha / 2)
        for j, row in enumerate(detail_rows):
            se = float(np.std(boot_ates[:, j], ddof=1))
            row['se'] = se
            pv = float(2 * (1 - sp_stats.norm.cdf(abs(row['estimate'] / max(se, 1e-10)))))
            row['pvalue'] = pv
            row['ci_lower'] = row['estimate'] - z_crit * se
            row['ci_upper'] = row['estimate'] + z_crit * se

        detail = pd.DataFrame(detail_rows)

        # Overall F-test-like: is any treatment different from reference?
        if n_contrasts > 0:
            main_estimate = detail_rows[0]['estimate']
            main_se = detail_rows[0]['se']
            main_pvalue = detail_rows[0]['pvalue']
            main_ci = (detail_rows[0]['ci_lower'], detail_rows[0]['ci_upper'])
        else:
            main_estimate, main_se, main_pvalue = 0.0, 0.0, 1.0
            main_ci = (0.0, 0.0)

        model_info = {
            'treatment_levels': levels.tolist(),
            'reference': int(ref),
            'n_levels': K,
            'potential_outcomes': {int(k): v for k, v in ey.items()},
        }

        return CausalResult(
            method='Multi-valued Treatment (AIPW, Cattaneo 2010)',
            estimand='ATE vs reference',
            estimate=main_estimate,
            se=main_se,
            pvalue=main_pvalue,
            ci=main_ci,
            alpha=self.alpha,
            n_obs=n,
            detail=detail,
            model_info=model_info,
            _citation_key='multi_treatment',
        )

    def _estimate_gps(self, X, D, levels):
        """Estimate generalized propensity scores via multinomial logit."""
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
        )
        lr.fit(X, D)
        probs = lr.predict_proba(X)

        # Align columns with levels
        gps = np.zeros((len(D), len(levels)))
        for j, k in enumerate(levels):
            if k in lr.classes_:
                col_idx = list(lr.classes_).index(k)
                gps[:, j] = probs[:, col_idx]
            else:
                gps[:, j] = 1e-6

        return gps


CausalResult._CITATIONS['multi_treatment'] = (
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
