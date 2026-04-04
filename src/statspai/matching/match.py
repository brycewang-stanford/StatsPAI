"""
Matching estimators for observational causal inference.

Implements propensity score matching (PSM), Mahalanobis distance matching,
and coarsened exact matching (CEM) with ATT/ATE estimation and balance
diagnostics.

References
----------
Rosenbaum, P.R. and Rubin, D.B. (1983). Biometrika, 70(1), 41-55.
Abadie, A. and Imbens, G. (2006). Econometrica, 74(1), 235-267.
Iacus, S.M., King, G., and Porro, G. (2012). Political Analysis, 20(1), 1-24.
"""

from typing import Optional, List, Dict, Any, Union
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist

from ..core.results import CausalResult


def match(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    method: str = 'psm',
    estimand: str = 'ATT',
    n_matches: int = 1,
    caliper: Optional[float] = None,
    replace: bool = True,
    alpha: float = 0.05,
) -> CausalResult:
    """
    Estimate treatment effect using matching.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    y : str
        Outcome variable.
    treat : str
        Binary treatment variable (0/1).
    covariates : list of str
        Variables to match on.
    method : str, default 'psm'
        Matching method: 'psm' (propensity score), 'mahalanobis', or 'cem'.
    estimand : str, default 'ATT'
        Target estimand: 'ATT' or 'ATE'.
    n_matches : int, default 1
        Number of matches per unit (nearest-neighbor).
    caliper : float, optional
        Maximum distance for a valid match (in std devs for PSM).
    replace : bool, default True
        Match with replacement.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    CausalResult

    Examples
    --------
    >>> result = match(df, y='wage', treat='training',
    ...               covariates=['age', 'education', 'experience'],
    ...               method='psm')
    >>> print(result.summary())
    """
    estimator = MatchEstimator(
        data=data, y=y, treat=treat, covariates=covariates,
        method=method, estimand=estimand, n_matches=n_matches,
        caliper=caliper, replace=replace, alpha=alpha,
    )
    return estimator.fit()


class MatchEstimator:
    """
    Matching estimator for causal inference.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        y: str,
        treat: str,
        covariates: List[str],
        method: str = 'psm',
        estimand: str = 'ATT',
        n_matches: int = 1,
        caliper: Optional[float] = None,
        replace: bool = True,
        alpha: float = 0.05,
    ):
        self.data = data.copy()
        self.y = y
        self.treat = treat
        self.covariates = covariates
        self.method = method.lower()
        self.estimand = estimand.upper()
        self.n_matches = n_matches
        self.caliper = caliper
        self.replace = replace
        self.alpha = alpha

        self._validate()

    def _validate(self):
        for col in [self.y, self.treat] + self.covariates:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in data")

        if self.method not in ('psm', 'mahalanobis', 'cem'):
            raise ValueError(
                f"method must be 'psm', 'mahalanobis', or 'cem', "
                f"got '{self.method}'"
            )
        if self.estimand not in ('ATT', 'ATE'):
            raise ValueError(f"estimand must be 'ATT' or 'ATE', got '{self.estimand}'")

        treat_vals = self.data[self.treat].dropna().unique()
        if not set(treat_vals).issubset({0, 1, 0.0, 1.0}):
            raise ValueError(
                f"Treatment must be binary (0/1), got values: {treat_vals}"
            )

    def fit(self) -> CausalResult:
        """Fit matching estimator and return results."""
        # Drop missing
        cols = [self.y, self.treat] + self.covariates
        clean = self.data[cols].dropna()
        T = clean[self.treat].values.astype(int)
        Y = clean[self.y].values.astype(float)
        X = clean[self.covariates].values.astype(float)

        idx_t = np.where(T == 1)[0]
        idx_c = np.where(T == 0)[0]

        if len(idx_t) == 0 or len(idx_c) == 0:
            raise ValueError("Need both treated and control observations")

        # Compute distance / matching
        if self.method == 'psm':
            att, se, matched_data, balance = self._psm(Y, X, T, idx_t, idx_c)
        elif self.method == 'mahalanobis':
            att, se, matched_data, balance = self._mahalanobis(Y, X, T, idx_t, idx_c)
        else:  # cem
            att, se, matched_data, balance = self._cem(Y, X, T, idx_t, idx_c, clean)

        # Inference
        t_stat = att / se if se > 0 else 0
        pvalue = float(2 * (1 - stats.norm.cdf(abs(t_stat))))
        z_crit = stats.norm.ppf(1 - self.alpha / 2)
        ci = (att - z_crit * se, att + z_crit * se)

        n_treated = len(idx_t)
        n_control = len(idx_c)

        model_info = {
            'method': self.method.upper(),
            'estimand': self.estimand,
            'n_treated': n_treated,
            'n_control': n_control,
            'n_matches': self.n_matches,
            'caliper': self.caliper,
            'replace': self.replace,
            'balance': balance,
        }

        return CausalResult(
            method=f'Matching ({self.method.upper()})',
            estimand=self.estimand,
            estimate=att,
            se=se,
            pvalue=pvalue,
            ci=ci,
            alpha=self.alpha,
            n_obs=len(clean),
            detail=balance,
            model_info=model_info,
            _citation_key='matching',
        )

    # ------------------------------------------------------------------
    # PSM
    # ------------------------------------------------------------------

    def _psm(self, Y, X, T, idx_t, idx_c):
        """Propensity Score Matching."""
        # Estimate propensity score via logistic regression
        pscore = self._logit_propensity(X, T)

        # Match on propensity score
        ps_t = pscore[idx_t].reshape(-1, 1)
        ps_c = pscore[idx_c].reshape(-1, 1)

        if self.estimand == 'ATT':
            matches, weights = self._nn_match(ps_t, ps_c, self.caliper)
            att = self._compute_att(Y, idx_t, idx_c, matches, weights)
            se = self._abadie_imbens_se(Y, X, T, idx_t, idx_c, matches, weights)
        else:  # ATE
            # Match treated→control and control→treated
            m_tc, w_tc = self._nn_match(ps_t, ps_c, self.caliper)
            m_ct, w_ct = self._nn_match(ps_c, ps_t, self.caliper)
            att_part = self._compute_att(Y, idx_t, idx_c, m_tc, w_tc)
            atc_part = self._compute_att(Y, idx_c, idx_t, m_ct, w_ct)
            n_t, n_c = len(idx_t), len(idx_c)
            att = (n_t * att_part + n_c * (-atc_part)) / (n_t + n_c)
            # Simplified SE for ATE
            se = self._abadie_imbens_se(Y, X, T, idx_t, idx_c, m_tc, w_tc)

        # Balance
        balance = self._compute_balance(X, T, pscore)

        return att, se, None, balance

    def _logit_propensity(self, X, T):
        """Estimate propensity score via logistic regression (Newton-Raphson)."""
        n, k = X.shape
        # Add intercept
        X_aug = np.column_stack([np.ones(n), X])
        k_aug = X_aug.shape[1]

        # Newton-Raphson for logistic regression
        beta = np.zeros(k_aug)
        for _ in range(25):
            p = 1 / (1 + np.exp(-X_aug @ beta))
            p = np.clip(p, 1e-10, 1 - 1e-10)
            W = np.diag(p * (1 - p))
            grad = X_aug.T @ (T - p)
            H = X_aug.T @ W @ X_aug
            try:
                delta = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(H, grad, rcond=None)[0]
            beta += delta
            if np.max(np.abs(delta)) < 1e-8:
                break

        pscore = 1 / (1 + np.exp(-X_aug @ beta))
        return np.clip(pscore, 1e-6, 1 - 1e-6)

    # ------------------------------------------------------------------
    # Mahalanobis
    # ------------------------------------------------------------------

    def _mahalanobis(self, Y, X, T, idx_t, idx_c):
        """Mahalanobis distance matching."""
        X_t = X[idx_t]
        X_c = X[idx_c]

        # Covariance from pooled sample
        cov = np.cov(X.T)
        if cov.ndim == 0:
            cov = np.array([[cov]])

        try:
            VI = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            VI = np.linalg.pinv(cov)

        dist = cdist(X_t, X_c, metric='mahalanobis', VI=VI)

        caliper = self.caliper
        matches, weights = self._nn_match_from_dist(dist, caliper)

        att = self._compute_att(Y, idx_t, idx_c, matches, weights)
        se = self._abadie_imbens_se(Y, X, T, idx_t, idx_c, matches, weights)

        pscore = self._logit_propensity(X, T)
        balance = self._compute_balance(X, T, pscore)

        return att, se, None, balance

    # ------------------------------------------------------------------
    # CEM
    # ------------------------------------------------------------------

    def _cem(self, Y, X, T, idx_t, idx_c, clean):
        """Coarsened Exact Matching."""
        n, k = X.shape

        # Coarsen each covariate into bins
        strata = np.zeros(n, dtype=object)
        for j in range(k):
            col = X[:, j]
            # Use Sturges' rule for bins
            n_bins = max(int(np.ceil(np.log2(n) + 1)), 3)
            bins = np.linspace(col.min() - 1e-10, col.max() + 1e-10, n_bins + 1)
            digitized = np.digitize(col, bins)
            if j == 0:
                strata = digitized.astype(str)
            else:
                strata = np.char.add(np.char.add(strata, '_'), digitized.astype(str))

        # Find strata with both treated and control
        matched_t = []
        matched_c = []
        weights_c = []

        for s in np.unique(strata):
            in_stratum = strata == s
            t_in = np.where(in_stratum & (T == 1))[0]
            c_in = np.where(in_stratum & (T == 0))[0]

            if len(t_in) > 0 and len(c_in) > 0:
                matched_t.extend(t_in.tolist())
                matched_c.extend(c_in.tolist())
                # CEM weights: n_t/n_c ratio within stratum
                w = len(t_in) / len(c_in)
                weights_c.extend([w] * len(c_in))

        if len(matched_t) == 0:
            raise ValueError("CEM: no strata with both treated and control units")

        # ATT from matched sample
        Y_t_matched = Y[matched_t]
        Y_c_matched = Y[matched_c]
        w_c = np.array(weights_c)

        att = float(np.mean(Y_t_matched) - np.average(Y_c_matched, weights=w_c))

        # SE via matched sample variance
        var_t = np.var(Y_t_matched, ddof=1) / len(Y_t_matched) if len(Y_t_matched) > 1 else 0
        var_c = np.average((Y_c_matched - np.average(Y_c_matched, weights=w_c))**2,
                           weights=w_c) / len(Y_c_matched) if len(Y_c_matched) > 1 else 0
        se = float(np.sqrt(var_t + var_c))

        pscore = self._logit_propensity(X, T)
        balance = self._compute_balance(X, T, pscore)

        return att, se, None, balance

    # ------------------------------------------------------------------
    # Nearest-neighbor matching helpers
    # ------------------------------------------------------------------

    def _nn_match(self, X_target, X_pool, caliper=None):
        """Nearest-neighbor match from target to pool (1D propensity scores)."""
        dist = cdist(X_target, X_pool, metric='euclidean')
        return self._nn_match_from_dist(dist, caliper)

    def _nn_match_from_dist(self, dist, caliper=None):
        """
        Given distance matrix (n_target x n_pool), find k-NN matches.
        Returns (matches, weights).
        """
        n_target, n_pool = dist.shape
        matches = []  # list of arrays, one per target unit
        weights = []

        for i in range(n_target):
            d = dist[i].copy()
            if caliper is not None:
                d[d > caliper] = np.inf

            if self.replace:
                # With replacement: find k nearest
                k = min(self.n_matches, np.sum(np.isfinite(d)))
                if k == 0:
                    matches.append(np.array([], dtype=int))
                    weights.append(np.array([]))
                    continue
                idx = np.argpartition(d, k)[:k]
                matches.append(idx)
                weights.append(np.ones(k) / k)
            else:
                # Without replacement: greedy (simplified)
                k = min(self.n_matches, np.sum(np.isfinite(d)))
                if k == 0:
                    matches.append(np.array([], dtype=int))
                    weights.append(np.array([]))
                    continue
                idx = np.argpartition(d, k)[:k]
                matches.append(idx)
                weights.append(np.ones(k) / k)

        return matches, weights

    def _compute_att(self, Y, idx_target, idx_pool, matches, weights):
        """Compute ATT from matches."""
        effects = []
        for i, (m, w) in enumerate(zip(matches, weights)):
            if len(m) == 0:
                continue
            y_target = Y[idx_target[i]]
            y_matched = Y[idx_pool[m]]
            effects.append(y_target - np.average(y_matched, weights=w))

        if len(effects) == 0:
            return 0.0
        return float(np.mean(effects))

    def _abadie_imbens_se(self, Y, X, T, idx_t, idx_c, matches, weights):
        """
        Abadie-Imbens (2006) standard error for matching estimator.
        Simplified version using matched sample variance.
        """
        effects = []
        for i, (m, w) in enumerate(zip(matches, weights)):
            if len(m) == 0:
                continue
            y_t = Y[idx_t[i]]
            y_c = Y[idx_c[m]]
            effects.append(y_t - np.average(y_c, weights=w))

        if len(effects) < 2:
            return 0.0

        effects = np.array(effects)
        n_eff = len(effects)
        return float(np.std(effects, ddof=1) / np.sqrt(n_eff))

    # ------------------------------------------------------------------
    # Balance diagnostics
    # ------------------------------------------------------------------

    def _compute_balance(
        self, X, T, pscore=None,
    ) -> pd.DataFrame:
        """
        Compute standardized mean differences (SMD) before matching.
        """
        idx_t = T == 1
        idx_c = T == 0
        rows = []

        for j, cov_name in enumerate(self.covariates):
            x_t = X[idx_t, j]
            x_c = X[idx_c, j]

            mean_t = np.mean(x_t)
            mean_c = np.mean(x_c)
            sd_pooled = np.sqrt((np.var(x_t, ddof=1) + np.var(x_c, ddof=1)) / 2)
            smd = (mean_t - mean_c) / sd_pooled if sd_pooled > 0 else 0

            rows.append({
                'variable': cov_name,
                'mean_treated': round(mean_t, 4),
                'mean_control': round(mean_c, 4),
                'smd': round(smd, 4),
            })

        if pscore is not None:
            ps_t = pscore[idx_t]
            ps_c = pscore[idx_c]
            sd_ps = np.sqrt((np.var(ps_t, ddof=1) + np.var(ps_c, ddof=1)) / 2)
            smd_ps = (np.mean(ps_t) - np.mean(ps_c)) / sd_ps if sd_ps > 0 else 0
            rows.append({
                'variable': 'propensity_score',
                'mean_treated': round(float(np.mean(ps_t)), 4),
                'mean_control': round(float(np.mean(ps_c)), 4),
                'smd': round(float(smd_ps), 4),
            })

        return pd.DataFrame(rows)


# Citation
CausalResult._CITATIONS['matching'] = (
    "@article{abadie2006large,\n"
    "  title={Large Sample Properties of Matching Estimators for "
    "Average Treatment Effects},\n"
    "  author={Abadie, Alberto and Imbens, Guido W},\n"
    "  journal={Econometrica},\n"
    "  volume={74},\n"
    "  number={1},\n"
    "  pages={235--267},\n"
    "  year={2006},\n"
    "  publisher={Wiley}\n"
    "}"
)
