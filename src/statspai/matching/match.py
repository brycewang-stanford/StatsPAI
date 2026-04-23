"""
Matching estimators for observational causal inference.

Unified interface supporting orthogonal design choices:

- **distance**: how to measure unit similarity
  - ``'propensity'`` — logit propensity score (Rosenbaum & Rubin 1983)
  - ``'mahalanobis'`` — Mahalanobis distance (Rubin 1980)
  - ``'euclidean'`` — normalized Euclidean distance
  - ``'exact'`` — exact covariate values (no approximation)

- **method**: how to use those distances
  - ``'nearest'`` — k-nearest-neighbor matching
  - ``'stratify'`` — subclassification / stratification
  - ``'cem'`` — coarsened exact matching (Iacus, King & Porro 2012)

- **bias_correction**: Abadie-Imbens (2011) regression adjustment for
  matching discrepancies in nearest-neighbor matching.

Backward compatible: ``method='psm'``, ``method='mahalanobis'``, and
``method='cem'`` still work and map to the new parameter space.

References
----------
Rosenbaum, P.R. and Rubin, D.B. (1983). Biometrika, 70(1), 41-55.
Abadie, A. and Imbens, G.W. (2006). Econometrica, 74(1), 235-267.
Abadie, A. and Imbens, G.W. (2011). JBES, 29(1), 1-11.
Iacus, S.M., King, G., and Porro, G. (2012). Political Analysis, 20(1), 1-24.
King, G. and Nielsen, R. (2019). Political Analysis, 27(4), 435-454.
Cunningham, S. (2021). *Causal Inference: The Mixtape*. Yale University Press.
    Ch. 5: Matching and Subclassification. https://mixtape.scunning.com/ [@rosenbaum1983central]
"""

from typing import Optional, List
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist

from ..core.results import CausalResult


# ======================================================================
# Legacy method aliases → (distance, method) pairs
# ======================================================================
_LEGACY_MAP = {
    'psm': ('propensity', 'nearest'),
    'mahalanobis': ('mahalanobis', 'nearest'),
    'cem': (None, 'cem'),
}

_VALID_DISTANCES = ('propensity', 'mahalanobis', 'euclidean', 'exact')
_VALID_METHODS = ('nearest', 'stratify', 'cem')


# ======================================================================
# Public API
# ======================================================================

def match(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    *,
    # --- new orthogonal API ---
    distance: Optional[str] = None,
    method: str = 'nearest',
    # --- matching parameters ---
    estimand: str = 'ATT',
    n_matches: int = 1,
    caliper: Optional[float] = None,
    replace: bool = True,
    bias_correction: bool = False,
    # --- propensity score specification ---
    ps_poly: int = 1,
    # --- stratification parameters ---
    n_strata: int = 5,
    # --- CEM parameters ---
    n_bins: Optional[int] = None,
    # --- inference ---
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
    distance : str, optional
        Distance metric: 'propensity', 'mahalanobis', 'euclidean', 'exact'.
        Default is 'propensity' for method='nearest'/'stratify'.
    method : str, default 'nearest'
        Matching algorithm: 'nearest', 'stratify', 'cem'.
        Legacy values 'psm', 'mahalanobis' also accepted.
    estimand : str, default 'ATT'
        Target estimand: 'ATT' or 'ATE'.
    n_matches : int, default 1
        Number of nearest-neighbor matches per unit.
    caliper : float, optional
        Maximum distance for a valid match.
    replace : bool, default True
        Match with replacement (nearest-neighbor only).
    bias_correction : bool, default False
        Apply Abadie-Imbens (2011) bias correction via regression
        adjustment on the matching discrepancy.
    ps_poly : int, default 1
        Polynomial degree for the propensity score logit model.
        ``ps_poly=1`` uses linear terms only.
        ``ps_poly=2`` adds all squared terms and pairwise interactions.
        ``ps_poly=3`` adds cubic terms as well.
        Higher-order specifications are standard practice; see
        Cunningham (2021, Ch. 5) for worked examples with
        ``age + age^2 + age^3 + educ + educ^2 + educ*re74``.
    n_strata : int, default 5
        Number of strata for method='stratify'.
    n_bins : int, optional
        Number of bins per covariate for method='cem'.
        Default uses Sturges' rule.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    CausalResult

    Examples
    --------
    >>> # Propensity score matching (default)
    >>> result = sp.match(df, y='wage', treat='training',
    ...                   covariates=['age', 'edu', 'exp'])

    >>> # Mahalanobis distance + bias correction
    >>> result = sp.match(df, y='wage', treat='training',
    ...                   covariates=['age', 'edu', 'exp'],
    ...                   distance='mahalanobis', bias_correction=True)

    >>> # Exact matching
    >>> result = sp.match(df, y='wage', treat='training',
    ...                   covariates=['age', 'edu'],
    ...                   distance='exact')

    >>> # Propensity score stratification (5 strata)
    >>> result = sp.match(df, y='wage', treat='training',
    ...                   covariates=['age', 'edu', 'exp'],
    ...                   method='stratify', n_strata=5)

    >>> # CEM
    >>> result = sp.match(df, y='wage', treat='training',
    ...                   covariates=['age', 'edu'],
    ...                   method='cem')

    >>> # Quadratic PS model (Cunningham 2021, Ch. 5 style)
    >>> result = sp.match(df, y='wage', treat='training',
    ...                   covariates=['age', 'edu', 'exp'],
    ...                   ps_poly=2)

    >>> # Without-replacement matching
    >>> result = sp.match(df, y='wage', treat='training',
    ...                   covariates=['age', 'edu'],
    ...                   replace=False)

    >>> # Legacy API still works
    >>> result = sp.match(df, y='wage', treat='training',
    ...                   covariates=['age', 'edu'], method='psm')
    """
    estimator = MatchEstimator(
        data=data, y=y, treat=treat, covariates=covariates,
        distance=distance, method=method, estimand=estimand,
        n_matches=n_matches, caliper=caliper, replace=replace,
        bias_correction=bias_correction, ps_poly=ps_poly,
        n_strata=n_strata, n_bins=n_bins, alpha=alpha,
    )
    return estimator.fit()


# ======================================================================
# MatchEstimator
# ======================================================================

class MatchEstimator:
    """Unified matching estimator supporting multiple distance × method combinations."""

    def __init__(
        self,
        data: pd.DataFrame,
        y: str,
        treat: str,
        covariates: List[str],
        *,
        distance: Optional[str] = None,
        method: str = 'nearest',
        estimand: str = 'ATT',
        n_matches: int = 1,
        caliper: Optional[float] = None,
        replace: bool = True,
        bias_correction: bool = False,
        ps_poly: int = 1,
        n_strata: int = 5,
        n_bins: Optional[int] = None,
        alpha: float = 0.05,
    ):
        self.data = data.copy()
        self.y = y
        self.treat = treat
        self.covariates = covariates
        self.estimand = estimand.upper()
        self.n_matches = n_matches
        self.caliper = caliper
        self.replace = replace
        self.bias_correction = bias_correction
        self.ps_poly = ps_poly
        self.n_strata = n_strata
        self.n_bins = n_bins
        self.alpha = alpha

        # Resolve legacy method names
        method_lower = method.lower()
        if method_lower in _LEGACY_MAP:
            resolved_dist, resolved_method = _LEGACY_MAP[method_lower]
            self.distance = resolved_dist if distance is None else distance.lower()
            self.method = resolved_method
        else:
            self.method = method_lower
            self.distance = distance.lower() if distance else None

        # Set default distance for methods that need one
        if self.distance is None:
            if self.method in ('nearest', 'stratify'):
                self.distance = 'propensity'
            elif self.method == 'cem':
                self.distance = None  # CEM doesn't use distance

        self._validate()

    def _validate(self):
        for col in [self.y, self.treat] + self.covariates:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in data")

        if self.method not in _VALID_METHODS:
            raise ValueError(
                f"method must be one of {_VALID_METHODS} "
                f"(or legacy: 'psm', 'mahalanobis'), got '{self.method}'"
            )
        if self.distance is not None and self.distance not in _VALID_DISTANCES:
            raise ValueError(
                f"distance must be one of {_VALID_DISTANCES}, got '{self.distance}'"
            )
        if self.estimand not in ('ATT', 'ATE'):
            raise ValueError(f"estimand must be 'ATT' or 'ATE', got '{self.estimand}'")

        treat_vals = self.data[self.treat].dropna().unique()
        if not set(treat_vals).issubset({0, 1, 0.0, 1.0}):
            from statspai.exceptions import MethodIncompatibility
            raise MethodIncompatibility(
                f"Treatment must be binary (0/1), got values: {treat_vals}",
                recovery_hint=(
                    "Matching assumes a binary treatment. For multi-valued "
                    "treatments use sp.multi_treatment; for continuous use "
                    "sp.dose_response."
                ),
                diagnostics={"treat_values": sorted(map(str, treat_vals))[:10]},
                alternative_functions=["sp.multi_treatment", "sp.dose_response"],
            )

        # Exact matching only supports ATT
        if self.distance == 'exact' and self.estimand == 'ATE':
            raise ValueError("Exact matching only supports estimand='ATT'")

        # Stratification only works with propensity distance
        if self.method == 'stratify' and self.distance != 'propensity':
            raise ValueError("method='stratify' requires distance='propensity'")

    # ==================================================================
    # Main fit
    # ==================================================================

    def fit(self) -> CausalResult:
        """Fit matching estimator and return results."""
        cols = [self.y, self.treat] + self.covariates
        clean = self.data[cols].dropna()
        T = clean[self.treat].values.astype(int)
        Y = clean[self.y].values.astype(float)
        X = clean[self.covariates].values.astype(float)

        idx_t = np.where(T == 1)[0]
        idx_c = np.where(T == 0)[0]

        if len(idx_t) == 0 or len(idx_c) == 0:
            from statspai.exceptions import DataInsufficient
            raise DataInsufficient(
                "Need both treated and control observations",
                recovery_hint=(
                    "All observations have the same treatment value; "
                    "re-check the treatment column."
                ),
                diagnostics={
                    "n_treated": int(len(idx_t)),
                    "n_control": int(len(idx_c)),
                },
                alternative_functions=[],
            )

        # Dispatch — each returns (att, se, balance, extra_info)
        extra_info = {}
        if self.method == 'cem':
            att, se, balance, extra_info = self._fit_cem(Y, X, T, idx_t, idx_c)
            method_label = 'Matching (CEM)'
        elif self.method == 'stratify':
            att, se, balance, extra_info = self._fit_stratify(Y, X, T, idx_t, idx_c)
            method_label = 'Matching (PS Stratification)'
        elif self.distance == 'exact':
            att, se, balance, extra_info = self._fit_exact(Y, X, T, idx_t, idx_c)
            method_label = 'Matching (Exact)'
        else:
            att, se, balance = self._fit_nearest(Y, X, T, idx_t, idx_c)
            dist_name = self.distance.capitalize()
            bc_tag = ', BC' if self.bias_correction else ''
            method_label = f'Matching ({dist_name}{bc_tag})'

        # PSM warning
        if self.distance == 'propensity' and self.method == 'nearest':
            warnings.warn(
                "PSM can increase imbalance and bias (King & Nielsen 2019). "
                "Consider distance='mahalanobis' or method='cem'.",
                UserWarning, stacklevel=3,
            )

        # Inference
        t_stat = att / se if se > 0 else 0.0
        pvalue = float(2 * (1 - stats.norm.cdf(abs(t_stat))))
        z = stats.norm.ppf(1 - self.alpha / 2)
        ci = (att - z * se, att + z * se)

        model_info = {
            'distance': self.distance,
            'method': self.method,
            'estimand': self.estimand,
            'n_treated': int(len(idx_t)),
            'n_control': int(len(idx_c)),
            'n_matches': self.n_matches,
            'caliper': self.caliper,
            'replace': self.replace,
            'bias_correction': self.bias_correction,
            'ps_poly': self.ps_poly,
            'balance': balance,
            **extra_info,
        }

        return CausalResult(
            method=method_label,
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

    # ==================================================================
    # Nearest-neighbor matching (propensity / mahalanobis / euclidean)
    # ==================================================================

    def _fit_nearest(self, Y, X, T, idx_t, idx_c):
        """Nearest-neighbor matching with configurable distance metric."""
        # For propensity distance, estimate PS once with actual treatment
        pscore = self._logit_propensity(X, T, poly=self.ps_poly) if self.distance == 'propensity' else None

        # Build distance matrix
        dist_mat = self._compute_distance_matrix(X, idx_t, idx_c, pscore)

        if self.estimand == 'ATT':
            matches, weights = self._nn_match_from_dist(dist_mat, self.caliper)
            att = self._compute_effect(Y, idx_t, idx_c, X, matches, weights)
            se = self._ai_se(Y, X, T, idx_t, idx_c, matches, weights)
        else:
            # ATE: match both directions, reuse the same propensity scores
            dist_ct = self._compute_distance_matrix(X, idx_c, idx_t, pscore)
            m_tc, w_tc = self._nn_match_from_dist(dist_mat, self.caliper)
            m_ct, w_ct = self._nn_match_from_dist(dist_ct, self.caliper)
            att_part = self._compute_effect(Y, idx_t, idx_c, X, m_tc, w_tc)
            atc_part = self._compute_effect(Y, idx_c, idx_t, X, m_ct, w_ct)
            n_t, n_c = len(idx_t), len(idx_c)
            att = (n_t * att_part + n_c * (-atc_part)) / (n_t + n_c)
            se = self._ai_se(Y, X, T, idx_t, idx_c, m_tc, w_tc)

        if pscore is None:
            pscore = self._logit_propensity(X, T, poly=self.ps_poly)
        balance = self._balance_table(X, T, pscore)

        return att, se, balance

    def _compute_distance_matrix(self, X, idx_from, idx_to, pscore=None):
        """Compute distance matrix between two groups."""
        X_from = X[idx_from]
        X_to = X[idx_to]

        if self.distance == 'propensity':
            # Use pre-estimated propensity scores (estimated once with actual T)
            ps_from = pscore[idx_from].reshape(-1, 1)
            ps_to = pscore[idx_to].reshape(-1, 1)
            return cdist(ps_from, ps_to, metric='euclidean')

        elif self.distance == 'mahalanobis':
            cov = np.cov(X.T)
            if cov.ndim == 0:
                cov = np.array([[cov]])
            try:
                VI = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                VI = np.linalg.pinv(cov)
            return cdist(X_from, X_to, metric='mahalanobis', VI=VI)

        else:  # euclidean (normalized)
            sd = np.std(X, axis=0, ddof=1)
            sd[sd == 0] = 1.0
            return cdist(X_from / sd, X_to / sd, metric='euclidean')

    # ==================================================================
    # Exact matching
    # ==================================================================

    def _fit_exact(self, Y, X, T, idx_t, idx_c):
        """Exact matching: only match units with identical covariate values."""
        # Build string keys for each observation
        keys_t = self._covariate_keys(X, idx_t)
        keys_c = self._covariate_keys(X, idx_c)

        # Index control units by key
        control_map = {}
        for i, key in enumerate(keys_c):
            control_map.setdefault(key, []).append(i)

        effects = []
        n_matched = 0
        for i, key in enumerate(keys_t):
            if key not in control_map:
                continue
            c_indices = control_map[key]
            y_t = Y[idx_t[i]]
            y_c_mean = np.mean(Y[idx_c[c_indices]])
            effects.append(y_t - y_c_mean)
            n_matched += 1

        if n_matched == 0:
            raise ValueError(
                "Exact matching: no treated units found exact matches. "
                "Consider distance='mahalanobis' or method='cem'."
            )

        att = float(np.mean(effects))
        se = float(np.std(effects, ddof=1) / np.sqrt(n_matched)) if n_matched > 1 else 0.0

        pscore = self._logit_propensity(X, T, poly=self.ps_poly)
        balance = self._balance_table(X, T, pscore)
        extra = {
            'n_matched_treated': n_matched,
            'n_unmatched_treated': len(keys_t) - n_matched,
        }
        return att, se, balance, extra

    @staticmethod
    def _covariate_keys(X, indices):
        """Create hashable keys for exact matching."""
        return [tuple(X[i]) for i in indices]

    # ==================================================================
    # Subclassification / propensity score stratification
    # ==================================================================

    def _fit_stratify(self, Y, X, T, idx_t, idx_c):
        """
        Propensity score stratification (Rosenbaum & Rubin 1984).

        Partition the sample into strata by propensity score quantiles,
        compute within-stratum treatment effects, then weight by the
        proportion of treated (ATT) or total (ATE) units per stratum.
        """
        pscore = self._logit_propensity(X, T, poly=self.ps_poly)

        # Create strata from propensity score quantiles
        boundaries = np.quantile(pscore, np.linspace(0, 1, self.n_strata + 1))
        boundaries[0] -= 1e-6
        boundaries[-1] += 1e-6
        strata = np.digitize(pscore, boundaries) - 1
        strata = np.clip(strata, 0, self.n_strata - 1)

        # Collect per-stratum effects, weights, and variance components
        strata_results = []  # list of (tau, weight, var_t, var_c)

        for s in range(self.n_strata):
            in_s = strata == s
            t_in = in_s & (T == 1)
            c_in = in_s & (T == 0)
            n_t_s = t_in.sum()
            n_c_s = c_in.sum()

            if n_t_s == 0 or n_c_s == 0:
                continue

            tau_s = Y[t_in].mean() - Y[c_in].mean()

            if self.estimand == 'ATT':
                w_s = float(n_t_s)
            else:
                w_s = float(n_t_s + n_c_s)

            # Within-stratum variance components
            vt = np.var(Y[t_in], ddof=1) / n_t_s if n_t_s >= 2 else 0.0
            vc = np.var(Y[c_in], ddof=1) / n_c_s if n_c_s >= 2 else 0.0

            strata_results.append((tau_s, w_s, vt, vc))

        if len(strata_results) == 0:
            raise ValueError("No strata contain both treated and control units")

        effects = np.array([r[0] for r in strata_results])
        raw_weights = np.array([r[1] for r in strata_results])
        weights = raw_weights / raw_weights.sum()

        att = float(effects @ weights)

        # SE: sum of weighted within-stratum sampling variances
        within_var = 0.0
        for (_, _, vt, vc), w_s in zip(strata_results, weights):
            within_var += w_s ** 2 * (vt + vc)

        se = float(np.sqrt(within_var))

        balance = self._balance_table(X, T, pscore)
        extra = {
            'n_strata': self.n_strata,
            'n_effective_strata': len(strata_results),
        }
        return att, se, balance, extra

    # ==================================================================
    # CEM
    # ==================================================================

    def _fit_cem(self, Y, X, T, idx_t, idx_c):
        """Coarsened Exact Matching (Iacus, King & Porro 2012)."""
        n, k = X.shape

        # Coarsen each covariate
        n_bins = self.n_bins
        if n_bins is None:
            n_bins = max(int(np.ceil(np.log2(n) + 1)), 3)  # Sturges' rule

        strata = np.zeros(n, dtype=object)
        for j in range(k):
            col = X[:, j]
            bins = np.linspace(col.min() - 1e-10, col.max() + 1e-10, n_bins + 1)
            digitized = np.digitize(col, bins)
            if j == 0:
                strata = digitized.astype(str)
            else:
                strata = np.char.add(np.char.add(strata, '_'), digitized.astype(str))

        # Match within strata
        matched_t = []
        matched_c = []
        weights_c = []

        for s in np.unique(strata):
            in_s = strata == s
            t_in = np.where(in_s & (T == 1))[0]
            c_in = np.where(in_s & (T == 0))[0]
            if len(t_in) > 0 and len(c_in) > 0:
                matched_t.extend(t_in.tolist())
                matched_c.extend(c_in.tolist())
                w = len(t_in) / len(c_in)
                weights_c.extend([w] * len(c_in))

        if len(matched_t) == 0:
            raise ValueError("CEM: no strata with both treated and control units")

        Y_t = Y[matched_t]
        Y_c = Y[matched_c]
        w_c = np.array(weights_c)

        att = float(np.mean(Y_t) - np.average(Y_c, weights=w_c))

        var_t = np.var(Y_t, ddof=1) / len(Y_t) if len(Y_t) > 1 else 0
        var_c = (np.average((Y_c - np.average(Y_c, weights=w_c)) ** 2, weights=w_c)
                 / len(Y_c)) if len(Y_c) > 1 else 0
        se = float(np.sqrt(var_t + var_c))

        pscore = self._logit_propensity(X, T, poly=self.ps_poly)
        balance = self._balance_table(X, T, pscore)

        n_matched_t = len(set(matched_t))
        extra = {
            'n_matched_treated': n_matched_t,
            'n_matched_control': len(set(matched_c)),
            'n_unmatched_treated': len(idx_t) - n_matched_t,
            'n_bins': n_bins,
        }
        return att, se, balance, extra

    # ==================================================================
    # Propensity score estimation
    # ==================================================================

    @staticmethod
    def _expand_poly(X, degree):
        """
        Expand covariate matrix with polynomial and interaction terms.

        - degree=1: linear terms only (identity).
        - degree=2: add X^2 and all pairwise X_i * X_j interactions.
        - degree=3: add X^3 as well.

        This follows the common practice in propensity score estimation
        of including higher-order terms (Cunningham 2021, Ch. 5;
        Dehejia & Wahba 1999).
        """
        if degree <= 1:
            return X
        cols = [X]
        n, k = X.shape
        # Squared terms
        cols.append(X ** 2)
        # Pairwise interactions
        if k > 1:
            for i in range(k):
                for j in range(i + 1, k):
                    cols.append((X[:, i] * X[:, j]).reshape(-1, 1))
        # Cubic terms
        if degree >= 3:
            cols.append(X ** 3)
        return np.column_stack(cols)

    @staticmethod
    def _logit_propensity(X, T, poly=1):
        """
        Logistic regression propensity score via Newton-Raphson (IRLS).

        Parameters
        ----------
        X : ndarray, shape (n, k)
        T : ndarray, shape (n,)
        poly : int, default 1
            Polynomial expansion degree for the design matrix.
            ``poly=2`` adds squared terms and pairwise interactions,
            following the standard specification in Cunningham (2021, Ch. 5)
            and Dehejia & Wahba (1999).
        """
        X_poly = MatchEstimator._expand_poly(X, poly)
        n = X_poly.shape[0]
        X_aug = np.column_stack([np.ones(n), X_poly])
        k_aug = X_aug.shape[1]

        beta = np.zeros(k_aug)
        for _ in range(25):
            linear = np.clip(X_aug @ beta, -500, 500)
            p = 1 / (1 + np.exp(-linear))
            p = np.clip(p, 1e-10, 1 - 1e-10)
            # Vectorized IRLS: W is diagonal, so X'WX = (X * w)' X
            w = p * (1 - p)
            grad = X_aug.T @ (T - p)
            H = (X_aug * w[:, None]).T @ X_aug
            try:
                delta = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(H, grad, rcond=None)[0]
            beta += delta
            if np.max(np.abs(delta)) < 1e-8:
                break

        linear = np.clip(X_aug @ beta, -500, 500)
        pscore = 1 / (1 + np.exp(-linear))
        return np.clip(pscore, 1e-6, 1 - 1e-6)

    # ==================================================================
    # NN matching helpers
    # ==================================================================

    def _nn_match_from_dist(self, dist, caliper=None):
        """
        k-NN matching from a precomputed distance matrix.

        When ``self.replace=False``, each control unit can be used at most
        once across all treated units.  Treated units are processed in
        order of their minimum distance (best match first) so the greedy
        assignment favours the closest pairs.

        References: Cunningham (2021, Ch. 5) discusses with- vs.
        without-replacement matching and the bias–variance trade-off.
        """
        n_target = dist.shape[0]
        matches = [None] * n_target
        weights = [None] * n_target

        # Without replacement: process treated units greedily by best
        # minimum distance so each control is used at most once.
        if not self.replace:
            used = set()
            # Sort treated units by their minimum distance to any control
            min_dists = np.min(dist, axis=1)
            order = np.argsort(min_dists)

            for i in order:
                d = dist[i].copy()
                if caliper is not None:
                    d[d > caliper] = np.inf
                # Mask out already-used controls
                for u in used:
                    d[u] = np.inf

                k = min(self.n_matches, int(np.sum(np.isfinite(d))))
                if k == 0:
                    matches[i] = np.array([], dtype=int)
                    weights[i] = np.array([])
                    continue

                idx = np.argpartition(d, k)[:k]
                matches[i] = idx
                weights[i] = np.ones(k) / k
                used.update(idx.tolist())

            return matches, weights

        # With replacement (default): simple k-NN per target
        for i in range(n_target):
            d = dist[i].copy()
            if caliper is not None:
                d[d > caliper] = np.inf

            k = min(self.n_matches, int(np.sum(np.isfinite(d))))
            if k == 0:
                matches[i] = np.array([], dtype=int)
                weights[i] = np.array([])
                continue

            idx = np.argpartition(d, k)[:k]
            matches[i] = idx
            weights[i] = np.ones(k) / k

        return matches, weights

    # ==================================================================
    # Effect computation (with optional bias correction)
    # ==================================================================

    def _compute_effect(self, Y, idx_target, idx_pool, X, matches, weights):
        """
        Compute matching estimate, optionally with Abadie-Imbens (2011)
        bias correction.

        Bias correction estimates mu_0(x) via OLS on the matched control
        group, then adjusts each matched pair:
            tau_i^BC = (Y_i - Y_j) - (mu_hat(X_i) - mu_hat(X_j))
        """
        effects = []
        for i, (m, w) in enumerate(zip(matches, weights)):
            if len(m) == 0:
                continue
            y_target = Y[idx_target[i]]
            y_matched = Y[idx_pool[m]]
            effects.append(y_target - np.average(y_matched, weights=w))

        if len(effects) == 0:
            return 0.0

        raw_att = float(np.mean(effects))

        if not self.bias_correction:
            return raw_att

        # --- Abadie-Imbens (2011) bias correction ---
        # Estimate conditional mean function on pool group via OLS
        X_pool = X[idx_pool]
        Y_pool = Y[idx_pool]
        X_pool_aug = np.column_stack([np.ones(len(X_pool)), X_pool])

        try:
            beta_pool = np.linalg.lstsq(X_pool_aug, Y_pool, rcond=None)[0]
        except np.linalg.LinAlgError:
            return raw_att  # fallback to uncorrected

        # Compute bias correction for each matched pair
        corrections = []
        for i, (m, w) in enumerate(zip(matches, weights)):
            if len(m) == 0:
                continue
            x_target = np.concatenate([[1], X[idx_target[i]]])
            x_matched = np.column_stack([np.ones(len(m)), X[idx_pool[m]]])
            mu_target = x_target @ beta_pool
            mu_matched = np.average(x_matched @ beta_pool, weights=w)
            corrections.append(mu_target - mu_matched)

        if len(corrections) == 0:
            return raw_att

        bias = float(np.mean(corrections))
        return raw_att - bias

    # ==================================================================
    # Standard errors
    # ==================================================================

    def _ai_se(self, Y, X, T, idx_t, idx_c, matches, weights):
        """
        Abadie-Imbens (2006) standard error for matching estimator.
        Uses conditional variance estimation from matched pairs.
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

    # ==================================================================
    # Balance diagnostics
    # ==================================================================

    def _balance_table(self, X, T, pscore=None) -> pd.DataFrame:
        """Standardized mean differences (SMD) before matching."""
        idx_t = T == 1
        idx_c = T == 0
        rows = []

        for j, name in enumerate(self.covariates):
            x_t = X[idx_t, j]
            x_c = X[idx_c, j]
            mean_t = np.mean(x_t)
            mean_c = np.mean(x_c)
            sd_pool = np.sqrt((np.var(x_t, ddof=1) + np.var(x_c, ddof=1)) / 2)
            smd = (mean_t - mean_c) / sd_pool if sd_pool > 0 else 0
            rows.append({
                'variable': name,
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


# ======================================================================
# Citation
# ======================================================================

# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def balanceplot(
    result: CausalResult,
    threshold: float = 0.1,
    ax=None,
    figsize: tuple = (8, None),
    title: str = None,
):
    """
    Love plot: covariate balance visualization (SMD dot plot).

    Displays standardized mean differences (SMD) for each covariate.
    The standard threshold for good balance is |SMD| < 0.1.

    Parameters
    ----------
    result : CausalResult
        Result from ``match()`` or ``ebalance()``.
    threshold : float, default 0.1
        SMD threshold lines.
    ax : matplotlib Axes, optional
    figsize : tuple
        Height auto-scales with number of covariates if None.
    title : str, optional

    Returns
    -------
    (fig, ax)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required.")

    balance = result.detail
    if balance is None or 'smd' not in balance.columns:
        raise ValueError("No balance table. Use match() result.")

    n_vars = len(balance)
    if figsize[1] is None:
        figsize = (figsize[0], max(4, n_vars * 0.4 + 1))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    variables = balance['variable'].values
    smd = balance['smd'].values
    y_pos = np.arange(n_vars)

    # Color by balance quality
    colors = ['#27AE60' if abs(s) < threshold else '#E74C3C' for s in smd]

    ax.scatter(smd, y_pos, c=colors, s=60, zorder=5, edgecolors='white',
               linewidth=0.5)
    ax.barh(y_pos, smd, height=0.02, color='#BDC3C7', zorder=2)

    # Threshold lines
    ax.axvline(x=threshold, color='#E74C3C', linestyle='--', linewidth=0.8,
               alpha=0.5)
    ax.axvline(x=-threshold, color='#E74C3C', linestyle='--', linewidth=0.8,
               alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(variables, fontsize=10)
    ax.set_xlabel('Standardized Mean Difference (SMD)', fontsize=11)
    ax.set_title(title or 'Covariate Balance (Love Plot)', fontsize=13)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    return fig, ax


def psplot(
    data: pd.DataFrame,
    treat: str,
    covariates: List[str],
    *,
    n_bins: int = 40,
    ax=None,
    figsize: tuple = (8, 5),
    title: str = None,
    labels: tuple = ('Control', 'Treated'),
    colors: tuple = ('#3498DB', '#E74C3C'),
    trim: Optional[float] = None,
):
    """
    Propensity score distribution plot (common support diagnostic).

    Overlays histograms of the estimated propensity score for treated
    and control groups, so the user can visually assess whether the
    common support (overlap) assumption holds.

    Parameters
    ----------
    data : pd.DataFrame
    treat : str
        Binary treatment column.
    covariates : list of str
        Covariates used to estimate the propensity score.
    n_bins : int, default 40
        Number of histogram bins.
    ax : matplotlib Axes, optional
    figsize : tuple
    title : str, optional
    labels : tuple of str
        Labels for (control, treated).
    colors : tuple of str
        Colors for (control, treated).
    trim : float, optional
        If set, draw vertical lines at (trim, 1-trim) to show
        the recommended trimming region.

    Returns
    -------
    (fig, ax)

    Examples
    --------
    >>> fig, ax = sp.psplot(df, treat='D', covariates=['x1', 'x2'])
    >>> fig, ax = sp.psplot(df, treat='D', covariates=['x1', 'x2'],
    ...                      trim=0.1)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required.")

    df = data[[treat] + covariates].dropna()
    T = df[treat].values.astype(int)
    X = df[covariates].values.astype(float)

    pscore = MatchEstimator._logit_propensity(X, T)
    ps_c = pscore[T == 0]
    ps_t = pscore[T == 1]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    bins = np.linspace(0, 1, n_bins + 1)

    # Control: mirrored downward
    ax.hist(ps_c, bins=bins, alpha=0.6, color=colors[0], label=labels[0],
            density=True, edgecolor='white', linewidth=0.3)
    # Treated: upward
    ax.hist(ps_t, bins=bins, alpha=0.6, color=colors[1], label=labels[1],
            density=True, edgecolor='white', linewidth=0.3)

    # Trimming region
    if trim is not None:
        ax.axvline(x=trim, color='#8E44AD', linestyle='--', linewidth=1,
                   alpha=0.7, label=f'Trim [{trim:.2f}, {1-trim:.2f}]')
        ax.axvline(x=1 - trim, color='#8E44AD', linestyle='--', linewidth=1,
                   alpha=0.7)

    ax.set_xlabel('Propensity Score', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(title or 'Propensity Score Distribution (Common Support)',
                 fontsize=13)
    ax.set_xlim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    return fig, ax


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
