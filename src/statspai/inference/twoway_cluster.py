"""
Two-Way Cluster-Robust Standard Errors (Cameron, Gelbach & Miller 2011).

Many panel datasets exhibit correlation along two dimensions (e.g., firm and
year, state and industry). Standard one-way cluster-robust SEs handle only
one dimension. This module implements the multi-way clustering correction:

    V_twoway = V_cluster1 + V_cluster2 - V_intersection

where V_intersection clusters on the (dim1 x dim2) interaction.

Each component uses the standard Liang-Zeger (1986) sandwich estimator with
the finite-sample correction G/(G-1) * (n-1)/(n-k).

If the resulting matrix is not positive semi-definite, an eigenvalue
correction sets negative eigenvalues to zero (Cameron et al. 2011, p. 241).

References
----------
Cameron, A.C., Gelbach, J.B. and Miller, D.L. (2011).
"Robust Inference with Multiway Clustering."
*Journal of Business & Economic Statistics*, 29(2), 238-249. [@cameron2011robust]

Liang, K.-Y. and Zeger, S.L. (1986).
"Longitudinal data analysis using generalized linear models."
*Biometrika*, 73(1), 13-22. [@liang1986longitudinal]
"""

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import EconometricResults
from ..exceptions import MethodIncompatibility


def _cluster_robust_variance(
    X: np.ndarray, residuals: np.ndarray, clusters: np.ndarray
) -> np.ndarray:
    """
    One-way cluster-robust variance matrix (Liang-Zeger 1986).

    V = (G/(G-1)) * ((n-1)/(n-k)) * (X'X)^{-1} B (X'X)^{-1},
    B = sum_g (u_g u_g'), u_g = sum_{i in g} X_i * e_i.

    Thin wrapper over the canonical ``core._vcov.cluster_robust_vcov``; the
    Liang-Zeger correction is its ``'liang_zeger'`` factor. Verified
    byte-identical to the prior hand-rolled implementation for non-missing
    cluster labels.
    """
    from ..core._vcov import cluster_robust_vcov

    return cluster_robust_vcov(X, residuals, clusters, correction="liang_zeger")


def _ensure_psd(V: np.ndarray) -> np.ndarray:
    """
    Eigenvalue correction for non-PSD matrices.

    Sets negative eigenvalues to zero and reconstructs the matrix.
    """
    eigvals, eigvecs = np.linalg.eigh(V)
    if np.any(eigvals < 0):
        eigvals = np.maximum(eigvals, 0.0)
        V = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return V


def _require_least_squares(result: EconometricResults) -> None:
    """Refuse a likelihood fit, whose sandwich has another bread.

    A logit / probit / Poisson / GLM result stores its design and response
    residuals too; ``(X'X)^-1 S'S (X'X)^-1`` with ``S = x e`` on them is not
    that model's cluster-robust variance (the bread is ``X' diag(dmu/deta) X``).
    """
    mi = result.model_info or {}
    di = result.data_info or {}
    link = mi.get("link")
    link_obj = di.get("link_obj")
    if link_obj is not None:
        link = getattr(link_obj, "name", link)
    family = mi.get("family")
    nonlinear = (
        isinstance(link, str) and link.lower() not in ("identity", "gaussian")
    ) or (isinstance(family, str) and family.lower() not in ("gaussian", "normal"))
    if nonlinear:
        raise MethodIncompatibility(
            "twoway_cluster computes the two-way cluster covariance of least "
            f"squares; this fit is a likelihood model (family={family!r}, "
            f"link={link!r}), whose sandwich needs the Hessian of its "
            "likelihood as the bread.",
            recovery_hint=(
                "Use it after sp.regress. For a likelihood model pass both "
                "cluster variables to the estimator's own cluster option "
                "where it offers two-way clustering."
            ),
            diagnostics={"family": family, "link": link},
        )


def twoway_cluster(
    result: EconometricResults,
    data: pd.DataFrame,
    cluster1: str,
    cluster2: str,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Compute two-way cluster-robust standard errors.

    Implements Cameron, Gelbach & Miller (2011):

        V_twoway = V_cluster1 + V_cluster2 - V_intersection

    Parameters
    ----------
    result : EconometricResults
        Fitted OLS result. Must have ``data_info`` containing
        ``'X'`` (design matrix), ``'y'`` (response), and ``'residuals'``.
    data : pd.DataFrame
        Original data containing the cluster variables.
    cluster1 : str
        Column name for the first clustering dimension.
    cluster2 : str
        Column name for the second clustering dimension.
    alpha : float, default 0.05
        Significance level for confidence intervals.

    Returns
    -------
    EconometricResults
        New results object with the same point estimates but two-way
        clustered standard errors, t-statistics, p-values, and
        confidence intervals.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> df = pd.DataFrame({
    ...     'x1': rng.normal(size=n),
    ...     'x2': rng.normal(size=n),
    ...     'firm': rng.integers(0, 15, n),
    ...     'year': rng.integers(2000, 2010, n),
    ... })
    >>> df['y'] = (1.0 + 0.5 * df['x1'] - 0.3 * df['x2']
    ...            + rng.normal(size=n))
    >>> result = sp.regress("y ~ x1 + x2", data=df)
    >>> tw = sp.twoway_cluster(result, data=df,
    ...                        cluster1="firm", cluster2="year")
    >>> print(tw.summary())  # doctest: +SKIP
    """
    # The sandwich below is the least-squares one: bread (X'X)^-1, scores
    # x_i e_i. On a likelihood fit (logit, poisson, glm ...) it is not that
    # model's variance.
    _require_least_squares(result)

    # --- Extract estimation objects ---
    X = np.asarray(result.data_info["X"], dtype=float)
    residuals = np.asarray(result.data_info["residuals"], dtype=float).ravel()

    # The rows the fit kept (it drops missing values) and its analytic
    # weights: the scores are those of the weighted design, on those rows.
    keys = data[[cluster1, cluster2]]
    kept = result.data_info.get("sample_index")
    if kept is not None and len(kept) == X.shape[0] and len(kept) != len(data):
        keys = keys.loc[kept]
    if len(keys) != X.shape[0]:
        raise MethodIncompatibility(
            f"twoway_cluster: {len(keys)} cluster keys for {X.shape[0]} fitted "
            "rows.",
            recovery_hint="Pass the DataFrame the model was fitted on.",
        )
    if keys.isna().any().any():
        raise MethodIncompatibility(
            "twoway_cluster: a cluster key is missing on the estimation sample.",
            recovery_hint="Drop those rows before fitting.",
        )
    w = result.data_info.get("analytic_weights")
    if w is not None:
        root = np.sqrt(np.asarray(w, dtype=float).ravel())
        X = X * root[:, None]
        residuals = residuals * root

    c1 = keys[cluster1].values
    c2 = keys[cluster2].values

    # Reuse the N-way core so intersection keys are factorized tuple-wise
    # instead of built from collision-prone string concatenation.
    from .multiway_cluster import multiway_cluster_vcov

    V_twoway = multiway_cluster_vcov(
        X,
        residuals,
        [c1, c2],
        df_adjust=True,
        n_params=X.shape[1],
        psd_correct=True,
    )

    # --- Build new results ---
    se = pd.Series(np.sqrt(np.diag(V_twoway)), index=result.params.index)

    G1 = len(np.unique(c1))
    G2 = len(np.unique(c2))
    df_t = min(G1, G2) - 1  # Conservative DoF

    model_info = dict(result.model_info)
    # sp.regress records its formula and weights only on its provenance,
    # which the new result does not carry: postestimation (margins of an
    # I(x**2) term, weighted averages) needs them.
    from ..postestimation._design import model_formula, model_setting

    for key, value in (
        ("formula", model_formula(result)),
        ("weights", model_setting(result, "weights")),
    ):
        if value is not None and model_info.get(key) is None:
            model_info[key] = value
    model_info["se_type"] = "twoway_cluster"
    model_info["cluster1"] = cluster1
    model_info["cluster2"] = cluster2
    model_info["n_clusters1"] = G1
    model_info["n_clusters2"] = G2

    data_info = dict(result.data_info)
    # The new covariance replaces the fit's under every key it is read
    # from (predict(what='confidence') reads var_cov). Inference is t with
    # min(G1, G2) - 1 degrees of freedom; df_resid stays n - k, the divisor
    # of the error variance in a prediction interval.
    data_info["var_cov"] = V_twoway
    data_info["vcov"] = V_twoway
    data_info["df_inference"] = df_t
    for stale in ("df_by_coefficient", "hotelling_df", "inference"):
        data_info.pop(stale, None)

    model_info["alpha"] = alpha
    new_result = EconometricResults(
        params=result.params.copy(),
        std_errors=se,
        model_info=model_info,
        data_info=data_info,
        diagnostics=dict(result.diagnostics),
    )

    # Recompute CIs at requested alpha
    t_crit = stats.t.ppf(1 - alpha / 2, df_t)
    new_result.conf_int_lower = new_result.params - t_crit * se
    new_result.conf_int_upper = new_result.params + t_crit * se

    return new_result
