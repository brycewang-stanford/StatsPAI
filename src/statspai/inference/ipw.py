"""
Inverse Probability Weighting (IPW) estimator for ATE / ATT / ATC.

Standalone IPW without outcome-model augmentation (for augmented/doubly-robust,
see :func:`statspai.inference.aipw`).

Implements the Horvitz-Thompson estimator with logistic propensity scores
and optional trimming, normalized weights, and bootstrapped standard errors.

References
----------
Horvitz, D.G. and Thompson, D.J. (1952).
"A Generalization of Sampling Without Replacement From a Finite Universe."
*Journal of the American Statistical Association*, 47(260), 663-685. [@horvitz1952generalization]

Hirano, K., Imbens, G.W. and Ridder, G. (2003).
"Efficient Estimation of Average Treatment Effects Using the Estimated
Propensity Score."
*Econometrica*, 71(4), 1161-1189. [@hirano2003efficient]

Crump, R.K., Hotz, V.J., Imbens, G.W. and Mitnik, O.A. (2009).
"Dealing with Limited Overlap in Estimation of Average Treatment Effects."
*Biometrika*, 96(1), 187-199. [@crump2009dealing]
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .._aliases import accepts_aliases
from ..core.results import CausalResult
from ..exceptions import MethodIncompatibility


@accepts_aliases(_strict=True, controls="covariates")
def ipw(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    estimand: str = "ATE",
    trim: float = 0.0,
    normalize: bool = True,
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    seed: Optional[int] = None,
    weights: Optional[str] = None,
    cluster: Optional[str] = None,
    se_method: str = "bootstrap",
) -> CausalResult:
    """
    Inverse Probability Weighting estimator for treatment effects.

    Estimates ATE, ATT, or ATC by weighting observations by the inverse
    of their propensity to receive the treatment they actually received.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    y : str
        Outcome variable.
    treat : str
        Binary treatment indicator (0/1).
    covariates : list of str
        Variables for the propensity score model (logistic regression).
    estimand : str, default 'ATE'
        'ATE' (average treatment effect), 'ATT' (on treated),
        or 'ATC' (on controls).
    trim : float, default 0.0
        Trim propensity scores to [trim, 1 - trim]. Common choices: 0.01, 0.05, 0.1.
        Crump et al. (2009) recommend dropping units with p outside [0.1, 0.9].
    normalize : bool, default True
        If True, use Hajek (normalised) weights. Generally recommended
        for finite-sample stability.
    n_bootstrap : int, default 500
        Number of bootstrap iterations for standard error estimation.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    seed : int, optional
        Random seed for reproducibility.
    weights : str, optional
        Column of sampling (probability) weights, Stata ``[pw=]``. The
        propensity logit is a weighted fit and each unit's IPW weight is
        multiplied by its sampling weight. Strictly positive; scale-free.
    cluster : str, optional
        Column identifying clusters. The bootstrap resamples whole clusters;
        the sandwich sums the influence function within clusters (Stata
        ``vce(cluster c)``).
    se_method : {'bootstrap', 'sandwich'}, default 'bootstrap'
        ``'sandwich'`` is the stacked M-estimation variance of the logit
        score and the normalised IPW means (divisor ``n``) -- the robust
        standard error of Stata ``teffects ipw``, deterministic and
        bootstrap-free. Requires ``normalize=True`` and ``trim=0``.

    Returns
    -------
    CausalResult
        With `.estimate`, `.se`, `.ci`, `.pvalue`, and propensity score
        diagnostics in `.model_info`.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> age = rng.normal(40, 10, n)
    >>> education = rng.normal(13, 3, n)
    >>> experience = rng.normal(15, 7, n)
    >>> p = 1 / (1 + np.exp(-(-2 + 0.03 * (age - 40) + 0.15 * (education - 13))))
    >>> training = (rng.uniform(size=n) < p).astype(int)
    >>> wage = 20 + 2.5 * training + 0.1 * age + 0.5 * education + rng.normal(0, 2, n)
    >>> df = pd.DataFrame({"wage": wage, "training": training, "age": age,
    ...                    "education": education, "experience": experience})
    >>> result = sp.ipw(df, y='wage', treat='training',
    ...                 covariates=['age', 'education', 'experience'],
    ...                 n_bootstrap=100, seed=0)
    >>> result.estimand
    'ATE'

    >>> # ATT with trimming
    >>> result = sp.ipw(df, y='wage', treat='training',
    ...                 covariates=['age', 'education'],
    ...                 estimand='ATT', trim=0.05, n_bootstrap=100, seed=0)
    >>> result.estimand
    'ATT'
    """
    estimand = estimand.upper()
    if estimand not in ("ATE", "ATT", "ATC"):
        raise ValueError(f"estimand must be 'ATE', 'ATT', or 'ATC', got '{estimand}'")

    if se_method not in ("bootstrap", "sandwich"):
        raise MethodIncompatibility(
            f"se_method must be 'bootstrap' or 'sandwich', got {se_method!r}"
        )
    if se_method == "sandwich" and (not normalize or trim > 0):
        raise MethodIncompatibility(
            "se_method='sandwich' is the M-estimation variance of the "
            "normalised (Hajek) IPW estimator without trimming; it requires "
            "normalize=True and trim=0.",
            recovery_hint="Use se_method='bootstrap' for trimmed or "
            "Horvitz-Thompson weights.",
        )
    rng = np.random.RandomState(seed)

    # --- Prepare data ---
    extra = [c for c in (weights, cluster) if c is not None]
    missing_cols = [c for c in [y, treat] + list(covariates) + extra if c not in data]
    if missing_cols:
        raise MethodIncompatibility(
            f"ipw: columns not found in data: {missing_cols}",
            diagnostics={"missing": missing_cols},
        )
    df = data[list(dict.fromkeys([y, treat] + list(covariates) + extra))].dropna()
    Y = df[y].values.astype(np.float64)
    T = df[treat].values.astype(np.float64)
    X = df[covariates].values.astype(np.float64)
    n = len(Y)
    # Sampling weights normalised to mean one; None keeps the unweighted
    # path byte-identical to earlier releases.
    sw: Optional[np.ndarray] = None
    if weights is not None:
        wv = df[weights].to_numpy(dtype=float)
        if not np.all(np.isfinite(wv)) or np.any(wv <= 0):
            raise MethodIncompatibility(
                "ipw: weights must be finite and strictly positive.",
                diagnostics={"weights": weights},
            )
        sw = wv * (n / wv.sum())
    groups: Optional[np.ndarray] = None
    if cluster is not None:
        groups = pd.factorize(df[cluster])[0]
        if groups.max() + 1 < 2:
            raise MethodIncompatibility(
                "ipw: cluster= needs at least two clusters.",
                diagnostics={"cluster": cluster},
            )

    if not set(np.unique(T)).issubset({0, 1}):
        raise ValueError(f"Treatment variable '{treat}' must be binary (0/1)")
    if T.sum() == 0 or T.sum() == n:
        raise ValueError(
            f"Treatment variable '{treat}' must contain both treated and "
            "control observations"
        )

    # --- Estimate propensity scores ---
    pscore = _estimate_propensity(X, T, sw)
    pscore_raw = np.asarray(pscore, dtype=float).copy()  # pre-trim, for overlap

    # --- Trim ---
    if trim > 0:
        pscore = np.clip(pscore, trim, 1 - trim)

    # --- Compute weights ---
    weights_1, weights_0 = _compute_weights(T, pscore, estimand, normalize, sw)

    # --- Point estimate ---
    estimate = float(np.sum(weights_1 * Y) - np.sum(weights_0 * Y))

    if se_method == "sandwich":
        se = _ipw_sandwich_se(X, T, Y, pscore, estimand, sw, groups)
        boot_estimates = None
    else:
        # --- Bootstrap SE (whole clusters when cluster= is given) ---
        boot_estimates = np.empty(n_bootstrap)
        if groups is not None:
            members = [np.flatnonzero(groups == g) for g in range(groups.max() + 1)]
        for b in range(n_bootstrap):
            if groups is None:
                idx = rng.choice(n, size=n, replace=True)
            else:
                pick = rng.choice(len(members), size=len(members), replace=True)
                idx = np.concatenate([members[g] for g in pick])
            Y_b, T_b, X_b = Y[idx], T[idx], X[idx]
            sw_b = None if sw is None else sw[idx]
            ps_b = _estimate_propensity(X_b, T_b, sw_b)
            if trim > 0:
                ps_b = np.clip(ps_b, trim, 1 - trim)
            w1, w0 = _compute_weights(T_b, ps_b, estimand, normalize, sw_b)
            boot_estimates[b] = np.sum(w1 * Y_b) - np.sum(w0 * Y_b)

        se = float(np.std(boot_estimates, ddof=1))
    t_crit = sp_stats.norm.ppf(1 - alpha / 2)
    ci = (estimate - t_crit * se, estimate + t_crit * se)
    pvalue = float(2 * sp_stats.norm.sf(abs(estimate / se))) if se > 0 else 1.0

    # --- Diagnostics ---
    n_treated = int(T.sum())
    n_control = int(n - n_treated)

    model_info = {
        "model_type": "IPW",
        "estimand": estimand,
        "n_treated": n_treated,
        "n_control": n_control,
        "pscore_mean_treated": float(pscore[T == 1].mean()),
        "pscore_mean_control": float(pscore[T == 0].mean()),
        "pscore_min": float(pscore.min()),
        "pscore_max": float(pscore.max()),
        "trim": trim,
        # Raw propensity distribution so result.violations() can assess overlap
        # (IPW is the most overlap-sensitive estimator). Read via the shared
        # _propensity_extreme_share helper; the trimming bound defaults to 0.01.
        "_pscore": pscore_raw,
        "trimming_threshold": trim,
        "normalized": normalize,
        "n_bootstrap": n_bootstrap if se_method == "bootstrap" else None,
        "se_method": se_method,
        "weights": weights,
        "cluster": cluster,
        "n_clusters": None if groups is None else int(groups.max() + 1),
    }

    _result = CausalResult(
        method=f"IPW ({estimand})",
        estimand=estimand,
        estimate=estimate,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        model_info=model_info,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.ipw",
            params={
                "y": y,
                "treat": treat,
                "covariates": list(covariates),
                "estimand": estimand,
                "trim": trim,
                "normalize": normalize,
                "n_bootstrap": n_bootstrap,
                "alpha": alpha,
                "seed": seed,
                "weights": weights,
                "cluster": cluster,
                "se_method": se_method,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


# ====================================================================== #
#  Internal helpers
# ====================================================================== #


def _estimate_propensity(
    X: np.ndarray, T: np.ndarray, sw: Optional[np.ndarray] = None
) -> np.ndarray:
    """Logistic regression propensity score (weighted MLE if ``sw``)."""
    if sw is not None:
        import statsmodels.api as sm

        X_const = sm.add_constant(X, has_constant="add")
        # freq_weights gives the pweighted likelihood's point estimates.
        res = sm.GLM(T, X_const, family=sm.families.Binomial(), freq_weights=sw).fit(
            tol=1e-12, maxiter=300
        )
        return np.clip(np.asarray(res.predict(X_const), dtype=float), 1e-8, 1 - 1e-8)
    try:
        import statsmodels.api as sm

        X_const = sm.add_constant(X, has_constant="add")
        model = sm.GLM(T, X_const, family=sm.families.Binomial())
        res = model.fit(maxiter=300, disp=0)
        if getattr(res, "converged", True) is False:
            raise RuntimeError("statsmodels GLM did not converge")
        ps = np.asarray(res.predict(X_const), dtype=float)
    except Exception:
        from sklearn.linear_model import LogisticRegression

        try:
            model = LogisticRegression(
                max_iter=5000,
                solver="newton-cg",
                penalty=None,
                tol=1e-10,
            )
        except TypeError:  # pragma: no cover - old scikit-learn compatibility
            model = LogisticRegression(
                max_iter=5000,
                solver="newton-cg",
                penalty="none",
                tol=1e-10,
            )
        model.fit(X, T)
        ps = model.predict_proba(X)[:, 1]
    # Safety clip to avoid division by zero
    return np.clip(ps, 1e-8, 1 - 1e-8)


def _ipw_sandwich_se(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    e: np.ndarray,
    estimand: str,
    sw: Optional[np.ndarray],
    groups: Optional[np.ndarray],
) -> float:
    """M-estimation SE of the normalised IPW contrast (Stata ``teffects ipw``).

    Stacks the (weighted) logit score ``w (T - e) x`` with the two Hajek
    means ``w a_k (Y - mu_k) = 0``, where ``a_1 = T/e, a_0 = (1-T)/(1-e)``
    (ATE), ``T, (1-T) e/(1-e)`` (ATT) or ``T (1-e)/e, 1-T`` (ATC). Row ``i``
    of the influence function of ``mu_k`` is
    ``[w a_k (Y - mu_k) + G_k' IF_gamma] / mean(w a_k)`` with
    ``G_k = mean(w (Y - mu_k) d a_k / d gamma)`` and
    ``IF_gamma = H^{-1} w (T - e) x``, ``H = mean(w e (1-e) x x')``.
    Divisor ``n``; with clusters the rows are summed within clusters first.
    """
    n = len(Y)
    w = np.ones(n) if sw is None else sw
    Xc = np.column_stack([np.ones(n), X])
    odds = e / (1 - e)
    if estimand == "ATE":
        a1, a0 = T / e, (1 - T) / (1 - e)
        da1, da0 = -T * (1 - e) / e, (1 - T) * odds
    elif estimand == "ATT":
        a1, a0 = T, (1 - T) * odds
        da1, da0 = np.zeros(n), (1 - T) * odds
    else:  # ATC
        a1, a0 = T * (1 - e) / e, 1 - T
        da1, da0 = -T * (1 - e) / e, np.zeros(n)
    H = (Xc * (w * e * (1 - e))[:, None]).T @ Xc / n
    if_gamma = np.linalg.solve(H, (Xc * (w * (T - e))[:, None]).T).T

    def _if(a: np.ndarray, da: np.ndarray) -> np.ndarray:
        mu = np.sum(w * a * Y) / np.sum(w * a)
        resid = Y - mu
        G = (Xc * (w * resid * da)[:, None]).mean(axis=0)
        rows: np.ndarray = (w * a * resid + if_gamma @ G) / np.mean(w * a)
        return rows

    u = _if(a1, da1) - _if(a0, da0)
    if groups is not None:
        u = np.bincount(groups, weights=u)
    return float(np.sqrt(np.sum(u**2)) / n)


def _compute_weights(
    T: np.ndarray,
    pscore: np.ndarray,
    estimand: str,
    normalize: bool,
    sw: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute IPW weights for treated (w1) and control (w0) groups.

    Returns (weights_1, weights_0) such that:
        estimate = sum(w1 * Y) - sum(w0 * Y)

    Sampling weights ``sw`` (mean one) multiply each unit's IPW weight.
    """
    n = len(T)
    if sw is not None:
        w1, w0 = _compute_weights(T, pscore, estimand, normalize=False)
        w1, w0 = w1 * n * sw, w0 * n * sw
        if normalize:
            return w1 / w1.sum(), w0 / w0.sum()
        return w1 / n, w0 / n

    if estimand == "ATE":
        # Horvitz-Thompson: w1 = T/p, w0 = (1-T)/(1-p)
        w1 = T / pscore
        w0 = (1 - T) / (1 - pscore)
    elif estimand == "ATT":
        # ATT: treated get weight 1, controls get weight p/(1-p)
        w1 = T.copy()
        w0 = (1 - T) * pscore / (1 - pscore)
    elif estimand == "ATC":
        # ATC: controls get weight 1, treated get weight (1-p)/p
        w1 = T * (1 - pscore) / pscore
        w0 = (1 - T).copy()

    if normalize:
        s1 = w1.sum()
        s0 = w0.sum()
        if s1 > 0:
            w1 = w1 / s1
        if s0 > 0:
            w0 = w0 / s0
    else:
        w1 = w1 / n
        w0 = w0 / n

    return w1, w0
