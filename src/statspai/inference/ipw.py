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

import warnings
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .._aliases import accepts_aliases
from ..core._covariates import expands_categorical_covariates as _expands_categorical
from ..core._validate import treatment_as_float as _treatment_as_float
from ..core.results import CausalResult
from ..exceptions import AssumptionWarning, MethodIncompatibility


@accepts_aliases(_strict=True, controls="covariates")
@_expands_categorical("covariates")
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
    ps_model: str = "logit",
    propensity: Optional[Union[float, str]] = None,
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
    estimand : {'ATE', 'ATT', 'ATC', 'ATO', 'ATM'}, default 'ATE'
        The population the weights target. With propensity score ``e``
        and tilting function ``h(e)``, treated units get ``h / e`` and
        controls ``h / (1 - e)``:

        - ``'ATE'``: ``h = 1``, the whole sample.
        - ``'ATT'``: ``h = e``, the treated.
        - ``'ATC'`` (also ``'ATU'``): ``h = 1 - e``, the untreated.
        - ``'ATO'``: ``h = e (1 - e)``, the overlap population of Li,
          Morgan and Zaslavsky (2018) [@li2018balancing]. Weights are bounded by one and the
          covariates of the logit are balanced exactly.
        - ``'ATM'``: ``h = min(e, 1 - e)``, the matching weights of Li and
          Greene (2013) [@li2013weighting], the population a 1:1 caliper match would keep.

        ``'ATO'`` and ``'ATM'`` are defined through normalised weights and
        require ``normalize=True``.
    trim : float, default 0.0
        Trim propensity scores to [trim, 1 - trim]. Common choices: 0.01, 0.05, 0.1.
        Crump et al. (2009) recommend dropping units with p outside [0.1, 0.9].
    normalize : bool, default True
        If True, use Hajek (normalised) weights. Generally recommended
        for finite-sample stability, and the estimate does not change when
        a constant is added to the outcome. ``False`` is Horvitz-Thompson:
        the weighted sums are divided by the size of the target population
        (``n`` for the ATE, the number of treated for the ATT, of controls
        for the ATC).
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
    ps_model : {'logit', 'probit'}, default 'logit'
        The binary model for the propensity score. ``'probit'`` is the
        treatment model of Stata ``teffects ipw (y) (d x, probit)``. The
        sandwich variance uses the score and the observed Hessian of the
        chosen model.
    propensity : float or str, optional
        Known probability of treatment: one number, or a column of
        per-unit design probabilities (a randomised trial). No propensity
        model is fitted, the bootstrap resamples the design probabilities
        with their rows, and the sandwich drops the first-stage term. An
        IPW estimator that *estimates* the propensity is the more precise
        of the two even when the truth is known, because the fitted
        propensity absorbs chance imbalance in the covariates; pass the
        known one when the design-based estimator is what is wanted, or
        use ``sp.aipw(propensity=)``, whose variance does not depend on
        the choice.

    Returns
    -------
    CausalResult
        With `.estimate`, `.se`, `.ci`, `.pvalue`, and propensity score
        diagnostics in `.model_info`, among them ``ess_treated`` and
        ``ess_control``: Kish's effective sample size of each arm's
        weights. When either falls below a fifth of the arm's size an
        :class:`~statspai.exceptions.AssumptionWarning` is raised: the
        estimate then rests on a few heavily weighted units.

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
    if estimand == "ATU":
        estimand = "ATC"
    if estimand not in ("ATE", "ATT", "ATC", "ATO", "ATM"):
        raise ValueError(
            "estimand must be 'ATE', 'ATT', 'ATC' (or 'ATU'), 'ATO' or 'ATM', "
            f"got '{estimand}'"
        )
    if estimand in ("ATO", "ATM") and not normalize:
        raise MethodIncompatibility(
            f"estimand={estimand!r} has no Horvitz-Thompson form: the overlap "
            "and matching-weight populations are defined by the weights "
            "themselves.",
            recovery_hint="Use normalize=True.",
        )

    if se_method not in ("bootstrap", "sandwich"):
        raise MethodIncompatibility(
            f"se_method must be 'bootstrap' or 'sandwich', got {se_method!r}"
        )
    if ps_model not in ("logit", "probit"):
        raise MethodIncompatibility(
            f"ps_model must be 'logit' or 'probit', got {ps_model!r}"
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
    if isinstance(propensity, str):
        extra.append(propensity)
    missing_cols = [c for c in [y, treat] + list(covariates) + extra if c not in data]
    if missing_cols:
        raise MethodIncompatibility(
            f"ipw: columns not found in data: {missing_cols}",
            diagnostics={"missing": missing_cols},
        )
    df = data[list(dict.fromkeys([y, treat] + list(covariates) + extra))].dropna()
    Y = df[y].values.astype(np.float64)
    T = _treatment_as_float(df[treat], function="ipw")
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

    # --- Estimate propensity scores (or take the design's) ---
    ps_known = propensity is not None
    if ps_known:
        if isinstance(propensity, str):
            pscore = df[propensity].to_numpy(dtype=float)
        else:
            pscore = np.full(n, float(propensity))
        if not np.all(np.isfinite(pscore)) or np.any((pscore <= 0) | (pscore >= 1)):
            raise MethodIncompatibility(
                "ipw: a known propensity must lie strictly between 0 and 1.",
                diagnostics={
                    "min": float(np.nanmin(pscore)),
                    "max": float(np.nanmax(pscore)),
                },
            )
    else:
        pscore = _estimate_propensity(X, T, sw, ps_model)
    pscore_raw = np.asarray(pscore, dtype=float).copy()  # pre-trim, for overlap

    # --- Trim ---
    if trim > 0:
        pscore = np.clip(pscore, trim, 1 - trim)

    # --- Compute weights ---
    weights_1, weights_0 = _compute_weights(T, pscore, estimand, normalize, sw)

    # --- Point estimate ---
    estimate = float(np.sum(weights_1 * Y) - np.sum(weights_0 * Y))

    # --- How many observations the weights leave each arm with ---
    ess_treated = _kish_ess(weights_1[T == 1])
    ess_control = _kish_ess(weights_0[T == 0])
    n1_arm, n0_arm = int(T.sum()), int(n - T.sum())
    ess_ratio_min = min(ess_treated / n1_arm, ess_control / n0_arm)
    if ess_ratio_min < _ESS_RATIO_WARN:
        thin = "control" if ess_control / n0_arm <= ess_treated / n1_arm else "treated"
        thin_ess, thin_n = (
            (ess_control, n0_arm) if thin == "control" else (ess_treated, n1_arm)
        )
        warnings.warn(
            AssumptionWarning(
                f"ipw: the weights leave the {thin} arm an effective sample "
                f"of {thin_ess:.0f} out of {thin_n} ({thin_ess / thin_n:.0%}); "
                f"propensity scores run from {float(pscore.min()):.3f} to "
                f"{float(pscore.max()):.3f}. The estimate rests on a few "
                "heavily weighted units and can move far from estimators "
                "that model the outcome.",
                recovery_hint=(
                    "Compare with sp.aipw; restrict to the overlap region "
                    "with trim= or sp.trimming; or target the overlap "
                    "population with estimand='ATO'."
                ),
                diagnostics={
                    "ess_treated": ess_treated,
                    "ess_control": ess_control,
                    "n_treated": n1_arm,
                    "n_control": n0_arm,
                    "pscore_min": float(pscore.min()),
                    "pscore_max": float(pscore.max()),
                },
                alternative_functions=["sp.aipw", "sp.trimming", "sp.overlap_weights"],
            ),
            stacklevel=3,
        )

    if se_method == "sandwich":
        se = _ipw_sandwich_se(
            X, T, Y, pscore, estimand, sw, groups, ps_model, ps_known=ps_known
        )
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
            if ps_known:
                ps_b = pscore[idx]
            else:
                ps_b = _estimate_propensity(X_b, T_b, sw_b, ps_model)
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
        "ess_treated": ess_treated,
        "ess_control": ess_control,
        "ess_ratio_min": float(ess_ratio_min),
        "trim": trim,
        # Raw propensity distribution so result.violations() can assess overlap
        # (IPW is the most overlap-sensitive estimator). Read via the shared
        # _propensity_extreme_share helper; the trimming bound defaults to 0.01.
        "_pscore": pscore_raw,
        "trimming_threshold": trim,
        "normalized": normalize,
        "n_bootstrap": n_bootstrap if se_method == "bootstrap" else None,
        "se_method": se_method,
        "ps_model": None if ps_known else ps_model,
        "propensity": "known" if ps_known else "estimated",
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
                "ps_model": ps_model,
                "propensity": propensity,
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

#: An arm whose Kish effective sample size is below this share of its
#: size draws a warning. Under a constant propensity the share is one; at
#: 0.2 four fifths of the arm's information is gone to weight dispersion.
_ESS_RATIO_WARN = 0.2


def _kish_ess(w: np.ndarray) -> float:
    """Kish's effective sample size of a weight vector, ``(sum w)^2 / sum w^2``."""
    total = float(np.sum(w))
    squares = float(np.sum(w**2))
    return total * total / squares if squares > 0 else 0.0


def _binomial_family(ps_model: str) -> Any:
    import statsmodels.api as sm

    if ps_model == "probit":
        return sm.families.Binomial(link=sm.families.links.Probit())
    return sm.families.Binomial()


def _estimate_propensity(
    X: np.ndarray,
    T: np.ndarray,
    sw: Optional[np.ndarray] = None,
    ps_model: str = "logit",
) -> np.ndarray:
    """Logit or probit propensity score (weighted MLE if ``sw``)."""
    if sw is not None:
        import statsmodels.api as sm

        X_const = sm.add_constant(X, has_constant="add")
        # freq_weights gives the pweighted likelihood's point estimates.
        res = sm.GLM(
            T, X_const, family=_binomial_family(ps_model), freq_weights=sw
        ).fit(tol=1e-12, maxiter=300)
        return np.clip(np.asarray(res.predict(X_const), dtype=float), 1e-8, 1 - 1e-8)
    if ps_model == "probit":
        import statsmodels.api as sm

        X_const = sm.add_constant(X, has_constant="add")
        res = sm.GLM(T, X_const, family=_binomial_family(ps_model)).fit(
            tol=1e-12, maxiter=300
        )
        if getattr(res, "converged", True) is False:
            raise MethodIncompatibility(
                "ipw: the probit propensity model did not converge.",
                recovery_hint="Check for separation or badly scaled "
                "covariates, or use ps_model='logit'.",
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
    except Exception as exc:
        from ..core._fallback import warn_fallback

        warn_fallback(
            "IPW propensity logit (statsmodels GLM)",
            exc,
            "refitting with scikit-learn's unpenalised logistic "
            "regression; check for separation or collinear covariates",
        )
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
    ps_model: str = "logit",
    ps_known: bool = False,
) -> float:
    """M-estimation SE of the normalised IPW contrast (Stata ``teffects ipw``).

    Stacks the (weighted) logit score ``w (T - e) x`` with the two Hajek
    means ``w a_k (Y - mu_k) = 0``, where ``a_1 = T/e, a_0 = (1-T)/(1-e)``
    (ATE), ``T, (1-T) e/(1-e)`` (ATT), ``T (1-e)/e, 1-T`` (ATC),
    ``T (1-e), (1-T) e`` (ATO) or ``T min(1, (1-e)/e), (1-T) min(1, e/(1-e))``
    (ATM; its kink at ``e = 1/2`` has probability zero). Row ``i``
    of the influence function of ``mu_k`` is
    ``[w a_k (Y - mu_k) + G_k' IF_gamma] / mean(w a_k)`` with
    ``G_k = mean(w (Y - mu_k) d a_k / d gamma)`` and
    ``IF_gamma = H^{-1} w (T - e) x``, ``H = mean(w e (1-e) x x')``.
    Divisor ``n``; with clusters the rows are summed within clusters first.

    For a probit the score is ``w lam x`` with ``lam = (T - e) f / (e (1-e))``
    and ``f`` the normal density at the index, ``H`` is the observed Hessian
    ``mean(w lam (lam + index) x x')`` and ``e (1-e)`` in ``d a_k / d gamma``
    becomes ``f``.

    With ``ps_known`` the propensity is a design quantity, there is no
    first-stage score to stack, and the ``G_k' IF_gamma`` term is dropped.
    """
    n = len(Y)
    w = np.ones(n) if sw is None else sw
    Xc = np.column_stack([np.ones(n), X])
    odds = e / (1 - e)
    if ps_model == "probit":
        index = sp_stats.norm.ppf(e)
        dens = sp_stats.norm.pdf(index)  # d e / d index
        score = (T - e) * dens / (e * (1 - e))
        curv = score * (score + index)  # minus the observed second derivative
    else:
        dens = e * (1 - e)
        score = T - e
        curv = dens
    # d(1/e) and d(e/(1-e)) with respect to the index; the logit forms are
    # written out so that path keeps its earlier floating-point result
    if ps_model == "probit":
        d_inv, d_odds = -T * dens / e**2, (1 - T) * dens / (1 - e) ** 2
    else:
        d_inv, d_odds = -T * (1 - e) / e, (1 - T) * odds
    if estimand == "ATE":
        a1, a0 = T / e, (1 - T) / (1 - e)
        da1, da0 = d_inv, d_odds
    elif estimand == "ATT":
        a1, a0 = T, (1 - T) * odds
        da1, da0 = np.zeros(n), d_odds
    elif estimand == "ATC":
        a1, a0 = T * (1 - e) / e, 1 - T
        da1, da0 = d_inv, np.zeros(n)
    elif estimand == "ATO":
        a1, a0 = T * (1 - e), (1 - T) * e
        da1, da0 = -T * dens, (1 - T) * dens
    else:  # ATM
        low = e < 0.5
        a1 = T * np.where(low, 1.0, (1 - e) / e)
        a0 = (1 - T) * np.where(low, odds, 1.0)
        da1 = np.where(low, 0.0, d_inv)
        da0 = np.where(low, d_odds, 0.0)
    H = (Xc * (w * curv)[:, None]).T @ Xc / n
    if_gamma = np.linalg.solve(H, (Xc * (w * score)[:, None]).T).T

    def _if(a: np.ndarray, da: np.ndarray) -> np.ndarray:
        mu = np.sum(w * a * Y) / np.sum(w * a)
        resid = Y - mu
        if ps_known:
            return np.asarray(w * a * resid / np.mean(w * a))
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
    if sw is not None:
        w1, w0 = _raw_weights(T, pscore, estimand)
        w1, w0 = w1 * sw, w0 * sw
        if normalize:
            return w1 / w1.sum(), w0 / w0.sum()
        denom = _target_size(T, estimand, sw)
        return w1 / denom, w0 / denom

    w1, w0 = _raw_weights(T, pscore, estimand)
    if normalize:
        s1 = w1.sum()
        s0 = w0.sum()
        if s1 > 0:
            w1 = w1 / s1
        if s0 > 0:
            w0 = w0 / s0
    else:
        # Horvitz-Thompson: divide by the size of the target population --
        # n for the ATE, the number of treated for the ATT and of controls
        # for the ATC. Dividing the ATT / ATC weights by n (as was done
        # through 1.38.0) returned P(T=1) x ATT and P(T=0) x ATC.
        denom = _target_size(T, estimand, None)
        w1 = w1 / denom
        w0 = w0 / denom

    return w1, w0


def _target_size(T: np.ndarray, estimand: str, sw: Optional[np.ndarray]) -> float:
    """(Weighted) number of units in the population the estimand averages over."""
    one = np.ones(len(T)) if sw is None else sw
    if estimand == "ATT":
        return float(np.sum(one * T))
    if estimand == "ATC":
        return float(np.sum(one * (1 - T)))
    return float(np.sum(one))


def _raw_weights(
    T: np.ndarray, pscore: np.ndarray, estimand: str
) -> tuple[np.ndarray, np.ndarray]:
    """Unscaled IPW weights of the treated and control terms."""

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
    elif estimand == "ATO":
        # overlap weights: the probability of the other arm
        w1 = T * (1 - pscore)
        w0 = (1 - T) * pscore
    elif estimand == "ATM":
        # matching weights: min(p, 1-p) over the probability of the own arm
        tilt = np.minimum(pscore, 1 - pscore)
        w1 = T * tilt / pscore
        w0 = (1 - T) * tilt / (1 - pscore)
    return w1, w0
