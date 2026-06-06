"""
Flexible covariate adjustment for RD via machine learning (Noack, Olma & Rothe 2025).

Implements ``rd_flex``: a wrapper around :func:`rdrobust` that residualises
the outcome with respect to a (potentially high-dimensional) covariate
vector ``W`` using a user-supplied sklearn-compatible learner with K-fold
cross-fitting.  Following Noack, Olma & Rothe (2025, arXiv:2107.07942 v5),
the adjusted outcome

    Ỹ_i = Y_i − η̂(W_i)

is fed into the canonical local-polynomial RD estimator.  Provided ``η`` is
continuous in ``W`` near the cutoff (a free-of-cutoff continuity condition),
the RD estimand is invariant; the optimal ``η*(W) = E[Y | X = c, W]``
minimises the asymptotic variance.  Cross-fitting prevents overfitting bias.

This complements :func:`rd_lasso`, :func:`rd_forest`, and :func:`rd_boost`,
which model the *conditional* treatment effect τ(x); ``rd_flex`` instead
*reduces variance of the average treatment effect at the cutoff* by using
covariates as predictors of the outcome.

References
----------
Noack, C., Olma, T., and Rothe, C. (2025).
"Flexible Covariate Adjustments in Regression Discontinuity Designs."
arXiv preprint arXiv:2107.07942 v5. [@noack2025flexible]
"""

from __future__ import annotations

from typing import Optional, List, Any
import warnings

import numpy as np
import pandas as pd

from ..core.results import CausalResult
from .rdrobust import rdrobust


def rd_flex(
    data: pd.DataFrame,
    y: str,
    x: str,
    c: float = 0.0,
    W: Optional[List[str]] = None,
    *,
    learner: str = "boost",
    sklearn_estimator: Optional[Any] = None,
    n_folds: int = 5,
    fuzzy: Optional[str] = None,
    bwselect: str = "mserd",
    kernel: str = "triangular",
    p: int = 1,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> CausalResult:
    """
    RD with flexible covariate adjustment via cross-fit ML residualisation.

    Estimates τ at the cutoff after residualising ``y`` with respect to
    ``W`` using ``learner`` (or a user-supplied sklearn estimator) under
    K-fold cross-fitting.  Empirically yields shorter confidence
    intervals than the unadjusted ``rdrobust`` whenever ``W`` predicts
    the outcome well.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome column.
    x : str
        Running variable column.
    c : float, default 0.0
        Cutoff value.
    W : list of str, optional
        Covariate column names used for the flexible adjustment. If
        ``None`` or empty, falls back to plain :func:`rdrobust`.
    learner : str, default ``'boost'``
        Built-in learner choice when ``sklearn_estimator`` is None:
        ``'boost'`` (gradient boosting), ``'forest'`` (random forest),
        ``'ridge'`` (ridge with alpha CV), or ``'lasso'`` (Lasso CV).
        High-dimensional covariates work best with ``'lasso'`` or
        ``'boost'``.
    sklearn_estimator : estimator, optional
        Any object exposing ``fit(X, y)`` and ``predict(X)``.  If
        provided, supersedes ``learner``.  ``random_state`` is *not*
        injected into a user-provided estimator.
    n_folds : int, default 5
        Number of cross-fitting folds.  Use ``1`` to disable
        cross-fitting (fits ``η`` on the full sample and predicts in-sample
        — only recommended for low-dimensional ``W``).
    fuzzy : str, optional
        Fuzzy treatment column.  When provided the same flexible
        residualisation is applied to both ``y`` and the treatment
        indicator (numerator and denominator of the Wald ratio).
    bwselect, kernel, p, cluster, alpha
        Forwarded to :func:`rdrobust`.
    random_state : int, optional
        Seed for the cross-fit splitter and for built-in learners.

    Returns
    -------
    CausalResult
        Standard RD result with ``model_info['flex']`` containing
        ``learner``, ``n_folds``, ``r2_y`` (cross-fitted out-of-sample
        R² for the outcome), ``var_reduction`` (estimated variance
        reduction relative to plain rdrobust), and ``W`` covariate
        names.

    Notes
    -----
    The Noack-Olma-Rothe estimator is a *boundary-free* covariate
    adjustment: η̂ need not be consistent for the cutoff conditional
    expectation in order for the resulting RD estimator to be
    consistent for τ_RD.  It is, however, asymptotically efficient
    among the class of valid covariate-adjusted estimators when
    ``η̂(W) → E[Y | X = c, W]`` in mean square.

    Examples
    --------
    >>> import statspai as sp
    >>> from statspai.datasets import load_lee2008
    >>> df = load_lee2008()
    >>> r = sp.rd_flex(df, y='vote', x='margin', c=0.0,
    ...                W=['lagvote', 'demvoteshare'],
    ...                learner='boost', n_folds=5)
    >>> r.summary()
    """
    if W is None or len(W) == 0:
        warnings.warn(
            "rd_flex called with no covariates; falling back to rdrobust.",
            UserWarning,
        )
        return rdrobust(
            data=data, y=y, x=x, c=c, fuzzy=fuzzy, p=p, kernel=kernel,
            bwselect=bwselect, cluster=cluster, alpha=alpha,
        )

    # --- Validate columns --------------------------------------------
    needed = [y, x] + list(W)
    if fuzzy is not None:
        needed.append(fuzzy)
    if cluster is not None:
        needed.append(cluster)
    missing = [c_ for c_ in needed if c_ not in data.columns]
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")  # pragma: no cover

    df = data.dropna(subset=needed).copy()
    Y = df[y].to_numpy(dtype=float)
    X = df[x].to_numpy(dtype=float)
    Wmat = df[W].to_numpy(dtype=float)
    D = df[fuzzy].to_numpy(dtype=float) if fuzzy is not None else None

    # --- Cross-fit residualisation -----------------------------------
    Y_resid, r2_y = _crossfit_residualise(
        Wmat, Y, learner, sklearn_estimator, n_folds, random_state,
    )
    df = df.assign(_yflex_=Y_resid)

    if fuzzy is not None:
        D_resid, _ = _crossfit_residualise(
            Wmat, D, learner, sklearn_estimator, n_folds, random_state,
        )
        df = df.assign(_dflex_=D_resid)
        # Apply rdrobust on residualised Y, with residualised D as fuzzy
        # treatment.  Wald ratio remains consistent because both
        # numerator and denominator are residualised by free-of-cutoff
        # adjustments.
        r_flex = rdrobust(
            data=df, y='_yflex_', x=x, c=c, fuzzy='_dflex_',
            p=p, kernel=kernel, bwselect=bwselect, cluster=cluster,
            alpha=alpha,
        )
    else:
        r_flex = rdrobust(
            data=df, y='_yflex_', x=x, c=c, p=p, kernel=kernel,
            bwselect=bwselect, cluster=cluster, alpha=alpha,
        )

    # Variance reduction relative to plain rdrobust
    r_plain = rdrobust(
        data=df, y=y, x=x, c=c, fuzzy=fuzzy,
        p=p, kernel=kernel, bwselect=bwselect, cluster=cluster,
        alpha=alpha,
    )
    se_plain = float(r_plain.se) if r_plain.se else np.nan
    se_flex = float(r_flex.se) if r_flex.se else np.nan
    var_reduction = (
        1.0 - (se_flex ** 2) / (se_plain ** 2)
        if (np.isfinite(se_plain) and np.isfinite(se_flex) and se_plain > 0)
        else np.nan
    )

    method_label = (
        f"RD with flexible covariate adjustment "
        f"(Noack-Olma-Rothe 2025; learner={learner}, n_folds={n_folds})"
    )

    flex_info = {
        "learner": learner,
        "n_folds": int(n_folds),
        "n_covariates": int(Wmat.shape[1]),
        "covariates": list(W),
        "r2_y": float(r2_y),
        "se_plain": se_plain,
        "se_flex": se_flex,
        "var_reduction": var_reduction,
        "summary_str": (
            "\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "  Flexible Covariate-Adjusted RD (Noack-Olma-Rothe 2025)\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"  Learner:                  {learner} ({n_folds}-fold CV)\n"
            f"  Covariates (W):           {len(W)} variables\n"
            f"  Cross-fit out-of-sample R²: {r2_y:.4f}\n"
            f"\n"
            f"  τ̂ (flex):                 {float(r_flex.estimate):.4f}\n"
            f"  SE (flex):                {se_flex:.4f}\n"
            f"  τ̂ (plain rdrobust):       {float(r_plain.estimate):.4f}\n"
            f"  SE (plain):               {se_plain:.4f}\n"
            f"  Variance reduction:       "
            f"{(var_reduction * 100 if np.isfinite(var_reduction) else float('nan')):.1f}%\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        ),
    }

    # Merge info from r_flex
    info = dict(r_flex.model_info or {})
    info["flex"] = flex_info
    info.setdefault("kernel", kernel)
    info.setdefault("cutoff", c)

    out = CausalResult(
        method=method_label,
        estimand="LATE" if fuzzy is not None else "ATE at cutoff",
        estimate=float(r_flex.estimate),
        se=se_flex,
        pvalue=float(r_flex.pvalue) if r_flex.pvalue is not None else None,
        ci=tuple(r_flex.ci) if r_flex.ci is not None else None,
        alpha=alpha,
        n_obs=int(r_flex.n_obs) if r_flex.n_obs is not None else None,
        model_info=info,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov
        _attach_prov(
            out,
            function="sp.rd.rd_flex",
            params={
                "y": y, "x": x, "c": c, "W": list(W), "learner": learner,
                "n_folds": n_folds, "fuzzy": fuzzy, "bwselect": bwselect,
                "kernel": kernel, "p": p, "cluster": cluster, "alpha": alpha,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass  # pragma: no cover
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_learner(name: str, random_state: Optional[int]):
    """Construct a default sklearn learner by name."""
    name = name.lower()
    try:
        if name == "boost":
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                random_state=random_state,
            )
        if name == "forest":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=300, max_depth=None, n_jobs=-1,
                random_state=random_state,
            )
        if name == "ridge":
            from sklearn.linear_model import RidgeCV
            return RidgeCV(alphas=np.logspace(-3, 3, 21))
        if name == "lasso":
            from sklearn.linear_model import LassoCV
            from ..compat.sklearn import lasso_cv_alphas_kwargs
            return LassoCV(cv=5, random_state=random_state,
                           max_iter=10_000, **lasso_cv_alphas_kwargs(50))
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "rd_flex requires scikit-learn. Install with: pip install scikit-learn"
        ) from exc
    raise ValueError(
        f"Unknown learner '{name}'. Choose: boost, forest, ridge, lasso."
    )


def _crossfit_residualise(
    W: np.ndarray,
    y: np.ndarray,
    learner: str,
    sklearn_estimator: Optional[Any],
    n_folds: int,
    random_state: Optional[int],
) -> tuple:
    """K-fold cross-fitted residualisation.  Returns (y - η̂(W), R²_oos)."""
    from sklearn.base import clone
    from sklearn.model_selection import KFold
    n = len(y)
    if n_folds <= 1:
        est = (clone(sklearn_estimator)
               if sklearn_estimator is not None
               else _make_learner(learner, random_state))
        est.fit(W, y)
        eta_hat = est.predict(W)
        ss_res = float(np.sum((y - eta_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return y - eta_hat, r2

    kf = KFold(n_splits=int(n_folds), shuffle=True, random_state=random_state)
    eta_hat = np.full(n, np.nan)
    base = (sklearn_estimator
            if sklearn_estimator is not None
            else _make_learner(learner, random_state))
    for train_idx, test_idx in kf.split(W):
        est = clone(base)
        est.fit(W[train_idx], y[train_idx])
        eta_hat[test_idx] = est.predict(W[test_idx])

    if not np.all(np.isfinite(eta_hat)):
        # Fold left a NaN — rare; fill with overall mean and warn.
        warnings.warn(
            "rd_flex: some out-of-fold predictions were NaN; "
            "replacing with sample mean.",
            UserWarning,
        )
        bad = ~np.isfinite(eta_hat)
        eta_hat[bad] = float(np.mean(y))

    ss_res = float(np.sum((y - eta_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return y - eta_hat, r2
