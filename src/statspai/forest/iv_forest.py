"""
Instrumental-variable generalized random forest (``grf::instrumental_forest``).

The forest estimates the conditional local average treatment effect

.. math::

    \\tau(x) = \\frac{\\mathrm{Cov}[Y, Z \\mid X = x]}{\\mathrm{Cov}[W, Z \\mid X = x]}

by solving, at every target point, the forest-weighted local IV moment
:math:`\\sum_i \\alpha_i(x)\\,(1, \\tilde Z_i)'(\\tilde Y_i - c - \\tau
\\tilde W_i) = 0` on nuisance-centred variables (Athey, Tibshirani and
Wager 2019, Sec. 5).  Trees are honest, split on the IV gradient
pseudo-outcome, and use the instrument in the stabilised split rule; the
pointwise variance is the bootstrap of little bags.

The average of :math:`\\tau(X)` -- the average conditional LATE, which
under monotonicity is a weighted complier effect -- is estimated from the
doubly-robust scores

.. math::

    \\Gamma_i = \\hat\\tau(X_i) + \\frac{Z_i - \\hat z(X_i)}
        {\\widehat{\\mathrm{Var}}(Z \\mid X_i)\\,\\hat\\Delta(X_i)}
        \\bigl(Y_i - \\hat m(X_i) - \\hat\\tau(X_i)(W_i - \\hat w(X_i))\\bigr),

where :math:`\\Delta(x) = \\mathrm{Cov}[W, Z \\mid x] / \\mathrm{Var}[Z \\mid x]`
is the compliance score (for a binary instrument, :math:`E[W \\mid x, Z=1] -
E[W \\mid x, Z=0]`), estimated by an auxiliary causal forest of ``W`` on
``Z`` unless supplied.  All training-row quantities are out-of-bag.

References
----------
[@athey2019generalized], [@aronow2013beyond], [@chernozhukov2022locally]
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .._aliases import accepts_aliases
from ..exceptions import DataInsufficient, MethodIncompatibility
from . import _grf_engine as engine
from ._grf_family import (
    ForestOptions,
    GRFFamilyForest,
    cluster_codes,
    observation_weights,
    oob_regression,
    require_finite,
    resolve_inputs,
    sample_weights,
    score_average,
    score_blp,
    user_nuisance,
    validate_alpha,
    validate_vcov_type,
    with_stream,
    z_crit,
)

_CONTEXT = "iv_forest"


class IVForestResult(GRFFamilyForest):
    """A fitted instrumental forest.

    Attributes
    ----------
    late : float
        Average conditional LATE, ``E[tau(X)]``, from the doubly-robust
        scores (grf ``average_treatment_effect`` on an instrumental forest).
    se, ci, pvalue : float, tuple, float
        Its (cluster-robust) standard error, confidence interval and p-value.
    cate : np.ndarray
        Out-of-bag ``tau(X_i)`` for the training rows.
    cate_variance : np.ndarray or None
        Little-bag variance of each out-of-bag ``tau(X_i)``.
    n_obs : int
    detail : dict
        Nuisance sources, first-stage strength, compliance-score range.

    Examples
    --------
    >>> import numpy as np, statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(600, 2)); Z = rng.binomial(1, 0.5, 600)
    >>> W = (Z + rng.normal(size=600) > 0.5).astype(float)
    >>> Y = (1 + X[:, 0]) * W + rng.normal(size=600)
    >>> fit = sp.iv_forest(y=Y, treat=W, instrument=Z, covariates=X,
    ...                    n_estimators=200)
    >>> fit.cate.shape
    (600,)
    """

    _citation_keys = ("athey2019generalized",)
    _context = _CONTEXT

    def __init__(self) -> None:
        self.late = float("nan")
        self.se = float("nan")
        self.ci = (float("nan"), float("nan"))
        self.pvalue = float("nan")
        self.alpha = 0.05
        self.cate: np.ndarray = np.zeros(0)
        self.cate_variance: Optional[np.ndarray] = None
        self.n_obs = 0
        self.feature_names: list = []
        self.detail: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    def predict(
        self, newdata: Any = None, estimate_variance: bool = False
    ) -> pd.DataFrame:
        """``tau(x)`` (and optionally its variance) at ``newdata``.

        ``newdata=None`` returns the out-of-bag predictions for the
        training rows.
        """
        if newdata is None:
            pred, var = self.cate, self.cate_variance
        else:
            if estimate_variance:
                self._require_little_bags()
            p, v = self._engine.predict(
                self._new_X(newdata), estimate_variance=bool(estimate_variance)
            )
            pred, var = p[:, 0], v[:, 0]
        out = pd.DataFrame({"predictions": np.asarray(pred, dtype=float)})
        if estimate_variance:
            if var is None:
                raise MethodIncompatibility(
                    "iv_forest: variance estimates need ci_group_size >= 2.",
                    recovery_hint="Refit with ci_group_size=2.",
                )
            out["variance_estimates"] = np.asarray(var, dtype=float)
        return out

    def get_scores(
        self,
        compliance_score: Any = None,
        debiasing_weights: Any = None,
    ) -> np.ndarray:
        """Doubly-robust scores ``Gamma_i`` for ``E[tau(X)]``.

        ``compliance_score`` (``Delta(X_i)``) or the full
        ``debiasing_weights`` may be supplied; otherwise the stored
        estimates from the fit are used.
        """
        n = self.n_obs
        if debiasing_weights is not None:
            g = user_nuisance(debiasing_weights, n, 1, "debiasing_weights", _CONTEXT)[
                :, 0
            ]
        else:
            if compliance_score is not None:
                delta = user_nuisance(
                    compliance_score, n, 1, "compliance_score", _CONTEXT
                )[:, 0]
            else:
                delta = self._compliance
            if np.any(np.abs(delta) < 1e-12):
                raise DataInsufficient(
                    "iv_forest: the compliance score is zero for some rows, so "
                    "the doubly-robust weights are undefined.",
                    recovery_hint=(
                        "The instrument does not move the treatment for those "
                        "covariate values; check first-stage strength."
                    ),
                )
            g = iv_debiasing_weights(self._Z, self._z_hat, self._z_var, delta)
        return iv_dr_scores(self._Y, self._W, self._y_hat, self._w_hat, self.cate, g)

    def average_treatment_effect(
        self,
        alpha: float = 0.05,
        compliance_score: Any = None,
        debiasing_weights: Any = None,
    ) -> Dict[str, Any]:
        """Average conditional LATE with its (cluster-robust) SE."""
        alpha = validate_alpha(alpha, f"{_CONTEXT}.average_treatment_effect()")
        scores = self.get_scores(compliance_score, debiasing_weights)
        out = score_average(scores, self._obs_weight, self._clusters, alpha)
        out.update(
            estimand="ACLATE",
            method="aipw_instrumental",
            n=self.n_obs,
            n_clusters=(
                None if self._clusters is None else int(self._clusters.max()) + 1
            ),
            alpha=alpha,
        )
        return out

    @accepts_aliases(_strict=True, vcov_type="vce")
    def best_linear_projection(
        self,
        A: Any = None,
        vce: str = "HC3",
        alpha: float = 0.05,
        compliance_score: Any = None,
        debiasing_weights: Any = None,
    ) -> pd.DataFrame:
        """Regress the doubly-robust scores on ``(1, A)``
        (``grf::best_linear_projection``)."""
        from .forest_tools import _projection_design

        alpha = validate_alpha(alpha, f"{_CONTEXT}.best_linear_projection()")
        vce = validate_vcov_type(vce, _CONTEXT)
        A_mat, names = _projection_design(A, self, _CONTEXT)
        scores = self.get_scores(compliance_score, debiasing_weights)
        return score_blp(
            scores,
            A_mat,
            names,
            self._obs_weight,
            self._clusters,
            vce,
            alpha,
            f"instrumental-forest DR scores on OOB CATE, {vce} robust SE",
        )

    def summary(self) -> str:
        lo, hi = self.ci
        return (
            "Instrumental forest (grf-style GRF)\n"
            "-----------------------------------\n"
            f"  N              : {self.n_obs}\n"
            f"  trees          : {self.num_trees}\n"
            f"  ACLATE         : {self.late:.4f}\n"
            f"  SE             : {self.se:.4f}\n"
            f"  {100 * (1 - self.alpha):.0f}% CI         : [{lo:.4f}, {hi:.4f}]\n"
            f"  p-value        : {self.pvalue:.4g}\n"
            f"  CATE (OOB)     : mean {np.mean(self.cate):.4f}, "
            f"sd {np.std(self.cate):.4f}"
        )

    def __repr__(self) -> str:
        return (
            f"IVForestResult(ACLATE={self.late:.4f}, se={self.se:.4f}, n={self.n_obs})"
        )


def iv_dr_scores(
    Y: np.ndarray,
    W: np.ndarray,
    y_hat: np.ndarray,
    w_hat: np.ndarray,
    tau: np.ndarray,
    debiasing_weights: np.ndarray,
) -> np.ndarray:
    """``Gamma_i = tau_i + g_i (Y_i - m_i - tau_i (W_i - w_i))`` with
    ``g_i = (Z_i - z_i) / (Var(Z | X_i) Delta(X_i))``."""
    return np.asarray(
        tau + debiasing_weights * (Y - y_hat - tau * (W - w_hat)), dtype=float
    )


def iv_debiasing_weights(
    Z: np.ndarray, z_hat: np.ndarray, z_var: np.ndarray, compliance: np.ndarray
) -> np.ndarray:
    """Riesz representer of the average conditional LATE."""
    return np.asarray((Z - z_hat) / (z_var * compliance), dtype=float)


def _compliance_forest(
    X: np.ndarray,
    Wc: np.ndarray,
    Zc: np.ndarray,
    common: Dict[str, Any],
    opts: ForestOptions,
) -> np.ndarray:
    """OOB ``Delta(X) = Cov(W, Z | X) / Var(Z | X)`` from a causal forest of
    ``W`` on ``Z`` (both already centred by their nuisances)."""
    cf = engine.train_forest(
        X,
        Wc,
        Zc,
        kind=engine.KIND_CAUSAL,
        num_trees=500,
        mtry=opts.mtry,
        min_node_size=5,
        ci_group_size=1,
        stabilize_splits=True,
        **with_stream(common, "compliance"),
    )
    delta, _ = cf.predict_oob(X)
    require_finite(delta, "compliance-score", _CONTEXT)
    return np.asarray(delta, dtype=float)


@accepts_aliases(
    _strict=True,
    n_trees="n_estimators",
    min_leaf="min_samples_leaf",
)
def iv_forest(
    data: Optional[pd.DataFrame] = None,
    y: Any = None,
    treat: Any = None,
    instrument: Any = None,
    covariates: Any = None,
    *,
    clusters: Any = None,
    weights: Any = None,
    equalize_cluster_weights: bool = False,
    Y_hat: Any = None,
    W_hat: Any = None,
    Z_hat: Any = None,
    compliance_score: Any = None,
    n_estimators: int = 2000,
    min_samples_leaf: int = 5,
    max_samples: float = 0.5,
    mtry: Optional[int] = None,
    honest: bool = True,
    honesty_fraction: float = 0.5,
    honesty_prune_leaves: bool = True,
    split_alpha: float = 0.05,
    imbalance_penalty: float = 0.0,
    stabilize_splits: bool = True,
    ci_group_size: int = 2,
    reduced_form_weight: float = 0.0,
    max_depth: Optional[int] = None,
    random_state: Optional[int] = 42,
    n_jobs: int = 1,
    alpha: float = 0.05,
    n_bootstrap: Optional[int] = None,
) -> IVForestResult:
    """
    Instrumental forest: heterogeneous local average treatment effects.

    A generalized random forest for ``tau(x) = Cov[Y, Z | x] / Cov[W, Z | x]``
    (``grf::instrumental_forest``), reporting out-of-bag conditional effects
    with little-bag variances and the doubly-robust average conditional
    LATE.

    Parameters
    ----------
    data : pd.DataFrame, optional
        Input data.  When omitted, ``y``, ``treat``, ``instrument`` and
        ``covariates`` are arrays.
    y, treat, instrument : str or array-like
        Outcome, treatment (binary or continuous) and instrument (binary or
        continuous).
    covariates : str, list of str or 2-D array
        Effect modifiers ``X``; the nuisances are also regressed on them.
    clusters : str or array-like, optional
        Cluster ids: trees sample whole clusters and every standard error
        is cluster-robust.
    weights : str or array-like, optional
        Sample weights (grf ``sample.weights``).
    equalize_cluster_weights : bool, default False
        Give every cluster the same weight (incompatible with ``weights``).
    Y_hat, W_hat, Z_hat : array-like, optional
        Precomputed ``E[Y|X]``, ``E[W|X]``, ``E[Z|X]``; by default each is the
        out-of-bag prediction of a regression forest with
        ``max(50, n_estimators // 4)`` trees.
    compliance_score : array-like, optional
        ``Delta(X_i)`` for the average-effect scores; by default estimated
        by an auxiliary 500-tree causal forest of ``W`` on ``Z``.
    n_estimators : int, default 2000
        Trees (grf ``num.trees``).
    min_samples_leaf : int, default 5
        grf ``min.node.size``.
    max_samples : float, default 0.5
        Fraction of clusters drawn per tree (grf ``sample.fraction``).
    mtry : int, optional
        Candidate variables per split (grf default ``min(ceil(sqrt(p) + 20), p)``).
    honest, honesty_fraction, honesty_prune_leaves
        grf honesty options.
    split_alpha : float, default 0.05
        grf ``alpha``: minimum share of the parent's instrument variation
        each child of a stabilised split keeps.
    imbalance_penalty : float, default 0
    stabilize_splits : bool, default True
        Use the instrument in the split constraints.
    ci_group_size : int, default 2
        Trees per little bag; 1 disables variance estimates.
    reduced_form_weight : float in [0, 1], default 0
        Mix the IV split criterion with the causal-forest criterion that
        treats ``W`` as exogenous (grf ``reduced.form.weight``).
    max_depth : int, optional
        Depth cap (no grf analogue).
    random_state : int, default 42
    n_jobs : int, default 1
    alpha : float, default 0.05
        Significance level of the reported interval.
    n_bootstrap : deprecated
        Ignored.  Standard errors now come from the doubly-robust scores.

    Returns
    -------
    IVForestResult
        ``.late`` / ``.se`` / ``.ci`` / ``.pvalue`` (average conditional
        LATE), ``.cate`` (OOB), ``.predict(newdata)``,
        ``.best_linear_projection(A)``, ``.variable_importance()``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 800
    >>> x = rng.normal(size=(n, 3)); z = rng.binomial(1, 0.5, n)
    >>> u = rng.normal(size=n)
    >>> w = (1.2 * z + 0.6 * u + rng.normal(scale=0.5, size=n) > 0.6).astype(float)
    >>> y = (1 + x[:, 0]) * w + u + rng.normal(size=n)
    >>> df = pd.DataFrame({"y": y, "w": w, "z": z, "x1": x[:, 0],
    ...                    "x2": x[:, 1], "x3": x[:, 2]})
    >>> fit = sp.iv_forest(df, y="y", treat="w", instrument="z",
    ...                    covariates=["x1", "x2", "x3"], n_estimators=200)
    >>> round(fit.late, 1)  # doctest: +SKIP
    1.0

    References
    ----------
    [@athey2019generalized], [@aronow2013beyond], [@chernozhukov2022locally]
    """
    if n_bootstrap is not None:
        warnings.warn(
            "iv_forest(n_bootstrap=) is deprecated and ignored: the LATE's "
            "standard error now comes from the doubly-robust scores, not a "
            "bootstrap of fixed residuals.",
            DeprecationWarning,
            stacklevel=3,
        )
    alpha = validate_alpha(alpha, _CONTEXT)
    rfw = float(reduced_form_weight)
    if not 0.0 <= rfw <= 1.0:
        raise MethodIncompatibility(
            f"{_CONTEXT}: reduced_form_weight must be in [0, 1].",
            recovery_hint="Use 0 (pure IV splitting) up to 1.",
            diagnostics={"reduced_form_weight": reduced_form_weight},
        )
    opts = ForestOptions(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_samples=max_samples,
        mtry=mtry,
        honest=honest,
        honesty_fraction=honesty_fraction,
        honesty_prune_leaves=honesty_prune_leaves,
        split_alpha=split_alpha,
        imbalance_penalty=imbalance_penalty,
        stabilize_splits=stabilize_splits,
        ci_group_size=ci_group_size,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    opts.validate(_CONTEXT)
    arrays, names, keep, n_input = resolve_inputs(
        _CONTEXT,
        data,
        {"y": y, "treat": treat, "instrument": instrument, "covariates": covariates},
        {"clusters": clusters, "weights": weights},
    )
    Y = arrays["y"][:, 0]
    W = arrays["treat"][:, 0]
    Z = arrays["instrument"][:, 0]
    X = arrays["covariates"]
    n = Y.size
    for nm, arr in (
        ("y", arrays["y"]),
        ("treat", arrays["treat"]),
        ("instrument", arrays["instrument"]),
    ):
        if arr.shape[1] != 1:
            raise MethodIncompatibility(
                f"{_CONTEXT}: {nm} must be a single column.",
                recovery_hint=f"Pass one {nm} column.",
            )
    if np.var(Z) <= 0:
        raise DataInsufficient(
            f"{_CONTEXT}: the instrument has no variation.",
            recovery_hint="Pass an instrument that varies across rows.",
        )
    cl = cluster_codes(arrays.get("clusters"), _CONTEXT)
    sw = sample_weights(arrays.get("weights"), equalize_cluster_weights, _CONTEXT)
    common = opts.engine_kwargs(cl, equalize_cluster_weights, sw)

    sources = {}
    nuis = {}
    for key, target, supplied in (
        ("Y_hat", Y, Y_hat),
        ("W_hat", W, W_hat),
        ("Z_hat", Z, Z_hat),
    ):
        if supplied is not None:
            supplied_arr = np.asarray(supplied, dtype=float).ravel()
            if supplied_arr.size == n_input and n_input != n:
                supplied_arr = supplied_arr[keep]
            nuis[key] = user_nuisance(supplied_arr, n, 1, key, _CONTEXT)[:, 0]
            sources[key] = "user-supplied"
        else:
            nuis[key] = oob_regression(X, target, opts, common, key, _CONTEXT)
            sources[key] = "regression forest (OOB)"
    Yc = Y - nuis["Y_hat"]
    Wc = W - nuis["W_hat"]
    Zc = Z - nuis["Z_hat"]

    forest = engine.train_forest(
        X,
        np.zeros(n),
        kind=engine.KIND_INSTRUMENTAL,
        num_trees=opts.n_estimators,
        mtry=opts.mtry,
        min_node_size=int(opts.min_samples_leaf),
        stabilize_splits=bool(opts.stabilize_splits),
        ci_group_size=int(opts.ci_group_size),
        max_depth=opts.max_depth,
        M=np.column_stack([Yc, Wc, Zc]),
        params=np.array([rfw]),
        **common,
    )
    tau, var = forest.predict_oob(X, estimate_variance=opts.ci_group_size > 1)
    tau = tau[:, 0]
    require_finite(tau, "instrumental", _CONTEXT)

    binary_z = bool(np.all(np.isin(np.unique(Z), (0.0, 1.0))))
    if binary_z:
        z_var = nuis["Z_hat"] * (1.0 - nuis["Z_hat"])
        z_var_source = "Z_hat (1 - Z_hat)"
    else:
        vf = engine.train_forest(
            X,
            Zc**2,
            kind=engine.KIND_REGRESSION,
            num_trees=500,
            ci_group_size=1,
            **with_stream(common, "var_z"),
        )
        z_var, _ = vf.predict_oob(X)
        require_finite(z_var, "Var(Z|X)", _CONTEXT)
        z_var_source = "regression forest of (Z - Z_hat)^2 (OOB)"
    if np.any(z_var <= 0):
        raise DataInsufficient(
            f"{_CONTEXT}: the estimated Var(Z | X) is not positive for "
            f"{int(np.sum(z_var <= 0))} row(s), so the doubly-robust weights "
            "are undefined.",
            recovery_hint=(
                "Instrument propensities of exactly 0 or 1 mean no instrument "
                "variation at those covariates; trim the sample or pass Z_hat."
            ),
        )
    if compliance_score is not None:
        cs = np.asarray(compliance_score, dtype=float).ravel()
        if cs.size == n_input and n_input != n:
            cs = cs[keep]
        delta = user_nuisance(cs, n, 1, "compliance_score", _CONTEXT)[:, 0]
        sources["compliance_score"] = "user-supplied"
    else:
        delta = _compliance_forest(X, Wc, Zc, common, opts)
        sources["compliance_score"] = "causal forest of W on Z (OOB)"

    res = IVForestResult()
    res._engine = forest
    res._X = X
    res._Y, res._W, res._Z = Y, W, Z
    res._y_hat, res._w_hat, res._z_hat = nuis["Y_hat"], nuis["W_hat"], nuis["Z_hat"]
    res._z_var = np.asarray(z_var, dtype=float)
    res._compliance = delta
    res._clusters = cl
    res._obs_weight = observation_weights(cl, equalize_cluster_weights, sw, n)
    res.feature_names = names["covariates"]
    res.n_obs = n
    res.alpha = alpha
    res.cate = tau
    res.cate_variance = var[:, 0] if opts.ci_group_size > 1 else None
    ate = res.average_treatment_effect(alpha=alpha)
    res.late = ate["estimate"]
    res.se = ate["se"]
    z = z_crit(alpha)
    res.ci = (res.late - z * res.se, res.late + z * res.se)
    res.pvalue = ate["pvalue"]
    first_stage = float(np.corrcoef(Zc, Wc)[0, 1]) if np.std(Wc) > 0 else float("nan")
    res.detail = {
        "nuisance_source": sources,
        "first_stage_corr": first_stage,
        "compliance_score_range": (float(np.min(delta)), float(np.max(delta))),
        "instrument": "binary" if binary_z else "continuous",
        "var_z_source": z_var_source,
        "n_dropped_missing": int(n_input - n),
        "n_clusters": None if cl is None else int(cl.max()) + 1,
        "options": dict(opts.__dict__),
        "reduced_form_weight": rfw,
    }
    if np.min(np.abs(delta)) < 0.05:
        warnings.warn(
            f"{_CONTEXT}: the estimated compliance score is near zero for "
            f"{int(np.sum(np.abs(delta) < 0.05))} row(s) (min |Delta| = "
            f"{float(np.min(np.abs(delta))):.3g}); the average-effect scores "
            "divide by it and may be unstable (weak instrument locally).",
            UserWarning,
            stacklevel=3,
        )
    from ..output._lineage import attach_provenance as _attach_prov

    _attach_prov(
        res,
        function="sp.iv_forest",
        params={
            "y": y if isinstance(y, str) else "<array>",
            "treat": treat if isinstance(treat, str) else "<array>",
            "instrument": instrument if isinstance(instrument, str) else "<array>",
            "covariates": list(names["covariates"]),
            "n_estimators": int(opts.n_estimators),
            "min_samples_leaf": int(opts.min_samples_leaf),
            "reduced_form_weight": rfw,
            "random_state": random_state,
            "alpha": alpha,
        },
        data=data,
        overwrite=False,
    )
    return res


def instrumental_forest(*args: Any, **kwargs: Any) -> IVForestResult:
    """Alias of :func:`iv_forest` under grf's name."""
    return iv_forest(*args, **kwargs)


instrumental_forest.__doc__ = iv_forest.__doc__

__all__ = [
    "iv_forest",
    "instrumental_forest",
    "IVForestResult",
    "iv_dr_scores",
    "iv_debiasing_weights",
]
