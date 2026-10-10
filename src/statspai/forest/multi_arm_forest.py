"""
Multi-arm causal forest (``grf::multi_arm_causal_forest``).

With ``K`` mutually exclusive arms and a reference arm ``0``, the forest
estimates the ``K - 1`` contrasts

.. math::

    \\tau_k(x) = E[Y(k) - Y(0) \\mid X = x], \\qquad k = 1, \\dots, K - 1,

jointly: it is the linear-model forest (:func:`statspai.lm_forest`) of the
nuisance-centred outcome on the centred arm indicators, the multi-arm
R-learner of Nie and Wager (2021) with GRF weights, whose trees split on
the gradient of all contrasts at once and apply the causal-forest split
constraints to every arm.  Propensities ``e_k(x)`` come from a probability
forest and ``E[Y | x]`` from a regression forest, both out-of-bag.

Average effects use the AIPW scores of each contrast,

.. math::

    \\Gamma_{ik} = \\hat\\tau_k(X_i)
      + \\Bigl(\\frac{1\\{W_i = k\\}}{\\hat e_k(X_i)}
             - \\frac{1\\{W_i = 0\\}}{\\hat e_0(X_i)}\\Bigr)
        \\bigl(Y_i - \\hat\\mu_{W_i}(X_i)\\bigr),

with ``mu_0 = m - sum_k e_k tau_k`` and ``mu_k = mu_0 + tau_k`` built from
the forest's own out-of-bag predictions.

References
----------
[@athey2019generalized], [@nie2021quasi], [@robins1994estimation]
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .._aliases import accepts_aliases
from ..exceptions import DataInsufficient, MethodIncompatibility
from . import _grf_engine as engine
from ._grf_family import (
    ForestOptions,
    GRFFamilyForest,
    cluster_codes,
    nuisance_trees,
    observation_weights,
    require_finite,
    resolve_inputs,
    sample_weights,
    score_average,
    score_blp,
    user_nuisance,
    validate_alpha,
    validate_vcov_type,
    with_stream,
)
from .lm_forest import _rows, fit_multi_causal, oob_multi_regression

_CONTEXT = "multi_arm_forest"


class MultiArmForestResult(GRFFamilyForest):
    """A fitted multi-arm causal forest.

    Attributes
    ----------
    arms : list
        Arm labels, reference first.
    reference : Any
        The reference arm.
    ate, ate_se, ci, pvalue : dict
        Doubly-robust average effect of each non-reference arm against the
        reference (keyed by arm label).
    cate : dict
        Out-of-bag ``tau_k(X_i)`` per non-reference arm.
    cate_variance : dict or None
        Little-bag variances of ``cate``.
    propensities : pd.DataFrame
        Out-of-bag ``e_k(X_i)``, one column per arm.

    Examples
    --------
    >>> import numpy as np, statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(600, 2)); A = rng.integers(0, 3, 600)
    >>> Y = X[:, 1] + (A == 1) - 0.5 * (A == 2) + rng.normal(size=600)
    >>> fit = sp.multi_arm_forest(y=Y, treat=A, covariates=X, n_estimators=200)
    >>> fit.contrasts
    [1, 2]
    """

    _citation_keys = ("athey2019generalized", "nie2021quasi")
    _context = _CONTEXT

    def __init__(self) -> None:
        self.arms: List[Any] = []
        self.reference: Any = None
        self.ate: Dict[Any, float] = {}
        self.ate_se: Dict[Any, float] = {}
        self.ci: Dict[Any, tuple] = {}
        self.pvalue: Dict[Any, float] = {}
        self.cate: Dict[Any, np.ndarray] = {}
        self.cate_variance: Optional[Dict[Any, np.ndarray]] = None
        self.alpha = 0.05
        self.n_obs = 0
        self.feature_names: List[str] = []
        self.propensities: pd.DataFrame = pd.DataFrame()
        self.detail: Dict[str, Any] = {}

    @property
    def contrasts(self) -> List[Any]:
        return [a for a in self.arms if a != self.reference]

    def predict(
        self, newdata: Any = None, estimate_variance: bool = False
    ) -> pd.DataFrame:
        """``tau_k(x)`` for every contrast (columns ``"<arm> - <reference>"``)."""
        cols = [f"{a} - {self.reference}" for a in self.contrasts]
        if newdata is None:
            pred = np.column_stack([self.cate[a] for a in self.contrasts])
            var = (
                None
                if self.cate_variance is None
                else np.column_stack([self.cate_variance[a] for a in self.contrasts])
            )
        else:
            if estimate_variance:
                self._require_little_bags()
            p, v = self._engine.predict(
                self._new_X(newdata), estimate_variance=bool(estimate_variance)
            )
            pred, var = p, v
        out = pd.DataFrame(np.asarray(pred, dtype=float), columns=cols)
        if estimate_variance:
            if var is None:
                raise MethodIncompatibility(
                    f"{_CONTEXT}: variance estimates need ci_group_size >= 2.",
                    recovery_hint="Refit with ci_group_size=2.",
                )
            for j, c in enumerate(cols):
                out[f"variance[{c}]"] = np.asarray(var, dtype=float)[:, j]
        return out

    def get_scores(self) -> np.ndarray:
        """AIPW scores, ``(n, K-1)``, one column per contrast."""
        tau = np.column_stack([self.cate[a] for a in self.contrasts])
        return multi_arm_scores(
            self._Y, self._arm_codes, self._y_hat, self._propensity, tau
        )

    def average_treatment_effect(self, alpha: float = 0.05) -> pd.DataFrame:
        """Doubly-robust average effect of each arm against the reference."""
        alpha = validate_alpha(alpha, f"{_CONTEXT}.average_treatment_effect()")
        scores = self.get_scores()
        rows = []
        for j, a in enumerate(self.contrasts):
            r = score_average(scores[:, j], self._obs_weight, self._clusters, alpha)
            rows.append(
                {
                    "contrast": f"{a} - {self.reference}",
                    "estimate": r["estimate"],
                    "se": r["se"],
                    "ci_low": r["ci_low"],
                    "ci_high": r["ci_high"],
                    "pvalue": r["pvalue"],
                }
            )
        out = pd.DataFrame(rows).set_index("contrast")
        out.attrs["method"] = "AIPW scores on OOB CATE (multi-arm)"
        return out

    @accepts_aliases(_strict=True, vcov_type="vce")
    def best_linear_projection(
        self,
        A: Any = None,
        vce: str = "HC3",
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """BLP of each contrast's CATE on ``(1, A)``, stacked with a
        ``contrast`` index level."""
        from .forest_tools import _projection_design

        alpha = validate_alpha(alpha, f"{_CONTEXT}.best_linear_projection()")
        vce = validate_vcov_type(vce, _CONTEXT)
        A_mat, names = _projection_design(A, self, _CONTEXT)
        scores = self.get_scores()
        frames = []
        for j, a in enumerate(self.contrasts):
            tab = score_blp(
                scores[:, j],
                A_mat,
                names,
                self._obs_weight,
                self._clusters,
                vce,
                alpha,
                f"multi-arm AIPW scores on OOB CATE, {vce} robust SE",
            )
            tab.index = pd.MultiIndex.from_product(
                [[f"{a} - {self.reference}"], tab.index], names=["contrast", "term"]
            )
            frames.append(tab)
        return pd.concat(frames)

    def summary(self) -> str:
        rows = [
            "Multi-arm causal forest (GRF engine)",
            f"  N          : {self.n_obs}",
            f"  trees      : {self.num_trees}",
            f"  reference  : {self.reference}",
        ]
        for a in self.contrasts:
            lo, hi = self.ci[a]
            rows.append(
                f"  {a} vs {self.reference}: ATE={self.ate[a]:+.4f}  "
                f"SE={self.ate_se[a]:.4f}  CI=[{lo:+.4f}, {hi:+.4f}]"
            )
        return "\n".join(rows)

    def __repr__(self) -> str:
        return f"MultiArmForestResult(K={len(self.arms)}, n={self.n_obs})"


def multi_arm_scores(
    Y: np.ndarray,
    arm_codes: np.ndarray,
    y_hat: np.ndarray,
    propensity: np.ndarray,
    tau: np.ndarray,
) -> np.ndarray:
    """AIPW scores ``(n, K-1)`` of each arm against arm code 0.

    ``mu_0 = m - sum_k e_k tau_k`` and ``mu_k = mu_0 + tau_k``; the score of
    contrast ``k`` is ``tau_k + (1{W=k}/e_k - 1{W=0}/e_0)(Y - mu_W)``.
    """
    e = np.asarray(propensity, dtype=float)
    tau = np.asarray(tau, dtype=float)
    ref = (arm_codes == 0).astype(float)
    mu0 = y_hat - np.sum(e[:, 1:] * tau, axis=1)
    mu_obs = mu0.copy()
    for j in range(tau.shape[1]):
        sel = arm_codes == j + 1
        mu_obs[sel] = mu0[sel] + tau[sel, j]
    resid = Y - mu_obs
    scores = np.empty_like(tau)
    for j in range(tau.shape[1]):
        ind = (arm_codes == j + 1).astype(float)
        scores[:, j] = tau[:, j] + (ind / e[:, j + 1] - ref / e[:, 0]) * resid
    return scores


def _oob_probabilities(
    X: np.ndarray,
    codes: np.ndarray,
    K: int,
    opts: ForestOptions,
    common: Dict[str, Any],
) -> np.ndarray:
    """Out-of-bag arm probabilities from a probability forest."""
    onehot = np.zeros((codes.size, K))
    onehot[np.arange(codes.size), codes] = 1.0
    mtry = None if opts.mtry is None else int(min(max(int(opts.mtry), 1), X.shape[1]))
    forest = engine.train_forest(
        X,
        np.zeros(codes.size),
        kind=engine.KIND_MULTI_REG,
        num_trees=nuisance_trees(opts.n_estimators),
        mtry=mtry,
        min_node_size=5,
        ci_group_size=1,
        M=onehot,
        params=np.array([float(K)]),
        **with_stream(common, "W_hat"),
    )
    pred, _ = forest.predict_oob(X)
    require_finite(pred, "W_hat (propensity)", _CONTEXT)
    return np.asarray(pred, dtype=float)


@accepts_aliases(_strict=True, n_trees="n_estimators", min_leaf="min_samples_leaf")
def multi_arm_forest(
    data: Optional[pd.DataFrame] = None,
    y: Any = None,
    treat: Any = None,
    covariates: Any = None,
    *,
    reference: Any = None,
    clusters: Any = None,
    weights: Any = None,
    equalize_cluster_weights: bool = False,
    Y_hat: Any = None,
    W_hat: Any = None,
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
    max_depth: Optional[int] = None,
    random_state: Optional[int] = 42,
    n_jobs: int = 1,
    alpha: float = 0.05,
    propensity_bounds: Optional[tuple] = None,
) -> MultiArmForestResult:
    """
    Multi-arm causal forest: CATEs of ``K - 1`` arms against a reference.

    Parameters
    ----------
    data : pd.DataFrame, optional
        Input data; when omitted, the other inputs are arrays.
    y : str or array-like
        Outcome.
    treat : str or array-like
        Arm labels (any values; at least two arms).
    covariates : str, list of str or 2-D array
        Covariates ``X``.
    reference : optional
        Reference arm; defaults to the smallest label (``0`` for integer
        arms, matching the previous API).
    clusters, weights, equalize_cluster_weights
        As in :func:`statspai.causal_forest`.
    Y_hat : array-like, optional
        ``E[Y | X]``; default out-of-bag regression forest.
    W_hat : array-like, optional
        ``(n, K)`` arm propensities in the order of ``.arms``; default
        out-of-bag probability forest.
    n_estimators, min_samples_leaf, max_samples, mtry, honest,
    honesty_fraction, honesty_prune_leaves, split_alpha, imbalance_penalty,
    stabilize_splits, ci_group_size, max_depth, random_state, n_jobs
        Forest options (grf ``num.trees``, ``min.node.size``,
        ``sample.fraction``, ..., ``alpha``).
    alpha : float, default 0.05
        Significance level for the reported intervals.
    propensity_bounds : (float, float), optional
        Clip the estimated propensities to these bounds in the average-
        effect scores (not done by default, as in grf).

    Returns
    -------
    MultiArmForestResult
        ``.ate`` / ``.ate_se`` / ``.ci`` (dicts keyed by arm), ``.cate``
        (OOB, per arm), ``.predict(newdata)``, ``.average_treatment_effect()``,
        ``.best_linear_projection(A)``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 600; x = rng.normal(size=(n, 3)); a = rng.integers(0, 3, n)
    >>> y = x[:, 1] + (1 + x[:, 0]) * (a == 1) - 0.5 * (a == 2) + rng.normal(size=n)
    >>> df = pd.DataFrame({"y": y, "a": a, "x1": x[:, 0], "x2": x[:, 1],
    ...                    "x3": x[:, 2]})
    >>> fit = sp.multi_arm_forest(df, y="y", treat="a",
    ...                           covariates=["x1", "x2", "x3"], n_estimators=200)
    >>> sorted(fit.ate)
    [1, 2]

    References
    ----------
    [@athey2019generalized], [@nie2021quasi]
    """
    alpha = validate_alpha(alpha, _CONTEXT)
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
    bounds = None
    if propensity_bounds is not None:
        if len(propensity_bounds) != 2:
            raise MethodIncompatibility(
                f"{_CONTEXT}: propensity_bounds must contain two values.",
                recovery_hint="Use propensity_bounds=(0.01, 0.99) or None.",
            )
        lo, hi = float(propensity_bounds[0]), float(propensity_bounds[1])
        if not 0 < lo < hi < 1:
            raise MethodIncompatibility(
                f"{_CONTEXT}: propensity_bounds must satisfy 0 < lower < upper < 1.",
                recovery_hint="Use bounds such as (0.01, 0.99).",
                diagnostics={"propensity_bounds": [lo, hi]},
            )
        bounds = (lo, hi)

    labels = treat
    if data is not None and isinstance(treat, str):
        if treat not in data.columns:
            raise MethodIncompatibility(
                f"{_CONTEXT}: Missing columns: {[treat]}",
                recovery_hint="Pass column names present in the input DataFrame.",
            )
        labels = data[treat].to_numpy()
    labels = np.asarray(labels).ravel()
    present = ~pd.isna(pd.Series(labels)).to_numpy()
    arms_all = list(np.unique(labels[present]))
    if len(arms_all) < 2:
        raise DataInsufficient(
            f"{_CONTEXT}: need at least two treatment arms.",
            recovery_hint="Pass a treatment with two or more distinct values.",
        )
    if reference is None:
        reference = arms_all[0]
    if reference not in arms_all:
        raise MethodIncompatibility(
            f"{_CONTEXT}: reference arm {reference!r} is not a treatment value.",
            recovery_hint=f"Choose one of {arms_all[:10]}.",
        )
    arms = [reference] + [a for a in arms_all if a != reference]
    code_of = {a: i for i, a in enumerate(arms)}
    codes_all = np.full(labels.size, np.nan)
    codes_all[present] = [code_of[v] for v in labels[present]]

    arrays, names, keep, n_input = resolve_inputs(
        _CONTEXT,
        data,
        {"y": y, "treat": codes_all, "covariates": covariates},
        {"clusters": clusters, "weights": weights},
    )
    X = arrays["covariates"]
    Y = arrays["y"]
    if Y.shape[1] != 1:
        raise MethodIncompatibility(
            f"{_CONTEXT}: y must be one column.",
            recovery_hint="Fit one outcome at a time, or use sp.lm_forest.",
        )
    Yv = Y[:, 0]
    codes = arrays["treat"][:, 0].astype(int)
    n = Yv.size
    K = len(arms)
    counts = np.bincount(codes, minlength=K)
    if np.any(counts == 0):
        raise DataInsufficient(
            f"{_CONTEXT}: some arms have no complete rows.",
            recovery_hint="Drop empty arms before fitting.",
            diagnostics={"arm_counts": dict(zip(map(str, arms), counts.tolist()))},
        )
    cl = cluster_codes(arrays.get("clusters"), _CONTEXT)
    sw = sample_weights(arrays.get("weights"), equalize_cluster_weights, _CONTEXT)
    common = opts.engine_kwargs(cl, equalize_cluster_weights, sw)

    if Y_hat is None:
        y_hat = oob_multi_regression(X, Y, opts, common, "Y_hat", _CONTEXT)[:, 0]
        y_src = "regression forest (OOB)"
    else:
        y_hat = user_nuisance(_rows(Y_hat, keep, n_input), n, 1, "Y_hat", _CONTEXT)[
            :, 0
        ]
        y_src = "user-supplied"
    if W_hat is None:
        e = _oob_probabilities(X, codes, K, opts, common)
        e_src = "probability forest (OOB)"
    else:
        e = user_nuisance(_rows(W_hat, keep, n_input), n, K, "W_hat", _CONTEXT)
        e_src = "user-supplied"
    onehot = np.zeros((n, K))
    onehot[np.arange(n), codes] = 1.0
    forest, coef, var = fit_multi_causal(
        _CONTEXT, X, (Yv - y_hat)[:, None], onehot[:, 1:] - e[:, 1:], opts, common
    )

    res = MultiArmForestResult()
    res._engine = forest
    res._X = X
    res._Y = Yv
    res._y_hat = y_hat
    res._arm_codes = codes
    e_scores = e if bounds is None else np.clip(e, bounds[0], bounds[1])
    if np.any(e_scores <= 0):
        raise DataInsufficient(
            f"{_CONTEXT}: some estimated arm propensities are 0, so the "
            "doubly-robust scores are undefined.",
            recovery_hint="Pass propensity_bounds=(0.01, 0.99) or trim the sample.",
        )
    res._propensity = e_scores
    res._clusters = cl
    res._obs_weight = observation_weights(cl, equalize_cluster_weights, sw, n)
    res.arms = [a.item() if hasattr(a, "item") else a for a in arms]
    res.reference = res.arms[0]
    res.alpha = alpha
    res.n_obs = n
    res.feature_names = names["covariates"]
    res.cate = {a: coef[:, j, 0] for j, a in enumerate(res.arms[1:])}
    res.cate_variance = (
        None if var is None else {a: var[:, j, 0] for j, a in enumerate(res.arms[1:])}
    )
    res.propensities = pd.DataFrame(e, columns=[str(a) for a in res.arms])
    ate_tab = res.average_treatment_effect(alpha=alpha)
    for j, a in enumerate(res.arms[1:]):
        row = ate_tab.iloc[j]
        res.ate[a] = float(row["estimate"])
        res.ate_se[a] = float(row["se"])
        res.ci[a] = (float(row["ci_low"]), float(row["ci_high"]))
        res.pvalue[a] = float(row["pvalue"])
    res.detail = {
        "nuisance_source": {"Y_hat": y_src, "W_hat": e_src},
        "propensity_ranges": {
            str(a): (float(e[:, j].min()), float(e[:, j].max()))
            for j, a in enumerate(res.arms)
        },
        "propensity_bounds": bounds,
        "arm_counts": {str(a): int(c) for a, c in zip(res.arms, counts)},
        "n_dropped_missing": int(n_input - n),
        "n_clusters": None if cl is None else int(cl.max()) + 1,
        "options": dict(opts.__dict__),
    }
    small = float(np.min(e))
    if bounds is None and small < 0.01:
        warnings.warn(
            f"{_CONTEXT}: the smallest estimated arm propensity is {small:.3g}; "
            "the average-effect scores divide by it (poor overlap). Consider "
            "propensity_bounds= or trimming.",
            UserWarning,
            stacklevel=3,
        )
    from ..output._lineage import attach_provenance as _attach_prov

    _attach_prov(
        res,
        function="sp.multi_arm_forest",
        params={
            "y": y if isinstance(y, str) else "<array>",
            "treat": treat if isinstance(treat, str) else "<array>",
            "covariates": list(names["covariates"]),
            "reference": str(res.reference),
            "n_estimators": int(opts.n_estimators),
            "min_samples_leaf": int(opts.min_samples_leaf),
            "random_state": random_state,
            "alpha": alpha,
        },
        data=data,
        overwrite=False,
    )
    return res


__all__ = ["multi_arm_forest", "MultiArmForestResult", "multi_arm_scores"]
