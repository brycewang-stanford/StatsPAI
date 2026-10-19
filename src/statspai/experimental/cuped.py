"""CUPED: variance reduction for A/B tests with pre-experiment data.

CUPED (Deng, Xu, Kohavi and Walker 2013) subtracts from the outcome its
linear projection on pre-experiment covariates, ``Y - (X - mean X)'theta``
with ``theta`` the coefficient of ``Y`` on ``X`` pooled over both arms, and
compares the adjusted outcome between arms. Because ``X`` is independent of
the assignment, the adjustment leaves the expected difference unchanged and
removes the share of the outcome variance that ``X`` explains (``1 - rho^2``
for one covariate correlated ``rho`` with the outcome).

``method='cupac'`` uses as the single covariate an out-of-fold machine
learning prediction of the outcome from the covariates (fitted ignoring the
treatment), which captures nonlinear signal. ``method='lin'`` is Lin's
interacted regression (:func:`sp.lm_lin`), which estimates a slope per arm
and is never less precise asymptotically.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from .._input_validation import check_columns
from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["cuped"]

# Verbatim from paper.bib (deng2013improving).
CausalResult._CITATIONS["cuped"] = (
    "@inproceedings{deng2013improving,\n"
    "  title={Improving the sensitivity of online controlled experiments by "
    "utilizing pre-experiment data},\n"
    "  author={Deng, Alex and Xu, Ya and Kohavi, Ron and Walker, Toby},\n"
    "  booktitle={Proceedings of the Sixth ACM International Conference on "
    "Web Search and Data Mining},\n"
    "  pages={123--132},\n"
    "  year={2013},\n"
    "  doi={10.1145/2433396.2433413}\n"
    "}"
)

_METHODS = ("cuped", "cupac", "lin")


def _cupac_feature(
    frame: pd.DataFrame,
    y: str,
    covariates: Sequence[str],
    learner: Any,
    n_folds: int,
    seed: Optional[int],
) -> np.ndarray:
    from sklearn.base import clone
    from sklearn.model_selection import KFold

    if learner is None:
        from sklearn.ensemble import GradientBoostingRegressor

        learner = GradientBoostingRegressor(random_state=seed)
    X = frame[list(covariates)].to_numpy(dtype=float)
    yv = frame[y].to_numpy(dtype=float)
    pred = np.empty(len(frame))
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train, test in folds.split(X):
        model = clone(learner).fit(X[train], yv[train])
        pred[test] = model.predict(X[test])
    return pred


def cuped(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: Sequence[str],
    *,
    method: str = "cuped",
    theta: Optional[Sequence[float]] = None,
    cluster: Optional[str] = None,
    learner: Any = None,
    n_folds: int = 5,
    seed: Optional[int] = None,
    alpha: float = 0.05,
) -> CausalResult:
    """Average treatment effect of an A/B test with CUPED variance reduction.

    Parameters
    ----------
    data : pandas.DataFrame
        One row per unit.
    y : str
        Outcome measured during the experiment.
    treat : str
        Treatment with exactly two values; the estimate is the larger value
        minus the smaller.
    covariates : sequence of str
        Numeric pre-experiment covariates (typically the same metric before
        the experiment). They must not be affected by the treatment.
    method : {'cuped', 'cupac', 'lin'}, default 'cuped'
        ``'cuped'``: pooled linear adjustment. ``'cupac'``: CUPED on an
        out-of-fold prediction of ``y`` from ``covariates``. ``'lin'``:
        :func:`sp.lm_lin` (interacted regression, HC2).
    theta : sequence of float, optional
        Fixed adjustment coefficients (one per covariate), e.g. estimated on
        pre-experiment data. Default: estimated from ``data`` pooled over
        both arms. Only with ``method='cuped'``.
    cluster : str, optional
        Unit of randomization when groups were assigned together; the
        variance of the adjusted difference is then the cluster-level one of
        :func:`sp.difference_in_means`.
    learner : sklearn regressor, optional
        Model for ``method='cupac'``; default gradient boosting.
    n_folds : int, default 5
        Cross-fitting folds for ``method='cupac'``.
    seed : int, optional
        Seed of the fold split and the default learner.
    alpha : float, default 0.05

    Returns
    -------
    CausalResult
        ``estimate``, ``se``, ``ci``, ``pvalue`` from
        :func:`sp.difference_in_means` applied to the adjusted outcome
        (Welch degrees of freedom), with ``model_info`` holding ``theta``,
        the unadjusted estimate and standard error, and
        ``variance_reduction`` (one minus the ratio of the adjusted to the
        unadjusted variance of the estimate).

    Notes
    -----
    The standard error treats ``theta`` as known, as Deng et al. do; its
    estimation changes the variance only at order ``1/n``. When the arms
    differ in size and the effect varies with the covariates, the pooled
    slope is not the variance-minimising one and ``method='lin'`` is the
    better choice.

    Examples
    --------
    >>> import numpy as np, pandas as pd, statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 2000
    >>> pre = rng.normal(size=n)
    >>> d = rng.permutation(np.repeat([0, 1], n // 2))
    >>> df = pd.DataFrame({"pre": pre, "d": d,
    ...                    "y": 0.2 * d + 0.9 * pre + rng.normal(scale=0.5, size=n)})
    >>> r = sp.cuped(df, "y", "d", ["pre"])
    >>> bool(r.model_info["variance_reduction"] > 0.7)
    True

    References
    ----------
    deng2013improving; lin2013agnostic
    """
    from ..inference.difference_in_means import difference_in_means

    key = str(method).lower()
    if key not in _METHODS:
        raise MethodIncompatibility(
            f"method must be one of {list(_METHODS)}; got {method!r}.",
            recovery_hint="Use method='cuped' (default), 'cupac' or 'lin'.",
        )
    covariates = [covariates] if isinstance(covariates, str) else list(covariates)
    if not covariates:
        raise MethodIncompatibility(
            "cuped: no covariates given.",
            recovery_hint="Without covariates use sp.difference_in_means.",
        )
    if key == "lin":
        if theta is not None:
            raise MethodIncompatibility(
                "theta= is not used by method='lin'.",
                recovery_hint="Drop theta= or use method='cuped'.",
            )
        return __import__("statspai.inference.lm_lin", fromlist=["lm_lin"]).lm_lin(
            data, y, treat, covariates, cluster=cluster, alpha=alpha
        )
    check_columns(
        data,
        {"y": y, "treat": treat, "covariates": covariates, "cluster": cluster},
        function="sp.cuped",
    )
    cols = list(dict.fromkeys([y, treat, *covariates] + ([cluster] if cluster else [])))
    frame = data[cols].dropna().reset_index(drop=True)
    non_numeric = [c for c in covariates if not pd.api.types.is_numeric_dtype(frame[c])]
    if non_numeric:
        raise MethodIncompatibility(
            f"cuped: covariates must be numeric; got {non_numeric}.",
            recovery_hint="Encode categorical covariates as indicators first.",
        )
    if len(frame) < len(covariates) + 3:
        raise DataInsufficient(
            f"cuped: {len(frame)} complete rows for {len(covariates)} covariates.",
            recovery_hint="Use fewer covariates.",
        )
    if key == "cupac":
        if theta is not None:
            raise MethodIncompatibility(
                "theta= is not used by method='cupac'.",
                recovery_hint="Drop theta= or use method='cuped'.",
            )
        if int(n_folds) < 2 or int(n_folds) > len(frame):
            raise MethodIncompatibility(
                f"n_folds must be between 2 and the number of rows; got {n_folds!r}.",
                recovery_hint="E.g. n_folds=5.",
            )
        Xmat = _cupac_feature(frame, y, covariates, learner, int(n_folds), seed)[
            :, None
        ]
        names = ["cupac_prediction"]
    else:
        Xmat = frame[covariates].to_numpy(dtype=float)
        names = covariates
    yv = frame[y].to_numpy(dtype=float)
    Xc = Xmat - Xmat.mean(axis=0)
    if theta is None:
        coef, _, rank, _ = np.linalg.lstsq(Xc, yv - yv.mean(), rcond=None)
        if rank < Xc.shape[1]:
            raise DataInsufficient(
                "cuped: the covariates are collinear.",
                recovery_hint="Drop covariates that are linear combinations of others.",
                diagnostics={"rank": int(rank), "n_covariates": Xc.shape[1]},
            )
        theta_v = np.asarray(coef, dtype=float)
    else:
        theta_v = np.asarray(list(np.atleast_1d(theta)), dtype=float)
        if theta_v.shape != (Xc.shape[1],) or not np.all(np.isfinite(theta_v)):
            raise MethodIncompatibility(
                f"theta must hold {Xc.shape[1]} finite numbers (one per covariate).",
                recovery_hint="Pass theta=[...] in the order of covariates.",
            )
    work = frame[[treat] + ([cluster] if cluster else [])].copy()
    work["_y_raw"] = yv
    work["_y_adj"] = yv - Xc @ theta_v
    raw = difference_in_means(work, "_y_raw", treat, cluster=cluster, alpha=alpha)
    adj = difference_in_means(work, "_y_adj", treat, cluster=cluster, alpha=alpha)
    reduction = 1.0 - (adj.se / raw.se) ** 2 if raw.se > 0 else float("nan")
    info = dict(adj.model_info)
    info.update(
        {
            "method": key,
            "theta": dict(zip(names, theta_v.tolist())),
            "theta_estimated": theta is None,
            "estimate_unadjusted": float(raw.estimate),
            "se_unadjusted": float(raw.se),
            "variance_reduction": float(reduction),
            "covariates": covariates,
        }
    )
    label = "CUPAC" if key == "cupac" else "CUPED"
    return CausalResult(
        method=f"{label} difference in means",
        estimand="ATE",
        estimate=float(adj.estimate),
        se=float(adj.se),
        pvalue=float(adj.pvalue),
        ci=tuple(adj.ci),
        alpha=alpha,
        n_obs=int(adj.n_obs),
        detail=adj.detail,
        model_info=info,
        _citation_key="cuped",
    )
