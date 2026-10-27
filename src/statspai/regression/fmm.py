"""
Finite mixture of regressions: ``sp.fmm`` (Stata ``fmm``).

The population is a mixture of ``k`` latent classes. Each class has its own
regression of ``y`` on the same covariates (linear with its own error
variance, logit or Poisson), and class membership follows a multinomial
logit in optional class covariates, Stata's ``lcprob()``:

    f(y_i | x_i, z_i) = sum_k pi_k(z_i) f_k(y_i | x_i' b_k).

The parameters are estimated by maximum likelihood [@mclachlan2000finite].
EM [@dempster1977maximum] from several starts supplies start values and
Newton-Raphson on the full likelihood takes the best of them to the
maximum, so the reported point is the MLE and its standard errors come from
the observed information matrix (Stata's default), or a robust / cluster
sandwich.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .._input_validation import missing_named_columns
from ..core._call_record import records_call
from ..core._collinear import drop_collinear
from ..core.utils import create_design_matrices
from ..exceptions import (
    AssumptionWarning,
    ColumnNotFound,
    ConvergenceFailure,
    ConvergenceWarning,
    DataInsufficient,
    IdentificationFailure,
    MethodIncompatibility,
    NumericalInstability,
)
from . import _fmm_core as core
from ._fmm_inference import _check_outcome, _cons, covariance, make_predictor, margins
from ._fmm_result import FMMResult
from ._optim_helpers import inverse_information, se_from_vcov

_VCE = ("oim", "robust", "cluster")
_ORDER = ("mean", "share", "none")


@records_call("fmm")
def fmm(
    formula: str,
    data: pd.DataFrame,
    k: int,
    *,
    family: str = "gaussian",
    class_covariates: Optional[Union[str, Sequence[str]]] = None,
    weights: Optional[str] = None,
    weights_as: Optional[str] = None,
    vce: Optional[str] = None,
    cluster: Optional[str] = None,
    n_starts: int = 10,
    seed: Optional[int] = None,
    max_iter: int = 200,
    tol: float = 1e-8,
    em_iter: int = 30,
    class_order: str = "mean",
    alpha: float = 0.05,
) -> FMMResult:
    """
    Finite mixture of linear, logit or Poisson regressions (Stata ``fmm``).

    Fits ``fmm k, lcprob(z): regress | logit | poisson y x`` by maximum
    likelihood. Every class has its own coefficients on the same
    right-hand side (and, for the Gaussian family, its own error
    variance); the class probabilities follow a multinomial logit in
    ``class_covariates`` with class 1 as the base.

    Parameters
    ----------
    formula : str
        Outcome equation, e.g. ``"y ~ x1 + x2"``. Factors, interactions and
        transformations are allowed; every class uses the same terms.
    data : pd.DataFrame
        Rows with a missing value in any variable used are dropped.
    k : int
        Number of latent classes (``k = 1`` is the one-class model, useful
        for a likelihood-ratio or BIC comparison).
    family : {"gaussian", "logit", "poisson"}, default "gaussian"
        Class density: Stata ``regress``, ``logit`` or ``poisson``. A logit
        outcome must be 0 / 1, a Poisson one non-negative.
    class_covariates : str or list of str, optional
        Columns that shift the class probabilities (Stata ``lcprob()``). A
        constant is always included.
    weights : str, optional
        Column of non-negative observation weights.
    weights_as : {"frequency", "sampling"}, optional
        How to read ``weights``. ``"frequency"`` (the default reading,
        Stata ``[fw=]`` / ``[iw=]``): the weighted log-likelihood, the
        information matrix and ``N`` (in BIC) scale with the weights.
        ``"sampling"`` (Stata ``[pw=]``): the same estimates with the robust
        sandwich, ``N`` the number of rows.
    vce : {"oim", "robust", "cluster"}, optional
        Covariance: the inverse observed information (default), the
        Huber-White sandwich times ``N/(N-1)``, or the cluster sandwich
        times ``G/(G-1)``, as Stata. ``None`` means ``"oim"``, ``"robust"``
        under ``weights_as="sampling"`` and ``"cluster"`` when ``cluster``
        is given.
    cluster : str, optional
        Cluster identifier for ``vce="cluster"``.
    n_starts : int, default 10
        Number of starts. The first is deterministic (a quantile split of
        the pooled residual, or of the outcome for logit / Poisson); the
        others split the sample along random directions drawn from
        ``seed``. Every start is taken to a maximum by EM and quasi-Newton,
        and Newton-Raphson polishes the one with the best log-likelihood.
    seed : int, optional
        Seed of the random starts. With well-separated classes the result
        does not depend on it.
    max_iter : int, default 200
        Iterations of the final Newton-Raphson from the best start (each
        start's quasi-Newton run is allowed ``20 * max_iter``).
    tol : float, default 1e-8
        Newton-Raphson stops after a step below ``tol * (1 + max|theta|)``.
    em_iter : int, default 30
        EM iterations per start; quasi-Newton (BFGS) then takes each start
        to its maximum.
    class_order : {"mean", "share", "none"}, default "mean"
        How classes are labelled, which fixes the solution among its ``k!``
        relabellings. ``"mean"``: ascending class-specific marginal mean of
        the outcome (``estat lcmean``); ``"share"``: descending marginal
        class probability (``estat lcprob``); ``"none"``: the labels of the
        best start. Class 1 is the base of the membership equation.
    alpha : float, default 0.05
        Level of the reported intervals.

    Returns
    -------
    FMMResult
        Coefficients by class (``coef_table(k)``, ``tidy()``), the class
        membership equation, marginal class probabilities and class means
        with delta-method standard errors (``class_probs``,
        ``class_means``), posterior and prior class probabilities per
        observation (``posterior``, ``prior``), ``loglik``, ``aic``,
        ``bic``, ``predict()`` and ``summary()``.

    Raises
    ------
    IdentificationFailure
        A class is (numerically) empty at the maximum, or the information
        matrix is singular there.
    NumericalInstability
        Every start ends in a degenerate solution (a Gaussian class whose
        variance collapses onto a few points, where the likelihood is
        unbounded).
    ConvergenceFailure
        Newton-Raphson did not reach Stata's ``nrtolerance`` criterion.

    Notes
    -----
    The mixture likelihood has local maxima. Each start runs ``em_iter``
    EM iterations and then Newton-Raphson; the start log-likelihoods are
    kept in ``start_logliks``. A ``ConvergenceWarning`` is issued when only
    one start reaches the best value, or when starts that reach it disagree
    on the estimates after relabelling (a flat or non-identified maximum).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 600
    >>> x = rng.normal(size=n)
    >>> c = rng.uniform(size=n) < 0.4
    >>> y = np.where(c, 6 - x, 1 + 2 * x) + rng.normal(size=n)
    >>> df = pd.DataFrame({"y": y, "x": x})
    >>> res = sp.fmm("y ~ x", df, k=2, seed=1)
    >>> res.class_probs["estimate"].round(1).tolist()
    [0.6, 0.4]
    >>> round(float(res.params["y:1.Class#x"]))
    2

    References
    ----------
    [@mclachlan2000finite] [@dempster1977maximum]
    """
    fam = str(family).lower()
    if fam not in core.FAMILIES:
        raise MethodIncompatibility(
            f"fmm: family must be one of {list(core.FAMILIES)}; got {family!r}.",
            recovery_hint="Use family='gaussian', 'logit' or 'poisson'.",
            diagnostics={"family": family},
        )
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or int(k) < 1:
        raise MethodIncompatibility(
            f"fmm: k must be a positive integer; got {k!r}.",
            recovery_hint="Pass the number of latent classes, e.g. k=2.",
        )
    K = int(k)
    order_key = str(class_order).lower()
    if order_key not in _ORDER:
        raise MethodIncompatibility(
            f"fmm: class_order must be one of {list(_ORDER)}; got {class_order!r}.",
            recovery_hint="Use class_order='mean', 'share' or 'none'.",
        )
    if int(n_starts) < 1:
        raise MethodIncompatibility(
            f"fmm: n_starts must be at least 1; got {n_starts!r}.",
            recovery_hint="Use n_starts=10 or more for a mixture likelihood.",
        )
    if not 0.0 < float(alpha) < 1.0:
        raise MethodIncompatibility(f"fmm: alpha must lie in (0, 1); got {alpha!r}.")

    from ..core._agent_summary import resolve_weights_as

    w_read = resolve_weights_as(
        weights_as, has_weights=weights is not None, native="frequency", entry="sp.fmm"
    )
    sampling = w_read == "sampling"
    vce_key = _resolve_vce(vce, cluster, sampling)

    zcols = (
        [class_covariates]
        if isinstance(class_covariates, str)
        else list(class_covariates or [])
    )
    extra = zcols + [c for c in (weights, cluster) if c is not None]
    missing = missing_named_columns(
        {"class_covariates": zcols, "weights": weights, "cluster": cluster},
        data.columns,
    )
    if missing:
        raise ColumnNotFound(
            f"fmm: column(s) {sorted(map(str, missing.values()))} not in data.",
            recovery_hint="Check the spelling against data.columns.",
            diagnostics={"missing": {k_: str(v) for k_, v in missing.items()}},
        )

    frame = data.reset_index(drop=True)
    keep = (
        frame[extra].notna().all(axis=1)
        if extra
        else pd.Series(True, index=frame.index)
    )
    y_df, X_df = create_design_matrices(formula, frame.loc[keep])
    rows = np.asarray(y_df.index)
    orig_index = data.index[rows]
    y = np.asarray(y_df, dtype=float).ravel()
    depvar = str(y_df.columns[0]) if hasattr(y_df, "columns") else "y"
    X, xnames, omitted, _ = drop_collinear(
        np.asarray(X_df, dtype=float),
        [str(c) for c in X_df.columns],
        "fmm",
        formula=formula,
        design_info=getattr(X_df, "design_info", None),
    )
    xnames = [_cons(c) for c in xnames]
    sub = frame.loc[rows]
    Z = np.column_stack(
        [sub[c].to_numpy(dtype=float) for c in zcols] + [np.ones(len(rows))]
    )
    znames = [str(c) for c in zcols] + ["_cons"]
    if zcols and np.linalg.matrix_rank(Z) < Z.shape[1]:
        raise IdentificationFailure(
            "fmm: the class covariates are collinear (with each other or the "
            "constant).",
            recovery_hint="Drop the redundant class covariate.",
            diagnostics={"class_covariates": zcols},
        )
    w = np.ones(len(rows)) if weights is None else sub[weights].to_numpy(dtype=float)
    if np.any(w < 0) or not np.all(np.isfinite(w)) or w.sum() <= 0:
        raise MethodIncompatibility(
            "fmm: weights must be finite and non-negative with a positive sum.",
            recovery_hint="Drop or fix the rows with negative weights.",
        )
    _check_outcome(y, fam)
    n_params = (K - 1) * Z.shape[1] + K * X.shape[1] + (K if fam == "gaussian" else 0)
    if len(y) <= n_params:
        raise DataInsufficient(
            f"fmm: {len(y)} observations for {n_params} parameters.",
            recovery_hint="Use fewer classes or fewer covariates.",
            diagnostics={"n_obs": int(len(y)), "n_params": int(n_params)},
        )
    model = core.MixtureModel(y, X, Z, K, fam, w)

    best, starts, failures = _multistart(
        model, int(n_starts), seed, int(em_iter), int(max_iter), float(tol)
    )
    theta, order = _canonical(model, best["theta"], order_key)
    theta, ll, H, it_final, info = core.newton(model, theta, 5, float(tol))
    core.require_converged(info["nr"])
    _check_degenerate(model, theta, xnames)
    _check_starts(starts, ll, n_starts)

    scores = model.scores(theta)
    clusters = sub[cluster].to_numpy() if cluster is not None else None
    V, n_clusters = covariance(H, scores, w, vce_key, sampling, clusters)
    spread = _check_agreement(
        model, starts, theta, ll, order_key, Ainv=inverse_information(H)
    )

    names = _param_names(K, depvar, xnames, znames, fam)
    J = np.eye(model.size)
    est = theta.copy()
    if model.n_scale:
        sl = slice(model.n_gamma + model.n_beta, model.size)
        var = np.exp(2.0 * theta[sl])
        est[sl] = var
        J[sl, sl] = np.diag(2.0 * var)
    Vr = J @ V @ J.T
    params = pd.Series(est, index=names)
    ses = pd.Series(se_from_vcov(Vr), index=names)

    n_ic = float(w.sum()) if (weights is not None and not sampling) else float(len(y))
    aic = -2.0 * ll + 2.0 * model.size
    bic = -2.0 * ll + np.log(n_ic) * model.size
    probs, means = margins(model, theta, V, float(alpha))

    with np.errstate(all="ignore"):
        _, log_pi, post, _, _ = model.joint(theta)
    labels = [f"{j}.Class" for j in range(1, K + 1)]
    fitted = model.class_mean(theta)

    tmp = FMMResult(
        formula=formula,
        depvar=depvar,
        family=fam,
        k=K,
        params=params,
        std_errors=ses,
        vcov=pd.DataFrame(Vr, index=names, columns=names),
        class_probs=probs,
        class_means=means,
        class_tables={},
        membership_table=pd.DataFrame(),
        posterior=pd.DataFrame(post, index=orig_index, columns=labels),
        prior=pd.DataFrame(np.exp(log_pi), index=orig_index, columns=labels),
        fitted=pd.DataFrame(fitted, index=orig_index, columns=labels),
        loglik=float(ll),
        aic=float(aic),
        bic=float(bic),
        n_obs=int(len(y)),
        n_ic=n_ic,
        df_model=int(model.size),
        vce=vce_key,
        converged=True,
        iterations=int(best["iterations"] + it_final),
        alpha=float(alpha),
        class_order=order_key,
        start_logliks=[s["ll"] for s in starts],
    )
    tidy = tmp.tidy()
    for j in range(1, K + 1):
        part = tidy[(tidy["class"] == j) & (tidy["equation"] != "class")]
        tmp.class_tables[j] = _table(part)
    mem = tidy[tidy["equation"] == "class"]
    tmp.membership_table = _table(mem, with_class=True) if len(mem) else pd.DataFrame()
    tmp.model_info.update(
        {
            "model_type": "FMM",
            "family": fam,
            "k": K,
            "formula": formula,
            "class_covariates": zcols,
            "weights": weights,
            "weights_as": w_read or ("frequency" if weights is not None else None),
            "vce": vce_key,
            "cluster": cluster,
            "n_clusters": n_clusters,
            "class_order": order_key,
            "relabelling": [int(o) + 1 for o in order],
            "omitted": omitted,
            "n_starts": int(n_starts),
            "seed": seed,
        }
    )
    tmp.diagnostics.update(
        {
            "start_logliks": [s["ll"] for s in starts],
            "failed_starts": failures,
            "n_best": int(sum(abs(s["ll"] - ll) <= _ll_tol(ll) for s in starts)),
            "nr_criterion": info["nr"],
            "best_starts_max_gap_se": spread,
            "min_class_share": float(probs["estimate"].min()),
        }
    )
    tmp._predictor = make_predictor(model, theta, formula, xnames, zcols, labels, X_df)
    return tmp


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _resolve_vce(vce: Optional[str], cluster: Optional[str], sampling: bool) -> str:
    if vce is None:
        key = "cluster" if cluster is not None else ("robust" if sampling else "oim")
    else:
        key = str(vce).lower()
    if key not in _VCE:
        raise MethodIncompatibility(
            f"fmm: vce must be one of {list(_VCE)}; got {vce!r}.",
            recovery_hint="Use vce='oim', 'robust' or 'cluster' with cluster=.",
        )
    if key == "cluster" and cluster is None:
        raise MethodIncompatibility(
            "fmm: vce='cluster' needs cluster=<column>.",
            recovery_hint="Pass the cluster identifier, e.g. cluster='firm'.",
        )
    if cluster is not None and key != "cluster":
        raise MethodIncompatibility(
            f"fmm: cluster= was given with vce={key!r}.",
            recovery_hint="Use vce='cluster' (or leave vce unset) with cluster=.",
        )
    if sampling and key == "oim":
        raise MethodIncompatibility(
            "fmm: sampling weights need a robust or cluster covariance; the "
            "inverse information is not a valid variance under them (Stata "
            "refuses vce(oim) with pweights).",
            recovery_hint="Leave vce unset, or use vce='robust' / 'cluster'.",
        )
    return key


def _ll_tol(ll: float) -> float:
    return 1e-6 * max(1.0, abs(ll))


def _multistart(
    model: core.MixtureModel,
    n_starts: int,
    seed: Optional[int],
    em_iter: int,
    max_iter: int,
    tol: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, str]]]:
    rngs = (
        np.random.default_rng(seed).spawn(max(n_starts - 1, 0)) if n_starts > 1 else []
    )
    starts: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    floor = 1e-10 * model.y_scale
    for s in range(n_starts):
        rng = None if s == 0 else rngs[s - 1]
        try:
            R = core.initial_posterior(model, rng)
            with np.errstate(all="ignore"):
                th, it_em = core.em(model, R, em_iter, 1e-6)
            th, ll = core.bfgs(model, th, 20 * max_iter)
        except (NumericalInstability, np.linalg.LinAlgError, FloatingPointError) as exc:
            failures.append(
                {"start": str(s), "error_type": type(exc).__name__, "message": str(exc)}
            )
            continue
        if model.n_scale and np.any(np.exp(2.0 * model.unpack(th)[2]) <= floor):
            failures.append(
                {
                    "start": str(s),
                    "error_type": "NumericalInstability",
                    "message": "a class variance collapsed to zero",
                }
            )
            continue
        if not np.isfinite(ll):
            failures.append(
                {
                    "start": str(s),
                    "error_type": "ConvergenceFailure",
                    "message": "non-finite log-likelihood",
                }
            )
            continue
        starts.append({"start": s, "theta": th, "ll": float(ll), "iterations": it_em})
    if not starts:
        kinds = sorted({f["error_type"] for f in failures})
        exc_cls = (
            NumericalInstability
            if "NumericalInstability" in kinds
            else ConvergenceFailure
        )
        raise exc_cls(
            f"fmm: none of the {n_starts} starts reached a non-degenerate maximum "
            f"({'; '.join(f['message'] for f in failures[:3])}).",
            recovery_hint="Use fewer classes, more starts (n_starts), or a "
            "different seed; a class that collapses onto a few points means "
            "the data do not support that many classes.",
            diagnostics={"failed_starts": failures},
        )
    best = max(starts, key=lambda d: d["ll"])
    try:
        th, ll, _, it_nr, info = core.newton(model, best["theta"], max_iter, tol)
    except NumericalInstability as exc:
        raise NumericalInstability(
            f"fmm: the best start is degenerate ({exc}).",
            recovery_hint="Use fewer classes or more starts.",
            diagnostics={"start_logliks": [d["ll"] for d in starts]},
        ) from exc
    core.require_converged(info["nr"])
    best = dict(best, theta=th, ll=float(ll), iterations=best["iterations"] + it_nr)
    return best, starts, failures


def _class_stat(model: core.MixtureModel, theta: np.ndarray, key: str) -> np.ndarray:
    wn = model.w / model.w.sum()
    if key == "share":
        return wn @ np.exp(np.real(model.class_log_prior(theta)))
    return wn @ np.real(model.class_mean(theta))


def _canonical(
    model: core.MixtureModel, theta: np.ndarray, key: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Relabel classes by the documented rule; returns (theta, order)."""
    if key == "none" or model.K == 1:
        return theta, np.arange(model.K)
    if key == "share":
        order = np.argsort(-_class_stat(model, theta, "share"), kind="stable")
    else:
        m = _class_stat(model, theta, "mean")
        s = _class_stat(model, theta, "share")
        order = np.lexsort((-s, m))
    return model.permute(theta, order), order


def _check_starts(starts: List[Dict[str, Any]], ll: float, n_starts: int) -> None:
    if n_starts < 2:
        return
    tol = _ll_tol(ll)
    n_best = sum(abs(s["ll"] - ll) <= tol for s in starts)
    if n_best < 2:
        warnings.warn(
            f"fmm: only one of {len(starts)} converged starts reached the best "
            f"log-likelihood {ll:.6f}; the maximum is not replicated. "
            f"Start log-likelihoods: {sorted(round(s['ll'], 4) for s in starts)}.",
            ConvergenceWarning,
            stacklevel=3,
        )


def _check_agreement(
    model: core.MixtureModel,
    starts: List[Dict[str, Any]],
    theta: np.ndarray,
    ll: float,
    order_key: str,
    Ainv: np.ndarray,
) -> float:
    """Largest gap, in OIM standard errors, between the final estimate and
    any start that reached the same log-likelihood (after relabelling).

    Starts that tie on the maximum but sit far apart mean a flat or
    non-identified maximum; a gap above half a standard error is reported.
    Each start's point is its quasi-Newton endpoint, within about 0.05 SE
    of its own maximum at the log-likelihood tolerance used here.
    """
    se = np.sqrt(np.maximum(np.diag(Ainv), 1e-300))
    gap = 0.0
    for s in starts:
        if abs(s["ll"] - ll) > _ll_tol(ll):
            continue
        th, _ = _canonical(
            model, s["theta"], order_key if order_key != "none" else "mean"
        )
        ref, _ = _canonical(model, theta, order_key if order_key != "none" else "mean")
        gap = max(gap, float(np.max(np.abs(th - ref) / se)))
    if gap > 0.5:
        warnings.warn(
            f"fmm: starts that reach the best log-likelihood {ll:.6f} disagree "
            f"on the estimates by up to {gap:.2g} standard errors after "
            "relabelling; the maximum is flat or not identified.",
            ConvergenceWarning,
            stacklevel=3,
        )
    return gap


def _check_degenerate(
    model: core.MixtureModel, theta: np.ndarray, xnames: List[str]
) -> None:
    with np.errstate(all="ignore"):
        post = np.real(model.joint(theta)[2])
    nk = model.w @ post
    n_class_params = model.p + (1 if model.n_scale else 0)
    small = np.where(nk < n_class_params)[0]
    if small.size:
        raise IdentificationFailure(
            f"fmm: class {int(small[0]) + 1} is empty at the maximum (posterior "
            f"count {float(nk[small[0]]):.3g} for {n_class_params} class "
            "parameters); its share has gone to zero.",
            recovery_hint="Use fewer classes (compare BIC across k).",
            diagnostics={"posterior_counts": nk.tolist()},
        )
    share = nk / model.w.sum()
    if share.min() < 0.01:
        warnings.warn(
            f"fmm: class {int(np.argmin(share)) + 1} holds {share.min():.2%} of the "
            "sample; its estimates rest on very few observations.",
            AssumptionWarning,
            stacklevel=3,
        )
    if model.family != "gaussian":
        _, B, _ = model.unpack(theta)
        if np.max(np.abs(B)) > 25.0:
            warnings.warn(
                "fmm: a class coefficient exceeds 25 in absolute value; the "
                "outcome is (quasi-)separated within a class and the "
                "estimate may not exist.",
                ConvergenceWarning,
                stacklevel=3,
            )


def _param_names(
    K: int, depvar: str, xnames: List[str], znames: List[str], fam: str
) -> List[str]:
    names = [f"{j}.Class:{z}" for j in range(2, K + 1) for z in znames]
    names += [f"{depvar}:{j}.Class#{x}" for j in range(1, K + 1) for x in xnames]
    if fam == "gaussian":
        names += [f"/var(e.{depvar})#{j}.Class" for j in range(1, K + 1)]
    return names


def _table(part: pd.DataFrame, with_class: bool = False) -> pd.DataFrame:
    cols = ["estimate", "std_error", "statistic", "p_value", "conf_low", "conf_high"]
    t = part.copy()
    if with_class:
        t.index = pd.MultiIndex.from_arrays(
            [t["class"], t["term"]], names=["class", "term"]
        )
    else:
        t.index = pd.Index(t["term"].tolist(), name="term")
    return t[cols]
