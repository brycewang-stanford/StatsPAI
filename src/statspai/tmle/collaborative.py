"""Collaborative targeted maximum likelihood estimation (C-TMLE).

A TMLE needs a propensity model only to remove the bias its outcome model
left behind. Adjusting the propensity for a covariate the outcome model
already handles, or one that does not affect the outcome at all (an
instrument), buys no bias reduction and costs variance, sometimes a lot
of it when it pushes propensities towards 0 or 1. C-TMLE therefore builds
the propensity model *in collaboration with* the outcome fit:

1. Start from the initial outcome regression and an intercept-only
   propensity, and target.
2. Greedily add to the propensity model the covariate whose targeted fit
   lowers the empirical loss of the outcome regression most. When no
   addition lowers it, the current targeted fit becomes the new initial
   fit and the search continues from there.
3. Choose how far along the sequence to go by cross-validation: the whole
   construction is repeated on each training fold and scored on the held
   out fold.

The result is the TMLE at the selected step, with the usual influence
function inference at that step's propensity model.

References
----------
[@vanderlaan2010collaborative], [@gruber2010application]
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.special import expit, logit

from .._aliases import accepts_aliases
from ..core._covariates import expands_categorical_covariates as _expands_categorical
from ..core._validate import treatment_as_float as _treatment_as_float
from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility
from ._targeting import cluster_se

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

_PENALTIES = ("none", "search", "variance", "variance+bias")


# ----------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------


def _logit_mle(X: np.ndarray, a: np.ndarray, max_iter: int = 60) -> np.ndarray:
    """Coefficients of an unpenalised logistic regression (with intercept).

    Newton steps on standardised columns; a step that does not raise the
    likelihood is halved. Under separation the coefficients run away and
    the iteration simply stops at ``max_iter`` with fitted probabilities
    near 0 and 1, which the caller truncates.
    """
    n = X.shape[0]
    if X.shape[1]:
        mu, sd = X.mean(axis=0), X.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        Z = np.column_stack([np.ones(n), (X - mu) / sd])
    else:
        mu = sd = np.zeros(0)
        Z = np.ones((n, 1))
    beta = np.zeros(Z.shape[1])
    beta[0] = logit(np.clip(a.mean(), 1e-6, 1 - 1e-6))

    def loglik(b: np.ndarray) -> float:
        eta = Z @ b
        return float(np.sum(a * eta - np.logaddexp(0.0, eta)))

    cur = loglik(beta)
    for _ in range(max_iter):
        p = expit(Z @ beta)
        w = np.maximum(p * (1 - p), 1e-10)
        grad = Z.T @ (a - p)
        H = (Z * w[:, None]).T @ Z + 1e-10 * np.eye(Z.shape[1])
        step = np.linalg.solve(H, grad)
        t = 1.0
        new = loglik(beta + step)
        while new < cur - 1e-12 and t > 1e-4:
            t *= 0.5
            new = loglik(beta + t * step)
        beta = beta + t * step
        cur = new
        # Converge on the step, not on the likelihood: a likelihood that
        # has stopped moving to ten digits still leaves the coefficients
        # short by about its square root.
        if float(np.max(np.abs(t * step))) < 1e-11:
            break
    # Back to the original column scale: (intercept, slopes).
    slopes = beta[1:] / sd if X.shape[1] else beta[1:]
    intercept = beta[0] - float(np.sum(slopes * mu)) if X.shape[1] else beta[0]
    return np.concatenate([[intercept], slopes])


def _propensity(
    W: np.ndarray, a: np.ndarray, cols: Sequence[int], W_eval: np.ndarray
) -> np.ndarray:
    cols = list(cols)
    beta = _logit_mle(W[:, cols], a)
    return np.asarray(expit(beta[0] + W_eval[:, cols] @ beta[1:]))


def _clever(a: np.ndarray, g: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clever covariate of the ATE at the observed arm, at A = 1 and at A = 0."""
    h1 = 1.0 / g
    h0 = -1.0 / (1.0 - g)
    return np.where(a == 1, h1, h0), h1, h0


def _epsilon(y: np.ndarray, off: np.ndarray, h: np.ndarray) -> float:
    eps = 0.0
    for _ in range(100):
        p = expit(off + eps * h)
        score = float(np.sum(h * (y - p)))
        info = float(np.sum(h * h * p * (1 - p)))
        if info < 1e-300:
            break
        step = score / info
        eps += step
        if abs(step) < 1e-12:
            break
    return eps


class _Fit:
    """An outcome fit on the unit scale: predictions at both arms."""

    __slots__ = ("q0", "q1")

    def __init__(self, q0: np.ndarray, q1: np.ndarray):
        self.q0, self.q1 = q0, q1

    def at(self, a: np.ndarray) -> np.ndarray:
        return np.where(a == 1, self.q1, self.q0)


def _rss(y: np.ndarray, a: np.ndarray, fit: _Fit) -> float:
    return float(np.sum((y - fit.at(a)) ** 2))


def _plug_in(fit: _Fit) -> float:
    return float(np.mean(fit.q1 - fit.q0))


def _influence(
    y: np.ndarray, a: np.ndarray, fit: _Fit, g: np.ndarray, psi: float
) -> np.ndarray:
    """Efficient influence function of the ATE at a targeted fit."""
    hA, _, _ = _clever(a, g)
    return hA * (y - fit.at(a)) + fit.q1 - fit.q0 - psi


class _Sequence:
    """The greedy sequence of targeted fits built on one sample.

    ``steps[k]`` records what is needed to replay step ``k`` on other rows:
    the propensity columns and, for every fluctuation applied since the
    initial fit, its propensity columns and its epsilon.
    """

    def __init__(
        self,
        y: np.ndarray,
        a: np.ndarray,
        W: np.ndarray,
        init: _Fit,
        bounds: Tuple[float, float],
        max_steps: Optional[int] = None,
        penalise: bool = True,
        order: Optional[Sequence[int]] = None,
    ):
        self.penalise = penalise
        self.order = None if order is None else list(order)
        self.y, self.a, self.W = y, a, W
        self.bounds = bounds
        self.init = init
        p = W.shape[1] if order is None else len(list(order))
        self.max_steps = p if max_steps is None else min(max_steps, p)
        # Each step: (columns of g, tuple of (columns, epsilon) fluctuations)
        self.steps: List[
            Tuple[Tuple[int, ...], Tuple[Tuple[Tuple[int, ...], float], ...]]
        ] = []
        self.losses: List[float] = []
        self.restarts = 0
        self._build()

    def _g(
        self, cols: Sequence[int], W_eval: Optional[np.ndarray] = None
    ) -> np.ndarray:
        g = _propensity(self.W, self.a, cols, self.W if W_eval is None else W_eval)
        return np.clip(g, *self.bounds)

    def _fluctuate(self, base: _Fit, g: np.ndarray) -> Tuple[_Fit, float]:
        hA, h1, h0 = _clever(self.a, g)
        eps = _epsilon(self.y, logit(base.at(self.a)), hA)
        return (
            _Fit(expit(logit(base.q0) + eps * h0), expit(logit(base.q1) + eps * h1)),
            eps,
        )

    def _criterion(self, fit: _Fit, g: np.ndarray) -> float:
        """Empirical loss of a targeted fit, with the variance penalty.

        The mean of the influence function is zero at a targeted fit, so
        the squared-bias term of the cross-validated criterion vanishes
        here and only the variance is added.
        """
        loss = _rss(self.y, self.a, fit)
        if self.penalise:
            ic = _influence(self.y, self.a, fit, g, _plug_in(fit))
            loss += float(np.var(ic))
        return loss

    def _build(self) -> None:
        base = self.init
        history: Tuple[Tuple[Tuple[int, ...], float], ...] = ()
        cols: Tuple[int, ...] = ()
        g0 = self._g(cols)
        fit, eps = self._fluctuate(base, g0)
        self.steps.append((cols, history + ((cols, eps),)))
        self.losses.append(self._criterion(fit, g0))
        remaining = (
            [j for j in range(self.W.shape[1])]
            if self.order is None
            else list(self.order)
        )
        while remaining and len(cols) < self.max_steps:
            best: Optional[Tuple[float, int, _Fit, float]] = None
            # Greedy: try every remaining covariate. Pre-ordered: only the
            # next one in the given order.
            pool = remaining if self.order is None else remaining[:1]
            for j in pool:
                g_j = self._g(cols + (j,))
                cand, e = self._fluctuate(base, g_j)
                loss = self._criterion(cand, g_j)
                if best is None or loss < best[0]:
                    best = (loss, j, cand, e)
            assert best is not None
            if best[0] >= self.losses[-1] - 1e-12 and base is not fit:
                # No covariate improves on the current targeted fit: make
                # that fit the starting point and search again. (The
                # fluctuation maximises a Bernoulli likelihood, so even
                # then the sum of squares is not guaranteed to fall; the
                # best candidate is accepted regardless and the
                # cross-validation decides whether the step is used.)
                base = fit
                history = self.steps[-1][1]
                self.restarts += 1
                continue
            loss, j, fit, eps = best
            cols = cols + (j,)
            remaining.remove(j)
            self.steps.append((cols, history + ((cols, eps),)))
            self.losses.append(loss)

    def replay(
        self, k: int, W_eval: np.ndarray, init_eval: _Fit
    ) -> Tuple[_Fit, np.ndarray]:
        """Step ``k``'s targeted fit and propensity on other rows."""
        cols, history = self.steps[k]
        q0, q1 = logit(init_eval.q0), logit(init_eval.q1)
        g = self._g(cols, W_eval)
        for h_cols, eps in history:
            g_h = self._g(h_cols, W_eval)
            _, h1, h0 = _clever(np.ones(len(g_h)), g_h)
            q0, q1 = q0 + eps * h0, q1 + eps * h1
        return _Fit(expit(q0), expit(q1)), g


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


@accepts_aliases(_strict=True, controls="covariates")
@_expands_categorical("covariates")
def ctmle(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    estimand: str = "ATE",
    outcome_library: "Optional[List[BaseEstimator]]" = None,
    Q: "Optional[np.ndarray]" = None,
    propensity_covariates: Optional[List[str]] = None,
    order: Optional[List[str]] = None,
    cv_folds: int = 5,
    fold_indices: "Optional[Any]" = None,
    penalty: str = "variance+bias",
    n_folds: int = 5,
    propensity_bounds: Tuple[float, float] = (0.025, 0.975),
    q_bound: float = 5e-4,
    alpha: float = 0.05,
    random_state: int = 42,
    cluster: Optional[str] = None,
    se_method: str = "influence",
    n_boot: int = 200,
) -> CausalResult:
    """Collaborative TMLE: the propensity model is chosen for the outcome fit.

    Use it when a full propensity model produces extreme weights. A
    covariate enters the propensity only if targeting with it improves the
    fit of the outcome regression, so instruments and covariates the
    outcome model already accounts for tend to stay out, while a
    confounder the outcome model missed is brought in.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome, binary or continuous (a continuous outcome is rescaled to
        the unit interval for the logistic fluctuation).
    treat : str
        Binary treatment (0/1).
    covariates : list of str
        Covariates of the outcome regression and, unless
        ``propensity_covariates`` says otherwise, the candidates for the
        propensity model.
    estimand : {'ATE'}, default 'ATE'
        Only the average treatment effect. For the effect on the treated
        use :func:`sp.tmle`; see Notes.
    outcome_library : list of sklearn estimators, optional
        Super Learner library for the initial outcome regression
        ``Q(A, W)``; the default library of :func:`sp.tmle` if omitted.
    Q : array of shape (n, 2), optional
        Initial outcome predictions ``[Q(0, W), Q(1, W)]`` on the outcome
        scale, replacing the Super Learner. They are then held fixed in
        the cross-validation, which is honest only if they were not fitted
        on these rows (or come from a parametric model with few
        parameters).
    propensity_covariates : list of str, optional
        Candidates for the propensity model. Default: ``covariates``.
    order : list of str, optional
        A fixed order in which the candidates are offered to the
        propensity model, replacing the greedy search (the pre-ordered,
        "scalable" C-TMLE of Ju et al.). Each step then costs one
        propensity fit instead of one per remaining candidate, so the
        whole sequence is linear in the number of candidates where the
        greedy search is quadratic. Candidates not listed are never
        offered. Put first what is most likely to confound.
    cv_folds : int, default 5
        Folds of the cross-validation that selects the step.
    fold_indices : array-like, optional
        One fold label per row of ``data``, replacing the random
        assignment (and ``cv_folds``).
    penalty : {'variance+bias', 'variance', 'search', 'none'}, default 'variance+bias'
        What is added to the residual sum of squares when candidates are
        compared. The variance of the estimator's influence function is
        what keeps a covariate that only makes the weights extreme out of
        the propensity model.

        - ``'search'``: the variance is added in the greedy search and the
          cross-validation compares residual sums of squares alone. This
          is what R ``ctmle::ctmleDiscrete`` does by default.
        - ``'variance'``: the variance is added in the cross-validation
          too.
        - ``'variance+bias'``: the cross-validation also adds ``n`` times
          the squared mean of the held-out influence function, the
          estimator's squared bias, as in Gruber and van der Laan (2010).
        - ``'none'``: residual sums of squares throughout.
    n_folds : int, default 5
        Internal folds of the Super Learner.
    propensity_bounds : tuple, default (0.025, 0.975)
        Truncation of every fitted propensity.
    q_bound : float, default 5e-4
        Truncation of the initial outcome predictions on the unit scale.
    alpha : float, default 0.05
    random_state : int, default 42
        Seeds the fold assignment and the Super Learner.
    cluster : str, optional
        Cluster column. Folds are formed from whole clusters and the
        standard error sums the influence function within clusters.
    se_method : {'influence', 'bootstrap'}, default 'influence'
        ``'bootstrap'`` reruns the whole procedure, selection included, on
        ``n_boot`` resamples of rows (of clusters with ``cluster=``); the
        standard error is the standard deviation of the estimates and the
        interval their percentile interval. Prefer it for inference: see
        Notes. Not with ``Q``.
    n_boot : int, default 200

    Returns
    -------
    CausalResult
        ``detail`` has one row per step of the sequence: the covariate
        added, the estimate at that step, the empirical and the
        cross-validated criterion with its three parts, and which step was
        selected. ``model_info`` holds ``selected_covariates`` (in the
        order they entered), ``candidate_order`` (the full greedy order),
        ``step``, ``n_restarts`` and the per-row ``influence_function``.
        After ``se_method='bootstrap'`` it also has
        ``selection_frequency``, the share of resamples in which each
        candidate entered the propensity model.

    Notes
    -----
    **What is selected and how.** The propensity models are logistic
    regressions on subsets of the candidates, nested along a greedy
    order. Step 0 is an intercept-only propensity. At each step the
    covariate that gives the lowest criterion is added; when none lowers
    it, the current targeted fit becomes the initial fit and the search
    goes on. The step is then chosen by cross-validation: the sequence is
    rebuilt on each training fold and its steps are scored on the held-out
    rows.

    **Inference.** The influence-function standard error is the standard
    deviation of the efficient influence function at the selected targeted
    fit and propensity model, over ``sqrt(n)``. It ignores the selection,
    and when the selected propensity model is not the true one (which is
    the point of the method) that function is not the estimator's
    influence function. In the simulations behind this function it was
    20% to 40% too small and its 95% interval covered 61% to 88% of the
    time; R ``ctmle`` behaves the same way. ``se_method='bootstrap'`` is
    the remedy.

    **When it helps and when it does not.** With an outcome model that is
    about right and a covariate that drives treatment but not the outcome,
    the root mean squared error was 27% below that of :func:`sp.tmle` with
    the full propensity model, and the bootstrap interval covered 94% of
    the time. With an outcome model that omits a strong confounder the
    confounder is brought into the propensity first, as it should be, but
    the error was no smaller than that of :func:`sp.tmle`. It is a tool
    for weak overlap, not a default.

    **Why only the ATE.** A collaborative ATT was implemented and tested.
    With an additive outcome model and effects that vary with a
    confounder, the intercept-only propensity already gives the lowest
    criterion, nothing rewards a step that would repair the missing
    heterogeneity, and the estimate was biased by 0.17 to 0.20 where
    :func:`sp.tmle` with the full propensity was unbiased. R ``ctmle``
    estimates the ATE only as well.

    **What it does not protect against.** A confounder that neither the
    outcome model nor the candidate list contains.

    Examples
    --------
    >>> import numpy as np, pandas as pd, statspai as sp
    >>> from sklearn.linear_model import LinearRegression
    >>> rng = np.random.default_rng(0)
    >>> n = 600
    >>> x1, z = rng.normal(size=n), rng.normal(size=n)
    >>> a = rng.binomial(1, 1 / (1 + np.exp(-(0.5 * x1 + 2.0 * z))))
    >>> yv = a + x1 + rng.normal(size=n)          # z is an instrument
    >>> df = pd.DataFrame({'y': yv, 'a': a, 'x1': x1, 'z': z})
    >>> res = sp.ctmle(df, y='y', treat='a', covariates=['x1', 'z'],
    ...                outcome_library=[LinearRegression()])
    >>> 'z' in res.model_info['selected_covariates']
    False

    References
    ----------
    [@vanderlaan2010collaborative], [@gruber2010application],
    [@ju2019scalable]
    """
    if not isinstance(estimand, str) or estimand.upper() != "ATE":
        raise MethodIncompatibility(
            f"ctmle: estimand={estimand!r} is not available; only 'ATE' is. "
            "For the effect on the treated the collaborative choice of the "
            "propensity model was tried and withdrawn: when the outcome "
            "model misses effect heterogeneity, no step of the sequence is "
            "rewarded for repairing it and the estimate keeps the bias.",
            recovery_hint="Use sp.tmle(estimand='ATT', se_method='bootstrap').",
            alternative_functions=["sp.tmle"],
        )
    estimand = "ATE"
    if not (0 < q_bound < 0.5):
        raise MethodIncompatibility("ctmle: q_bound must lie in (0, 0.5)")
    if penalty not in _PENALTIES:
        raise MethodIncompatibility(
            f"ctmle: penalty must be one of {_PENALTIES}, got {penalty!r}"
        )
    if se_method not in ("influence", "bootstrap"):
        raise MethodIncompatibility(
            f"ctmle: se_method must be 'influence' or 'bootstrap', got {se_method!r}"
        )
    if se_method == "bootstrap":
        if Q is not None:
            raise MethodIncompatibility(
                "ctmle: se_method='bootstrap' refits the outcome regression on "
                "every resample; with Q supplied its sampling error would be "
                "left out.",
                recovery_hint="Drop Q, or use se_method='influence'.",
            )
        if int(n_boot) < 20:
            raise MethodIncompatibility(
                "ctmle: n_boot must be at least 20 for a standard error."
            )
    if int(cv_folds) < 2:
        raise MethodIncompatibility("ctmle: cv_folds must be at least 2")
    g_names = list(
        covariates if propensity_covariates is None else propensity_covariates
    )
    order_idx: Optional[List[int]] = None
    if order is not None:
        unknown = [c for c in order if c not in g_names]
        if unknown or len(set(order)) != len(order) or not len(order):
            raise MethodIncompatibility(
                "ctmle: order must list distinct propensity candidates.",
                diagnostics={"unknown": unknown, "candidates": g_names},
            )
        order_idx = [g_names.index(c) for c in order]
    design = [c for c in (cluster,) if c is not None]
    cols = list(dict.fromkeys([y, treat] + list(covariates) + g_names + design))
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"ctmle: columns not found in data: {missing}",
            diagnostics={"missing": missing},
        )
    keep = data[cols].notna().all(axis=1).to_numpy()
    clean = data.loc[keep, cols]
    Y = clean[y].to_numpy(dtype=float)
    A = _treatment_as_float(clean[treat], function="ctmle")
    if set(np.unique(A)) != {0.0, 1.0}:
        raise MethodIncompatibility(
            "ctmle: treatment must be binary (0/1) with both arms"
        )
    Wq = clean[list(covariates)].to_numpy(dtype=float)
    Wg = clean[g_names].to_numpy(dtype=float)
    n = len(Y)
    if n < 10 * cv_folds:
        raise DataInsufficient(
            f"ctmle: {n} complete rows are too few for {cv_folds} folds."
        )

    binary = set(np.unique(Y)) <= {0.0, 1.0}
    y_min, y_rng = (0.0, 1.0) if binary else (float(Y.min()), float(Y.max() - Y.min()))
    if y_rng <= 0:
        raise MethodIncompatibility("ctmle: the outcome is constant")
    Ys = (Y - y_min) / y_rng

    def unit(v: np.ndarray) -> np.ndarray:
        return np.clip((v - y_min) / y_rng, q_bound, 1 - q_bound)

    def initial_fit(train: np.ndarray, test: np.ndarray) -> _Fit:
        """Outcome predictions at both arms for ``test`` rows, unit scale."""
        if Q is not None:
            return _Fit(unit(Q_arr[test, 0]), unit(Q_arr[test, 1]))
        from .super_learner import SuperLearner

        sl = SuperLearner(
            library=outcome_library,
            n_folds=n_folds,
            task="classification" if binary else "regression",
            random_state=random_state,
        )
        sl.fit(np.column_stack([A[train], Wq[train]]), Ys[train])
        m = int(np.sum(test)) if test.dtype == bool else len(test)
        q1 = sl.predict(np.column_stack([np.ones(m), Wq[test]]))
        q0 = sl.predict(np.column_stack([np.zeros(m), Wq[test]]))
        return _Fit(
            np.clip(q0, q_bound, 1 - q_bound), np.clip(q1, q_bound, 1 - q_bound)
        )

    if Q is not None:
        Q_arr = np.asarray(Q, dtype=float)
        if Q_arr.shape != (len(data), 2) and Q_arr.shape != (n, 2):
            raise MethodIncompatibility(
                f"ctmle: Q must have shape (n, 2); got {Q_arr.shape}."
            )
        if Q_arr.shape[0] == len(data) and len(data) != n:
            Q_arr = Q_arr[keep]

    bounds = (float(propensity_bounds[0]), float(propensity_bounds[1]))
    everyone = np.ones(n, dtype=bool)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full_init = initial_fit(everyone, everyone)
        penalise = penalty != "none"
        seq = _Sequence(
            Ys, A, Wg, full_init, bounds, penalise=penalise, order=order_idx
        )

        # Cross-validation of the step: rebuild the sequence on each
        # training fold and score every step on the held-out rows.
        rng = np.random.default_rng(random_state)
        if fold_indices is not None:
            codes = None if cluster is None else pd.factorize(clean[cluster])[0]
            fold = pd.factorize(np.asarray(fold_indices)[keep], sort=True)[0]
            cv_folds = int(fold.max()) + 1
        elif cluster is not None:
            codes = pd.factorize(clean[cluster])[0]
            n_groups = int(codes.max()) + 1
            if n_groups < cv_folds:
                raise DataInsufficient(
                    f"ctmle: {n_groups} clusters are fewer than cv_folds={cv_folds}."
                )
            fold = (rng.permutation(n_groups) % cv_folds)[codes]
        else:
            codes = None
            fold = rng.permutation(n) % cv_folds
        K = len(seq.steps)
        cv_loss = np.zeros(K)
        cv_ic: List[List[np.ndarray]] = [[] for _ in range(K)]
        cv_count = np.zeros(K)
        for v in range(cv_folds):
            tr, te = fold != v, fold == v
            if len(np.unique(A[tr])) < 2:
                continue
            init_tr = initial_fit(tr, tr)
            init_te = initial_fit(tr, te)
            seq_v = _Sequence(
                Ys[tr],
                A[tr],
                Wg[tr],
                init_tr,
                bounds,
                max_steps=K - 1,
                penalise=penalise,
                order=order_idx,
            )
            for k in range(K):
                fit_te, g_te = seq_v.replay(k, Wg[te], init_te)
                cv_loss[k] += _rss(Ys[te], A[te], fit_te)
                cv_count[k] += 1
                # Influence function of the step-k estimator on the held-out
                # rows, centred at the training-sample plug-in.
                fit_tr, _ = seq_v.replay(k, Wg[tr], init_tr)
                cv_ic[k].append(
                    _influence(Ys[te], A[te], fit_te, g_te, _plug_in(fit_tr))
                )
    if not np.all(cv_count > 0):
        raise DataInsufficient(
            "ctmle: no cross-validation fold had both treatment arms in its "
            "training rows."
        )
    # The cross-validated criterion: residual sum of squares, plus the
    # variance of the influence function and n times its squared mean. The
    # last two are the variance and the squared bias of the estimator at
    # that step, on the scale of a sum over n rows of squared errors of
    # size 1/n, which is what makes them commensurate with the first.
    cv_var = np.array([float(np.var(np.concatenate(v))) for v in cv_ic])
    cv_bias = np.array([float(np.mean(np.concatenate(v))) for v in cv_ic])
    criterion = cv_loss.copy()
    # 'search' penalises the greedy search only (R ctmle's default).
    if penalty in ("variance", "variance+bias"):
        criterion = criterion + cv_var
    if penalty == "variance+bias":
        criterion = criterion + n * cv_bias**2
    k_star = int(np.argmin(criterion))
    fit, g = seq.replay(k_star, Wg, full_init)

    # Plug-in and influence function on the unit scale, then rescaled.
    psi_u = _plug_in(fit)
    eif = _influence(Ys, A, fit, g, psi_u)
    psi = psi_u * y_rng
    eif = eif * y_rng
    se = cluster_se(codes, n)(eif)
    z = float(sp_stats.norm.ppf(1 - alpha / 2))
    pvalue = float(2 * sp_stats.norm.sf(abs(psi / se))) if se > 0 else np.nan

    # Every step of the sequence, so the choice can be inspected.
    rows = []
    for k in range(K):
        fit_k, _ = seq.replay(k, Wg, full_init)
        rows.append(
            {
                "step": k,
                "added": g_names[seq.steps[k][0][-1]] if k else "(intercept)",
                "estimate": _plug_in(fit_k) * y_rng,
                "empirical_criterion": seq.losses[k],
                "cv_rss": float(cv_loss[k]),
                "cv_variance": float(cv_var[k]),
                "cv_bias": float(cv_bias[k]),
                "cv_criterion": float(criterion[k]),
                "selected": k == k_star,
            }
        )
    candidates = pd.DataFrame(rows)
    entered = [g_names[j] for j in seq.steps[-1][0]]
    selected = [g_names[j] for j in seq.steps[k_star][0]]
    g1 = g
    model_info: Dict[str, Any] = {
        "estimand": estimand,
        "selected_covariates": selected,
        "candidate_order": entered,
        "search": "greedy" if order is None else "pre-ordered",
        "step": k_star,
        "n_steps": K,
        "penalty": penalty,
        "cv_criterion": [float(v) for v in criterion],
        "cv_loss": [float(v) for v in cv_loss],
        "cv_variance": [float(v) for v in cv_var],
        "cv_bias": [float(v) for v in cv_bias],
        "empirical_loss": [float(v) for v in seq.losses],
        "n_restarts": int(seq.restarts),
        "cv_folds": int(cv_folds),
        "propensity_bounds": tuple(propensity_bounds),
        "propensity_min": float(g1.min()),
        "propensity_max": float(g1.max()),
        "outcome_type": "binary" if binary else "continuous",
        "nuisance_source": {"Q": "supplied" if Q is not None else "super_learner"},
        "se_method": (
            "efficient_influence_function"
            if codes is None
            else "cluster_efficient_influence_function"
        ),
        "cluster": cluster,
        "n_treated": int(np.sum(A == 1)),
        "n_control": int(np.sum(A == 0)),
        "influence_function": np.asarray(eif, dtype=float),
        "influence_function_scale": "estimate",
    }
    result = CausalResult(
        method="C-TMLE (van der Laan & Gruber 2010)",
        estimand=estimand,
        estimate=float(psi),
        se=float(se),
        pvalue=pvalue,
        ci=(float(psi - z * se), float(psi + z * se)),
        alpha=alpha,
        n_obs=n,
        detail=candidates,
        model_info=model_info,
        _citation_key="tmle",
    )
    if se_method == "bootstrap":
        _bootstrap(
            result,
            clean,
            n_boot=int(n_boot),
            seed=random_state,
            cluster=cluster,
            alpha=alpha,
            kwargs=dict(
                y=y,
                treat=treat,
                covariates=list(covariates),
                estimand=estimand,
                outcome_library=outcome_library,
                propensity_covariates=propensity_covariates,
                order=order,
                cv_folds=cv_folds,
                penalty=penalty,
                n_folds=n_folds,
                propensity_bounds=propensity_bounds,
                q_bound=q_bound,
                alpha=alpha,
                cluster=cluster,
            ),
        )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            result,
            function="sp.ctmle",
            params={
                "y": y,
                "treat": treat,
                "covariates": list(covariates),
                "estimand": estimand,
                "propensity_covariates": propensity_covariates,
                "order": order,
                "cv_folds": cv_folds,
                "penalty": penalty,
                "se_method": se_method,
                "n_boot": n_boot if se_method == "bootstrap" else None,
                "n_folds": n_folds,
                "propensity_bounds": list(propensity_bounds),
                "q_bound": q_bound,
                "alpha": alpha,
                "random_state": random_state,
                "cluster": cluster,
            },
            data=data,
            overwrite=False,
        )
    except (ImportError, AttributeError, TypeError, ValueError):  # pragma: no cover
        pass
    return result


def _bootstrap(
    result: CausalResult,
    clean: pd.DataFrame,
    n_boot: int,
    seed: int,
    cluster: Optional[str],
    alpha: float,
    kwargs: Dict[str, Any],
) -> None:
    """Replace the inference of ``result`` by bootstrap inference, in place.

    Every resample reruns the whole procedure: outcome regression, greedy
    sequence, cross-validated choice of the step, targeting. The spread of
    the estimates therefore includes the selection, which the
    influence-function standard error leaves out.
    """
    rng = np.random.default_rng(seed)
    n = len(clean)
    frame = clean.reset_index(drop=True)
    members: List[np.ndarray] = []
    if cluster is not None:
        codes = pd.factorize(frame[cluster])[0]
        members = [np.flatnonzero(codes == g) for g in range(codes.max() + 1)]
    draws: List[float] = []
    picked: Dict[str, int] = {}
    failed = 0
    for b in range(n_boot):
        if members:
            pick = rng.integers(0, len(members), len(members))
            idx = np.concatenate([members[g] for g in pick])
            sample = frame.iloc[idx].reset_index(drop=True)
            sample[cluster] = np.repeat(
                np.arange(len(pick)), [len(members[g]) for g in pick]
            )
        else:
            sample = frame.iloc[rng.integers(0, n, n)].reset_index(drop=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rep = ctmle(sample, random_state=seed + 1 + b, **kwargs)
        except (
            ValueError,
            np.linalg.LinAlgError,
            MethodIncompatibility,
            DataInsufficient,
        ):
            failed += 1
            continue
        draws.append(float(rep.estimate))
        for name in rep.model_info["selected_covariates"]:
            picked[name] = picked.get(name, 0) + 1
    if failed > 0.1 * n_boot or len(draws) < 20:
        raise DataInsufficient(
            f"ctmle: {failed} of {n_boot} bootstrap resamples could not be fitted.",
            recovery_hint="Use se_method='influence' or check both arms are "
            "well represented.",
            diagnostics={"n_boot": n_boot, "n_failed": failed},
        )
    boot = np.asarray(draws)
    info = result.model_info
    info["se_influence"] = float(result.se)
    info["ci_influence"] = tuple(float(v) for v in result.ci)
    result.se = float(np.std(boot, ddof=1))
    result.ci = (
        float(np.quantile(boot, alpha / 2)),
        float(np.quantile(boot, 1 - alpha / 2)),
    )
    result.pvalue = (
        float(2 * sp_stats.norm.sf(abs(result.estimate / result.se)))
        if result.se > 0
        else np.nan
    )
    info["se_method"] = "bootstrap" if cluster is None else "cluster_bootstrap"
    info["n_boot"] = n_boot
    info["n_boot_failed"] = failed
    info["bootstrap_estimates"] = boot
    # How often each candidate entered the propensity model across resamples.
    info["selection_frequency"] = {k: v / len(boot) for k, v in sorted(picked.items())}
