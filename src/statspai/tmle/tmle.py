"""
Targeted Maximum Likelihood Estimation (TMLE) for causal inference.

TMLE is a two-step semiparametric estimator:

1. **Initial estimate**: Fit outcome model Q(Y | A, W) and propensity
   model g(A | W) using flexible ML (Super Learner).

2. **Targeting step**: Update the initial outcome estimate along the
   least-favourable submodel using the clever covariate:
       H(A, W) = A/g(W) - (1-A)/(1-g(W))
   Fit epsilon by regressing Y on H with offset logit(Q_bar).

3. **Plug-in estimate**: Compute ATE as mean(Q*(1,W) - Q*(0,W))
   using the targeted (updated) outcome predictions.

The resulting estimator is:
- Doubly robust: consistent if either Q or g is correct
- Semiparametrically efficient: achieves the efficiency bound
- Regular and asymptotically linear with known influence function

References
----------
van der Laan, M. J. & Rose, S. (2011).
Targeted Learning. Springer Series in Statistics. [@vanderlaan2011targeted]

van der Laan, M. J. & Rubin, D. (2006).
Targeted Maximum Likelihood Learning.
International Journal of Biostatistics, 2(1). [@vanderlaan2006targeted]
"""

from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.special import expit, logit

# sklearn is imported lazily inside the functions that need it so that
# ``import statspai`` doesn't pull ~245 sklearn submodules through this
# file when the user never touches tmle. ``BaseEstimator`` only appears
# in type annotations here and is gated behind ``TYPE_CHECKING``.
if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

from .._aliases import accepts_aliases
from ..core.results import CausalResult
from ..exceptions import MethodIncompatibility
from .super_learner import SuperLearner

# ======================================================================
# Public API
# ======================================================================


@accepts_aliases(_strict=True, controls="covariates")
def tmle(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    outcome_library: "Optional[List[BaseEstimator]]" = None,
    propensity_library: "Optional[List[BaseEstimator]]" = None,
    n_folds: int = 5,
    estimand: str = "ATE",
    alpha: float = 0.05,
    propensity_bounds: Tuple[float, float] = (0.025, 0.975),
    random_state: int = 42,
    Q: "Optional[np.ndarray]" = None,
    g1W: "Optional[np.ndarray]" = None,
    fluctuation: str = "single",
    q_bound: float = 1e-5,
    fold_indices: "Optional[Any]" = None,
    weights: Optional[str] = None,
    cluster: Optional[str] = None,
) -> CausalResult:
    """
    Estimate causal effects using TMLE with Super Learner.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    y : str
        Outcome variable (binary or continuous).
    treat : str
        Binary treatment variable (0/1).
    covariates : list of str
        Covariate names.
    outcome_library : list of sklearn estimators, optional
        Candidate learners for the outcome model Q(Y|A,W).
        If None, uses a default diverse library.
    propensity_library : list of sklearn estimators, optional
        Candidate learners for propensity model g(A|W).
        If None, uses a default diverse library.
    n_folds : int, default 5
        Cross-validation folds for Super Learner.
    estimand : str, default 'ATE'
        'ATE' or 'ATT'.
    alpha : float, default 0.05
        Significance level.
    propensity_bounds : tuple, default (0.025, 0.975)
        Bounds for propensity score truncation.
    random_state : int, default 42
    q_bound : float, default 1e-5
        The initial outcome predictions, on the [0, 1] scale the logistic
        fluctuation works on (continuous outcomes are min-max rescaled),
        are truncated to ``[q_bound, 1 - q_bound]`` before targeting.
        ``tmle::tmle`` truncates at ``1 - alpha = 5e-4`` (its default
        ``alpha = 0.9995``); pass ``q_bound=5e-4`` to reproduce it. The
        truncation only binds when an initial prediction falls outside
        the observed outcome range, which is common for linear ``Q`` fits
        near the extremes.
    fold_indices : array-like of int, optional
        Opt-in cross-validated TMLE (CV-TMLE). One label per row of
        ``data``, the labels being ``0, ..., K-1`` with ``K >= 2`` (the
        convention ``sp.dml(fold_indices=...)`` uses); rows dropped for
        missing values are dropped from the vector too. When given, the
        initial outcome model ``Q(Y | A, W)`` and the propensity
        ``g(A | W)`` are Super Learners fitted on the rows *outside* each
        fold and predicted on the rows inside it, so every unit's initial
        predictions are out of fold. The fluctuation is then fitted once on
        the pooled out-of-fold predictions, the plug-in is the mean of the
        targeted out-of-fold counterfactual predictions, and the standard
        error is the sample standard deviation of the efficient influence
        function at those targeted fits over ``sqrt(n)``. ``n_folds`` keeps
        its meaning (the Super Learner's internal CV folds, now run inside
        each training set). Cannot be combined with ``Q`` / ``g1W``.
        ``result.model_info["cross_fitted"]`` is ``True`` and
        ``model_info["n_cv_folds"]`` is ``K``; the per-fold ensemble
        weights are in ``model_info["sl_outcome_weights_by_fold"]`` /
        ``["sl_propensity_weights_by_fold"]`` (entry ``k`` is the Super
        Learner fitted on the rows with ``fold != k``), and the full-sample
        ``sl_*_weights`` keys are ``None``. A continuous outcome is min-max
        rescaled to ``[0, 1]`` with bounds taken from the *full* retained
        sample, as on the default path, before the per-fold fits; this is
        a fixed affine map, not a fitted nuisance. A Super Learner failure
        inside a training complement is re-raised naming the fold.
    weights : str, optional
        Observation (sampling) weights column, R ``tmle``'s ``obsWeights``:
        normalised to mean one, passed to every Super Learner fit, used in
        the fluctuation regression, and multiplying the plug-in average and
        the influence function. ``estimand='ATE'`` only.
    cluster : str, optional
        Cluster column. The standard error sums the influence function
        within clusters: ``G/(G-1) sum_g (S_g - mean S)^2 / n^2``, which
        is R ``tmle``'s ``id=`` variance when clusters are of equal size
        (``tmle`` averages the influence function within ``id`` instead, a
        repeated-measures convention that differs with unequal sizes).

    Returns
    -------
    CausalResult

    Notes
    -----
    **How the nuisances are fitted.** By default (``fold_indices=None``)
    this is *not* a cross-validated TMLE: the Super Learner for ``Q`` is
    fitted on the full sample with the treatment as a covariate, the
    Super Learner for ``g`` is fitted on the full sample, and both are
    evaluated on the same rows they were trained on. ``n_folds`` only
    controls the internal cross-validation the Super Learner uses to pick
    its ensemble weights; the final learners are refitted on all rows.
    ``model_info["cross_fitted"]`` is ``False`` on this path. Pass
    ``fold_indices`` for out-of-fold initial fits.

    **ATT.** With ``estimand='ATT'`` only the outcome model is targeted;
    ``g`` is held at its (truncated) initial fit and is not updated. The
    clever covariate is ``H(A, W) = A - (1 - A) g / (1 - g)`` (so
    ``H(0, W) = -g / (1 - g)``), and the reported estimate uses the
    estimating-equation form
    ``mean(A (Y - Q*(0,W)) / p - (1 - A) g (Y - Q*(0,W)) / ((1 - g) p))``
    with ``p`` the treated share of the retained sample, not the plug-in
    ``mean over treated of Q*(1,W) - Q*(0,W)``. The SE is the standard
    deviation of the corresponding influence function (the summand minus
    ``psi A / p``) over ``sqrt(n)``.

    References
    ----------
    van der Laan, M. J. and Rose, S. (2011). *Targeted Learning: Causal
    Inference for Observational and Experimental Data*. Springer.
    [@vanderlaan2011targeted]

    Examples
    --------
    >>> import statspai as sp, numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> X = rng.normal(size=(n, 3))
    >>> treatment = rng.binomial(1, 1 / (1 + np.exp(-X[:, 0])))
    >>> outcome = 2.0 * treatment + X @ np.array([1.0, -0.5, 0.3]) + rng.normal(size=n)
    >>> df = pd.DataFrame(X, columns=["x1", "x2", "x3"])
    >>> df["outcome"], df["treatment"] = outcome, treatment
    >>> result = sp.tmle(df, y="outcome", treat="treatment",
    ...                  covariates=["x1", "x2", "x3"])
    >>> print(result.summary())
    """
    est = TMLE(
        data=data,
        y=y,
        treat=treat,
        covariates=covariates,
        outcome_library=outcome_library,
        propensity_library=propensity_library,
        n_folds=n_folds,
        estimand=estimand,
        alpha=alpha,
        propensity_bounds=propensity_bounds,
        random_state=random_state,
        Q=Q,
        g1W=g1W,
        fluctuation=fluctuation,
        q_bound=q_bound,
        fold_indices=fold_indices,
        weights=weights,
        cluster=cluster,
    )
    _result = est.fit()
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.tmle",
            params={
                "y": y,
                "treat": treat,
                "covariates": list(covariates),
                "n_folds": n_folds,
                "estimand": estimand,
                "alpha": alpha,
                "propensity_bounds": list(propensity_bounds),
                "random_state": random_state,
                "q_bound": q_bound,
                "cross_fitted": fold_indices is not None,
                "weights": weights,
                "cluster": cluster,
                "outcome_library": (
                    [type(m).__name__ for m in outcome_library]
                    if outcome_library
                    else None
                ),
                "propensity_library": (
                    [type(m).__name__ for m in propensity_library]
                    if propensity_library
                    else None
                ),
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


def _fit_in_fold(
    sl: SuperLearner, X: np.ndarray, y: np.ndarray, k: int, what: str
) -> None:
    """Fit a CV-TMLE nuisance on fold ``k``'s training complement.

    A failure is re-raised, same class where possible, with the fold named
    so the caller can see which partition cell broke the fit.
    """
    try:
        sl.fit(X, y)
    except Exception as exc:
        from ..exceptions import StatsPAIError

        msg = (
            f"sp.tmle (CV-TMLE): fitting the {what} Super Learner on the "
            f"training complement of fold {k} (rows with fold != {k}) failed: "
            f"{getattr(exc, 'message', None) or exc}"
        )
        if isinstance(exc, StatsPAIError):
            raise type(exc)(
                msg,
                recovery_hint=exc.recovery_hint,
                diagnostics={**exc.diagnostics, "cv_fold": k},
                alternative_functions=exc.alternative_functions,
            ) from exc
        try:
            new_exc = type(exc)(msg)
        except Exception:
            new_exc = RuntimeError(msg)
        raise new_exc from exc


# ======================================================================
# TMLE Estimator
# ======================================================================


class TMLE:
    """
    Targeted Maximum Likelihood Estimation.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
    treat : str
    covariates : list of str
    outcome_library : list of sklearn estimators, optional
    propensity_library : list of sklearn estimators, optional
    n_folds : int
    estimand : str
    alpha : float
    propensity_bounds : tuple
    random_state : int
    fold_indices : array-like of int, optional
        Opt-in cross-validated TMLE partition (labels ``0..K-1``, one per
        row of ``data``); see :func:`statspai.tmle`. ``None`` (default)
        fits Q and g on the full sample, i.e. not cross-fitted.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> x1 = rng.normal(size=n)
    >>> x2 = rng.normal(size=n)
    >>> ps = 1 / (1 + np.exp(-(0.5 * x1 - 0.3 * x2)))
    >>> treat = rng.binomial(1, ps)
    >>> y = 1.0 * treat + 0.8 * x1 - 0.5 * x2 + rng.normal(size=n)
    >>> df = pd.DataFrame({'y': y, 'treat': treat, 'x1': x1, 'x2': x2})
    >>> est = sp.TMLE(df, y='y', treat='treat', covariates=['x1', 'x2'],
    ...               n_folds=2, random_state=0)
    >>> res = est.fit()
    >>> bool(hasattr(res, 'estimate'))
    True
    >>> bool(res.se > 0)
    True
    """

    def __init__(
        self,
        data: pd.DataFrame,
        y: str,
        treat: str,
        covariates: List[str],
        outcome_library: "Optional[List[BaseEstimator]]" = None,
        propensity_library: "Optional[List[BaseEstimator]]" = None,
        n_folds: int = 5,
        estimand: str = "ATE",
        alpha: float = 0.05,
        propensity_bounds: Tuple[float, float] = (0.025, 0.975),
        random_state: int = 42,
        Q: "Optional[np.ndarray]" = None,
        g1W: "Optional[np.ndarray]" = None,
        fluctuation: str = "single",
        q_bound: float = 1e-5,
        fold_indices: "Optional[Any]" = None,
        weights: Optional[str] = None,
        cluster: Optional[str] = None,
    ):
        if not (0 < q_bound < 0.5):
            raise MethodIncompatibility(
                f"tmle: q_bound must lie in (0, 0.5), got {q_bound!r}"
            )
        self.q_bound = float(q_bound)
        if fluctuation not in ("single", "per_arm"):
            raise ValueError(
                f"tmle: unknown fluctuation={fluctuation!r}; use "
                "'single' or 'per_arm'."
            )
        self.fluctuation = fluctuation
        self.Q_init = None if Q is None else np.asarray(Q, dtype=np.float64)
        self.g1W_init = None if g1W is None else np.asarray(g1W, dtype=np.float64)
        self.data = data
        self.y = y
        self.treat = treat
        self.covariates = covariates
        self.outcome_library = outcome_library
        self.propensity_library = propensity_library
        self.n_folds = n_folds
        self.estimand = estimand
        self.alpha = alpha
        self.propensity_bounds = propensity_bounds
        self.random_state = random_state
        if fold_indices is not None and (Q is not None or g1W is not None):
            raise MethodIncompatibility(
                "tmle: fold_indices cross-fits the Super Learner nuisances, "
                "but Q / g1W were supplied, so no nuisance would be fitted "
                "on the partition.",
                recovery_hint=(
                    "Drop fold_indices when passing precomputed Q / g1W (make "
                    "those out-of-fold yourself), or drop Q / g1W."
                ),
            )
        self.fold_indices = fold_indices
        if weights is not None and fold_indices is not None:
            raise MethodIncompatibility(
                "tmle: weights= is not implemented for the cross-fitted "
                "(fold_indices) path; its per-fold Super Learners would be "
                "fitted unweighted.",
                recovery_hint="Drop fold_indices, or drop weights=.",
            )
        if weights is not None and estimand != "ATE":
            raise MethodIncompatibility(
                "tmle: weights= is implemented for estimand='ATE' only.",
                recovery_hint="Drop weights= or use estimand='ATE'.",
            )
        self.weights = weights
        self.cluster = cluster

    def fit(self) -> CausalResult:
        """Run TMLE and return causal effect estimates."""
        # Prepare data
        design = [c for c in (self.weights, self.cluster) if c is not None]
        cols = list(dict.fromkeys([self.y, self.treat] + self.covariates + design))
        missing = [c for c in cols if c not in self.data.columns]
        if missing:
            raise ValueError(f"Columns not found in data: {missing}")

        clean = self.data[cols].dropna()
        Y = clean[self.y].values.astype(np.float64)
        A = clean[self.treat].values.astype(np.float64)
        W = clean[self.covariates].values.astype(np.float64)
        n = len(Y)
        # Observation weights normalised to mean one (R tmle's obsWeights);
        # None keeps the unweighted path byte-identical.
        obs_w: "Optional[np.ndarray]" = None
        if self.weights is not None:
            obs_w = clean[self.weights].to_numpy(dtype=np.float64)
            if not np.all(np.isfinite(obs_w)) or np.any(obs_w <= 0):
                raise MethodIncompatibility(
                    "tmle: weights must be finite and strictly positive."
                )
            obs_w = obs_w * (n / obs_w.sum())
        cl_codes: "Optional[np.ndarray]" = None
        if self.cluster is not None:
            cl_codes = pd.factorize(clean[self.cluster])[0]
            if cl_codes.max() + 1 < 2:
                raise MethodIncompatibility(
                    "tmle: cluster= needs at least two clusters."
                )

        unique_a = np.unique(A)
        if not (len(unique_a) == 2 and set(unique_a.astype(int)) == {0, 1}):
            raise ValueError(f"Treatment must be binary (0/1), got: {unique_a}")

        # Opt-in CV-TMLE partition, aligned with the input rows and then
        # subset to the complete cases kept above.
        folds: "Optional[np.ndarray]" = None
        if self.fold_indices is not None:
            from ..core._validate import validate_fold_indices

            folds = validate_fold_indices(
                self.fold_indices,
                len(self.data),
                context="sp.tmle",
                keep=self.data[cols].notna().all(axis=1).to_numpy(),
                binary_target=A,
            )

        # Detect if outcome is binary
        is_binary_outcome = set(np.unique(Y)) <= {0.0, 1.0}

        # For continuous outcomes, bound Y to [0,1] for logistic fluctuation
        if not is_binary_outcome:
            y_min, y_max = Y.min(), Y.max()
            y_range = y_max - y_min
            Y_scaled = (Y - y_min) / (y_range + 1e-10)
        else:
            Y_scaled = Y
            y_min, y_max, y_range = 0.0, 1.0, 1.0

        # ---------------------------------------------------------------
        # Step 1: Initial estimates via Super Learner
        # ---------------------------------------------------------------

        # Outcome model: Q(Y | A, W). A caller-supplied Q short-circuits
        # the Super Learner entirely, which is what makes the targeting
        # step comparable across implementations: with the initial fit
        # held fixed, only the fluctuation and the plug-in remain. Column
        # order matches ``tmle::tmle``'s Q argument, [Q(0,W), Q(1,W)].
        AW = np.column_stack([A, W])
        W1 = np.column_stack([np.ones(n), W])
        W0 = np.column_stack([np.zeros(n), W])

        if self.Q_init is not None:
            Q_in = self.Q_init
            if Q_in.ndim != 2 or Q_in.shape != (n, 2):
                raise ValueError(
                    f"tmle: Q must have shape (n, 2) = ({n}, 2) with columns "
                    f"[Q(0,W), Q(1,W)]; got {Q_in.shape}."
                )
            if not is_binary_outcome:
                Q_bar_0 = (Q_in[:, 0] - y_min) / (y_range + 1e-10)
                Q_bar_1 = (Q_in[:, 1] - y_min) / (y_range + 1e-10)
            else:
                Q_bar_0 = Q_in[:, 0].copy()
                Q_bar_1 = Q_in[:, 1].copy()
            Q_bar_A = np.where(A == 1, Q_bar_1, Q_bar_0)
        elif folds is not None:
            # CV-TMLE: each unit's initial predictions come from a Super
            # Learner that never saw it.
            Q_bar_A = np.empty(n)
            Q_bar_1 = np.empty(n)
            Q_bar_0 = np.empty(n)
            sl_Q_folds = []
            for k in range(int(folds.max()) + 1):
                tr = np.flatnonzero(folds != k)
                te = np.flatnonzero(folds == k)
                sl_Q_k = SuperLearner(
                    library=self.outcome_library,
                    n_folds=self.n_folds,
                    task="classification" if is_binary_outcome else "regression",
                    random_state=self.random_state,
                )
                _fit_in_fold(sl_Q_k, AW[tr], Y_scaled[tr], k, "outcome model Q")
                Q_bar_A[te] = sl_Q_k.predict(AW[te])
                Q_bar_1[te] = sl_Q_k.predict(W1[te])
                Q_bar_0[te] = sl_Q_k.predict(W0[te])
                sl_Q_folds.append(sl_Q_k)
        else:
            sl_Q = SuperLearner(
                library=self.outcome_library,
                n_folds=self.n_folds,
                task="classification" if is_binary_outcome else "regression",
                random_state=self.random_state,
            )
            sl_Q.fit(AW, Y_scaled, sample_weight=obs_w)
            Q_bar_A = sl_Q.predict(AW)  # Q(A_i, W_i) for observed A
            Q_bar_1 = sl_Q.predict(W1)  # Q(1, W_i)
            Q_bar_0 = sl_Q.predict(W0)  # Q(0, W_i)

        # Bound predictions
        eps_bound = self.q_bound
        Q_bar_A = np.clip(Q_bar_A, eps_bound, 1 - eps_bound)
        Q_bar_1 = np.clip(Q_bar_1, eps_bound, 1 - eps_bound)
        Q_bar_0 = np.clip(Q_bar_0, eps_bound, 1 - eps_bound)

        # Propensity model: g(A | W)
        if self.g1W_init is not None:
            g_hat_raw = np.asarray(self.g1W_init, dtype=np.float64).ravel()
            if g_hat_raw.shape[0] != n:
                raise ValueError(
                    f"tmle: g1W must have length n = {n}; " f"got {g_hat_raw.shape[0]}."
                )
        elif folds is not None:
            g_hat_raw = np.empty(n)
            sl_g_folds = []
            for k in range(int(folds.max()) + 1):
                tr = np.flatnonzero(folds != k)
                te = np.flatnonzero(folds == k)
                sl_g_k = SuperLearner(
                    library=self.propensity_library,
                    n_folds=self.n_folds,
                    task="classification",
                    random_state=self.random_state,
                )
                _fit_in_fold(sl_g_k, W[tr], A[tr], k, "propensity model g")
                g_hat_raw[te] = sl_g_k.predict(W[te])
                sl_g_folds.append(sl_g_k)
        else:
            sl_g = SuperLearner(
                library=self.propensity_library,
                n_folds=self.n_folds,
                task="classification",
                random_state=self.random_state,
            )
            sl_g.fit(W, A, sample_weight=obs_w)
            g_hat_raw = sl_g.predict(W)
        g_hat = np.clip(g_hat_raw, self.propensity_bounds[0], self.propensity_bounds[1])

        # Overlap diagnostics + loud warning when many propensities hit
        # the truncation bounds — the AIPW score blows up at e≈0 / e≈1,
        # so heavy clipping silently changes the estimand from the ATE
        # in the full population to an ATE on the trimmed sample.
        n_clip_lo = int(np.sum(g_hat_raw < self.propensity_bounds[0]))
        n_clip_hi = int(np.sum(g_hat_raw > self.propensity_bounds[1]))
        clip_share = (n_clip_lo + n_clip_hi) / max(n, 1)
        self._propensity_diagnostics = {
            "pscore_min": float(np.min(g_hat_raw)),
            "pscore_max": float(np.max(g_hat_raw)),
            "pscore_p01": float(np.quantile(g_hat_raw, 0.01)),
            "pscore_p99": float(np.quantile(g_hat_raw, 0.99)),
            "n_clipped_below": n_clip_lo,
            "n_clipped_above": n_clip_hi,
            "clip_share": float(clip_share),
            "propensity_bounds": tuple(self.propensity_bounds),
        }
        if clip_share > 0.05:
            import warnings

            warnings.warn(
                f"sp.tmle: {n_clip_lo + n_clip_hi}/{n} "
                f"({100 * clip_share:.1f}%) propensity scores hit the "
                f"{self.propensity_bounds} clip — overlap is poor and "
                f"the ATE / SE may be biased toward the trimmed "
                f"sample. Inspect "
                f"result.model_info['propensity_diagnostics'] and "
                f"consider sp.overlap_plot() / a more flexible "
                f"propensity model.",
                UserWarning,
                stacklevel=2,
            )

        # ---------------------------------------------------------------
        # Step 2: Targeting step (fluctuation parameter epsilon)
        # ---------------------------------------------------------------

        # Clever covariate H(A, W)
        if self.estimand == "ATE":
            H_A = A / g_hat - (1 - A) / (1 - g_hat)
            H_1 = 1.0 / g_hat
            H_0 = -1.0 / (1 - g_hat)
        else:  # ATT
            H_A = A - (1 - A) * g_hat / (1 - g_hat)
            H_1 = np.ones(n)
            H_0 = -g_hat / (1 - g_hat)

        # Logistic fluctuation model. Two parameterisations are in use in
        # the literature and they are NOT the same in finite samples:
        #
        #   'single'  (default, van der Laan & Rubin 2006): one clever
        #             covariate H(A,W) = A/g - (1-A)/(1-g) and a scalar
        #             epsilon. Solves the full EIF equation in one
        #             dimension.
        #   'per_arm' (the R ``tmle`` package's convention): two clever
        #             covariates, A/g and -(1-A)/(1-g), fitted jointly,
        #             giving a 2-vector epsilon and solving the treated
        #             and control score equations separately.
        #
        # Both are valid TMLEs and are asymptotically equivalent; they
        # differ at finite n. 'single' is the documented StatsPAI default
        # and its numbers are unchanged. 'per_arm' exists so results can
        # be reconciled with ``tmle::tmle`` (Track A module 72).
        logit_Q_A = logit(Q_bar_A)

        if self.fluctuation == "single":
            epsilon = self._fit_epsilon(Y_scaled, logit_Q_A, H_A, weights=obs_w)
            epsilon_vec = np.array([float(epsilon)])
            Q_star_A = expit(logit_Q_A + epsilon * H_A)
            Q_star_1 = expit(logit(Q_bar_1) + epsilon * H_1)
            Q_star_0 = expit(logit(Q_bar_0) + epsilon * H_0)
        else:
            # Per-arm covariates, evaluated at the observed A for fitting
            # and at each arm for the counterfactual updates.
            if self.estimand == "ATE":
                H1_A = A / g_hat
                H0_A = -(1 - A) / (1 - g_hat)
                H1_at1, H0_at1 = 1.0 / g_hat, np.zeros(n)
                H1_at0, H0_at0 = np.zeros(n), -1.0 / (1 - g_hat)
            else:  # ATT
                H1_A = A.astype(np.float64)
                H0_A = -(1 - A) * g_hat / (1 - g_hat)
                H1_at1, H0_at1 = np.ones(n), np.zeros(n)
                H1_at0, H0_at0 = np.zeros(n), -g_hat / (1 - g_hat)
            H_mat = np.column_stack([H1_A, H0_A])
            epsilon_vec = self._fit_epsilon_multi(
                Y_scaled, logit_Q_A, H_mat, weights=obs_w
            )
            e1, e0 = float(epsilon_vec[0]), float(epsilon_vec[1])
            Q_star_A = expit(logit_Q_A + e1 * H1_A + e0 * H0_A)
            Q_star_1 = expit(logit(Q_bar_1) + e1 * H1_at1 + e0 * H0_at1)
            Q_star_0 = expit(logit(Q_bar_0) + e1 * H1_at0 + e0 * H0_at0)
            epsilon = e1  # scalar slot keeps the legacy field populated

        # ---------------------------------------------------------------
        # Step 3: Plug-in estimate
        # ---------------------------------------------------------------

        if not is_binary_outcome:
            # Rescale back to original Y scale
            Q_star_1_orig = Q_star_1 * (y_range + 1e-10) + y_min
            Q_star_0_orig = Q_star_0 * (y_range + 1e-10) + y_min
            Q_star_A_orig = Q_star_A * (y_range + 1e-10) + y_min
        else:
            Q_star_1_orig = Q_star_1
            Q_star_0_orig = Q_star_0
            Q_star_A_orig = Q_star_A

        if self.estimand == "ATE" and obs_w is not None:
            # R tmle: mu_a = mean(w * Q*_a), IC = w * (... - (mu1 - mu0)).
            psi = float(np.mean(obs_w * (Q_star_1_orig - Q_star_0_orig)))
            EIF = obs_w * (
                (Q_star_1_orig - Q_star_0_orig)
                + A * (Y - Q_star_A_orig) / g_hat
                - (1 - A) * (Y - Q_star_A_orig) / (1 - g_hat)
                - psi
            )
        elif self.estimand == "ATE":
            psi = float(np.mean(Q_star_1_orig - Q_star_0_orig))

            # Efficient influence function
            EIF = (
                (Q_star_1_orig - Q_star_0_orig)
                + A * (Y - Q_star_A_orig) / g_hat
                - (1 - A) * (Y - Q_star_A_orig) / (1 - g_hat)
                - psi
            )
        else:  # ATT
            p_treat = np.mean(A)
            psi = float(
                np.mean(
                    A * (Y - Q_star_0_orig) / p_treat
                    - (1 - A) * g_hat * (Y - Q_star_0_orig) / ((1 - g_hat) * p_treat)
                )
            )

            EIF = (
                A * (Y - Q_star_0_orig) / p_treat
                - (1 - A) * g_hat * (Y - Q_star_0_orig) / ((1 - g_hat) * p_treat)
                - psi * A / p_treat
            )

        # Standard error from influence function
        if cl_codes is None:
            se = float(np.std(EIF, ddof=1) / np.sqrt(n))
        else:
            # Cluster sums of the influence function, centred, with the
            # G/(G-1) factor: equals the line above when every row is its
            # own cluster, and R tmle's id= variance for equal cluster sizes.
            S = np.bincount(cl_codes, weights=EIF)
            G = S.shape[0]
            se = float(np.sqrt(G / (G - 1) * np.sum((S - S.mean()) ** 2)) / n)

        if se > 0:
            z_stat = psi / se
            pvalue = float(2 * sp_stats.norm.sf(abs(z_stat)))
        else:
            pvalue = 0.0

        z_crit = sp_stats.norm.ppf(1 - self.alpha / 2)
        ci = (psi - z_crit * se, psi + z_crit * se)

        # Model info
        model_info = {
            "estimand": self.estimand,
            "se_method": (
                "efficient_influence_function"
                if cl_codes is None
                else "cluster_efficient_influence_function"
            ),
            "weights": self.weights,
            "cluster": self.cluster,
            "n_clusters": None if cl_codes is None else int(cl_codes.max() + 1),
            "propensity_mean": float(np.mean(g_hat)),
            "propensity_std": float(np.std(g_hat)),
            "propensity_bounds": self.propensity_bounds,
            "propensity_diagnostics": self._propensity_diagnostics,
            "outcome_type": "binary" if is_binary_outcome else "continuous",
            "n_folds": self.n_folds,
            "Q_star_1_mean": float(np.mean(Q_star_1_orig)),
            "Q_star_0_mean": float(np.mean(Q_star_0_orig)),
            "n_treated": int(np.sum(A == 1)),
            "n_control": int(np.sum(A == 0)),
            # Absent when the caller supplied the corresponding nuisance
            # directly: there is no Super Learner to report weights for,
            # and ``sl_Q`` / ``sl_g`` are never bound on that path.
            # One ensemble weight vector for the full-sample fit. Under
            # fold_indices there is one Super Learner per training
            # complement: these are None and the ``*_by_fold`` lists hold
            # entry k = weights of the ensemble fitted on rows with
            # fold != k.
            "sl_outcome_weights": (
                None
                if self.Q_init is not None or folds is not None
                else sl_Q.weights_.tolist()
            ),
            "sl_propensity_weights": (
                None
                if self.g1W_init is not None or folds is not None
                else sl_g.weights_.tolist()
            ),
            "sl_outcome_weights_by_fold": (
                [m.weights_.tolist() for m in sl_Q_folds] if folds is not None else None
            ),
            "sl_propensity_weights_by_fold": (
                [m.weights_.tolist() for m in sl_g_folds] if folds is not None else None
            ),
            "nuisance_source": {
                "Q": (
                    "supplied"
                    if self.Q_init is not None
                    else (
                        "super_learner_out_of_fold"
                        if folds is not None
                        else "super_learner"
                    )
                ),
                "g1W": (
                    "supplied"
                    if self.g1W_init is not None
                    else (
                        "super_learner_out_of_fold"
                        if folds is not None
                        else "super_learner"
                    )
                ),
            },
            # True only for the opt-in CV-TMLE path; the default fits both
            # nuisances on the full sample and evaluates them in sample.
            "cross_fitted": folds is not None,
            "n_cv_folds": None if folds is None else int(folds.max()) + 1,
            "fluctuation": self.fluctuation,
            "q_bound": self.q_bound,
            # ``epsilon`` stays the scalar it has always been under the
            # default single-covariate fluctuation. Under 'per_arm' there
            # is no scalar fluctuation parameter, so it is None rather
            # than an arbitrary element of the pair; ``epsilon_vec``
            # carries the full vector in both modes.
            "epsilon": (float(epsilon) if self.fluctuation == "single" else None),
            "epsilon_vec": [float(v) for v in np.atleast_1d(epsilon_vec)],
        }

        # One fitted Super Learner, or one per fold under CV-TMLE.
        self._sl_Q: Any
        self._sl_g: Any
        if folds is not None:
            self._sl_Q = sl_Q_folds
            self._sl_g = sl_g_folds
        else:
            self._sl_Q = None if self.Q_init is not None else sl_Q
            self._sl_g = None if self.g1W_init is not None else sl_g
        self._epsilon = epsilon
        self._epsilon_vec = np.atleast_1d(epsilon_vec)

        return CausalResult(
            method=(
                "CV-TMLE (van der Laan & Rose 2011)"
                if folds is not None
                else "TMLE (van der Laan & Rose 2011)"
            ),
            estimand=self.estimand,
            estimate=psi,
            se=se,
            pvalue=pvalue,
            ci=ci,
            alpha=self.alpha,
            n_obs=n,
            detail=None,
            model_info=model_info,
            _citation_key="tmle",
        )

    def _fit_epsilon_multi(
        self,
        Y: np.ndarray,
        logit_Q: np.ndarray,
        H: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-10,
        weights: "Optional[np.ndarray]" = None,
    ) -> np.ndarray:
        """Fit a vector fluctuation parameter by Newton-Raphson.

        The multivariate counterpart of :meth:`_fit_epsilon`: a logistic
        regression of ``Y`` on the clever-covariate columns of ``H`` with
        ``logit_Q`` as a fixed offset and no intercept. Used by
        ``fluctuation='per_arm'``, which is the parameterisation the R
        ``tmle`` package uses.

        Failing to converge means the plug-in is not fully de-biased, so
        it warns rather than returning a silently untargeted fit.
        """
        eps = np.zeros(H.shape[1], dtype=np.float64)
        converged = False
        ow = np.ones(len(Y)) if weights is None else weights
        for _ in range(max_iter):
            p = expit(logit_Q + H @ eps)
            score = H.T @ (ow * (Y - p))
            w = ow * p * (1.0 - p)
            hessian = -(H * w[:, None]).T @ H
            try:
                delta = np.linalg.solve(hessian, -score)
            except np.linalg.LinAlgError:  # pragma: no cover
                break
            eps = eps + delta
            if np.max(np.abs(delta)) < tol:
                converged = True
                break
        if not converged:
            import warnings

            resid = float(np.max(np.abs(H.T @ (Y - expit(logit_Q + H @ eps)))))
            warnings.warn(
                "TMLE: Newton iteration on the per-arm fluctuation "
                f"parameters did not converge in {max_iter} steps "
                f"(max |score| = {resid:.3g}). The plug-in estimate is "
                "not fully target-de-biased; inspect overlap and the "
                "initial Q fit.",
                UserWarning,
                stacklevel=3,
            )
        return eps

    def _fit_epsilon(
        self,
        Y: np.ndarray,
        logit_Q: np.ndarray,
        H: np.ndarray,
        max_iter: int = 50,
        tol: float = 1e-8,
        weights: "Optional[np.ndarray]" = None,
    ) -> Any:
        """
        Fit the fluctuation parameter epsilon via Newton-Raphson.

        Logistic model: P(Y=1 | H) = expit(logit_Q + epsilon * H)
        MLE for epsilon using iteratively reweighted least squares.
        """
        epsilon = 0.0
        converged = False

        for it in range(max_iter):
            p = expit(logit_Q + epsilon * H)
            # Score: sum(w * H * (Y - p)); Hessian: -sum(w * H^2 * p * (1-p))
            if weights is None:
                score = np.sum(H * (Y - p))
                hessian = -np.sum(H**2 * p * (1 - p))
            else:
                score = np.sum(weights * H * (Y - p))
                hessian = -np.sum(weights * H**2 * p * (1 - p))

            if abs(hessian) < 1e-15:
                # Singular Hessian — exit but flag.
                break

            delta = -score / hessian
            epsilon += delta

            if abs(delta) < tol:
                converged = True
                break

        if not converged:
            # Targeting failed to converge in `max_iter` Newton steps.
            # Returning the current epsilon means the plug-in is not
            # fully target-de-biased; warn the caller rather than fail
            # silently. Asymptotic theory still applies once nuisance
            # rates are good enough, but finite-sample inference may be
            # less reliable. Surface the latest score to help debugging.
            import warnings

            final_score = float(np.sum(H * (Y - expit(logit_Q + epsilon * H))))
            warnings.warn(
                f"TMLE: Newton iteration on the fluctuation parameter "
                f"epsilon did not converge in {max_iter} steps "
                f"(final |score|={abs(final_score):.2e}, "
                f"epsilon={epsilon:.3e}). The plug-in estimate may not "
                f"satisfy the targeting equation; coverage may be "
                f"affected. Check propensity overlap or tighten "
                f"`propensity_bounds=`.",
                UserWarning,
                stacklevel=3,
            )

        return epsilon


# ======================================================================
# Citation
# ======================================================================

CausalResult._CITATIONS["tmle"] = (
    # Kept verbatim in sync with paper.bib (single source of truth per
    # CLAUDE.md §10). Touching this block requires touching paper.bib
    # too. We register all three: the methodology references in the
    # docstring (vanderlaan2011targeted, vanderlaan2006targeted) and
    # the Super Learner reference used internally (vanderlaan2007super).
    "@book{vanderlaan2011targeted,\n"
    "  title={Targeted Learning: Causal Inference for Observational "
    "and Experimental Data},\n"
    "  author={van der Laan, Mark J. and Rose, Sherri},\n"
    "  year={2011},\n"
    "  publisher={Springer},\n"
    "  series={Springer Series in Statistics}\n"
    "}\n\n"
    "@article{vanderlaan2006targeted,\n"
    "  title={Targeted Maximum Likelihood Learning},\n"
    "  author={van der Laan, Mark J. and Rubin, Daniel},\n"
    "  journal={The International Journal of Biostatistics},\n"
    "  year={2006},\n"
    "  doi={10.2202/1557-4679.1043}\n"
    "}\n\n"
    "@article{vanderlaan2007super,\n"
    "  title={Super Learner},\n"
    "  author={van der Laan, Mark J. and Polley, Eric C. and "
    "Hubbard, Alan E.},\n"
    "  journal={Statistical Applications in Genetics and Molecular Biology},\n"
    "  year={2007},\n"
    "  doi={10.2202/1544-6115.1309}\n"
    "}"
)
