"""
Causal survival forest (Cui, Kosorok, Sverdrup, Wager and Zhu 2023).

Heterogeneous treatment effects on a right-censored outcome, for a binary
treatment under unconfoundedness and conditionally independent censoring.
The estimand at horizon ``h`` is either

* ``target="RMST"``: ``tau(x) = E[min(T(1), h) - min(T(0), h) | X = x]``, or
* ``target="survival_probability"``:
  ``tau(x) = P[T(1) > h | X = x] - P[T(0) > h | X = x]``.

Method (paper Sec. 2.3 and eq. 11-14)
-------------------------------------
Nuisances, all out-of-bag: the propensity ``e(x)`` (regression forest);
the conditional survival functions of the event and of censoring given
``(X, W)`` (survival forests on ``[X, W]``; the counterfactual-arm curves
are read from the same out-of-bag trees with ``W`` switched).  From them
``Q_w(s | x) = E[y(T) | X = x, W = w, T > s]`` and ``m(x) = e mu_1 + (1 - e)
mu_0``.  The censoring-robust score of the causal-forest equation is linear
in ``tau``, ``psi_i(tau) = A_i - tau B_i``, with

.. math::

    A_i = (W_i - e_i)\\Bigl[\\frac{\\Delta^h_i y_i + (1 - \\Delta^h_i)
    Q_i(U_i \\wedge h) - m_i}{S^C_i(U_i \\wedge h)}
    - \\int_0^{U_i \\wedge h} \\frac{\\lambda^C_i(s)}{S^C_i(s)}
      \\bigl(Q_i(s) - m_i\\bigr)\\,ds\\Bigr],
    \\qquad
    B_i = (W_i - e_i)^2 \\Bigl[\\frac{1}{S^C_i(U_i \\wedge h)}
    - \\int_0^{U_i \\wedge h} \\frac{\\lambda^C_i(s)}{S^C_i(s)}\\,ds\\Bigr],

``Delta^h = 1{event or U >= h}``.  Discretisation follows the reference
implementation (``grf``, maintained by the method's authors): the integral
is the sum over censoring-grid times ``c_k <= U^h`` of
``[log S^C(c_{k-1}) - log S^C(c_k)] / S^C(c_k) (Q(c_k) - m)``, ``S^C(U^h)`` is
the curve's value at the last grid time at or below ``U^h``, and the
bracket of ``B_i`` is replaced by its population value, one (the identity
``1/S^C(u) = 1 + int_0^u lambda^C / S^C``), so ``B_i = (W_i - e_i)^2``.  The forest
solves ``sum_i alpha_i(x) (A_i - tau B_i) = 0`` with trees split on
``rho_i = (A_i - tau_P B_i) / mean(B)`` (eq. 14), stabilised on the
treatment and requiring ``split_alpha`` of the parent's size in failures in
each child; variances come from little bags (Sec. 3.1).  The average effect
uses ``Gamma_i = tau(X_i) + psi_i(tau(X_i)) / (e_i (1 - e_i))`` (eq. 27).

References
----------
[@cui2023estimating], [@athey2019generalized]
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .._aliases import accepts_aliases
from ..exceptions import DataInsufficient, MethodIncompatibility
from ..forest import _grf_engine as engine
from ..forest._grf_family import (
    ForestOptions,
    GRFFamilyForest,
    cluster_codes,
    nuisance_trees,
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
from ..forest.survival_forest import (
    check_survival_inputs,
    failure_grid,
    train_survival_engine,
)

_CONTEXT = "causal_survival_forest"
_SURVIVAL_FOREST_ALTERNATIVES = [
    "sp.survival.causal_survival_forest",
    "sp.causal_survival_forest",
    "sp.survival.cox",
]


def _csf_error(
    message: str,
    *,
    diagnostics: Optional[Dict[str, Any]] = None,
    recovery_hint: str = "Check causal-survival-forest inputs.",
) -> MethodIncompatibility:
    return MethodIncompatibility(
        message,
        recovery_hint=recovery_hint,
        diagnostics=diagnostics,
        alternative_functions=_SURVIVAL_FOREST_ALTERNATIVES,
    )


class CausalSurvivalForestResult(GRFFamilyForest):
    """A fitted causal survival forest.

    Attributes
    ----------
    ate : float
        Doubly-robust average effect (RMST difference or survival-
        probability difference at ``horizon``).
    ate_rmst : float
        Same as ``ate`` when ``target="RMST"`` (kept for compatibility);
        NaN otherwise.
    se, ci, pvalue : float, tuple, float
    cate : np.ndarray
        Out-of-bag ``tau(X_i)``.
    cate_variance : np.ndarray or None
    horizon : float
    target : str
    n_obs, n_trees : int
    detail : dict

    Examples
    --------
    >>> import numpy as np, statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(600, 2)); W = rng.binomial(1, 0.5, 600)
    >>> T = rng.exponential(1 / np.exp(0.5 * X[:, 0] - 0.5 * W))
    >>> C = rng.exponential(3, 600)
    >>> fit = sp.causal_survival_forest(time=np.minimum(T, C),
    ...                                 event=(T <= C).astype(int), treat=W,
    ...                                 covariates=X, horizon=1.0,
    ...                                 n_estimators=200)
    >>> fit.horizon
    1.0
    """

    _citation_keys = ("cui2023estimating",)
    _context = _CONTEXT

    def __init__(self) -> None:
        self.ate = float("nan")
        self.ate_rmst = float("nan")
        self.se = float("nan")
        self.ci = (float("nan"), float("nan"))
        self.pvalue = float("nan")
        self.alpha = 0.05
        self.cate: np.ndarray = np.zeros(0)
        self.cate_variance: Optional[np.ndarray] = None
        self.horizon = float("nan")
        self.target = "RMST"
        self.n_obs = 0
        self.n_trees = 0
        self.feature_names: list = []
        self.detail: Dict[str, Any] = {}

    def predict(
        self, newdata: Any = None, estimate_variance: bool = False
    ) -> pd.DataFrame:
        """``tau(x)`` at ``newdata`` (OOB for the training rows when None)."""
        if newdata is None:
            pred, var = self.cate, self.cate_variance
        else:
            p, v = self._engine.predict(
                self._new_X(newdata), estimate_variance=bool(estimate_variance)
            )
            pred, var = p[:, 0], v[:, 0]
        out = pd.DataFrame({"predictions": np.asarray(pred, dtype=float)})
        if estimate_variance:
            if var is None:
                raise MethodIncompatibility(
                    f"{_CONTEXT}: variance estimates need ci_group_size >= 2.",
                    recovery_hint="Refit with ci_group_size=2.",
                )
            out["variance_estimates"] = np.asarray(var, dtype=float)
        return out

    def get_scores(self) -> np.ndarray:
        """Doubly-robust scores ``tau(X_i) + (A_i - tau B_i) / (e_i (1 - e_i))``."""
        return csf_dr_scores(self.cate, self._A, self._B, self._e_scores)

    def average_treatment_effect(self, alpha: float = 0.05) -> Dict[str, Any]:
        alpha = validate_alpha(alpha, f"{_CONTEXT}.average_treatment_effect()")
        out = score_average(self.get_scores(), self._obs_weight, self._clusters, alpha)
        out.update(
            estimand=f"{self.target} difference at horizon {self.horizon:g}",
            method="aipcw",
            n=self.n_obs,
            alpha=alpha,
        )
        return out

    @accepts_aliases(_strict=True, vcov_type="vce")
    def best_linear_projection(
        self, A: Any = None, vce: str = "HC3", alpha: float = 0.05
    ) -> pd.DataFrame:
        from ..forest.forest_tools import _projection_design

        alpha = validate_alpha(alpha, f"{_CONTEXT}.best_linear_projection()")
        vce = validate_vcov_type(vce, _CONTEXT)
        A_mat, names = _projection_design(A, self, _CONTEXT)
        return score_blp(
            self.get_scores(),
            A_mat,
            names,
            self._obs_weight,
            self._clusters,
            vce,
            alpha,
            f"causal-survival DR scores on OOB CATE, {vce} robust SE",
        )

    def summary(self) -> str:
        lo, hi = self.ci
        label = "ATE(RMST)" if self.target == "RMST" else "ATE(S(h))"
        return (
            "Causal Survival Forest (Cui et al. 2023)\n"
            "----------------------------------------\n"
            f"  target          : {self.target}\n"
            f"  horizon (h)     : {self.horizon:.4g}\n"
            f"  trees           : {self.n_trees}\n"
            f"  n               : {self.n_obs}\n"
            f"  {label:<15} : {self.ate:.4f}  (SE={self.se:.4f})\n"
            f"  {100 * (1 - self.alpha):.0f}% CI          : [{lo:.4f}, {hi:.4f}]\n"
            f"  p-value         : {self.pvalue:.4g}"
        )

    def __repr__(self) -> str:
        return (
            f"CausalSurvivalForestResult(target={self.target}, ATE={self.ate:.4f}, "
            f"se={self.se:.4f})"
        )


# --------------------------------------------------------------------------- #
#  Score construction
# --------------------------------------------------------------------------- #


def _area(S: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """``A(t_k) = int_0^{t_k} S`` for right-continuous step curves ``S``
    (value 1 before the grid), at ``t_0 = 0, t_1, ..., t_F``."""
    times = np.concatenate([[0.0], grid])
    padded = np.column_stack([np.ones(S.shape[0]), S])
    seg = np.diff(times)
    area = np.zeros_like(padded)
    area[:, 1:] = np.cumsum(padded[:, :-1] * seg[None, :], axis=1)
    return area


def _q_values(
    row: int,
    s: np.ndarray,
    Spad: np.ndarray,
    Apad: np.ndarray,
    times: np.ndarray,
    grid: np.ndarray,
    h: float,
    A_h: float,
    S_h: float,
    rmst: bool,
) -> np.ndarray:
    """``Q(s) = E[y(T) | T > s]`` for one row's event curve."""
    k = np.searchsorted(grid, s, side="right")
    S_s = Spad[row, k]
    if rmst:
        A_s = Apad[row, k] + S_s * (s - times[k])
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(S_s > 0, s + (A_h - A_s) / S_s, s)
        return np.asarray(np.minimum(q, h), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(S_s > 0, np.minimum(S_h / S_s, 1.0), 0.0)


def csf_scores(
    Y: np.ndarray,
    D: np.ndarray,
    W: np.ndarray,
    e: np.ndarray,
    S1: np.ndarray,
    S0: np.ndarray,
    event_grid: np.ndarray,
    SC: np.ndarray,
    censor_grid: np.ndarray,
    horizon: float,
    target: str,
) -> Dict[str, np.ndarray]:
    """Numerator ``A`` and denominator ``B`` of the CSF estimating equation
    (Cui et al. 2023, eq. 11) from the nuisance estimates.

    ``S1``/``S0`` are event survival curves under ``W = 1``/``W = 0`` on
    ``event_grid``; ``SC`` the censoring survival curve at the observed ``W``
    on ``censor_grid`` (right-continuous, ``P[C > t]``).
    """
    n = Y.size
    rmst = target == "RMST"
    h = float(horizon)
    Uh = np.minimum(Y, h)
    Dh = ((D > 0.5) | (Y >= h)).astype(float)
    if rmst:
        y_obs = Uh
    else:
        y_obs = ((Y > h) | ((Y == h) & (D < 0.5))).astype(float)
    times = np.concatenate([[0.0], event_grid])
    kh = int(np.searchsorted(event_grid, h, side="right"))
    Sobs = np.where((W > 0.5)[:, None], S1, S0)
    out_mu = {}
    for label, S in (("mu1", S1), ("mu0", S0)):
        Spad = np.column_stack([np.ones(n), S])
        Apad = _area(S, event_grid)
        A_h = Apad[:, kh] + Spad[:, kh] * (h - times[kh])
        out_mu[label] = A_h if rmst else Spad[:, kh]
    m = e * out_mu["mu1"] + (1.0 - e) * out_mu["mu0"]
    Spad = np.column_stack([np.ones(n), Sobs])
    Apad = _area(Sobs, event_grid)
    A_h_obs = Apad[:, kh] + Spad[:, kh] * (h - times[kh])
    S_h_obs = Spad[:, kh]
    SCpad = np.column_stack([np.ones(n), SC])
    # Censoring survival at U^h (grid convention: the value at the last
    # censoring-grid time at or below U^h, i.e. after a censoring jump at U^h).
    k_u = np.searchsorted(censor_grid, Uh, side="right")
    sc_u = SCpad[np.arange(n), k_u]
    if np.any(sc_u <= 0):
        bad = int(np.sum(sc_u <= 0))
        raise DataInsufficient(
            f"{_CONTEXT}: the estimated censoring survival at min(T, h) is zero "
            f"for {bad} row(s), so the censoring weights are undefined "
            "(positivity of Cui et al. 2023, Assumption 6, fails at this "
            "horizon).",
            recovery_hint="Choose a shorter horizon.",
            diagnostics={"n_rows": bad, "horizon": h},
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        log_sc = np.log(SCpad)
        # lambda^C(c_k) / S^C(c_k) with the backward difference of -log S^C.
        haz_w = (log_sc[:, :-1] - log_sc[:, 1:]) / SCpad[:, 1:]
    integral_q = np.zeros(n)
    for i in range(n):
        cnt = int(k_u[i])
        if cnt == 0:
            continue
        wi = haz_w[i, :cnt]
        if not np.all(np.isfinite(wi)):
            raise DataInsufficient(
                f"{_CONTEXT}: the censoring survival reaches zero before "
                "min(T, h) for some rows.",
                recovery_hint="Choose a shorter horizon.",
            )
        q = _q_values(
            i,
            censor_grid[:cnt],
            Spad,
            Apad,
            times,
            event_grid,
            h,
            A_h_obs[i],
            S_h_obs[i],
            rmst,
        )
        integral_q[i] = float(np.sum(wi * (q - m[i])))
    q_u = np.array(
        [
            _q_values(
                i,
                np.array([Uh[i]]),
                Spad,
                Apad,
                times,
                event_grid,
                h,
                A_h_obs[i],
                S_h_obs[i],
                rmst,
            )[0]
            for i in range(n)
        ]
    )
    first = Dh * y_obs + (1.0 - Dh) * q_u
    resid_w = W - e
    A = resid_w * ((first - m) / sc_u - integral_q)
    # With the true censoring law the bracket 1/S^C(U) - int lambda^C/S^C is
    # identically one; the denominator uses that identity (as grf does).
    B = resid_w**2
    return {
        "A": A,
        "B": B,
        "m": m,
        "mu1": out_mu["mu1"],
        "mu0": out_mu["mu0"],
        "sc_at_u": sc_u,
        "Dh": Dh,
    }


def csf_dr_scores(
    tau: np.ndarray, A: np.ndarray, B: np.ndarray, e: np.ndarray
) -> np.ndarray:
    """``Gamma_i = tau_i + (A_i - tau_i B_i) / (e_i (1 - e_i))`` (eq. 27)."""
    return np.asarray(tau + (A - tau * B) / (e * (1.0 - e)), dtype=float)


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #


@accepts_aliases(
    _strict=True,
    n_trees="n_estimators",
    min_leaf="min_samples_leaf",
    seed="random_state",
)
def causal_survival_forest(
    data: Optional[pd.DataFrame] = None,
    time: Any = None,
    event: Any = None,
    treat: Any = None,
    covariates: Any = None,
    horizon: Optional[float] = None,
    *,
    target: str = "RMST",
    failure_times: Any = None,
    clusters: Any = None,
    weights: Any = None,
    equalize_cluster_weights: bool = False,
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
    propensity_bounds: Optional[Sequence[float]] = None,
    random_state: Optional[int] = 42,
    n_jobs: int = 1,
    alpha: float = 0.05,
) -> CausalSurvivalForestResult:
    """
    Causal survival forest: heterogeneous effects on censored survival.

    Parameters
    ----------
    data : pd.DataFrame, optional
        Input data; when omitted, the other inputs are arrays.
    time : str or array-like
        Observed time ``min(T, C)``, non-negative.
    event : str or array-like
        1 = event observed, 0 = censored.
    treat : str or array-like
        Binary treatment.
    covariates : str, list of str or 2-D array
        Covariates ``X`` (confounders and effect modifiers).
    horizon : float, optional
        ``h`` of the estimand.  Defaults to the 80th percentile of observed
        event times (recorded in ``detail``); grf requires it -- choose it
        from the study design.
    target : {"RMST", "survival_probability"}, default "RMST"
    failure_times : array-like, optional
        Grid for the survival nuisance forests (default: observed times).
    clusters, weights, equalize_cluster_weights
        As in :func:`statspai.causal_forest`.
    W_hat : array-like, optional
        Precomputed propensities; default out-of-bag regression forest.
    n_estimators : int, default 2000
    min_samples_leaf : int, default 5
    max_samples, mtry, honest, honesty_fraction, honesty_prune_leaves,
    imbalance_penalty, stabilize_splits, ci_group_size, max_depth
        Forest options (grf names).
    split_alpha : float, default 0.05
        grf ``alpha``; each child must also hold ``split_alpha`` of the
        parent's size in failures.
    propensity_bounds : (float, float), optional
        Clip ``e`` in the average-effect score denominators (off by
        default, as in grf).
    random_state : int, default 42
    n_jobs : int, default 1
    alpha : float, default 0.05
        Significance level of the reported interval.

    Returns
    -------
    CausalSurvivalForestResult
        ``.ate`` / ``.se`` / ``.ci`` / ``.pvalue``, ``.cate`` (OOB),
        ``.predict(newdata)``, ``.best_linear_projection(A)``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 600; x = rng.normal(size=(n, 2)); w = rng.binomial(1, 0.5, n)
    >>> T = rng.exponential(1 / np.exp(0.5 * x[:, 0] - 0.5 * w))
    >>> C = rng.exponential(3, n)
    >>> df = pd.DataFrame({"t": np.minimum(T, C), "d": (T <= C).astype(int),
    ...                    "w": w, "x1": x[:, 0], "x2": x[:, 1]})
    >>> csf = sp.causal_survival_forest(df, time="t", event="d", treat="w",
    ...                                 covariates=["x1", "x2"], horizon=1.0,
    ...                                 n_estimators=200)
    >>> csf.target
    'RMST'

    References
    ----------
    [@cui2023estimating], [@athey2019generalized]
    """
    alpha = validate_alpha(alpha, _CONTEXT)
    if data is not None and not isinstance(data, pd.DataFrame):
        raise _csf_error(
            "causal_survival_forest data must be a pandas DataFrame.",
            diagnostics={"type": type(data).__name__},
            recovery_hint="Pass a pandas DataFrame with survival outcome rows.",
        )
    tgt = str(target).replace(".", "_").lower()
    if tgt in ("rmst",):
        tgt = "RMST"
    elif tgt in ("survival_probability", "survival_prob"):
        tgt = "survival_probability"
    else:
        raise _csf_error(
            f"target must be 'RMST' or 'survival_probability', got {target!r}.",
            recovery_hint="Use target='RMST' or target='survival_probability'.",
        )
    bounds = None
    if propensity_bounds is not None:
        if len(propensity_bounds) != 2:
            raise _csf_error(
                "propensity_bounds must contain exactly two values.",
                diagnostics={"propensity_bounds": list(propensity_bounds)},
                recovery_hint="Use propensity_bounds=(0.05, 0.95) or None.",
            )
        p_lo, p_hi = float(propensity_bounds[0]), float(propensity_bounds[1])
        if not (0 < p_lo < p_hi < 1):
            raise _csf_error(
                "propensity_bounds must satisfy 0 < lower < upper < 1.",
                diagnostics={"propensity_bounds": [p_lo, p_hi]},
                recovery_hint="Use bounds such as (0.05, 0.95).",
            )
        bounds = (p_lo, p_hi)
    if horizon is not None:
        horizon = float(horizon)
        if not np.isfinite(horizon) or horizon <= 0:
            raise _csf_error(
                "horizon must be a positive finite number.",
                diagnostics={"horizon": horizon},
                recovery_hint="Pass a positive RMST horizon.",
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
    if isinstance(covariates, str) and not covariates:
        raise _csf_error(
            "covariates[0] must be a non-empty column-name string.",
            recovery_hint="Pass covariates as column-name strings.",
        )
    arrays, names, keep, n_input = resolve_inputs(
        _CONTEXT,
        data,
        {"time": time, "event": event, "treat": treat, "covariates": covariates},
        {"clusters": clusters, "weights": weights},
    )
    Y = arrays["time"][:, 0]
    D = arrays["event"][:, 0]
    W = arrays["treat"][:, 0]
    X = arrays["covariates"]
    n = Y.size
    check_survival_inputs(Y, D, _CONTEXT)
    if not set(np.unique(W).tolist()).issubset({0.0, 1.0}):
        raise _csf_error(
            "treat must be binary 0/1.",
            diagnostics={"treat_values": np.unique(W)[:10].tolist()},
            recovery_hint="Encode treatment as binary 0/1 before fitting.",
        )
    if np.unique(W).size < 2:
        raise DataInsufficient(
            "causal_survival_forest requires both treatment arms.",
            recovery_hint="Provide complete rows from treated and control units.",
            diagnostics={"treat_values": np.unique(W).tolist()},
            alternative_functions=_SURVIVAL_FOREST_ALTERNATIVES,
        )
    horizon_source = "user"
    if horizon is None:
        horizon = float(np.quantile(Y[D > 0.5], 0.80))
        horizon_source = "default: 80th percentile of observed event times"
    if horizon <= float(np.min(Y)):
        raise _csf_error(
            "horizon must exceed the smallest observed time.",
            diagnostics={"horizon": horizon, "min_time": float(np.min(Y))},
            recovery_hint="Pass a later horizon.",
        )

    cl = cluster_codes(arrays.get("clusters"), _CONTEXT)
    sw = sample_weights(arrays.get("weights"), equalize_cluster_weights, _CONTEXT)
    common = opts.engine_kwargs(cl, equalize_cluster_weights, sw)
    sources = {}
    if W_hat is None:
        e = oob_regression(X, W, opts, common, "W_hat", _CONTEXT)
        sources["W_hat"] = "regression forest (OOB)"
    else:
        arr = np.asarray(W_hat, dtype=float).ravel()
        if arr.size == n_input and n_input != n:
            arr = arr[keep]
        e = user_nuisance(arr, n, 1, "W_hat", _CONTEXT)[:, 0]
        sources["W_hat"] = "user-supplied"

    XW = np.column_stack([X, W])
    s_trees = nuisance_trees(opts.n_estimators)
    s_mtry = None if opts.mtry is None else int(min(max(opts.mtry, 1), XW.shape[1]))
    # The nuisance curves use the full grids: truncating a grid at the
    # horizon would round every later event down onto the last grid time
    # and count it as a failure before the horizon.
    event_grid = failure_grid(Y, D, failure_times)
    if not np.any(event_grid <= horizon):
        raise DataInsufficient(
            f"{_CONTEXT}: no events at or before the horizon.",
            recovery_hint="Choose a later horizon.",
        )
    sf_event = train_survival_engine(
        XW,
        Y,
        D,
        event_grid,
        num_trees=s_trees,
        min_node_size=15,
        mtry=s_mtry,
        sample_weight=sw,
        common=with_stream(common, "event"),
    )
    XW1 = XW.copy()
    XW1[:, -1] = 1.0
    XW0 = XW.copy()
    XW0[:, -1] = 0.0
    S1 = sf_event.predict_survival(XW1, oob=True)
    S0 = sf_event.predict_survival(XW0, oob=True)
    require_finite(S1, "event survival (W=1)", _CONTEXT)
    require_finite(S0, "event survival (W=0)", _CONTEXT)
    Dc = 1.0 - D
    censor_grid = np.unique(Y[Dc > 0.5])
    if not np.any(censor_grid < horizon):
        SC = np.ones((n, 0))
        censor_grid = np.zeros(0)
        sources["censoring"] = "no censoring before the horizon"
    else:
        sf_cens = train_survival_engine(
            XW,
            Y,
            Dc,
            censor_grid,
            num_trees=s_trees,
            min_node_size=15,
            mtry=s_mtry,
            sample_weight=sw,
            common=with_stream(common, "censoring"),
        )
        SC = sf_cens.predict_survival(XW, oob=True)
        require_finite(SC, "censoring survival", _CONTEXT)
        sources["censoring"] = "survival forest on [X, W] (OOB)"
    sources["event"] = "survival forest on [X, W] (OOB, both arms)"
    sc = csf_scores(Y, D, W, e, S1, S0, event_grid, SC, censor_grid, horizon, tgt)
    A, B = sc["A"], sc["B"]
    if not (np.isfinite(A).all() and np.isfinite(B).all()):
        raise DataInsufficient(
            f"{_CONTEXT}: non-finite censoring-adjusted scores.",
            recovery_hint="Choose a shorter horizon or check overlap.",
        )
    forest = engine.train_forest(
        X,
        np.zeros(n),
        kind=engine.KIND_CAUSAL_SURV,
        num_trees=opts.n_estimators,
        mtry=opts.mtry,
        min_node_size=int(opts.min_samples_leaf),
        stabilize_splits=bool(opts.stabilize_splits),
        ci_group_size=int(opts.ci_group_size),
        max_depth=opts.max_depth,
        M=np.column_stack([A, B, W - e, sc["Dh"]]),
        params=np.zeros(1),
        **common,
    )
    tau, var = forest.predict_oob(X, estimate_variance=opts.ci_group_size > 1)
    tau = tau[:, 0]
    require_finite(tau, "causal survival", _CONTEXT)

    res = CausalSurvivalForestResult()
    res._engine = forest
    res._X = X
    res._A, res._B = A, B
    e_scores = e if bounds is None else np.clip(e, bounds[0], bounds[1])
    if np.any((e_scores <= 0) | (e_scores >= 1)):
        raise DataInsufficient(
            f"{_CONTEXT}: estimated propensities of 0 or 1 make the "
            "average-effect scores undefined.",
            recovery_hint="Pass propensity_bounds=(0.05, 0.95) or trim the sample.",
        )
    res._e_scores = e_scores
    res._clusters = cl
    res._obs_weight = observation_weights(cl, equalize_cluster_weights, sw, n)
    res.feature_names = names["covariates"]
    res.n_obs = n
    res.n_trees = forest.num_trees
    res.alpha = alpha
    res.horizon = float(horizon)
    res.target = tgt
    res.cate = tau
    res.cate_variance = var[:, 0] if opts.ci_group_size > 1 else None
    ate = res.average_treatment_effect(alpha=alpha)
    res.ate = ate["estimate"]
    res.ate_rmst = res.ate if tgt == "RMST" else float("nan")
    res.se = ate["se"]
    z = z_crit(alpha)
    res.ci = (res.ate - z * res.se, res.ate + z * res.se)
    res.pvalue = ate["pvalue"]
    res.detail = {
        "horizon_source": horizon_source,
        "nuisance_source": sources,
        "propensity_range": (float(e.min()), float(e.max())),
        "propensity_bounds": bounds,
        "min_censoring_survival_at_u": float(np.min(sc["sc_at_u"])),
        "n_events": int(np.sum(D > 0.5)),
        "n_effective_complete": int(np.sum(sc["Dh"] > 0.5)),
        "n_dropped_missing": int(n_input - n),
        "n_clusters": None if cl is None else int(cl.max()) + 1,
        "options": dict(opts.__dict__),
    }
    if float(np.min(sc["sc_at_u"])) < 0.05:
        warnings.warn(
            f"{_CONTEXT}: the estimated censoring survival falls to "
            f"{float(np.min(sc['sc_at_u'])):.3g} before min(T, horizon) for some "
            "rows; their inverse-censoring weights are large and the "
            "estimates unstable. A shorter horizon is safer.",
            UserWarning,
            stacklevel=3,
        )
    from ..output._lineage import attach_provenance as _attach_prov

    _attach_prov(
        res,
        function="sp.survival.causal_survival_forest",
        params={
            "time": time if isinstance(time, str) else "<array>",
            "event": event if isinstance(event, str) else "<array>",
            "treat": treat if isinstance(treat, str) else "<array>",
            "covariates": list(names["covariates"]),
            "horizon": float(horizon),
            "target": tgt,
            "n_estimators": int(opts.n_estimators),
            "min_samples_leaf": int(opts.min_samples_leaf),
            "propensity_bounds": None if bounds is None else list(bounds),
            "random_state": random_state,
            "alpha": alpha,
        },
        data=data,
        overwrite=False,
    )
    return res


# Backward-compatible alias matching grf naming.
causal_survival = causal_survival_forest


__all__ = [
    "causal_survival_forest",
    "causal_survival",
    "CausalSurvivalForestResult",
    "csf_scores",
    "csf_dr_scores",
]
