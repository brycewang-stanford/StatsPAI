"""Covariate-balancing propensity score (CBPS) for a binary treatment.

The propensity score is a logistic model ``pi(X; beta) = expit(X'beta)``
whose coefficients are chosen so that the inverse-probability weights it
implies balance the covariate means, optionally together with the logit
score equation [@imai2014covariate]. With ``n`` units, ``n1`` of them
treated, the sample moments of Imai and Ratkovic (2014) are

    score   (eq. 7):   s(beta) = X'(T - pi) / n
    balance (eq. 10):  b(beta) = X'w / n,   w = T/pi - (1 - T)/(1 - pi)
    balance (eq. 11):  b(beta) = X'w / n,   w = (n/n1) (T - pi)/(1 - pi)

(eq. 10 for the ATE, eq. 11 for the ATT). ``variant='exact'`` is the
just-identified CBPS, ``b(beta) = 0``; ``variant='over'`` stacks
``g = [s; b]`` and minimises the GMM criterion ``g' Sigma^{-1} g`` with the
model-implied moment covariance ``Sigma`` of eq. (16) (ATE) or (17) (ATT).

Two observations organise the computation:

* The just-identified moments are the gradient of a strictly concave
  function of ``beta`` (for the ATE ``sum_T (eta - e^{-eta}) -
  sum_C (eta + e^{eta})``, for the ATT ``sum_T eta - sum_C e^{eta}``, with
  ``eta = X'beta``), so the solution is unique when it exists and Newton's
  method with a line search on that function finds it.
* The over-identified fit is the two-step estimator (``CBPS(twostep =
  TRUE)``, the reference's default): ``Sigma`` is evaluated once, at
  ``a * beta_MLE`` with ``a`` the minimiser over ``[0.8, 1.1]`` of the
  continuously-updated criterion along that ray, and then held fixed. This
  convention was established from the reference's outputs (its reported
  ``J`` and ``mle.J`` are reproduced by exactly this ``Sigma``, and the
  scale sits on 0.8 or 1.1 when the unconstrained minimiser lies outside);
  see ``docs/dev/cleanroom/cbps.md``.

Both criteria are invariant to a non-singular linear reparametrisation of
the covariates, so they are solved on centred and scaled covariates and
the coefficients mapped back.

The effect is the Hajek (self-normalised) inverse-probability-weighted
contrast; its standard error is the standard deviation of the estimate
over nonparametric bootstrap refits.

Clean-room implementation written from Imai and Ratkovic (2014); the
reference (R ``CBPS`` 0.24, GPL >= 2) was used only as a black box.

References
----------
[@imai2014covariate]
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize
from scipy import stats as sp_stats

from ..core._covariates import expands_categorical_covariates as _expands_categorical
from ..core._validate import require_binary_treatment
from ..core._validate import treatment_as_float as _treatment_as_float
from ..core.results import CausalResult
from ..exceptions import (
    ColumnNotFound,
    ConvergenceWarning,
    DataInsufficient,
    MethodIncompatibility,
)

# Propensity scores used for weighting are kept inside [floor, 1 - floor].
_PS_FLOOR = 1e-6
# Search interval for the scale of the two-step weighting point.
_SCALE_BOUNDS = (0.8, 1.1)
# Linear index clip that only guards exp() against overflow.
_ETA_CLIP = 35.0
_RANK_TOL = 1e-9


@_expands_categorical("covariates")
def cbps(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    estimand: Literal["ATE", "ATT"] = "ATE",
    variant: Literal["exact", "over"] = "over",
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    seed: Optional[int] = None,
    add_intercept: bool = True,
    trim: float = 0.0,
) -> CausalResult:
    """Covariate-Balancing Propensity Score estimator (Imai-Ratkovic 2014).

    Fits a logistic propensity score whose inverse-probability weights
    balance the covariate means and reports the Hajek-weighted treatment
    effect.

    Parameters
    ----------
    data : pandas.DataFrame
        Analysis data. Rows with a missing outcome, treatment or covariate
        are dropped.
    y : str
        Outcome column.
    treat : str
        Treatment column, coded 0/1.
    covariates : list of str
        Covariates of the propensity model; they are also the balanced
        moments. Categorical columns are expanded to indicators.
    estimand : {'ATE', 'ATT'}, default 'ATE'
        ``'ATE'`` balances both arms towards the pooled sample (eq. 10 of
        Imai and Ratkovic 2014); ``'ATT'`` reweights the controls towards
        the treated (eq. 11).
    variant : {'exact', 'over'}, default 'over'
        ``'exact'``: just-identified CBPS, the balance conditions alone
        (exact mean balance when a solution exists). ``'over'``:
        over-identified CBPS, balance conditions plus the logit score,
        two-step GMM with the model-implied moment covariance.
    n_bootstrap : int, default 500
        Nonparametric bootstrap replications for the standard error (the
        whole CBPS fit is repeated on each resample). With fewer than two
        replications the standard error, p-value and interval are NaN.
    alpha : float, default 0.05
        Level of the normal-approximation confidence interval.
    seed : int, optional
        Seed of the bootstrap resampling.
    add_intercept : bool, default True
        Add a constant to the propensity model. With ``False`` the
        covariates must contain exactly one constant column.
    trim : float, default 0.0
        When positive, propensity scores are clipped to
        ``[trim, 1 - trim]`` before the weights are formed (units are not
        dropped). Scores are always kept inside ``[1e-6, 1 - 1e-6]``.

    Returns
    -------
    CausalResult
        ``estimate`` is the Hajek IPW contrast. ``model_info`` holds the
        propensity coefficients (``beta``, in the order intercept then
        covariates), ``std_mean_diff_after`` (weighted mean difference over
        the pooled within-arm standard deviation, per design column),
        ``gmm_objective``, ``converged``, the score range and the
        bootstrap counts.

    Raises
    ------
    MethodIncompatibility
        Unknown ``estimand`` / ``variant``, a treatment not coded 0/1, a
        design with no or several constant columns, or a covariate that is
        a linear combination of the others.
    DataInsufficient
        One treatment arm is empty after dropping missing rows.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=400)
    >>> d = rng.binomial(1, 1 / (1 + np.exp(-0.5 * x)))
    >>> df = pd.DataFrame({"y": 1 + 2 * d + x + rng.normal(size=400),
    ...                    "d": d, "x": x})
    >>> res = sp.cbps(df, y="y", treat="d", covariates=["x"],
    ...               variant="exact", n_bootstrap=50, seed=1)
    >>> abs(res.model_info["std_mean_diff_after"]["x"]) < 1e-8
    True

    References
    ----------
    [@imai2014covariate]

    Clean-room implementation; the reference (R ``CBPS`` 0.24, GPL >= 2)
    was used only as a black box.
    """
    if estimand not in ("ATE", "ATT"):
        raise MethodIncompatibility(
            f"cbps: estimand must be 'ATE' or 'ATT'; got {estimand!r}",
            recovery_hint="Pass estimand='ATE' or estimand='ATT'.",
        )
    if variant not in ("exact", "over"):
        raise MethodIncompatibility(
            f"cbps: variant must be 'exact' or 'over'; got {variant!r}",
            recovery_hint=(
                "Pass variant='exact' (balance conditions only) or "
                "variant='over' (balance conditions plus the logit score)."
            ),
        )
    covariates = list(covariates)
    missing = [c for c in [y, treat] + covariates if c not in data.columns]
    if missing:
        raise ColumnNotFound(
            f"cbps: column(s) {missing} not found in data",
            recovery_hint="Check the outcome, treatment and covariate names.",
        )

    frame = data[[y, treat] + covariates].dropna()
    T = _treatment_as_float(frame[treat], function="cbps", argument=treat)
    require_binary_treatment(T, function="cbps", argument=treat)
    n_treated = int(T.sum())
    n_control = int(T.size - n_treated)
    if n_treated == 0 or n_control == 0:
        raise DataInsufficient(
            f"cbps: {treat} has {n_treated} treated and {n_control} control "
            "rows after dropping missing values; both arms are needed",
            recovery_hint=(
                "Check the treatment coding and the rows dropped for missing "
                "outcome or covariate values."
            ),
        )
    Y = frame[y].to_numpy(dtype=float)
    X, names = _design(frame, covariates, add_intercept)
    att = estimand == "ATT"

    fit = _fit_cbps(X, T, att, variant)
    if not fit["converged"]:
        warnings.warn(
            f"cbps: the {variant} CBPS fit did not converge; the "
            "coefficients are the last iterate. "
            + (
                "The balance equations may have no solution (little overlap)."
                if variant == "exact"
                else "Check overlap and covariate scaling."
            ),
            ConvergenceWarning,
            stacklevel=2,
        )
    beta = fit["beta"]
    ps = _scores(X @ beta, trim)
    if np.any(ps <= _PS_FLOOR * (1 + 1e-9)) or np.any(ps >= 1 - _PS_FLOOR * (1 + 1e-9)):
        warnings.warn(
            "cbps: some propensity scores reached the 1e-6 bound "
            "(near-separation); the weights of those units are extreme.",
            ConvergenceWarning,
            stacklevel=2,
        )
    w1, w0 = _arm_weights(T, ps, att)
    estimate = _hajek(Y, w1, w0)
    smd = {name: _std_mean_diff(X[:, j], T, w1, w0) for j, name in enumerate(names)}

    boot = _bootstrap(Y, T, X, att, variant, trim, n_bootstrap, seed)
    se = float(np.std(boot["draws"], ddof=1)) if len(boot["draws"]) >= 2 else np.nan
    if np.isfinite(se) and se > 0:
        z = sp_stats.norm.ppf(1 - alpha / 2)
        ci = (estimate - z * se, estimate + z * se)
        pvalue = float(2 * sp_stats.norm.sf(abs(estimate) / se))
    else:
        ci, pvalue = (np.nan, np.nan), np.nan
    if boot["retries"]:
        warnings.warn(
            f"cbps: {boot['retries']} bootstrap resample(s) could not be fitted "
            f"({boot['nonconverged']} did not converge) and were redrawn; "
            f"{len(boot['draws'])} of {n_bootstrap} replications were used.",
            ConvergenceWarning,
            stacklevel=2,
        )

    model_info = {
        "model_type": f"CBPS ({variant})",
        "estimand": estimand,
        "beta": beta,
        "n_treated": n_treated,
        "n_control": n_control,
        "pscore_min": float(ps.min()),
        "pscore_max": float(ps.max()),
        "std_mean_diff_after": smd,
        "converged": bool(fit["converged"]),
        "gmm_objective": float(fit["objective"]),
        "n_bootstrap": int(n_bootstrap),
        "n_bootstrap_success": len(boot["draws"]),
        "n_bootstrap_nonconverged": int(boot["nonconverged"]),
        "n_bootstrap_retries": int(boot["retries"]),
    }
    if variant == "over":
        model_info["weighting_point_scale"] = float(fit["scale"])
    return CausalResult(
        method=f"CBPS ({variant}, {estimand})",
        estimand=estimand,
        estimate=float(estimate),
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=int(T.size),
        detail=None,
        model_info=model_info,
    )


# ======================================================================
# Design matrix
# ======================================================================


def _design(
    frame: pd.DataFrame, covariates: List[str], add_intercept: bool
) -> Tuple[np.ndarray, List[str]]:
    """Propensity design matrix and its column names.

    Exactly one column must be constant: the added intercept, or (with
    ``add_intercept=False``) one supplied by the caller.
    """
    Xc = frame[covariates].to_numpy(dtype=float).reshape(len(frame), -1)
    if add_intercept:
        X = np.column_stack([np.ones(len(frame)), Xc])
        names = ["_intercept"] + covariates
    else:
        X, names = Xc, list(covariates)
    constant = [j for j in range(X.shape[1]) if np.ptp(X[:, j]) == 0]
    if len(constant) != 1:
        raise MethodIncompatibility(
            f"cbps: the propensity design must contain exactly one constant "
            f"column; found {len(constant)} "
            f"({[names[j] for j in constant]})",
            recovery_hint=(
                "With add_intercept=True drop constant covariates; with "
                "add_intercept=False include exactly one constant column."
            ),
        )
    if X.shape[0] <= X.shape[1]:
        raise DataInsufficient(
            f"cbps: {X.shape[0]} rows for {X.shape[1]} propensity coefficients",
            recovery_hint="Use fewer covariates or more data.",
        )
    _, sv, _ = np.linalg.svd(_Standardiser(X).Z, full_matrices=False)
    if sv[-1] <= _RANK_TOL * sv[0]:
        raise MethodIncompatibility(
            "cbps: the propensity design is not full rank: a covariate is a "
            f"linear combination of the others (singular values of the "
            f"standardised design from {sv[0]:.3g} down to {sv[-1]:.3g})",
            recovery_hint=(
                "Drop the redundant covariate: the balance conditions cannot "
                "tell collinear columns apart."
            ),
        )
    return X, names


class _Standardiser:
    """Centre and scale the non-constant columns of ``X`` (sample sd).

    ``Z gamma = X beta`` with ``beta = to_original(gamma)``. The CBPS
    criteria are invariant to this reparametrisation; it only conditions
    the numerical problem (incomes in tens of thousands next to binaries).
    """

    def __init__(self, X: np.ndarray):
        self.const = np.array([np.ptp(X[:, j]) == 0 for j in range(X.shape[1])])
        self.mean = np.where(self.const, 0.0, X.mean(axis=0))
        sd = X.std(axis=0, ddof=1) if X.shape[0] > 1 else np.ones(X.shape[1])
        self.scale = np.where(self.const, 1.0, sd)
        self.Z = (X - self.mean) / self.scale
        self.const_value = X[0, self.const] if self.const.any() else None

    def to_original(self, gamma: np.ndarray) -> np.ndarray:
        beta = gamma / self.scale
        if self.const.any():
            j = int(np.flatnonzero(self.const)[0])
            shift = float(
                np.sum(
                    gamma[~self.const]
                    * self.mean[~self.const]
                    / self.scale[~self.const]
                )
            )
            cv = float(np.asarray(self.const_value)[0])
            beta[j] = (gamma[j] * cv - shift) / cv
        out: np.ndarray = beta
        return out


# ======================================================================
# Moments (Imai & Ratkovic 2014, eqs. 7, 10, 11, 16, 17)
# ======================================================================


def _expit(eta: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * np.clip(eta, -_ETA_CLIP, _ETA_CLIP)))


def _score_moments(
    gamma: np.ndarray, Z: np.ndarray, T: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Logit score ``Z'(T - pi)/n`` and its Jacobian."""
    n = T.size
    p = _expit(Z @ gamma)
    jac = -(Z * (p * (1 - p))[:, None]).T @ Z / n
    return Z.T @ (T - p) / n, jac


def _balance_moments(
    gamma: np.ndarray, Z: np.ndarray, T: np.ndarray, att: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """Balance moments ``Z'w/n`` (eq. 10 or 11) and their Jacobian."""
    n = T.size
    eta = np.clip(Z @ gamma, -_ETA_CLIP, _ETA_CLIP)
    if att:
        r = n / T.sum()
        odds = np.exp(eta)
        w = r * (T - (1 - T) * odds)  # (n/n1)(T - pi)/(1 - pi)
        dw = -r * (1 - T) * odds
    else:
        e_neg, e_pos = np.exp(-eta), np.exp(eta)
        w = T * (1 + e_neg) - (1 - T) * (1 + e_pos)  # T/pi - (1-T)/(1-pi)
        dw = -T * e_neg - (1 - T) * e_pos
    return Z.T @ w / n, (Z * dw[:, None]).T @ Z / n


def _moment_covariance(
    gamma: np.ndarray, Z: np.ndarray, T: np.ndarray, att: bool
) -> np.ndarray:
    """Model-implied covariance of ``[score; balance]`` (eq. 16 / 17)."""
    n = T.size
    p = _expit(Z @ gamma)
    v = p * (1 - p)

    def cross(a: np.ndarray) -> np.ndarray:
        out: np.ndarray = (Z * a[:, None]).T @ Z / n
        return out

    if att:
        r = n / T.sum()
        s12, s22 = r * cross(p), r * r * cross(p / (1 - p))
    else:
        s12, s22 = cross(np.ones(n)), cross(1 / v)
    return np.block([[cross(v), s12], [s12.T, s22]])


def _whitener(S: np.ndarray) -> np.ndarray:
    """``W`` with ``W'W = S^+`` (pseudo-inverse on the positive eigenspace)."""
    lam, vec = np.linalg.eigh(0.5 * (S + S.T))
    keep = lam > 1e-12 * max(lam.max(), 1e-300)
    W: np.ndarray = (vec[:, keep] / np.sqrt(lam[keep])).T
    return W


def _stacked(
    gamma: np.ndarray, Z: np.ndarray, T: np.ndarray, att: bool
) -> Tuple[np.ndarray, np.ndarray]:
    s, ds = _score_moments(gamma, Z, T)
    b, db = _balance_moments(gamma, Z, T, att)
    return np.concatenate([s, b]), np.vstack([ds, db])


def _cue_criterion(gamma: np.ndarray, Z: np.ndarray, T: np.ndarray, att: bool) -> float:
    """Continuously-updated criterion ``g' Sigma(gamma)^+ g``."""
    g, _ = _stacked(gamma, Z, T, att)
    r = _whitener(_moment_covariance(gamma, Z, T, att)) @ g
    return float(r @ r)


# ======================================================================
# Solvers
# ======================================================================


def _logit_mle(Z: np.ndarray, T: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """Logit maximum likelihood by Newton's method with step halving."""
    gamma = np.zeros(Z.shape[1])

    def loglik(g: np.ndarray) -> float:
        eta = np.clip(Z @ g, -_ETA_CLIP, _ETA_CLIP)
        return float(T @ eta - np.logaddexp(0.0, eta).sum())

    ll = loglik(gamma)
    for _ in range(max_iter):
        score, jac = _score_moments(gamma, Z, T)
        try:
            step = -np.linalg.solve(jac, score)
        except np.linalg.LinAlgError:
            break
        t = 1.0
        while t > 1e-10:
            cand = gamma + t * step
            ll_new = loglik(cand)
            if ll_new >= ll - 1e-12 * abs(ll):
                break
            t *= 0.5
        gamma, ll = cand, ll_new
        if np.max(np.abs(t * step)) < 1e-12 * (1 + np.max(np.abs(gamma))):
            break
    return gamma


def _solve_exact(
    Z: np.ndarray, T: np.ndarray, att: bool, start: np.ndarray, max_iter: int = 200
) -> Tuple[np.ndarray, bool]:
    """Just-identified CBPS: maximise the concave potential whose gradient
    is the balance moment vector. Returns ``(gamma, converged)``."""
    n = T.size
    weight = n / T.sum() if att else 1.0

    def potential(g: np.ndarray) -> float:
        eta = np.clip(Z @ g, -_ETA_CLIP, _ETA_CLIP)
        if att:
            val = eta[T == 1].sum() - np.exp(eta[T == 0]).sum()
        else:
            val = np.sum(eta[T == 1] - np.exp(-eta[T == 1])) - np.sum(
                eta[T == 0] + np.exp(eta[T == 0])
            )
        return float(weight * val / n)

    gamma = np.asarray(start, dtype=float).copy()
    f = potential(gamma)
    for _ in range(max_iter):
        g, H = _balance_moments(gamma, Z, T, att)
        if np.max(np.abs(g)) < 1e-14:
            break
        try:
            step = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        slope = float(g @ step)
        if slope <= 0:  # not an ascent direction: Hessian lost definiteness
            break
        t = 1.0
        while t > 1e-12:
            cand = gamma + t * step
            f_new = potential(cand)
            if f_new >= f + 1e-4 * t * slope:
                break
            t *= 0.5
        else:
            break
        gamma, f = cand, f_new
        if t * np.max(np.abs(step)) < 1e-15 * (1 + np.max(np.abs(gamma))):
            break
    g, _ = _balance_moments(gamma, Z, T, att)
    converged = bool(np.all(np.isfinite(gamma)) and np.max(np.abs(g)) < 1e-8)
    return gamma, converged


def _two_step_scale(
    Z: np.ndarray, T: np.ndarray, att: bool, gamma_mle: np.ndarray
) -> float:
    """Scale ``a`` in ``[0.8, 1.1]`` minimising the CUE criterion at
    ``a * gamma_mle``; ``Sigma`` of the two-step fit is evaluated there."""
    res = optimize.minimize_scalar(
        lambda a: _cue_criterion(a * gamma_mle, Z, T, att),
        bounds=_SCALE_BOUNDS,
        method="bounded",
        options={"xatol": 1e-10},
    )
    return float(res.x)


def _solve_over(
    Z: np.ndarray,
    T: np.ndarray,
    att: bool,
    starts: List[np.ndarray],
    weight_point: np.ndarray,
) -> Tuple[np.ndarray, float, bool]:
    """Two-step over-identified CBPS: minimise ``||W g(gamma)||^2`` with
    ``W'W = Sigma(weight_point)^+`` from each start; keep the best."""
    W = _whitener(_moment_covariance(weight_point, Z, T, att))

    def resid(g: np.ndarray) -> np.ndarray:
        r: np.ndarray = W @ _stacked(g, Z, T, att)[0]
        return r

    def jac(g: np.ndarray) -> np.ndarray:
        J: np.ndarray = W @ _stacked(g, Z, T, att)[1]
        return J

    best: Optional[Tuple[np.ndarray, float, bool]] = None
    for s in starts:
        try:
            r = optimize.least_squares(
                resid,
                s,
                jac=jac,
                method="lm",
                xtol=1e-15,
                ftol=1e-15,
                gtol=1e-15,
                max_nfev=2000,
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        q = float(r.fun @ r.fun)
        if np.isfinite(q) and (best is None or q < best[1]):
            best = (r.x, q, r.status > 0)
    if best is None:
        return np.asarray(starts[0], dtype=float), np.nan, False
    return best


def _fit_cbps(X: np.ndarray, T: np.ndarray, att: bool, variant: str) -> Dict[str, Any]:
    """CBPS coefficients (original basis), objective and convergence."""
    std = _Standardiser(X)
    Z = std.Z
    gamma_mle = _logit_mle(Z, T)
    gamma_exact, exact_ok = _solve_exact(Z, T, att, gamma_mle)
    if variant == "exact":
        g, _ = _balance_moments(gamma_exact, Z, T, att)
        S = _moment_covariance(gamma_exact, Z, T, att)[Z.shape[1] :, Z.shape[1] :]
        r = _whitener(S) @ g
        return {
            "beta": std.to_original(gamma_exact),
            "objective": float(r @ r),
            "converged": exact_ok,
            "scale": np.nan,
        }
    scale = _two_step_scale(Z, T, att, gamma_mle)
    starts = [scale * gamma_mle, gamma_mle]
    if exact_ok:
        starts.append(gamma_exact)
    gamma, q, ok = _solve_over(Z, T, att, starts, scale * gamma_mle)
    return {
        "beta": std.to_original(gamma),
        "objective": q,
        "converged": bool(ok),
        "scale": scale,
    }


# ======================================================================
# Weights, effect, balance, bootstrap
# ======================================================================


def _scores(eta: np.ndarray, trim: float) -> np.ndarray:
    lo = max(_PS_FLOOR, float(trim))
    return np.clip(_expit(eta), lo, 1 - lo)


def _arm_weights(
    T: np.ndarray, ps: np.ndarray, att: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """Unnormalised arm weights: ATE ``T/pi`` and ``(1-T)/(1-pi)``; ATT
    ``T`` and ``(1-T) pi/(1-pi)``."""
    if att:
        return T.copy(), (1 - T) * ps / (1 - ps)
    return T / ps, (1 - T) / (1 - ps)


def _hajek(Y: np.ndarray, w1: np.ndarray, w0: np.ndarray) -> float:
    return float(w1 @ Y / w1.sum() - w0 @ Y / w0.sum())


def _std_mean_diff(
    x: np.ndarray, T: np.ndarray, w1: np.ndarray, w0: np.ndarray
) -> float:
    """Weighted treated-minus-control mean over the pooled (unweighted,
    divisor n) within-arm standard deviation; 0 for a constant column."""
    if np.ptp(x) == 0:
        return 0.0
    diff = w1 @ x / w1.sum() - w0 @ x / w0.sum()
    pooled = np.sqrt(0.5 * (x[T == 1].var() + x[T == 0].var()))
    return float(diff / pooled) if pooled > 0 else float("nan")


def _bootstrap(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    att: bool,
    variant: str,
    trim: float,
    n_bootstrap: int,
    seed: Optional[int],
) -> Dict[str, Any]:
    """Refit CBPS on ``n_bootstrap`` resamples of the rows.

    A resample that cannot be fitted (one arm empty, rank-deficient design)
    or whose fit does not converge is redrawn, at most ``n_bootstrap + 10``
    times in total; the counts are reported.
    """
    draws: List[float] = []
    retries = nonconverged = 0
    if n_bootstrap < 1:
        return {"draws": draws, "retries": 0, "nonconverged": 0}
    rng = np.random.default_rng(seed)
    n = T.size
    budget = n_bootstrap + 10
    while len(draws) < n_bootstrap and retries <= budget:
        idx = rng.integers(0, n, size=n)
        Tb, Xb = T[idx], X[idx]
        n1 = Tb.sum()
        try:
            if n1 == 0 or n1 == n:
                raise ValueError("one arm empty")
            std = _Standardiser(Xb)
            if std.const.sum() != 1:
                raise ValueError("constant column lost")
            sv = np.linalg.svd(std.Z, compute_uv=False)
            if sv[-1] <= _RANK_TOL * sv[0]:
                raise ValueError("rank deficient")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = _fit_cbps(Xb, Tb, att, variant)
        except (ValueError, np.linalg.LinAlgError):
            retries += 1
            continue
        if not fit["converged"]:
            retries += 1
            nonconverged += 1
            continue
        ps = _scores(Xb @ fit["beta"], trim)
        w1, w0 = _arm_weights(Tb, ps, att)
        draws.append(_hajek(Y[idx], w1, w0))
    return {"draws": draws, "retries": retries, "nonconverged": nonconverged}


__all__ = ["cbps"]
