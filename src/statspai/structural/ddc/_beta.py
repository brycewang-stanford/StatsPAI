"""The discount factor of ``sp.ddc``: a profile likelihood over ``beta``.

For each ``beta`` the utility parameters (and, with ``stage='full'``, the
increment probabilities) are re-estimated; the profile is the maximised
log-likelihood as a function of ``beta``. ``beta='estimate'`` maximises it,
which is the joint maximum likelihood estimate of ``beta`` and the other
parameters. The search runs on ``tau = -log(1 - beta)``, so the region near
one, where Rust's estimates sit, gets as much resolution as the rest.

Rust (1987, p. 1023) could not estimate ``beta`` precisely: it is nearly
collinear with the replacement cost, the information matrix is close to
singular, and the estimate tended to one. The interval reported here is
therefore the likelihood-ratio interval read off the profile, not a Wald
interval, and the standard errors of the other parameters treat ``beta``
as known.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import numpy as np
from scipy import optimize, stats

from ...exceptions import AssumptionWarning, MethodIncompatibility

__all__ = ["beta_search", "objective"]

#: Upper end of the search for ``beta='estimate'`` (Rust's largest value).
BETA_MAX = 0.999999
_TAU_MAX = -np.log1p(-BETA_MAX)


def objective(fit: Any) -> float:
    """Profile objective: the full log-likelihood under stage='full'."""
    full = fit.diagnostics.get("loglik_full")
    return float(fit.loglik if full is None else full)


def _check(values: Sequence[float]) -> List[float]:
    out = [float(b) for b in values]
    if not out:
        raise MethodIncompatibility(
            "beta grid is empty", recovery_hint="pass at least one discount factor"
        )
    bad = [b for b in out if not 0.0 <= b < 1.0]
    if bad:
        raise MethodIncompatibility(
            f"discount factors outside [0, 1): {bad}",
            recovery_hint="every grid value must satisfy 0 <= beta < 1",
        )
    return out


def _attach(best: Any, fits: Dict[float, Any], how: str, extra: Dict[str, Any]) -> Any:
    grid = sorted(fits)
    names = list(best.params.index)
    best.diagnostics["beta_profile"] = {
        "beta": grid,
        "loglik": [objective(fits[b]) for b in grid],
        "converged": [bool(fits[b].converged) for b in grid],
        "params": {n: [float(fits[b].params[n]) for b in grid] for n in names},
    }
    best.diagnostics["beta_search"] = how
    best.diagnostics.update(extra)
    return best


def _lr_interval(
    h: Callable[[float], float], tau_hat: float, l_max: float, alpha: float
) -> Tuple[float, float]:
    """Likelihood-ratio interval for beta, clipped to [0, BETA_MAX]."""
    crit = 0.5 * float(stats.chi2.ppf(1 - alpha, 1))

    def drop(t: float) -> float:
        return float(l_max - h(t) - crit)

    lo, hi = 0.0, _TAU_MAX
    if tau_hat > 0 and drop(0.0) > 0:
        lo = optimize.brentq(drop, 0.0, tau_hat, xtol=1e-6)
    if tau_hat < _TAU_MAX and drop(_TAU_MAX) > 0:
        hi = optimize.brentq(drop, tau_hat, _TAU_MAX, xtol=1e-6)
    return float(-np.expm1(-lo)), float(-np.expm1(-hi))


def beta_search(
    fit_at: Callable[[float], Any],
    beta: Union[str, Sequence[float]],
    alpha: float,
) -> Any:
    """Fit over a grid of ``beta`` (or maximise over it) and keep the best."""
    fits: Dict[float, Any] = {}

    def fit(b: float) -> Any:
        if b not in fits:
            fits[b] = fit_at(b)
        return fits[b]

    if not isinstance(beta, str):
        for b in _check(beta):
            fit(b)
        best_b = max(fits, key=lambda b: objective(fits[b]))
        return _attach(fits[best_b], fits, "grid", {})

    if beta != "estimate":
        raise MethodIncompatibility(
            f"beta={beta!r} is not supported",
            recovery_hint="pass a number, a list of numbers, or 'estimate'",
        )

    def h(tau: float) -> float:
        return objective(fit(float(-np.expm1(-tau))))

    taus = np.linspace(0.0, _TAU_MAX, 13)
    vals = np.array([h(t) for t in taus])
    i = int(np.argmax(vals))
    a, b = taus[max(i - 1, 0)], taus[min(i + 1, taus.size - 1)]
    res = optimize.minimize_scalar(
        lambda t: -h(t), bounds=(a, b), method="bounded", options={"xatol": 1e-6}
    )
    tau_hat = float(res.x) if -res.fun >= vals[i] else float(taus[i])
    beta_hat = float(-np.expm1(-tau_hat))
    l_max = h(tau_hat)
    ci = _lr_interval(h, tau_hat, l_max, alpha)
    at_bound = tau_hat >= _TAU_MAX - 1e-3 or tau_hat <= 1e-3
    if at_bound:
        warnings.warn(
            AssumptionWarning(
                f"the profile likelihood is maximised at the boundary "
                f"beta = {beta_hat:.6g} of the search range [0, {BETA_MAX}]",
                recovery_hint=(
                    "beta is weakly identified (nearly collinear with the "
                    "utility constants); fix it from outside information and "
                    "report the profile in diagnostics['beta_profile']"
                ),
            ),
            stacklevel=3,
        )
    return _attach(
        fits[beta_hat],
        fits,
        "estimate",
        {
            "beta_hat": beta_hat,
            "beta_ci": list(ci),
            "beta_at_bound": bool(at_bound),
            "beta_loglik": l_max,
        },
    )
