"""Design analysis: power, sign errors and exaggeration (Gelman and Carlin)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import integrate, stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import MethodIncompatibility


@dataclass
class RetrodesignResult(ResultProtocolMixin):
    """Result of :func:`retrodesign`.

    Attributes
    ----------
    table : pd.DataFrame
        One row per hypothesised effect: ``effect``, ``se``, ``power``,
        ``type_s`` and ``type_m``.
    power : float
        Probability that the estimate is statistically significant
        (first row).
    type_s : float
        Probability that a significant estimate has the wrong sign.
    type_m : float
        Exaggeration ratio: expected absolute value of a significant
        estimate divided by the true effect.
    alpha, dof : float

    Examples
    --------
    >>> import statspai as sp
    >>> out = sp.retrodesign(0.1, 3.28)
    >>> round(out.power, 3), round(out.type_s, 2), round(out.type_m)
    (0.05, 0.46, 77)
    """

    table: pd.DataFrame
    power: float
    type_s: float
    type_m: float
    alpha: float
    dof: float
    notes: list = field(default_factory=list)

    _citation_keys = ("gelman2014beyond",)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "power": self.power,
            "type_s": self.type_s,
            "type_m": self.type_m,
            "alpha": self.alpha,
            "dof": None if np.isinf(self.dof) else self.dof,
            "table": self.table.to_dict(orient="records"),
        }

    def summary(self) -> str:
        lines = [
            "Design analysis (Gelman and Carlin)",
            f"alpha = {self.alpha:g}"
            + ("" if np.isinf(self.dof) else f"    degrees of freedom = {self.dof:g}"),
            "",
            self.table.to_string(index=False, float_format=lambda v: f"{v:.4g}"),
            "",
            "power:  probability the estimate is statistically significant",
            "type_s: probability a significant estimate has the wrong sign",
            "type_m: expected |significant estimate| / true effect",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()


def _upper_partial_mean(c: np.ndarray, dof: float) -> np.ndarray:
    """``E[T 1(T > c)]`` for a standard normal or Student-t variable."""
    if np.isinf(dof):
        return np.asarray(stats.norm.pdf(c))
    return np.asarray(dof / (dof - 1.0) * (1.0 + c**2 / dof) * stats.t.pdf(c, dof))


def _exact_abs_mean(mu: float, crit: float, dof: float) -> float:
    """``E[|Z + mu| 1(|Z + mu| > crit * U)]`` with ``U = sqrt(chi2_dof / dof)``.

    The numerator of the exaggeration ratio of a t-test in units of the
    true standard error: the known-variance expression at critical value
    ``crit * u``, averaged over the estimated-to-true ratio ``u``.
    """
    root = np.sqrt(dof)

    def integrand(u: float) -> float:
        c = crit * u
        inner = (
            mu * (stats.norm.sf(c - mu) - stats.norm.cdf(-c - mu))
            + stats.norm.pdf(c - mu)
            + stats.norm.pdf(c + mu)
        )
        return float(inner * stats.chi.pdf(u * root, dof) * root)

    # U concentrates around one with standard deviation near
    # 1 / sqrt(2 dof); integrate over twelve of them on each side
    half = 12.0 / np.sqrt(2.0 * dof)
    value, _ = integrate.quad(
        integrand,
        max(0.0, 1.0 - half),
        1.0 + half,
        points=[1.0],
        epsabs=0.0,
        epsrel=1e-10,
        limit=200,
    )
    return float(value)


def retrodesign(
    effect: Any,
    se: Any,
    alpha: float = 0.05,
    dof: Optional[float] = None,
    method: str = "exact",
) -> RetrodesignResult:
    """Power, sign-error rate and exaggeration ratio of a design.

    For a hypothesised true effect and the standard error a study
    delivers, answers three questions about its statistically
    significant results: how often there is one (power), how often it
    points the wrong way (type S error) and by what factor it overstates
    the truth on average (type M error, the exaggeration ratio). With
    low power, significance selects the estimates that happen to be far
    from zero, so a published significant estimate is biased away from
    zero and may well have the wrong sign.

    Parameters
    ----------
    effect : float or array-like
        Hypothesised true effect, from outside information (earlier
        literature, a plausible range), never the estimate of the study
        under review. Several values give one row each.
    se : float or array-like
        Standard error of the estimate.
    alpha : float, default 0.05
        Two-sided significance level.
    dof : float, optional
        Degrees of freedom of the standard error, when it is estimated
        and the test is a t-test. Omitted: the standard error is known
        and the test is a z-test.
    method : {'exact', 'shifted'}, default 'exact'
        Only matters with ``dof``. ``'exact'``: the estimate is normal
        around the true effect, its estimated standard error is
        independent of it with ``dof`` degrees of freedom, and the test
        compares their ratio with the t critical value, so the test
        statistic is noncentral t. ``'shifted'``: the estimate is
        ``effect + se * t_dof`` and is compared with ``se`` times the
        critical value, the approximation in the code published with
        Gelman and Carlin (2014).

    Returns
    -------
    RetrodesignResult

    Notes
    -----
    Nothing is simulated. With ``mu = effect / se`` and ``z`` the
    critical value, the known-variance case is ``power = P(Z > z - mu) +
    P(Z < -z - mu)``, ``type_s`` the smaller term over the sum, and the
    exaggeration ratio follows from ``E[Z 1(Z > c)] = pdf(c)``. With
    ``dof`` and ``method='exact'`` power and sign error are noncentral-t
    probabilities and the exaggeration ratio integrates the
    known-variance expression over the distribution of the estimated
    standard error. With ``method='shifted'`` the partial expectation
    of Student's t, ``dof / (dof - 1) (1 + c^2 / dof) pdf(c)``, gives a
    closed form.

    The two methods differ noticeably at low power and few degrees of
    freedom (effect 0.5, se 1, 20 degrees of freedom: exaggeration 4.63
    exact, 5.18 shifted). A simulation of the t-test itself reproduces
    the exact values.

    Examples
    --------
    >>> import statspai as sp
    >>> out = sp.retrodesign([0.5, 2.8], se=1.0)
    >>> [round(v, 2) for v in out.table["power"]]
    [0.08, 0.8]
    >>> round(float(out.table["type_m"].iloc[1]), 2)
    1.13

    References
    ----------
    gelman2014beyond
    """
    a = np.atleast_1d(np.asarray(effect, dtype=float))
    s = np.atleast_1d(np.asarray(se, dtype=float))
    try:
        a, s = np.broadcast_arrays(a, s)
    except ValueError as exc:
        raise MethodIncompatibility(
            "effect and se must have the same length, or one of them be a "
            "single number."
        ) from exc
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(s)) or np.any(s <= 0):
        raise MethodIncompatibility("effect must be finite and se positive.")
    if np.any(a == 0):
        raise MethodIncompatibility(
            "A true effect of exactly zero has no sign to get wrong and no "
            "size to exaggerate; every significant result is then a false "
            "positive, with probability alpha."
        )
    if not 0.0 < alpha < 1.0:
        raise MethodIncompatibility(f"alpha must be in (0, 1); got {alpha}.")
    nu = np.inf if dof is None else float(dof)
    if not nu > 1:
        raise MethodIncompatibility(
            "dof must exceed 1 for the exaggeration ratio to be finite."
        )
    if method not in ("exact", "shifted"):
        raise MethodIncompatibility("method must be 'exact' or 'shifted'.")
    # past a million degrees of freedom the t-test is the z-test to seven
    # digits, and the quadrature below has nothing left to resolve
    known = np.isinf(nu) or (method == "exact" and nu >= 1e6)
    dist = stats.norm if known else stats.t(nu)
    z = float(dist.ppf(1.0 - alpha / 2.0))
    mu = np.abs(a) / s
    if known:
        p_hi = dist.sf(z - mu)
        p_lo = dist.cdf(-z - mu)
        partial = stats.norm.pdf(z - mu) + stats.norm.pdf(z + mu)
        numerator = mu * (p_hi - p_lo) + partial
    elif method == "shifted":
        p_hi = dist.sf(z - mu)
        p_lo = dist.cdf(-z - mu)
        partial = _upper_partial_mean(z - mu, nu) + _upper_partial_mean(z + mu, nu)
        numerator = mu * (p_hi - p_lo) + partial
    else:
        p_hi = stats.nct.sf(z, nu, mu)
        p_lo = stats.nct.cdf(-z, nu, mu)
        numerator = np.array([_exact_abs_mean(float(m), z, nu) for m in mu])
    power = p_hi + p_lo
    type_s = p_lo / power
    type_m = numerator / (power * mu)
    table = pd.DataFrame(
        {"effect": a, "se": s, "power": power, "type_s": type_s, "type_m": type_m}
    )
    return RetrodesignResult(
        table=table,
        power=float(power[0]),
        type_s=float(type_s[0]),
        type_m=float(type_m[0]),
        alpha=float(alpha),
        dof=nu,
    )


__all__ = ["RetrodesignResult", "retrodesign"]
