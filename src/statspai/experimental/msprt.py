"""Always-valid inference for A/B tests: the mixture sequential probability ratio test.

The mixture SPRT of Johari, Koomen, Pekelis and Walsh (2022) lets an
experimenter look at the data after every observation and stop whenever the
p-value is small, without inflating the type I error. With the difference
in arm means ``theta_n`` and its variance ``v_n``, the likelihood ratio of
the Gaussian model integrated over a normal prior ``N(theta_0, tau^2)`` on
the effect is

    Lambda_n = sqrt(v_n / (v_n + tau^2))
               * exp(tau^2 (theta_n - theta_0)^2 / (2 v_n (v_n + tau^2))).

Under the null it is a nonnegative martingale with mean one, so by Ville's
inequality ``P(sup_n Lambda_n >= 1/alpha) <= alpha``. The always-valid
p-value is ``min(1, min_{m <= n} 1 / Lambda_m)`` and the confidence sequence
is the set of ``theta_0`` not rejected, intersected over time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .._input_validation import check_columns
from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["msprt", "MSPRTResult", "msprt_lambda"]


def msprt_lambda(estimate, variance, tau: float, theta0: float = 0.0):
    """Mixture likelihood ratio ``Lambda`` for estimates with known variances.

    Parameters
    ----------
    estimate, variance : float or array-like
        Estimate of the effect and its variance at each look.
    tau : float
        Standard deviation of the normal mixing distribution of the effect,
        in the units of the estimate.
    theta0 : float, default 0
        Null value.

    Returns
    -------
    numpy.ndarray or float
        ``Lambda`` at each look (computed on the log scale).

    Examples
    --------
    >>> import statspai as sp
    >>> round(float(sp.msprt_lambda(0.0, 1.0, tau=1.0)), 6)
    0.707107
    """
    est = np.asarray(estimate, dtype=float)
    v = np.asarray(variance, dtype=float)
    t2 = float(tau) ** 2
    log_lam = 0.5 * np.log(v / (v + t2)) + t2 * (est - theta0) ** 2 / (
        2.0 * v * (v + t2)
    )
    out = np.exp(np.minimum(log_lam, 700.0))
    return float(out) if out.ndim == 0 else out


@dataclass
class MSPRTResult(ResultProtocolMixin):
    """Result of :func:`msprt`.

    Attributes
    ----------
    path : pd.DataFrame
        One row per look: ``n``, ``n_treated``, ``n_control``, ``estimate``,
        ``se``, ``likelihood_ratio``, ``p_value`` (always valid, running
        minimum) and ``ci_lower`` / ``ci_upper`` (the confidence sequence,
        running intersection).
    p_value : float
        Always-valid p-value at the last look.
    ci : tuple
        Confidence sequence at the last look.
    rejected : bool
        Whether the null was rejected at some look.
    stop_look : int or None
        First look (row of ``path``) at which ``p_value <= alpha``.

    Examples
    --------
    >>> import numpy as np, pandas as pd, statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> d = rng.integers(0, 2, size=1000)
    >>> df = pd.DataFrame({"d": d, "y": 0.5 * d + rng.normal(size=1000)})
    >>> r = sp.msprt(df, "y", "d", tau=0.5, sigma=1.0)
    >>> r.rejected
    True
    """

    _citation_keys = ("johari2022always",)

    path: pd.DataFrame
    estimate: float
    p_value: float
    ci: tuple
    rejected: bool
    stop_look: Optional[int]
    alpha: float
    tau: float
    theta0: float
    variance: str
    detail: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        stop = "never" if self.stop_look is None else f"look {self.stop_look}"
        return (
            "Mixture SPRT (always-valid inference)\n"
            f"  looks = {len(self.path)}, tau = {self.tau:g}, "
            f"theta0 = {self.theta0:g}, variance: {self.variance}\n"
            f"  estimate = {self.estimate:.4f}, always-valid p = "
            f"{self.p_value:.4g}, {100 * (1 - self.alpha):.0f}% CS = "
            f"[{self.ci[0]:.4f}, {self.ci[1]:.4f}]\n"
            f"  first crossing of alpha = {self.alpha:g}: {stop}"
        )

    def __repr__(self) -> str:
        return (
            f"MSPRTResult(estimate={self.estimate:.4f}, p_value={self.p_value:.4g}, "
            f"rejected={self.rejected})"
        )


def msprt(
    data: pd.DataFrame,
    y: str,
    treat: str,
    *,
    tau: float,
    sigma: Optional[float] = None,
    time: Optional[str] = None,
    theta0: float = 0.0,
    alpha: float = 0.05,
    min_per_arm: int = 2,
) -> MSPRTResult:
    """Always-valid p-values and confidence sequences for a two-arm test.

    Parameters
    ----------
    data : pandas.DataFrame
        One row per observation, in arrival order (or sortable by
        ``time``).
    y : str
        Outcome.
    treat : str
        Treatment with exactly two values; the effect is the mean under the
        larger value minus the mean under the smaller.
    tau : float
        Standard deviation of the normal mixing distribution on the effect,
        in outcome units: roughly the size of effect the test should be
        most sensitive to. Validity does not depend on it; power does.
    sigma : float, optional
        Known outcome standard deviation, common to the arms. Then the test
        is exact for Gaussian outcomes. Default: running per-arm sample
        variances (the plug-in version), whose validity is asymptotic.
    time : str, optional
        Column giving the analysis time of each row; the test is evaluated
        once per distinct value (batched looks) instead of after every row.
    theta0 : float, default 0
        Null value of the effect.
    alpha : float, default 0.05
        Level of the test and one minus the coverage of the confidence
        sequence.
    min_per_arm : int, default 2
        Looks with fewer observations in an arm are skipped (the variance
        is undefined); at least 2 with the plug-in variance.

    Returns
    -------
    MSPRTResult

    Notes
    -----
    Looking at the p-value after every observation and stopping the first
    time it falls below ``alpha`` keeps the type I error at most ``alpha``;
    a fixed-horizon t-test used the same way does not. The price is power:
    at a pre-planned horizon a fixed test is more powerful.

    Examples
    --------
    >>> import numpy as np, pandas as pd, statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> d = rng.integers(0, 2, size=1000)
    >>> df = pd.DataFrame({"d": d, "y": rng.normal(size=1000)})
    >>> r = sp.msprt(df, "y", "d", tau=0.2, sigma=1.0)
    >>> r.p_value > 0.05
    True

    References
    ----------
    johari2022always
    """
    if not (np.isfinite(tau) and tau > 0):
        raise MethodIncompatibility(
            f"tau must be a positive number; got {tau!r}.",
            recovery_hint=(
                "tau is the prior standard deviation of the effect in outcome "
                "units, e.g. the effect size you hope to detect."
            ),
        )
    if not (0 < alpha < 1):
        raise MethodIncompatibility(
            f"alpha must be in (0, 1); got {alpha!r}.", recovery_hint="E.g. alpha=0.05."
        )
    if sigma is not None and not (np.isfinite(sigma) and sigma > 0):
        raise MethodIncompatibility(
            f"sigma must be a positive number; got {sigma!r}.",
            recovery_hint="Drop sigma= to use the running sample variances.",
        )
    min_arm = max(int(min_per_arm), 1 if sigma is not None else 2)
    check_columns(data, {"y": y, "treat": treat, "time": time}, function="sp.msprt")
    cols = [y, treat] + ([time] if time else [])
    frame = data[cols].dropna()
    if time:
        frame = frame.sort_values(time, kind="stable")
    frame = frame.reset_index(drop=True)
    levels = np.sort(frame[treat].unique())
    if len(levels) != 2:
        raise MethodIncompatibility(
            f"sp.msprt: {treat!r} must take exactly two values; found {len(levels)}.",
            recovery_hint="Compare two arms at a time.",
        )
    d = (frame[treat].to_numpy() == levels[1]).astype(float)
    yv = frame[y].to_numpy(dtype=float)

    n1 = np.cumsum(d)
    n0 = np.cumsum(1.0 - d)
    s1 = np.cumsum(d * yv)
    s0 = np.cumsum((1.0 - d) * yv)
    q1 = np.cumsum(d * yv**2)
    q0 = np.cumsum((1.0 - d) * yv**2)
    if time:
        last = np.r_[frame[time].to_numpy()[1:] != frame[time].to_numpy()[:-1], True]
    else:
        last = np.ones(len(frame), dtype=bool)
    ok = last & (n1 >= min_arm) & (n0 >= min_arm)
    if not ok.any():
        raise DataInsufficient(
            f"sp.msprt: no look has {min_arm} or more observations in each arm.",
            recovery_hint="Collect more data or lower min_per_arm (with sigma=).",
        )
    idx = np.flatnonzero(ok)
    n1, n0, s1, s0, q1, q0 = (a[idx] for a in (n1, n0, s1, s0, q1, q0))
    m1, m0 = s1 / n1, s0 / n0
    est = m1 - m0
    if sigma is not None:
        v = sigma**2 * (1.0 / n1 + 1.0 / n0)
        vlabel = "known sigma"
    else:
        var1 = np.maximum(q1 - n1 * m1**2, 0.0) / (n1 - 1.0)
        var0 = np.maximum(q0 - n0 * m0**2, 0.0) / (n0 - 1.0)
        v = var1 / n1 + var0 / n0
        vlabel = "plug-in sample variances"
    if np.any(v <= 0):
        raise DataInsufficient(
            "sp.msprt: an arm has zero sample variance at a look.",
            recovery_hint="Pass sigma= or raise min_per_arm.",
        )
    lam = msprt_lambda(est, v, tau, theta0)
    lam = np.atleast_1d(lam)
    p = np.minimum.accumulate(np.minimum(1.0, 1.0 / lam))
    t2 = tau**2
    half = np.sqrt(
        v * (v + t2) / t2 * (2.0 * np.log(1.0 / alpha) + np.log((v + t2) / v))
    )
    lo = np.maximum.accumulate(est - half)
    hi = np.minimum.accumulate(est + half)
    path = pd.DataFrame(
        {
            "n": (n1 + n0).astype(int),
            "n_treated": n1.astype(int),
            "n_control": n0.astype(int),
            "estimate": est,
            "se": np.sqrt(v),
            "likelihood_ratio": lam,
            "p_value": p,
            "ci_lower": lo,
            "ci_upper": hi,
        }
    )
    cross = np.flatnonzero(p <= alpha)
    return MSPRTResult(
        path=path,
        estimate=float(est[-1]),
        p_value=float(p[-1]),
        ci=(float(lo[-1]), float(hi[-1])),
        rejected=bool(cross.size),
        stop_look=int(cross[0]) + 1 if cross.size else None,
        alpha=float(alpha),
        tau=float(tau),
        theta0=float(theta0),
        variance=vlabel,
        detail={
            "treatment_levels": [levels[0], levels[1]],
            "n_looks": int(len(path)),
            "ci_empty": bool(lo[-1] > hi[-1]),
        },
    )
