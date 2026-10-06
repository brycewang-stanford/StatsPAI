"""Long-run variance of a time series.

The long-run variance ``J = sum over h of gamma(h)`` is the spectral density
at frequency zero times ``2 pi``. It replaces the variance in every
statement about a sample mean of dependent data: ``sqrt(T) (mean - mu)`` is
asymptotically ``N(0, J)``. :func:`lrvar` estimates it with a kernel
(Bartlett, Parzen, quadratic spectral, Tukey-Hanning or truncated), a fixed
or data-driven bandwidth, and optional VAR prewhitening.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["lrvar", "LongRunVariance"]

_KERNELS: Dict[str, str] = {
    "bartlett": "bartlett",
    "newey-west": "bartlett",
    "parzen": "parzen",
    "qs": "qs",
    "quadratic spectral": "qs",
    "quadratic-spectral": "qs",
    "tukey-hanning": "tukey-hanning",
    "tukey": "tukey-hanning",
    "hanning": "tukey-hanning",
    "truncated": "truncated",
}
# Andrews (1991), eq. (6.2): optimal bandwidth c * (alpha(q) * T)^(1/(2q+1))
_ANDREWS: Dict[str, Tuple[float, int]] = {
    "bartlett": (1.1447, 1),
    "parzen": (2.6614, 2),
    "tukey-hanning": (1.7462, 2),
    "qs": (1.3221, 2),
    "truncated": (0.6611, 2),
}
# Newey and West (1994), Table I: lag of the pilot estimate c * (T/100)^e
_NW_PILOT: Dict[str, Tuple[float, int]] = {
    "bartlett": (2.0 / 9.0, 1),
    "parzen": (4.0 / 25.0, 2),
    "qs": (2.0 / 25.0, 2),
}
_RULES = ("andrews", "newey-west", "rule", "sw")


@dataclass
class LongRunVariance(ResultProtocolMixin):
    """Long-run variance returned by :func:`statspai.lrvar`.

    Attributes
    ----------
    lrvar : float or numpy.ndarray
        The long-run variance; a ``K x K`` matrix for ``K > 1`` series.
    var_mean : float or numpy.ndarray
        Variance of the sample mean, ``lrvar / n_obs``.
    se_mean : float or numpy.ndarray
        Its square root (of the diagonal, for several series).
    variance : float or numpy.ndarray
        The ordinary variance ``gamma(0)`` with divisor ``n_obs``.
    kernel : str
    bandwidth : float
        Scale ``S`` of the kernel: lag ``j`` has weight ``k(j / S)``.
    bandwidth_rule : str
        ``'fixed'`` or the automatic rule that produced ``bandwidth``.
    prewhite : int
    adjust : bool
    n_obs : int
    names : list of str

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> x = np.array([1.0, 2.0, 4.0, 3.0, 5.0, 4.0, 6.0, 5.0])
    >>> fit = sp.lrvar(x, kernel="bartlett", bandwidth=2)
    >>> round(float(fit), 4)
    3.3047
    """

    lrvar: Any
    var_mean: Any
    se_mean: Any
    variance: Any
    kernel: str
    bandwidth: float
    bandwidth_rule: str
    prewhite: int
    adjust: bool
    n_obs: int
    names: List[str] = field(default_factory=list)

    def __float__(self) -> float:
        if np.ndim(self.lrvar) != 0:
            raise TypeError(
                "the long-run variance of several series is a matrix; " "read .lrvar"
            )
        return float(self.lrvar)

    def to_frame(self) -> pd.DataFrame:
        """The long-run covariance matrix as a labelled DataFrame."""
        mat = np.atleast_2d(np.asarray(self.lrvar, dtype=float))
        return pd.DataFrame(mat, index=self.names, columns=self.names)

    def summary(self) -> str:
        """Plain-text report of the estimate and how it was computed."""
        lines = [
            "Long-run variance",
            f"  kernel            {self.kernel}",
            f"  bandwidth         {self.bandwidth:.4f} ({self.bandwidth_rule})",
            (
                f"  prewhitening      VAR({self.prewhite})"
                if self.prewhite
                else "  prewhitening      none"
            ),
            f"  small-sample adj. {'yes' if self.adjust else 'no'}",
            f"  observations      {self.n_obs}",
        ]
        if np.ndim(self.lrvar) == 0:
            ratio = float(self.lrvar) / float(self.variance)
            lines += [
                f"  variance gamma(0) {float(self.variance):.6g}",
                f"  long-run variance {float(self.lrvar):.6g}",
                f"  ratio J/gamma(0)  {ratio:.4f}",
                f"  var(mean) = J/T   {float(self.var_mean):.6g}",
                f"  se(mean)          {float(self.se_mean):.6g}",
            ]
        else:
            lines.append("  long-run covariance matrix:")
            lines += ["    " + ln for ln in self.to_frame().to_string().splitlines()]
            se = ", ".join(f"{v:.6g}" for v in np.asarray(self.se_mean))
            lines.append(f"  se(mean)          {se}")
        return "\n".join(lines)


def _matrix(
    data: Union[pd.DataFrame, pd.Series, np.ndarray, Sequence[float]],
    y: Union[None, str, Sequence[str]],
) -> Tuple[np.ndarray, List[str]]:
    if isinstance(data, pd.DataFrame):
        cols = (
            list(data.columns) if y is None else [y] if isinstance(y, str) else list(y)
        )
        missing = [c for c in cols if c not in data.columns]
        if missing:
            raise MethodIncompatibility(
                f"sp.lrvar: {missing} not in the columns of the data.",
                recovery_hint="Pass column names of data in y=.",
            )
        x = data[cols].to_numpy(dtype=float, na_value=np.nan)
        names = [str(c) for c in cols]
    else:
        x = np.asarray(data, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        if x.ndim != 2:
            raise MethodIncompatibility(
                "sp.lrvar: the data must be one series or a T x K array.",
                recovery_hint="Pass a 1-D or 2-D array.",
            )
        if isinstance(data, pd.Series) and data.name is not None:
            names = [str(data.name)]
        else:
            names = [f"x{i + 1}" for i in range(x.shape[1])]
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise DataInsufficient(
            "sp.lrvar: the data are empty.", recovery_hint="Check the input."
        )
    if not np.isfinite(x).all():
        raise MethodIncompatibility(
            "sp.lrvar: the data contain missing or infinite values; "
            "autocovariances need consecutive periods.",
            recovery_hint="Drop the incomplete rows at the ends of the "
            "sample, or analyse the longest complete stretch.",
        )
    return x, names


def _kernel_weights(x: np.ndarray, kernel: str) -> np.ndarray:
    """Kernel ``k(x)`` at non-negative arguments."""
    x = np.abs(np.asarray(x, dtype=float))
    if kernel == "truncated":
        return np.where(x <= 1.0, 1.0, 0.0)
    if kernel == "bartlett":
        return np.where(x <= 1.0, 1.0 - x, 0.0)
    if kernel == "parzen":
        inner = 1.0 - 6.0 * x**2 + 6.0 * x**3
        outer = 2.0 * (1.0 - x) ** 3
        return np.where(x <= 0.5, inner, np.where(x <= 1.0, outer, 0.0))
    if kernel == "tukey-hanning":
        return np.where(x <= 1.0, (1.0 + np.cos(np.pi * x)) / 2.0, 0.0)
    # quadratic spectral
    out = np.ones_like(x)
    nz = x > 0
    z = 6.0 * np.pi * x[nz] / 5.0
    out[nz] = 3.0 / z**2 * (np.sin(z) / z - np.cos(z))
    return out


def _prewhiten(u: np.ndarray, order: int) -> Tuple[np.ndarray, np.ndarray]:
    """Residuals of a VAR(order) without intercept, and ``(I - sum A)^-1``."""
    n, k = u.shape
    lagged = np.hstack([u[order - j : n - j] for j in range(1, order + 1)])
    coef, *_ = np.linalg.lstsq(lagged, u[order:], rcond=None)
    resid = u[order:] - lagged @ coef
    a_sum = np.zeros((k, k))
    for j in range(order):
        a_sum += coef[j * k : (j + 1) * k].T
    gap = np.eye(k) - a_sum
    if abs(np.linalg.det(gap)) < 1e-10:
        raise MethodIncompatibility(
            "sp.lrvar: the prewhitening VAR has a unit root, so the "
            "recoloured estimate is not defined.",
            recovery_hint="Difference the series or use prewhite=0.",
        )
    return resid, np.linalg.inv(gap)


def _bw_andrews(u: np.ndarray, kernel: str, n: int) -> float:
    """Andrews (1991) plug-in bandwidth from an AR(1) fitted to each series.

    The AR(1) is fitted by least squares with an intercept.
    """
    num1 = num2 = den = 0.0
    for a in range(u.shape[1]):
        col = u[:, a]
        lag = col[:-1] - col[:-1].mean()
        lead = col[1:] - col[1:].mean()
        ss = float(lag @ lag)
        if ss <= 0:
            raise DataInsufficient(
                "sp.lrvar: a series is constant.",
                recovery_hint="The long-run variance needs variation.",
            )
        rho = float(lead @ lag) / ss
        if abs(rho) >= 1.0:
            raise MethodIncompatibility(
                "sp.lrvar: the AR(1) coefficient used by the Andrews "
                f"bandwidth is {rho:.4f}, outside the stationary region.",
                recovery_hint="Difference the series, or pass a number "
                "in bandwidth=.",
            )
        resid = lead - rho * lag
        s4 = float(resid @ resid / resid.size) ** 2
        den += s4 / (1.0 - rho) ** 4
        num1 += 4.0 * rho**2 * s4 / ((1.0 - rho) ** 6 * (1.0 + rho) ** 2)
        num2 += 4.0 * rho**2 * s4 / (1.0 - rho) ** 8
    const, q = _ANDREWS[kernel]
    alpha = (num1 if q == 1 else num2) / den
    return float(const * (alpha * n) ** (1.0 / (2 * q + 1)))


def _bw_newey_west(u: np.ndarray, kernel: str, n: int, prewhite: int) -> float:
    """Newey and West (1994) non-parametric plug-in bandwidth."""
    if kernel not in _NW_PILOT:
        raise MethodIncompatibility(
            f"sp.lrvar: bandwidth='newey-west' is not defined for the "
            f"{kernel} kernel.",
            recovery_hint="Use the Bartlett, Parzen or QS kernel, or "
            "bandwidth='andrews'.",
        )
    power, q = _NW_PILOT[kernel]
    scale = 3.0 if prewhite else 4.0
    m = int(np.floor(scale * (n / 100.0) ** power))
    h = u.sum(axis=1)
    m = min(m, h.size - 1)
    sig = np.array([float(h[: h.size - j] @ h[j:]) for j in range(m + 1)])
    lags = np.arange(1, m + 1)
    s0 = sig[0] + 2.0 * sig[1:].sum()
    sq = 2.0 * float((lags**q * sig[1:]).sum())
    if s0 == 0:
        raise MethodIncompatibility(
            "sp.lrvar: the pilot estimate of the Newey-West bandwidth is zero.",
            recovery_hint="Use bandwidth='andrews' or pass a number.",
        )
    const = _ANDREWS[kernel][0]
    exponent = 1.0 / (2 * q + 1)
    return float(const * ((sq / s0) ** 2) ** exponent * n**exponent)


def _kernel_sum(u: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``sum_j w_j (G_j + G_j')`` with ``G_j = sum_t u_t u_{t-j}'`` (not / n)."""
    n = u.shape[0]
    total = 0.5 * weights[0] * (u.T @ u)
    for j in range(1, min(weights.size, n)):
        if weights[j] != 0.0:
            total = total + weights[j] * (u[: n - j].T @ u[j:])
    return np.asarray(total + total.T)


def lrvar(
    data: Union[pd.DataFrame, pd.Series, np.ndarray, Sequence[float]],
    y: Union[None, str, Sequence[str]] = None,
    *,
    kernel: str = "bartlett",
    bandwidth: Union[float, str] = "andrews",
    prewhite: int = 0,
    demean: bool = True,
    adjust: bool = False,
    integer_lag: bool = False,
    tol: float = 1e-7,
) -> LongRunVariance:
    """Long-run variance of one or several series.

    Estimates ``J = sum over all h of gamma(h)``, the limit of
    ``T * Var(sample mean)``, by a weighted sum of sample autocovariances
    ``J_hat = sum_{|j| < T} k(j / S) gamma_hat(j)``.

    Parameters
    ----------
    data : DataFrame, Series or array
        The series in time order; a ``T x K`` array or several columns give
        the ``K x K`` long-run covariance matrix.
    y : str or sequence of str, optional
        Column(s) of a DataFrame. Default: every column.
    kernel : str, default 'bartlett'
        ``'bartlett'``, ``'parzen'``, ``'qs'`` (quadratic spectral),
        ``'tukey-hanning'`` or ``'truncated'``. Bartlett, Parzen and QS
        give a non-negative estimate; the other two need not.
    bandwidth : float or str, default 'andrews'
        The scale ``S`` of the kernel, or a rule that chooses it:

        * ``'andrews'``: the plug-in bandwidth of Andrews (1991) from an
          AR(1) fitted to each series, with the constant of the kernel.
        * ``'newey-west'``: the non-parametric plug-in of Newey and West
          (1994); Bartlett, Parzen and QS only.
        * ``'rule'``: lag ``L = floor(4 (T/100)^(2/9))``, ``S = L + 1``.
        * ``'sw'``: lag ``L = floor(0.75 T^(1/3))``, ``S = L + 1``.

        With the Bartlett kernel ``S = L + 1`` gives the weights
        ``1 - j / (L + 1)`` of a Newey-West estimator with ``L`` lags.
    prewhite : int, default 0
        Order of a VAR fitted without intercept to the (demeaned) series
        before the kernel is applied; the estimate on its residuals is
        recoloured with ``(I - sum A_j)^-1`` on both sides (Andrews and
        Monahan 1992).
    demean : bool, default True
        Subtract the sample mean of each series. ``False`` is for series
        known to have mean zero, such as moment conditions at the truth.
    adjust : bool, default False
        Multiply by ``T / (T - 1)`` for the estimated mean. Needs
        ``demean=True``.
    integer_lag : bool, default False
        Replace an automatic bandwidth ``S`` by ``floor(S) + 1``, so that
        the Bartlett weights are ``1 - j / (floor(S) + 1)``. This is what
        R ``sandwich::NeweyWest`` does with its automatic bandwidth.
    tol : float, default 1e-7
        Kernel weights after the last one larger than ``tol`` in absolute
        value are dropped. Only the QS kernel, which never reaches zero,
        is affected.

    Returns
    -------
    LongRunVariance
        ``lrvar`` (a float for one series, a ``K x K`` array otherwise),
        ``var_mean`` (``lrvar / T``), ``se_mean``, ``variance``,
        ``bandwidth``, ``bandwidth_rule``, ``kernel``, ``n_obs`` and
        ``summary()``. ``float(result)`` works for one series.

    Raises
    ------
    MethodIncompatibility
        Unknown kernel or rule, missing values, a non-positive bandwidth,
        a rule not defined for the kernel, or a unit root in the
        prewhitening VAR.
    DataInsufficient
        Too few observations for the lags requested.

    Notes
    -----
    Autocovariances have divisor ``T`` whatever the lag, also after
    prewhitening has used up the first ``prewhite`` observations. The
    Andrews rule fits each AR(1) by least squares with an intercept and
    uses the number of observations left after prewhitening; the
    Newey-West rule uses ``T``, a pilot lag of ``floor(c (T/100)^e)`` with
    ``c = 4`` (``3`` after prewhitening) and ``e = 2/9`` (Bartlett),
    ``4/25`` (Parzen) or ``2/25`` (QS). These are the conventions of R
    ``sandwich`` (3.1-1), whose estimates this function reproduces to
    rounding error on an intercept-only regression ``m = lm(x ~ 1)``,
    where ``lrvar = T * vcov``:

    * ``kernHAC(m, kernel=, bw=S, prewhite=, adjust=)`` is
      ``sp.lrvar(x, kernel=, bandwidth=S, prewhite=, adjust=)``; with
      ``bw=bwAndrews`` it is ``bandwidth='andrews'``, with
      ``bw=bwNeweyWest`` it is ``bandwidth='newey-west'``.
    * ``NeweyWest(m, lag=L, prewhite=, adjust=)`` is
      ``kernel='bartlett', bandwidth=L + 1``; with the default
      ``lag=NULL`` it is ``bandwidth='newey-west', integer_lag=True``.
    * ``sandwich::lrvar(x, type='Andrews')`` is ``kernel='qs',
      bandwidth='andrews', prewhite=1, adjust=True`` and returns
      ``var_mean``; ``type='Newey-West'`` is ``kernel='bartlett',
      bandwidth='newey-west', integer_lag=True, prewhite=1, adjust=True``.

    For several series ``'andrews'`` and ``'newey-west'`` weight every
    series equally, in the units of the data: rescaling one series changes
    the bandwidth. ``adjust=True`` is ``T / (T - 1)`` for any number of
    series, one estimated mean per series; ``sandwich`` applied to
    ``lm(cbind(x1, ..., xK) ~ 1)`` uses ``T / (T - K)``.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> e = rng.normal(size=500)
    >>> x = np.zeros(500)
    >>> for t in range(1, 500):
    ...     x[t] = 0.5 * x[t - 1] + e[t]
    >>> fit = sp.lrvar(x, kernel="qs", bandwidth="andrews", prewhite=1)
    >>> bool(2.0 < float(fit) < 8.0)   # truth: 1 / (1 - 0.5)^2 = 4
    True
    >>> fit.kernel
    'qs'

    References
    ----------
    [@newey1987simple],
    [@andrews1991heteroskedasticity],
    [@andrews1992improved],
    [@newey1994automatic],
    [@zeileis2020sandwich],
    [@neusser2016time]
    """
    x, names = _matrix(data, y)
    n, k = x.shape
    key = _KERNELS.get(str(kernel).strip().lower())
    if key is None:
        raise MethodIncompatibility(
            f"sp.lrvar: kernel={kernel!r} is not known.",
            recovery_hint="Use 'bartlett', 'parzen', 'qs', 'tukey-hanning' "
            "or 'truncated'.",
        )
    prewhite = int(prewhite)
    if prewhite < 0:
        raise MethodIncompatibility(
            f"sp.lrvar: prewhite={prewhite} is negative.",
            recovery_hint="Use 0 (none) or 1.",
        )
    if adjust and not demean:
        raise MethodIncompatibility(
            "sp.lrvar: adjust=True corrects for the estimated mean and "
            "needs demean=True.",
            recovery_hint="Drop adjust=True or set demean=True.",
        )
    if n < max(4, (k + 1) * prewhite + 3):
        raise DataInsufficient(
            f"sp.lrvar: {n} observations are too few.",
            recovery_hint="Use a longer series.",
        )
    u = x - x.mean(axis=0) if demean else x.copy()
    variance = u.T @ u / n
    recolour: Optional[np.ndarray] = None
    if prewhite:
        u, recolour = _prewhiten(u, prewhite)
    m = u.shape[0]

    if isinstance(bandwidth, str):
        rule = bandwidth.strip().lower().replace("_", "-")
        if rule not in _RULES:
            raise MethodIncompatibility(
                f"sp.lrvar: bandwidth={bandwidth!r} is not a known rule.",
                recovery_hint=f"Use a positive number or one of {_RULES}.",
            )
        if rule == "andrews":
            bw = _bw_andrews(u, key, m)
        elif rule == "newey-west":
            bw = _bw_newey_west(u, key, n, prewhite)
        elif rule == "rule":
            bw = float(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))) + 1.0
        else:
            bw = float(np.floor(0.75 * n ** (1.0 / 3.0))) + 1.0
        if integer_lag and rule in ("andrews", "newey-west"):
            bw = float(np.floor(bw)) + 1.0
    else:
        rule = "fixed"
        bw = float(bandwidth)
    if not np.isfinite(bw) or bw <= 0:
        raise MethodIncompatibility(
            f"sp.lrvar: the bandwidth is {bw}; it must be positive.",
            recovery_hint="Pass a positive number in bandwidth=. A series "
            "without autocorrelation gives an automatic bandwidth of 0: "
            "its long-run variance is its variance.",
        )

    weights = _kernel_weights(np.arange(m) / bw, key)
    keep = np.flatnonzero(np.abs(weights) > tol)
    weights = weights[: keep[-1] + 1] if keep.size else weights[:1]
    total = _kernel_sum(u, weights)
    if recolour is not None:
        total = recolour @ total @ recolour.T
    lr = total / n
    if adjust:
        lr = lr * n / (n - 1)
    if k == 1:
        value: Any = float(lr[0, 0])
        var_mean: Any = value / n
        if value < 0:
            raise MethodIncompatibility(
                f"sp.lrvar: the {key} kernel gave a negative long-run "
                f"variance ({value:.6g}).",
                recovery_hint="Use the Bartlett, Parzen or QS kernel, " "which cannot.",
                diagnostics={"lrvar": value, "bandwidth": bw},
            )
        se_mean: Any = float(np.sqrt(var_mean))
        gamma0: Any = float(variance[0, 0])
    else:
        value = lr
        var_mean = lr / n
        diag = np.diag(var_mean)
        if (diag < 0).any():
            raise MethodIncompatibility(
                f"sp.lrvar: the {key} kernel gave a negative long-run "
                "variance on the diagonal.",
                recovery_hint="Use the Bartlett, Parzen or QS kernel, " "which cannot.",
                diagnostics={"diagonal": diag.tolist(), "bandwidth": bw},
            )
        se_mean = np.sqrt(diag)
        gamma0 = variance
    return LongRunVariance(
        lrvar=value,
        var_mean=var_mean,
        se_mean=se_mean,
        variance=gamma0,
        kernel=key,
        bandwidth=bw,
        bandwidth_rule=rule,
        prewhite=prewhite,
        adjust=bool(adjust),
        n_obs=int(n),
        names=names,
    )
