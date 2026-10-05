"""Power of the t test, from the noncentral t distribution.

``sp.power_rct`` uses the normal approximation that is standard in the
experimental-design literature. For the t test itself the power has an exact
form: under the alternative the statistic is noncentral t, with
noncentrality ``delta / se`` and the degrees of freedom of the test. That is
what Stata's ``power onemean`` / ``twomeans`` / ``pairedmeans`` and R's
``pwr::pwr.t.test`` compute, and what this module reproduces. The two agree
in large samples; in small ones the normal approximation overstates power
(with 15 per arm and an effect of one standard deviation, 0.78 against 0.75)
and so understates the sample a design needs.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

from scipy import optimize, stats

from ..exceptions import MethodIncompatibility
from .power import PowerResult

__all__ = ["power_ttest"]

_TYPES = {
    "two-sample": "two-sample",
    "two_sample": "two-sample",
    "twosample": "two-sample",
    "twomeans": "two-sample",
    "one-sample": "one-sample",
    "one_sample": "one-sample",
    "onesample": "one-sample",
    "onemean": "one-sample",
    "paired": "paired",
    "pairedmeans": "paired",
}
_SIDES = {
    "two-sided": "two-sided",
    "two.sided": "two-sided",
    "two_sided": "two-sided",
    "twosided": "two-sided",
    "greater": "greater",
    "less": "less",
}


def _sizes(n: float, kind: str, ratio: float) -> Tuple[float, float]:
    """(n1, n2) for a two-sample design of total size n; (n, 0) otherwise."""
    if kind != "two-sample":
        return n, 0.0
    n1 = n / (1.0 + ratio)
    return n1, n - n1


def _power(
    n: float, d: float, alpha: float, kind: str, side: str, ratio: float
) -> float:
    """Power at total size ``n`` (may be fractional) and standardised ``d``."""
    n1, n2 = _sizes(n, kind, ratio)
    if kind == "two-sample":
        return _power_sizes(n1, n2, d, alpha, side)
    return _tail_power(n1 - 1.0, d * math.sqrt(n1), alpha, side)


def _tail_power(df: float, ncp: float, alpha: float, side: str) -> float:
    """Rejection probability of the t test under noncentrality ``ncp``."""
    if df <= 0:
        return float("nan")
    if side == "two-sided":
        crit = stats.t.isf(alpha / 2.0, df)
        return float(stats.nct.sf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp))
    crit = stats.t.isf(alpha, df)
    if side == "greater":
        return float(stats.nct.sf(crit, df, ncp))
    return float(stats.nct.cdf(-crit, df, ncp))


def power_ttest(
    n: Optional[float] = None,
    delta: Optional[float] = None,
    power: Optional[float] = None,
    *,
    sd: float = 1.0,
    alpha: float = 0.05,
    type: str = "two-sample",
    alternative: str = "two-sided",
    ratio: float = 1.0,
) -> PowerResult:
    """Power, sample size or detectable difference of a t test.

    Give two of ``n``, ``delta`` and ``power``; the third is computed. The
    calculation uses the noncentral t distribution, as Stata's ``power
    onemean`` / ``twomeans`` / ``pairedmeans`` and R's ``pwr.t.test`` do.

    Parameters
    ----------
    n : float, optional
        Sample size: the **total** over both groups for a two-sample test
        (Stata's ``n()``; ``pwr.t.test`` takes the size of one group), the
        number of observations for a one-sample test, the number of pairs
        for a paired test.
    delta : float, optional
        The difference to detect, in the units of the outcome: the
        difference between the two means, the distance of the mean from its
        null value, or the mean of the paired differences. With the default
        ``sd=1`` it is the standardised effect (Cohen's d).
    power : float, optional
        Target power, between ``alpha`` and 1.
    sd : float, default 1
        Standard deviation of the outcome, common to the two groups; of the
        paired differences for ``type='paired'``.
    alpha : float, default 0.05
        Significance level.
    type : {'two-sample', 'one-sample', 'paired'}
    alternative : {'two-sided', 'greater', 'less'}
        ``'greater'`` and ``'less'`` are one-sided tests in the direction of
        a positive and a negative ``delta``.
    ratio : float, default 1
        ``n2 / n1`` for a two-sample test (Stata's ``nratio()``).

    Returns
    -------
    PowerResult
        ``power``, ``n`` and ``effect_size`` (``delta / sd``). When ``n`` is
        solved for it is rounded up to whole observations in each group, as
        Stata reports it, and ``power`` is the power actually achieved at
        that size; ``params['n_exact']`` keeps the fractional solution that
        ``pwr.t.test`` prints. ``params`` also holds ``n1``, ``n2``,
        ``delta`` and ``df``.

    Notes
    -----
    The two-sided power counts rejections in both tails. The detectable
    difference is returned as a positive number (negative for
    ``alternative='less'``).

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.power_ttest(n=128, delta=0.5)        # 64 per group
    >>> round(res.power, 4)
    0.8015
    >>> sp.power_ttest(delta=0.5, power=0.8).n        # total, both groups
    128
    >>> round(sp.power_ttest(n=128, power=0.8).params["delta"], 4)
    0.4991
    >>> sp.power_ttest(delta=0.3, power=0.9, alpha=0.01, type="paired").n
    169
    """
    kind = _TYPES.get(str(type).lower().replace(" ", "-"))
    side = _SIDES.get(str(alternative).lower().replace(" ", "-"))
    if kind is None:
        raise MethodIncompatibility(
            f"power_ttest: unknown type {type!r}.",
            recovery_hint="Use 'two-sample', 'one-sample' or 'paired'.",
        )
    if side is None:
        raise MethodIncompatibility(
            f"power_ttest: unknown alternative {alternative!r}.",
            recovery_hint="Use 'two-sided', 'greater' or 'less'.",
        )
    given = sum(v is not None for v in (n, delta, power))
    if given != 2:
        raise MethodIncompatibility(
            "power_ttest: give exactly two of n, delta and power; the third "
            f"is computed ({given} given).",
            recovery_hint="e.g. sp.power_ttest(delta=0.5, power=0.8)",
        )
    if not (0.0 < alpha < 1.0):
        raise MethodIncompatibility("power_ttest: alpha must be between 0 and 1.")
    if not (sd > 0 and math.isfinite(sd)):
        raise MethodIncompatibility("power_ttest: sd must be positive.")
    if not (ratio > 0 and math.isfinite(ratio)):
        raise MethodIncompatibility("power_ttest: ratio must be positive.")
    if power is not None and not (alpha < power < 1.0):
        raise MethodIncompatibility(
            f"power_ttest: power must lie between alpha ({alpha}) and 1.",
            recovery_hint="A test has power alpha when the effect is zero.",
        )
    min_n = 4.0 if kind == "two-sample" else 2.0
    if n is not None and not (n >= min_n):
        raise MethodIncompatibility(
            f"power_ttest: n must be at least {int(min_n)} for a {kind} test."
        )

    def signed(d: float) -> float:
        """The effect as the one-sided test sees it."""
        return -d if side == "less" else d

    n_exact: Optional[float] = None
    if power is None:
        assert n is not None and delta is not None
        d = float(delta) / sd
        achieved = _power(float(n), d, alpha, kind, side, ratio)
        n_out: Any = n
    elif n is None:
        assert delta is not None
        d = float(delta) / sd
        if d == 0 or (side != "two-sided" and signed(d) < 0):
            raise MethodIncompatibility(
                "power_ttest: no sample size reaches that power: delta is "
                "zero, or on the wrong side of a one-sided test.",
                recovery_hint="Check the sign of delta against alternative=.",
            )

        def gap_n(m: float) -> float:
            return _power(m, d, alpha, kind, side, ratio) - float(power)

        lo = min_n * (1 + 1e-9)
        hi = max(8.0, min_n * 2)
        while gap_n(hi) < 0:
            hi *= 2.0
            if hi > 1e12:
                raise MethodIncompatibility(
                    "power_ttest: the sample size needed exceeds 1e12."
                )
        n_exact = (
            lo if gap_n(lo) >= 0 else float(optimize.brentq(gap_n, lo, hi, xtol=1e-10))
        )
        # Whole observations in each group, the first group rounded up and
        # the second set by the ratio, as Stata reports.
        if kind == "two-sample":
            n1 = math.ceil(n_exact / (1.0 + ratio) - 1e-9)
            n2 = math.ceil(n1 * ratio - 1e-9)
            n_out = int(n1 + n2)
            achieved = _power_sizes(n1, n2, d, alpha, side)
        else:
            n_out = int(math.ceil(n_exact - 1e-9))
            achieved = _power(float(n_out), d, alpha, kind, side, ratio)
    else:
        assert n is not None

        def gap_d(dd: float) -> float:
            return _power(float(n), signed(dd), alpha, kind, side, ratio) - float(power)

        hi = 1.0
        while gap_d(hi) < 0:
            hi *= 2.0
            if hi > 1e8:  # pragma: no cover - unreachable for power < 1
                raise MethodIncompatibility("power_ttest: no detectable effect.")
        d = signed(float(optimize.brentq(gap_d, 0.0, hi, xtol=1e-12)))
        achieved = float(power)
        n_out = n

    n1f, n2f = _sizes(float(n_out), kind, ratio)
    if kind == "two-sample" and n_exact is not None:
        n1f = float(math.ceil(n_exact / (1.0 + ratio) - 1e-9))
        n2f = float(n_out) - n1f
    params: Dict[str, Any] = {
        "type": kind,
        "alternative": side,
        "alpha": alpha,
        "sd": sd,
        "delta": d * sd,
        "ratio": ratio if kind == "two-sample" else None,
        "n1": n1f,
        "n2": n2f if kind == "two-sample" else None,
        "df": (n1f + n2f - 2.0) if kind == "two-sample" else n1f - 1.0,
        "n_exact": n_exact,
        "distribution": "noncentral t",
    }
    return PowerResult(
        power_val=float(achieved),
        n=n_out,
        effect_size=float(d),
        design=f"t test ({kind})",
        params=params,
    )


def _power_sizes(n1: float, n2: float, d: float, alpha: float, side: str) -> float:
    """Two-sample power at given group sizes."""
    ncp = d / math.sqrt(1.0 / n1 + 1.0 / n2)
    return _tail_power(n1 + n2 - 2.0, ncp, alpha, side)
