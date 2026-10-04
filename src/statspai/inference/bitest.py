"""Exact binomial probability test.

Stata's ``bitest`` / ``bitesti`` and R's ``binom.test``. The two-sided
p-value is the total probability of every outcome no more likely than the
one observed, which is the definition both use; it is not twice the
smaller tail unless the null is one half.
"""

from __future__ import annotations

from typing import ClassVar, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["bitest", "BiTestResult"]


class BiTestResult(ResultProtocolMixin):
    """Outcome of :func:`bitest`.

    Attributes
    ----------
    n_obs, successes : int
        Number of trials and of successes.
    p_null : float
        Probability of success under the null.
    estimate : float
        Observed share of successes.
    expected : float
        ``n_obs * p_null``.
    pvalue : float
        Two-sided p-value. ``pvalue_greater`` is ``Pr(k >= successes)`` and
        ``pvalue_less`` is ``Pr(k <= successes)`` under the null.
    k_opposite : int or None
        The outcome in the opposite tail at which the two-sided region
        starts (Stata's ``r(k_opp)``); ``None`` when that tail is empty.
    ci : tuple of float
        Clopper-Pearson ``1 - alpha`` interval for the share.

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.bitest(n=41, successes=25)
    >>> round(res.pvalue, 4)
    0.211
    """

    _citation_keys: ClassVar[Tuple[str, ...]] = ()

    def __init__(
        self,
        *,
        n_obs: int,
        successes: int,
        p_null: float,
        pvalue: float,
        pvalue_less: float,
        pvalue_greater: float,
        k_opposite: Optional[int],
        ci: Tuple[float, float],
        alpha: float,
    ) -> None:
        self.method = "Binomial probability test"
        self.estimand = "share of successes"
        self.n_obs = n_obs
        self.successes = successes
        self.p_null = p_null
        self.estimate = successes / n_obs
        self.expected = n_obs * p_null
        self.pvalue = pvalue
        self.pvalue_less = pvalue_less
        self.pvalue_greater = pvalue_greater
        self.k_opposite = k_opposite
        self.ci = ci
        self.alpha = alpha

    def summary(self) -> str:
        k = self.successes
        if self.k_opposite is None:
            region = f"k <= {k}" if k <= self.expected else f"k >= {k}"
        else:
            lo, hi = sorted((k, self.k_opposite))
            region = f"k <= {lo} or k >= {hi}"
        lines = [
            self.method,
            "=" * len(self.method),
            f"N = {self.n_obs}   observed k = {k}   expected k = "
            f"{self.expected:g}   assumed p = {self.p_null:.5f}   observed p = "
            f"{self.estimate:.5f}",
            "",
            f"  Pr(k >= {k}) = {self.pvalue_greater:.6f}  (one-sided test)",
            f"  Pr(k <= {k}) = {self.pvalue_less:.6f}  (one-sided test)",
            f"  Pr({region}) = {self.pvalue:.6f}  (two-sided test)",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"BiTestResult(k={self.successes}, n={self.n_obs}, "
            f"p0={self.p_null:g}, p={self.pvalue:.4g})"
        )


def bitest(
    data: Optional[pd.DataFrame] = None,
    y: Optional[str] = None,
    p: float = 0.5,
    *,
    n: Optional[int] = None,
    successes: Optional[int] = None,
    alpha: float = 0.05,
) -> BiTestResult:
    """Exact binomial test that a share equals ``p``.

    ==========================================  ======================
    Call                                        Stata
    ==========================================  ======================
    ``sp.bitest(df, "z", p=0.3)``               ``bitest z == 0.3``
    ``sp.bitest(n=41, successes=25)``           ``bitesti 41 25 0.5``
    ==========================================  ======================

    Parameters
    ----------
    data : DataFrame, optional
    y : str, optional
        A 0/1 column; missing values are dropped. Give ``data`` and ``y``,
        or ``n`` and ``successes``.
    p : float, default 0.5
        Probability of success under the null.
    n, successes : int, optional
        Number of trials and of successes, for the immediate form.
    alpha : float, default 0.05
        One minus the level of the Clopper-Pearson interval.

    Returns
    -------
    BiTestResult

    Notes
    -----
    The two-sided p-value sums the probabilities of all outcomes that are
    no more likely than the observed one. Stata's ``bitest`` and R's
    ``binom.test`` use the same definition, and the numbers agree with
    Stata 18 to 1e-14 (``tests/test_bitest.py``).

    Examples
    --------
    >>> import pandas as pd
    >>> import statspai as sp
    >>> df = pd.DataFrame({"z": [1] * 4 + [0] * 25})
    >>> res = sp.bitest(df, "z", p=0.3)
    >>> (res.n_obs, res.successes)
    (29, 4)
    >>> round(res.pvalue, 4)
    0.0672
    """
    if not 0 < p < 1:
        raise MethodIncompatibility("bitest: p must lie strictly between 0 and 1.")
    from_data = data is not None or y is not None
    from_counts = n is not None or successes is not None
    if from_data == from_counts:
        raise MethodIncompatibility(
            "bitest: give data= and y=, or n= and successes=, not both."
        )
    if from_data:
        if data is None or y is None:
            raise MethodIncompatibility("bitest: data= and y= go together.")
        col = pd.to_numeric(data[y], errors="coerce").dropna().to_numpy(dtype=float)
        if not np.isin(col, (0.0, 1.0)).all():
            raise MethodIncompatibility(
                f"bitest: {y!r} must be a 0/1 variable; found "
                f"{np.unique(col)[:5].tolist()}."
            )
        n_obs, k = int(col.size), int(col.sum())
    else:
        if n is None or successes is None:
            raise MethodIncompatibility("bitest: n= and successes= go together.")
        n_obs, k = int(n), int(successes)
        if k < 0 or k > n_obs:
            raise MethodIncompatibility("bitest: successes must lie between 0 and n.")
    if n_obs < 1:
        raise DataInsufficient("bitest: no observations.")

    pmf = stats.binom.pmf(np.arange(n_obs + 1), n_obs, p)
    # Outcomes "no more likely" than the observed one, with the relative
    # tolerance R's binom.test uses so that ties in probability count.
    region = pmf <= pmf[k] * (1 + 1e-7)
    other_tail = np.arange(n_obs + 1)[region & (np.arange(n_obs + 1) != k)]
    if k <= n_obs * p:
        other_tail = other_tail[other_tail > k]
        k_opp = int(other_tail.min()) if other_tail.size else None
    else:
        other_tail = other_tail[other_tail < k]
        k_opp = int(other_tail.max()) if other_tail.size else None
    lower = float(stats.beta.ppf(alpha / 2, k, n_obs - k + 1)) if k > 0 else 0.0
    upper = float(stats.beta.ppf(1 - alpha / 2, k + 1, n_obs - k)) if k < n_obs else 1.0
    return BiTestResult(
        n_obs=n_obs,
        successes=k,
        p_null=float(p),
        pvalue=float(min(1.0, pmf[region].sum())),
        pvalue_less=float(stats.binom.cdf(k, n_obs, p)),
        pvalue_greater=float(stats.binom.sf(k - 1, n_obs, p)),
        k_opposite=k_opp,
        ci=(lower, upper),
        alpha=alpha,
    )
