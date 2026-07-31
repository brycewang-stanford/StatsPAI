"""Pre-trend equivalence testing (Liu, Wang & Xu 2024).

A failure to reject "no pre-trend" is not evidence that pre-trends are absent —
it is often just low power, which is Roth's (2022) point. The counterfactual-
estimator literature answers this by *reversing the null*: instead of testing
whether pre-period effects are zero, test whether they are demonstrably
**small**. That is what R ``fect``'s diagnostic panel reports, and it is the
current standard for validating parallel trends in TSCS designs.

Three statistics, all ported from ``fect:::diagtest``:

``f_stat`` / ``f_pvalue``
    The conventional joint test that every pre-period effect is zero.
    ``psi = D' S^-1 D`` scaled by ``(N_bar - k) / ((N_bar - 1) k)`` and
    referred to ``F(k, N_bar - k)``. A small p-value says the pre-trend is
    detectable; a large one says only that it was not detected.

``f_equivalence_pvalue``
    The same statistic read against a *non-central* ``F`` with non-centrality
    ``N_bar * f_threshold``. Here a **small** p-value is the reassuring
    outcome: it rejects the null that the pre-trend is at least as large as
    the threshold.

``tost_pvalue``
    Two one-sided tests per pre-period against ``±tost_threshold``, taking the
    least favourable (maximum) p-value across periods. Again, small is
    reassuring: it rejects "at least one pre-period effect is bigger than the
    threshold".

Because the equivalence tests invert the usual logic, the threshold is a
substantive judgement about what counts as a negligible pre-trend. The
dimensionless ``f_threshold`` carries ``fect``'s 0.6 default; the TOST
threshold is in outcome units and has no defensible universal default, so it
must be supplied — omitting it reports the F tests and leaves TOST as ``None``
rather than inventing a scale.

Verified against R ``fect`` 2.4.1: on identical inputs the F statistic,
its p-value and the TOST p-value match to ~1e-15, and the non-central F
equivalence p-value to ~2e-10 (SciPy vs R's non-central F).

References
----------
Liu, L., Wang, Y. and Xu, Y. (2024). "A Practical Guide to Counterfactual
Estimators for Causal Inference with Time-Series Cross-Sectional Data."
*American Journal of Political Science*, 68(1), 160-176.
[@liu2024practical]

Roth, J. (2022). "Pretest with Caution: Event-Study Estimates after Testing
for Parallel Trends." *American Economic Review: Insights*, 4(3), 305-322.
[@roth2022pretest]
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np
from scipy import stats

from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = [
    "EquivalenceResult",
    "pretrend_equivalence",
    "pretrends_equivalence",
]


class EquivalenceResult(NamedTuple):
    """Joint pre-trend test plus its two equivalence counterparts."""

    f_stat: float
    f_pvalue: float
    f_equivalence_pvalue: float
    tost_pvalue: Optional[float]
    df1: int
    df2: int
    n_bar: int
    f_threshold: float
    tost_threshold: Optional[float]
    event_times: np.ndarray

    def verdict(self, alpha: float = 0.05) -> str:
        """One-line reading of the detection and equivalence tests together.

        Equivalence is judged on the TOST when a threshold was supplied,
        because that bound is in interpretable outcome units. The
        non-central-F test uses a dimensionless bound (0.6 by default) that
        rejects easily, so it is only used as the fallback.
        """
        detected = self.f_pvalue < alpha
        equivalent = (
            self.tost_pvalue < alpha
            if self.tost_pvalue is not None
            else self.f_equivalence_pvalue < alpha
        )
        if detected and not equivalent:
            return "pre-trend detected"
        if detected and equivalent:
            return "pre-trend detected but bounded below the threshold"
        if equivalent:
            return "no pre-trend detected, and bounded below the threshold"
        return (
            "no pre-trend detected, but not bounded either — the test is "
            "uninformative (low power)"
        )


def pretrend_equivalence(
    pre_estimates: np.ndarray,
    pre_cov: np.ndarray,
    n_bar: int,
    pre_se: Optional[np.ndarray] = None,
    f_threshold: float = 0.6,
    tost_threshold: Optional[float] = None,
    event_times: Optional[np.ndarray] = None,
) -> EquivalenceResult:
    """Joint and equivalence tests on a vector of pre-period effects.

    Parameters
    ----------
    pre_estimates
        Pre-period event-study coefficients, excluding the normalised
        reference period.
    pre_cov
        Their joint covariance matrix. The *joint* covariance is what makes
        the F test valid; a diagonal built from the standard errors would
        ignore the (substantial) cross-period correlation.
    n_bar
        Number of treated units contributing to the pre-periods. This enters
        the finite-sample scaling and the non-centrality parameter.
    pre_se
        Per-period standard errors used by the TOST. Defaults to
        ``sqrt(diag(pre_cov))``.
    f_threshold
        Dimensionless effect-size bound for the F equivalence test
        (``fect``'s default is 0.6).
    tost_threshold
        Equivalence bound in **outcome units**. Required for the TOST; when
        omitted the TOST is reported as ``None``.
    event_times
        Optional labels, carried through for reporting.
    """
    d = np.asarray(pre_estimates, dtype=float).ravel()
    cov = np.asarray(pre_cov, dtype=float)
    k = d.size

    if k < 1:
        raise DataInsufficient(
            "pre-trend equivalence testing needs at least one pre-period "
            "effect (excluding the normalised reference period).",
            recovery_hint="Widen the event-study window.",
            diagnostics={"n_pre": int(k)},
        )
    if cov.shape != (k, k):
        raise MethodIncompatibility(
            f"pre_cov must be {k}x{k}; got {cov.shape}.",
            recovery_hint="Pass the joint covariance of the pre-period "
            "coefficients.",
            diagnostics={"shape": list(cov.shape)},
        )
    if n_bar <= k:
        raise DataInsufficient(
            f"the F test needs more treated units than pre-periods; got "
            f"n_bar={n_bar} with {k} pre-periods.",
            recovery_hint="Shorten the pre-period window or use a design "
            "with more treated units.",
            diagnostics={"n_bar": int(n_bar), "n_pre": int(k)},
        )

    try:
        psi = float(d @ np.linalg.solve(cov, d))
    except np.linalg.LinAlgError as exc:
        raise DataInsufficient(
            "the pre-period covariance matrix is singular, so the joint F "
            "test is not identified.",
            recovery_hint="Drop a collinear pre-period or shorten the " "window.",
            diagnostics={"n_pre": int(k)},
        ) from exc

    scale = (n_bar - k) / ((n_bar - 1) * k)
    f_stat = psi * scale
    df2 = int(n_bar - k)
    f_pvalue = float(stats.f.sf(f_stat, k, df2))
    f_equiv = float(stats.ncf.cdf(f_stat, k, df2, n_bar * f_threshold))

    tost_p: Optional[float] = None
    if tost_threshold is not None:
        thr = float(tost_threshold)
        if thr <= 0:
            raise MethodIncompatibility(
                f"tost_threshold must be positive; got {thr}.",
                recovery_hint="Pass a positive equivalence bound in outcome " "units.",
                diagnostics={"tost_threshold": thr},
            )
        se = (
            np.sqrt(np.diag(cov))
            if pre_se is None
            else np.asarray(pre_se, dtype=float).ravel()
        )
        # Two one-sided tests against -thr and +thr; the binding one is the
        # larger p-value, and across periods the least favourable period.
        p_low = 1.0 - stats.norm.cdf((d + thr) / se)
        p_high = 1.0 - stats.norm.cdf((thr - d) / se)
        tost_p = float(np.max(np.maximum(p_low, p_high)))

    return EquivalenceResult(
        f_stat=float(f_stat),
        f_pvalue=f_pvalue,
        f_equivalence_pvalue=f_equiv,
        tost_pvalue=tost_p,
        df1=int(k),
        df2=df2,
        n_bar=int(n_bar),
        f_threshold=float(f_threshold),
        tost_threshold=None if tost_threshold is None else float(tost_threshold),
        event_times=(
            np.arange(-k, 0) if event_times is None else np.asarray(event_times)
        ),
    )


def pretrends_equivalence(
    result,
    f_threshold: float = 0.6,
    tost_threshold: Optional[float] = None,
    alpha: float = 0.05,
) -> EquivalenceResult:
    """Pre-trend equivalence tests for a fitted DiD result.

    Answers the question a plain pre-trend test cannot: *is the pre-trend
    demonstrably small*, as opposed to merely not detected? A failure to
    reject "no pre-trend" is frequently just low power (Roth 2022), so
    reporting it alone overstates the evidence for parallel trends.

    Needs the **joint** covariance of the pre-period event-study
    coefficients, which is recovered from a Callaway-Sant'Anna fit's
    influence functions. Results without influence functions cannot support
    the joint F test and raise rather than silently substituting a diagonal.

    Parameters
    ----------
    result : CausalResult
        A DiD fit carrying an event study and influence functions (e.g.
        :func:`sp.callaway_santanna`).
    f_threshold : float, default 0.6
        Dimensionless effect-size bound for the F equivalence test
        (``fect``'s default).
    tost_threshold : float, optional
        Equivalence bound in outcome units. Without it the TOST is skipped —
        there is no defensible universal default for "how big a pre-trend is
        too big", and inventing one would be a substantive judgement made on
        the user's behalf. ``fect`` uses ``0.36 * residual SD``.
    alpha : float, default 0.05
        Level used by :meth:`EquivalenceResult.verdict`.

    Returns
    -------
    EquivalenceResult
        ``f_stat`` / ``f_pvalue`` (conventional joint test),
        ``f_equivalence_pvalue`` and ``tost_pvalue`` (equivalence tests,
        where *small* is the reassuring outcome), plus ``.verdict()``.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_did(n_units=200, n_periods=8, staggered=True, seed=42)
    >>> df['first_treat'] = df['first_treat'].fillna(0)
    >>> cs = sp.callaway_santanna(df, y='y', g='first_treat', t='time', i='unit')
    >>> eq = sp.pretrends_equivalence(cs)
    >>> 0.0 <= eq.f_pvalue <= 1.0
    True
    >>> isinstance(eq.verdict(), str)
    True

    References
    ----------
    liu2024practical, roth2022pretest
    """
    from ._flci import event_study_moments

    moments = event_study_moments(result)
    if moments is None:
        raise MethodIncompatibility(
            "pretrends_equivalence needs the joint pre-period covariance, "
            "which is recovered from influence functions; this result does "
            "not carry them.",
            recovery_hint="Pass a sp.callaway_santanna result, or compute "
            "the tests directly with "
            "statspai.did._equivalence.pretrend_equivalence.",
            diagnostics={"method": getattr(result, "method", None)},
        )

    beta, sigma, times = moments
    pre = times < 0
    if int(pre.sum()) < 2:
        raise DataInsufficient(
            "pre-trend equivalence testing needs at least two pre-periods "
            "(one is absorbed as the normalisation reference).",
            recovery_hint="Widen the event-study window.",
            diagnostics={"n_pre": int(pre.sum())},
        )

    # fect drops the earliest pre-period when the window covers them all: it
    # is the implicit normalisation and carries no independent information.
    idx = np.where(pre)[0]
    idx = idx[np.argsort(times[idx])][1:]

    detail = getattr(result, "detail", None)
    model_info = getattr(result, "model_info", None) or {}
    n_bar = None
    if detail is not None and "group" in detail.columns:
        sizes = model_info.get("cohort_sizes")
        if sizes is not None:
            treated = [g for g in detail["group"].unique() if g != 0]
            n_bar = int(sum(float(sizes.get(g, 0.0)) for g in treated))
    if not n_bar:
        n_bar = int(model_info.get("n_units", getattr(result, "n_obs", 0)))

    return pretrend_equivalence(
        pre_estimates=beta[idx],
        pre_cov=sigma[np.ix_(idx, idx)],
        n_bar=n_bar,
        f_threshold=f_threshold,
        tost_threshold=tost_threshold,
        event_times=times[idx],
    )
