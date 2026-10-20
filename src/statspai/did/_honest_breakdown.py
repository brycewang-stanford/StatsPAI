"""Breakdown value of the Rambachan-Roth sensitivity analysis.

:func:`breakdown_m` inverts the confidence set :func:`honest_did` reports:
the largest ``M`` (or ``Mbar``) at which it still excludes zero.
"""

import warnings

import numpy as np
from scipy import stats

from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility
from ._arp_sets import rm_confidence_set
from ._flci import breakdown_m_sd, event_study_moments
from ._honest_common import (
    _extract_event_study,
    _require_alpha,
    _require_int,
    _require_string,
    _rm_inputs,
)


def breakdown_m(
    result: CausalResult,
    e: int = 0,
    method: str = "smoothness",
    alpha: float = 0.05,
) -> float:
    """
    Compute the breakdown value of M.

    The breakdown M* is the largest violation magnitude under which
    the treatment effect at relative time ``e`` remains statistically
    significant. Larger M* = more robust.

    Parameters
    ----------
    result : CausalResult
        DID result with event study.
    e : int, default 0
        Relative time period.
    method : {'smoothness', 'relative_magnitude'}, default 'smoothness'
        The restriction whose confidence set is inverted; the same meaning
        as in :func:`honest_did`.
    alpha : float, default 0.05

    Returns
    -------
    float
        Breakdown value M*. The effect is significant for all M < M*.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> rows = []  # staggered panel: half treated at t=5, half never
    >>> for i in range(30):
    ...     g = 5 if i < 15 else 0
    ...     ui = rng.normal(0, 1)
    ...     for t in range(1, 9):
    ...         post = 1 if (g != 0 and t >= g) else 0
    ...         y = ui + 0.3 * t + 2.0 * post + rng.normal(0, 0.5)
    ...         rows.append({"unit": i, "time": t, "y": y, "g": g})
    >>> df = pd.DataFrame(rows)
    >>> r = sp.sun_abraham(df, y='y', g='g', t='time', i='unit')
    >>> m_star = sp.breakdown_m(r, e=0)
    >>> bool(m_star >= 0)  # parallel trends can deviate by up to M* per period
    True

    Notes
    -----
    M* = sup{M : 0 not in CI(M)}, where CI(M) is exactly the interval
    :func:`honest_did` reports for the same ``result``, ``e``, ``method`` and
    ``alpha`` with ``backend='native'``. The breakdown value is a property of
    that confidence set, so it is computed by inverting it:

    * ``method='smoothness'`` with a recoverable event-study covariance (a
      Callaway-Sant'Anna fit or its ``aggte(type='dynamic')``): the
      Rambachan-Roth fixed-length confidence interval under ``Delta^SD(M)``,
      inverted by root-finding on the bound that faces zero. This is what
      R ``HonestDiD`` reports when its ``createSensitivityResults(method =
      'FLCI')`` grid is refined to the crossing.
    * ``method='smoothness'`` without the covariance: :func:`honest_did` falls
      back to the worst-case-bias interval ``theta_hat +/- (M (e+1) +
      z SE)``, and so does this function, giving the closed form
      ``(|theta_hat| - z SE) / (e + 1)``. It warns, as :func:`honest_did`
      does.
    * ``method='relative_magnitude'`` with the covariance: the
      conditional least-favourable hybrid confidence set under
      ``Delta^RM(Mbar)`` (HonestDiD's default), bisected in ``Mbar`` to
      1e-3. The set is nested in ``Mbar``, so the crossing is unique.
    * ``method='relative_magnitude'`` without it: the worst-case-bias
      approximation ``theta_hat +/- (Mbar max|pre| + z SE)``, giving
      ``(|theta_hat| - z SE) / max|pre|``, with a warning.

    Through 1.28.0 this function returned ``(|theta_hat| - z SE) / (e + 1)``
    for every input: ``method`` was validated and then ignored, and the
    smoothness value did not invert the FLCI that :func:`honest_did` reports
    for the same fit (on the event study in
    ``tests/reference_parity/test_did_synth_R_parity.py``, at ``e = 1`` it
    returned 0.0967 where the FLCI breakdown -- and R ``HonestDiD`` -- give
    0.0646; on mpdta's Callaway-Sant'Anna fit at ``e = 0``, 0.0193 against
    0.0080).

    See Rambachan & Roth (2023, *ReStud*) [@rambachan2023more].
    """
    e = _require_int(e, argument="e")
    alpha = _require_alpha(alpha)
    method = _require_string(method, argument="method")
    method_norm = method.lower()
    if method_norm == "relative_magnitudes":
        method_norm = "relative_magnitude"
    if method_norm not in {"smoothness", "relative_magnitude"}:
        raise MethodIncompatibility(
            f"method must be 'smoothness' or 'relative_magnitude', got '{method}'",
            recovery_hint="Use method='smoothness' or method='relative_magnitude'.",
            diagnostics={"method": method},
        )
    es = _extract_event_study(result)

    target = es[es["relative_time"] == e]
    if len(target) == 0:
        raise DataInsufficient(
            f"No estimate at relative time e={e}",
            recovery_hint="Choose an `e` value present in the event-study table.",
            diagnostics={"e": e, "available": es["relative_time"].tolist()},
        )

    theta = float(target["att"].iloc[0])
    se = float(target["se"].iloc[0])
    z_crit = stats.norm.ppf(1 - alpha / 2)

    if method_norm == "smoothness":
        moments = event_study_moments(result)
        if moments is not None:
            beta, sigma, times = moments
            post_mask = times >= 0
            n_pre = int((~post_mask).sum())
            post_times = times[post_mask]
            if n_pre >= 1 and e in set(post_times.tolist()):
                order = np.concatenate(
                    [np.where(~post_mask)[0], np.where(post_mask)[0]]
                )
                return breakdown_m_sd(
                    beta[order],
                    sigma[np.ix_(order, order)],
                    n_pre=n_pre,
                    n_post=int(post_mask.sum()),
                    l_post=(post_times == e).astype(float),
                    alpha=alpha,
                )
        warnings.warn(
            "breakdown_m(method='smoothness'): the event-study covariance is "
            "unavailable, so this inverts honest_did's worst-case-bias "
            "fallback interval (theta_hat +/- M*(e+1) +/- z*SE) rather than "
            "the Rambachan-Roth FLCI. Pass a Callaway-Sant'Anna result, which "
            "carries influence functions.",
            UserWarning,
            stacklevel=2,
        )
        n_drift = max(e + 1, 1)
        # M* such that |theta| - M* x n_drift - z x SE = 0
        return float(max((abs(theta) - z_crit * se) / n_drift, 0.0))

    inputs = _rm_inputs(event_study_moments(result), e)
    if inputs is not None:
        return _breakdown_rm(*inputs, alpha=alpha)

    warnings.warn(
        "breakdown_m(method='relative_magnitude'): the event-study covariance "
        "is unavailable, so this inverts honest_did's worst-case-bias fallback "
        "interval (theta_hat +/- Mbar*max|pre| +/- z*SE), not the "
        "Rambachan-Roth conditional / hybrid confidence set.",
        UserWarning,
        stacklevel=2,
    )
    pre_atts = es.loc[es["relative_time"] < 0, "att"].to_numpy(dtype=float)
    max_pre = float(np.max(np.abs(pre_atts))) if pre_atts.size else 0.0
    scale = max_pre if max_pre > 0 else se
    return float(max((abs(theta) - z_crit * se) / scale, 0.0))


def _breakdown_rm(
    b, s, n_pre, n_post, l_post, *, alpha: float, tol: float = 1e-3
) -> float:
    """Largest Mbar whose C-LF Delta^RM set excludes zero (bisection).

    The set is nested in Mbar, so ``0 in CI(Mbar)`` is monotone and the
    crossing can be bracketed and bisected to ``tol``.
    """

    def excludes_zero(m_bar: float) -> bool:
        lo, hi, _, _ = rm_confidence_set(
            b, s, n_pre, n_post, m_bar, l_vec=l_post, alpha=alpha, method="C-LF"
        )
        return bool(np.isfinite(lo) and not (lo <= 0 <= hi))

    if not excludes_zero(0.0):
        return 0.0
    hi = 1.0
    while excludes_zero(hi):
        hi *= 2.0
        if hi > 1e4:
            return float("inf")
    lo = hi / 2.0 if hi > 1.0 else 0.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if excludes_zero(mid):
            lo = mid
        else:
            hi = mid
    return float(lo)
