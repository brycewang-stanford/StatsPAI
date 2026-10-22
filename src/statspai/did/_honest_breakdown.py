"""Breakdown value of the Rambachan-Roth sensitivity analysis.

:func:`breakdown_m` inverts the confidence set :func:`honest_did` reports:
the largest ``M`` (or ``Mbar``) at which it still excludes zero.
"""

import warnings
from typing import Any, Optional, Tuple

import numpy as np
from scipy import stats

from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility
from ._arp_sets import rm_confidence_set
from ._flci import breakdown_m_sd, event_study_moments
from ._honest_common import (
    _HYBRID_OF_HM,
    _coerce_l_vec,
    _coerce_window,
    _extract_event_study,
    _require_alpha,
    _require_int,
    _require_string,
    _resolve_restriction,
    _rm_inputs,
    _window_filter,
)


def breakdown_m(
    result: CausalResult,
    e: int = 0,
    method: str = "smoothness",
    alpha: float = 0.05,
    honestdid_method: Optional[str] = None,
    bias_direction: Optional[str] = None,
    monotonicity_direction: Optional[str] = None,
    l_vec: Optional[Any] = None,
    window: Optional[Tuple[float, float]] = None,
    seed: int = 0,
    *,
    sigma: Any = None,
    event_times: Any = None,
    num_pre_periods: Optional[int] = None,
) -> float:
    """
    Compute the breakdown value of M.

    The breakdown M* is the largest violation magnitude under which
    the treatment effect at relative time ``e`` (or the target ``l_vec``,
    e.g. the average post-treatment effect) remains statistically
    significant. Larger M* = more robust.

    Parameters
    ----------
    result : CausalResult or array-like
        DID result with event study, or the event-study coefficients when
        ``sigma`` (or ``event_times`` / ``num_pre_periods``) is given.
    e : int, default 0
        Relative time period.
    method : str, default 'smoothness'
        The restriction whose confidence set is inverted, as in
        :func:`honest_did`: ``'smoothness'``, ``'relative_magnitude'`` or
        ``'smoothness_relative_magnitude'`` (alias ``'sdrm'``: HonestDiD's
        "deviation from linear trend", ``Delta^SDRM``; needs two or more
        pre-periods).
    alpha : float, default 0.05
    honestdid_method : {'C-LF', 'Conditional', 'FLCI', 'C-F'}, optional
        Confidence set to invert, with :func:`honest_did`'s defaults: the
        FLCI for plain smoothness, ``'C-F'`` for smoothness with a sign or
        shape restriction, ``'C-LF'`` for the relative-magnitude families.
        ``'Conditional'`` involves no simulation.
    bias_direction : {'positive', 'negative'}, optional
        Sign restriction on the post-treatment violation (HonestDiD's
        ``biasDirection``), as in :func:`honest_did`. Not together with
        ``monotonicity_direction``. Needs the joint event-study covariance.
    monotonicity_direction : {'increasing', 'decreasing'}, optional
        Shape restriction on the violation (HonestDiD's
        ``monotonicityDirection``), as in :func:`honest_did`. Needs the joint
        event-study covariance.
    l_vec : sequence of float or ``'average'``, optional
        Target ``l' tau_post`` instead of the single period ``e``, with the
        meaning it has in ``honest_did``. One weight per post-treatment event
        time of the (windowed) event study, in increasing order; ``'average'``
        is the mean of the post-treatment effects (HonestDiD's ``l_vec``).
        ``e`` is ignored when given.
    window : tuple of (lo, hi), optional
        Use only the event times with ``lo <= t <= hi``, as in
        :func:`honest_did`.
    seed : int, default 0
        Seed of the simulated first-stage critical value of the ``'C-LF'``
        hybrid (1,000 draws, HonestDiD's ``seed`` and default), so the value
        is the same on every run. Other methods do not simulate.
    sigma, event_times, num_pre_periods : optional
        Take the event study from coefficients instead of a fitted result:
        ``result`` is then the coefficient vector (reference period
        excluded) and ``sigma`` its joint covariance, or a fitted regression
        with ``event_times={coefficient name: event time}``. Same
        conventions as :func:`honest_did_from_moments`, whose
        ``breakdown=True`` stores the same number, so a breakdown value can
        be computed from the estimates and standard errors printed in a
        paper (a diagonal ``sigma`` when the paper reports no covariance).

        .. versionadded:: 1.40.1

    Returns
    -------
    float
        Breakdown value M*. The effect is significant for all M < M*.
        ``inf`` when the set excludes zero however large M is (a sign
        restriction can bound the violation on the side that matters).

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
      ``Delta^RM(Mbar)`` (HonestDiD's default): the first ``Mbar`` at which
      it contains zero, by a log-grid scan in ``Mbar`` and bisection to
      1e-3 relative (the set need not be nested in ``Mbar``).
    * ``method='relative_magnitude'`` without it: the worst-case-bias
      approximation ``theta_hat +/- (Mbar max|pre| + z SE)``, giving
      ``(|theta_hat| - z SE) / max|pre|``, with a warning.
    * every other restriction set -- a sign or shape restriction,
      ``Delta^SDRM``, or a test-inverted interval (``'Conditional'``,
      ``'C-LF'``, ``'C-F'``) under smoothness: the same test-inverted set
      :func:`honest_did` reports, bisected in ``M`` (``Mbar``) to 1e-3
      (``M``: 1e-3 standard deviations of the target). These sets are
      nested in ``M``, so the crossing is unique; the value is the largest
      ``M`` at which the reported set excludes zero, so ``honest_did`` at
      that ``M`` has ``rejects_zero=True`` and just above it ``False``.
      They need the joint covariance; without it this raises, as
      :func:`honest_did` does. R ``HonestDiD`` has no breakdown function:
      its sensitivity grid under ``method = "Conditional"`` brackets this
      value (``tests/reference_parity/test_honest_breakdown_R_parity.py``).

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
    seed = _require_int(seed, argument="seed")
    if sigma is not None or event_times is not None or num_pre_periods is not None:
        from ._honest_moments import event_study_from_moments

        result = event_study_from_moments(
            result,
            sigma,
            event_times=event_times,
            num_pre_periods=num_pre_periods,
            alpha=alpha,
        )
    method = _require_string(method, argument="method")
    win = _coerce_window(window)
    family, hm, bias, mono = _resolve_restriction(
        method, honestdid_method, bias_direction, monotonicity_direction
    )
    if hm == "FLCI" and (bias is not None or mono is not None):
        warnings.warn(
            "breakdown_m: honestdid_method='FLCI' does not use the sign or "
            "shape restriction (honest_did warns the same); this inverts the "
            "FLCI under smoothness alone. Use honestdid_method='C-F' (the "
            "default with a restriction), 'Conditional' or 'C-LF'.",
            UserWarning,
            stacklevel=2,
        )
        bias = mono = None
    es = _extract_event_study(result)
    if win is not None:
        es = es[(es["relative_time"] >= win[0]) & (es["relative_time"] <= win[1])]
    moments = _window_filter(event_study_moments(result), win)

    lv: Optional[np.ndarray] = None
    if l_vec is not None:
        # The target of honest_did(l_vec=...): l' tau_post.
        if moments is not None:
            post_t = moments[2][moments[2] >= 0]
        else:
            post_t = np.sort(
                es.loc[es["relative_time"] >= 0, "relative_time"].to_numpy(dtype=int)
            )
        lv = _coerce_l_vec(l_vec, post_t)
        if moments is not None:
            b_all, S_all, t_all = moments
            pm = t_all >= 0
            theta = float(lv @ b_all[pm])
            se = float(np.sqrt(lv @ S_all[np.ix_(pm, pm)] @ lv))
        else:
            post_es = es[es["relative_time"] >= 0].sort_values("relative_time")
            theta = float(lv @ post_es["att"].to_numpy(dtype=float))
            se = float(np.sqrt(lv**2 @ post_es["se"].to_numpy(dtype=float) ** 2))
        # the worst-case fallback drifts to the furthest weighted period
        e = int(post_t[np.flatnonzero(lv != 0)].max())
    else:
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

    plain_rm_clf = family == "rm" and bias is None and mono is None and hm == "C-LF"
    if hm != "FLCI" and not plain_rm_clf:
        # Every set other than the two defaults (the FLCI under smoothness,
        # C-LF under relative magnitudes), which keep their original paths.
        return _breakdown_restricted(
            _rm_inputs(moments, e, lv), family, hm, bias, mono, alpha, seed=seed
        )

    if family == "smoothness":
        if moments is not None:
            beta, sigma, times = moments
            post_mask = times >= 0
            n_pre = int((~post_mask).sum())
            post_times = times[post_mask]
            if n_pre >= 1 and (lv is not None or e in set(post_times.tolist())):
                order = np.concatenate(
                    [np.where(~post_mask)[0], np.where(post_mask)[0]]
                )
                return breakdown_m_sd(
                    beta[order],
                    sigma[np.ix_(order, order)],
                    n_pre=n_pre,
                    n_post=int(post_mask.sum()),
                    l_post=(post_times == e).astype(float) if lv is None else lv,
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

    inputs = _rm_inputs(moments, e, lv)
    if inputs is not None:
        return _breakdown_rm(*inputs, alpha=alpha, seed=seed)

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


def _first_inclusion(excludes_zero, unit: float = 1.0, rtol: float = 1e-3) -> float:
    """Smallest ``M`` at which the set contains zero, for sets not nested in M.

    HonestDiD's test-inverted sets under a sign or shape restriction are not
    always nested in ``M``: on the fixture of
    ``test_honest_breakdown_R_parity.py`` R's own ``Delta^RMNB`` set contains
    zero at ``Mbar = 0.0068``, excludes it at ``0.007`` and contains it again
    at ``0.01``. Bisection on a doubling bracket would then return some
    crossing, not the first one. Scan ``M`` upwards on a log grid (four
    points per octave from ``unit / 1024`` to ``1e4 * unit``) and bisect the
    first bracket in which the set comes to contain zero, to ``rtol``
    relative to its upper end (absolute floor ``1e-6 * unit``). An
    inclusion interval narrower than the grid step before that bracket is
    not seen.
    """
    if not excludes_zero(0.0):
        return 0.0
    prev = 0.0
    for m in unit * 2.0 ** (np.arange(-40, 54) / 4.0):
        if not excludes_zero(float(m)):
            lo, hi = prev, float(m)
            while hi - lo > max(rtol * hi, 1e-6 * unit):
                mid = 0.5 * (lo + hi)
                if excludes_zero(mid):
                    lo = mid
                else:
                    hi = mid
            return float(lo)
        prev = float(m)
    return float("inf")


def _excludes_zero(lo: float, hi: float) -> bool:
    # The same verdict as honest_did's ``rejects_zero`` column.
    return bool(np.isfinite(lo) and not (lo <= 0 <= hi))


def _breakdown_rm(
    b,
    s,
    n_pre,
    n_post,
    l_post,
    *,
    alpha: float,
    tol: float = 1e-3,
    seed: int = 0,
) -> float:
    """First Mbar at which the C-LF Delta^RM set contains zero.

    The test-inverted set is not guaranteed to be nested in Mbar (the
    least-favourable critical value and the union over Delta^RM pieces both
    move with Mbar), so ``0 in CI(Mbar)`` need not be monotone and a
    doubling bracket could bisect some later crossing. The log-grid scan of
    :func:`_first_inclusion` (as for every other restriction) finds the first
    one, to ``tol`` relative. Through 1.39 this path bisected a doubling
    bracket to ``tol`` absolute; on nested inputs the two agree to that
    tolerance.
    """

    def excludes_zero(m_bar: float) -> bool:
        lo, hi, _, _ = rm_confidence_set(
            b,
            s,
            n_pre,
            n_post,
            m_bar,
            l_vec=l_post,
            alpha=alpha,
            method="C-LF",
            seed=seed,
        )
        return _excludes_zero(lo, hi)

    return _first_inclusion(excludes_zero, 1.0, rtol=tol)


def _breakdown_restricted(
    inputs: Optional[tuple],
    family: str,
    hm: str,
    bias: Optional[str],
    mono: Optional[str],
    alpha: float,
    seed: int = 0,
) -> float:
    """Breakdown value of the test-inverted sets of :mod:`._honest_sets`.

    The same sets ``honest_did``'s native path reports for this restriction
    and method (default grid, extended at an accepted end); the first ``M``
    at which they contain zero is located by :func:`_first_inclusion`. Mbar
    is unitless; the smoothness ``M`` is in the target's units, so its scan
    is scaled by the target's standard deviation.
    """
    from ._honest_sets import (
        delta_name,
        rm_family_confidence_set,
        sd_family_confidence_set,
    )

    if inputs is None:
        raise MethodIncompatibility(
            f"breakdown_m: {delta_name(family, bias, mono)} with "
            f"honestdid_method={hm!r} needs the joint covariance of the "
            "event-study coefficients, which this result does not expose.",
            recovery_hint="Refit with an estimator that stores it (see "
            "sp.event_study_vcov), or pass the coefficients and their joint "
            "covariance to sp.honest_did_from_moments.",
            diagnostics={"restriction": delta_name(family, bias, mono)},
        )
    b, s, n_pre, n_post, l_post = inputs
    hybrid = _HYBRID_OF_HM[hm]
    empty_at: list = []

    def _outside(m: float, lo: float, hi: float) -> bool:
        # "Zero is not in the set": an empty set (every grid point rejected,
        # i.e. the pre-period coefficients reject the restriction at this M)
        # does not contain zero, so the scan moves on to larger M.
        if not np.isfinite(lo):
            empty_at.append(float(m))
            return True
        return not (lo <= 0 <= hi)

    def _finish(m_star: float) -> float:
        below = [m for m in empty_at if m <= m_star]
        if below:
            warnings.warn(
                f"breakdown_m ({delta_name(family, bias, mono)}, {hm}): the "
                f"confidence set is empty at M = {min(below):.6g}"
                + (f" .. {max(below):.6g}" if len(below) > 1 else "")
                + ": the pre-period coefficients reject the restriction "
                "there. The breakdown value is the first M at which the set "
                "contains zero; below it the set excludes zero or is empty.",
                UserWarning,
                stacklevel=3,
            )
        return m_star

    if family == "smoothness":

        def excludes_zero(m: float) -> bool:
            lo, hi, _, _ = sd_family_confidence_set(
                b,
                s,
                n_pre,
                n_post,
                float(m),
                l_post,
                alpha=alpha,
                hybrid=hybrid,
                bias=bias,
                mono=mono,
                seed=seed,
                max_expand=10,
            )
            return _outside(m, lo, hi)

        post = np.asarray(s, dtype=float)[n_pre:, n_pre:]
        sd_theta = float(np.sqrt(l_post @ post @ l_post))
        return _finish(_first_inclusion(excludes_zero, sd_theta))

    def excludes_zero_rm(m_bar: float) -> bool:
        lo, hi, _, _ = rm_family_confidence_set(
            b,
            s,
            n_pre,
            n_post,
            float(m_bar),
            l_post,
            alpha=alpha,
            hybrid=hybrid,
            base=family,
            bias=bias,
            mono=mono,
            seed=seed,
            max_expand=10,
        )
        return _outside(m_bar, lo, hi)

    return _finish(_first_inclusion(excludes_zero_rm, 1.0))
