"""
Rambachan & Roth (2023) honest parallel trends sensitivity analysis.

Computes robust confidence sets for DID treatment effects under
violations of the parallel trends assumption. Instead of testing
whether pre-trends are exactly zero, this approach asks: "How large
could the parallel trends violation be, and would we still draw the
same conclusion?"

The key parameter M (or Delta) bounds the magnitude of possible
violations: |delta_t - delta_{t-1}| <= M for all t, where delta_t is
the deviation from parallel trends at time t.

References
----------
Rambachan, A. and Roth, J. (2023).
"A More Credible Approach to Parallel Trends."
*Review of Economic Studies*, 90(5), 2555-2591. [@rambachan2023more]

Roth, J. (2022).
"Pretest with Caution: Event-Study Estimates after Testing for
Parallel Trends."
*American Economic Review: Insights*, 4(3), 305-322. [@roth2022pretest]

Layout: this module holds :func:`honest_did` and its native tables; the
argument checks and the event-study extractor are in ``_honest_common``, the
R backend in ``_honest_r``, :func:`breakdown_m` in ``_honest_breakdown`` and
the array interface (:func:`honest_did_from_moments`) in ``_honest_moments``.
All of them are re-exported here.
"""

import warnings
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility
from ._arp_sets import rm_confidence_set
from ._flci import event_study_moments, flci_delta_sd
from ._honest_breakdown import _breakdown_rm, breakdown_m  # noqa: F401
from ._honest_common import (  # noqa: F401  (re-exported for callers)
    _FAMILY_OF_METHOD,
    _HM_CANONICAL,
    _HYBRID_OF_HM,
    _INTERVAL_OF_METHOD,
    _MOMENTS_METHOD,
    _RESTRICTION_LABEL,
    _RM_DEFAULT_GRID,
    _SOURCE_TOKENS,
    _coerce_l_vec,
    _coerce_m_grid,
    _coerce_window,
    _extract_event_study,
    _require_alpha,
    _require_int,
    _require_string,
    _resolve_restriction,
    _rm_inputs,
    _scope_attrs,
    _target_kind,
    _vcov_source,
    _window_filter,
)
from ._honest_moments import (  # noqa: F401
    _moments_from_fit,
    event_study_from_moments,
    honest_did_from_moments,
    honest_did_from_result,
)
from ._honest_r import _find_rscript, _honest_did_r_backend  # noqa: F401


def honest_did(
    result: CausalResult,
    e: int = 0,
    m_grid: Optional[List[float]] = None,
    method: str = "smoothness",
    alpha: float = 0.05,
    backend: str = "native",
    honestdid_method: Optional[str] = None,
    l_vec: Optional[Any] = None,
    window: Optional[Tuple[float, float]] = None,
    grid_lb: Optional[float] = None,
    grid_ub: Optional[float] = None,
    grid_points: int = 1000,
    grid_expand: bool = True,
    bias_direction: Optional[str] = None,
    monotonicity_direction: Optional[str] = None,
) -> pd.DataFrame:
    """
    Rambachan & Roth (2023) sensitivity analysis for parallel trends.

    Given a DID result (from Callaway-Sant'Anna or Sun-Abraham),
    computes robust confidence intervals for the treatment effect at
    relative time ``e`` under different magnitudes of parallel trends
    violations.

    Parameters
    ----------
    result : CausalResult
        A fitted DID result that contains event study estimates in
        ``result.model_info['event_study']``.
    e : int, default 0
        Relative time period for which to compute robust CIs.
        0 = on-impact effect, 1 = one period after, etc.
    m_grid : list of float, optional
        Grid of M values (maximum violation magnitude) to evaluate.
        Default for ``'smoothness'``: [0, 0.5σ, 1σ, 1.5σ, 2σ, 3σ] where σ
        is the SE at e. Default for ``'relative_magnitude'``: ten equally
        spaced M̄ from 0 to 2 (``HonestDiD``'s ``Mbarvec``), since M̄ is a
        unitless ratio to the largest pre-period violation.
    method : str, default 'smoothness'
        Type of restriction on violations:
        - ``'smoothness'``: |Δ²δ_t| ≤ M (``Delta^SD``: the slope of the
          violation changes by at most M per period)
        - ``'relative_magnitude'``: |Δδ_post| ≤ M̄ × max|Δδ_pre|
          (``Delta^RM``)
        - ``'smoothness_relative_magnitude'`` (alias ``'sdrm'``):
          |Δ²δ_post| ≤ M̄ × max|Δ²δ_pre| (``Delta^SDRM``, HonestDiD's
          ``bound = "deviation from linear trend"``); needs two or more
          pre-periods.
        Each can be intersected with a sign restriction
        (``bias_direction``) or a shape restriction
        (``monotonicity_direction``).
    alpha : float, default 0.05
        Significance level.
    backend : {'native', 'honestdid', 'r'}, default 'native'
        ``'native'`` solves the problem in Python.

        For ``method='smoothness'`` it computes the true Rambachan-Roth
        **fixed-length confidence interval**: the optimal affine estimator
        under ``Δ^SD(M)``, obtained from a convex program over the
        event-study covariance. With ``HonestDiD``'s Monte-Carlo
        folded-normal quantile replaced by the exact one, the half-length
        agrees with R ``HonestDiD::findOptimalFLCI`` to 1e-8 and the bounds
        to ~2e-5 (HonestDiD's coarser search over ``h``); against the
        shipped package the gap is that quantile's simulation error, ~1e-3
        (``tests/reference_parity/test_did_synth_R_parity.py``). This needs
        the joint event-study covariance, which is recovered from a
        Callaway-Sant'Anna fit's influence functions; if it is
        unavailable the function falls back to a worst-case-bias interval
        and warns.

        For ``method='relative_magnitude'`` it inverts the Andrews-Roth-Pakes
        conditional test (``honestdid_method='Conditional'``) or the
        conditional least-favourable hybrid (``'C-LF'``, the default, as in
        ``HonestDiD``) over the union of ``Δ^RM`` pieces, on
        ``HonestDiD``'s default grid of 1,000 points spanning ±20 standard
        deviations of the target. On identical inputs the ``'Conditional'``
        set equals R ``HonestDiD::createSensitivityResults_relativeMagnitudes``
        exactly (same accepted grid points); the ``'C-LF'`` hybrid simulates
        its first-stage critical value (NumPy draws, not R's), so it agrees
        up to that Monte Carlo error. Like the FLCI it needs the joint
        event-study covariance; without it the function falls back to the
        worst-case-bias approximation ``θ̂ ± M̄·max|δ_pre| ± z·SE`` and warns.
        A set that reaches an end of the default grid is not truncated
        there: with ``grid_expand=True`` that side is extended at the same
        step until the set closes (see ``grid_expand``).

        ``'honestdid'``/``'r'`` delegates to the R ``HonestDiD`` package
        through ``Rscript``. It is handed the same full covariance when
        one is recoverable; only when it is not does it fall back to
        ``diag(se²)``, which discards the (large, positive) cross-period
        covariance and materially changes the confidence set.
    honestdid_method : {'C-LF', 'Conditional', 'FLCI', 'C-F'}, optional
        Inference method. The default preserves HonestDiD defaults:
        ``'C-LF'`` for the relative-magnitude restrictions, ``'FLCI'`` for
        ``method='smoothness'``, and ``'C-F'`` for ``method='smoothness'``
        with a sign or shape restriction (the FLCI ignores those, so asking
        for it then warns, as HonestDiD does). Both backends implement all
        four for smoothness (``'Conditional'``: the Andrews-Roth-Pakes
        conditional test; ``'C-F'``: the conditional-FLCI hybrid;
        ``'C-LF'``: the conditional least-favourable hybrid) and
        ``'C-LF'`` / ``'Conditional'`` for the relative-magnitude families.
        ``'Conditional'`` involves no simulation and accepts the same grid
        points as HonestDiD; ``'C-LF'`` simulates its first-stage critical
        value (NumPy draws, not R's) and so agrees up to that Monte Carlo
        error.
    l_vec : sequence of float or ``'average'``, optional
        Target ``l' tau_post`` instead of the single period ``e`` (HonestDiD's
        ``l_vec``): one weight per post-treatment event time of the
        (windowed) event study, in increasing order.  ``'average'`` is the
        mean of the post-treatment effects.  ``e`` is ignored when given.
        Both backends: ``backend='r'`` passes the weights to HonestDiD.
    window : tuple of (lo, hi), optional
        Use only the event times with ``lo <= t <= hi`` (inclusive; must
        straddle zero), e.g. ``(-5, 5)`` for an analysis run on Stata
        ``estat event, window(-5 5)``.  The leads outside the window no
        longer calibrate the restriction.  Both backends.
    grid_lb, grid_ub : float, optional
        Test-inverted sets only (every interval except the FLCI): the ends of
        the grid for ``l' tau_post`` (HonestDiD's ``grid.lb`` / ``grid.ub``).
        Default ``-/+ 20 sd(l' tau_post)`` for the relative-magnitude
        families; for smoothness the identified set at ``beta = 0`` widened
        by ``20 sd`` (``'Conditional'`` / ``'C-LF'``) or the level
        ``alpha/10`` FLCI (``'C-F'``), as HonestDiD lays them out. A bound
        given here is honoured as is and never extended. Native backend;
        ``backend='r'`` passes them for the relative-magnitude families.
    grid_points : int, default 1000
        Points on that grid (HonestDiD's ``gridPoints``).
    grid_expand : bool, default True
        Native backend, test-inverted sets: when the accepted set reaches an
        end of the default grid, extend that side by another ``grid_points``
        points at the same step (up to 10 times) instead of reporting the
        grid end as the bound. HonestDiD (R) and versions of StatsPAI before
        1.33 report the grid end, which understates the interval whenever
        the set is wider than the grid; ``False`` reproduces that. A set
        still open after the extensions is reported with a warning and
        ``attrs['open_at']``. The ``'C-F'`` set lies inside its first-stage
        FLCI, its grid, and is never extended. ``backend='r'`` runs
        HonestDiD's own grid: widen it there with ``grid_lb`` / ``grid_ub``.
    bias_direction : {'positive', 'negative'}, optional
        Sign restriction on the post-treatment violation: ``delta_post >= 0``
        or ``<= 0`` (HonestDiD's ``biasDirection``; ``Delta^SDPB``,
        ``Delta^RMNB`` ...). Use it when the confound can only push the
        estimate one way. Not together with ``monotonicity_direction``.
        Needs the joint event-study covariance.
    monotonicity_direction : {'increasing', 'decreasing'}, optional
        Shape restriction: the violation ``delta`` is monotone in event time
        through the reference period (HonestDiD's ``monotonicityDirection``;
        ``Delta^SDI``, ``Delta^RMD`` ...). Not together with
        ``bias_direction``. Needs the joint event-study covariance.

    Returns
    -------
    pd.DataFrame
        Columns: M, ci_lower, ci_upper, rejects_zero.
        Each row is a different M value.  For test-inverted sets,
        ``attrs['grid_extended_at']`` lists the ``M`` whose grid was
        extended and ``attrs['open_at']`` those whose set still reaches the
        grid edge (bound unknown). ``attrs['delta']`` names the restriction
        set as HonestDiD does (``'DeltaSD'``, ``'DeltaSDRMPB'`` ...).

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
    >>> sensitivity = sp.honest_did(r, e=0)
    >>> list(sensitivity.columns)
    ['M', 'ci_lower', 'ci_upper', 'rejects_zero']
    >>> # If rejects_zero=True even at M=1 sigma, the result is robust

    Notes
    -----
    The method constructs the identified set for the treatment effect
    under the restriction that parallel trends violations are bounded:

    For ``method='smoothness'``:
        The post-treatment violation δ_post is bounded by extrapolating
        pre-trend violations. If pre-trends show a slope, the post
        period could continue that slope by at most M per period.

        The robust CI at level 1-α is:
        [θ̂ - bias_bound - z_{α/2} × SE,  θ̂ + bias_bound + z_{α/2} × SE]

        where bias_bound = M × (e + 1) is the worst-case bias under
        the smoothness restriction.

    For ``method='relative_magnitude'``:
        The post-treatment violation is bounded as a multiple of the
        largest pre-treatment violation observed. M̄ = 1 means post
        violations can be as large as the largest pre violation.

    See Rambachan & Roth (2023, *ReStud*), Section 2.

    References
    ----------
    Rambachan, A. and Roth, J. (2023). A more credible approach to parallel
    trends. *Review of Economic Studies*. [@rambachan2023more]
    """
    e = _require_int(e, argument="e")
    method = _require_string(method, argument="method")
    alpha = _require_alpha(alpha)
    backend = _require_string(backend, argument="backend")
    backend_norm = backend.lower().replace("-", "_")
    win = _coerce_window(window)
    source = _vcov_source(result)
    family, hm, bias, mono = _resolve_restriction(
        method, honestdid_method, bias_direction, monotonicity_direction
    )
    if hm == "FLCI" and (bias is not None or mono is not None):
        warnings.warn(
            "honest_did: honestdid_method='FLCI' does not use the sign or shape "
            "restriction (HonestDiD warns the same); the interval below is "
            "the FLCI under smoothness alone. Use honestdid_method='C-F' (the "
            "default with a restriction), 'Conditional' or 'C-LF' to impose it.",
            UserWarning,
            stacklevel=2,
        )
        bias = mono = None
    shape = "sign" if bias is not None else ("monotone" if mono is not None else "none")
    varying_base = (getattr(result, "model_info", None) or {}).get(
        "base_period"
    ) == "varying"
    if varying_base:
        warnings.warn(
            "honest_did: this event study uses base_period='varying', so "
            "each pre-treatment coefficient is a one-period change, not the "
            "pre-trend path relative to the reference period that the "
            "smoothness and relative-magnitude restrictions bound. The "
            "intervals below are not the Rambachan-Roth ones. Refit with "
            "callaway_santanna(..., base_period='universal').",
            UserWarning,
            stacklevel=2,
        )

    def _stamp(out: pd.DataFrame, code_path: str) -> pd.DataFrame:
        if varying_base:
            out.attrs["base_period"] = "varying"
        # What sp.validation_scope reads back from the table.
        out.attrs.update(
            _scope_attrs(
                out,
                method=_RESTRICTION_LABEL[family],
                alpha=alpha,
                l_vec=l_vec,
                window=win,
                code_path=code_path,
                source=source,
                honestdid_method=hm,
                custom_grid=(
                    grid_lb is not None or grid_ub is not None or grid_points != 1000
                ),
                shape=shape,
            )
        )
        if "delta" not in out.attrs:
            from ._honest_sets import delta_name

            out.attrs["delta"] = (
                "DeltaSD"
                if out.attrs.get("interval") in ("flci", "worst_case_bias")
                and family == "smoothness"
                else delta_name(family, bias, mono)
            )
        return out

    if backend_norm in {"r", "honestdid", "honest_did", "honestdid_r"}:
        return _stamp(
            _honest_did_r_backend(
                result=result,
                e=e,
                m_grid=m_grid,
                method=_RESTRICTION_LABEL[family],
                alpha=alpha,
                honestdid_method=hm,
                l_vec=l_vec,
                window=win,
                grid_lb=grid_lb,
                grid_ub=grid_ub,
                grid_points=grid_points,
                bias_direction=bias,
                monotonicity_direction=mono,
            ),
            "r",
        )
    if backend_norm not in {"native", "statspai"}:
        raise MethodIncompatibility(
            "backend must be 'native', 'honestdid', or 'r'.",
            recovery_hint=(
                "Use backend='native' unless exact R HonestDiD parity is " "required."
            ),
            diagnostics={"backend": backend},
        )

    _mi = getattr(result, "model_info", None) or {}
    if _mi.get("event_vcov") is not None and "cohorts" in _mi:
        # A raw sp.etwfe fit stores cohort-by-period cells; its event study
        # is the event-time aggregation of those cells.
        from .wooldridge_did import etwfe_emfx

        result = etwfe_emfx(result, type="event", include_leads=True)

    es = _extract_event_study(result)
    if win is not None:
        es = es[(es["relative_time"] >= win[0]) & (es["relative_time"] <= win[1])]
    if not (es["relative_time"] < 0).any():
        # Both restrictions are calibrated by the pre-treatment coefficients.
        # Without any, the old fallback returned theta +/- z*SE (relative
        # magnitudes) under a "covariance unavailable" warning, which reads
        # as a robustness result. The R backend raises the same error.
        raise DataInsufficient(
            "honest_did needs at least one pre-treatment event-study "
            "coefficient; this event study has none"
            + (f" inside window={win}" if win is not None else "")
            + ".",
            recovery_hint=(
                "Refit the event study with leads (e.g. callaway_santanna with "
                "base_period='universal', or an event study that reports "
                "pre-periods), or widen `window`."
            ),
            diagnostics={
                "available": sorted(int(t) for t in es["relative_time"].tolist()),
                "window": win,
            },
        )
    z_crit = stats.norm.ppf(1 - alpha / 2)
    moments = _window_filter(event_study_moments(result), win)

    # Target: the period e, or l' tau_post over the post-treatment periods.
    lv: Optional[np.ndarray] = None
    if l_vec is not None:
        if moments is not None:
            post_t = moments[2][moments[2] >= 0]
        else:
            post_t = np.sort(
                es.loc[es["relative_time"] >= 0, "relative_time"].to_numpy(dtype=int)
            )
        lv = _coerce_l_vec(l_vec, post_t)
        if moments is not None:
            b, S, t = moments
            pm = t >= 0
            theta_hat = float(lv @ b[pm])
            se_hat = float(np.sqrt(lv @ S[np.ix_(pm, pm)] @ lv))
        else:
            post_es = es[es["relative_time"] >= 0].sort_values("relative_time")
            theta_hat = float(lv @ post_es["att"].to_numpy(dtype=float))
            se_hat = float(np.sqrt(lv**2 @ post_es["se"].to_numpy(dtype=float) ** 2))
        # the worst-case fallback drifts to the furthest weighted period
        e = int(post_t[np.flatnonzero(lv != 0)].max())
    else:
        target_row = es[es["relative_time"] == e]
        if len(target_row) == 0:
            raise DataInsufficient(
                f"No event study estimate at relative time e={e}",
                recovery_hint="Choose an `e` value present in the event-study table.",
                diagnostics={"e": e, "available": es["relative_time"].tolist()},
            )
        theta_hat = float(target_row["att"].iloc[0])
        se_hat = float(target_row["se"].iloc[0])

    # Pre-treatment estimates (for calibrating M)
    pre = es[es["relative_time"] < 0].sort_values("relative_time")
    pre_atts = pre["att"].values

    # Default M grid: multiples of SE (smoothness) or HonestDiD's M-bar
    # grid (relative magnitudes).
    m_grid = _coerce_m_grid(
        m_grid, se_hat=se_hat, relative_magnitude=(family != "smoothness")
    )

    rows = []

    if hm != "FLCI" and (family != "rm" or bias is not None or mono is not None):
        # Test-inverted sets beyond plain relative magnitudes: smoothness
        # with the conditional test or a hybrid, a sign or shape
        # restriction, or relative magnitudes of second differences.
        return _stamp(
            _native_restricted_table(
                moments,
                e,
                lv,
                m_grid,
                alpha,
                family,
                hm,
                bias,
                mono,
                grid_lb=grid_lb,
                grid_ub=grid_ub,
                grid_points=grid_points,
                grid_expand=grid_expand,
            ),
            "native",
        )

    if family == "smoothness":
        # Preferred path: the actual Rambachan-Roth fixed-length confidence
        # interval, which needs the joint event-study covariance.  It is
        # recoverable from a Callaway-Sant'Anna fit's influence functions;
        # when it is not (no influence functions attached) we fall back to the
        # worst-case-bias approximation below and say so.
        _moments = moments
        if _moments is not None:
            _beta, _sigma, _times = _moments
            _post_mask = _times >= 0
            _n_pre = int((~_post_mask).sum())
            _post_times = _times[_post_mask]
            if _n_pre >= 1 and (lv is not None or e in set(_post_times.tolist())):
                _l_post = lv if lv is not None else (_post_times == e).astype(float)
                # Reorder to pre-then-post, which is what flci_delta_sd wants.
                _order = np.concatenate(
                    [np.where(~_post_mask)[0], np.where(_post_mask)[0]]
                )
                _b = _beta[_order]
                _s = _sigma[np.ix_(_order, _order)]
                for M in m_grid:
                    _res = flci_delta_sd(
                        _b,
                        _s,
                        n_pre=_n_pre,
                        n_post=int(_post_mask.sum()),
                        m_bar=float(M),
                        l_post=_l_post,
                        alpha=alpha,
                    )
                    rows.append(
                        {
                            "M": float(M),
                            "ci_lower": float(_res.ci_lower),
                            "ci_upper": float(_res.ci_upper),
                            "rejects_zero": not (_res.ci_lower <= 0 <= _res.ci_upper),
                        }
                    )
                _out = pd.DataFrame(rows)
                _out.attrs["interval"] = "flci"
                return _stamp(_out, "native")

        warnings.warn(
            "honest_did(method='smoothness'): the event-study covariance is "
            "unavailable, so this falls back to a worst-case-bias interval "
            "(theta_hat +/- M*(e+1) +/- z*SE) rather than the Rambachan-Roth "
            "FLCI. That approximation ignores the pre-period covariance and "
            "can be narrower than the true confidence set, overstating "
            "robustness. The estimator did not expose a joint event-study "
            "covariance (see sp.event_study_vcov); refit with an estimator "
            "that does, or pass backend='r'.",
            UserWarning,
            stacklevel=2,
        )

        # Fallback: worst-case bias = M × (number of post periods from base).
        n_drift = max(e + 1, 1)

        for M in m_grid:
            bias_bound = M * n_drift
            ci_lower = theta_hat - bias_bound - z_crit * se_hat
            ci_upper = theta_hat + bias_bound + z_crit * se_hat
            rejects = not (ci_lower <= 0 <= ci_upper)

            rows.append(
                {
                    "M": float(M),
                    "ci_lower": float(ci_lower),
                    "ci_upper": float(ci_upper),
                    "rejects_zero": rejects,
                }
            )

    else:
        # Preferred path: the Rambachan-Roth / ARP confidence set, which needs
        # the joint event-study covariance (as the FLCI does).
        _rm = _native_rm_table(
            moments,
            e,
            m_grid,
            alpha,
            hm,
            l_vec=lv,
            grid_lb=grid_lb,
            grid_ub=grid_ub,
            grid_points=grid_points,
            grid_expand=grid_expand,
        )
        if _rm is not None:
            return _stamp(_rm, "native")

        warnings.warn(
            "honest_did(method='relative_magnitude'): the event-study "
            "covariance is unavailable, so this falls back to a worst-case-bias "
            "interval (theta_hat +/- Mbar*max|pre| +/- z*SE) rather than the "
            "Rambachan-Roth conditional / hybrid confidence set. It ignores the "
            "pre-period covariance and can overstate robustness. Refit with an "
            "estimator that exposes the joint event-study covariance (see "
            "sp.event_study_vcov), or pass backend='r'.",
            UserWarning,
            stacklevel=2,
        )
        # Relative magnitude: |δ_post| ≤ M̄ × max|δ_pre|
        if len(pre_atts) == 0:
            max_pre = 0
        else:
            max_pre = np.max(np.abs(pre_atts))

        for M_bar in m_grid:
            bias_bound = M_bar * max_pre if max_pre > 0 else M_bar * se_hat
            ci_lower = theta_hat - bias_bound - z_crit * se_hat
            ci_upper = theta_hat + bias_bound + z_crit * se_hat
            rejects = not (ci_lower <= 0 <= ci_upper)

            rows.append(
                {
                    "M": float(M_bar),
                    "ci_lower": float(ci_lower),
                    "ci_upper": float(ci_upper),
                    "rejects_zero": rejects,
                }
            )

    _out = pd.DataFrame(rows)
    _out.attrs["interval"] = "worst_case_bias"
    return _stamp(_out, "native")


def _native_restricted_table(
    moments: Optional[tuple],
    e: int,
    l_vec: Optional[np.ndarray],
    m_grid: Sequence[float],
    alpha: float,
    family: str,
    hm: str,
    bias: Optional[str],
    mono: Optional[str],
    grid_lb: Optional[float] = None,
    grid_ub: Optional[float] = None,
    grid_points: int = 1000,
    grid_expand: bool = True,
) -> pd.DataFrame:
    """Native table for the restriction sets of :mod:`._honest_sets`.

    These sets have no worst-case-bias stand-in, so a missing joint
    covariance is an error rather than a warned fallback.
    """
    from ._honest_sets import delta_name, restricted_table

    inputs = _rm_inputs(moments, e, l_vec)
    if inputs is None:
        raise MethodIncompatibility(
            f"honest_did: {delta_name(family, bias, mono)} with "
            f"honestdid_method={hm!r} needs the joint covariance of the "
            "event-study coefficients, which this result does not expose.",
            recovery_hint="Refit with an estimator that stores it (see "
            "sp.event_study_vcov), pass the coefficients and covariance to "
            "sp.honest_did_from_moments, or use backend='r'.",
            diagnostics={"restriction": delta_name(family, bias, mono)},
        )
    grid_points = _require_int(grid_points, argument="grid_points")
    if grid_points < 2:
        raise MethodIncompatibility(
            "grid_points must be at least 2.",
            recovery_hint="HonestDiD's default is 1000.",
            diagnostics={"grid_points": grid_points},
        )
    out = restricted_table(
        inputs,
        m_grid,
        alpha,
        _HYBRID_OF_HM[hm],
        family,
        bias,
        mono,
        grid_lb,
        grid_ub,
        grid_points,
        grid_expand,
    )
    if out.attrs["open_at"]:
        warnings.warn(
            f"honest_did ({out.attrs['delta']}, {hm}): the confidence set "
            "reaches the edge of the test-inversion grid at M = "
            f"{out.attrs['open_at']}; the reported bound is the grid edge, not "
            "the true bound. "
            + (
                "Pass grid_expand=True (the default) or a wider grid_lb / grid_ub."
                if not grid_expand or grid_lb is not None or grid_ub is not None
                else "The set did not close within 10 extensions and may be "
                "unbounded at this M."
            ),
            UserWarning,
            stacklevel=3,
        )
    return out


def _native_rm_table(
    moments: Optional[tuple],
    e: int,
    m_grid: Sequence[float],
    alpha: float,
    honestdid_method: Optional[str],
    l_vec: Optional[np.ndarray] = None,
    grid_lb: Optional[float] = None,
    grid_ub: Optional[float] = None,
    grid_points: int = 1000,
    grid_expand: bool = True,
) -> Optional[pd.DataFrame]:
    """Native Delta^RM confidence sets over ``m_grid`` (None if no covariance)."""
    inputs = _rm_inputs(moments, e, l_vec)
    if inputs is None:
        return None
    b, s, n_pre, n_post, l_post = inputs
    rm_method = "C-LF" if honestdid_method is None else str(honestdid_method)
    grid_points = _require_int(grid_points, argument="grid_points")
    if grid_points < 2:
        raise MethodIncompatibility(
            "grid_points must be at least 2.",
            recovery_hint="HonestDiD's default is 1000.",
            diagnostics={"grid_points": grid_points},
        )
    rows = []
    open_at: List[float] = []
    extended_at: List[float] = []
    for m_bar in m_grid:
        lo, hi, grid, accept = rm_confidence_set(
            b,
            s,
            n_pre,
            n_post,
            float(m_bar),
            l_vec=l_post,
            alpha=alpha,
            method=rm_method,
            grid_points=grid_points,
            grid_lb=grid_lb,
            grid_ub=grid_ub,
            max_expand=10 if grid_expand else 0,
        )
        if grid.size > grid_points:
            extended_at.append(float(m_bar))
        if accept.size and (accept[0] == 1 or accept[-1] == 1):
            open_at.append(float(m_bar))
        rows.append(
            {
                "M": float(m_bar),
                "ci_lower": lo,
                "ci_upper": hi,
                "rejects_zero": bool(np.isfinite(lo) and not (lo <= 0 <= hi)),
            }
        )
    if open_at:
        warnings.warn(
            "honest_did(method='relative_magnitude'): the confidence set reaches "
            f"the edge of the test-inversion grid at Mbar = {open_at}; the "
            "reported bound is the grid edge, not the true bound. "
            + (
                "Pass grid_expand=True (the default) or a wider grid_lb / grid_ub."
                if not grid_expand or grid_lb is not None or grid_ub is not None
                else "The set did not close within 10 extensions and may be "
                "unbounded at this Mbar."
            ),
            UserWarning,
            stacklevel=3,
        )
    out = pd.DataFrame(rows)
    out.attrs["grid_extended_at"] = extended_at
    out.attrs["open_at"] = open_at
    out.attrs["interval"] = (
        "arp_conditional" if rm_method.upper() == "CONDITIONAL" else "arp_c_lf"
    )
    return out


# Citation
# Citation
CausalResult._CITATIONS["honest_did"] = (
    "@article{rambachan2023more,\n"
    "  title={A More Credible Approach to Parallel Trends},\n"
    "  author={Rambachan, Ashesh and Roth, Jonathan},\n"
    "  journal={Review of Economic Studies},\n"
    "  volume={90},\n"
    "  number={5},\n"
    "  pages={2555--2591},\n"
    "  year={2023},\n"
    "  publisher={Oxford University Press}\n"
    "}"
)
