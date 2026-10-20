"""Array interface of the Rambachan-Roth sensitivity analysis.

:func:`event_study_from_moments` wraps a coefficient vector and its joint
covariance as an event-study result; :func:`honest_did_from_moments` runs
:func:`honest_did` on it (R ``HonestDiD::createSensitivityResults``).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..core.results import CausalResult
from ..exceptions import MethodIncompatibility
from ._honest_common import _MOMENTS_METHOD


def event_study_from_moments(
    betahat: Any,
    sigma: Any = None,
    event_times: Optional[Union[Sequence[int], Dict[str, int]]] = None,
    num_pre_periods: Optional[int] = None,
    alpha: float = 0.05,
) -> CausalResult:
    """An event-study result built from coefficients and their covariance.

    The common input of :func:`honest_did_from_moments`,
    :func:`pretrends_power` and :func:`pretrends_slope_for_power` when the
    event study was not fitted by a StatsPAI event-study estimator: a
    hand-built lead/lag regression, or numbers taken from a paper. See
    :func:`honest_did_from_moments` for the arguments.
    """
    if hasattr(betahat, "params") and not isinstance(betahat, pd.Series):
        betahat, sigma, event_times = _moments_from_fit(betahat, sigma, event_times)
    if sigma is None:
        raise MethodIncompatibility(
            "sigma (the joint covariance of betahat) is required.",
            recovery_hint="Pass sigma=, or a fitted result with "
            "event_times={name: time}.",
            diagnostics={},
        )
    b = np.asarray(betahat, dtype=float).ravel()
    S = np.asarray(sigma, dtype=float)
    if S.shape != (b.size, b.size):
        raise MethodIncompatibility(
            f"sigma must be {b.size} x {b.size} to match betahat; got {S.shape}.",
            recovery_hint="Pass the joint covariance of the event-study coefficients.",
            diagnostics={"n_beta": int(b.size), "sigma_shape": list(S.shape)},
        )
    if not (np.all(np.isfinite(b)) and np.all(np.isfinite(S))):
        raise MethodIncompatibility(
            "betahat and sigma must be finite.",
            recovery_hint="Drop the omitted reference period before passing.",
            diagnostics={},
        )
    if event_times is None:
        if num_pre_periods is None:
            raise MethodIncompatibility(
                "Pass event_times, or num_pre_periods for HonestDiD's layout.",
                recovery_hint="e.g. event_times=[-4, -3, -2, 0, 1, 2].",
                diagnostics={},
            )
        k = int(num_pre_periods)
        if not 1 <= k < b.size:
            raise MethodIncompatibility(
                "num_pre_periods must leave at least one post period.",
                recovery_hint="Check the coefficient count.",
                diagnostics={"num_pre_periods": k, "n_beta": int(b.size)},
            )
        times = np.r_[np.arange(-k - 1, -1), np.arange(b.size - k)]
    else:
        times = np.asarray(event_times, dtype=int).ravel()
        if times.size != b.size or len(set(times.tolist())) != b.size:
            raise MethodIncompatibility(
                "event_times must hold one distinct time per coefficient.",
                recovery_hint="Drop the omitted reference period from both.",
                diagnostics={"n_beta": int(b.size), "n_times": int(times.size)},
            )
    if not ((times < 0).any() and (times >= 0).any()):
        raise MethodIncompatibility(
            "Need both pre-treatment (< 0) and post-treatment (>= 0) coefficients.",
            recovery_hint="HonestDiD calibrates the restriction on the leads.",
            diagnostics={"event_times": times.tolist()},
        )
    order = np.argsort(times)
    b, S, times = b[order], S[np.ix_(order, order)], times[order]
    se = np.sqrt(np.clip(np.diag(S), 0.0, None))
    detail = pd.DataFrame({"relative_time": times, "att": b, "se": se})
    res = CausalResult(
        method=_MOMENTS_METHOD,
        estimand="ATT",
        estimate=float(b[times >= 0][0]),
        se=float(se[times >= 0][0]),
        pvalue=float("nan"),
        ci=(float("nan"), float("nan")),
        alpha=alpha,
        n_obs=0,
        detail=detail,
        model_info={
            "aggregation": "dynamic",
            "vcov": S,
            # the block the pre-trend functions read
            "vcv_pre": S[np.ix_(times < 0, times < 0)],
        },
    )
    return res


def honest_did_from_moments(
    betahat: Any,
    sigma: Any = None,
    event_times: Optional[Union[Sequence[int], Dict[str, int]]] = None,
    num_pre_periods: Optional[int] = None,
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
    """Rambachan-Roth sensitivity from a coefficient vector and its covariance.

    The array interface of R ``HonestDiD::createSensitivityResults(betahat,
    sigma, numPrePeriods, numPostPeriods, ...)``: pass the event-study
    coefficients of *any* estimator -- a ``reghdfe`` / ``sp.hdfe_ols`` /
    ``sp.feols`` event study, a stacked regression, estimates copied from a
    paper -- with their joint covariance, and run the same solvers as
    :func:`honest_did`.

    Parameters
    ----------
    betahat : sequence of float, or a fitted regression
        Event-study coefficients, the reference period excluded. Or a fitted
        result with ``params`` and a full covariance (``sp.hdfe_ols``,
        ``sp.regress``, ``sp.feols`` ...) together with ``event_times`` as a
        ``{coefficient name: event time}`` mapping: the coefficients and
        their covariance block are taken from the fit, so a hand-built
        intensity event study (``exposure x 1[t = k]`` regressors) goes
        straight in.
    sigma : array-like
        Their joint covariance matrix (``len(betahat)`` square). Use the full
        matrix: the pre-period covariance is what calibrates the restriction.
        Omitted when ``betahat`` is a fitted result.
    event_times : sequence of int or dict, optional
        Relative time of each coefficient (negative = pre-treatment,
        ``>= 0`` = post); a ``{name: time}`` dict with a fitted ``betahat``.
        If omitted, ``num_pre_periods`` must be given and the
        times are ``-num_pre_periods-1 .. -2`` for the leads (``-1`` is the
        omitted reference) and ``0, 1, ...`` for the lags -- HonestDiD's
        layout.
    num_pre_periods : int, optional
        Number of leading pre-treatment coefficients when ``event_times`` is
        omitted.
    e, m_grid, method, alpha, backend, honestdid_method, l_vec, window
        As in :func:`honest_did`.
    grid_lb, grid_ub, grid_points, grid_expand
        Test-inversion grid, as in :func:`honest_did`.
    bias_direction, monotonicity_direction
        Sign or shape restriction on the violation, as in :func:`honest_did`
        (HonestDiD's ``biasDirection`` / ``monotonicityDirection``).

    Returns
    -------
    pd.DataFrame
        Columns ``M, ci_lower, ci_upper, rejects_zero`` (as :func:`honest_did`).

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> b = np.array([0.02, -0.01, 0.00, 0.30, 0.35])
    >>> V = np.diag([0.02, 0.02, 0.02, 0.03, 0.04]) ** 2
    >>> out = sp.honest_did_from_moments(
    ...     b, V, event_times=[-4, -3, -2, 0, 1], e=0, m_grid=[0.0, 0.02]
    ... )
    >>> list(out.columns)[:3]
    ['M', 'ci_lower', 'ci_upper']
    """
    res = event_study_from_moments(
        betahat,
        sigma,
        event_times=event_times,
        num_pre_periods=num_pre_periods,
        alpha=alpha,
    )
    from .honest_did import honest_did

    return honest_did(
        res,
        e=e,
        m_grid=m_grid,
        method=method,
        alpha=alpha,
        backend=backend,
        honestdid_method=honestdid_method,
        l_vec=l_vec,
        window=window,
        grid_lb=grid_lb,
        grid_ub=grid_ub,
        grid_points=grid_points,
        grid_expand=grid_expand,
        bias_direction=bias_direction,
        monotonicity_direction=monotonicity_direction,
    )


def _moments_from_fit(fit: Any, sigma: Any, event_times: Any) -> tuple:
    """(betahat, sigma, times) from a fitted regression and ``{name: time}``."""
    from ..postestimation._covariance import coefficient_covariance

    if sigma is not None or not isinstance(event_times, dict) or not event_times:
        raise MethodIncompatibility(
            "With a fitted result as betahat, pass event_times as a "
            "{coefficient name: event time} dict and no sigma.",
            recovery_hint="e.g. event_times={'lead3': -3, 'lead2': -2, "
            "'lag0': 0, 'lag1': 1}",
            diagnostics={"type": type(fit).__name__},
        )
    params = fit.params
    names = list(event_times)
    missing = [nm for nm in names if nm not in params.index]
    if missing:
        raise MethodIncompatibility(
            f"Coefficients not in the fit: {missing}.",
            recovery_hint=f"Available: {list(params.index)[:20]}",
            diagnostics={"missing": missing},
        )
    V, _ = coefficient_covariance(fit)
    if V is None:
        raise MethodIncompatibility(
            "The fit carries no covariance matrix consistent with its SEs.",
            recovery_hint="Pass betahat and sigma arrays directly.",
            diagnostics={"type": type(fit).__name__},
        )
    idx = [list(params.index).index(nm) for nm in names]
    return (
        params.to_numpy(dtype=float)[idx],
        np.asarray(V, dtype=float)[np.ix_(idx, idx)],
        [int(event_times[nm]) for nm in names],
    )


def honest_did_from_result(result: CausalResult, **kwargs: Any) -> pd.DataFrame:
    """:func:`honest_did` under the name of the MCP tool.

    The MCP server exposes ``honest_did_from_result(result_id=...)``; this is
    the same analysis on a result object, so scripts written against the tool
    names run unchanged. All keyword arguments go to :func:`honest_did`.

    Examples
    --------
    >>> import statspai as sp
    >>> sp.honest_did_from_result is not sp.honest_did
    True
    """
    from .honest_did import honest_did

    return honest_did(result, **kwargs)
