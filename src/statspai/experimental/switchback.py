"""Switchback experiments: design and design-based analysis.

A switchback experiment treats one unit (a city, a marketplace) and switches
it between treatment and control over time. Nothing is assumed about how the
outcome evolves; the only randomness is the assignment path, which the
experimenter drew from a known distribution. The effect of a treatment may
carry over into the next ``m`` periods.

This module implements the design and the analysis of Bojinov, Simchi-Levi
and Zhao (2023) for *regular* switchback experiments, where a fair or biased
coin is flipped at a set of randomization points and the assignment stays put
in between:

- :func:`switchback_design` gives the randomization points, including the
  minimax-optimal ones (their Theorem 2), and draws an assignment path.
- :func:`switchback` estimates the average lag-``m`` effect with the
  Horvitz-Thompson estimator (their equation 4), tests the sharp null by
  re-drawing assignment paths (Algorithm 1) and, under the optimal design,
  reports the conservative standard error of Corollary 1.

References
----------
bojinov2023design
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["switchback", "switchback_design"]


# ---------------------------------------------------------------------- #
#  Design
# ---------------------------------------------------------------------- #


def _optimal_points(n_periods: int, m: int) -> np.ndarray:
    """Randomization points of Theorem 2 as a boolean vector of length T.

    ``m = 0``: every period. ``m > 0`` with ``T = n m`` and ``n >= 4``:
    periods ``1, 2m+1, 3m+1, ..., (n-2)m+1`` (1-based).
    """
    points = np.zeros(n_periods, dtype=bool)
    if m == 0:
        points[:] = True
        return points
    n = n_periods // m
    if n * m != n_periods or n < 4:
        raise MethodIncompatibility(
            f"The optimal switchback design needs a horizon that is a multiple "
            f"of m with at least 4 blocks; got {n_periods} periods and m={m}.",
            recovery_hint=(
                f"Use a horizon of n * {m} periods with n >= 4, or pass "
                "design='every' / an integer block length."
            ),
            diagnostics={"n_periods": n_periods, "m": m},
        )
    one_based = [1] + [k * m + 1 for k in range(2, n - 1)]
    points[np.asarray(one_based) - 1] = True
    return points


def _resolve_points(
    design: Union[str, int, Sequence[Any], np.ndarray], n_periods: int, m: int
) -> np.ndarray:
    if isinstance(design, str):
        if design == "optimal":
            return _optimal_points(n_periods, m)
        if design == "every":
            return np.ones(n_periods, dtype=bool)
        raise MethodIncompatibility(
            f"design must be 'optimal', 'every', a block length or a boolean "
            f"vector; got {design!r}."
        )
    if isinstance(design, (int, np.integer)) and not isinstance(design, bool):
        if design < 1:
            raise MethodIncompatibility("A block length must be at least 1.")
        every = np.zeros(n_periods, dtype=bool)
        every[:: int(design)] = True
        return every
    points: np.ndarray = np.asarray(design)
    if points.shape != (n_periods,):
        raise MethodIncompatibility(
            f"The randomization-point vector must have one entry per period "
            f"({n_periods}); got shape {points.shape}."
        )
    if points.dtype != bool:
        if not np.isin(points, (0, 1)).all():
            raise MethodIncompatibility(
                "The randomization-point vector must be boolean (True at a "
                "period where the coin is flipped)."
            )
        points = points.astype(bool)
    if not points[0]:
        raise MethodIncompatibility(
            "The first period must be a randomization point: the assignment "
            "of period 1 has to come from somewhere."
        )
    return points


def _draw_path(
    points: np.ndarray, p: float, rng: np.random.Generator, size: int = 1
) -> np.ndarray:
    """Assignment paths, one per row: a coin at each randomization point."""
    block = np.cumsum(points) - 1
    coins = rng.random((size, int(points.sum()))) < p
    return coins[:, block].astype(int)


def switchback_design(
    n_periods: int,
    m: int = 0,
    design: Union[str, int, Sequence[Any], np.ndarray] = "optimal",
    p: float = 0.5,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Randomization points and an assignment path for a switchback experiment.

    Parameters
    ----------
    n_periods : int
        Number of periods ``T`` in the experiment.
    m : int, default 0
        Order of the carryover effect: a treatment may affect the outcome of
        its own period and of the next ``m``.
    design : {'optimal', 'every'}, int, or boolean vector, default 'optimal'
        Where the coin is flipped. ``'optimal'`` is the minimax design of
        Bojinov, Simchi-Levi and Zhao (2023, Theorem 2): every period when
        ``m = 0``; periods ``1, 2m+1, 3m+1, ..., (n-2)m+1`` when ``m > 0``
        and ``n_periods = n m`` with ``n >= 4``. ``'every'`` flips in every
        period, an integer ``b`` flips every ``b`` periods, and a boolean
        vector marks the periods directly.
    p : float, default 0.5
        Probability of treatment at each flip. The optimal design uses a
        fair coin (their Theorem 1).
    seed : int, optional
        Seed for the assignment draw.

    Returns
    -------
    pd.DataFrame
        One row per period: ``period`` (1-based), ``randomize`` (True where
        the coin is flipped) and ``treat`` (the drawn assignment). Run the
        experiment with ``treat``, add the outcome column, and pass the
        frame to :func:`switchback` with ``design='randomize'``.

    Examples
    --------
    >>> import statspai as sp
    >>> plan = sp.switchback_design(12, m=2, seed=0)
    >>> plan.loc[plan["randomize"], "period"].tolist()
    [1, 5, 7, 9]

    References
    ----------
    bojinov2023design
    """
    n_periods, m = int(n_periods), int(m)
    if n_periods < 1 or m < 0:
        raise MethodIncompatibility("n_periods must be positive and m non-negative.")
    if not 0.0 < p < 1.0:
        raise MethodIncompatibility(f"p must lie strictly between 0 and 1; got {p}.")
    points = _resolve_points(design, n_periods, m)
    path = _draw_path(points, p, np.random.default_rng(seed))[0]
    return pd.DataFrame(
        {
            "period": np.arange(1, n_periods + 1),
            "randomize": points,
            "treat": path,
        }
    )


# ---------------------------------------------------------------------- #
#  Analysis
# ---------------------------------------------------------------------- #


def _ht_estimates(
    paths: np.ndarray, y: np.ndarray, points: np.ndarray, m: int, p: float
) -> np.ndarray:
    """Horvitz-Thompson estimate of the lag-``m`` effect for each path.

    ``paths`` is (n_paths, T). For period ``t`` (0-based, ``t >= m``) the
    window ``t-m..t`` is all-treated with probability ``p ** c_t``, where
    ``c_t`` is the number of coins that decide the window: the randomization
    points inside ``t-m+1..t`` plus the one that set period ``t-m``.
    """
    n_periods = y.shape[0]
    cum = np.cumsum(points)
    n_coins = (cum[m:] - cum[: n_periods - m]) + 1
    csum = np.concatenate(
        [np.zeros((paths.shape[0], 1), dtype=int), np.cumsum(paths, axis=1)], axis=1
    )
    window_sum = csum[:, m + 1 :] - csum[:, : n_periods - m]
    all_one = window_sum == m + 1
    all_zero = window_sum == 0
    y_use = y[m:]
    contrib = all_one * (y_use / p**n_coins) - all_zero * (y_use / (1.0 - p) ** n_coins)
    return np.asarray(contrib.sum(axis=1) / (n_periods - m), dtype=float)


def _conservative_variance(
    y: np.ndarray, d: np.ndarray, n_periods: int, m: int
) -> float:
    """Corollary 1: the estimator of the variance upper bound.

    Blocks ``k = 0..n-2`` are periods ``(k+1)m+1 .. (k+2)m`` (1-based); the
    first block of ``m`` periods has no complete window and is left out.
    """
    n = n_periods // m
    block_sums = y.reshape(n, m).sum(axis=1)[1:]
    # W at periods km+1, k = 1..n-1 (the first period of each kept block)
    w_first = d.reshape(n, m)[1:, 0]
    same = w_first[1:-1] == w_first[:-2]  # W_{(k+1)m+1} == W_{km+1}, k = 1..n-3
    total = (
        8.0 * block_sums[0] ** 2
        + 32.0 * float(np.sum(block_sums[1:-1] ** 2 * same))
        + 8.0 * block_sums[-1] ** 2
    )
    return float(total / (n_periods - m) ** 2)


def _worst_case_variance(points: np.ndarray, m: int, p: float, bound: float) -> float:
    """Largest variance of the estimator over outcomes bounded by ``bound``.

    With every potential outcome equal to ``bound`` (the dominating case of
    Lemma 1 of the paper) the estimator is ``bound / (T - m)`` times the sum
    over periods of ``Z_t = 1{window all treated} / p^c_t - 1{window all
    control} / (1 - p)^c_t``, with ``c_t`` the number of coins behind window
    ``t``. ``E[Z_t] = 0``, two windows with no coin in common are
    independent, and two windows that share ``s`` coins have
    ``E[Z_t Z_u] = p^-s + (1 - p)^-s``. The variance is the sum of those
    terms. It holds for any regular design and any ``p``, and agrees with
    full enumeration of the assignment paths.
    """
    n_periods = points.shape[0]
    block = np.cumsum(points) - 1
    lo = block[: n_periods - m]  # coin that set period t - m
    hi = block[m:]  # coin that set period t
    total = 0.0
    q = 1.0 - p
    for start in range(0, lo.shape[0], 2000):  # bounded memory for long horizons
        sl = slice(start, start + 2000)
        shared = (
            np.minimum(hi[sl, None], hi[None, :])
            - np.maximum(lo[sl, None], lo[None, :])
            + 1
        )
        s_pos = shared[shared > 0].astype(float)
        total += float(np.sum(p**-s_pos + q**-s_pos))
    return float(bound**2 * total / (n_periods - m) ** 2)


def switchback(
    data: pd.DataFrame,
    y: str,
    treat: str,
    m: int = 0,
    design: Union[str, int, Sequence[Any], np.ndarray] = "optimal",
    p: float = 0.5,
    time: Optional[str] = None,
    n_draws: int = 10_000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
    outcome_bound: Optional[float] = None,
) -> CausalResult:
    r"""Design-based analysis of a switchback experiment.

    Estimates the average lag-``m`` effect: the mean over periods of the
    outcome after ``m + 1`` consecutive treated periods minus the outcome
    after ``m + 1`` consecutive control periods,

    .. math::

        \tau_m = \frac{1}{T-m} \sum_{t=m+1}^{T}
                 \left[ Y_t(\mathbf{1}_{m+1}) - Y_t(\mathbf{0}_{m+1}) \right],

    with the Horvitz-Thompson estimator of Bojinov, Simchi-Levi and Zhao
    (2023, equation 4). A period contributes when its whole window was
    treated (or control), weighted by the inverse of the probability of that
    event under the design. No outcome model is involved, so trends,
    seasonality and serial correlation in the outcome do not bias it.

    Parameters
    ----------
    data : pd.DataFrame
        One row per period, in time order (or sortable by ``time``).
    y : str
        Outcome column.
    treat : str
        Assignment column (0/1), constant between randomization points.
    m : int, default 0
        Order of the carryover effect assumed in the analysis.
    design : {'optimal', 'every'}, int, boolean vector, or str, default 'optimal'
        The design the assignment was drawn from: a design name or block
        length as in :func:`switchback_design`, a boolean vector, or the
        name of a boolean column of ``data`` marking the randomization
        points. This must be the design that was actually run; the
        estimator's weights are its probabilities.
    p : float, default 0.5
        Probability of treatment at each coin flip.
    time : str, optional
        Column to sort by. Default: the row order of ``data``.
    n_draws : int, default 10000
        Assignment paths drawn for the randomization test. ``0`` skips it.
    alpha : float, default 0.05
        Level of the confidence interval.
    seed : int, optional
        Seed for the randomization test.
    outcome_bound : float, optional
        A number ``B`` that no outcome can exceed in absolute value under
        any assignment (it must be at least the largest observed
        ``|y|``). With it, a design other than the optimal one gets a
        standard error and an interval; see Returns. The estimand does not
        change when a constant is subtracted from the outcome, so centring
        the outcome at a value chosen before the experiment makes ``B``,
        and the interval, smaller.

    Returns
    -------
    CausalResult
        * ``estimate`` -- the Horvitz-Thompson estimate of :math:`\tau_m`.
        * ``se``, ``ci``, ``pvalue`` -- under the optimal design with
          ``m >= 1``, the conservative standard error of their Corollary 1,
          the normal interval and the two-sided normal p-value built on it
          (their Theorem 3). The interval is conservative: the estimator is
          of an upper bound of the variance. For any other design the paper
          gives no variance estimator. Without ``outcome_bound``, ``se``
          and ``ci`` are NaN and ``pvalue`` is the randomization p-value.
          With ``outcome_bound``, ``se`` is the square root of the largest
          variance the estimator can have under that design when outcomes
          are bounded by ``B`` (the paper's Lemma 1 shows which outcomes
          attain it), and ``ci`` is the Chebyshev interval
          ``estimate +/- se / sqrt(alpha)``. It is valid in finite samples
          and makes no distributional assumption, which is also why it is
          wide; ``pvalue`` stays the randomization p-value.
        * ``model_info['worst_case_se']`` -- that worst-case standard error,
          whenever ``outcome_bound`` is given.
        * ``model_info['randomization_pvalue']`` -- exact test of the sharp
          null of no effect of any assignment on any outcome (their
          Algorithm 1): the share of re-drawn assignment paths whose
          estimate, on the observed outcomes, is larger in absolute value
          than the one observed.
        * ``model_info['n_windows_treated']`` / ``['n_windows_control']`` --
          periods whose window was all treated / all control.

    Notes
    -----
    With ``m`` smaller than the true carryover the estimator is biased;
    with ``m`` larger it stays unbiased and loses precision, so err on the
    side of a larger ``m``. A regression of the outcome on the treatment
    and its lags answers the same question under a linear model of the
    carryover and of the outcome's dynamics; this estimator needs neither,
    at the price of a larger variance.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> plan = sp.switchback_design(120, m=2, seed=1)
    >>> rng = np.random.default_rng(1)
    >>> d = plan["treat"].to_numpy()
    >>> lagged = d + np.r_[0, d[:-1]] + np.r_[0, 0, d[:-2]]
    >>> plan["y"] = 10 + 1.0 * lagged + rng.normal(size=120)
    >>> res = sp.switchback(plan, y="y", treat="treat", m=2,
    ...                     design="randomize", seed=1)
    >>> bool(np.isfinite(res.estimate) and res.se > 0)
    True

    References
    ----------
    bojinov2023design
    """
    m = int(m)
    if m < 0:
        raise MethodIncompatibility("m must be non-negative.")
    if not 0.0 < p < 1.0:
        raise MethodIncompatibility(f"p must lie strictly between 0 and 1; got {p}.")
    missing = [c for c in (y, treat, time) if c is not None and c not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"switchback: columns not found in data: {missing}",
            diagnostics={"missing": missing},
        )
    frame = data if time is None else data.sort_values(time, kind="stable")
    if frame[[y, treat]].isna().any().any():
        raise MethodIncompatibility(
            "switchback: the outcome and the assignment must be observed in "
            "every period; a missing period changes the windows around it.",
            recovery_hint="Analyse a stretch of consecutive complete periods.",
        )
    y_arr = frame[y].to_numpy(dtype=float)
    d_raw = frame[treat].to_numpy()
    from ..core._validate import require_binary_treatment

    require_binary_treatment(d_raw, function="switchback")
    d_arr = d_raw.astype(int)
    n_periods = y_arr.shape[0]
    if n_periods <= m + 1:
        raise DataInsufficient(
            f"switchback: {n_periods} periods are too few for m={m}.",
            diagnostics={"n_periods": n_periods, "m": m},
        )

    if isinstance(design, str) and design not in ("optimal", "every"):
        if design not in frame.columns:
            raise MethodIncompatibility(
                f"switchback: design={design!r} is neither a design name "
                "('optimal', 'every') nor a column of data."
            )
        design_label = f"column {design!r}"
        points = _resolve_points(frame[design].to_numpy(), n_periods, m)
    else:
        design_label = design if isinstance(design, str) else "custom"
        if isinstance(design, (int, np.integer)) and not isinstance(design, bool):
            design_label = f"every {int(design)} periods"
        points = _resolve_points(design, n_periods, m)

    # The assignment may change only where the coin was flipped.
    changes = np.flatnonzero(np.diff(d_arr) != 0) + 1
    off_design = changes[~points[changes]]
    if off_design.size:
        raise MethodIncompatibility(
            f"switchback: the assignment changes at period(s) "
            f"{(off_design[:5] + 1).tolist()}, which are not randomization "
            "points of the stated design.",
            recovery_hint=(
                "Pass the design that generated the assignment: design= "
                "takes a boolean column marking the periods where the coin "
                "was flipped."
            ),
            diagnostics={"n_off_design_changes": int(off_design.size)},
        )

    estimate = float(_ht_estimates(d_arr[None, :], y_arr, points, m, p)[0])
    csum = np.concatenate([[0], np.cumsum(d_arr)])
    window_sum = csum[m + 1 :] - csum[: n_periods - m]
    n_treated_windows = int(np.sum(window_sum == m + 1))
    n_control_windows = int(np.sum(window_sum == 0))

    randomization_p = float("nan")
    if n_draws and n_draws > 0:
        rng = np.random.default_rng(seed)
        draws = _draw_path(points, p, rng, size=int(n_draws))
        null = _ht_estimates(draws, y_arr, points, m, p)
        randomization_p = float(np.mean(np.abs(null) > abs(estimate)))

    is_optimal = bool(
        m >= 1
        and p == 0.5
        and n_periods % m == 0
        and n_periods // m >= 4
        and np.array_equal(points, _optimal_points(n_periods, m))
    )
    worst_case_se: Optional[float] = None
    if outcome_bound is not None:
        bound = float(outcome_bound)
        largest = float(np.max(np.abs(y_arr)))
        if not np.isfinite(bound) or bound < largest:
            raise MethodIncompatibility(
                f"switchback: outcome_bound={outcome_bound} is below the "
                f"largest observed |{y}| ({largest:.6g}); it must bound every "
                "outcome under every assignment.",
                recovery_hint="Pass a bound that the outcome cannot exceed.",
            )
        worst_case_se = float(np.sqrt(_worst_case_variance(points, m, p, bound)))
    if is_optimal:
        se = float(np.sqrt(_conservative_variance(y_arr, d_arr, n_periods, m)))
        z = stats.norm.ppf(1 - alpha / 2)
        ci = (estimate - z * se, estimate + z * se)
        pvalue = float(2 * stats.norm.sf(abs(estimate) / se)) if se > 0 else np.nan
        inference = "conservative normal (Corollary 1, Theorem 3)"
    elif worst_case_se is not None:
        se = worst_case_se
        half = se / np.sqrt(alpha)
        ci = (estimate - half, estimate + half)
        pvalue = randomization_p
        inference = (
            "worst-case variance under bounded outcomes, Chebyshev interval; "
            "p-value from the randomization test"
        )
    else:
        se = float("nan")
        ci = (float("nan"), float("nan"))
        pvalue = randomization_p
        inference = "randomization test of the sharp null (Algorithm 1)"

    model_info: Dict[str, Any] = {
        "m": m,
        "p": p,
        "design": design_label,
        "optimal_design": is_optimal,
        "n_periods": n_periods,
        "n_randomization_points": int(points.sum()),
        "randomization_points": (np.flatnonzero(points) + 1).tolist(),
        "n_windows_treated": n_treated_windows,
        "n_windows_control": n_control_windows,
        "randomization_pvalue": randomization_p,
        "worst_case_se": worst_case_se,
        "outcome_bound": outcome_bound,
        "n_draws": int(n_draws or 0),
        "inference": inference,
    }
    return CausalResult(
        method="Switchback experiment (Horvitz-Thompson, lag-m effect)",
        estimand=f"Average lag-{m} effect",
        estimate=estimate,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=n_periods,
        detail=None,
        model_info=model_info,
        _citation_key="switchback",
    )


# Kept in sync with paper.bib (key bojinov2023design).
CausalResult._CITATIONS["switchback"] = (
    "@article{bojinov2023design,\n"
    "  title={Design and Analysis of Switchback Experiments},\n"
    "  author={Bojinov, Iavor and Simchi-Levi, David and Zhao, Jinglong},\n"
    "  journal={Management Science},\n"
    "  volume={69},\n"
    "  number={7},\n"
    "  pages={3759--3777},\n"
    "  year={2023},\n"
    "  doi={10.1287/mnsc.2022.4583}\n"
    "}"
)
