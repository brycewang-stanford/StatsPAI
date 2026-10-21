"""Bandwidth selection for geographically weighted regression.

``gwr_bandwidth`` minimises an information criterion (AICc, AIC, BIC) or the
leave-one-out cross-validation score of :func:`statspai.spatial.gwr.gwr`
over the bandwidth by golden-section search (Kiefer 1953,
``kiefer1953sequential``), the selection method of Brunsdon, Fotheringham
and Charlton (1996, ``brunsdon1996geographically``) as implemented in R
``GWmodel``'s ``bw.gwr`` (Gollini et al. 2015, ``gollini2015gwmodel``).

Clean-room implementation (2026-10-20), written from the papers, the
``GWmodel`` help pages and black-box runs of ``GWmodel`` 2.4-1; no
``GWmodel`` source was read (spec: ``docs/dev/cleanroom/gwr_bandwidth.md``).
The criterion is not unimodal on the integer lattice of adaptive bandwidths,
so the selected bandwidth depends on the search path; the path below
reproduces the probes that ``GWmodel``'s search evaluates, which were
recorded from the reference by passing it an instrumented objective
(``tests/reference_parity/test_gwr_golden_search_blackbox.py``).
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, List, Literal, Optional, Tuple, get_args

import numpy as np

from ...exceptions import MethodIncompatibility, NumericalInstability
from .gwr import GWRResult, KernelName, gwr

Criterion = Literal["AICc", "AIC", "BIC", "CV"]

_GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0  # 0.618..., the golden-section ratio
_CRITERIA = {"AICc": "aicc", "AIC": "aic", "BIC": "bic", "CV": "cv"}
_ADAPTIVE_MIN = 20.0
_FIXED_MIN_FRACTION = 1.0 / 5000.0


def _criterion(result: GWRResult, criterion: Criterion) -> float:
    """Value of ``criterion`` for a fitted GWR (smaller is better)."""
    return float(getattr(result, _CRITERIA[criterion]))


def _golden_section_search(
    objective: Callable[[float], float],
    lower: float,
    upper: float,
    integer: bool,
    tol: float = 1e-4,
) -> Tuple[float, List[float]]:
    """Golden-section minimisation of ``objective`` on ``[lower, upper]``.

    The two interior points sit a golden fraction ``d = 0.618 (upper -
    lower)`` from either end. Each step keeps the sub-interval around the
    better point, shrinks ``d`` by the golden ratio and evaluates one new
    point at distance ``d`` from the end that moved. The search stops when
    the two interior values differ by at most ``tol`` or when ``d`` itself
    has fallen to ``tol``; it returns the better interior point (the lower
    one on a tie) and the list of evaluated points.

    With ``integer=True`` (adaptive bandwidths, counted in neighbours) the
    evaluated points are integers: the first upper point is rounded down
    and the first lower point to the nearest integer; afterwards a new upper
    point is rounded to the nearest integer and a new lower point down, and
    the integer values are what the search keeps. These rounding rules are
    the ones that reproduce, probe for probe, the points evaluated by the
    reference implementation on 40 recorded searches.
    """
    probes: List[float] = []

    def evaluate(x: float) -> float:
        probes.append(x)
        return float(objective(x))

    step = _GOLDEN * (upper - lower)
    hi_pt = lower + step
    lo_pt = upper - step
    if integer:
        hi_pt = float(np.floor(hi_pt))
        lo_pt = float(np.rint(lo_pt))
    f_hi = evaluate(hi_pt)
    f_lo = evaluate(lo_pt)
    while abs(f_hi - f_lo) > tol and step > tol:
        step *= _GOLDEN
        if f_hi < f_lo:
            # Minimum lies above lo_pt: drop [lower, lo_pt].
            lower = lo_pt
            lo_pt, f_lo = hi_pt, f_hi
            hi_pt = lower + step
            if integer:
                hi_pt = float(np.rint(hi_pt))
            f_hi = evaluate(hi_pt)
        else:
            # Minimum lies below hi_pt: drop [hi_pt, upper].
            upper = hi_pt
            hi_pt, f_hi = lo_pt, f_lo
            lo_pt = upper - step
            if integer:
                lo_pt = float(np.floor(lo_pt))
            f_lo = evaluate(lo_pt)
    best = hi_pt if f_hi < f_lo else lo_pt
    return best, probes


def _max_pairwise_distance(coords: np.ndarray) -> float:
    """Largest Euclidean distance between two points."""
    c = np.asarray(coords, dtype=float)
    best = 0.0
    # Row blocks keep memory at O(block * n) for large n.
    block = max(1, int(2_000_000 // max(len(c), 1)))
    for start in range(0, len(c), block):
        diff = c[start : start + block, None, :] - c[None, :, :]
        best = max(best, float(np.sqrt(np.max(np.einsum("ijk,ijk->ij", diff, diff)))))
    return best


def gwr_bandwidth(
    coords: Any,
    y: Any,
    X: Any,
    kernel: KernelName = "bisquare",
    fixed: bool = False,
    criterion: Criterion = "AICc",
    bw_min: Optional[float] = None,
    bw_max: Optional[float] = None,
    add_constant: bool = True,
    tol: float = 1e-4,
) -> float:
    """Select a GWR bandwidth by golden-section search.

    Minimises ``criterion`` of :func:`statspai.spatial.gwr.gwr` over the
    bandwidth, as R ``GWmodel::bw.gwr`` does.

    Parameters
    ----------
    coords : (n, 2) array-like
        Projected point coordinates.
    y : (n,) array-like
        Outcome.
    X : (n, p) array-like
        Regressors (without a constant unless ``add_constant=False``).
    kernel : {'bisquare', 'gaussian', 'exponential'}
        Kernel passed to :func:`gwr`.
    fixed : bool, default False
        ``False``: adaptive bandwidth, a number of nearest neighbours
        (searched over integers). ``True``: a fixed distance.
    criterion : {'AICc', 'AIC', 'BIC', 'CV'}
        Score to minimise; ``'CV'`` is the leave-one-out residual sum of
        squares. ``GWmodel`` offers ``'AICc'`` and ``'CV'``.
    bw_min, bw_max : float, optional
        Search interval. Defaults: ``[20, n]`` neighbours (adaptive) or
        ``[bw_max / 5000, bw_max]`` with ``bw_max`` defaulting to the
        largest distance between two points (fixed), the reference's
        interval. With ``n <= 20`` observations the adaptive interval
        ``[20, n]`` would be empty and starts at ``k + 2`` instead (``k``
        local coefficients, the constant included). Adaptive searches can
        end one or two neighbours outside the interval, as the reference's
        do, because their points are rounded.
    add_constant : bool, default True
        Passed to :func:`gwr`.
    tol : float, default 1e-4
        The search stops when two interior criterion values differ by at
        most ``tol`` or the golden step falls to ``tol``.

    Returns
    -------
    float
        The selected bandwidth.

    Notes
    -----
    A probed bandwidth that admits no GWR fit (one neighbour or less) scores
    ``+inf``, with a ``RuntimeWarning``; if no probe can be fitted the first
    error is raised.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> coords = rng.uniform(0, 10, size=(60, 2))
    >>> X = rng.normal(size=(60, 1))
    >>> y = 1 + (0.5 + 0.1 * coords[:, 0]) * X[:, 0] + rng.normal(0, 0.3, 60)
    >>> bw = sp.gwr_bandwidth(coords, y, X)
    >>> bool(bw > 0)
    True

    References
    ----------
    brunsdon1996geographically; gollini2015gwmodel; kiefer1953sequential.
    """
    if criterion not in _CRITERIA:
        raise MethodIncompatibility(
            f"criterion must be one of {sorted(_CRITERIA)}, got {criterion!r}",
            recovery_hint="Use criterion='AICc', 'AIC', 'BIC' or 'CV'.",
            diagnostics={"criterion": criterion},
        )
    if kernel not in get_args(KernelName):
        raise MethodIncompatibility(
            f"kernel must be one of {list(get_args(KernelName))}, got {kernel!r}",
            recovery_hint="Use kernel='bisquare', 'gaussian' or 'exponential'.",
            diagnostics={"kernel": kernel},
        )
    coords_arr = np.asarray(coords, dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()
    n = y_arr.size
    if fixed:
        upper = _max_pairwise_distance(coords_arr) if bw_max is None else float(bw_max)
        lower = upper * _FIXED_MIN_FRACTION if bw_min is None else float(bw_min)
    else:
        X_arr = np.asarray(X, dtype=float)
        n_coef = (X_arr.shape[1] if X_arr.ndim == 2 else 1) + int(add_constant)
        # [20, n] is empty for n <= 20; start just above the local model size.
        default_lower = _ADAPTIVE_MIN if n > _ADAPTIVE_MIN else float(n_coef + 2)
        lower = default_lower if bw_min is None else float(bw_min)
        upper = float(n) if bw_max is None else float(bw_max)
    if not upper > lower:
        raise MethodIncompatibility(
            f"empty bandwidth search interval [{lower}, {upper}]",
            recovery_hint="Pass bw_min < bw_max (for adaptive bandwidths, "
            "bw_max defaults to the number of observations).",
            diagnostics={"bw_min": lower, "bw_max": upper, "n": n},
        )

    failures: List[BaseException] = []

    def objective(bw: float) -> float:
        # A probe can fall where no GWR exists (a bandwidth of one neighbour
        # or less, or below zero once an integer search has drifted past
        # its lower end); it scores +inf so the search moves away from it.
        try:
            fit = gwr(
                coords_arr,
                y_arr,
                X,
                bw=bw,
                kernel=kernel,
                fixed=fixed,
                add_constant=add_constant,
            )
        except (
            NumericalInstability,
            MethodIncompatibility,
            np.linalg.LinAlgError,
        ) as exc:
            failures.append(exc)
            return float("inf")
        return _criterion(fit, criterion)

    best, probes = _golden_section_search(
        objective, lower, upper, integer=not fixed, tol=tol
    )
    if failures:
        if len(failures) == len(probes):
            raise failures[0]
        warnings.warn(
            f"gwr_bandwidth: {len(failures)} of {len(probes)} probed bandwidths "
            f"admit no GWR fit (first: {failures[0]}); they were scored +inf.",
            RuntimeWarning,
            stacklevel=2,
        )
    return float(best)
