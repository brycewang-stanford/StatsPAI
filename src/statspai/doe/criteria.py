"""Space-filling measures of a design: ``sp.design_criteria``."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from ..exceptions import DataInsufficient, MethodIncompatibility
from ._common import pair_sqdist, to_unit, upper


def maximin_distance(X: np.ndarray) -> float:
    """Smallest distance between two runs."""
    return float(np.sqrt(upper(pair_sqdist(X)).min()))


def reciprocal_distance(X: np.ndarray, r: Optional[float] = None) -> float:
    """``(mean over pairs of d^-r)^(1/r)``, the smooth stand-in for maximin."""
    r = float(2 * X.shape[1] if r is None else r)
    d2 = upper(pair_sqdist(X))
    if d2.min() <= 0:
        return float("inf")
    # work on logs: d^-r overflows for close pairs and large r
    lg = -0.5 * r * np.log(d2)
    m = lg.max()
    return float(np.exp((m + np.log(np.exp(lg - m).mean())) / r))


def _log_maxpro_terms(X: np.ndarray, delta: float) -> np.ndarray:
    n = X.shape[0]
    iu = np.triu_indices(n, k=1)
    lg = np.zeros(iu[0].size)
    for col in X.T:
        diff = col[iu[0]] - col[iu[1]]
        with np.errstate(divide="ignore"):
            lg -= np.log(diff * diff + delta)
    return lg


def maxpro_criterion(X: np.ndarray, delta: float = 0.0) -> float:
    """``(mean over pairs of 1 / prod_l (x_il - x_jl)^2)^(1/p)``."""
    lg = _log_maxpro_terms(X, delta)
    m = lg.max()
    if not np.isfinite(m):
        return float("inf")
    return float(np.exp((m + np.log(np.exp(lg - m).mean())) / X.shape[1]))


def wraparound_discrepancy(X: np.ndarray) -> float:
    """Wrap-around L2 discrepancy (Hickernell 1998)."""
    n, p = X.shape
    prod = np.ones((n, n))
    for col in X.T:
        a = np.abs(col[:, None] - col[None, :])
        prod *= 1.5 - a * (1.0 - a)
    return float(np.sqrt(max(-((4.0 / 3.0) ** p) + prod.sum() / n**2, 0.0)))


def centered_discrepancy(X: np.ndarray) -> float:
    """Centered L2 discrepancy (Hickernell 1998)."""
    n, p = X.shape
    a = np.abs(X - 0.5)
    one = np.prod(1.0 + 0.5 * a - 0.5 * a * a, axis=1).sum()
    prod = np.ones((n, n))
    for j in range(p):
        prod *= (
            1.0
            + 0.5 * a[:, j][:, None]
            + 0.5 * a[:, j][None, :]
            - 0.5 * np.abs(X[:, j][:, None] - X[:, j][None, :])
        )
    return float(
        np.sqrt(max((13.0 / 12.0) ** p - 2.0 * one / n + prod.sum() / n**2, 0.0))
    )


def fill_distance(X: np.ndarray, n_eval: int = 0, seed: int = 0) -> float:
    """Largest distance from a point of the cube to its nearest run.

    Evaluated on a scrambled Sobol' set plus the corners, so it is a lower
    bound that tightens with ``n_eval``.
    """
    from scipy.stats import qmc

    p = X.shape[1]
    m = int(n_eval) if n_eval else int(min(2**17, max(2**12, 2 ** (9 + p))))
    m = 1 << int(np.ceil(np.log2(m)))
    pts = qmc.Sobol(p, scramble=True, seed=seed).random(m)
    if p <= 12:
        corners = np.array(np.meshgrid(*[[0.0, 1.0]] * p)).reshape(p, -1).T
        pts = np.vstack([pts, corners])
    best = 0.0
    for start in range(0, pts.shape[0], 4096):
        d2 = pair_sqdist(pts[start : start + 4096], X).min(axis=1)
        best = max(best, float(d2.max()))
    return float(np.sqrt(best))


def energy_to_sample(sample: np.ndarray, points: np.ndarray) -> float:
    """Energy distance between ``points`` and ``sample`` up to a constant.

    ``2 mean|y - x| - mean|x - x'|`` with ``y`` over the sample and ``x``
    over the points; the term that involves the sample alone is dropped,
    as it does not depend on the points. Both are scaled by the mean and
    standard deviation of the sample.
    """
    mu = sample.mean(axis=0)
    sd = sample.std(axis=0, ddof=1)
    sd = np.where(sd > 0, sd, 1.0)
    S = (sample - mu) / sd
    P = (points - mu) / sd
    cross = 0.0
    for start in range(0, S.shape[0], 8192):
        cross += float(np.sqrt(pair_sqdist(S[start : start + 8192], P)).sum())
    cross /= S.shape[0] * P.shape[0]
    within = float(np.sqrt(pair_sqdist(P)).sum()) / P.shape[0] ** 2
    return float(2.0 * cross - within)


def all_criteria(
    X: np.ndarray, delta: float = 0.0, r: Optional[float] = None, fill: bool = True
) -> Dict[str, float]:
    """The measures reported with every design."""
    n, p = X.shape
    out: Dict[str, float] = {
        "maximin": maximin_distance(X),
        "reciprocal_distance": reciprocal_distance(X, r),
        "maxpro": maxpro_criterion(X, delta),
        "wraparound_discrepancy": wraparound_discrepancy(X),
        "centered_discrepancy": centered_discrepancy(X),
    }
    proj = min(float(np.diff(np.sort(X[:, j])).min()) for j in range(p))
    out["min_projected_distance"] = proj
    if fill:
        out["fill_distance"] = fill_distance(X)
    if p > 1 and n > 2:
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.corrcoef(X, rowvar=False)
        off = np.abs(c[np.triu_indices(p, k=1)])
        out["max_abs_correlation"] = float(np.nanmax(off)) if off.size else 0.0
    return out


def design_criteria(
    design: Any,
    bounds: Optional[Mapping[str, Tuple[float, float]]] = None,
    delta: float = 0.0,
    r: Optional[float] = None,
    target: Any = None,
) -> pd.Series:
    """How well a design fills the experimental region.

    Parameters
    ----------
    design : DataFrame, array or DesignResult
        One row per run. Arrays and frames are taken to be on the unit
        cube unless ``bounds`` is given.
    bounds : dict, optional
        ``{factor: (lower, upper)}``; the design is scaled to the unit
        cube with them before anything is computed.
    delta : float, default 0
        Added to each squared difference in the MaxPro criterion. With
        the default a design that repeats a level of a factor has an
        infinite criterion; a small positive value makes designs with
        repeated levels comparable.
    r : float, optional
        Power of the reciprocal distance. Default: twice the number of
        factors.
    target : DataFrame or array, optional
        A sample from the distribution the runs should represent. When
        given, ``energy_distance`` to that sample is added. This one is
        computed in the units of the sample, not on the unit cube.

    Returns
    -------
    Series
        ``maximin`` (smallest distance between two runs, larger is
        better), ``reciprocal_distance`` (its smooth stand-in, smaller is
        better), ``maxpro`` (smaller is better; penalises runs that are
        close in any projection), ``wraparound_discrepancy`` and
        ``centered_discrepancy`` (L2 discrepancies, smaller is
        more uniform), ``min_projected_distance`` (smallest gap between
        two runs along a single factor), ``fill_distance`` (largest
        distance from a point of the region to its nearest run, smaller
        is better), ``max_abs_correlation``.

    Notes
    -----
    ``maximin``, ``reciprocal_distance``, ``maxpro`` and
    ``wraparound_discrepancy`` reproduce ``maximin.crit``,
    ``maxpro.crit`` and ``uniform.crit`` of the R package ``SFDesign``;
    ``energy_distance`` reproduces ``twinning::energy``.

    ``fill_distance`` is evaluated on a finite set of points and is
    therefore a lower bound of the true value.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> grid = np.array([[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]])
    >>> crit = sp.design_criteria(grid)
    >>> float(crit["maximin"])
    0.5
    >>> bool(np.isinf(crit["maxpro"]))  # two runs share a level
    True

    References
    ----------
    johnson1990minimax; morris1995exploratory; joseph2015maximum;
    hickernell1998generalized
    """
    X, _, _, _ = to_unit(design, bounds)
    if X.shape[0] < 2:
        raise DataInsufficient("A design needs at least two runs to be measured.")
    if delta < 0:
        raise MethodIncompatibility(f"delta must be non-negative; got {delta}.")
    out = all_criteria(X, delta=delta, r=r)
    if target is not None:
        frame = design.design if hasattr(design, "design") else design
        if isinstance(target, pd.DataFrame) and isinstance(frame, pd.DataFrame):
            # two tables: match the columns by name, not by position
            if set(map(str, target.columns)) != set(map(str, frame.columns)):
                raise MethodIncompatibility(
                    "target and the design do not have the same columns."
                )
            target = target[list(frame.columns)]
        S = np.asarray(
            (
                target.to_numpy(dtype=float)
                if isinstance(target, pd.DataFrame)
                else target
            ),
            dtype=float,
        )
        raw = np.asarray(
            frame.to_numpy(dtype=float) if isinstance(frame, pd.DataFrame) else frame,
            dtype=float,
        )
        if S.ndim == 1:
            S = S[:, None]
        if raw.ndim == 1:
            raw = raw[:, None]
        if S.shape[1] != raw.shape[1]:
            raise MethodIncompatibility(
                f"target has {S.shape[1]} columns, the design {raw.shape[1]}."
            )
        out["energy_distance"] = energy_to_sample(S, raw)
    return pd.Series(out, name="criterion")
