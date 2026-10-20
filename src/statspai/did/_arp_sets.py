"""Grid inversion of the Andrews-Roth-Pakes tests (Honest DiD, native).

:func:`union_confidence_set` inverts :func:`statspai.did._arp.accept_polyhedron`
over a grid of ``theta = l' tau_post`` for a union of polyhedral pieces,
extending the grid at an accepted end and re-centring it when the estimate
lies outside; :func:`rm_confidence_set` is that inversion for HonestDiD's
relative-magnitudes restriction ``Delta^RM(Mbar)``. The grid layout follows
R ``HonestDiD`` 0.2.8 (MIT licence; notice in THIRD_PARTY_NOTICES.md); see
:mod:`statspai.did._arp` for the tests themselves.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np

from ._arp import accept_polyhedron, create_a_rm

__all__ = ["union_confidence_set", "rm_confidence_set"]


def union_confidence_set(
    pieces: List[Tuple[np.ndarray, np.ndarray]],
    betahat: np.ndarray,
    sigma: np.ndarray,
    n_pre: int,
    n_post: int,
    l_vec: np.ndarray,
    alpha: float,
    hybrid: str,
    lb: float,
    ub: float,
    grid_points: int,
    *,
    kappa: Optional[float] = None,
    seed: int = 0,
    flci: Optional[Tuple[np.ndarray, float]] = None,
    nonuis_post_rows_only: bool = False,
    expand_lo: bool = False,
    expand_hi: bool = False,
    max_expand: int = 0,
    recentre_sd: Optional[float] = None,
    progress: Optional[Callable[[int], None]] = None,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Confidence set for ``l' tau_post`` under the union of ``pieces``.

    ``pieces`` are ``(A, d)`` polyhedra; a grid point is accepted when any
    piece accepts it (HonestDiD takes the row-wise maximum of the pieces'
    acceptance vectors). The grid is ``grid_points`` values on ``[lb, ub]``.
    A side flagged in ``expand_lo`` / ``expand_hi`` whose end point is
    accepted is extended by another ``grid_points`` points at the same step,
    up to ``max_expand`` times, so the reported bound is the set's end point
    rather than the grid's; only the new points are tested. With
    ``recentre_sd``, a grid that accepts nothing because the estimate lies
    outside it is moved to ``+/- recentre_sd`` around the estimate.

    Returns ``(lower, upper, grid, accept)``: the smallest and largest
    accepted grid values (``nan`` if none) and the acceptance vector.
    """
    betahat = np.asarray(betahat, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float)
    l_vec = np.asarray(l_vec, dtype=float).ravel()
    if kappa is None:
        kappa = alpha / 10.0
    grid = np.linspace(lb, ub, int(grid_points))

    def union_accept(pts: np.ndarray) -> np.ndarray:
        accept = np.zeros(pts.size)
        for a, d in pieces:
            acc = accept_polyhedron(
                betahat,
                sigma,
                n_pre,
                n_post,
                l_vec,
                a,
                d,
                alpha,
                hybrid,
                kappa,
                pts,
                seed=seed,
                skip=accept == 1,
                flci=flci,
                nonuis_post_rows_only=nonuis_post_rows_only,
            )
            accept = np.maximum(accept, acc)
            if progress is not None:
                progress(1)
        return accept

    accept = union_accept(grid)
    if grid.size > 1:
        step = float(grid[1] - grid[0])
        for _ in range(int(max_expand)):
            grow_lo = expand_lo and accept[0] == 1
            grow_hi = expand_hi and accept[-1] == 1
            if not (grow_lo or grow_hi):
                break
            n_new = int(grid_points)
            if grow_lo:
                new = grid[0] - step * np.arange(n_new, 0, -1)
                grid = np.concatenate([new, grid])
                accept = np.concatenate([union_accept(new), accept])
            if grow_hi:
                new = grid[-1] + step * np.arange(1, n_new + 1)
                grid = np.concatenate([grid, new])
                accept = np.concatenate([accept, union_accept(new)])
    hit = grid[accept == 1]
    if hit.size == 0 and recentre_sd is not None:
        # HonestDiD centres its default grid at zero. An estimate more than
        # 20 standard deviations from zero lies outside it, so nothing is
        # accepted and the set would be reported as empty. Test the same
        # number of points around the estimate instead.
        centre = float(l_vec @ betahat[n_pre:])
        if not lb <= centre <= ub:
            grid = np.linspace(
                centre - recentre_sd, centre + recentre_sd, int(grid_points)
            )
            accept = union_accept(grid)
            hit = grid[accept == 1]
    if hit.size == 0:
        return float("nan"), float("nan"), grid, accept
    return float(hit.min()), float(hit.max()), grid, accept


def _ci_fixed_s(
    betahat,
    sigma,
    n_pre,
    n_post,
    l_vec,
    m_bar,
    s,
    max_positive,
    alpha,
    hybrid,
    kappa,
    grid,
    seed,
    skip=None,
) -> np.ndarray:
    """Acceptance vector of one ``Delta^RM`` piece over ``grid``."""
    a = create_a_rm(n_pre, n_post, m_bar, s, max_positive)
    return accept_polyhedron(
        betahat,
        sigma,
        n_pre,
        n_post,
        l_vec,
        a,
        np.zeros(a.shape[0]),
        alpha,
        hybrid,
        kappa,
        grid,
        seed=seed,
        skip=skip,
        nonuis_post_rows_only=True,
    )


def rm_confidence_set(
    betahat: np.ndarray,
    sigma: np.ndarray,
    n_pre: int,
    n_post: int,
    m_bar: float,
    l_vec: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    method: str = "C-LF",
    grid_points: int = 1000,
    grid_lb: Optional[float] = None,
    grid_ub: Optional[float] = None,
    seed: int = 0,
    progress: Optional[Callable[[int], None]] = None,
    max_expand: int = 0,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Confidence set for ``l' tau_post`` under ``Delta^RM(Mbar)``.

    Returns ``(lower, upper, grid, accept)``: the smallest and largest
    accepted grid values (``nan`` if none is accepted) and the full
    acceptance vector, so a caller can detect a non-convex or
    boundary-open set.

    The default grid is HonestDiD's: ``grid_points`` values spanning
    ``+/-20 sd(l' tau_post)``. With ``max_expand > 0``, a side whose end
    point is accepted -- and whose bound the caller did not fix through
    ``grid_lb`` / ``grid_ub`` -- is extended by another ``grid_points``
    points at the same step (40 sd on the default grid), up to
    ``max_expand`` times, so the reported bound is the set's end point
    rather than the grid's. Only the new points are tested.

    If the default grid accepts nothing because the estimate itself lies
    outside it, the grid is moved to ``+/-20 sd`` around the estimate.
    """
    betahat = np.asarray(betahat, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float)
    if l_vec is None:
        l_vec = np.eye(n_post)[0]
    l_vec = np.asarray(l_vec, dtype=float).ravel()
    method_u = method.upper().replace("_", "-")
    if method_u in {"C-LF", "LF"}:
        hybrid = "LF"
    elif method_u in {"CONDITIONAL", "ARP"}:
        hybrid = "ARP"
    else:
        raise ValueError("method must be 'C-LF' or 'Conditional'")
    post = sigma[n_pre:, n_pre:]
    sd_theta = float(np.sqrt(l_vec @ post @ l_vec))
    lb = -20.0 * sd_theta if grid_lb is None else float(grid_lb)
    ub = 20.0 * sd_theta if grid_ub is None else float(grid_ub)
    pieces = []
    for s in range(-(n_pre - 1), 1):
        for positive in (True, False):
            a = create_a_rm(n_pre, n_post, m_bar, s, positive)
            pieces.append((a, np.zeros(a.shape[0])))
    return union_confidence_set(
        pieces,
        betahat,
        sigma,
        n_pre,
        n_post,
        l_vec,
        alpha,
        hybrid,
        lb,
        ub,
        grid_points,
        seed=seed,
        nonuis_post_rows_only=True,
        expand_lo=grid_lb is None,
        expand_hi=grid_ub is None,
        max_expand=max_expand,
        recentre_sd=(20.0 * sd_theta if grid_lb is None and grid_ub is None else None),
        progress=progress,
    )
