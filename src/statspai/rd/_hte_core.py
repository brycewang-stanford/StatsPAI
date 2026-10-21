"""Least-squares core of :func:`statspai.rd.rdhte`.

Clean-room implementation (2026-10-21) of the interacted local polynomial
estimator of Calonico, Cattaneo, Farrell, Palomba and Titiunik (2025, bib
keys ``calonico2025treatment``, ``calonico2025rdhte``): on the window
``|x - c| <= h_side`` regress ``y`` on the full interaction
``[1, T] x [1, Xc, ..., Xc^r] x [1, W]`` with kernel weights, and read off the
coefficients of ``T`` and ``T:W``. R ``rdhte`` 0.2.0 (GPL-3) was used only as
a black box; see ``docs/dev/cleanroom/rdhte.md``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..exceptions import DataInsufficient, MethodIncompatibility, NumericalInstability

__all__ = [
    "KERNELS",
    "canonical_kernel",
    "normalize_vce",
    "InteractedFit",
    "interacted_fit",
    "evaluation_grid",
]

KERNELS = ("triangular", "uniform", "epanechnikov")
_KERNEL_ALIASES = {
    "tri": "triangular",
    "triangular": "triangular",
    "uni": "uniform",
    "uniform": "uniform",
    "epa": "epanechnikov",
    "epanechnikov": "epanechnikov",
}
_VCE_ALLOWED = ("hc0", "hc1", "hc2", "hc3", "cr1", "cr2", "cr3")


def canonical_kernel(kernel: str) -> str:
    """Full kernel name; refuses anything outside the three RD kernels."""
    key = str(kernel).lower()
    if key not in _KERNEL_ALIASES:
        raise MethodIncompatibility(
            "kernel must be 'triangular', 'uniform', or 'epanechnikov', "
            f"got {kernel!r}",
            recovery_hint="Use one of 'triangular', 'uniform', 'epanechnikov'.",
            diagnostics={"kernel": kernel},
        )
    return _KERNEL_ALIASES[key]


def _kernel(u: np.ndarray, kernel: str) -> np.ndarray:
    a = np.abs(u)
    inside = a <= 1.0
    if kernel == "triangular":
        return np.where(inside, 1.0 - a, 0.0)
    if kernel == "epanechnikov":
        return np.where(inside, 0.75 * (1.0 - u * u), 0.0)
    return np.where(inside, 0.5, 0.0)


def normalize_vce(vce: Optional[str], clustered: bool) -> str:
    """Variance type after the documented remapping (rdhte help page).

    Defaults: ``'hc3'``, or ``'cr1'`` with a cluster. With a cluster,
    ``hc0`` / ``hc1`` become ``cr1``; without one, ``crK`` becomes ``hcK``;
    both remaps warn. Clustered HC2/HC3 are not implemented.
    """
    if vce is None:
        return "cr1" if clustered else "hc3"
    v = str(vce).lower()
    if v not in _VCE_ALLOWED:
        raise MethodIncompatibility(
            f"vce must be one of hc0-hc3 / cr1; got {vce!r}",
            recovery_hint="Use vce='hc3' (default), 'hc0', 'hc1', 'hc2', or "
            "cluster= with vce='cr1'.",
            diagnostics={"vce": vce},
        )
    if clustered:
        if v in ("hc0", "hc1"):
            warnings.warn(
                f"rdhte: vce={v!r} with cluster= is computed as 'cr1'.",
                UserWarning,
                stacklevel=3,
            )
            return "cr1"
        if v == "cr1":
            return v
        raise MethodIncompatibility(
            f"rdhte: clustered vce={v!r} (CR2/CR3) is not implemented.",
            recovery_hint="Use vce='cr1' with cluster=.",
            diagnostics={"vce": vce},
        )
    if v.startswith("cr"):
        new = "hc" + v[2:]
        warnings.warn(
            f"rdhte: vce={v!r} without cluster= is computed as {new!r}.",
            UserWarning,
            stacklevel=3,
        )
        return new
    return v


@dataclass
class InteractedFit:
    """Coefficients and covariance of one interacted least-squares fit."""

    beta: np.ndarray
    vcov: np.ndarray
    treat_index: List[int]  # columns T, T:W_1, ..., T:W_d (in that order)
    n_window: int
    n_side: Tuple[int, int]


def _design(
    xc: np.ndarray, treat: np.ndarray, W: Optional[np.ndarray], order: int
) -> Tuple[np.ndarray, List[int]]:
    """Full interaction ``[1, T] x [1, xc, .., xc^order] x [1, W]``."""
    n = xc.size
    powers = [np.ones(n)] + [xc**j for j in range(1, order + 1)]
    mods = [np.ones(n)] + ([] if W is None else [W[:, j] for j in range(W.shape[1])])
    cols: List[np.ndarray] = []
    treat_index: List[int] = []
    for t_on, t_col in ((0, np.ones(n)), (1, treat)):
        for j, pw in enumerate(powers):
            for m, mv in enumerate(mods):
                if t_on and j == 0:
                    treat_index.append(len(cols))
                cols.append(t_col * pw * mv)
    return np.column_stack(cols), treat_index


def interacted_fit(
    y: np.ndarray,
    xc: np.ndarray,
    W: Optional[np.ndarray],
    h_left: np.ndarray,
    h_right: np.ndarray,
    order: int,
    kernel: str,
    vce: str,
    cluster: Optional[np.ndarray] = None,
) -> InteractedFit:
    """Kernel-weighted interacted fit on the window and its sandwich covariance.

    ``h_left`` / ``h_right`` are per-observation bandwidths (subgroups carry
    their own).
    """
    treat = (xc >= 0).astype(float)
    hh = np.where(treat == 1.0, h_right, h_left)
    u = xc / hh
    inside = np.abs(u) <= 1.0
    X_all, treat_index = _design(xc, treat, W, order)
    X = X_all[inside]
    Y = y[inside]
    w = (_kernel(u, kernel) / hh)[inside]
    n, k = X.shape
    n_left = int(np.sum(inside & (treat == 0.0)))
    n_right = int(np.sum(inside & (treat == 1.0)))
    if n <= k or min(n_left, n_right) == 0:
        raise DataInsufficient(
            f"Not enough observations inside the h window ({n}) for the "
            f"order-{order} interacted fit",
            recovery_hint="Widen the bandwidth or lower the polynomial order.",
            diagnostics={"n_window": n, "n_params": k, "order": order},
        )
    XtW = X.T * w
    A = XtW @ X
    rank = np.linalg.matrix_rank(A)
    if rank < k:
        raise NumericalInstability(
            f"rdhte: the interacted design is rank deficient ({rank} < {k}) "
            "inside the bandwidth window.",
            recovery_hint="A moderator is constant (or collinear) near the "
            "cutoff; drop it, widen the bandwidth or lower p.",
            diagnostics={"rank": int(rank), "n_params": k},
        )
    A_inv = np.linalg.inv(A)
    beta = A_inv @ (XtW @ Y)
    resid = Y - X @ beta
    scores = X * (w * resid)[:, None]
    if vce == "cr1":
        codes = np.unique(np.asarray(cluster)[inside], return_inverse=True)[1]
        G = int(codes.max()) + 1
        if G < 2:
            raise DataInsufficient(
                "rdhte: fewer than two clusters inside the bandwidth window.",
                recovery_hint="Use more clusters or drop cluster=.",
                diagnostics={"n_clusters": G},
            )
        sums = np.zeros((G, k))
        np.add.at(sums, codes, scores)
        meat = (G / (G - 1.0)) * ((n - 1.0) / (n - k)) * (sums.T @ sums)
    elif vce in ("hc2", "hc3"):
        lev = np.einsum("ij,jk,ik->i", X, A_inv, X) * w
        power = 0.5 if vce == "hc2" else 1.0
        adj = scores / ((1.0 - lev) ** power)[:, None]
        meat = adj.T @ adj
    else:
        factor = n / (n - 1.0) if vce == "hc0" else n / (n - float(k))
        meat = factor * (scores.T @ scores)
    vcov = A_inv @ meat @ A_inv
    return InteractedFit(
        beta=beta,
        vcov=0.5 * (vcov + vcov.T),
        treat_index=treat_index,
        n_window=n,
        n_side=(n_left, n_right),
    )


def evaluation_grid(Z: np.ndarray, n_eval: int) -> np.ndarray:
    """Default CATE evaluation points (StatsPAI convention).

    One moderator: quantiles at ``linspace(0.1, 0.9, n_eval)``. ``d`` moderators:
    the product of per-column quantiles at ``linspace(0.1, 0.9, k)`` with
    ``k = max(3, floor(n_eval ** (1/d)))``, thinned to ``n_eval`` rows at
    ``round(linspace(0, G - 1, n_eval))`` when larger.
    """
    d = Z.shape[1]
    if d == 1:
        _ret: np.ndarray = np.percentile(Z[:, 0], np.linspace(10.0, 90.0, n_eval))[
            :, None
        ]
        return _ret
    k = max(3, int(n_eval ** (1.0 / d)))
    probs = np.linspace(0.1, 0.9, k)
    axes = [np.quantile(Z[:, j], probs) for j in range(d)]
    mesh = np.meshgrid(*axes, indexing="ij")
    grid = np.column_stack([m.ravel() for m in mesh])
    if grid.shape[0] > n_eval:
        rows = np.round(np.linspace(0, grid.shape[0] - 1, n_eval)).astype(int)
        grid = grid[rows]
    return grid


def subgroup_levels(values: Sequence) -> List:
    """Sorted distinct levels (category order for a pandas Categorical)."""
    cats = getattr(getattr(values, "cat", None), "categories", None)
    if cats is not None:
        present = set(values.dropna().unique())
        return [c for c in cats if c in present]
    return sorted(set(values))
