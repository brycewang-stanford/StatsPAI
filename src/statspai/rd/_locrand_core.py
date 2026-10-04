"""Shared primitives for local-randomization RD (``rd/locrand.py``).

Three things live here so that ``rdrandinf``, ``rdwinselect`` and
``rdsensitivity`` cannot drift apart:

* the **statistic as a linear functional**. The difference in means, its
  kernel-weighted version and the difference in intercepts of side-specific
  polynomial fits are all ``g @ y`` for a weight vector ``g`` that depends
  only on the score. That is what makes the randomization distribution and
  the test-inversion confidence interval cheap: one matrix product each.
* the **randomization draws**, in bounded-memory chunks.
* the **nested window sequence** used for window selection.

Formulas were fixed against ``rdlocrand`` 2.0 by output only (its source is
GPL and was not consulted): the observed statistics and large-sample
p-values agree to 1e-13 on the U.S. Senate data for every kernel,
polynomial order and evaluation point tried. See
``tests/reference_parity/test_rdlocrand_extensions_parity.py``.
"""

from typing import Iterator, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

from ..exceptions import DataInsufficient, MethodIncompatibility

KERNELS = ("uniform", "triangular", "epanechnikov")
_KERNEL_ALIASES = {
    "uniform": "uniform",
    "uni": "uniform",
    "triangular": "triangular",
    "tri": "triangular",
    "epanechnikov": "epanechnikov",
    "epan": "epanechnikov",
}

# Draws are generated in blocks of at most this many cells so that a window
# holding tens of thousands of observations does not allocate a
# (n_perms x n) matrix in one piece.
_CHUNK_CELLS = 4_000_000


def canonical_kernel(kernel: str) -> str:
    try:
        return _KERNEL_ALIASES[str(kernel).lower()]
    except KeyError:
        raise MethodIncompatibility(
            f"Unknown kernel {kernel!r}. Choose from 'uniform', 'triangular', "
            "'epanechnikov' (alias 'epan')."
        ) from None


def kernel_weights(
    xc: np.ndarray, t: np.ndarray, bw_left: float, bw_right: float, kernel: str
) -> np.ndarray:
    """Kernel weights on the centred score.

    The bandwidth is the distance from the cutoff to the window edge on the
    observation's own side, so an asymmetric window gets two bandwidths.
    """
    kernel = canonical_kernel(kernel)
    if kernel == "uniform":
        return np.ones_like(xc, dtype=float)
    bw = np.where(t == 1, bw_right, bw_left)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(bw > 0, xc / bw, 0.0)
    if kernel == "triangular":
        return np.asarray(np.maximum(1.0 - np.abs(u), 0.0), dtype=float)
    return np.asarray(np.maximum(0.75 * (1.0 - u**2), 0.0), dtype=float)


def linear_functional(
    xc: np.ndarray,
    t: np.ndarray,
    w: np.ndarray,
    p: int,
    eval_left: float = 0.0,
    eval_right: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Weights ``g`` with ``g @ y`` = difference in fitted values at the cutoff.

    Each side gets its own weighted polynomial of order ``p`` in the score,
    evaluated at ``eval_right`` / ``eval_left`` (centred, so 0 is the
    cutoff). With ``p = 0`` this is the weighted difference in means.

    Returns ``(g, leverage)``; the leverage is that of the weighted fit and
    feeds the HC2 variance.
    """
    g = np.zeros(len(xc), dtype=float)
    lev = np.zeros(len(xc), dtype=float)
    for side, ev, sign in ((1, eval_right, 1.0), (0, eval_left, -1.0)):
        m = t == side
        X = np.vander(xc[m] - ev, p + 1, increasing=True)
        ws = w[m]
        XtWX = X.T @ (X * ws[:, None])
        try:
            A = np.linalg.inv(XtWX)
        except np.linalg.LinAlgError:
            raise DataInsufficient(
                f"Polynomial of order {p} is not identified on the "
                f"{'right' if side else 'left'} of the cutoff inside this "
                "window (too few distinct score values with positive kernel "
                "weight). Lower p or widen the window."
            ) from None
        if not np.all(np.isfinite(A)) or np.linalg.cond(XtWX) > 1e13:
            raise DataInsufficient(
                f"Polynomial of order {p} is not identified on the "
                f"{'right' if side else 'left'} of the cutoff inside this "
                "window (too few distinct score values with positive kernel "
                "weight). Lower p or widen the window."
            )
        H = (X @ A) * ws[:, None]  # row i: e_i' X A X' W restricted to i
        g[m] = sign * H[:, 0]
        lev[m] = np.einsum("ij,ij->i", X @ A, X) * ws
    return g, lev


def hc2_se(
    y: np.ndarray,
    xc: np.ndarray,
    t: np.ndarray,
    w: np.ndarray,
    p: int,
    eval_left: float = 0.0,
    eval_right: float = 0.0,
) -> float:
    """HC2 standard error of ``g @ y``; Welch's SE when p = 0 and unweighted."""
    var = 0.0
    for side, ev in ((1, eval_right), (0, eval_left)):
        m = t == side
        X = np.vander(xc[m] - ev, p + 1, increasing=True)
        ws = w[m]
        A = np.linalg.inv(X.T @ (X * ws[:, None]))
        beta = A @ (X.T @ (ws * y[m]))
        resid = y[m] - X @ beta
        lev = np.einsum("ij,ij->i", X @ A, X) * ws
        denom = 1.0 - lev
        if np.any(denom <= 1e-12):
            return float("nan")
        meat = (X * (ws**2 * resid**2 / denom)[:, None]).T @ X
        var += float((A @ meat @ A)[0, 0])
    return float(np.sqrt(var))


def ranksum_from_labels(ranks: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Standardised control-group rank sum for each row of ``labels``.

    Uses the empirical variance of the midranks, which is what keeps the
    statistic right on tied data (see ``locrand._ranksum_stat``).
    """
    n = ranks.shape[0]
    n1 = labels.sum(axis=1).astype(float)
    n0 = n - n1
    s2 = float(np.var(ranks, ddof=1)) if n > 1 else 0.0
    t_stat = (1.0 - labels) @ ranks
    var_t = n0 * n1 * s2 / n
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (t_stat - n0 * (n + 1) / 2.0) / np.sqrt(var_t)
    return np.where(var_t > 0, out, 0.0)


def ks_from_labels(y: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Two-sample Kolmogorov-Smirnov statistic for each row of ``labels``."""
    order = np.argsort(y, kind="mergesort")
    ys = y[order]
    L = labels[:, order].astype(float)
    n1 = L.sum(axis=1, keepdims=True)
    n0 = L.shape[1] - n1
    with np.errstate(divide="ignore", invalid="ignore"):
        gap = np.cumsum(L, axis=1) / n1 - np.cumsum(1.0 - L, axis=1) / n0
    # The ECDFs are compared at the last observation of each tie block.
    last = np.r_[ys[1:] != ys[:-1], True]
    return np.asarray(np.nanmax(np.abs(gap[:, last]), axis=1), dtype=float)


def chunks(n_draws: int, n: int) -> Iterator[int]:
    """Block sizes that keep each draw matrix under ``_CHUNK_CELLS`` cells."""
    step = max(1, min(n_draws, _CHUNK_CELLS // max(n, 1)))
    done = 0
    while done < n_draws:
        size = min(step, n_draws - done)
        yield size
        done += size


def permutation_indices(rng: np.random.Generator, size: int, n: int) -> np.ndarray:
    """``size`` independent permutations of ``range(n)``, one per row."""
    return np.asarray(rng.permuted(np.tile(np.arange(n), (size, 1)), axis=1))


def bernoulli_labels(
    rng: np.random.Generator, size: int, prob: np.ndarray
) -> np.ndarray:
    """Independent Bernoulli assignments; rows with an empty arm are dropped."""
    lab = (rng.random((size, prob.shape[0])) < prob[None, :]).astype(np.int8)
    n1 = lab.sum(axis=1)
    return np.asarray(lab[(n1 > 0) & (n1 < prob.shape[0])])


def asymptotic_power(d: float, se: float, alpha: float) -> float:
    """Power of the two-sided z-test against a shift of ``d``."""
    if not np.isfinite(se) or se <= 0:
        return float("nan")
    z = sp_stats.norm.ppf(1 - alpha / 2)
    return float(1 - sp_stats.norm.cdf(z - d / se) + sp_stats.norm.cdf(-z - d / se))


# ----------------------------------------------------------------------
# Window sequence
# ----------------------------------------------------------------------


def _kth_distance(dist_sorted: np.ndarray, k: int) -> Optional[float]:
    """Distance of the k-th closest observation, or None if there are fewer."""
    if k < 1 or k > dist_sorted.shape[0]:
        return None
    return float(dist_sorted[k - 1])


def window_sequence(
    x: np.ndarray,
    c: float,
    *,
    nwindows: int,
    obsmin: int = 10,
    wmin: Optional[float] = None,
    wobs: Optional[int] = None,
    wstep: Optional[float] = None,
    wasymmetric: bool = False,
) -> List[Tuple[float, float]]:
    """Nested windows around the cutoff, as half-widths ``(left, right)``.

    The first window is ``wmin`` if given, otherwise the smallest window
    holding at least ``obsmin`` observations on each side. Each later window
    either grows by ``wstep`` or, by default, by enough to add at least
    ``wobs`` observations on each side (5 unless stated). Symmetric windows
    take the wider of the two sides; ``wasymmetric`` lets each side grow on
    its own.

    The sequence stops early when the data run out, so fewer than
    ``nwindows`` windows may come back.
    """
    if wobs is not None and wstep is not None:
        raise MethodIncompatibility("Pass at most one of wobs= and wstep=.")
    if wasymmetric and (wmin is not None or wstep is not None):
        raise MethodIncompatibility(
            "wasymmetric=True builds each side from observation counts and "
            "cannot be combined with wmin= or wstep=."
        )
    left = np.sort(c - x[x < c])
    right = np.sort(x[x >= c] - c)
    step_obs = 5 if (wobs is None and wstep is None) else wobs

    def count(side: np.ndarray, w: float) -> int:
        return int(np.searchsorted(side, w, side="right"))

    if wmin is not None:
        if wmin <= 0:
            raise MethodIncompatibility("wmin must be positive.")
        cur_l = cur_r = float(wmin)
    else:
        first_l = _kth_distance(left, obsmin)
        first_r = _kth_distance(right, obsmin)
        if first_l is None or first_r is None:
            raise DataInsufficient(
                f"Fewer than obsmin={obsmin} observations on one side of the "
                f"cutoff ({left.shape[0]} left, {right.shape[0]} right)."
            )
        if wasymmetric:
            cur_l, cur_r = first_l, first_r
        else:
            cur_l = cur_r = max(first_l, first_r)

    out: List[Tuple[float, float]] = [(cur_l, cur_r)]
    while len(out) < nwindows:
        if cur_l >= left[-1] and cur_r >= right[-1]:
            break  # the window already holds every observation
        if wstep is not None:
            nxt_l = nxt_r = cur_l + float(wstep)
        else:
            assert step_obs is not None
            want_l = _kth_distance(left, count(left, cur_l) + step_obs)
            want_r = _kth_distance(right, count(right, cur_r) + step_obs)
            if want_l is None or want_r is None:
                break
            if wasymmetric:
                nxt_l, nxt_r = want_l, want_r
            else:
                nxt_l = nxt_r = max(want_l, want_r)
        out.append((nxt_l, nxt_r))
        cur_l, cur_r = nxt_l, nxt_r
    return out
