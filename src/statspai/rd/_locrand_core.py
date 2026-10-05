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

Formulas were fixed against ``rdlocrand`` 2.0 and then 3.0 by output only
(its source is GPL and was not consulted): the observed statistics and
large-sample p-values agree to 1e-12 on the U.S. Senate data for every
kernel, polynomial order, evaluation point and HC variance tried. See
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


def hc_se(
    y: np.ndarray,
    xc: np.ndarray,
    t: np.ndarray,
    w: np.ndarray,
    p: int,
    eval_left: float = 0.0,
    eval_right: float = 0.0,
    vce: str = "hc2",
) -> float:
    """Heteroskedasticity-robust standard error of ``g @ y``.

    ``g @ y`` is the difference of the two side-specific polynomial fits at
    their evaluation points, which is the coefficient on treatment in the
    regression of the outcome on treatment, the polynomial and their
    interaction. That design is block-diagonal across the sides, so the
    leverages are the side-specific ones. ``'hc2'`` divides each squared
    residual by ``1 - h`` (Welch's standard error when ``p = 0`` and the
    weights are equal), ``'hc3'`` by ``(1 - h)^2``, and ``'hc1'`` scales the
    whole variance by ``n / (n - k)`` with ``k = 2 (p + 1)`` coefficients.
    """
    if vce not in ("hc1", "hc2", "hc3"):
        raise MethodIncompatibility(f"vce must be 'hc1', 'hc2' or 'hc3', got {vce!r}")
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
        if vce == "hc1":
            scale = np.ones_like(denom)
        elif np.any(denom <= 1e-12):
            return float("nan")
        else:
            scale = 1.0 / denom if vce == "hc2" else 1.0 / denom**2
        meat = (X * (ws**2 * resid**2 * scale)[:, None]).T @ X
        var += float((A @ meat @ A)[0, 0])
    if vce == "hc1":
        n, k = int(y.shape[0]), 2 * (p + 1)
        if n <= k:
            return float("nan")
        var *= n / (n - k)
    return float(np.sqrt(var))


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
    return hc_se(y, xc, t, w, p, eval_left, eval_right, "hc2")


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


def mass_point_windows(
    x: np.ndarray, c: float, *, nwindows: int
) -> List[Tuple[float, float]]:
    """Windows at successive support points of a discrete score.

    Window ``k`` reaches the ``k``-th distinct value below the cutoff and
    the ``k``-th distinct value at or above it, as half-widths
    ``(left, right)``. Stops when either side runs out of support points.
    """
    left = np.unique(c - x[x < c])
    right = np.unique(x[x >= c] - c)
    k = min(int(nwindows), left.shape[0], right.shape[0])
    if k < 1:
        raise DataInsufficient(
            "Need observations on both sides of the cutoff to build "
            "mass-point windows."
        )
    return [(float(left[i]), float(right[i])) for i in range(k)]


def hotelling_t2(Z: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Two-sample Hotelling T-squared for each row of ``labels``.

    ``T2 = n1 n0 / n * d' S^{-1} d`` with ``d`` the difference in mean
    vectors and ``S`` the pooled covariance. The pooled sums of squares are
    the total ones minus the between-group part, so only ``d`` has to be
    recomputed for a relabelling.
    """
    n, k = Z.shape
    lab = np.atleast_2d(labels).astype(float)
    n1 = lab.sum(axis=1)
    n0 = n - n1
    centred = Z - Z.mean(axis=0)
    total = centred.T @ centred
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (lab @ centred) / n1[:, None] - ((1.0 - lab) @ centred) / n0[:, None]
        scale = n1 * n0 / n
        pooled = (
            total[None, :, :] - scale[:, None, None] * d[:, :, None] * d[:, None, :]
        ) / (n - 2)
        out = np.full(lab.shape[0], np.nan)
        ok = (n1 > 0) & (n0 > 0)
        try:
            sol = np.linalg.solve(pooled[ok], d[ok][:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            return out
        out[ok] = scale[ok] * np.einsum("ij,ij->i", d[ok], sol)
    return out


def hotelling_pvalue_f(t2: float, n: int, k: int) -> float:
    """Large-sample (F) p-value of a two-sample Hotelling T-squared."""
    if not np.isfinite(t2) or n - k - 1 <= 0:
        return float("nan")
    f_stat = (n - k - 1) / ((n - 2) * k) * t2
    return float(sp_stats.f.sf(f_stat, k, n - k - 1))


def ks_exact_pvalue(y1: np.ndarray, y0: np.ndarray, stat: float) -> float:
    """Exact two-sided p-value of the two-sample Kolmogorov-Smirnov statistic,
    valid with ties.

    Conditional on the pooled sample, every assignment of ``len(y1)`` of the
    observations to the first group is equally likely under the null. Each
    is a lattice path from (0, 0) to (m, n); the two empirical distribution
    functions can only be compared where a block of tied values ends, so a
    path stays "inside" if ``|i / m - j / n| < stat`` at those points
    (Schroer and Trenkler's algorithm). Without ties every point is such a
    point and this is the classical exact distribution.
    """
    m, n = len(y1), len(y0)
    pooled = np.sort(np.concatenate([y1, y0]))
    # check[k] is True when the k-th pooled observation (1-based) closes a
    # block of tied values.
    check = np.r_[False, pooled[1:] != pooled[:-1], True]
    # The statistic is a multiple of 1 / (m n); compare against the grid
    # point just below it so that rounding cannot move a boundary path.
    q = (0.5 + np.floor(stat * m * n - 1e-7)) / (m * n)
    row = np.zeros(n + 1)
    j = np.arange(n + 1)
    for i in range(m + 1):
        new = np.zeros(n + 1)
        outside = check[i + j] & (np.abs(i / m - j / n) >= q)
        for jj in range(n + 1):
            if i == 0 and jj == 0:
                value = 1.0
            else:
                value = (row[jj] if i > 0 else 0.0) + (new[jj - 1] if jj > 0 else 0.0)
            new[jj] = 0.0 if outside[jj] else value
        row = new
    from scipy.special import comb

    total = comb(m + n, m, exact=False)
    return float(min(1.0, max(0.0, 1.0 - row[n] / total)))
