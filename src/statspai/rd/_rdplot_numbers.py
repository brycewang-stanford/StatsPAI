"""Numbers behind :func:`statspai.rdplot` (binned means and global fits).

Clean-room rewrite; reference (R rdrobust 4.0.0 rdplot, GPL-3) used as a
black box. Every rule below comes from the papers, the package's help
pages, or from comparing outputs of the reference on test data; the
record of which is which is ``docs/dev/cleanroom/rdplot.md``.

What is computed
----------------
* The number of bins on each side, by the eight selectors of
  Calonico, Cattaneo and Titiunik (bib key ``calonico2015optimal``):
  integrated-MSE optimal (``es``, ``qs``) or
  mimicking-variance (``esmv``, ``qsmv``) choices, for evenly spaced (ES)
  or quantile spaced (QS) partitions, with the unknown constants estimated
  either by spacings (concomitants of the order statistics) or by
  polynomial regression (the ``pr`` variants).
* The partition, the per-bin sample means of ``x`` and ``y``, their
  standard errors and Student-t intervals (cluster-robust on request).
* The kernel-weighted global polynomial of order ``p`` on each side,
  optionally with covariates entering linearly and evaluated at their
  sample means.

Plotting lives in :func:`statspai.rd.rdrobust.rdplot`.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from ..exceptions import DataInsufficient, MethodIncompatibility
from ._core import _kernel_fn

__all__ = ["rdplot_numbers", "kernel_weights", "gram_inverse"]

_BINSELECT = ("es", "espr", "esmv", "esmvpr", "qs", "qspr", "qsmv", "qsmvpr")
_KERNELS = {
    "uni": "uniform",
    "uniform": "uniform",
    "tri": "triangular",
    "triangular": "triangular",
    "epa": "epanechnikov",
    "epanechnikov": "epanechnikov",
}
# Polynomial order of the preliminary fits behind the bin selectors
# (black box: the selected numbers of bins do not move with ``p``).
_SELECTOR_ORDER = 4
# Number of grid points per side in ``vars_poly`` (black box).
_GRID = 500
# Minimum sample size (black box: the reference stops below 20).
_MIN_N = 20
# A side has mass points when fewer than this share of its values are
# distinct (black box: 79 distinct of 100 switches the rule, 80 does not).
_MASS_SHARE = 0.8


# --------------------------------------------------------------------------- #
#  small helpers
# --------------------------------------------------------------------------- #
def _kernel_name(kernel: str) -> str:
    key = str(kernel).lower()
    if key not in _KERNELS:
        raise MethodIncompatibility(
            f"kernel must be 'triangular', 'epanechnikov' or 'uniform', got {kernel!r}.",
            recovery_hint="Use kernel='uniform' (the rdplot default), "
            "'triangular' or 'epanechnikov'.",
            diagnostics={"kernel": kernel},
        )
    return _KERNELS[key]


def kernel_weights(x: np.ndarray, c: float, h: float, kernel: str) -> np.ndarray:
    """Kernel weights ``K((x - c) / h) / h`` for the global fits.

    Parameters
    ----------
    x : array_like
        Running variable.
    c : float
        Cutoff.
    h : float
        Bandwidth (positive).
    kernel : str
        ``'uniform'`` / ``'uni'`` (``K = 1/2`` on ``|u| <= 1``),
        ``'triangular'`` / ``'tri'`` (``K = 1 - |u|``) or
        ``'epanechnikov'`` / ``'epa'`` (``K = 3/4 (1 - u^2)``). Case is
        ignored.

    Returns
    -------
    numpy.ndarray
        The weights, zero outside ``[c - h, c + h]``.

    Examples
    --------
    >>> import numpy as np
    >>> from statspai.rd._rdplot_numbers import kernel_weights
    >>> kernel_weights(np.array([-0.5, 0.0, 0.25, 2.0]), 0.0, 0.5, "epa")
    array([0.   , 1.5  , 1.125, 0.   ])
    """
    name = _kernel_name(kernel)
    u = (np.asarray(x, dtype=float) - float(c)) / float(h)
    return _kernel_fn(u, name) / float(h)


def gram_inverse(X: np.ndarray) -> Optional[np.ndarray]:
    """Inverse of ``X'X`` by Cholesky, or ``None`` if not positive definite.

    Parameters
    ----------
    X : numpy.ndarray
        Design matrix, ``n x k``.

    Returns
    -------
    numpy.ndarray or None
        ``(X'X)^{-1}``, or ``None`` when ``X'X`` is singular.

    Examples
    --------
    >>> import numpy as np
    >>> from statspai.rd._rdplot_numbers import gram_inverse
    >>> gram_inverse(np.ones((5, 2))) is None
    True
    """
    X = np.asarray(X, dtype=float)
    G = X.T @ X
    try:
        L = np.linalg.cholesky(G)
    except np.linalg.LinAlgError:
        return None
    Linv = np.linalg.solve(L, np.eye(G.shape[0]))
    return Linv.T @ Linv


def _powers(u: np.ndarray, order: int) -> np.ndarray:
    return np.vander(np.asarray(u, dtype=float), order + 1, increasing=True)


def _lsq(X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
    """(Weighted) least squares; minimum-norm solution when rank deficient."""
    if w is not None:
        sw = np.sqrt(w)
        X = X * sw[:, None]
        y = y * sw if y.ndim == 1 else y * sw[:, None]
    coef: np.ndarray = np.linalg.lstsq(X, y, rcond=None)[0]
    return coef


def _pair(value: Any, name: str) -> Tuple[Any, Any]:
    arr = np.atleast_1d(np.asarray(value, dtype=float)).ravel()
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    if arr.size == 2:
        return float(arr[0]), float(arr[1])
    raise MethodIncompatibility(
        f"{name} takes one value or one per side (left, right); got {arr.size} values.",
        recovery_hint=f"Pass {name}=value or {name}=(left, right).",
        diagnostics={name: arr.tolist()},
    )


def _positive_pair(value: Any, name: str, integer: bool = False) -> Tuple[Any, Any]:
    a, b = _pair(value, name)
    for v in (a, b):
        if not np.isfinite(v) or v <= 0 or (integer and v != math.floor(v)):
            kind = "positive integers" if integer else "positive numbers"
            raise MethodIncompatibility(
                f"{name} must be {kind}; got {value!r}.",
                recovery_hint=f"Pass {name} as {kind}, one value or (left, right).",
                diagnostics={name: [a, b]},
            )
    if integer:
        return int(a), int(b)
    return a, b


# --------------------------------------------------------------------------- #
#  number of bins (Calonico, Cattaneo and Titiunik 2015, sections 3-5)
# --------------------------------------------------------------------------- #
def _selector_constants(
    xs: np.ndarray, ys: np.ndarray, c: float, length: float, n: int
) -> Dict[str, float]:
    """Plug-in constants of the bin selectors on one side.

    ``xs`` is sorted (stable) and ``ys`` holds the concomitants. Returns
    the bias constants ``B_es`` / ``B_qs``, the variance constants by
    spacings (``V_es`` / ``V_qs``) and by polynomial regression
    (``V_espr`` / ``V_qspr``), and the sample variance ``var_y``.
    """
    N = xs.size
    u = xs - c
    dx = np.diff(xs)
    dy = np.diff(ys)
    mid = (u[1:] + u[:-1]) / 2.0
    order = _SELECTOR_ORDER

    beta1 = _lsq(_powers(u, order), ys)
    beta2 = _lsq(_powers(u, order), ys * ys)
    # derivative of the order-4 fit
    dcoef = beta1[1:] * np.arange(1, order + 1)

    def slope(t: np.ndarray) -> np.ndarray:
        d: np.ndarray = _powers(t, order - 1) @ dcoef
        return d

    def cond_var(t: np.ndarray) -> np.ndarray:
        P = _powers(t, order)
        v: np.ndarray = P @ beta2 - (P @ beta1) ** 2
        return v

    var_y = float(np.var(ys, ddof=1)) if N > 1 else float("nan")

    # Eq. (8)/(10): bias constant, ES; eq. (18)/(20): QS.
    B_es = length**2 / (12.0 * n) * float(np.sum(slope(u) ** 2))
    B_qs = N**2 / (24.0 * n) * float(np.sum(dx**2 * slope(mid) ** 2))
    # Eq. (7)/(9) and (17)/(19): spacings variance constants.
    V_es = float(np.sum(dx * dy**2)) / (2.0 * length)
    V_qs = float(np.sum(dy**2)) / (2.0 * N)
    # Remarks 1-2: polynomial-regression variance constants, with
    # sigma^2(x) = E[Y^2 | x] - E[Y | x]^2 from order-4 fits. Negative
    # estimates of sigma^2(x) are replaced by the sample variance of y
    # (black box).
    s_mid = cond_var(mid)
    s_obs = cond_var(u)
    s_mid = np.where(s_mid < 0, var_y, s_mid)
    s_obs = np.where(s_obs < 0, var_y, s_obs)
    V_espr = float(np.sum(dx * s_mid)) / length
    V_qspr = float(np.mean(s_obs))
    return {
        "B_es": B_es,
        "B_qs": B_qs,
        "V_es": V_es,
        "V_qs": V_qs,
        "V_espr": V_espr,
        "V_qspr": V_qspr,
        "var_y": var_y,
    }


def _ceil_count(value: float) -> Optional[int]:
    if not np.isfinite(value):
        return None
    return max(int(math.ceil(value)), 1)


def _side_bins(
    const: Dict[str, float], method: str, n: int
) -> Tuple[Optional[int], Optional[int]]:
    """IMSE-optimal and mimicking-variance numbers of bins on one side."""
    spacing = "V_es" if method.startswith("es") else "V_qs"
    if method.endswith("pr"):
        spacing += "pr"
    V = const[spacing]
    B = const["B_es"] if method.startswith("es") else const["B_qs"]
    if not (np.isfinite(V) and V > 0):
        return None, None
    j_imse = _ceil_count((2.0 * B / V) ** (1.0 / 3.0) * n ** (1.0 / 3.0))
    j_mv = _ceil_count(const["var_y"] / V * n / math.log(n) ** 2)
    return j_imse, j_mv


# --------------------------------------------------------------------------- #
#  partitions
# --------------------------------------------------------------------------- #
def _even_edges(lo: float, hi: float, J: int) -> np.ndarray:
    """``lo, lo + d, lo + 2d, ...`` with ``d = (hi - lo) / J``, capped at ``hi``."""
    step = (hi - lo) / J
    edges = lo + np.arange(J + 1) * step
    capped: np.ndarray = np.minimum(edges, hi)
    return capped


def _quantile_edges(values: np.ndarray, J: int) -> np.ndarray:
    """Sample quantiles at ``p_j = j * (1/J)``, ``j = 0..J`` (Hyndman-Fan type 7).

    With order statistics ``x_(1) <= ... <= x_(m)``, position
    ``r = 1 + (m - 1) p``, ``k = floor(r)`` and ``g = r - k``, the quantile
    is ``(1 - g) x_(k) + g x_(k+1)`` (``x_(k)`` when the two are equal).
    Evaluated in exactly this order of operations, and with ``p_j``
    formed as ``j`` times ``1/J`` (not ``j / J``), the interior edges equal
    the reference's bit for bit (black box). That decides on which side of
    an edge an observation lying on it falls.
    """
    srt = np.sort(values)
    m = srt.size
    probs = np.arange(J + 1) * (1.0 / J)
    pos = 1.0 + (m - 1) * probs
    k = np.floor(pos).astype(int)
    g = pos - k
    a = srt[np.clip(k - 1, 0, m - 1)]
    b = srt[np.clip(k, 0, m - 1)]
    return np.where(b != a, (1.0 - g) * a + g * b, a)


def _assign(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin index ``j`` with ``edges[j] <= x < edges[j + 1]``; last bin closed."""
    J = edges.size - 1
    idx = np.searchsorted(edges, x, side="right") - 1
    return np.clip(idx, 0, J - 1)


def _bin_table(
    xs: np.ndarray,
    ys: np.ndarray,
    edges: np.ndarray,
    level: float,
    cluster: Optional[np.ndarray],
) -> Dict[str, np.ndarray]:
    """Per-bin counts, means, standard errors and t intervals (non-empty bins)."""
    idx = _assign(xs, edges)
    J = edges.size - 1
    cols: Dict[str, list] = {
        k: []
        for k in (
            "rdplot_mean_bin",
            "rdplot_mean_x",
            "rdplot_mean_y",
            "rdplot_min_bin",
            "rdplot_max_bin",
            "rdplot_se_y",
            "rdplot_N",
            "rdplot_ci_l",
            "rdplot_ci_r",
        )
    }
    tail = 1.0 - (1.0 - level / 100.0) / 2.0
    for j in range(J):
        m = idx == j
        Nj = int(m.sum())
        if Nj == 0:
            continue
        yj = ys[m]
        mean_y = float(np.mean(yj))
        if cluster is None:
            df = Nj - 1
            # one observation: standard error 0 and a degenerate interval,
            # as the reference reports
            se = float(np.std(yj, ddof=1) / np.sqrt(Nj)) if Nj > 1 else 0.0
        else:
            _, g = np.unique(cluster[m], return_inverse=True)
            G = int(g.max()) + 1
            df = G - 1
            if G > 1:
                score = np.bincount(g, weights=yj - mean_y, minlength=G)
                se = float(np.sqrt(G / (G - 1.0) * np.sum(score**2)) / Nj)
            else:
                se = float("nan")
        if df > 0:
            q = float(stats.t.ppf(tail, df))
        else:
            q = 0.0 if se == 0.0 else float("nan")
        lo_e, hi_e = float(edges[j]), float(edges[j + 1])
        cols["rdplot_mean_bin"].append((lo_e + hi_e) / 2.0)
        cols["rdplot_mean_x"].append(float(np.mean(xs[m])))
        cols["rdplot_mean_y"].append(mean_y)
        cols["rdplot_min_bin"].append(lo_e)
        cols["rdplot_max_bin"].append(hi_e)
        cols["rdplot_se_y"].append(se)
        cols["rdplot_N"].append(Nj)
        cols["rdplot_ci_l"].append(mean_y - q * se)
        cols["rdplot_ci_r"].append(mean_y + q * se)
    out = {k: np.asarray(v, dtype=float) for k, v in cols.items()}
    out["rdplot_N"] = np.asarray(cols["rdplot_N"], dtype=np.int64)
    return out


# --------------------------------------------------------------------------- #
#  main entry
# --------------------------------------------------------------------------- #
def rdplot_numbers(
    y: np.ndarray,
    x: np.ndarray,
    c: float = 0.0,
    p: int = 4,
    nbins: Optional[Any] = None,
    binselect: str = "esmv",
    scale: Optional[Any] = None,
    kernel: str = "uni",
    h: Optional[Any] = None,
    weights: Optional[np.ndarray] = None,
    covs: Optional[np.ndarray] = None,
    ci: float = 95.0,
    masspoints: str = "adjust",
    cluster: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Binned means and global polynomial fits of an RD plot.

    Parameters
    ----------
    y, x : array_like
        Outcome and running variable. Rows with a missing value in ``y``,
        ``x``, ``weights``, ``covs`` or ``cluster`` are dropped.
    c : float, default 0.0
        Cutoff; must lie strictly inside the range of ``x``.
    p : int, default 4
        Order of the global polynomial fitted on each side.
    nbins : int or (int, int), optional
        Number of bins left and right of the cutoff. When given, ``scale``
        is ignored and ``binselect`` only decides the spacing (``es*``
        evenly, ``qs*`` by quantiles); the selected numbers are still
        reported in ``J_IMSE`` / ``J_MV``.
    binselect : str, default 'esmv'
        ``'es'``, ``'espr'``, ``'esmv'``, ``'esmvpr'``, ``'qs'``,
        ``'qspr'``, ``'qsmv'`` or ``'qsmvpr'``: evenly (``es``) or quantile
        (``qs``) spaced bins; IMSE-optimal or mimicking-variance (``mv``)
        number; constants by spacings or polynomial regression (``pr``).
    scale : float or (float, float), optional
        Factor applied to the selected number of bins on each side; the
        number used is ``ceil(scale * J)`` (equal to ``scale * J`` for the
        integer factors the reference accepts).
    kernel : str, default 'uni'
        Kernel of the global fits: uniform, triangular or Epanechnikov.
    h : float or (float, float), optional
        Support of the global fits on each side, ``[c - h_l, c)`` and
        ``[c, c + h_r]``; defaults to the range of ``x`` on that side.
    weights : array_like, optional
        Observation weights multiplying the kernel in the global fits.
    covs : array_like, optional
        Covariates (``n`` or ``n x d``) entering the global fits linearly,
        with one coefficient vector for both sides, evaluated at their
        sample means. Columns collinear with the polynomial or with earlier
        covariates are dropped with a warning.
    ci : float, default 95.0
        Level (in percent) of the per-bin t intervals.
    masspoints : {'adjust', 'check', 'off'}, default 'adjust'
        With ``'adjust'``, a side whose values are less than 80% distinct
        switches a spacings rule to its polynomial-regression version.
    cluster : array_like, optional
        Cluster labels; the per-bin standard errors become cluster-robust
        (``sqrt(G/(G-1) sum_g s_g^2) / N`` with ``t(G - 1)`` intervals).

    Returns
    -------
    dict
        ``coef`` ((p+1) x 2 array, columns left / right, powers of
        ``x - c``), ``coef_covs`` (array of the kept covariates or None),
        ``vars_bins`` (dict of arrays, one entry per non-empty bin, left to
        right; a single-observation bin has standard error 0), ``vars_poly``
        (dict with ``rdplot_x`` / ``rdplot_y``, 500 points per side),
        ``J``, ``J_IMSE``, ``J_MV`` (tuples left / right), ``binselect``
        (the rule actually used), ``mass_points_adjusted``, ``p``, ``c``,
        ``h``, ``N`` and ``N_h``.

    Examples
    --------
    >>> import numpy as np
    >>> from statspai.rd._rdplot_numbers import rdplot_numbers
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(-1, 1, 400)
    >>> y = x + (x >= 0) + rng.normal(0, 0.3, 400)
    >>> out = rdplot_numbers(y, x, nbins=(5, 5))
    >>> out["J"], int(out["vars_bins"]["rdplot_N"].sum())
    ((5, 5), 400)

    References
    ----------
    calonico2015optimal
    """
    if binselect not in _BINSELECT:
        raise MethodIncompatibility(
            f"binselect must be one of {_BINSELECT}; got {binselect!r}",
            recovery_hint="Use 'esmv' (the default) or another listed rule.",
            diagnostics={"binselect": binselect},
        )
    if masspoints not in ("adjust", "check", "off"):
        raise MethodIncompatibility(
            f"masspoints must be 'adjust', 'check' or 'off'; got {masspoints!r}",
            recovery_hint="Use masspoints='adjust' (the default).",
            diagnostics={"masspoints": masspoints},
        )
    kname = _kernel_name(kernel)
    if isinstance(p, bool) or int(p) != p or p < 0:
        raise MethodIncompatibility(
            f"p must be a non-negative integer; got {p!r}",
            recovery_hint="Pass p=4 (the default) or another order >= 0.",
            diagnostics={"p": p},
        )
    p = int(p)
    if ci is None or not np.isfinite(ci) or not 0 < float(ci) < 100:
        raise MethodIncompatibility(
            f"ci must be a level in percent, strictly between 0 and 100; got {ci!r}",
            recovery_hint="Pass ci=95 for 95% intervals.",
            diagnostics={"ci": ci},
        )

    yv = np.asarray(y, dtype=float).ravel()
    xv = np.asarray(x, dtype=float).ravel()
    n_in = xv.size
    wv = None if weights is None else np.asarray(weights, dtype=float).ravel()
    Z = None
    if covs is not None:
        Z = np.asarray(covs, dtype=float)
        Z = Z.reshape(-1, 1) if Z.ndim == 1 else Z
    cl = None if cluster is None else np.asarray(cluster, dtype=object).ravel()
    for name, arr in (
        ("y", yv),
        ("weights", wv),
        ("covs", Z),
        ("cluster", cl),
    ):
        if arr is not None and arr.shape[0] != n_in:
            raise MethodIncompatibility(
                f"{name} has {arr.shape[0]} rows but x has {n_in}.",
                recovery_hint=f"Pass {name} aligned with x, one row per observation.",
                diagnostics={"n_x": n_in, f"n_{name}": int(arr.shape[0])},
            )
    ok = np.isfinite(yv) & np.isfinite(xv)
    if wv is not None:
        ok &= np.isfinite(wv)
    if Z is not None:
        ok &= np.all(np.isfinite(Z), axis=1)
    if cl is not None:
        ok &= np.array([not _is_missing(v) for v in cl], dtype=bool)
    yv, xv = yv[ok], xv[ok]
    wv = None if wv is None else wv[ok]
    Z = None if Z is None else Z[ok]
    cl = None if cl is None else cl[ok]
    if wv is not None and np.any(wv < 0):
        raise MethodIncompatibility(
            "weights must be non-negative.",
            recovery_hint="Drop or fix the rows with negative weights.",
            diagnostics={"n_negative": int(np.sum(wv < 0))},
        )

    n = xv.size
    if n < _MIN_N:
        raise DataInsufficient(
            "Not enough observations to perform bin calculations",
            recovery_hint=f"rdplot needs at least {_MIN_N} complete observations.",
            diagnostics={"n": int(n), "minimum": _MIN_N},
        )
    x_min, x_max = float(np.min(xv)), float(np.max(xv))
    c = float(c)
    if not x_min < c < x_max:
        raise MethodIncompatibility(
            "c should be set within the range of x",
            recovery_hint="Pick a cutoff strictly between min(x) and max(x).",
            diagnostics={"c": c, "x_min": x_min, "x_max": x_max},
        )
    left = xv < c
    N_l, N_r = int(left.sum()), int((~left).sum())
    if min(N_l, N_r) < 2:
        raise DataInsufficient(
            "Not enough observations on one side of the cutoff to form bins",
            recovery_hint="Each side needs at least two observations.",
            diagnostics={"N_left": N_l, "N_right": N_r},
        )

    h_l, h_r = (c - x_min, x_max - c) if h is None else _positive_pair(h, "h")
    scale_l, scale_r = (1.0, 1.0) if scale is None else _positive_pair(scale, "scale")
    manual = None if nbins is None else _positive_pair(nbins, "nbins", integer=True)

    # ---- mass points --------------------------------------------------- #
    method = binselect
    mass = False
    if masspoints in ("adjust", "check"):
        share = [np.unique(xv[m]).size / m.sum() for m in (left, ~left)]
        has_mass = min(share) < _MASS_SHARE
        if has_mass:
            warnings.warn(
                "Mass points detected in the running variable.",
                stacklevel=2,
            )
            if masspoints == "adjust":
                mass = True
                if not method.endswith("pr"):
                    method = method + "pr"

    # ---- number of bins ------------------------------------------------ #
    sides: List[Dict[str, Any]] = []
    for m, length, scl, label in (
        (left, c - x_min, scale_l, "below"),
        (~left, x_max - c, scale_r, "above"),
    ):
        order = np.argsort(xv[m], kind="stable")
        xs, ys = xv[m][order], yv[m][order]
        const = _selector_constants(xs, ys, c, length, n)
        j_imse, j_mv = _side_bins(const, method, n)
        if j_imse is None or j_mv is None or const["var_y"] == 0:
            warnings.warn(
                f"Not enough variability in the outcome variable {label} the threshold",
                stacklevel=2,
            )
            j_imse, j_mv, J = 1, 1, 1
        else:
            J = j_mv if "mv" in method else j_imse
            J = max(int(math.ceil(scl * J)), 1)
        sides.append({"J": J, "J_IMSE": j_imse, "J_MV": j_mv, "const": const})
    if manual is not None:
        for s, J in zip(sides, manual):
            if s["const"]["var_y"] != 0:
                s["J"] = J
    J_l, J_r = sides[0]["J"], sides[1]["J"]

    # ---- partitions and per-bin statistics ------------------------------ #
    xl, yl = xv[left], yv[left]
    xr, yr = xv[~left], yv[~left]
    if method.startswith("es"):
        e_l = _even_edges(x_min, c, J_l)
        e_r = _even_edges(c, x_max, J_r)
    else:
        e_l = _quantile_edges(xl, J_l)
        e_r = _quantile_edges(xr, J_r)
    level = float(ci)
    tb_l = _bin_table(xl, yl, e_l, level, None if cl is None else cl[left])
    tb_r = _bin_table(xr, yr, e_r, level, None if cl is None else cl[~left])
    vars_bins = {k: np.concatenate([tb_l[k], tb_r[k]]) for k in tb_l}

    # ---- global polynomial ---------------------------------------------- #
    coef, coef_covs, poly_y, N_h = _global_fit(yv, xv, c, p, h_l, h_r, kname, wv, Z)
    grid_l = np.linspace(c - h_l, c, _GRID)
    grid_r = np.linspace(c, c + h_r, _GRID)
    shift_l, shift_r = poly_y
    vars_poly = {
        "rdplot_x": np.concatenate([grid_l, grid_r]),
        "rdplot_y": np.concatenate(
            [
                _powers(grid_l - c, p) @ coef[:, 0] + shift_l,
                _powers(grid_r - c, p) @ coef[:, 1] + shift_r,
            ]
        ),
    }

    return {
        "coef": coef,
        "coef_covs": coef_covs,
        "vars_bins": vars_bins,
        "vars_poly": vars_poly,
        "J": (int(J_l), int(J_r)),
        "J_IMSE": (int(sides[0]["J_IMSE"]), int(sides[1]["J_IMSE"])),
        "J_MV": (int(sides[0]["J_MV"]), int(sides[1]["J_MV"])),
        "binselect": method,
        "mass_points_adjusted": bool(mass),
        "p": p,
        "c": c,
        "h": (float(h_l), float(h_r)),
        "N": (N_l, N_r),
        "N_h": N_h,
    }


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    try:
        return bool(np.isnan(v))
    except (TypeError, ValueError):
        return False


def _global_fit(
    yv: np.ndarray,
    xv: np.ndarray,
    c: float,
    p: int,
    h_l: float,
    h_r: float,
    kernel: str,
    wv: Optional[np.ndarray],
    Z: Optional[np.ndarray],
) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple[float, float], Tuple[int, int]]:
    """Kernel-weighted order-p fits on each side (optionally with covariates).

    Returns the (p+1) x 2 coefficient matrix on powers of ``x - c``, the
    covariate coefficients, the covariate contribution added to each
    side's curve, and the number of observations inside each window.
    """
    left = xv < c
    kern = np.where(
        left,
        kernel_weights(xv, c, h_l, kernel),
        kernel_weights(xv, c, h_r, kernel),
    )
    # observations with a positive kernel weight form the window; with the
    # default ``h`` the extreme observation sits at ``|u| = 1`` exactly
    use_l = left & (kern > 0)
    use_r = (~left) & (kern > 0)
    N_h = (int(use_l.sum()), int(use_r.sum()))
    kw = kern if wv is None else kern * wv
    coef = np.full((p + 1, 2), np.nan)
    if Z is None:
        for j, m in enumerate((use_l, use_r)):
            if m.any():
                coef[:, j] = _lsq(_powers(xv[m] - c, p), yv[m], kw[m])
        return coef, None, (0.0, 0.0), N_h

    use = use_l | use_r
    P = _powers(xv[use] - c, p)
    L = left[use][:, None].astype(float)
    base = np.hstack([P * L, P * (1.0 - L)])
    keep = _independent_columns(base, Z[use], kw[use])
    if len(keep) < Z.shape[1]:
        dropped = [j for j in range(Z.shape[1]) if j not in keep]
        warnings.warn(
            f"rdplot: covariate column(s) {dropped} are collinear with the "
            "polynomial or with earlier covariates and were dropped; "
            "coef_covs holds the remaining columns.",
            stacklevel=3,
        )
    Zk = Z[:, keep]
    X = np.hstack([base, Zk[use]])
    beta = _lsq(X, yv[use], kw[use])
    coef[:, 0] = beta[: p + 1]
    coef[:, 1] = beta[p + 1 : 2 * (p + 1)]
    gamma = beta[2 * (p + 1) :]
    # covariates evaluated at their means over the whole sample (black box)
    shift = float(Zk.mean(axis=0) @ gamma) if gamma.size else 0.0
    return coef, gamma, (shift, shift), N_h


def _independent_columns(base: np.ndarray, Z: np.ndarray, w: np.ndarray) -> List[int]:
    """Indices of covariate columns kept, first of each dependent set first."""
    sw = np.sqrt(w)[:, None]
    current = base * sw
    keep: List[int] = []
    for j in range(Z.shape[1]):
        col = Z[:, j] * sw[:, 0]
        norm = float(np.linalg.norm(col))
        if norm == 0.0:
            continue
        resid = col - current @ np.linalg.lstsq(current, col, rcond=None)[0]
        if float(np.linalg.norm(resid)) > 1e-8 * norm:
            keep.append(j)
            current = np.hstack([current, col[:, None]])
    return keep
