"""Calonico-Cattaneo-Titiunik MSE/CER bandwidth selection.

A faithful port of ``rdrobust::rdbwselect`` (R package rdrobust 4.0.0), which
supersedes the single-step rule-of-thumb previously used in
``statspai.rd.bandwidth``.

Why the old formula could not work
----------------------------------
It computed one closed form,

    h = (C_K * sigma^2 / (n * f_c * m''^2)) ** (1/5)

whose exponent ``1/5`` equals CCT's ``1/(2p+3)`` **only when p == 1**, and
which produced no separate bias bandwidth ``b`` at all.  Measured against R on
``rdrobust_RDsenate``: ``h`` came out 2.8-4.8x too narrow, identical for p=1
and p=2, and ``b == h`` in every one of 36 specifications -- driving the
headline effect to 12.39 where R reports 7.41.

The real procedure is a **three-stage cascade**, each stage a call to the same
V/B/R kernel at a different polynomial order, each feeding the next stage's
bias window:

    stage 1   d_bw = _vbr(o=q+1, nu=q+1, o_B=q+2, h_V=c_bw, h_B=range)
    stage 2   b_bw = _vbr(o=q,   nu=p+1, o_B=q+1, h_V=c_bw, h_B=d_bw)
    stage 3   h_bw = _vbr(o=p,   nu=deriv, o_B=q, h_V=c_bw, h_B=b_bw)

with ``value = (V / (B**2 + scale * R)) ** (1 / (2*o + 3))`` at each stage.

Three defaults in the R implementation are easy to miss and each one silently
shifts the answer; all three are reproduced here and were confirmed by an
R-side replication that matches ``rdbwselect`` to 0.000e+00 (see
``tests/reference_parity/_fixtures/_verify_rdbwselect_cascade.R``):

1. ``stdvars = FALSE`` is the default -- the running variable is **not**
   standardised, and ``BWp = min(sd(x), IQR/1.349)`` uses the raw scale.
2. ``masspoints = "adjust"`` is the default -- the reference bandwidth uses the
   count of **unique** running-variable values, not ``n``; and a ``bwcheck=10``
   floor engages once either side is >=20% tied.
3. **Stage 1 passes ``scale = 0``**, stages 2 and 3 pass ``scaleregul``.
   Inside the kernel ``R = scale * 2 * (o + 1 - nu) * BWreg``, so stage 1's
   regularisation is switched off entirely.  Passing ``scaleregul`` to all
   three stages leaves ``h`` about 11% low (17.754 -> 16.026).

References
----------
calonico2014robust, calonico2019regression
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = ["cct_bandwidth", "BW_SELECTORS"]

# R's six MSE/CER variants, plus the two combination forms.
BW_SELECTORS = (
    "mserd",
    "msetwo",
    "msesum",
    "msecomb1",
    "msecomb2",
    "cerrd",
    "certwo",
    "cersum",
    "cercomb1",
    "cercomb2",
)

# rdbwselect's kernel constants for the reference bandwidth c_bw.
_C_C = {"triangular": 2.576, "uniform": 1.843, "epanechnikov": 2.34}


def _kweight(x: np.ndarray, c: float, h: float, kernel: str) -> np.ndarray:
    """``rdrobust_kweight``: kernel weights, divided by ``h``."""
    u = (x - c) / h
    inside = np.abs(u) <= 1
    if kernel == "epanechnikov":
        return 0.75 * (1 - u**2) * inside / h
    if kernel == "uniform":
        return 0.5 * inside / h
    return (1 - np.abs(u)) * inside / h


def _vander(u: np.ndarray, p: int) -> np.ndarray:
    """``.rdrobust_vander``: columns ``u**0 .. u**p`` (no factorials)."""
    if p < 1:
        return np.ones((len(u), 1))
    out = np.ones((len(u), p + 1))
    for j in range(1, p + 1):
        out[:, j] = out[:, j - 1] * u
    return out


def _qr_xx_inv(x: np.ndarray) -> np.ndarray:
    """``qrXXinv``: ``(x'x)^-1`` via Cholesky, pseudo-inverse on failure."""
    g = x.T @ x
    try:
        L = np.linalg.cholesky(g)
        Li = np.linalg.inv(L)
        return Li.T @ Li
    except np.linalg.LinAlgError:
        return np.linalg.pinv(g)


def _nn_residuals(
    x: np.ndarray,
    y: np.ndarray,
    dups: np.ndarray,
    dupsid: np.ndarray,
    matches: int,
) -> np.ndarray:
    """``rdrobust_res`` with ``vce='nn'``: nearest-neighbour residuals.

    ``x`` must be sorted ascending; ``dups[i]`` is the run length of ties at
    ``i`` and ``dupsid[i]`` the 1-based position within that run.  Ties are
    consumed whole, which is why the two arrays are needed rather than a plain
    k-NN search.
    """
    n = len(y)
    res = np.empty(n)
    limit = min(matches, n - 1)
    for pos in range(n):
        rpos = dups[pos] - dupsid[pos]
        lpos = dupsid[pos] - 1
        while lpos + rpos < limit:
            if pos - lpos - 1 < 0:
                rpos += dups[pos + rpos + 1]
            elif pos + rpos + 1 > n - 1:
                lpos += dups[pos - lpos - 1]
            else:
                dl = x[pos] - x[pos - lpos - 1]
                dr = x[pos + rpos + 1] - x[pos]
                if dl > dr:
                    rpos += dups[pos + rpos + 1]
                elif dl < dr:
                    lpos += dups[pos - lpos - 1]
                else:
                    rpos += dups[pos + rpos + 1]
                    lpos += dups[pos - lpos - 1]
        lo = max(0, pos - lpos)
        hi = min(n - 1, pos + rpos)
        idx = np.arange(lo, hi + 1)
        y_j = y[idx].sum() - y[pos]
        ji = len(idx) - 1
        res[pos] = np.sqrt(ji / (ji + 1.0)) * (y[pos] - y_j / ji)
    return res


_VCE_KINDS = frozenset({"nn", "hc0", "hc1", "hc2", "hc3"})


def _hc_scale(
    vce: str,
    R: np.ndarray,
    invG: np.ndarray,
    W: np.ndarray,
    d: int,
    cluster: Optional[np.ndarray],
) -> np.ndarray:
    """The per-observation residual weight for ``hc0``--``hc3``.

    Mirrors ``rdrobust_res``: ``hc1`` is the degrees-of-freedom correction
    and is switched OFF when clusters are present (R defers to the cluster
    factor instead of applying both), ``hc2``/``hc3`` divide by the leverage
    of the weighted design.
    """
    n = R.shape[0]
    if vce == "hc0":
        return np.ones(n)
    if vce == "hc1":
        if cluster is not None or n <= d:
            return np.ones(n)
        return np.full(n, np.sqrt(n / (n - d)))
    hii = np.einsum("ij,ij->i", R @ invG, R * W[:, None])
    denom = np.maximum(1.0 - hii, 1e-08)
    return np.sqrt(1.0 / denom) if vce == "hc2" else 1.0 / denom


def _vce_meat(
    RX: np.ndarray,
    res: np.ndarray,
    cluster: Optional[np.ndarray],
    k_df: Optional[int] = None,
) -> np.ndarray:
    """``rdrobust_vce``'s meat matrix, heteroskedastic or clustered.

    Without clusters this is ``crossprod(res * RX)``. With them the scores
    are summed within cluster *before* the outer product, and R's
    finite-sample factor ``((n-1)/(n-k)) * (g/(g-1))`` is applied.

    ``k_df`` overrides the ``k`` in that factor. The bias-corrected variance
    is the one place it differs: the operator ``Q_q`` has ``p+1`` columns but
    R charges it ``q+1`` degrees of freedom (its ``k_override``), because the
    correction was estimated from a ``q``-order fit. Worth ~6e-4 on the SE.
    """
    if cluster is None:
        S = res[:, None] * RX
        return S.T @ S
    n, k = RX.shape
    if k_df is None:
        k_df = k
    codes = pd.factorize(cluster, sort=True)[0]
    g = int(codes.max()) + 1 if n else 0
    if g < 2:
        raise ValueError(
            f"cluster-robust variance needs at least 2 clusters on each "
            f"side of the cutoff, found {g} within the bandwidth"
        )
    # scores[j] = sum over cluster j of res_i * RX_i  -> (g x k)
    scores = np.zeros((g, k))
    np.add.at(scores, codes, res[:, None] * RX)
    factor = ((n - 1) / (n - k_df)) * (g / (g - 1.0)) if n > k_df else g / (g - 1.0)
    return factor * (scores.T @ scores)


def _runs(x_sorted: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """R's ``rle``-derived ``dups`` / ``dupsid`` for a sorted vector."""
    n = len(x_sorted)
    if n == 0:
        return np.zeros(0, int), np.zeros(0, int)
    change = np.empty(n, bool)
    change[0] = True
    change[1:] = x_sorted[1:] != x_sorted[:-1]
    starts = np.flatnonzero(change)
    lengths = np.diff(np.append(starts, n))
    dups = np.repeat(lengths, lengths)
    dupsid = np.arange(n) - np.repeat(starts, lengths) + 1
    return dups.astype(int), dupsid.astype(int)


def _vbr(
    y: np.ndarray,
    x: np.ndarray,
    c: float,
    o: int,
    nu: int,
    o_B: int,
    h_V: float,
    h_B: float,
    scale: float,
    kernel: str,
    dups: np.ndarray,
    dupsid: np.ndarray,
    nnmatch: int,
    Z: Optional[np.ndarray] = None,
    vce: str = "nn",
    C: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """``rdrobust_bw``: variance ``V``, bias ``B``, regularisation ``R``.

    Sharp RD. ``Z`` supplies covariates, which are partialled out of the
    local polynomial Frisch-Waugh style: the residual maker is the vector
    ``s = [1, -gamma]`` where ``gamma`` solves the covariate block of the
    weighted normal equations after projecting out the polynomial basis.

    Only ``V`` depends on ``vce``/``C`` -- the bias and regularisation
    windows are pure least squares. That is why clustering shifts the
    bandwidth at all: it enters the numerator of ``V / (B^2 + scale*R)``.
    """
    # ---- variance window ------------------------------------------------
    w = _kweight(x, c, h_V, kernel)
    ind = w > 0
    eY, eX, eW = y[ind], x[ind], w[ind]
    R_V = _vander(eX - c, o)
    invG_V = _qr_xx_inv(R_V * np.sqrt(eW)[:, None])
    RX = R_V * eW[:, None]

    # Covariate partialling-out (R's dZ branch).
    s_vec = np.array([1.0])
    D_V = eY[:, None]
    if Z is not None and Z.shape[1] > 0:
        eZ = Z[ind]
        D_V = np.column_stack([eY, eZ])
        U = RX.T @ D_V
        ZWD = (eZ * eW[:, None]).T @ D_V
        colsZ = slice(1, 1 + eZ.shape[1])
        UiGU = U[:, colsZ].T @ (invG_V @ U)
        ZWZ = ZWD[:, colsZ] - UiGU[:, colsZ]
        ZWY = ZWD[:, 0] - UiGU[:, 0]
        gamma = np.linalg.solve(ZWZ, ZWY)
        s_vec = np.concatenate([[1.0], -gamma])

    eC = None if C is None else C[ind]
    if vce == "nn":
        res_V = (
            np.column_stack(
                [
                    _nn_residuals(eX, D_V[:, j], dups[ind], dupsid[ind], nnmatch)
                    for j in range(D_V.shape[1])
                ]
            )
            @ s_vec
        )
    else:
        beta_V = invG_V @ (RX.T @ D_V)
        res_V = (
            _hc_scale(vce, R_V, invG_V, eW, o + 1, eC)[:, None] * (D_V - R_V @ beta_V)
        ) @ s_vec

    V_V = (invG_V @ _vce_meat(RX, res_V, eC) @ invG_V)[nu, nu]

    v = RX.T @ (((eX - c) / h_V) ** (o + 1))
    Hp = h_V ** np.arange(o + 1)
    BConst = (Hp * (invG_V @ v))[nu]

    # ---- bias window ----------------------------------------------------
    w_b = _kweight(x, c, h_B, kernel)
    ind_b = w_b > 0
    eY_b, eX_b, eW_b = y[ind_b], x[ind_b], w_b[ind_b]
    R_B = _vander(eX_b - c, o_B)
    invG_B = _qr_xx_inv(R_B * np.sqrt(eW_b)[:, None])
    if Z is not None and Z.shape[1] > 0:
        D_B = np.column_stack([eY_b, Z[ind_b]])
        beta_B_full = invG_B @ ((R_B * eW_b[:, None]).T @ D_B)
        beta_B = beta_B_full @ s_vec
    else:
        beta_B = invG_B @ ((R_B * eW_b[:, None]).T @ eY_b)

    BWreg = 0.0
    if scale > 0:
        # The bias window needs the same covariate residual maker as the
        # variance window; using the raw Y residuals here leaves h ~39%
        # too narrow on a design where covariates bind. It also carries the
        # same vce/cluster treatment -- the regularisation term is a
        # sandwich variance too, and leaving it at nn while V used hc left
        # h 8e-3 off with V and B both already exact.
        D_Bz = (
            eY_b[:, None]
            if Z is None or Z.shape[1] == 0
            else np.column_stack([eY_b, Z[ind_b]])
        )
        eC_b = None if C is None else C[ind_b]
        if vce == "nn":
            res_B = (
                np.column_stack(
                    [
                        _nn_residuals(
                            eX_b, D_Bz[:, j], dups[ind_b], dupsid[ind_b], nnmatch
                        )
                        for j in range(D_Bz.shape[1])
                    ]
                )
                @ s_vec
            )
        else:
            beta_B_all = invG_B @ ((R_B * eW_b[:, None]).T @ D_Bz)
            res_B = (
                _hc_scale(vce, R_B, invG_B, eW_b, o_B + 1, eC_b)[:, None]
                * (D_Bz - R_B @ beta_B_all)
            ) @ s_vec
        RX_B = R_B * eW_b[:, None]
        V_B = (invG_B @ _vce_meat(RX_B, res_B, eC_b) @ invG_B)[o + 1, o + 1]
        BWreg = 3.0 * BConst**2 * V_B

    B = np.sqrt(2 * (o + 1 - nu)) * BConst * beta_B[o + 1]
    V = (2 * nu + 1) * h_V ** (2 * nu + 1) * V_V
    R = scale * (2 * (o + 1 - nu)) * BWreg
    return {"V": float(V), "B": float(B), "R": float(R), "rate": 1.0 / (2 * o + 3)}


def _combine(v_l: Dict, v_r: Dict, form: str, scale: float) -> float:
    """Assemble one side's or both sides' V/B/R into a bandwidth."""
    if form == "two_left":
        return float((v_l["V"] / (v_l["B"] ** 2 + scale * v_l["R"])) ** v_l["rate"])
    if form == "two_right":
        return float((v_r["V"] / (v_r["B"] ** 2 + scale * v_r["R"])) ** v_r["rate"])
    num = v_l["V"] + v_r["V"]
    bias = (v_r["B"] + v_l["B"]) if form == "sum" else (v_r["B"] - v_l["B"])
    den = bias**2 + scale * (v_r["R"] + v_l["R"])
    return float((num / den) ** v_l["rate"])


def cct_bandwidth(
    y: np.ndarray,
    x: np.ndarray,
    c: float = 0.0,
    p: int = 1,
    q: Optional[int] = None,
    deriv: int = 0,
    kernel: str = "triangular",
    bwselect: str = "mserd",
    scaleregul: float = 1.0,
    nnmatch: int = 3,
    masspoints: str = "adjust",
    bwrestrict: bool = True,
    covs: Optional[np.ndarray] = None,
    vce: str = "nn",
    cluster: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """MSE/CER-optimal bandwidths, matching ``rdrobust::rdbwselect``.

    ``vce`` and ``cluster`` are not cosmetic here. The cascade's ``V`` term
    is a sandwich variance, so clustering widens or narrows every stage --
    the bandwidth is not separable from the variance estimator. R makes an
    additional substitution that is easy to miss: supplying ``cluster`` with
    ``vce='nn'`` (the default) silently promotes it to ``'cr1'``, whose
    residuals are ``hc1``, **not** nearest-neighbour. Reproduced below.

    Returns
    -------
    dict
        ``h_left``, ``h_right``, ``b_left``, ``b_right``.  The ``*rd`` and
        ``*sum`` variants return a common bandwidth on both sides; ``*two``
        returns side-specific ones.

    Examples
    --------
    >>> import numpy as np
    >>> from statspai.rd._cct_bandwidth import cct_bandwidth
    >>> rng = np.random.default_rng(0)
    >>> n = 2000
    >>> x = rng.uniform(-1, 1, n)
    >>> y = 0.5 * x + 1.0 * (x >= 0) + rng.normal(0, 0.5, n)
    >>> bw = cct_bandwidth(y, x, c=0.0, p=1)
    >>> sorted(bw)
    ['b_left', 'b_right', 'h_left', 'h_right']
    >>> bool(bw["b_left"] > bw["h_left"] > 0)  # bias window is the wider one
    True

    The bandwidth responds to the polynomial order, which is the property the
    superseded rule of thumb lacked:

    >>> h1 = cct_bandwidth(y, x, c=0.0, p=1)["h_left"]
    >>> h2 = cct_bandwidth(y, x, c=0.0, p=2)["h_left"]
    >>> bool(abs(h1 - h2) > 1e-6)
    True
    """
    kernel = {"tri": "triangular", "uni": "uniform", "epa": "epanechnikov"}.get(
        kernel, kernel
    )
    if kernel not in _C_C:
        raise ValueError(
            f"kernel must be 'triangular', 'uniform' or 'epanechnikov', "
            f"got {kernel!r}"
        )
    if bwselect not in BW_SELECTORS:
        raise ValueError(f"bwselect must be one of {BW_SELECTORS}, got {bwselect!r}")
    if q is None:
        q = p + 1

    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    Z = None if covs is None else np.asarray(covs, dtype=float)
    if Z is not None and Z.ndim == 1:
        Z = Z[:, None]
    Cc = None if cluster is None else np.asarray(cluster)
    vce_eff = str(vce).lower()
    if vce_eff not in _VCE_KINDS:
        raise ValueError(f"vce must be one of {sorted(_VCE_KINDS)}, got {vce_eff!r}")
    if Cc is not None and vce_eff in ("nn", "hc0", "hc1"):
        # R promotes cluster + nn/hc0/hc1 to cr1, whose residuals are hc1's.
        vce_eff = "hc1"
    ok = np.isfinite(y) & np.isfinite(x)
    if Z is not None:
        ok &= np.isfinite(Z).all(axis=1)
    y, x = y[ok], x[ok]
    if Z is not None:
        Z = Z[ok]
    if Cc is not None:
        Cc = Cc[ok]
    n = len(x)

    # Reference bandwidth. stdvars=FALSE => raw scale; masspoints='adjust'
    # => the unique-value count replaces n.
    x_iq = np.quantile(x, 0.75, method="lower") - np.quantile(x, 0.25, method="lower")
    BWp = min(np.std(x, ddof=1), x_iq / 1.349)
    C_c = _C_C[kernel]

    left, right = x < c, x >= c
    M_l = len(np.unique(x[left]))
    M_r = len(np.unique(x[right]))
    M = M_l + M_r
    c_bw = C_c * BWp * (M if masspoints == "adjust" else n) ** (-0.2)

    x_min, x_max = x.min(), x.max()
    if bwrestrict:
        c_bw = min(c_bw, max(abs(c - x_min), abs(c - x_max)))
    # bwcheck floor: engages when either side is heavily tied.
    if masspoints == "adjust" and (
        1 - M_l / max(left.sum(), 1) >= 0.2 or 1 - M_r / max(right.sum(), 1) >= 0.2
    ):
        xu_l = np.sort(np.unique(x[left]))[::-1]
        xu_r = np.unique(x[right])
        k_l, k_r = min(10, M_l), min(10, M_r)
        c_bw = max(
            c_bw,
            abs(xu_l - c)[k_l - 1] + 1e-8,
            abs(xu_r - c)[k_r - 1] + 1e-8,
        )

    # Side data, sorted (the nn residuals require it).
    ol = np.argsort(x[left], kind="mergesort")
    orr = np.argsort(x[right], kind="mergesort")
    Xl, Yl = x[left][ol], y[left][ol]
    Xr, Yr = x[right][orr], y[right][orr]
    Zl = None if Z is None else Z[left][ol]
    Zr = None if Z is None else Z[right][orr]
    Cl = None if Cc is None else Cc[left][ol]
    Cr = None if Cc is None else Cc[right][orr]
    dl, dil = _runs(Xl)
    dr, dir_ = _runs(Xr)
    range_l, range_r = abs(c - x_min), abs(c - x_max)

    def bw(Y, X, o, nu, o_B, h_B, scale, dups, dupsid, Zs=None, Cs=None):
        return _vbr(
            Y,
            X,
            c,
            o,
            nu,
            o_B,
            c_bw,
            h_B,
            scale,
            kernel,
            dups,
            dupsid,
            nnmatch,
            Zs,
            vce_eff,
            Cs,
        )

    two = bwselect in ("msetwo", "certwo")
    form = "sum" if bwselect in ("msesum", "cersum") else "diff"

    # bwrestrict clamps EVERY stage's output, not just c_bw: an intermediate
    # bandwidth wider than the data range would otherwise feed the next stage
    # a window that reaches past the support. Missing this leaves msesum/p=2/
    # epanechnikov 1.1% off in h and 3.3% in b, while the other 34 cells match.
    bw_max_l, bw_max_r = abs(c - x_min), abs(c - x_max)
    bw_max = max(bw_max_l, bw_max_r)
    bw_min_l = bw_min_r = None
    if masspoints == "adjust" and (
        1 - M_l / max(left.sum(), 1) >= 0.2 or 1 - M_r / max(right.sum(), 1) >= 0.2
    ):
        xu_l = np.sort(np.unique(x[left]))[::-1]
        xu_r = np.unique(x[right])
        bw_min_l = abs(xu_l - c)[min(10, M_l) - 1] + 1e-8
        bw_min_r = abs(xu_r - c)[min(10, M_r) - 1] + 1e-8

    def clamp(v, side=None):
        if bwrestrict:
            v = min(v, {"l": bw_max_l, "r": bw_max_r}.get(side, bw_max))
        if bw_min_l is not None:
            v = max(
                v,
                (
                    bw_min_l
                    if side == "l"
                    else bw_min_r
                    if side == "r"
                    else max(bw_min_l, bw_min_r)
                ),
            )
        return v

    # stage 1 -- note scale = 0 here, not scaleregul.
    D_l = bw(Yl, Xl, q + 1, q + 1, q + 2, range_l, 0.0, dl, dil, Zl, Cl)
    D_r = bw(Yr, Xr, q + 1, q + 1, q + 2, range_r, 0.0, dr, dir_, Zr, Cr)
    if two:
        d_l = clamp(_combine(D_l, D_r, "two_left", scaleregul), "l")
        d_r = clamp(_combine(D_l, D_r, "two_right", scaleregul), "r")
    else:
        d_l = d_r = clamp(_combine(D_l, D_r, form, scaleregul))

    # stage 2 -- bias bandwidth b.
    B_l = bw(Yl, Xl, q, p + 1, q + 1, d_l, scaleregul, dl, dil, Zl, Cl)
    B_r = bw(Yr, Xr, q, p + 1, q + 1, d_r, scaleregul, dr, dir_, Zr, Cr)
    if two:
        b_l = clamp(_combine(B_l, B_r, "two_left", scaleregul), "l")
        b_r = clamp(_combine(B_l, B_r, "two_right", scaleregul), "r")
    else:
        b_l = b_r = clamp(_combine(B_l, B_r, form, scaleregul))

    # stage 3 -- main bandwidth h.
    H_l = bw(Yl, Xl, p, deriv, q, b_l, scaleregul, dl, dil, Zl, Cl)
    H_r = bw(Yr, Xr, p, deriv, q, b_r, scaleregul, dr, dir_, Zr, Cr)
    if two:
        h_l = clamp(_combine(H_l, H_r, "two_left", scaleregul), "l")
        h_r = clamp(_combine(H_l, H_r, "two_right", scaleregul), "r")
    else:
        h_l = h_r = clamp(_combine(H_l, H_r, form, scaleregul))

    # CER variants shrink the MSE bandwidth by n^{-eps}.
    if bwselect.startswith("cer"):
        cer = n ** (-(p / ((3 + p) * (3 + 2 * p))))
        h_l, h_r = h_l * cer, h_r * cer

    return {"h_left": h_l, "h_right": h_r, "b_left": b_l, "b_right": b_r}


# ══════════════════════════════════════════════════════════════════════
#  CCT bias correction (defect E)
# ══════════════════════════════════════════════════════════════════════


def cct_bias_corrected(
    y: np.ndarray,
    x: np.ndarray,
    c: float,
    h_l: float,
    h_r: float,
    b_l: float,
    b_r: float,
    p: int,
    q: int,
    deriv: int,
    kernel: str,
    nnmatch: int = 3,
    covs: Optional[np.ndarray] = None,
    vce: str = "nn",
    cluster: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float]:
    """CCT bias-corrected estimate and its SEs, matching ``rdrobust``.

    Returns ``(tau_conventional, tau_bias_corrected, se_conventional,
    se_robust)``. Sharp RD; ``covs``, ``vce`` and ``cluster`` are supported,
    ``fuzzy`` is not.

    ``vce`` selects the residual construction, following ``rdrobust_res``:
    ``'nn'`` uses nearest-neighbour residuals (no fitted values, so it needs
    no bandwidth-specific refit), while ``'hc0'``–``'hc3'`` use ordinary
    regression residuals with the usual leverage corrections. The two are not
    interchangeable — under ``hc*`` the robust variance uses residuals from
    the ``q``-order fit on ``b``, not the ``p``-order fit on ``h``, which is
    why ``res_b`` is computed separately below.

    ``cluster`` switches the meat matrix from ``sum_i r_i^2 X_i X_i'`` to
    ``sum_g (X_g' r_g)(X_g' r_g)'`` with R's finite-sample factor
    ``((n-1)/(n-k)) * (g/(g-1))``.

    Examples
    --------
    >>> import numpy as np
    >>> from statspai.rd._cct_bandwidth import (
    ...     cct_bandwidth, cct_bias_corrected)
    >>> rng = np.random.default_rng(0)
    >>> n = 2000
    >>> x = rng.uniform(-1, 1, n)
    >>> y = 0.5 * x + 1.0 * (x >= 0) + rng.normal(0, 0.5, n)
    >>> bw = cct_bandwidth(y, x, c=0.0, p=1)
    >>> tau, tau_bc, se, se_rb = cct_bias_corrected(
    ...     y, x, 0.0, bw["h_left"], bw["h_right"],
    ...     bw["b_left"], bw["b_right"], p=1, q=2, deriv=0,
    ...     kernel="triangular")
    >>> bool(abs(tau - 1.0) < 3 * se)  # true jump is 1.0, within 3 SE
    True
    >>> bool(se > 0 and se_rb > 0)
    True

    The bias-corrected estimator is **not** a ``q``-order refit on bandwidth
    ``b``.  It is the ``p``-order fit on ``h`` with the design weights replaced
    by a corrected operator that subtracts an estimated ``(p+1)``-order
    curvature term measured on the ``b`` window::

        L      = (R_p' W_h) u^(p+1),      u = (x - c) / h
        Q_q    = R_p' W_h - h^(p+1) * L e_{p+2}' (invG_q R_q') W_b
        beta_bc = invG_p @ (Q_q' D)

    Running a plain ``q``-order regression on ``b`` instead -- which is what
    StatsPAI did through 1.20.x -- estimates a different quantity, and its
    variance is not the robust variance either.
    """
    vce = str(vce).lower()
    if vce not in _VCE_KINDS:
        raise ValueError(f"vce must be one of {sorted(_VCE_KINDS)}, got {vce!r}")
    if cluster is not None and vce in ("nn", "hc0", "hc1"):
        # Same promotion rdrobust applies: cluster + nn/hc0/hc1 -> cr1, whose
        # residuals are hc1's. Keeping nn residuals under a clustered meat
        # matrix understates the SE by ~10x, because nearest-neighbour
        # differencing removes exactly the within-cluster correlation the
        # cluster sum is there to capture.
        vce = "hc1"
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    Zall = None if covs is None else np.asarray(covs, float)
    if Zall is not None and Zall.ndim == 1:
        Zall = Zall[:, None]
    Call = None if cluster is None else np.asarray(cluster)
    sides = []
    for side, h, b in (("l", h_l, b_l), ("r", h_r, b_r)):
        m = (x < c) if side == "l" else (x >= c)
        # Sort the SIDE first: nn residuals need ascending x, and the tie runs
        # must be measured on the whole side and then subset by the window --
        # computing them on the window instead miscounts ties that straddle
        # the boundary, which costs ~4% on both SEs.
        xs, ys = x[m], y[m]
        o = np.argsort(xs, kind="mergesort")
        xs, ys = xs[o], ys[o]
        Zs = None if Zall is None else Zall[m][o]
        Cs = None if Call is None else Call[m][o]
        dups_side, dupsid_side = _runs(xs)
        W_h = _kweight(xs, c, h, kernel)
        W_b = _kweight(xs, c, b, kernel)
        keep = (W_h > 0) | (W_b > 0)
        eX, eY = xs[keep], ys[keep]
        eZ = None if Zs is None else Zs[keep]
        eC = None if Cs is None else Cs[keep]
        W_h, W_b = W_h[keep], W_b[keep]
        e_dups, e_dupsid = dups_side[keep], dupsid_side[keep]

        R_q = _vander(eX - c, q)
        R_p = R_q[:, : p + 1]
        u = (eX - c) / h
        L = (R_p * W_h[:, None]).T @ (u ** (p + 1))
        invG_q = _qr_xx_inv(R_q * np.sqrt(W_b)[:, None])
        invG_p = _qr_xx_inv(R_p * np.sqrt(W_h)[:, None])

        e_p1 = np.zeros(q + 1)
        e_p1[p + 1] = 1.0
        # Q_q = R_p'W_h - h^(p+1) * L e_p1' (invG_q R_q') W_b     (k_p x n)
        M = (invG_q @ R_q.T) * W_b[None, :]
        Q_q = (R_p * W_h[:, None]).T - h ** (p + 1) * np.outer(L, e_p1) @ M

        RXp0 = R_p * W_h[:, None]
        Dz = (
            eY[:, None] if eZ is None or eZ.shape[1] == 0 else np.column_stack([eY, eZ])
        )
        # Covariate partialling-out. Unlike the bandwidth cascade, which
        # solves for gamma one side at a time, rdrobust POOLS the two sides
        # here: ZWZ_p = ZWZ_p_l + ZWZ_p_r, giving a single s_Y shared by both.
        # Solving per side instead leaves the estimate ~1e-3 off with the
        # bandwidth already exact -- a small enough gap to look like noise.
        ZWZ = ZWY = None
        if eZ is not None and eZ.shape[1] > 0:
            U = RXp0.T @ Dz
            ZWD = (eZ * W_h[:, None]).T @ Dz
            colsZ = slice(1, 1 + eZ.shape[1])
            UiGU = U[:, colsZ].T @ (invG_p @ U)
            ZWZ = ZWD[:, colsZ] - UiGU[:, colsZ]
            ZWY = ZWD[:, 0] - UiGU[:, 0]
        sides.append(
            dict(
                h=h,
                b=b,
                eX=eX,
                eZ=eZ,
                eC=eC,
                W_h=W_h,
                W_b=W_b,
                Dz=Dz,
                R_p=R_p,
                R_q=R_q,
                invG_p=invG_p,
                invG_q=invG_q,
                Q_q=Q_q,
                RXp0=RXp0,
                ZWZ=ZWZ,
                ZWY=ZWY,
                e_dups=e_dups,
                e_dupsid=e_dupsid,
            )
        )

    if sides[0]["ZWZ"] is not None:
        gamma = np.linalg.solve(
            sides[0]["ZWZ"] + sides[1]["ZWZ"], sides[0]["ZWY"] + sides[1]["ZWY"]
        )
        s_vec = np.concatenate([[1.0], -gamma])
    else:
        s_vec = np.array([1.0])

    out = []
    for sd in sides:
        h, b = sd["h"], sd["b"]
        eX, eZ, eC, Dz = sd["eX"], sd["eZ"], sd["eC"], sd["Dz"]
        W_h, W_b = sd["W_h"], sd["W_b"]
        R_p, R_q = sd["R_p"], sd["R_q"]
        invG_p, invG_q, Q_q = sd["invG_p"], sd["invG_q"], sd["Q_q"]
        RXp0 = sd["RXp0"]
        e_dups, e_dupsid = sd["e_dups"], sd["e_dupsid"]

        beta_p = (invG_p @ (RXp0.T @ Dz)) @ s_vec
        beta_bc = (invG_p @ (Q_q @ Dz)) @ s_vec

        # Variances. rdrobust_vce with C=NULL is crossprod((res %*% s) * RX);
        # the CONVENTIONAL one uses RX = R_p * W_h, the ROBUST one swaps in
        # the bias-correction operator Q_q. Reusing the conventional RX (or a
        # plain q-order refit) is what left the robust SE ~13% off.
        if vce == "nn":
            res_mat = np.column_stack(
                [
                    _nn_residuals(eX, Dz[:, j], e_dups, e_dupsid, nnmatch)
                    for j in range(Dz.shape[1])
                ]
            )
            # rdrobust sets res_b = res_h under nn: nearest-neighbour
            # residuals do not depend on a fitted value, so there is no
            # b-window refit to do.
            res_h = res_b = res_mat @ s_vec
        else:
            beta_p_full = invG_p @ (RXp0.T @ Dz)
            beta_q_full = invG_q @ ((R_q * W_b[:, None]).T @ Dz)
            res_h = (
                _hc_scale(vce, R_p, invG_p, W_h, p + 1, eC)[:, None]
                * (Dz - R_p @ beta_p_full)
            ) @ s_vec
            res_b = (
                _hc_scale(vce, R_q, invG_q, W_b, q + 1, eC)[:, None]
                * (Dz - R_q @ beta_q_full)
            ) @ s_vec
        RXp = R_p * W_h[:, None]
        V_cl = invG_p @ _vce_meat(RXp, res_h, eC) @ invG_p
        if eC is not None and h == b:
            # R takes a different route when h == b under clustering: the
            # robust variance is the q-order sandwich on the h window rather
            # than the Q_q operator.
            invG_q_h = _qr_xx_inv(R_q * np.sqrt(W_h)[:, None])
            RXq = R_q * W_h[:, None]
            V_rb = invG_q_h @ _vce_meat(RXq, res_b, eC) @ invG_q_h
        else:
            V_rb = invG_p @ _vce_meat(Q_q.T, res_b, eC, q + 1) @ invG_p
        out.append(
            (beta_p[deriv], beta_bc[deriv], V_cl[deriv, deriv], V_rb[deriv, deriv])
        )

    fac = float(math.factorial(deriv)) if deriv else 1.0
    tau_cl = fac * (out[1][0] - out[0][0])
    tau_bc = fac * (out[1][1] - out[0][1])
    se_cl = fac * np.sqrt(out[0][2] + out[1][2])
    se_rb = fac * np.sqrt(out[0][3] + out[1][3])
    return float(tau_cl), float(tau_bc), float(se_cl), float(se_rb)
