"""Smooth terms of ``sp.gam`` beyond the univariate P-spline.

Every term exposes the same small interface, which is all the fitting
engine in ``gam.py`` uses:

``label``            name shown in ``smooth_terms``
``cols``             its slice of the coefficient vector
``k``                number of basis functions before any constraint
``roots``            list of matrices ``R_j`` with penalty ``sum_j lambda_j
                     R_j'R_j``; one smoothing parameter per entry
``term_design(df)``  design columns on a data frame
``partial_frame(g)`` coordinates and design rows for plotting the term

The constructions follow the published descriptions (random effects as a
ridge penalty; tensor products of marginal P-splines, Eilers and Marx;
thin plate regression splines, Wood [@wood2003thin]; the test of a smooth
term, Wood [@wood2013pvalues]) and are checked against
``mgcv::gam`` as a black box at given smoothing parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

from ..exceptions import DataInsufficient, MethodIncompatibility

_DEGREE = 3
_DIFF = 2
#: thin plate splines are built on at most this many distinct values
_TP_MAX_KNOTS = 2000


def _slice() -> slice:
    return slice(0, 0)


def ps_knots(x: np.ndarray, k: int) -> np.ndarray:
    """Evenly spaced cubic B-spline knots for ``k`` basis functions."""
    lo, hi = float(np.min(x)), float(np.max(x))
    span = hi - lo
    lo, hi = lo - 0.001 * span, hi + 0.001 * span
    inner = k - (_DEGREE - 1)
    dx = (hi - lo) / (inner - 1)
    return np.asarray(lo + dx * np.arange(-_DEGREE, inner + _DEGREE), dtype=float)


def ps_basis(knots: np.ndarray, k: int, x: np.ndarray) -> np.ndarray:
    basis = BSpline(knots, np.eye(k), _DEGREE, extrapolate=True)
    return np.asarray(basis(np.asarray(x, dtype=float)), dtype=float)


def _sum_to_zero(raw: np.ndarray) -> np.ndarray:
    """Columns spanning the coefficient vectors whose fitted values sum to
    zero over the sample."""
    c = raw.sum(axis=0)[:, None]
    Qfull, _ = np.linalg.qr(c, mode="complete")
    return np.asarray(Qfull[:, 1:], dtype=float)


# ---------------------------------------------------------------- random effect


@dataclass
class RandomEffect:
    """``s(g, bs="re")``: one coefficient per level of ``g``, shrunk toward
    zero by a ridge penalty. Its smoothing parameter is the ratio of the
    residual variance to the variance of the effects."""

    var: str
    levels: List[str]
    cols: slice = field(default_factory=_slice)
    kind: str = "re"

    @property
    def label(self) -> str:
        return f"s({self.var})"

    @property
    def k(self) -> int:
        return len(self.levels) + 1  # so that k - 1 is what it can use

    @property
    def roots(self) -> List[np.ndarray]:
        return [np.eye(len(self.levels))]

    def term_design(self, data: pd.DataFrame) -> np.ndarray:
        codes = pd.Categorical(data[self.var].astype(str), categories=self.levels).codes
        Z = np.zeros((len(data), len(self.levels)))
        seen = codes >= 0  # a level the fit never saw predicts at zero
        Z[np.flatnonzero(seen), codes[seen]] = 1.0
        return Z

    def partial_frame(
        self, grid: Optional[Sequence[Any]]
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        levels = self.levels if grid is None else [str(g) for g in grid]
        frame = pd.DataFrame({self.var: levels})
        return frame, self.term_design(frame)


def make_random_effect(var: str, values: pd.Series) -> RandomEffect:
    levels = sorted(values.astype(str).unique())
    if len(levels) < 2:
        raise DataInsufficient(
            f"gam: s({var}, bs='re') needs at least two levels.",
            diagnostics={"levels": len(levels)},
        )
    return RandomEffect(var=var, levels=levels)


# ---------------------------------------------------------------- tensor product


@dataclass
class Tensor:
    """``te(x, z)``: a surface built from the products of two marginal
    P-spline bases, with one penalty (and smoothing parameter) for
    wiggliness in each direction.

    Two conventions are mgcv's, so that fits can be compared. Each margin
    is re-expressed so that its coefficients are the values of the
    marginal function at ``k`` points spread evenly over the data (the
    identity that completes each penalty, ``S1 x I`` and ``I x S2``, is
    then an identity between function values, not between B-spline
    coefficients). And each marginal penalty is divided by its largest
    eigenvalue, which puts the two directions on one scale.
    """

    var: str
    var2: str
    k1: int
    k2: int
    knots1: np.ndarray
    knots2: np.ndarray
    map1: np.ndarray  # B-spline coefficients -> function values, inverted
    map2: np.ndarray
    Q: np.ndarray
    root1: np.ndarray
    root2: np.ndarray
    cols: slice = field(default_factory=_slice)
    kind: str = "te"

    @property
    def label(self) -> str:
        return f"te({self.var},{self.var2})"

    @property
    def k(self) -> int:
        return self.k1 * self.k2

    @property
    def roots(self) -> List[np.ndarray]:
        return [self.root1, self.root2]

    def raw(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        B1 = ps_basis(self.knots1, self.k1, x) @ self.map1
        B2 = ps_basis(self.knots2, self.k2, z) @ self.map2
        # row-wise Kronecker product, second margin varying fastest
        return np.asarray(
            (B1[:, :, None] * B2[:, None, :]).reshape(len(B1), self.k1 * self.k2)
        )

    def term_design(self, data: pd.DataFrame) -> np.ndarray:
        x = data[self.var].to_numpy(dtype=float)
        z = data[self.var2].to_numpy(dtype=float)
        return np.asarray(self.raw(x, z) @ self.Q, dtype=float)

    def partial_frame(self, grid: Optional[Any]) -> Tuple[pd.DataFrame, np.ndarray]:
        if grid is None:

            def inside(knots: np.ndarray) -> np.ndarray:
                inner = knots[_DEGREE : len(knots) - _DEGREE]
                pad = 0.001 * (inner[-1] - inner[0]) / 1.002
                return np.linspace(inner[0] + pad, inner[-1] - pad, 25)

            gx, gz = np.meshgrid(
                inside(self.knots1), inside(self.knots2), indexing="ij"
            )
            frame = pd.DataFrame({self.var: gx.ravel(), self.var2: gz.ravel()})
        else:
            frame = pd.DataFrame(grid)[[self.var, self.var2]].astype(float)
        return frame, self.term_design(frame)


def _margin(x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Knots, the map to the function-value parametrisation, and the root
    of the marginal penalty in that parametrisation, scaled so that its
    largest eigenvalue is one."""
    knots = ps_knots(x, k)
    at = np.linspace(float(np.min(x)), float(np.max(x)), k)
    to_values = np.linalg.inv(ps_basis(knots, k, at))
    root = np.diff(np.eye(k), n=_DIFF, axis=0) @ to_values
    top = float(np.linalg.eigvalsh(root.T @ root).max())
    return knots, to_values, root / np.sqrt(top)


def make_tensor(
    var: str, x: np.ndarray, var2: str, z: np.ndarray, k1: int, k2: int
) -> Tensor:
    for name, values, kk in ((var, x, k1), (var2, z, k2)):
        if kk < _DEGREE + 1:
            raise MethodIncompatibility(
                f"gam: te(...) needs k >= {_DEGREE + 1} in each direction.",
                diagnostics={"k": kk},
            )
        distinct = np.unique(values).size
        if distinct < kk:
            raise DataInsufficient(
                f"gam: {name} has {distinct} distinct values, fewer than the "
                f"k = {kk} basis functions of its margin.",
                recovery_hint="Lower k for the tensor product.",
            )
    knots1, map1, r1 = _margin(x, k1)
    knots2, map2, r2 = _margin(z, k2)
    proto = Tensor(
        var=var,
        var2=var2,
        k1=k1,
        k2=k2,
        knots1=knots1,
        knots2=knots2,
        map1=map1,
        map2=map2,
        Q=np.eye(k1 * k2),
        root1=np.kron(r1, np.eye(k2)),
        root2=np.kron(np.eye(k1), r2),
    )
    Q = _sum_to_zero(proto.raw(x, z))
    proto.root1 = proto.root1 @ Q
    proto.root2 = proto.root2 @ Q
    proto.Q = Q
    return proto


# ---------------------------------------------------------------- thin plate


def _eta(r: np.ndarray) -> np.ndarray:
    """Thin plate spline kernel in one dimension, second-derivative
    penalty: ``|r|^3 / 12``."""
    return np.asarray(np.abs(r) ** 3 / 12.0, dtype=float)


@dataclass
class ThinPlate:
    """``s(x, bs="tp")``: the rank-``k`` thin plate regression spline, the
    best low-rank approximation to the full thin plate spline of ``x``
    (no knots to place). mgcv's default basis."""

    var: str
    k: int
    knots: np.ndarray  # the distinct values the kernel is centred on
    UZ: np.ndarray  # maps kernel evaluations to the wiggly part of the basis
    shift: float
    Q: np.ndarray
    root: np.ndarray
    cols: slice = field(default_factory=_slice)
    by: Optional[str] = None
    level: Any = None
    kind: str = "tp"

    @property
    def label(self) -> str:
        return f"s({self.var})"

    @property
    def roots(self) -> List[np.ndarray]:
        return [self.root]

    def raw(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        E = _eta(x[:, None] - self.knots[None, :])
        return np.column_stack([E @ self.UZ, np.ones(len(x)), x - self.shift])

    def term_design(self, data: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.raw(data[self.var].to_numpy(dtype=float)) @ self.Q)

    def partial_frame(
        self, grid: Optional[Sequence[float]]
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        xs = (
            np.linspace(self.knots.min(), self.knots.max(), 100)
            if grid is None
            else np.asarray(grid, dtype=float)
        )
        frame = pd.DataFrame({self.var: xs})
        return frame, self.term_design(frame)


def make_thin_plate(var: str, x: np.ndarray, k: int) -> ThinPlate:
    if k < 3:
        raise MethodIncompatibility(f"gam: s({var}, bs='tp') needs k >= 3.")
    xu = np.unique(x)
    if xu.size < k:
        raise DataInsufficient(
            f"gam: {var} has {xu.size} distinct values, fewer than the "
            f"k = {k} basis functions of its smooth.",
            recovery_hint=f"Use s({var}, k=...) with a smaller k.",
        )
    if xu.size > _TP_MAX_KNOTS:
        # the eigen-decomposition is cubic in the number of knots; an even
        # spread of the distinct values stands in for all of them
        pick = np.unique(
            np.round(np.linspace(0, xu.size - 1, _TP_MAX_KNOTS)).astype(int)
        )
        xu = xu[pick]
    shift = float(np.mean(x))
    E = _eta(xu[:, None] - xu[None, :])
    T = np.column_stack([np.ones(xu.size), xu - shift])
    vals, vecs = np.linalg.eigh(E)
    order = np.argsort(-np.abs(vals))[:k]
    U, D = vecs[:, order], vals[order]
    # the wiggly coefficients are orthogonal to constants and linear terms
    Qc, _ = np.linalg.qr(U.T @ T, mode="complete")
    Z = Qc[:, T.shape[1] :]
    UZ = U @ Z
    pen = Z.T @ (D[:, None] * Z)
    pen = (pen + pen.T) / 2.0
    w, V = np.linalg.eigh(pen)
    if np.min(w) < -1e-8 * max(1.0, float(np.max(np.abs(w)))):
        raise MethodIncompatibility(
            f"gam: the thin plate penalty of {var} is not positive; k = {k} "
            "is too small for the truncation to be valid.",
            recovery_hint="Raise k, or use the default P-spline basis.",
        )
    root_wiggly = np.sqrt(np.maximum(w, 0.0))[:, None] * V.T
    ncol = Z.shape[1] + 2
    root_full = np.zeros((Z.shape[1], ncol))
    root_full[:, : Z.shape[1]] = root_wiggly
    proto = ThinPlate(
        var=var, k=k, knots=xu, UZ=UZ, shift=shift, Q=np.eye(ncol), root=root_full
    )
    Q = _sum_to_zero(proto.raw(x))
    proto.Q = Q
    proto.root = root_full @ Q
    return proto


# ---------------------------------------------------------------- term tests


def mixture_tail(weights: Sequence[float], dfs: Sequence[float], t: float) -> float:
    """``P(sum_j w_j chi2_{h_j} > t)`` by Imhof's inversion of the
    characteristic function [@imhof1961computing]. Weights may be
    negative, which is how a ratio to an independent chi-square is turned
    into a tail at zero."""
    from scipy import integrate

    w = np.asarray(weights, dtype=float)
    h = np.asarray(dfs, dtype=float)

    def integrand(u: float) -> float:
        theta = 0.5 * np.sum(h * np.arctan(w * u)) - 0.5 * t * u
        log_rho = 0.25 * np.sum(h * np.log1p((w * u) ** 2))
        if log_rho > 700.0:  # the integrand is zero to machine precision
            return 0.0
        return float(np.sin(theta) / (u * np.exp(log_rho)))

    value, _ = integrate.quad(integrand, 0.0, np.inf, limit=500, epsabs=1e-12)
    return float(min(1.0, max(0.0, 0.5 + value / np.pi)))


def smooth_term_test(
    X: np.ndarray,
    beta: np.ndarray,
    V: np.ndarray,
    ref_df: float,
    resid_df: Optional[float],
) -> Tuple[float, float]:
    """Wald-type test that a smooth term is zero (Wood 2013).

    ``X`` holds the term's (weighted) design columns, ``beta`` and ``V``
    its coefficients and their Bayesian covariance. The statistic is
    ``f' V_f^{r-} f`` for the fitted term ``f = X beta``, with a
    pseudo-inverse of rank ``r = ref_df``. A fractional ``r`` is handled by
    the block ``[[1, rho], [rho, nu]]`` on the two eigen-directions it
    falls between, which makes the null distribution a weighted sum of
    chi-squares with weights summing to ``r``. With an estimated scale
    (``resid_df`` given) the reference is the ratio of that sum to an
    independent chi-square.

    The cross term of the block changes sign with the (arbitrary) signs of
    the two eigenvectors. As in mgcv, the p-value is the average over the
    two signs; the statistic returned is the average as well, where mgcv
    prints whichever of the two its linear algebra produced.
    Returns ``(statistic, p-value)``.
    """
    R = np.asarray(np.linalg.qr(X, mode="r"))
    f = R @ beta
    Vf = R @ V @ R.T
    Vf = (Vf + Vf.T) / 2.0
    d, U = np.linalg.eigh(Vf)
    order = np.argsort(-d)
    d, U = d[order], U[:, order]
    q = int(np.sum(d > d[0] * 1e-12))
    r = float(min(ref_df, q))
    k = int(np.floor(r + 1e-10))
    nu = r - k
    z = U.T @ f

    def tail(weights: List[float], dfs: List[float], stat: float) -> float:
        if resid_df is None:
            return mixture_tail(weights, dfs, stat)
        # P(Q / (chi2_m / m) > stat) = P(Q - (stat / m) chi2_m > 0)
        return mixture_tail(weights + [-stat / resid_df], dfs + [float(resid_df)], 0.0)

    if nu < 1e-8 or k >= q:
        k = min(k, q)
        stat = float(np.sum(z[:k] ** 2 / d[:k]))
        return stat, tail([1.0], [float(k)], stat)
    if k == 0:
        # less than one degree of freedom: a single scaled direction
        stat = float(nu * z[0] ** 2 / d[0])
        return stat, tail([nu], [1.0], stat)
    rho = np.sqrt(nu * (1.0 - nu) / 2.0)
    a, b = z[k - 1 : k + 1] / np.sqrt(d[k - 1 : k + 1])
    base = float(np.sum(z[: k - 1] ** 2 / d[: k - 1]) + a * a + nu * b * b)
    cross = 2.0 * rho * a * b
    nu1 = (nu + 1.0 + np.sqrt(1.0 - nu**2)) / 2.0
    nu2 = nu + 1.0 - nu1
    weights, dfs = [1.0, nu1, nu2], [float(k - 1), 1.0, 1.0]
    if k == 1:
        weights, dfs = weights[1:], dfs[1:]
    pval = 0.5 * (
        tail(list(weights), list(dfs), base + cross)
        + tail(list(weights), list(dfs), base - cross)
    )
    return base, float(pval)
