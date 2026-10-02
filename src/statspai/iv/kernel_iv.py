"""
Kernel IV regression with a uniform band.

Estimates the structural function ``h(D)`` in ``Y = h(D) + u`` when ``D``
is endogenous and an instrument ``Z`` is available, by kernel smoothing
only. The estimator is a control-function one, for the triangular model

    D = m(Z) + V,        E[u | Z, V] = E[u | V] = lambda(V),

under which ``E[Y | D, V] = h(D) + lambda(V)``:

1. ``V`` is the residual of a local-linear regression of ``D`` on ``Z``;
2. ``g(d, v) = E[Y | D = d, V = v]`` is estimated by a bivariate
   local-linear regression at each grid point ``d`` and each observed
   residual ``v``;
3. ``h(d_j) - h(d_{j-1})`` is the average of ``g(d_j, v) - g(d_{j-1}, v)``
   over the residuals with support at both grid points, which removes
   ``lambda`` exactly under additivity, and the level is set by
   ``E[Y] = E[h(D)]``.

The estimate is linear in ``Y`` given the first stage and the bandwidths,
so the uniform band comes from a multiplier bootstrap of the additive fit's
residuals. It does not carry the first stage's estimation error.

See also Lob et al. (2025, arXiv 2511.21603) on uniform inference for
kernel instrumental-variable regression; the estimator here is the
control-function one described above, not an RKHS two-stage estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin


@dataclass
class KernelIVResult(ResultProtocolMixin):
    """Output of kernel IV regression.

    Returned by :func:`sp.kernel_iv`. Holds the treatment grid, the
    estimated structural function ``h_hat`` on that grid, the uniform
    confidence band (``ci_low`` / ``ci_high``), and the bandwidth.
    Call ``.summary()`` for a formatted preview.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 300
    >>> z = rng.normal(size=n)
    >>> u = rng.normal(size=n)
    >>> d = 0.8 * z + 0.5 * u + rng.normal(size=n)
    >>> y = np.sin(d) + u + 0.3 * rng.normal(size=n)
    >>> df = pd.DataFrame({'y': y, 'd': d, 'z': z})
    >>> res = sp.kernel_iv(df, y='y', treat='d', instrument='z',
    ...                    n_boot=50)
    >>> isinstance(res, sp.KernelIVResult)
    True
    >>> res.h_hat.shape  # structural function on a 30-point grid
    (30,)
    """

    grid: np.ndarray  # (n_grid,) treatment values
    h_hat: np.ndarray  # (n_grid,) structural function estimate
    ci_low: np.ndarray
    ci_high: np.ndarray
    bandwidth: float
    n_obs: int

    def summary(self) -> str:
        rows = [
            "Kernel IV Regression (uniform CI)",
            "=" * 42,
            f"  N = {self.n_obs}, bandwidth = {self.bandwidth:.4f}",
            "  d        h(d)     95% UCB",
        ]
        for d, h, lo, hi in zip(
            self.grid[:5], self.h_hat[:5], self.ci_low[:5], self.ci_high[:5]
        ):
            rows.append(f"  {d:+.3f}  {h:+.4f}   [{lo:+.4f}, {hi:+.4f}]")
        if len(self.grid) > 5:
            rows.append(f"  ... (+{len(self.grid) - 5} more)")
        return "\n".join(rows)


def _kern(u: Any) -> Any:
    return np.exp(-0.5 * u * u)


def _local_linear(y: np.ndarray, x: np.ndarray, at: np.ndarray, h: float) -> Any:
    """Local-linear regression of ``y`` on ``x`` evaluated at ``at``."""
    out = np.empty(len(at))
    for start in range(0, len(at), 512):
        pts = at[start : start + 512]
        u = x[None, :] - pts[:, None]
        w = _kern(u / h)
        s0, s1, s2 = w.sum(1), (w * u).sum(1), (w * u * u).sum(1)
        t0, t1 = w @ y, (w * u) @ y
        den = s0 * s2 - s1 * s1
        flat = np.abs(den) < 1e-12 * np.maximum(s0 * s2, 1e-300)
        out[start : start + 512] = np.where(
            flat,
            t0 / np.maximum(s0, 1e-300),
            (s2 * t0 - s1 * t1) / np.where(flat, 1, den),
        )
    return out


def _control_function_operator(
    D: np.ndarray,
    V: np.ndarray,
    grid: np.ndarray,
    h_d: float,
    h_v: float,
    ridge: float,
    rows: np.ndarray,
    trim: float = 0.02,
) -> np.ndarray:
    """The ``(n_grid, n)`` matrix ``L`` with ``h_hat = L @ Y``.

    ``rows`` indexes the residuals ``v_i`` the surface is evaluated at.
    """
    n = len(D)
    dV = V[None, :] - V[rows][:, None]  # [i, k] = V_k - v_i
    Kv = _kern(dV / h_v)
    Kv1 = Kv * dV
    Kv2 = Kv1 * dV
    eye = np.eye(3)[None]

    def surface(d: float) -> Any:
        u = D - d
        a = _kern(u / h_d)
        b = a * u
        c = b * u
        S00, S10, S20 = Kv @ a, Kv @ b, Kv @ c
        S01, S11, S02 = Kv1 @ a, Kv1 @ b, Kv2 @ a
        A = np.stack(
            [
                np.stack([S00, S10, S01], 1),
                np.stack([S10, S20, S11], 1),
                np.stack([S01, S11, S02], 1),
            ],
            1,
        )
        ok = S00 > trim * S00.max()
        coef = np.zeros((len(rows), 3))
        if ok.any():
            scale = np.trace(A[ok], axis1=1, axis2=2)[:, None, None] / 3.0
            rhs = np.zeros((int(ok.sum()), 3, 1))
            rhs[:, 0, 0] = 1.0
            coef[ok] = np.linalg.solve(A[ok] + ridge * 1e-6 * scale * eye, rhs)[:, :, 0]
        return a, b, coef, ok

    def row_vector(a: Any, b: Any, coef: Any, mask: Any) -> Any:
        m = mask / mask.sum()
        return (
            a * ((coef[:, 0] * m) @ Kv)
            + b * ((coef[:, 1] * m) @ Kv)
            + a * ((coef[:, 2] * m) @ Kv1)
        )

    L = np.zeros((len(grid), n))
    prev = surface(float(grid[0]))
    broken = False
    for j in range(1, len(grid)):
        cur = surface(float(grid[j]))
        both = prev[3] & cur[3]
        if broken or both.sum() < 5:
            L[j] = np.nan
            broken = True
        else:
            L[j] = (
                L[j - 1]
                + row_vector(cur[0], cur[1], cur[2], both)
                - row_vector(prev[0], prev[1], prev[2], both)
            )
        prev = cur

    # Level: E[Y] = E[h(D)] over the observations inside the grid.
    good = np.isfinite(L).all(1)
    if good.sum() >= 2:
        g_ok = grid[good]
        inside = (D >= g_ok[0]) & (D <= g_ok[-1])
        if inside.sum() >= 5:
            pos = np.clip(np.searchsorted(g_ok, D[inside]) - 1, 0, len(g_ok) - 2)
            frac = (D[inside] - g_ok[pos]) / (g_ok[pos + 1] - g_ok[pos])
            p_bar = np.zeros(len(g_ok))
            np.add.at(p_bar, pos, 1 - frac)
            np.add.at(p_bar, pos + 1, frac)
            p_bar /= inside.sum()
            shift = inside / inside.sum() - p_bar @ L[good]
            L[good] = L[good] + shift[None, :]
    return L


def kernel_iv(
    data: pd.DataFrame,
    y: str,
    treat: str,
    instrument: str,
    grid: Optional[np.ndarray] = None,
    bandwidth: Optional[float] = None,
    ridge: float = 1e-3,
    alpha: float = 0.05,
    n_boot: int = 100,
    seed: int = 0,
) -> KernelIVResult:
    """
    Kernel IV regression of Y on D instrumented by Z.

    A control-function estimator of ``h`` in ``Y = h(D) + u`` under the
    triangular model ``D = m(Z) + V`` with ``E[u | Z, V] = E[u | V]``: the
    first-stage residual ``V`` is the control variable, ``E[Y | D, V]`` is
    fitted by bivariate local-linear regression, and ``h`` is recovered by
    differencing that surface along ``D`` over common support. See the
    module docstring for the steps.

    Parameters
    ----------
    data : pd.DataFrame
    y, treat, instrument : str
    grid : array, optional
        Grid of treatment values to evaluate h(d); defaults to the
        empirical 5–95th percentile range with 30 points. Must be
        increasing.
    bandwidth : float, optional
        Bandwidth for the treatment in the second stage. Defaults to a
        rule-of-thumb value at the two-dimensional rate,
        ``1.06 sd(D) n^(-1/6)``. The residual and the instrument use the
        same rule on their own scales.
    ridge : float, default 1e-3
        Relative ridge added to each local 3x3 system (times ``1e-6`` of
        its mean diagonal), for numerical stability only.
    alpha : float
    n_boot : int, default 100
        Multiplier-bootstrap replications for the uniform band.
    seed : int

    Returns
    -------
    KernelIVResult

    Notes
    -----
    Until the 2026-10 fix this function did not use the instrument: it
    returned the kernel regression of ``Y`` on ``D``, the confounded
    ``E[Y | D]``. On ``Y = sin(D) + u`` with ``D = 0.8 Z + 0.5 u + e`` and
    3,000 observations, the root mean squared error of the shape over
    ``[-1.5, 1.5]`` was 0.25; it is now 0.05.

    The model is restrictive in one direction (an additive first-stage
    error that carries all the endogeneity) and free in the other (no
    ill-posed inversion, so no regularisation bias in ``h``). For the
    series estimator that needs only ``E[u | Z] = 0`` see ``sp.iv.npiv``.
    Memory is ``O(min(n, 1500) * n)``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 300
    >>> z = rng.normal(size=n)
    >>> u = rng.normal(size=n)
    >>> d = 0.8 * z + 0.5 * u + rng.normal(size=n)
    >>> y = np.sin(d) + u + 0.3 * rng.normal(size=n)
    >>> df = pd.DataFrame({'y': y, 'd': d, 'z': z})
    >>> res = sp.kernel_iv(df, y='y', treat='d', instrument='z',
    ...                    n_boot=50)
    >>> res.n_obs
    300
    >>> res.h_hat.shape  # structural function on a 30-point grid
    (30,)
    >>> text = res.summary()  # h(d) with uniform 95% band
    """
    df = data[[y, treat, instrument]].dropna().reset_index(drop=True)
    Y = df[y].to_numpy(float)
    D = df[treat].to_numpy(float)
    Z = df[instrument].to_numpy(float)
    n = len(df)
    if n < 30:
        raise ValueError(f"kernel_iv needs at least 30 complete rows, got {n}.")

    def rule(x: np.ndarray, rate: float) -> float:
        return float(1.06 * x.std(ddof=1) * n**rate)

    if grid is None:
        grid = np.linspace(np.quantile(D, 0.05), np.quantile(D, 0.95), 30)
    grid = np.asarray(grid, dtype=float)
    if len(grid) < 2 or np.any(np.diff(grid) <= 0):
        raise ValueError("grid must hold at least two increasing treatment values.")
    rng = np.random.default_rng(seed)

    # Stage 1: control variable.
    V = D - _local_linear(D, Z, Z, rule(Z, -1 / 5))
    if V.std(ddof=1) < 1e-10 * max(D.std(ddof=1), 1e-300):
        raise ValueError(
            "kernel_iv: the treatment is an exact function of the instrument, "
            "so there is no first-stage residual to control for."
        )
    h_d = float(bandwidth) if bandwidth is not None else rule(D, -1 / 6)
    h_v = rule(V, -1 / 6)
    bandwidth = h_d

    # Stage 2: h_hat = L @ Y.
    rows = np.arange(n) if n <= 1500 else np.sort(rng.choice(n, 1500, replace=False))
    L = _control_function_operator(D, V, grid, h_d, h_v, ridge, rows)
    h_hat = L @ Y

    # Uniform band: multiplier bootstrap of the additive fit's residuals.
    ok = np.isfinite(h_hat)
    if ok.sum() >= 2:
        h_at_d = np.interp(D, grid[ok], h_hat[ok])
    else:
        h_at_d = np.full(n, np.nanmean(Y))
    lam = _local_linear(Y - h_at_d, V, V, h_v)
    fitted = h_at_d + lam
    resid = Y - fitted
    boot = np.full((n_boot, len(grid)), np.nan)
    for b in range(n_boot):
        boot[b] = L @ (fitted + rng.choice([-1.0, 1.0], size=n) * resid)
    if n_boot >= 2:
        sd = np.nanstd(boot, axis=0, ddof=1)
    else:
        sd = np.full(len(grid), np.nan)
    with np.errstate(all="ignore"):
        ratio = np.abs(boot - h_hat) / np.where(sd > 0, sd, np.nan)
        sup_band = (
            np.nanquantile(np.nanmax(ratio, axis=1), 1 - alpha)
            if n_boot >= 2 and np.isfinite(ratio).any()
            else np.nan
        )
    if not np.isfinite(sup_band):
        from scipy import stats

        sup_band = stats.norm.ppf(1 - alpha / 2)
    ci_low = h_hat - sup_band * sd
    ci_high = h_hat + sup_band * sd

    _result = KernelIVResult(
        grid=grid,
        h_hat=h_hat,
        ci_low=ci_low,
        ci_high=ci_high,
        bandwidth=float(bandwidth),
        n_obs=n,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.iv.kernel_iv",
            params={
                "y": y,
                "treat": treat,
                "instrument": instrument,
                "bandwidth": bandwidth,
                "ridge": ridge,
                "alpha": alpha,
                "n_boot": n_boot,
                "seed": seed,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result
