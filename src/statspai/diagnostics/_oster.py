"""Oster (2019) coefficient-stability estimator -- the exact solution.

Shared by :func:`sp.oster_bounds` and :func:`sp.oster_delta`. A port of the
Mata code of Oster's Stata ``psacalc`` 2.1 (functions ``bound``,
``d1quadsol`` and ``dnot1cubsol``), which R ``robomit`` also ports. The
inputs are those ``psacalc`` computes from the data:

* ``beta_o``, ``r_o`` -- treatment coefficient and R-squared of the short
  regression of ``y`` on the treatment and the "unrelated" controls
  (``psacalc``'s ``mcontrol()``; none by default);
* ``beta_t``, ``r_t`` -- the same from the long regression with all controls;
* ``sigma_yy`` -- the sample variance of ``y``;
* ``sigma_xx`` -- the sample variance of the treatment after residualising it
  on the unrelated controls (the plain variance of the treatment without
  them);
* ``t_x`` -- the sample variance of the residual of the treatment regressed
  on every other regressor of the long model.

The three variances are ratios of sums of squares, so the divisor (N - 1 in
Stata ``summarize`` and R ``var``) cancels out of every formula.

Oster's first-order approximation, ``beta* ~= beta_t - delta (beta_o - beta_t)
(R_max - R_t) / (R_t - R_o)``, is kept in :func:`oster_approx_beta` /
:func:`oster_approx_delta` for the case where only coefficients and R-squared
values are known (e.g. read off a published table).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from ..exceptions import MethodIncompatibility


def oster_inputs(
    data: pd.DataFrame,
    y: str,
    treat: str,
    controls: Sequence[str],
    mcontrol: Sequence[str] = (),
) -> Dict[str, float]:
    """Regression inputs of the Oster estimator, computed as ``psacalc`` does."""
    controls = [c for c in controls if c not in mcontrol]
    cols = list(dict.fromkeys([y, treat, *mcontrol, *controls]))
    df = data[cols].dropna()
    Y = df[y].to_numpy(dtype=float)
    D = df[treat].to_numpy(dtype=float)
    n = Y.size
    one = np.ones((n, 1))
    M = df[list(mcontrol)].to_numpy(dtype=float) if mcontrol else np.empty((n, 0))
    C = df[controls].to_numpy(dtype=float) if controls else np.empty((n, 0))

    def _ols(X: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.asarray(np.linalg.lstsq(X, v, rcond=None)[0], dtype=float)

    tss = float(np.sum((Y - Y.mean()) ** 2))

    X_short = np.hstack([one, D[:, None], M])
    b_short = _ols(X_short, Y)
    r_o = 1.0 - float(np.sum((Y - X_short @ b_short) ** 2)) / tss

    X_long = np.hstack([one, D[:, None], M, C])
    b_long = _ols(X_long, Y)
    r_t = 1.0 - float(np.sum((Y - X_long @ b_long) ** 2)) / tss

    X_m = np.hstack([one, M])
    e_xm = D - X_m @ _ols(X_m, D)
    X_rest = np.hstack([one, M, C])
    e_xr = D - X_rest @ _ols(X_rest, D)

    return {
        "beta_o": float(b_short[1]),
        "r_o": float(r_o),
        "beta_t": float(b_long[1]),
        "r_t": float(r_t),
        "sigma_yy": float(np.var(Y, ddof=1)),
        "sigma_xx": float(np.var(e_xm, ddof=1)),
        "t_x": float(np.var(e_xr, ddof=1)),
        "n": float(n),
        # Stata `regress` e(r2_a) of the long regression.
        "r2_a": 1.0 - (1.0 - r_t) * (n - 1) / (n - np.linalg.matrix_rank(X_long)),
    }


def _within_rank(
    Z: np.ndarray, other_codes: Sequence[np.ndarray], group_codes: np.ndarray
) -> int:
    """Rank of ``[Z, dummies(other_codes)]`` after removing group means.

    This is the number of regressors Stata ``xtreg, fe`` keeps (its
    ``e(df_m)`` minus ``N_g - 1``) when the other fixed effects enter as
    ``i.`` dummies: ``rank(W' (I - P_g) W)`` for ``W = [Z, D]``, built from
    group sums so the dummies never need to be demeaned densely.
    """
    from scipy import sparse

    n = Z.shape[0]
    blocks = [sparse.csr_matrix(Z)]
    for codes in other_codes:
        c = pd.factorize(np.asarray(codes))[0]
        blocks.append(
            sparse.csr_matrix((np.ones(n), (np.arange(n), c)), shape=(n, c.max() + 1))
        )
    W = sparse.hstack(blocks).tocsr()
    g = pd.factorize(np.asarray(group_codes))[0]
    G = g.max() + 1
    Dg = sparse.csr_matrix((np.ones(n), (np.arange(n), g)), shape=(n, G))
    C = (Dg.T @ W).toarray()
    ng = np.bincount(g, minlength=G).astype(float)
    M = (W.T @ W).toarray() - C.T @ (C / ng[:, None])
    d = np.sqrt(np.clip(np.diag(M), 0.0, None))
    live = d > 1e-10 * max(1.0, float(d.max()) if d.size else 1.0)
    if not live.any():
        return 0
    Ms = M[np.ix_(live, live)] / np.outer(d[live], d[live])
    eig = np.linalg.eigvalsh((Ms + Ms.T) / 2.0)
    return int(np.sum(eig > eig.max() * max(Ms.shape) * 1e-12))


def oster_inputs_fe(
    data: pd.DataFrame,
    y: str,
    treat: str,
    controls: Sequence[str],
    absorb: Sequence[str],
    absorb_controls: Sequence[str] = (),
    cluster: Optional[str] = None,
) -> Dict[str, float]:
    """``psacalc`` inputs after a fixed-effects regression (``xtreg, fe``).

    ``absorb`` are the effects in both regressions (``xtreg``'s panel
    variable, or ``areg``'s ``absorb()``); ``absorb_controls`` are fixed
    effects that are *controls* -- in the long regression only, as
    ``i.ind#i.year`` dummies are in ``xtreg y d x i.ind#i.year, fe``.
    As ``psacalc`` does after ``xtreg, fe``: the coefficients and R-squared
    values are within (the R-squared relative to the within sum of squares
    of ``y``), ``sigma_yy`` is the variance of ``y`` itself, ``sigma_xx``
    the variance of the within-transformed treatment and ``t_x`` that of
    the treatment residualised on every other regressor and effect.

    Also returns ``r2_a`` -- Stata ``xtreg, fe``'s ``e(r2_a)``,
    ``1 - (1 - R2_within)(N - 1)/(N - N_g - K)`` with ``K`` the exact rank
    of the regressors and control dummies given the panel effect (one
    ``absorb`` effect only) -- the usual base of ``R_max = 1.3 r2_a``.
    Singletons are kept, as ``xtreg`` keeps them. With ``cluster`` (Stata
    ``vce(cluster c)``) and the panel nested in it, ``xtreg`` stops
    charging the ``N_g`` panel means: ``r2_a`` then uses ``N - K - 1``.
    """
    from ..panel.hdfe import Absorber

    absorb = list(absorb)
    absorb_controls = list(absorb_controls)
    extra = [cluster] if cluster is not None else []
    cols = list(dict.fromkeys([y, treat, *controls, *absorb, *absorb_controls, *extra]))
    df = data[cols].dropna()
    Y = df[y].to_numpy(dtype=float)
    D = df[treat].to_numpy(dtype=float)
    C = df[list(controls)].to_numpy(dtype=float) if controls else np.empty((len(df), 0))
    n = Y.size

    ab_s = Absorber(df[absorb], drop_singletons=False, tol=1e-12)
    ab_l = Absorber(df[absorb + absorb_controls], drop_singletons=False, tol=1e-12)
    Yw_s, Dw_s = ab_s.demean(Y), ab_s.demean(D)
    Yw_l, Dw_l = ab_l.demean(Y), ab_l.demean(D)
    Cw_l = ab_l.demean(C) if C.shape[1] else C

    def _ols(X: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.asarray(np.linalg.lstsq(X, v, rcond=None)[0], dtype=float)

    tss_w = float(np.sum(Yw_s**2))
    b_s = _ols(Dw_s[:, None], Yw_s)
    r_o = 1.0 - float(np.sum((Yw_s - Dw_s * b_s[0]) ** 2)) / tss_w
    X_l = np.column_stack([Dw_l, Cw_l])
    b_l = _ols(X_l, Yw_l)
    rss_l = float(np.sum((Yw_l - X_l @ b_l) ** 2))
    r_t = 1.0 - rss_l / tss_w
    e_x = Dw_l - (Cw_l @ _ols(Cw_l, Dw_l) if C.shape[1] else 0.0)

    out = {
        "beta_o": float(b_s[0]),
        "r_o": float(r_o),
        "beta_t": float(b_l[0]),
        "r_t": float(r_t),
        "sigma_yy": float(np.var(Y, ddof=1)),
        "sigma_xx": float(np.var(Dw_s, ddof=1)),
        "t_x": float(np.var(e_x, ddof=1)),
        "n": float(n),
    }
    if len(absorb) == 1:
        g = df[absorb[0]].to_numpy()
        n_g = int(pd.Series(g).nunique())
        k = _within_rank(
            np.column_stack([D, C]), [df[a].to_numpy() for a in absorb_controls], g
        )
        nested = cluster is not None and bool(
            (df.groupby(absorb[0])[cluster].nunique() <= 1).all()
        )
        df_r = n - k - 1 if nested else n - n_g - k
        out["r2_a"] = 1.0 - (1.0 - r_t) * (n - 1) / df_r
        out["panel_nested_in_cluster"] = float(nested)
        out["n_groups"] = float(n_g)
        out["rank"] = float(k)
    return out


def resolve_r_max(r_max: object, r2_long: float, r2_a: float = float("nan")) -> float:
    """``r_max`` as a number, or a ``"<mult>*r2"`` / ``"<mult>*r2_a"`` rule."""
    if r_max is None or isinstance(r_max, (int, float, np.floating)):
        return r_max  # type: ignore[return-value]
    text = str(r_max).replace(" ", "").lower()
    mult_s, _, base = text.partition("*")
    if base not in ("r2", "r2_a"):
        mult_s, base = base, mult_s  # also accept "r2_a*1.3"
    try:
        mult = float(mult_s)
    except ValueError:
        raise MethodIncompatibility(
            f"r_max={r_max!r}: use a number or '<multiplier>*r2' / "
            "'<multiplier>*r2_a' (e.g. '1.3*r2_a')."
        ) from None
    if base == "r2":
        return mult * r2_long
    if base == "r2_a":
        if not np.isfinite(r2_a):
            raise MethodIncompatibility(
                "r_max='...*r2_a' needs the adjusted R-squared, available "
                "from data with absorb= (one panel effect) or from data "
                "without absorbed effects."
            )
        return mult * r2_a
    raise MethodIncompatibility(f"r_max={r_max!r} not understood.")


def oster_delta_exact(inp: Dict[str, float], r_max: float, beta: float = 0.0) -> float:
    """delta such that the bias-adjusted coefficient equals ``beta``.

    This is psacalc's ``bound``.
    """
    bo_m_bt = inp["beta_o"] - inp["beta_t"]
    rt_m_ro_t_syy = (inp["r_t"] - inp["r_o"]) * inp["sigma_yy"]
    rm_m_rt_t_syy = (r_max - inp["r_t"]) * inp["sigma_yy"]
    sxx, tx = inp["sigma_xx"], inp["t_x"]
    bt_m_b = inp["beta_t"] - beta
    num = (
        bt_m_b * rt_m_ro_t_syy * tx
        + bt_m_b * sxx * tx * bo_m_bt**2
        + 2.0 * bt_m_b**2 * (tx * bo_m_bt * sxx)
        + bt_m_b**3 * (tx * sxx - tx**2)
    )
    den = (
        rm_m_rt_t_syy * bo_m_bt * sxx
        + bt_m_b * rm_m_rt_t_syy * (sxx - tx)
        + bt_m_b**2 * (tx * bo_m_bt * sxx)
        + bt_m_b**3 * (tx * sxx - tx**2)
    )
    return float(num / den) if den != 0 else float(np.inf)


def oster_beta_exact(
    inp: Dict[str, float], r_max: float, delta: float = 1.0
) -> Dict[str, object]:
    """Bias-adjusted coefficient beta*(delta, R_max) and its alternative roots.

    delta = 1 solves psacalc's quadratic, any other delta its cubic. The
    reported root is the one closest to the controlled coefficient among the
    roots whose bias has the same sign as the observed coefficient movement
    (Oster's assumption 3) -- psacalc's selection rule, reproduced exactly,
    including its handling when no root satisfies the sign condition.
    Returns ``{"beta": ..., "alternatives": [...], "roots": [...]}``.
    """
    bo = inp["beta_o"]
    bt = inp["beta_t"]
    bo_m_bt = bo - bt
    rt_m_ro_t_syy = (inp["r_t"] - inp["r_o"]) * inp["sigma_yy"]
    rm_m_rt_t_syy = (r_max - inp["r_t"]) * inp["sigma_yy"]
    sxx, tx = inp["sigma_xx"], inp["t_x"]

    if delta == 1:
        cap_theta = (
            rm_m_rt_t_syy * (sxx - tx) - rt_m_ro_t_syy * tx - sxx * tx * bo_m_bt**2
        )
        d1_1 = 4.0 * rm_m_rt_t_syy * bo_m_bt**2 * sxx**2 * tx
        d1_2 = -2.0 * tx * bo_m_bt * sxx
        disc = np.sqrt(cap_theta**2 + d1_1)
        beta1 = bt - (-cap_theta - disc) / d1_2
        beta2 = bt - (-cap_theta + disc) / d1_2
        if (beta1 - bt) ** 2 < (beta2 - bt) ** 2:
            betax, alt = beta1, beta2
        else:
            betax, alt = beta2, beta1
        if np.sign(betax - bt) != np.sign(bt - bo):
            betax, alt = alt, betax
        return {
            "beta": float(betax),
            "alternatives": [float(alt)],
            "roots": [float(beta1), float(beta2)],
        }

    denom = (delta - 1.0) * (tx * sxx - tx**2)
    A = tx * bo_m_bt * sxx * (delta - 2.0) / denom
    B = (
        delta * rm_m_rt_t_syy * (sxx - tx) - rt_m_ro_t_syy * tx - sxx * tx * bo_m_bt**2
    ) / denom
    C = (rm_m_rt_t_syy * delta * bo_m_bt * sxx) / denom
    Q = (A**2 - 3.0 * B) / 9.0
    R = (2.0 * A**3 - 9.0 * A * B + 27.0 * C) / 54.0
    D = R**2 - Q**3
    if D < 0:
        theta = np.arccos(R / np.sqrt(Q**3))
        sols = np.array(
            [
                -2.0 * np.sqrt(Q) * np.cos(theta / 3.0) - A / 3.0,
                -2.0 * np.sqrt(Q) * np.cos((theta + 2.0 * np.pi) / 3.0) - A / 3.0,
                -2.0 * np.sqrt(Q) * np.cos((theta - 2.0 * np.pi) / 3.0) - A / 3.0,
            ]
        )
        betas = bt - sols
        dists = (betas - bt) ** 2
        for i in range(3):
            if np.sign(betas[i] - bt) != np.sign(bt - bo):
                dists[i] = dists.max() + 1.0
        order = np.argsort(dists, kind="stable")
        return {
            "beta": float(betas[order[0]]),
            "alternatives": [float(betas[order[1]]), float(betas[order[2]])],
            "roots": [float(b) for b in betas],
        }
    t1 = -R + np.sqrt(D)
    t2 = -R - np.sqrt(D)
    sol = (
        np.sign(t1) * abs(t1) ** (1.0 / 3.0)
        + np.sign(t2) * abs(t2) ** (1.0 / 3.0)
        - A / 3.0
    )
    return {"beta": float(bt - sol), "alternatives": [], "roots": [float(bt - sol)]}


def oster_approx_beta(
    beta_o: float, beta_t: float, r_o: float, r_t: float, r_max: float, delta: float
) -> float:
    """Oster's first-order approximation to beta*(delta, R_max)."""
    denom = r_t - r_o
    if abs(denom) < 1e-12:
        return float(beta_t)
    return float(beta_t - delta * (beta_o - beta_t) * (r_max - r_t) / denom)


def oster_approx_delta(
    beta_o: float, beta_t: float, r_o: float, r_t: float, r_max: float
) -> float:
    """delta that sets the approximate beta* to zero."""
    denom = (beta_o - beta_t) * (r_max - r_t)
    if abs(denom) < 1e-12:
        return float(np.inf)
    return float(beta_t * (r_t - r_o) / denom)
