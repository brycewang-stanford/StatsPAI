"""Oster (2019) coefficient-stability estimator -- the exact solution.

Shared by :func:`sp.oster_bounds` and :func:`sp.oster_delta`. The exact
delta and beta* (:func:`oster_delta_exact`, :func:`oster_beta_exact`) solve
the equations of Oster (2019) [@oster2019unobservable] in closed form; they
were rewritten clean-room from the paper and the MIT-licensed R package
``robomit`` (notice in ``THIRD_PARTY_NOTICES.md``), with Oster's Stata
``psacalc`` 2.1 used only as a black box (``docs/dev/cleanroom/oster.md``).
The inputs are those ``psacalc`` computes from the data (checked on its output):

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


def _oster_moments(inp: Dict[str, float], r_max: float) -> Dict[str, float]:
    """The scaled moments that enter Oster's (2019) bias equations."""
    syy = float(inp["sigma_yy"])
    return {
        "b_o": float(inp["beta_o"]),
        "b_t": float(inp["beta_t"]),
        "sxx": float(inp["sigma_xx"]),
        "tx": float(inp["t_x"]),
        "gap": float(inp["beta_o"]) - float(inp["beta_t"]),
        "r_move": (float(inp["r_t"]) - float(inp["r_o"])) * syy,
        "r_room": (float(r_max) - float(inp["r_t"])) * syy,
    }


def oster_delta_exact(inp: Dict[str, float], r_max: float, beta: float = 0.0) -> float:
    """delta such that the bias-adjusted coefficient equals ``beta``.

    Oster's (2019) Proposition 2 solved for delta: with the bias
    ``nu = beta_t - beta`` that the unobservables would have to explain,
    delta is a ratio of two cubic polynomials in ``nu`` whose coefficients
    are the moments of :func:`_oster_moments`. Written from the paper and
    the MIT-licensed R package ``robomit`` (notice in
    ``THIRD_PARTY_NOTICES.md``); matches Stata ``psacalc`` 2.1 output,
    used as a black box (``docs/dev/cleanroom/oster.md``).
    """
    k = _oster_moments(inp, r_max)
    sxx, tx, gap = k["sxx"], k["tx"], k["gap"]
    nu = k["b_t"] - float(beta)
    cubic = nu**3 * (tx * sxx - tx**2)
    numer = (
        nu * k["r_move"] * tx
        + nu * sxx * tx * gap**2
        + 2.0 * nu**2 * tx * gap * sxx
        + cubic
    )
    denom = (
        k["r_room"] * gap * sxx
        + nu * k["r_room"] * (sxx - tx)
        + nu**2 * tx * gap * sxx
        + cubic
    )
    return float(numer / denom)


def _rank_roots(candidates: Sequence[float], b_t: float, b_o: float) -> list:
    """Candidates ordered by squared distance to ``beta_t``.

    A candidate whose bias ``candidate - beta_t`` does not have the sign of
    the observed movement ``beta_t - beta_o`` (Oster's assumption 3) is
    pushed behind the others; each such candidate, taken in the order
    given, is assigned one more than the largest distance so far.
    """
    dist = [(c - b_t) ** 2 for c in candidates]
    for i, c in enumerate(candidates):
        if np.sign(c - b_t) != np.sign(b_t - b_o):
            dist[i] = max(dist) + 1.0
    order = sorted(range(len(candidates)), key=lambda i: dist[i])
    return [candidates[i] for i in order]


def oster_beta_exact(
    inp: Dict[str, float], r_max: float, delta: float = 1.0
) -> Dict[str, object]:
    """Bias-adjusted coefficient beta*(delta, R_max) and its alternative roots.

    The bias ``nu = beta_t - beta*`` solves Oster's (2019) equation: a
    quadratic when ``delta = 1`` and a cubic otherwise (solved in closed
    form, trigonometrically when it has three real roots and by Cardano's
    formula when it has one). The reported root is the one closest to the
    controlled coefficient among the roots whose bias has the same sign as
    the observed coefficient movement (Oster's assumption 3); with
    ``delta = 1`` the closer root is swapped for the other when it has the
    wrong sign. A negative quadratic discriminant gives NaN. Written from
    the paper and the MIT-licensed R package ``robomit`` (notice in
    ``THIRD_PARTY_NOTICES.md``); matches Stata ``psacalc`` 2.1 output,
    used as a black box (``docs/dev/cleanroom/oster.md``).

    Returns ``{"beta": ..., "alternatives": [...], "roots": [...]}``:
    ``roots`` in solution order, ``alternatives`` the other roots in the
    order of the selection rule.
    """
    k = _oster_moments(inp, r_max)
    b_o, b_t, sxx, tx, gap = k["b_o"], k["b_t"], k["sxx"], k["tx"], k["gap"]
    if delta == 1.0:
        mid = k["r_room"] * (sxx - tx) - k["r_move"] * tx - sxx * tx * gap**2
        disc = mid**2 + 4.0 * k["r_room"] * gap**2 * sxx**2 * tx
        lead = -2.0 * tx * gap * sxx
        with np.errstate(invalid="ignore"):
            half = np.sqrt(disc)
        roots = [b_t - (-mid - half) / lead, b_t - (-mid + half) / lead]
        near, far = (
            roots if (roots[0] - b_t) ** 2 < (roots[1] - b_t) ** 2 else roots[::-1]
        )
        if np.sign(near - b_t) != np.sign(b_t - b_o):
            near, far = far, near
        return {
            "beta": float(near),
            "alternatives": [float(far)],
            "roots": [float(r) for r in roots],
        }
    # delta != 1: nu^3 + a nu^2 + b nu + c = 0.
    scale = (delta - 1.0) * (tx * sxx - tx**2)
    a = tx * gap * sxx * (delta - 2.0) / scale
    b = (
        delta * k["r_room"] * (sxx - tx) - k["r_move"] * tx - sxx * tx * gap**2
    ) / scale
    c = k["r_room"] * delta * gap * sxx / scale
    q = (a**2 - 3.0 * b) / 9.0
    r = (2.0 * a**3 - 9.0 * a * b + 27.0 * c) / 54.0
    if r**2 - q**3 < 0:
        angle = np.arccos(r / np.sqrt(q**3))
        shifts = (0.0, 2.0 * np.pi, -2.0 * np.pi)
        nus = [-2.0 * np.sqrt(q) * np.cos((angle + s) / 3.0) - a / 3.0 for s in shifts]
        roots = [b_t - nu for nu in nus]
        ranked = _rank_roots(roots, b_t, b_o)
        return {
            "beta": float(ranked[0]),
            "alternatives": [float(x) for x in ranked[1:]],
            "roots": [float(x) for x in roots],
        }
    s = np.sqrt(r**2 - q**3)
    nu = np.cbrt(-r + s) + np.cbrt(-r - s) - a / 3.0
    return {"beta": float(b_t - nu), "alternatives": [], "roots": [float(b_t - nu)]}


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
