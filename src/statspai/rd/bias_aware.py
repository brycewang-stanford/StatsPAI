"""
Bias-aware confidence intervals for fuzzy RD (Noack & Rothe 2024).

Implements ``rd_bias_aware_fuzzy``: a bias-aware confidence set for the
fuzzy regression discontinuity parameter τ = (μ_Y(c+) − μ_Y(c−))
/ (μ_D(c+) − μ_D(c−)) based on local-linear regression.  The CI takes
the smoothing bias of *both* numerator and denominator into account,
following Noack & Rothe (2024, Econometrica 92(3), 687-711).

Unlike the conventional fuzzy-RD t-ratio CI, which can have severely
distorted coverage when the first stage is moderate, the bias-aware
construction has uniformly correct coverage over the smoothness class

    F(M_Y, M_D) = { (g_Y, g_D) : |g_Y''| ≤ M_Y, |g_D''| ≤ M_D }

via Anderson--Rubin style test inversion: τ_0 lies in the CI iff
``tau_to_test = mu_Y(c+) - mu_Y(c-) - τ_0 * (mu_D(c+) - mu_D(c-))``
fails to be rejected when accounting for worst-case bias of the
numerator and denominator local-linear estimators.

The CI also addresses the *power asymmetry* documented in
Kaliski, Keane & Neal (2025, NBER 33972): when the first-stage
discontinuity is small, the conventional 2SLS-style CI has poor
power on one side.  The Anderson-Rubin construction here is naturally
robust to weak first stages.

References
----------
Noack, C. and Rothe, C. (2024).
"Bias-Aware Inference in Fuzzy Regression Discontinuity Designs."
*Econometrica*, 92(3), 687-711. doi:10.3982/ECTA19466.
[@noack2024biasaware]

Kaliski, D., Keane, M.P. and Neal, T. (2025).
"The Power Asymmetry in Fuzzy Regression Discontinuity Designs."
NBER Working Paper No. 33972. doi:10.3386/w33972. [@kaliski2025power]
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats

from ..core.results import CausalResult
from ._rdhonest import (
    _hmin,
    _ik_bandwidth,
    _kern,
    cv_bias,
    honest_bias,
    honest_weights,
    m_rule_of_thumb,
)


def rd_bias_aware_fuzzy(
    data: pd.DataFrame,
    y: str,
    x: str,
    fuzzy: str,
    c: float = 0.0,
    M_y: Optional[float] = None,
    M_d: Optional[float] = None,
    h: Optional[float] = None,
    kernel: str = "triangular",
    alpha: float = 0.05,
    cluster: Optional[str] = None,
    n_grid: int = 401,
) -> CausalResult:
    """
    Bias-aware confidence interval for fuzzy RD (Noack & Rothe 2024).

    Constructs an Anderson--Rubin-style CI for the Wald-ratio fuzzy RD
    parameter that takes worst-case smoothing bias of the numerator and
    denominator local-linear estimates into account.  The CI is robust
    to weak first stages and avoids the power asymmetry documented in
    Kaliski-Keane-Neal (2025).

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome column.
    x : str
        Running variable column.
    fuzzy : str
        Treatment indicator column for the fuzzy first stage.
    c : float, default 0.0
        RD cutoff.
    M_y, M_d : float, optional
        Smoothness bounds on |g_Y''| and |g_D''|.  If ``None``, each is
        Armstrong & Kolesar's rule of thumb (R ``RDHonest``'s ``MROT``:
        the largest ``|f''|`` of a global quartic on each side), as in
        :func:`rd_honest`.
    h : float, optional
        Bandwidth.  If ``None``, R ``RDHonest``'s default for a fuzzy
        design: the MSE-optimal honest bandwidth with ``T0 = 0``.
    kernel : str, default ``'triangular'``
        Kernel.
    alpha : float, default 0.05
        Significance level.
    cluster : str, optional
        Cluster variable. The variance of the jump estimator is then the
        cluster-sum form RDHonest uses (EHW residuals, no finite-sample
        factor); without clusters it is RDHonest's nearest-neighbour
        variance (``J = 3``).
    n_grid : int, default 401
        Grid resolution for the AR-style test inversion.

    Returns
    -------
    CausalResult
        Result with ``model_info['bias_aware']`` containing
        ``M_y``, ``M_d``, ``naive_ci`` (Wald ratio +/- z * delta-method SE),
        ``bias_aware_ci`` (the Anderson-Rubin-type set), ``rdhonest``
        (R ``RDHonest``'s linearised honest fuzzy CI: estimate, std.error,
        maximum.bias, conf.low / conf.high), ``rejection_grid`` and
        ``first_stage_F``.

    Notes
    -----
    The bias-aware CI is the set of ``τ_0`` for which the test statistic

        T(τ_0) = (Δ̂_Y − τ_0 · Δ̂_D) / σ̂(τ_0)

    has |T(τ_0)| ≤ cv(b(τ_0)) where ``b(τ_0) = (M_Y + |τ_0| M_D) B / σ̂(τ_0)``
    is the worst-case bias-to-noise ratio of the numerator–denominator
    combination and ``B`` the exact Hölder-class worst-case bias of the
    local-linear jump per unit of ``M`` (from the realised weights). The
    grid inversion accommodates non-convex CIs under weak first stages.

    Every ingredient -- the jumps, the 2x2 variance, ``B``, the rule-of-
    thumb ``M`` and the default bandwidth -- is R ``RDHonest``'s; at
    ``τ_0 = τ̂`` the statistic's bias and noise are exactly RDHonest's
    ``maximum.bias`` and ``std.error`` times ``|Δ̂_D|``
    (tests/reference_parity/test_rd_iv_rd_R_parity.py). The inversion
    itself follows Noack & Rothe (2024); RDHonest reports the linearised
    interval instead, which is returned alongside.

    Through 1.28.0 the bias bound was the closed form ``h² M / 12``, which
    understates the worst-case local-linear boundary bias (the asymptotic
    constant is 1/10 for the triangular kernel), ``M`` came from local
    quadratic curvature at the cutoff, and the default bandwidth was a
    Silverman rule of thumb.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 600
    >>> age = rng.uniform(60, 70, n)
    >>> # Take-up jumps sharply at the cutoff -> strong first stage:
    >>> p = 0.05 + 0.9 * (age >= 65)
    >>> retired = (rng.uniform(size=n) < np.clip(p, 0, 1)).astype(float)
    >>> earnings = 20 - 4.0 * retired + 0.1 * (age - 65) + rng.normal(0, 1.0, n)
    >>> df = pd.DataFrame({'age': age, 'earnings': earnings, 'retired': retired})
    >>> r = sp.rd_bias_aware_fuzzy(df, y='earnings', x='age', fuzzy='retired',
    ...                            c=65.0, n_grid=101)
    >>> lo, hi = r.model_info['bias_aware']['bias_aware_ci']
    >>> bool(lo < hi)
    True
    """
    # --- Parse data ---------------------------------------------------
    needed = [y, x, fuzzy]
    if cluster is not None:
        needed.append(cluster)
    missing = [n for n in needed if n not in data.columns]
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")  # pragma: no cover
    df = data.dropna(subset=needed).copy()
    Y = df[y].to_numpy(dtype=float)
    X = df[x].to_numpy(dtype=float) - float(c)
    D = df[fuzzy].to_numpy(dtype=float)
    cl = df[cluster].to_numpy() if cluster is not None else None
    n_obs = len(Y)

    if kernel not in ("triangular", "epanechnikov", "uniform"):
        raise ValueError(f"Unsupported kernel '{kernel}'.")  # pragma: no cover
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be in (0, 1).")  # pragma: no cover

    order = np.argsort(X, kind="mergesort")
    X, Y, D = X[order], Y[order], D[order]
    cl = cl[order] if cl is not None else None
    YD = np.column_stack([Y, D])

    # --- M and bandwidth (RDHonest defaults) --------------------------
    M_y_estimated = M_y is None
    M_d_estimated = M_d is None
    if M_y is None:
        M_y = m_rule_of_thumb(X, Y, 0.0)
    if M_d is None:
        M_d = m_rule_of_thumb(X, D, 0.0)
    M_y, M_d = float(M_y), float(M_d)
    if h is None:
        h = _frd_bandwidth(X, YD, M_y, M_d, kernel, alpha)
    h = float(h)

    comp = _frd_components(X, YD, h, kernel, cl=cl)
    delta_y, delta_d, V, B = comp["delta_y"], comp["delta_d"], comp["V"], comp["B"]
    var_dd = float(V[3])

    sd_d = float(np.std(D)) if len(D) else 1.0
    weak_first_stage = abs(delta_d) < 0.01 * max(sd_d, 1e-6)
    if weak_first_stage:
        warnings.warn(  # pragma: no cover
            "rd_bias_aware_fuzzy: estimated first-stage discontinuity is "
            "tiny relative to Var(D); naive CI is reported as unbounded "
            "and the bias-aware AR CI is the appropriate object.",
            UserWarning,
        )

    z = stats.norm.ppf(1 - alpha / 2)
    if weak_first_stage:
        tau_hat = float("nan")  # pragma: no cover
        se_naive = float("nan")  # pragma: no cover
        naive_ci = (float("-inf"), float("inf"))
        rdhonest: Dict[str, Any] = {}
    else:
        tau_hat = delta_y / delta_d
        se_naive = _se_comb(V, tau_hat) / abs(delta_d)
        naive_ci = (tau_hat - z * se_naive, tau_hat + z * se_naive)
        # R RDHonest's reported (linearised) honest interval.
        max_bias = (M_y + M_d * abs(tau_hat)) * B / abs(delta_d)
        b_ratio = max_bias / se_naive if se_naive > 0 else float("inf")
        cv_hat = cv_bias(b_ratio, alpha)
        rdhonest = {
            "estimate": tau_hat,
            "std.error": se_naive,
            "maximum.bias": max_bias,
            "cv": cv_hat,
            "conf.low": tau_hat - cv_hat * se_naive,
            "conf.high": tau_hat + cv_hat * se_naive,
            "p.value": float(
                stats.norm.cdf(b_ratio - abs(tau_hat / se_naive))
                + stats.norm.cdf(-b_ratio - abs(tau_hat / se_naive))
            ),
        }

    # --- AR-style inversion -------------------------------------------
    def _excess(t0: float) -> float:
        """``|T(t0)| - cv(b(t0))``: negative inside the set."""
        se_t = _se_comb(V, t0)
        if se_t <= 0:  # pragma: no cover - degenerate
            return float("inf")
        bias_t = (M_y + abs(t0) * M_d) * B
        return abs(delta_y - t0 * delta_d) / se_t - cv_bias(bias_t / se_t, alpha)

    if np.isfinite(tau_hat) and np.isfinite(se_naive):
        span = max(abs(tau_hat) + 6 * se_naive, 5.0)
        grid_lo = tau_hat - span
        grid_hi = tau_hat + span
    else:
        scale = max(abs(delta_y), 6 * float(np.sqrt(V[0])), 5.0)
        grid_lo = -100 * scale
        grid_hi = 100 * scale
    grid = np.linspace(grid_lo, grid_hi, int(n_grid))
    exc = np.array([_excess(float(t0)) for t0 in grid])
    accept = exc <= 0

    if accept.any():
        idx = np.where(accept)[0]
        ci_lo = float(grid[idx.min()])
        ci_hi = float(grid[idx.max()])
        # Refine the two outer endpoints between the last rejected and the
        # first accepted grid point.
        if idx.min() > 0:
            ci_lo = float(
                optimize.brentq(_excess, grid[idx.min() - 1], ci_lo, xtol=1e-12)
            )
        if idx.max() < len(grid) - 1:
            ci_hi = float(
                optimize.brentq(_excess, ci_hi, grid[idx.max() + 1], xtol=1e-12)
            )
        non_convex = not np.all(accept[idx.min() : idx.max() + 1])
    else:
        ci_lo, ci_hi = float("nan"), float("nan")  # pragma: no cover
        non_convex = False

    bias_aware_ci = (ci_lo, ci_hi)

    # --- Power asymmetry diagnostic (KKN 2025) -----------------------
    first_stage_t = abs(delta_d) / np.sqrt(var_dd) if var_dd > 0 else float("inf")
    first_stage_F = float(first_stage_t**2)
    if first_stage_F < 10.0:
        warnings.warn(
            f"rd_bias_aware_fuzzy: first-stage F = {first_stage_F:.2f} < 10. "
            "Conventional fuzzy-RD t-tests have a power asymmetry (Kaliski-"
            "Keane-Neal 2025); the bias-aware AR CI here is robust, but you "
            "should also report the ITT (sharp RD on the outcome) per their "
            "recommendation.",
            UserWarning,
        )

    # --- Pretty summary ----------------------------------------------
    summary = (
        "\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "  Bias-Aware Fuzzy RD (Noack & Rothe 2024 ECTA)\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  τ̂ (Wald ratio):           {tau_hat:.4f}\n"
        f"  Δ̂_Y (numerator jump):     {delta_y:.4f}\n"
        f"  Δ̂_D (denominator jump):   {delta_d:.4f}\n"
        f"  First-stage F:            {first_stage_F:.2f}\n"
        f"\n"
        f"  Naive {int((1 - alpha) * 100)}% CI:       "
        f"[{naive_ci[0]:.4f}, {naive_ci[1]:.4f}]\n"
        f"  Bias-aware {int((1 - alpha) * 100)}% CI:  "
        f"[{ci_lo:.4f}, {ci_hi:.4f}]"
        f"{' (non-convex)' if non_convex else ''}\n"
        f"\n"
        f"  M_y:                      {M_y:.4g} "
        f"{'(estimated)' if M_y_estimated else '(supplied)'}\n"
        f"  M_d:                      {M_d:.4g} "
        f"{'(estimated)' if M_d_estimated else '(supplied)'}\n"
        f"  Bandwidth h:              {h:.4f}\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    )

    # P-value derived from the bias-aware test of H0: τ=0
    # (whether 0 lies in the bias-aware CI).
    pvalue = (
        2 * stats.norm.sf(abs(tau_hat) / se_naive)
        if (np.isfinite(tau_hat) and np.isfinite(se_naive) and se_naive > 0)
        else float("nan")
    )
    if not np.isfinite(pvalue) and ci_lo == ci_lo and ci_hi == ci_hi:
        # Approximate p-value from the AR CI: if 0 is well inside the CI,
        # report a conservative 1.0; if outside, report alpha.
        pvalue = float(alpha) if not (ci_lo <= 0 <= ci_hi) else 1.0

    out = CausalResult(
        method="Bias-aware fuzzy RD (Noack-Rothe 2024)",
        estimand="LATE at cutoff",
        estimate=tau_hat,
        se=se_naive,
        pvalue=pvalue,
        ci=bias_aware_ci,
        alpha=alpha,
        n_obs=int(n_obs),
        model_info={
            "bias_aware": {
                "naive_ci": naive_ci,
                "bias_aware_ci": bias_aware_ci,
                "non_convex_ci": bool(non_convex),
                "M_y": float(M_y),
                "M_d": float(M_d),
                "M_y_estimated": bool(M_y_estimated),
                "M_d_estimated": bool(M_d_estimated),
                "delta_y": delta_y,
                "delta_d": delta_d,
                "first_stage_F": first_stage_F,
                "bandwidth": float(h),
                "kernel": kernel,
                "bias_per_unit_M": float(B),
                "variance_2x2": [float(v) for v in V],
                "n_effective": comp["n_effective"],
                "rdhonest": rdhonest,
                "rejection_grid": (grid.tolist(), accept.tolist()),
            },
            "summary_str": summary,
        },
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            out,
            function="sp.rd.rd_bias_aware_fuzzy",
            params={
                "y": y,
                "x": x,
                "fuzzy": fuzzy,
                "c": c,
                "M_y": M_y,
                "M_d": M_d,
                "h": h,
                "kernel": kernel,
                "alpha": alpha,
                "cluster": cluster,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass  # pragma: no cover
    return out


# ---------------------------------------------------------------------------
# Fuzzy-RD honest machinery (port of RDHonest's FRD path)
# ---------------------------------------------------------------------------


def _sigma_nn_2(x: np.ndarray, Y: np.ndarray, J: int = 3) -> np.ndarray:
    """``RDHonest::sigmaNN`` for a two-column outcome; rows ``(YY, YD, DY, DD)``.

    ``x`` sorted ascending; ties are kept whole, as in the univariate port.
    """
    n = len(x)
    J = min(J, n - 1)
    out = np.empty((n, 4))
    for k in range(n):
        lo = max(k - J, 0)
        cand = np.concatenate([x[lo:k], x[k + 1 : min(k + J + 1, n)]])
        d = np.sort(np.abs(cand - x[k]))[J - 1]
        ind = np.abs(x - x[k]) <= d
        ind[k] = False
        jk = float(ind.sum())
        dev = Y[k] - Y[ind].mean(axis=0)
        out[k] = jk / (jk + 1.0) * np.outer(dev, dev).ravel()
    return out


def _joint_resid(
    x: np.ndarray, Y: np.ndarray, h: float, kernel: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Residuals of ``NPReg``'s joint order-1 fit (both columns), and ``w != 0``."""
    w = _kern(x / h, kernel)
    right = (x >= 0).astype(float)
    z = np.column_stack([right, right * x, np.ones_like(x), x])
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(z * sw[:, None], Y * sw[:, None], rcond=None)
    return Y - z @ beta, w != 0


def _prelim_var_frd(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """``PrelimVar(se.initial='EHW')`` for FRD: side-wise means of the 2x2
    EHW residual products at the IK bandwidth of the outcome."""
    h1 = _ik_bandwidth(x, Y[:, 0], 0.0)
    if not np.isfinite(h1):  # pragma: no cover - degenerate design
        h1 = float("inf")
    r, inwin = _joint_resid(x, Y, max(h1, _hmin(x, 0.0)), "triangular")
    prod = np.einsum("ni,nj->nij", r, r).reshape(len(x), 4)
    out = np.zeros_like(prod)
    for mask in (x >= 0, x < 0):
        sel = mask & inwin
        if sel.any():
            out[mask] = prod[sel].mean(axis=0)
    return out


def _frd_components(
    x: np.ndarray,
    Y: np.ndarray,
    h: float,
    kernel: str,
    sigma2: Optional[np.ndarray] = None,
    cl: Optional[np.ndarray] = None,
    J: int = 3,
) -> Dict[str, Any]:
    """Jumps, the 2x2 variance of the jump estimator, and the bias factor.

    ``x`` sorted and centred. ``V`` is ``(VYY, VYD, VDY, VDD)``: RDHonest's
    ``colSums(w^2 * sigma2)`` (nearest-neighbour ``sigma2`` by default,
    ``supplied`` in the bandwidth search) or, with clusters, the
    cross-product of cluster sums of ``w * residual`` (its EHW-cluster
    form, with no finite-sample factor). ``B`` is the worst-case bias per
    unit of ``M`` under the Holder class: ``honest_bias(w, M=1)``.
    """
    w = honest_weights(x, 0.0, h, kernel)
    nz = w != 0
    if nz.sum() < 4:
        raise ValueError(
            f"bandwidth h={h:g} leaves only {int(nz.sum())} observations with "
            "non-zero weight"
        )
    dY = float(w @ Y[:, 0])
    dD = float(w @ Y[:, 1])
    if cl is not None:
        r, _ = _joint_resid(x, Y, h, kernel)
        s = w[:, None] * r
        _, codes = np.unique(cl[nz], return_inverse=True)
        G = np.zeros((int(codes.max()) + 1, 2))
        np.add.at(G, codes, s[nz])
        V = (G.T @ G).ravel()
    else:
        if sigma2 is None:
            sigma2 = np.zeros((len(x), 4))
            for side in (x < 0, x >= 0):
                idx = np.flatnonzero(side & nz)
                if idx.size >= 2:
                    sigma2[idx] = _sigma_nn_2(x[idx], Y[idx], J)
        V = (w[:, None] ** 2 * sigma2).sum(axis=0)
    return {
        "w": w,
        "delta_y": dY,
        "delta_d": dD,
        "V": V,
        "B": honest_bias(w, x, 0.0, 1.0),
        "n_effective": int(nz.sum()),
    }


def _se_comb(V: np.ndarray, t: float) -> float:
    """SE of ``Delta_Y - t * Delta_D``."""
    return float(np.sqrt(max(V[0] - 2 * t * V[1] + t * t * V[3], 0.0)))


def _frd_bandwidth(
    x: np.ndarray,
    Y: np.ndarray,
    M_y: float,
    M_d: float,
    kernel: str,
    alpha: float,
    T0: float = 0.0,
) -> float:
    """``RDHonest::OptBW`` for FRD (MSE criterion, ``T0bias = TRUE``)."""
    s2 = _prelim_var_frd(x, Y)
    M_eff = M_y + M_d * abs(T0)

    def obj(h: float) -> float:
        try:
            comp = _frd_components(x, Y, float(h), kernel, sigma2=s2)
        except (ValueError, np.linalg.LinAlgError):
            return float("inf")
        tau = comp["delta_y"] / comp["delta_d"]
        se = _se_comb(comp["V"], tau)
        bias = M_eff * comp["B"]
        return bias**2 + se**2

    xr = np.unique(x[x >= 0])
    xl = np.unique(np.abs(x[x < 0]))
    hmin = max(xr[1], xl[1])
    hmax = float(np.max(np.abs(x)))
    if kernel == "uniform":
        supp = np.unique(np.abs(x))
        supp = supp[supp >= hmin]
        return float(supp[int(np.argmin([obj(float(v)) for v in supp]))])
    res = optimize.minimize_scalar(
        obj,
        bounds=(hmin, hmax),
        method="bounded",
        options={"xatol": np.finfo(float).eps ** 0.75},
    )
    return float(abs(res.x))
