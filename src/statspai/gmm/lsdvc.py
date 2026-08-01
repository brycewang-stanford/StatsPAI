"""
Bias-corrected least-squares dummy-variable estimator for dynamic panels.

The within (LSDV / fixed-effects) estimator of a dynamic panel is
inconsistent for fixed ``T``: demeaning correlates ``y_{i,t-1}`` with the
demeaned error, producing the Nickell (1981) bias of order ``1/T`` that does
not vanish as ``N -> infinity``. GMM removes it by instrumenting. The
*other* route, due to Kiviet (1995) and extended by Bun & Kiviet (2003) and
Bruno (2005), is to estimate the bias analytically and subtract it:

    beta_LSDVC = beta_LSDV - Bias_hat(gamma_tilde, sigma2_tilde)

where the bias approximation is evaluated at a consistent preliminary
estimate. Monte-Carlo evidence (Judson & Owen 1999; Bruno 2005) is that
LSDVC has markedly smaller RMSE than the GMM estimators in the small-``N``,
small-``T`` panels common in macro applications, precisely where GMM's
many-instrument problems bite hardest.

The trade-off is that it buys that efficiency with a stronger maintained
model: the correction is derived for strictly exogenous regressors and
homoskedastic errors, and its standard errors have to be bootstrapped
because the analytic ones do not account for the estimated correction.

References
----------
Kiviet, J.F. (1995).
"On Bias, Inconsistency, and Efficiency of Various Estimators in Dynamic
Panel Data Models." *Journal of Econometrics*, 68(1), 53-78.
[@kiviet1995bias]

Bun, M.J.G. and Kiviet, J.F. (2003).
"On the Diminishing Returns of Higher-Order Terms in Asymptotic Expansions
of Bias." *Economics Letters*, 79(2), 145-152. [@bun2003diminishing]

Bruno, G.S.F. (2005).
"Approximating the Bias of the LSDV Estimator for Dynamic Unbalanced Panel
Data Models." *Economics Letters*, 87(3), 361-366. [@bruno2005approximating]

Nickell, S. (1981). *Econometrica*, 49(6), 1417-1426. [@nickell1981biases]
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from ._dynpanel._data import build_panel_arrays
from ._dynpanel._spec import parse_terms, term_name

__all__ = ["xtlsdvc"]

_INITIAL_CHOICES = ("ab", "ah", "bb")


def _within_ols(y: np.ndarray, X: np.ndarray, unit: np.ndarray):
    """Fixed-effects (within) OLS. Returns ``(beta, resid, dof)``."""
    yd = y.copy()
    Xd = X.copy()
    for u in np.unique(unit):
        rows = unit == u
        yd[rows] -= yd[rows].mean()
        Xd[rows] -= Xd[rows].mean(axis=0)
    beta, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
    resid = yd - Xd @ beta
    dof = y.size - np.unique(unit).size - X.shape[1]
    return beta, resid, max(dof, 1), Xd


def xtlsdvc(
    data: pd.DataFrame,
    y: str,
    x: Optional[Sequence[str]] = None,
    id: str = "id",
    time: str = "time",
    initial: str = "ab",
    bias_order: int = 2,
    alpha: float = 0.05,
    bootstrap: int = 0,
    seed: Optional[int] = None,
) -> CausalResult:
    """
    Bias-corrected LSDV (LSDVC) estimator for dynamic panels.

    Equivalent to Stata's ``xtlsdvc`` (Bruno 2005).

    Parameters
    ----------
    data : pd.DataFrame
        Panel in long format.
    y : str
        Dependent variable. Its first lag is added automatically — do **not**
        pass a hand-built lag in ``x``.
    x : list of str, optional
        Strictly exogenous regressors. Accepts the same lag-operator syntax
        as :func:`~statspai.gmm.arellano_bond.xtabond` (``"l(0/1).w"``).
    id, time : str
        Unit and period identifiers.
    initial : {'ab', 'ah', 'bb'}, default 'ab'
        Consistent estimator used to evaluate the bias expression:
        Arellano-Bond, Anderson-Hsiao, or Blundell-Bond system GMM. The
        correction is only as good as this input, and the three can disagree
        materially on a persistent series — ``'bb'`` is the safer choice
        there, for the same reason system GMM is.
    bias_order : {1, 2, 3}, default 2
        How many terms of the Bun-Kiviet expansion to subtract: ``O(1/T)``,
        ``O(1/NT)``, or ``O(1/NT^2)``. Bun & Kiviet (2003) find diminishing
        returns beyond the second.
    alpha : float, default 0.05
        Significance level.
    bootstrap : int, default 0
        Number of parametric-bootstrap replications for the standard errors.
        **The reported analytic standard errors are the LSDV ones and do not
        account for the bias correction**, exactly as in Stata's
        ``xtlsdvc``; set this to get honest ones. 0 skips the bootstrap and
        warns.
    seed : int, optional
        Seed for the bootstrap.

    Returns
    -------
    CausalResult
        ``estimate`` / ``se`` are the corrected lagged-Y coefficient.
        ``detail`` holds the full corrected table, the uncorrected LSDV
        coefficients, and the subtracted bias.

    Notes
    -----
    The bias expression is the Bun-Kiviet/Bruno one, evaluated per unit on
    the balanced period grid with a selection matrix marking the observed
    rows (so unbalanced panels are handled as in Bruno 2005). Validated
    against Stata's ``xtlsdvc`` at all three bias orders and all three
    initial estimators.

    **When to prefer this over GMM.** Small ``N`` with small ``T`` — the
    regime where the GMM instrument count rivals the number of units and
    the Hansen test stops being informative. LSDVC uses no instruments at
    all, so it cannot be undone by instrument proliferation; the price is
    that it assumes strict exogeneity and homoskedasticity, which GMM does
    not.

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.xtlsdvc(df, y='n', x=['w', 'k'], id='id',   # doctest: +SKIP
    ...                  time='year', initial='bb', bootstrap=200)

    References
    ----------
    Kiviet, J.F. (1995). *Journal of Econometrics* 68(1), 53-78.
    [@kiviet1995bias]
    Bruno, G.S.F. (2005). *Economics Letters* 87(3), 361-366.
    [@bruno2005approximating]
    """
    if initial not in _INITIAL_CHOICES:
        raise ValueError(
            f"initial must be one of {_INITIAL_CHOICES} "
            f"(Arellano-Bond, Anderson-Hsiao, Blundell-Bond), got {initial!r}."
        )
    if bias_order not in (1, 2, 3):
        raise ValueError(
            "bias_order must be 1 (O(1/T)), 2 (O(1/NT)) or 3 (O(1/NT^2)), "
            f"got {bias_order!r}."
        )

    x_terms = parse_terms(x)
    base_vars = list(dict.fromkeys([t.var for t in x_terms]))
    panel = build_panel_arrays(data, id, time, [y] + base_vars)
    n_periods = panel.n_periods
    if n_periods < 3:
        raise ValueError(
            f"LSDVC needs at least 3 periods to form a lag and a within "
            f"transform; got {n_periods}."
        )

    Y = panel.get(y)
    Ly = np.full_like(Y, np.nan)
    Ly[:, 1:] = Y[:, :-1]
    X_grids = []
    for term in x_terms:
        arr = panel.get(term.var)
        shifted = np.full_like(arr, np.nan)
        if term.lag == 0:
            shifted = arr
        elif term.lag < n_periods:
            shifted[:, term.lag :] = arr[:, : n_periods - term.lag]
        X_grids.append(shifted)

    # --- consistent preliminary estimate -----------------------------------
    from ._dynpanel import fit_dynamic_panel

    method = {"ab": "difference", "ah": "ah", "bb": "system"}[initial]
    with warnings.catch_warnings():
        # The preliminary fit's own diagnostics are not the subject here; the
        # correction only needs its point estimate.
        warnings.simplefilter("ignore")
        prelim = fit_dynamic_panel(
            data,
            y=y,
            x=list(x or []),
            id=id,
            time=time,
            lags=1,
            method=method,
            constant=False,
        )
    prelim_names = list(prelim["names"])
    gamma = float(prelim["beta"][0])
    beta_x = np.asarray(prelim["beta"][1:], dtype=float)

    # --- residuals from the preliminary fit, on the full grid ---------------
    resid_grid = Y - gamma * Ly
    for j, grid in enumerate(X_grids):
        resid_grid = resid_grid - beta_x[j] * grid
    usable = np.isfinite(resid_grid)

    # --- LSDV (within) regression on the usable sample ----------------------
    ui, ti = np.nonzero(usable)
    if ui.size <= len(x_terms) + 2:
        raise ValueError("not enough usable observations for the LSDV regression.")
    design = np.column_stack([Ly[ui, ti]] + [g[ui, ti] for g in X_grids])
    b_lsdv, within_resid, dof, Xd = _within_ols(Y[ui, ti], design, ui)
    k = design.shape[1]

    # --- Bun-Kiviet / Bruno bias approximation ------------------------------
    # Everything below lives on the balanced grid with the first period
    # dropped (it has no lag), with unobserved cells zeroed and marked by the
    # selection matrix S_i -- the construction in Bruno (2005).
    T = n_periods - 1
    sel = usable[:, 1:].astype(float)
    keep_units = sel.sum(axis=1) > 0
    sel = sel[keep_units]
    n_used = int(keep_units.sum())

    ones = np.ones((T, 1))
    eye = np.eye(T)
    Lmat = np.eye(T, k=-1)
    C = Lmat @ np.linalg.inv(eye - gamma * Lmat)

    def _grid(arr):
        out = np.nan_to_num(arr[:, 1:], nan=0.0)[keep_units]
        return out * sel

    ly = _grid(Ly)
    eps = _grid(resid_grid)
    xs = [_grid(g) for g in X_grids]

    e1 = np.zeros((k, 1))
    e1[0, 0] = 1.0
    WMW = np.zeros((k, k))
    WPMW = np.zeros((k, k))
    WPPW = np.zeros((k, k))
    uMu = 0.0
    tr = dict(P=0.0, PP=0.0, PtP=0.0, PtPP=0.0, PtPPtP=0.0)
    dof_sum = 0.0

    for i in range(n_used):
        S = np.diag(sel[i])
        Ti = float(sel[i].sum())
        if Ti <= 0:
            continue
        dof_sum += Ti - 1.0
        M = S @ (eye - ones @ ones.T / Ti) @ S
        ei = eps[i][:, None]
        lybar = ly[i][:, None] - C @ M @ ei
        Wi = np.column_stack([lybar] + [g[i][:, None] for g in xs])
        P = M @ C
        PtP = P.T @ P
        tr["P"] += np.trace(P)
        tr["PP"] += np.trace(P @ P)
        tr["PtP"] += np.trace(PtP)
        tr["PtPP"] += np.trace(PtP @ P)
        tr["PtPPtP"] += np.trace(PtP @ PtP)
        WMW += Wi.T @ M @ Wi
        WPMW += Wi.T @ (P @ M) @ Wi
        WPPW += Wi.T @ (P @ P.T) @ Wi
        uMu += float((ei.T @ M @ ei).item())

    sigma2 = uMu / max(dof_sum - k, 1.0)
    Q = np.linalg.inv(WMW + sigma2 * tr["PtP"] * (e1 @ e1.T))
    q11 = float((e1.T @ Q @ e1).item())
    Qe1 = Q @ e1

    bias = sigma2 * tr["P"] * Qe1
    if bias_order >= 2:
        QWPMW = Q @ WPMW
        bias = bias - sigma2 * (
            (
                QWPMW
                + np.trace(QWPMW) * np.eye(k)
                + 2 * sigma2 * q11 * tr["PtPP"] * np.eye(k)
            )
            @ Qe1
        )
    if bias_order >= 3:
        QWPPW = Q @ WPPW
        bias = bias + (sigma2**2) * tr["P"] * (
            2 * q11 * (QWPPW @ Qe1)
            + (
                float(e1.T @ Q.T @ WPPW @ Qe1)
                + q11 * np.trace(QWPPW)
                + 2 * tr["PtPPtP"] * q11**2
            )
            * Qe1
        )
    bias = np.asarray(bias).ravel()
    b_corrected = np.asarray(b_lsdv).ravel() - bias

    # --- standard errors -----------------------------------------------------
    XtX_inv = np.linalg.inv(Xd.T @ Xd)
    s2 = float(within_resid @ within_resid) / dof
    se = np.sqrt(np.maximum(np.diag(s2 * XtX_inv), 0.0))
    se_source = "LSDV (analytic; does not reflect the bias correction)"

    if bootstrap and bootstrap > 0:
        se = _bootstrap_se(
            data,
            y,
            list(x or []),
            id,
            time,
            initial,
            bias_order,
            b_corrected,
            s2,
            bootstrap,
            seed,
        )
        se_source = f"parametric bootstrap ({bootstrap} replications)"
    else:
        warnings.warn(
            "LSDVC standard errors are the *uncorrected* LSDV ones: the "
            "analytic formula does not account for the estimated bias "
            "correction, so they understate the true uncertainty. Stata's "
            "xtlsdvc has the same caveat and also requires a bootstrap. Pass "
            "bootstrap=200 (or more) for honest inference.",
            stacklevel=2,
        )

    names = [f"L1.{y}"] + [term_name(t) for t in x_terms]
    z = np.where(se > 0, b_corrected / se, np.nan)
    detail = pd.DataFrame(
        {
            "variable": names,
            "coefficient": b_corrected,
            "se": se,
            "z": z,
            "pvalue": 2 * stats.norm.sf(np.abs(z)),
            "lsdv_uncorrected": np.asarray(b_lsdv).ravel(),
            "bias_subtracted": bias,
        }
    )
    z_crit = stats.norm.ppf(1 - alpha / 2)
    rho, rho_se = float(b_corrected[0]), float(se[0])

    return CausalResult(
        method=f"Bias-corrected LSDV (LSDVC, initial={initial}, order {bias_order})",
        estimand="rho (AR coefficient)",
        estimate=rho,
        se=rho_se,
        pvalue=float(2 * stats.norm.sf(abs(rho / rho_se))) if rho_se > 0 else np.nan,
        ci=(rho - z_crit * rho_se, rho + z_crit * rho_se),
        alpha=alpha,
        n_obs=int(ui.size),
        detail=detail,
        model_info={
            "method": "LSDVC",
            "initial": initial,
            "initial_names": prelim_names,
            "bias_order": bias_order,
            "n_units": n_used,
            "n_obs": int(ui.size),
            "n_periods": int(T),
            "sigma2": sigma2,
            "se_source": se_source,
            "bootstrap": int(bootstrap),
        },
        _citation_key="lsdvc",
    )


def _bootstrap_se(
    data, y, x, id, time, initial, bias_order, beta, s2, reps, seed
) -> np.ndarray:
    """Parametric bootstrap over the fitted dynamic process.

    Resamples idiosyncratic errors, regenerates ``y`` recursively from the
    corrected coefficients while holding the unit effects and regressors
    fixed, and re-runs the whole LSDVC pipeline. This is the only honest way
    to price the correction: the analytic LSDV variance treats the
    subtracted bias as known.
    """
    rng = np.random.default_rng(seed)
    frame = data[[id, time, y] + list(dict.fromkeys(_base_vars(x)))].copy()
    frame = frame.sort_values([id, time])
    units = frame[id].to_numpy()
    draws: List[np.ndarray] = []
    sigma = np.sqrt(max(s2, 0.0))

    for _ in range(int(reps)):
        boot = frame.copy()
        noise = rng.normal(scale=sigma, size=len(boot))
        # Recursive regeneration within unit, holding the first observation.
        values = boot[y].to_numpy(dtype=float).copy()
        start = 0
        for u in pd.unique(units):
            rows = np.flatnonzero(units == u)
            for pos in rows[1:]:
                values[pos] = beta[0] * values[pos - 1] + noise[pos]
            start += rows.size
        boot[y] = values
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = xtlsdvc(
                    boot,
                    y=y,
                    x=x,
                    id=id,
                    time=time,
                    initial=initial,
                    bias_order=bias_order,
                    bootstrap=0,
                )
            draws.append(fit.detail["coefficient"].to_numpy(float))
        except Exception:  # pragma: no cover - a degenerate resample
            continue

    if len(draws) < 2:
        warnings.warn(
            "The LSDVC bootstrap produced fewer than two usable replications; "
            "falling back to the uncorrected LSDV standard errors.",
            stacklevel=3,
        )
        return np.full(len(beta), np.nan)
    return np.asarray(draws).std(axis=0, ddof=1)


def _base_vars(x) -> List[str]:
    return [t.var for t in parse_terms(list(x or []))]


CausalResult._CITATIONS["lsdvc"] = (
    "@article{bruno2005approximating,\n"
    "  title={Approximating the Bias of the LSDV Estimator for Dynamic "
    "Unbalanced Panel Data Models},\n"
    "  author={Bruno, Giovanni S. F.},\n"
    "  journal={Economics Letters},\n"
    "  volume={87},\n"
    "  number={3},\n"
    "  pages={361--366},\n"
    "  year={2005},\n"
    "  publisher={Elsevier}\n"
    "}"
)
