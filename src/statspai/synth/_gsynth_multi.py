"""Generalized synthetic control with several treated units and covariates.

Xu (2017): an interactive fixed effects model is estimated on the
never-treated units,

    Y_it = X_it' beta + mu + alpha_i + xi_t + lambda_i' f_t + e_it,

each treated unit's intercept and loadings are then estimated from its own
pre-treatment periods given ``beta``, ``xi`` and the factors, and its
untreated outcome after treatment is the model's prediction.

The control-group fit alternates between least squares for ``beta`` and,
for the residual matrix, the two-way within transformation followed by a
rank-``r`` truncation. Given ``beta`` that pair is the exact minimiser over
additive effects and a rank-``r`` component, so the iteration is
alternating least squares on the Bai (2009) objective.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from ..exceptions import ConvergenceWarning, DataInsufficient, MethodIncompatibility


def _ife_fit(
    Y: np.ndarray,
    X: Optional[np.ndarray],
    r: int,
    tol: float = 1e-9,
    max_iter: int = 5000,
) -> Dict[str, Any]:
    """Interactive fixed effects on a balanced ``N x T`` panel.

    ``X`` is ``N x T x p`` or ``None``. Returns ``beta``, the grand mean
    ``mu``, unit effects ``alpha``, period effects ``xi``, the factors
    ``F`` (``T x r``, orthonormal columns), loadings ``L`` (``N x r``) and
    the fitted matrix.
    """
    N, T = Y.shape
    p = 0 if X is None else X.shape[2]
    r = int(max(0, min(r, min(N, T) - 1)))

    def decompose(E: np.ndarray) -> Tuple[Any, ...]:
        mu = float(E.mean())
        row = E.mean(axis=1)
        col = E.mean(axis=0)
        dm = E - row[:, None] - col[None, :] + mu
        if r > 0:
            U, s, Vt = np.linalg.svd(dm, full_matrices=False)
            F = Vt[:r].T
            L = U[:, :r] * s[:r]
            low = L @ F.T
        else:
            F = np.empty((T, 0))
            L = np.empty((N, 0))
            low = np.zeros_like(E)
        return mu, row - mu, col - mu, F, L, low

    beta = np.zeros(p)
    converged = True
    n_iter = 0
    if p:
        assert X is not None
        Xf = X.reshape(N * T, p)
        # Start from the two-way within estimator.
        Xd = X - X.mean(axis=1, keepdims=True) - X.mean(axis=0, keepdims=True)
        Xd = (Xd + X.mean(axis=(0, 1), keepdims=True)).reshape(N * T, p)
        Yd = (Y - Y.mean(axis=1, keepdims=True) - Y.mean(axis=0, keepdims=True)) + (
            Y.mean()
        )
        beta = np.linalg.lstsq(Xd, Yd.ravel(), rcond=None)[0]
        XtX = Xf.T @ Xf
        converged = False
        scale = max(float(np.abs(Y).max()), 1.0)
        for n_iter in range(1, max_iter + 1):
            E = Y - X @ beta
            mu, alpha, xi, F, L, low = decompose(E)
            fe_fit = mu + alpha[:, None] + xi[None, :] + low
            new = np.linalg.solve(XtX, Xf.T @ (Y - fe_fit).ravel())
            step = float(np.max(np.abs(X @ (new - beta)))) / scale
            beta = new
            if step < tol:
                converged = True
                break
    E = Y - (X @ beta if p else 0.0)
    mu, alpha, xi, F, L, low = decompose(E)
    fitted = (X @ beta if p else 0.0) + mu + alpha[:, None] + xi[None, :] + low
    return {
        "beta": beta,
        "mu": mu,
        "alpha": alpha,
        "xi": xi,
        "F": F,
        "L": L,
        "fitted": fitted,
        "resid": Y - fitted,
        "r": r,
        "converged": converged,
        "n_iter": n_iter,
    }


def _project_treated(
    Y_tr: np.ndarray,
    X_tr: Optional[np.ndarray],
    pre: np.ndarray,
    fit: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Untreated-outcome predictions for treated units.

    ``pre`` is a boolean ``N_tr x T`` mask of the periods used to estimate
    each unit's intercept and loadings. Returns the ``N_tr x T`` matrix of
    predictions, the intercepts and the loadings.
    """
    n_tr, T = Y_tr.shape
    base = np.broadcast_to(fit["mu"] + fit["xi"][None, :], (n_tr, T)).copy()
    if X_tr is not None and X_tr.shape[2]:
        base = base + X_tr @ fit["beta"]
    F = fit["F"]
    r = F.shape[1]
    design = np.column_stack([np.ones(T), F])
    alpha = np.empty(n_tr)
    lam = np.empty((n_tr, r))
    for i in range(n_tr):
        rows = pre[i]
        coef = np.linalg.lstsq(design[rows], (Y_tr[i] - base[i])[rows], rcond=None)[0]
        alpha[i] = coef[0]
        lam[i] = coef[1:]
    return base + alpha[:, None] + lam @ F.T, alpha, lam


def _loo_mspe(
    Y_tr: np.ndarray,
    X_tr: Optional[np.ndarray],
    pre: np.ndarray,
    fit: Dict[str, Any],
) -> float:
    """Leave-one-pre-period-out prediction error of the treated units."""
    n_tr, T = Y_tr.shape
    base = np.broadcast_to(fit["mu"] + fit["xi"][None, :], (n_tr, T)).copy()
    if X_tr is not None and X_tr.shape[2]:
        base = base + X_tr @ fit["beta"]
    design = np.column_stack([np.ones(T), fit["F"]])
    sq: List[float] = []
    for i in range(n_tr):
        rows = np.flatnonzero(pre[i])
        target = Y_tr[i] - base[i]
        A = design[rows]
        # Leave-one-out residuals of a linear fit from its hat matrix.
        Q, _ = np.linalg.qr(A)
        h = np.sum(Q**2, axis=1)
        res = target[rows] - Q @ (Q.T @ target[rows])
        if np.any(h > 1 - 1e-10):
            return float("inf")
        sq.extend(((res / (1.0 - h)) ** 2).tolist())
    return float(np.mean(sq))


def _att(
    Y_tr: np.ndarray, Y0_hat: np.ndarray, post: np.ndarray, rel: np.ndarray
) -> Tuple[float, pd.Series]:
    eff = Y_tr - Y0_hat
    avg = float(eff[post].mean())
    by = pd.Series(eff[post], index=rel[post]).groupby(level=0).mean().sort_index()
    return avg, by


def gsynth_multi(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treat: str,
    covariates: Optional[Sequence[str]] = None,
    n_factors: Optional[int] = None,
    max_factors: int = 5,
    inference: str = "auto",
    n_boot: int = 200,
    min_T0: int = 5,
    seed: Optional[int] = None,
    alpha: float = 0.05,
    tol: float = 1e-9,
) -> CausalResult:
    """Generalized synthetic control for a treatment indicator column."""
    covariates = list(covariates or [])
    cols = [outcome, unit, time, treat, *covariates]
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"gsynth: column(s) not in data: {missing}.",
            diagnostics={"missing_columns": missing},
        )
    if inference not in ("auto", "parametric", "nonparametric", "none"):
        raise MethodIncompatibility(
            f"gsynth: unknown inference {inference!r}.",
            recovery_hint="Use 'auto', 'parametric', 'nonparametric' or 'none'.",
        )
    df = data[cols]
    if df.duplicated([unit, time]).any():
        raise MethodIncompatibility(
            "gsynth: duplicate (unit, time) rows.",
            recovery_hint="Keep one row per unit and period.",
        )
    units = np.array(sorted(df[unit].unique()))
    times = np.array(sorted(df[time].unique()))
    N, T = len(units), len(times)
    wide = df.set_index([unit, time])
    full = pd.MultiIndex.from_product([units, times], names=[unit, time])
    wide = wide.reindex(full)
    Y = wide[outcome].to_numpy(dtype=float).reshape(N, T)
    D = wide[treat].to_numpy(dtype=float).reshape(N, T)
    X = (
        np.stack(
            [wide[c].to_numpy(dtype=float).reshape(N, T) for c in covariates], axis=2
        )
        if covariates
        else None
    )
    if np.isnan(Y).any() or np.isnan(D).any() or (X is not None and np.isnan(X).any()):
        raise MethodIncompatibility(
            "gsynth with treat= needs a balanced panel without missing "
            "outcomes, treatment or covariates.",
            recovery_hint="Drop incomplete units, or use sp.fect, which "
            "handles unbalanced panels.",
            alternative_functions=["sp.fect"],
        )
    if not np.isin(np.unique(D), (0.0, 1.0)).all():
        raise MethodIncompatibility(
            "gsynth: treat must be a 0/1 indicator.",
            recovery_hint="Code treat as 1 in treated unit-periods.",
        )
    if np.any(np.diff(D, axis=1) < 0):
        raise MethodIncompatibility(
            "gsynth: treatment switches off for some unit. The method "
            "takes treatment as absorbing.",
            recovery_hint="Use sp.fect, which allows treatment reversals.",
            alternative_functions=["sp.fect"],
        )
    ever = D.max(axis=1) == 1
    T0 = (D == 0).sum(axis=1)  # number of pre-treatment periods
    if (ever & (T0 == 0)).any():
        raise DataInsufficient(
            "gsynth: a unit is treated in every period.",
            recovery_hint="Drop always-treated units.",
        )
    short = ever & (T0 < int(min_T0))
    if short.any():
        warnings.warn(
            f"gsynth: dropped {int(short.sum())} treated unit(s) with fewer "
            f"than min_T0={int(min_T0)} pre-treatment periods.",
            RuntimeWarning,
            stacklevel=3,
        )
    tr = np.flatnonzero(ever & ~short)
    co = np.flatnonzero(~ever)
    n_tr, n_co = len(tr), len(co)
    if n_tr == 0:
        raise DataInsufficient(
            "gsynth: no treated unit with enough pre-treatment periods.",
            recovery_hint="Lower min_T0 or check the treat column.",
        )
    if n_co < 3:
        raise DataInsufficient(
            f"gsynth: {n_co} never-treated unit(s); at least 3 are needed.",
            recovery_hint="The factors are estimated from never-treated units.",
        )
    Y_co, Y_tr = Y[co], Y[tr]
    X_co = None if X is None else X[co]
    X_tr = None if X is None else X[tr]
    pre = D[tr] == 0
    post = ~pre
    T0_min = int(T0[tr].min())
    rel = np.arange(T)[None, :] - T0[tr][:, None]  # 0 = first treated period

    # --- number of factors ---
    cv_table: Optional[pd.DataFrame] = None
    r_max = int(max(0, min(int(max_factors), T0_min - 2, n_co - 1, T - 1)))
    if n_factors is None:
        rows = []
        fits = {}
        for r in range(0, r_max + 1):
            fits[r] = _ife_fit(Y_co, X_co, r, tol=tol)
            rows.append({"n_factors": r, "mspe": _loo_mspe(Y_tr, X_tr, pre, fits[r])})
        cv_table = pd.DataFrame(rows)
        r_use = int(cv_table.loc[cv_table["mspe"].idxmin(), "n_factors"])
        fit = fits[r_use]
    else:
        r_use = int(n_factors)
        if r_use < 0 or r_use > T0_min - 2:
            raise MethodIncompatibility(
                f"gsynth: n_factors={r_use} needs at least {r_use + 2} "
                f"pre-treatment periods for every treated unit; the "
                f"shortest has {T0_min}.",
                recovery_hint="Use fewer factors or raise min_T0.",
            )
        fit = _ife_fit(Y_co, X_co, r_use, tol=tol)
    if not fit["converged"]:
        warnings.warn(
            "gsynth: the interactive fixed effects iteration did not "
            "converge on the control group. This happens when a covariate is "
            "close to a unit effect times a period effect, which the factors "
            "can absorb: its coefficient is then weakly identified. Check "
            "model_info['beta'], or drop the covariate.",
            ConvergenceWarning,
            stacklevel=3,
        )
    Y0_hat, alpha_tr, lam_tr = _project_treated(Y_tr, X_tr, pre, fit)
    att, att_rel = _att(Y_tr, Y0_hat, post, rel)
    eff = Y_tr - Y0_hat
    pre_rmse = float(np.sqrt(np.mean(eff[pre] ** 2)))

    # --- inference ---
    mode = inference
    if mode == "auto":
        mode = "parametric" if n_tr < 40 else "nonparametric"
    rng = np.random.default_rng(seed)
    boot_avg: List[float] = []
    boot_rel: List[pd.Series] = []
    if mode == "parametric":
        # Xu (2017), Algorithm 2. Prediction errors of treated units are
        # simulated by treating each control unit as treated in turn.
        err_p = np.empty((n_co, T))
        for j in range(n_co):
            others = np.delete(np.arange(n_co), j)
            draw = rng.choice(others, size=len(others), replace=True)
            f_j = _ife_fit(
                Y_co[draw], None if X_co is None else X_co[draw], r_use, tol=1e-6
            )
            pre_j = np.zeros((1, T), dtype=bool)
            pre_j[0, :T0_min] = True
            pred, _, _ = _project_treated(
                Y_co[[j]], None if X_co is None else X_co[[j]], pre_j, f_j
            )
            err_p[j] = Y_co[j] - pred[0]
        resid_co = fit["resid"]
        for _ in range(int(n_boot)):
            Yb_co = fit["fitted"] + resid_co[rng.integers(0, n_co, n_co)]
            Yb_tr = Y0_hat + err_p[rng.integers(0, n_co, n_tr)]
            fb = _ife_fit(Yb_co, X_co, r_use, tol=1e-6)
            pb, _, _ = _project_treated(Yb_tr, X_tr, pre, fb)
            a, by = _att(Yb_tr, pb, post, rel)
            boot_avg.append(a)
            boot_rel.append(by)
    elif mode == "nonparametric":
        if n_tr < 10:
            warnings.warn(
                f"gsynth: the nonparametric bootstrap resamples the {n_tr} "
                "treated units; with so few its standard errors are "
                "unreliable. Use inference='parametric'.",
                RuntimeWarning,
                stacklevel=3,
            )
        for _ in range(int(n_boot)):
            ic = rng.integers(0, n_co, n_co)
            it = rng.integers(0, n_tr, n_tr)
            fb = _ife_fit(Y_co[ic], None if X_co is None else X_co[ic], r_use, tol=1e-6)
            pb, _, _ = _project_treated(
                Y_tr[it], None if X_tr is None else X_tr[it], pre[it], fb
            )
            a, by = _att(Y_tr[it], pb, post[it], rel[it])
            boot_avg.append(a - att)
            boot_rel.append(by - att_rel.reindex(by.index))

    z = float(stats.norm.ppf(1 - alpha / 2))
    detail = pd.DataFrame(
        {
            "relative_time": att_rel.index.to_numpy(),
            "att": att_rel.to_numpy(),
            "n_units": pd.Series(rel[post]).value_counts().sort_index().to_numpy(),
        }
    )
    if boot_avg:
        draws = np.asarray(boot_avg)
        se = float(draws.std(ddof=1))
        ci = (
            att - float(np.quantile(draws, 1 - alpha / 2)),
            att - float(np.quantile(draws, alpha / 2)),
        )
        tail = min(float(np.mean(draws >= att)), float(np.mean(draws <= att)))
        pvalue = min(1.0, 2.0 * tail)
        if pvalue == 0.0:
            pvalue = float(1.0 / (len(draws) + 1))
        rel_draws = pd.concat(boot_rel, axis=1).reindex(att_rel.index)
        detail["se"] = rel_draws.std(axis=1, ddof=1).to_numpy()
        detail["ci_lower"] = (
            att_rel.to_numpy() - rel_draws.quantile(1 - alpha / 2, axis=1).to_numpy()
        )
        detail["ci_upper"] = (
            att_rel.to_numpy() - rel_draws.quantile(alpha / 2, axis=1).to_numpy()
        )
    else:
        se = float("nan")
        ci = (float("nan"), float("nan"))
        pvalue = float("nan")

    unit_ids = units[tr].tolist()
    info: Dict[str, Any] = {
        "backend": "native",
        "native_convention": "interactive fixed effects on never-treated "
        "units; treated loadings from own pre-treatment periods (Xu 2017)",
        "n_factors": r_use,
        "n_factors_source": "cross-validation" if n_factors is None else "user",
        "cv_table": cv_table,
        "beta": pd.Series(fit["beta"], index=covariates, dtype=float),
        "treated_units": unit_ids,
        "n_treated_units": n_tr,
        "n_donors": n_co,
        "n_pre_periods": {u: int(t0) for u, t0 in zip(unit_ids, T0[tr])},
        "min_pre_periods": T0_min,
        "pre_treatment_rmse": pre_rmse,
        "inference": mode,
        "n_boot": int(n_boot) if boot_avg else 0,
        "ci_normal": (att - z * se, att + z * se),
        "grand_mean": fit["mu"],
        "time_fe": pd.Series(fit["xi"], index=times),
        "factors": pd.DataFrame(fit["F"], index=times),
        "loadings_control": pd.DataFrame(fit["L"], index=units[co]),
        "loadings_treated": pd.DataFrame(lam_tr, index=units[tr]),
        "treated_unit_fe": pd.Series(alpha_tr, index=units[tr]),
        "counterfactual": pd.DataFrame(Y0_hat, index=units[tr], columns=times),
        "effects": pd.DataFrame(
            np.where(post, eff, np.nan), index=units[tr], columns=times
        ),
        "ife_iterations": int(fit["n_iter"]),
        "times": times.tolist(),
    }
    # Averages over the treated units by calendar period, in the layout the
    # single-unit result and the synth plots use.
    first = int(T0[tr].min())
    mean_y, mean_cf = Y_tr.mean(axis=0), Y0_hat.mean(axis=0)
    info["trajectory"] = pd.DataFrame(
        {"time": times, "treated": mean_y, "synthetic": mean_cf}
    )
    info["effects_by_period"] = pd.DataFrame(
        {
            "time": times[first:],
            "treated": mean_y[first:],
            "counterfactual": mean_cf[first:],
            "effect": np.nanmean(np.where(post, eff, np.nan)[:, first:], axis=0),
        }
    )
    info["Y_treated"], info["Y_synth"] = mean_y, mean_cf
    info["treatment_time"] = times[first]
    info["treated_unit"] = unit_ids[0] if n_tr == 1 else unit_ids
    info["n_post_periods"] = int(T - first)
    info["pre_treatment_mspe"] = pre_rmse**2
    if boot_avg:
        info["att_boot"] = np.asarray(boot_avg)
    return CausalResult(
        method="Generalized Synthetic Control (Xu 2017)",
        estimand="ATT",
        estimate=att,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=int(N * T),
        detail=detail,
        model_info=info,
        _citation_key="gsynth",
    )
