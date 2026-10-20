"""Synthetic DiD with staggered adoption: ``sp.sdid(..., treat=...)``.

Clean-room implementation (2026-10-20) of the staggered-adoption design of
Arkhangelsky et al. (2021, Appendix A) as set out in Clarke et al. (2024):
one SDID block fit per adoption cohort against the never-treated units, the
cohort effects aggregated with weights proportional to treated units times
post-treatment periods (their eq. 7), projected covariates (Kranz 2022 as
described by Clarke et al.), and the bootstrap, jackknife and placebo
variance estimators of Algorithms 2-4. Stata ``sdid`` 2.0.2 (GPL-3) was
used only as a black box (its outputs and error codes on our data) and
through its help file. Specification: ``docs/dev/cleanroom/sdid_staggered.md``.

References
----------
[@arkhangelsky2021synthetic]; [@clarke2024synthetic].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .._input_validation import check_columns
from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility

_SE_METHODS = ("placebo", "bootstrap", "jackknife", "noinference")


@dataclass
class _Panel:
    """Balanced panel in matrix form, rows = units, columns = periods."""

    Y: np.ndarray  # (N, T)
    X: Optional[np.ndarray]  # (N, T, K) or None
    adopt: np.ndarray  # (N,) column index of first treated period, -1 = never
    units: List[Any]
    times: List[Any]


@dataclass
class _Fit:
    att: float
    tau: Dict[int, float]  # adoption column -> cohort effect
    weight: Dict[int, float]  # adoption column -> n_treated * n_post
    omega: Dict[int, np.ndarray]  # over the never-treated rows, in row order
    lam: Dict[int, np.ndarray]  # over the cohort's pre periods
    beta: Optional[np.ndarray]


def _build_panel(
    data: pd.DataFrame,
    y: str,
    unit: str,
    time: str,
    treatment: str,
    covariates: Sequence[str],
) -> _Panel:
    """Validate the panel (balanced, absorbing 0/1 treatment) and pivot it."""
    # labels are the sp.sdid argument names (outcome= / treat=)
    check_columns(
        data,
        {
            "outcome": y,
            "unit": unit,
            "time": time,
            "treat": treatment,
            "covariates": list(covariates),
        },
        function="sp.sdid",
    )
    df = data[[unit, time, y, treatment, *covariates]].copy()
    if df.duplicated([unit, time]).any():
        raise MethodIncompatibility(
            "sdid: duplicate (unit, time) rows; a balanced panel is required.",
            recovery_hint="Collapse or drop the duplicate rows.",
        )
    for col, role in [(y, "dependent variable"), (treatment, "treatment variable")]:
        if df[col].isna().any():
            raise DataInsufficient(
                f"sdid: missing values in the {role} {col!r}; a balanced panel "
                "without missing observations is required (Stata r(416)).",
                recovery_hint="Drop the incomplete units or impute first.",
            )
    for col in covariates:
        if df[col].isna().any():
            raise DataInsufficient(
                f"sdid: missing values in covariate {col!r} (Stata r(416)).",
                recovery_hint="Drop the rows with a missing covariate first, as "
                "Stata's examples do (`drop if lngdp == .`).",
            )
        if pd.to_numeric(df[col], errors="coerce").std() == 0:
            raise MethodIncompatibility(
                f"sdid: covariate {col!r} is constant in the estimation sample "
                "(Stata r(416)).",
                recovery_hint="Remove constant covariates.",
            )
    w = pd.to_numeric(df[treatment], errors="coerce")
    if not w.isin([0, 1]).all():
        raise MethodIncompatibility(
            f"sdid: treatment {treatment!r} takes values other than 0 and 1 "
            "(Stata r(450)).",
            recovery_hint="Pass a 0/1 indicator that switches on at adoption.",
        )
    df[treatment] = w.astype(int)

    times = sorted(df[time].unique().tolist())
    units = sorted(df[unit].unique().tolist(), key=lambda u: (str(type(u)), u))
    if len(df) != len(times) * len(units):
        raise MethodIncompatibility(
            "sdid: the panel is unbalanced (Stata r(451)).",
            recovery_hint="Keep units observed in every period.",
        )
    wide_w = df.pivot(index=unit, columns=time, values=treatment).loc[units, times]
    W = wide_w.to_numpy(dtype=int)
    if (np.diff(W, axis=1) < 0).any():
        raise MethodIncompatibility(
            "sdid: some units switch from treated back to untreated; treatment "
            "must be absorbing (Stata r(459)).",
            recovery_hint="Recode the treatment so it stays on after adoption.",
        )
    if W[:, 0].any():
        raise MethodIncompatibility(
            "sdid: some units are treated in the first period; units treated "
            "throughout the panel cannot enter SDID (Stata r(459)).",
            recovery_hint="Drop the always-treated units.",
        )
    ever = W.any(axis=1)
    if not ever.any():
        raise MethodIncompatibility(
            "sdid: all units are controls (Stata r(459)).",
            recovery_hint="Check the treatment indicator.",
        )
    if ever.all():
        raise MethodIncompatibility(
            "sdid: there is no never-treated unit to build the synthetic "
            "control from.",
            recovery_hint="SDID needs never-treated units; use a staggered "
            "estimator with not-yet-treated controls (sp.callaway_santanna).",
        )
    adopt = np.where(ever, W.argmax(axis=1), -1)
    short = sorted({times[a] for a in adopt[ever] if a < 2})
    if short:
        raise DataInsufficient(
            f"sdid: cohort(s) adopting at {short} have fewer than two "
            "pre-treatment periods, so the noise level that scales the "
            "regularisation is undefined.",
            recovery_hint="Drop those cohorts or extend the pre-period.",
        )
    Y = (
        df.pivot(index=unit, columns=time, values=y)
        .loc[units, times]
        .to_numpy(dtype=float)
    )
    X = None
    if covariates:
        X = np.stack(
            [
                df.pivot(index=unit, columns=time, values=c)
                .loc[units, times]
                .to_numpy(dtype=float)
                for c in covariates
            ],
            axis=-1,
        )
    return _Panel(Y=Y, X=X, adopt=adopt, units=units, times=times)


def _project(
    Y: np.ndarray, X: np.ndarray, never: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Projected covariate adjustment (Kranz 2022; Clarke et al. 2024, Sec. 2.2).

    ``beta`` is the OLS coefficient of the outcome on the covariates with
    unit and period fixed effects, fitted on the never-treated units only
    (black box: Stata ``sdid ..., covariates(x, projected)`` returns this
    ``beta``, not the one fitted on every untreated observation). The
    outcome of every unit is then replaced by ``Y - X beta`` and SDID runs
    on it. On a balanced block the two-way fixed effects are removed
    exactly by double demeaning.
    """
    Yc = Y[never]
    Xc = X[never]

    def two_way_within(A: np.ndarray) -> np.ndarray:
        return (
            A
            - A.mean(axis=0, keepdims=True)
            - A.mean(axis=1, keepdims=True)
            + A.mean(axis=(0, 1), keepdims=True)
        )

    k = X.shape[-1]
    y_w = two_way_within(Yc).ravel()
    x_w = np.stack([two_way_within(Xc[..., j]).ravel() for j in range(k)], axis=1)
    beta, *_ = np.linalg.lstsq(x_w, y_w, rcond=None)
    return Y - X @ beta, beta


def _cohort_blocks(
    Y: np.ndarray, never: np.ndarray, treated: np.ndarray, T0: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Y_co, Y_tr = Y[never], Y[treated]
    return Y_co[:, :T0], Y_co[:, T0:], Y_tr[:, :T0], Y_tr[:, T0:]


def _fit_optimized(Y: np.ndarray, X: np.ndarray, adopt: np.ndarray) -> _Fit:
    """Block design with covariates in the weight problem (R ``synthdid``)."""
    from .sdid import _sdid_optimized_covariates

    never = adopt < 0
    a = int(adopt[~never][0])
    order = np.r_[np.flatnonzero(never), np.flatnonzero(~never)]
    tau, beta, lam, omega, iterations = _sdid_optimized_covariates(
        Y[order], X[order], int(never.sum()), a
    )
    weight = float((~never).sum() * (Y.shape[1] - a))
    fit = _Fit(tau, {a: tau}, {a: weight}, {a: omega}, {a: lam}, beta)
    fit.iterations = iterations  # type: ignore[attr-defined]
    return fit


def _fit(
    panel_Y: np.ndarray,
    X: Optional[np.ndarray],
    adopt: np.ndarray,
    method: str,
    covariate_method: Optional[str] = "projected",
) -> _Fit:
    """Point estimate: one block fit per adoption cohort, eq. (7) aggregate."""
    from .sdid import _compute_weights, _estimate_tau

    never = adopt < 0
    if X is not None and covariate_method == "optimized":
        return _fit_optimized(panel_Y, X, adopt)
    Y, beta = _project(panel_Y, X, never) if X is not None else (panel_Y, None)
    T = Y.shape[1]
    tau, weight, omega, lam = {}, {}, {}, {}
    for a in np.unique(adopt[~never]):
        treated = adopt == a
        co_pre, co_post, tr_pre, tr_post = _cohort_blocks(Y, never, treated, int(a))
        om, la = _compute_weights(
            co_pre,
            co_post,
            tr_pre,
            method,
            int(never.sum()),
            int(a),
            n_tr=int(treated.sum()),
        )
        a = int(a)
        tau[a] = _estimate_tau(co_pre, co_post, tr_pre, tr_post, om, la)
        weight[a] = float(treated.sum() * (T - a))
        omega[a], lam[a] = om, la
    return _Fit(_aggregate(tau, weight), tau, weight, omega, lam, beta)


def _aggregate(tau: Dict[int, float], weight: Dict[int, float]) -> float:
    total = sum(weight.values())
    return float(sum(tau[a] * weight[a] for a in tau) / total)


def _fixed_weight_tau(
    Y: np.ndarray,
    controls: np.ndarray,
    treated: np.ndarray,
    a: int,
    omega: np.ndarray,
    lam: np.ndarray,
) -> float:
    """Cohort effect with given weights on a subset of the units.

    ``controls`` indexes the never-treated units kept (``omega`` is
    restricted to them and rescaled to sum to one); ``treated`` the
    cohort's treated units kept. NaN when the kept controls carry no weight.
    """
    from .sdid import _estimate_tau

    mass = float(omega.sum())
    if not mass > 0 or treated.size == 0:
        return float("nan")
    co_pre, co_post, tr_pre, tr_post = _cohort_blocks(Y, controls, treated, int(a))
    return _estimate_tau(co_pre, co_post, tr_pre, tr_post, omega / mass, lam)


def _jackknife(panel: _Panel, method: str, fit: _Fit) -> Tuple[float, Dict[int, float]]:
    """Leave-one-unit-out jackknife with the weights held fixed.

    Algorithm 3 of Arkhangelsky et al. (2021), in the staggered form of
    Clarke et al. (2024): for each unit ``i`` the cohort effects are
    recomputed without ``i`` using the full-sample unit weights (restricted
    to the remaining controls and rescaled) and time weights; projected
    covariates are re-estimated without ``i``. The cohort effects are
    aggregated with the eq. (7) weights of the remaining sample, and

    ``V = (N - 1)/N * sum_i (ATT^(-i) - mean_j ATT^(-j))^2``

    over all ``N`` units, the usual jackknife centred at the mean of the
    leave-one-out estimates (black box: this is what Stata ``sdid``
    reports, to 1e-15; Algorithm 3 as printed centres at the full-sample
    estimate, which differs by a term of order ``1/N^2``). Each cohort's
    variance is the same expression over its own leave-one-out effects.
    """
    Y0, X, adopt = panel.Y, panel.X, panel.adopt
    N, T = Y0.shape
    never_all = np.flatnonzero(adopt < 0)
    cohorts = sorted(fit.tau)
    att_loo = np.empty(N)
    tau_loo = {a: np.empty(N) for a in cohorts}
    for i in range(N):
        keep = np.ones(N, dtype=bool)
        keep[i] = False
        if X is not None:
            never_keep = (adopt < 0) & keep
            Y, _ = _project(Y0, X, never_keep)
        else:
            Y = Y0
        ctrl_mask = np.isin(never_all, np.flatnonzero(keep))
        controls = never_all[ctrl_mask]
        num = den = 0.0
        for a in cohorts:
            treated = np.flatnonzero((adopt == a) & keep)
            tau = _fixed_weight_tau(
                Y, controls, treated, a, fit.omega[a][ctrl_mask], fit.lam[a]
            )
            tau_loo[a][i] = tau
            w = float(treated.size * (T - a))
            num += tau * w
            den += w
        att_loo[i] = num / den
    scale = (N - 1) / N

    def jk_se(values: np.ndarray) -> float:
        return float(np.sqrt(scale * np.sum((values - values.mean()) ** 2)))

    return jk_se(att_loo), {a: jk_se(tau_loo[a]) for a in cohorts}


def _replicate(
    Y: np.ndarray,
    X: Optional[np.ndarray],
    adopt: np.ndarray,
    method: str,
) -> Tuple[float, Dict[int, float]]:
    """One bootstrap or placebo replicate: Algorithm 1 refitted in full
    (weights re-estimated, projected covariates re-estimated)."""
    fit = _fit(Y, X, adopt, method, "projected" if X is not None else None)
    return fit.att, dict(fit.tau)


def _resampling_se(
    panel: _Panel,
    method: str,
    fit: _Fit,
    se_method: str,
    n_reps: int,
    rng: np.random.Generator,
) -> Tuple[float, Dict[int, float], int]:
    """Bootstrap (Algorithm 2) or placebo (Algorithm 4) standard errors.

    Bootstrap: draw ``N`` units with replacement; a draw without treated
    or without never-treated units is discarded and drawn again. Placebo:
    on the never-treated units only, the first ``N_tr`` of a random
    permutation receive the adoption dates of the treated units (in the
    panel's unit order), and the estimator is applied to that placebo
    panel. (These draw conventions keep the seeded streams of earlier
    releases.) The standard error is the standard deviation of the ``B``
    replicate ATTs with divisor ``B`` (the algorithms' variance); each
    cohort's is the same over the replicates in which it appears.
    Returns ``(se, cohort_se, B)``.
    """
    Y, X, adopt = panel.Y, panel.X, panel.adopt
    N = Y.shape[0]
    never = np.flatnonzero(adopt < 0)
    treated_dates = adopt[adopt >= 0]
    atts: List[float] = []
    taus: Dict[int, List[float]] = {a: [] for a in fit.tau}
    while len(atts) < n_reps:
        if se_method == "bootstrap":
            idx = rng.integers(0, N, size=N)
            a_b = adopt[idx]
            if not (a_b < 0).any() or not (a_b >= 0).any():
                continue
            Y_b = Y[idx]
            X_b = None if X is None else X[idx]
        else:
            chosen = rng.permutation(never.size)[: treated_dates.size]
            a_b = np.full(never.size, -1)
            a_b[chosen] = treated_dates
            Y_b = Y[never]
            X_b = None if X is None else X[never]
        att_b, tau_b = _replicate(Y_b, X_b, a_b, method)
        atts.append(att_b)
        for a, value in tau_b.items():
            taus.setdefault(a, []).append(value)

    def sd(values: Sequence[float]) -> float:
        arr = np.asarray(values, dtype=float)
        return float(arr.std(ddof=0)) if arr.size >= 2 else float("nan")

    return sd(atts), {a: sd(taus[a]) for a in fit.tau}, len(atts)


def _check_se_method(panel: _Panel, se_method: str) -> None:
    """Refuse a variance estimator the design cannot support.

    From the ``sdid`` help (Stata error 451 in each case, black box):
    the bootstrap needs more than one treated unit; the jackknife needs at
    least two treated units in every adoption cohort (a cohort's
    leave-one-out effect is undefined otherwise); the placebo needs more
    never-treated than treated units.
    """
    adopt = panel.adopt
    n_tr = int((adopt >= 0).sum())
    n_co = int((adopt < 0).sum())
    if se_method == "bootstrap" and n_tr < 2:
        raise MethodIncompatibility(
            "sdid: the bootstrap standard error needs more than one treated "
            "unit (Stata r(451)).",
            recovery_hint="Use se_method='placebo'.",
            diagnostics={"n_treated": n_tr},
        )
    if se_method == "jackknife":
        cohorts, counts = np.unique(adopt[adopt >= 0], return_counts=True)
        single = [panel.times[int(a)] for a, c in zip(cohorts, counts) if c < 2]
        if single:
            raise MethodIncompatibility(
                "sdid: the jackknife standard error needs at least two treated "
                f"units in every adoption cohort; cohort(s) {single} have one "
                "(Stata r(451)).",
                recovery_hint="Use se_method='placebo' or 'bootstrap'.",
                diagnostics={"single_unit_cohorts": single},
            )
    if se_method == "placebo" and n_co <= n_tr:
        raise MethodIncompatibility(
            f"sdid: the placebo standard error needs more control units than "
            f"treated units ({n_co} controls, {n_tr} treated; Stata r(451)).",
            recovery_hint="Use se_method='bootstrap' or 'jackknife'.",
            diagnostics={"n_control": n_co, "n_treated": n_tr},
        )


def sdid_from_treatment(
    data: pd.DataFrame,
    *,
    y: str,
    unit: str,
    time: str,
    treatment: str,
    method: str,
    covariates: Optional[Sequence[str]],
    covariate_method: Optional[str],
    se_method: str,
    n_reps: int,
    seed: Optional[int],
    alpha: float,
) -> CausalResult:
    """Entry point behind ``sp.sdid(..., treat=...)``; see module docstring."""
    if se_method not in _SE_METHODS:
        raise MethodIncompatibility(
            f"sdid: se_method must be one of {_SE_METHODS}, got {se_method!r}.",
            recovery_hint="Use 'placebo', 'bootstrap', 'jackknife' or 'noinference'.",
        )
    covariates = list(covariates or [])
    if covariates:
        if covariate_method is None:
            raise MethodIncompatibility(
                "sdid: covariates need covariate_method= ('projected' or "
                "'optimized'). The two give different estimates, and Stata's "
                "default is 'optimized', so neither is chosen silently.",
                recovery_hint="Pass covariate_method='projected' (a regression "
                "adjustment, with standard errors) or 'optimized'.",
            )
        if covariate_method not in ("projected", "optimized"):
            raise MethodIncompatibility(
                f"sdid: covariate_method={covariate_method!r} is not one of "
                "'projected', 'optimized'.",
                recovery_hint="Pass covariate_method='projected' or 'optimized'.",
            )
        if covariate_method == "optimized":
            if method != "sdid":
                raise MethodIncompatibility(
                    "sdid: covariate_method='optimized' is defined for "
                    f"method='sdid', not {method!r}.",
                    recovery_hint="Use covariate_method='projected'.",
                )
            if se_method != "noinference":
                raise MethodIncompatibility(
                    "sdid: covariate_method='optimized' returns the point "
                    "estimate only; its resampling standard errors are not "
                    f"implemented (se_method={se_method!r}).",
                    recovery_hint="Pass se_method='noinference', or use "
                    "covariate_method='projected' for standard errors.",
                )
    elif covariate_method is not None:
        raise MethodIncompatibility(
            "sdid: covariate_method= was given without covariates=.",
            recovery_hint="Pass covariates=[...] or drop covariate_method.",
        )

    panel = _build_panel(data, y, unit, time, treatment, covariates)
    _check_se_method(panel, se_method)
    if covariate_method == "optimized" and len(set(panel.adopt[panel.adopt >= 0])) > 1:
        raise MethodIncompatibility(
            "sdid: covariate_method='optimized' is implemented for a single "
            "adoption date.",
            recovery_hint="Use covariate_method='projected' with staggered "
            "adoption.",
        )
    fit = _fit(panel.Y, panel.X, panel.adopt, method, covariate_method)

    rng = np.random.default_rng(seed)
    cohort_se: Dict[int, float] = {a: float("nan") for a in fit.tau}
    draws = None
    if se_method == "jackknife":
        se, cohort_se = _jackknife(panel, method, fit)
    elif se_method in ("bootstrap", "placebo"):
        se, cohort_se, draws = _resampling_se(
            panel, method, fit, se_method, n_reps, rng
        )
    else:
        se = float("nan")

    z = stats.norm.ppf(1 - alpha / 2)
    if np.isfinite(se) and se > 0:
        pvalue = float(2 * stats.norm.sf(abs(fit.att / se)))
        ci = (fit.att - z * se, fit.att + z * se)
    else:
        pvalue, ci = float("nan"), (float("nan"), float("nan"))

    times, units = panel.times, panel.units
    never_units = [u for u, a in zip(units, panel.adopt) if a < 0]
    cohorts = sorted(fit.tau)
    total_w = sum(fit.weight.values())
    tau_table = pd.DataFrame(
        {
            "adoption": [times[a] for a in cohorts],
            "tau": [fit.tau[a] for a in cohorts],
            "se": [cohort_se.get(a, float("nan")) for a in cohorts],
            "n_treated": [int((panel.adopt == a).sum()) for a in cohorts],
            "T_post": [len(times) - a for a in cohorts],
            "weight": [fit.weight[a] / total_w for a in cohorts],
        }
    )
    unit_weights = pd.DataFrame(
        {times[a]: fit.omega[a] for a in cohorts},
        index=pd.Index(never_units, name="unit"),
    )
    time_weights = pd.DataFrame(
        {times[a]: pd.Series(fit.lam[a], index=times[:a]) for a in cohorts}
    )
    labels = {
        "sdid": "Synthetic Difference-in-Differences",
        "sc": "Synthetic Control",
        "did": "Difference-in-Differences",
    }
    model_info: Dict[str, Any] = {
        "estimator": method,
        "estimator_label": labels[method],
        "design": "staggered" if len(cohorts) > 1 else "block",
        "backend": "native",
        "reference_backend": "Stata sdid 2.0.2",
        "validation_note": (
            "treat= path: cohort-by-cohort block fits with the block "
            "estimator's weight solver, aggregated by treated units x post "
            "periods, as Stata sdid. Point estimates and the jackknife SE "
            "reproduce Stata sdid; bootstrap / placebo match it in "
            "distribution only."
        ),
        "tau_by_cohort": tau_table,
        "unit_weights": unit_weights,
        "time_weights": time_weights,
        "n_treated": int((panel.adopt >= 0).sum()),
        "n_control": len(never_units),
        "control_units": never_units,
        "treated_units": [u for u, a in zip(units, panel.adopt) if a >= 0],
        "all_times": times,
        "se_method": se_method,
        "n_reps": n_reps if se_method in ("placebo", "bootstrap") else None,
        "resampling_draws": draws,
        "covariates": covariates or None,
        "covariate_method": covariate_method,
        "covariate_iterations": getattr(fit, "iterations", None),
        "covariate_beta": (
            pd.Series(fit.beta, index=covariates) if fit.beta is not None else None
        ),
    }
    return CausalResult(
        method=f"{labels[method]} (Arkhangelsky et al. 2021; Clarke et al. 2024)",
        estimand="ATT",
        estimate=fit.att,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=len(data),
        model_info=model_info,
        _citation_key="sdid",
    )
