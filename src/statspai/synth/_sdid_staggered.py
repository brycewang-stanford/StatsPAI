"""SDID from a treatment indicator: staggered adoption, projected covariates.

``sp.sdid(..., treat="W")`` follows Stata ``sdid`` (Pailanir & Clarke),
the implementation accompanying [@clarke2024synthetic]:

* **Design.** Units adopt at possibly different periods (absorbing
  treatment). Each adoption cohort ``a`` is a block design -- its treated
  units against the never-treated units over the full panel -- fitted with
  the same weight solver as the block estimator (``sdid._compute_weights``).
  The ATT aggregates the cohort effects with weights proportional to treated
  units x post periods (eq. 7 of the paper).
* **Covariates** (``covariate_method="projected"``): the outcome
  is regressed on the covariates with unit and period fixed effects on the
  never-treated units only, and ``Y - X beta`` replaces ``Y``. The
  never-treated sample is what ``sdid.ado``'s ``projected()`` uses (it selects
  on the unit-level ever-treated flag); Stata's default ``optimized`` method is
  not implemented.
* **Inference.** ``jackknife`` holds each cohort's omega (renormalised after
  a control is dropped) and lambda fixed, re-projects covariates on every
  leave-one-out sample, and is deterministic -- it reproduces Stata exactly.
  ``bootstrap`` (units resampled with replacement, redrawn without a treated
  or a control unit) and ``placebo`` (as many controls as treated units given
  the treated units' adoption periods) re-solve the weights from the original
  ones with regularisation recomputed on each replication, as ``sdid.ado``
  does. Their random draws cannot match Stata's; only their distribution does.
  ``noinference`` returns the point estimate only. The applicability rules are
  ``sdid.ado``'s: jackknife needs two treated units in every cohort, bootstrap
  more than one treated unit when there is a single cohort, placebo more
  controls than treated units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

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
    """Validate as ``sdid.ado`` does and pivot to matrices."""
    for col in [y, unit, time, treatment, *covariates]:
        if col not in data.columns:
            raise MethodIncompatibility(
                f"sdid: column {col!r} is not in the data.",
                recovery_hint="Check the column names passed to sp.sdid.",
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
    """``sdid.ado``'s ``projected()``: beta from a two-way FE regression on the
    never-treated units, then ``Y - X beta`` for every unit.

    On a balanced panel two-way demeaning gives the fixed-effects coefficient
    exactly (Frisch-Waugh-Lovell).
    """
    Yc, Xc = Y[never], X[never]

    def demean(A: np.ndarray) -> np.ndarray:
        out: np.ndarray = (
            A
            - A.mean(axis=1, keepdims=True)
            - A.mean(axis=0, keepdims=True)
            + A.mean(axis=(0, 1), keepdims=True)
        )
        return out

    y = demean(Yc).ravel()
    Z = np.column_stack([demean(Xc[..., k]).ravel() for k in range(X.shape[-1])])
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    return Y - X @ beta, beta


def _cohort_blocks(
    Y: np.ndarray, never: np.ndarray, treated: np.ndarray, T0: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Y_co, Y_tr = Y[never], Y[treated]
    return Y_co[:, :T0], Y_co[:, T0:], Y_tr[:, :T0], Y_tr[:, T0:]


def _fit(
    panel_Y: np.ndarray, X: Optional[np.ndarray], adopt: np.ndarray, method: str
) -> _Fit:
    """Point estimate: one block fit per adoption cohort, eq. (7) aggregate."""
    from .sdid import _compute_weights, _estimate_tau

    never = adopt < 0
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


def _refit(
    Y: np.ndarray,
    X: Optional[np.ndarray],
    adopt: np.ndarray,
    method: str,
    omega0: Dict[int, np.ndarray],
    lam0: Dict[int, np.ndarray],
) -> Tuple[float, Dict[int, float]]:
    """One bootstrap / placebo replication (``synthdid(data, inference=1, ...)``).

    ``omega0[a]`` is already aligned to this replication's never-treated rows.
    Weights are re-solved from those starting values with the regularisation
    recomputed on the replication, as ``sdid.ado`` does.
    """
    from .sdid import _sum_normalize, _synthdid_opts, _synthdid_refit

    never = adopt < 0
    if X is not None:
        Y, _ = _project(Y, X, never)
    T = Y.shape[1]
    tau, weight = {}, {}
    for a in np.unique(adopt[~never]):
        a = int(a)
        treated = adopt == a
        block = np.vstack([Y[never], Y[treated]])
        n0 = int(never.sum())
        opts = _synthdid_opts(block, n0, a, method)
        tau[a] = _synthdid_refit(block, n0, a, _sum_normalize(omega0[a]), lam0[a], opts)
        weight[a] = float(treated.sum() * (T - a))
    return _aggregate(tau, weight), tau


def _jackknife(panel: _Panel, method: str, fit: _Fit) -> Tuple[float, Dict[int, float]]:
    """Fixed-weight leave-one-unit-out jackknife (``sdid.ado`` jk branch)."""
    from .sdid import _sum_normalize

    never = panel.adopt < 0
    control_rows = np.flatnonzero(never)
    N, T = panel.Y.shape
    atts: List[float] = []
    taus: Dict[int, List[float]] = {a: [] for a in fit.tau}
    for i in range(N):
        keep = np.arange(N) != i
        Y, adopt = panel.Y[keep], panel.adopt[keep]
        X = panel.X[keep] if panel.X is not None else None
        nv = adopt < 0
        if X is not None:
            Y, _ = _project(Y, X, nv)
        drop_ctrl = control_rows != i
        tau, weight = {}, {}
        for a in fit.tau:
            treated = adopt == a
            om = _sum_normalize(fit.omega[a][drop_ctrl])
            la = fit.lam[a]
            co_pre, co_post, tr_pre, tr_post = _cohort_blocks(Y, nv, treated, a)
            post = tr_post.mean() - om @ co_post.mean(axis=1)
            pre = tr_pre.mean(axis=0) @ la - om @ (co_pre @ la)
            tau[a] = float(post - pre)
            weight[a] = float(treated.sum() * (T - a))
            taus[a].append(tau[a])
        atts.append(_aggregate(tau, weight))

    def se(values: Sequence[float]) -> float:
        x = np.asarray(values, dtype=float)
        n = x.size
        return float(np.sqrt((n - 1) / n * np.sum((x - x.mean()) ** 2)))

    return se(atts), {a: se(v) for a, v in taus.items()}


def _resampling_se(
    panel: _Panel,
    method: str,
    fit: _Fit,
    se_method: str,
    n_reps: int,
    rng: np.random.Generator,
) -> Tuple[float, Dict[int, float], int]:
    """Bootstrap or placebo SE: ``sqrt((B-1)/B) * sd`` over replications."""
    never = panel.adopt < 0
    control_rows = np.flatnonzero(never)
    ctrl_pos = {r: k for k, r in enumerate(control_rows)}
    N = panel.Y.shape[0]
    atts: List[float] = []
    taus: Dict[int, List[float]] = {a: [] for a in fit.tau}
    draws = 0
    treated_adopt = panel.adopt[~never]
    while len(atts) < n_reps:
        draws += 1
        if draws > 50 * n_reps:
            raise MethodIncompatibility(
                f"sdid: could not complete {n_reps} {se_method} replications "
                f"after {draws - 1} draws.",
                recovery_hint="Use se_method='jackknife' or 'noinference'.",
            )
        if se_method == "bootstrap":
            rows = rng.integers(0, N, N)
            adopt = panel.adopt[rows]
            if (adopt < 0).all() or (adopt >= 0).all():
                continue  # sdid.ado redraws: all controls or all treated
            ctrl_src = rows[adopt < 0]
        else:  # placebo
            chosen = rng.permutation(control_rows)[: treated_adopt.size]
            adopt = np.full(control_rows.size, -1)
            pos = np.searchsorted(control_rows, chosen)
            adopt[pos] = treated_adopt
            rows = control_rows
            ctrl_src = control_rows[adopt < 0]
        Y = panel.Y[rows]
        X = panel.X[rows] if panel.X is not None else None
        idx = [ctrl_pos[r] for r in ctrl_src]
        omega0 = {a: fit.omega[a][idx] for a in fit.omega}
        att, tau = _refit(Y, X, adopt, method, omega0, fit.lam)
        atts.append(att)
        for a, v in tau.items():
            taus.setdefault(a, []).append(v)

    def se(values: Sequence[float]) -> float:
        x = np.asarray(values, dtype=float)
        return float(np.std(x)) if x.size > 1 else float("nan")

    return se(atts), {a: se(v) for a, v in taus.items() if a in fit.tau}, draws


def _check_se_method(panel: _Panel, se_method: str) -> None:
    """``sdid.ado``'s refusals (all r(451))."""
    ever = panel.adopt >= 0
    counts = pd.Series(panel.adopt[ever]).value_counts()
    if se_method == "jackknife" and (counts < 2).any():
        singles = sorted(panel.times[a] for a in counts[counts < 2].index)
        raise MethodIncompatibility(
            "sdid: the jackknife standard error needs at least two treated "
            f"units in every adoption cohort; cohort(s) {singles} have one "
            "(Stata r(451)).",
            recovery_hint="Use se_method='placebo' or 'bootstrap'.",
        )
    if se_method == "bootstrap" and len(counts) == 1 and int(counts.iloc[0]) == 1:
        raise MethodIncompatibility(
            "sdid: the bootstrap standard error needs more than one treated "
            "unit when there is a single adoption period (Stata r(451)).",
            recovery_hint="Use se_method='placebo'.",
        )
    if se_method == "placebo" and (~ever).sum() <= ever.sum():
        raise MethodIncompatibility(
            "sdid: the placebo standard error needs more control units than "
            "treated units (Stata r(451)).",
            recovery_hint="Use se_method='bootstrap' or 'jackknife'.",
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
                "sdid: covariates need covariate_method=. Only 'projected' "
                "is implemented; Stata's default is 'optimized', "
                "so it is not chosen silently.",
                recovery_hint="Pass covariate_method='projected'.",
            )
        if covariate_method != "projected":
            raise MethodIncompatibility(
                f"sdid: covariate_method={covariate_method!r} is not implemented; "
                "only 'projected' is.",
                recovery_hint="Pass covariate_method='projected'.",
            )
    elif covariate_method is not None:
        raise MethodIncompatibility(
            "sdid: covariate_method= was given without covariates=.",
            recovery_hint="Pass covariates=[...] or drop covariate_method.",
        )

    panel = _build_panel(data, y, unit, time, treatment, covariates)
    _check_se_method(panel, se_method)
    fit = _fit(panel.Y, panel.X, panel.adopt, method)

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
