"""
BCFLong — hierarchical Bayesian Causal Forest for longitudinal data.

Alessi, Zorzetto et al. (arXiv:2508.08418, 2025) extend BCF to
longitudinal/panel data with a two-level hierarchy:

* Level 1 (observation): ``Y_{it} = mu_t(X_{it}) + tau_t(X_{it}) D_{it} + u_i + e_{it}``
* Level 2 (subject)    : ``u_i ~ Normal(0, sigma_u^2)`` (random intercept)

Both ``mu_t`` and ``tau_t`` can evolve across time, which is critical
when the treatment effect itself is dynamic (e.g. learning curves,
washout, compliance drift). The paper ran it on multiple-sclerosis
clinical-trial data; we reproduce the core ideas here with an efficient
bootstrap-based posterior that avoids PyMC as a hard dependency.

References
----------
Alessi, Zorzetto et al. (arXiv:2508.08418, 2025).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from scipy import stats

from ..core.results import CausalResult


__all__ = ["bcf_longitudinal", "BCFLongResult"]


@dataclass
class BCFLongResult:
    """Longitudinal BCF output container."""
    per_time_ate: pd.DataFrame   # columns: time, ate, se, ci_low, ci_high
    average_ate: float
    average_se: float
    average_ci: tuple
    individual_cate: pd.DataFrame  # per-(unit, time) CATE
    model_info: Dict = field(default_factory=dict)

    def summary(self) -> str:
        lo, hi = self.average_ci
        lines = [
            "BCFLong — Longitudinal Bayesian Causal Forest",
            "=" * 64,
            f"  Average ATE (over time) : {self.average_ate:.6f}",
            f"  SE                      : {self.average_se:.6f}",
            f"  95% CI                  : [{lo:.6f}, {hi:.6f}]",
            "",
            "Per-time ATE:",
            self.per_time_ate.to_string(index=False, float_format="%.4f"),
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"<BCFLongResult: {len(self.per_time_ate)} time points, "
            f"avg ATE = {self.average_ate:.4f}>"
        )


def _estimate_propensity(
    X: np.ndarray, D: np.ndarray, random_state: int,
) -> np.ndarray:
    clf = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, random_state=random_state,
    )
    clf.fit(X, D)
    p = clf.predict_proba(X)[:, 1]
    return np.clip(p, 0.01, 0.99)


def _fit_mu_tau_at_time(
    X: np.ndarray, Y: np.ndarray, D: np.ndarray, e: np.ndarray,
    unit_effect: np.ndarray,
    n_trees_mu: int = 200, n_trees_tau: int = 50,
    random_state: int = 42,
) -> tuple:
    """Fit BCF prognostic/treatment forests at a single time point.

    Uses a T-learner variant for ``tau`` to avoid the "mu absorbs tau"
    failure mode when D is a free forest feature:

    * mu_0(X, e) = E[Y - u_i | X, e, D=0] fit on controls.
    * mu_1(X, e) = E[Y - u_i | X, e, D=1] fit on treated.
    * mu(X, e)  = (1-e)*mu_0 + e*mu_1  — BCF-style propensity weighting.
    * tau(X)    = shrunk forest on mu_1(X,e) - mu_0(X,e).
    """
    X_full = np.column_stack([X, e])
    Y_resid = Y - unit_effect
    treated = D == 1
    control = ~treated
    rs = int(random_state) % (2 ** 31)

    def _fit(mask):
        if mask.sum() < 5:
            return None
        rf = RandomForestRegressor(
            n_estimators=n_trees_mu, max_depth=None,
            min_samples_leaf=5, random_state=rs,
        )
        rf.fit(X_full[mask], Y_resid[mask])
        return rf

    rf_0 = _fit(control)
    rf_1 = _fit(treated)
    if rf_0 is None or rf_1 is None:
        # Too few per arm — degenerate.
        mu_hat = np.zeros_like(Y)
        tau_hat = np.zeros_like(Y)
        return mu_hat, tau_hat
    mu0 = rf_0.predict(X_full)
    mu1 = rf_1.predict(X_full)
    # Propensity-weighted mu (BCF trick).
    mu_hat = (1.0 - e) * mu0 + e * mu1
    # Shrinkage forest for tau on the per-unit CATE.
    diff = mu1 - mu0
    rf_tau = RandomForestRegressor(
        n_estimators=n_trees_tau, max_depth=None,
        min_samples_leaf=5, random_state=rs,
    )
    rf_tau.fit(X, diff)
    tau_hat = rf_tau.predict(X)
    return mu_hat, tau_hat


def bcf_longitudinal(
    data: pd.DataFrame,
    *,
    outcome: str,
    treatment: str,
    unit: str,
    time: str,
    covariates: Sequence[str],
    n_trees_mu: int = 200,
    n_trees_tau: int = 50,
    n_bootstrap: int = 100,
    alpha: float = 0.05,
    random_state: int = 42,
) -> BCFLongResult:
    """Longitudinal Bayesian Causal Forest (BCFLong).

    Parameters
    ----------
    data : DataFrame
        Long-format panel. ``(unit, time)`` should be unique per row.
    outcome, treatment, unit, time : str
    covariates : sequence of str
    n_trees_mu, n_trees_tau : int
        BCF regularisation: ``n_trees_tau < n_trees_mu`` shrinks toward
        homogeneous effects.
    n_bootstrap : int, default 100
    alpha : float, default 0.05
    random_state : int, default 42

    Returns
    -------
    BCFLongResult

    Notes
    -----
    The hierarchy is handled by iteratively:

    1. Estimate unit random intercepts ``u_i`` as the unit-demeaned
       average of current residuals.
    2. Fit the mu/tau forests at each time slice on ``Y - u_i``.
    3. Update residuals and iterate (2 passes suffice in practice).

    Point estimates come from the final pass; uncertainty comes from
    nonparametric unit-level cluster-bootstrap — the correct resampling
    unit for panel data.

    References
    ----------
    Alessi, Zorzetto et al. (arXiv:2508.08418, 2025).
    Hahn, Murray, Carvalho (2020), Bayesian Analysis.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas DataFrame.")
    required = {outcome, treatment, unit, time, *covariates}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns in data: {sorted(missing)}")
    df = data.copy()
    if df[[unit, time]].duplicated().any():
        raise ValueError(
            f"({unit}, {time}) must be unique per row — found duplicates."
        )

    time_vals = sorted(df[time].unique())
    if len(time_vals) < 2:
        raise ValueError(
            "`bcf_longitudinal` requires >= 2 distinct time points. "
            "For a single time point use sp.bcf()."
        )

    def _single_pass(df_in: pd.DataFrame, seed: int) -> tuple:
        """Run two-pass BCFLong on one bootstrap draw (or original data)."""
        units = df_in[unit].unique()
        u_effect = {u: 0.0 for u in units}
        tau_panel: Dict = {}
        mu_panel: Dict = {}
        for pass_i in range(2):
            mu_panel, tau_panel = {}, {}
            for t in time_vals:
                slice_t = df_in.loc[df_in[time] == t]
                if slice_t.empty:
                    continue
                X_t = slice_t[list(covariates)].to_numpy(dtype=float)
                Y_t = slice_t[outcome].to_numpy(dtype=float)
                D_t = slice_t[treatment].to_numpy(dtype=int)
                ue_t = np.array([u_effect[u] for u in slice_t[unit]])
                if np.unique(D_t).size < 2:
                    # No treatment variation at this time: ATE undefined.
                    mu_panel[t] = np.zeros_like(Y_t)
                    tau_panel[t] = np.zeros_like(Y_t)
                    continue
                e_t = _estimate_propensity(X_t, D_t, seed + int(hash(t)) % 1000)
                mu_hat, tau_hat = _fit_mu_tau_at_time(
                    X_t, Y_t, D_t, e_t, ue_t,
                    n_trees_mu=n_trees_mu, n_trees_tau=n_trees_tau,
                    random_state=seed,
                )
                mu_panel[t] = mu_hat
                tau_panel[t] = tau_hat
            # Update unit random effect via unit-level mean of residual.
            resid_rows = []
            for t in time_vals:
                slice_t = df_in.loc[df_in[time] == t]
                if slice_t.empty:
                    continue
                idx = slice_t.index
                Y_t = slice_t[outcome].to_numpy(dtype=float)
                D_t = slice_t[treatment].to_numpy(dtype=float)
                mu_hat = mu_panel[t]
                tau_hat = tau_panel[t]
                r = Y_t - mu_hat - tau_hat * D_t
                resid_rows.append(pd.DataFrame({
                    "unit": slice_t[unit].to_numpy(),
                    "r": r,
                }, index=idx))
            rd = pd.concat(resid_rows, axis=0)
            u_effect = rd.groupby("unit")["r"].mean().to_dict()
        return tau_panel, mu_panel, u_effect

    # --- Point estimates ---------------------------------------------------
    tau_panel, mu_panel, u_effect = _single_pass(df, random_state)
    per_time_ate_rows = []
    individual_rows = []
    for t in time_vals:
        slice_t = df.loc[df[time] == t]
        if slice_t.empty or t not in tau_panel:
            continue
        tau_hat = tau_panel[t]
        per_time_ate_rows.append({
            "time": t,
            "ate_point": float(tau_hat.mean()),
            "n": int(len(slice_t)),
        })
        individual_rows.append(pd.DataFrame({
            unit: slice_t[unit].to_numpy(),
            time: t,
            "cate": tau_hat,
        }))
    per_time_ate = pd.DataFrame(per_time_ate_rows)
    individual_cate = pd.concat(individual_rows, axis=0, ignore_index=True)

    # --- Cluster bootstrap for SEs ----------------------------------------
    rng = np.random.default_rng(random_state)
    all_units = df[unit].unique()
    boot_point = []
    boot_per_time = {t: [] for t in time_vals}
    for b in range(n_bootstrap):
        draw = rng.choice(all_units, size=len(all_units), replace=True)
        boot_parts = []
        for i, u in enumerate(draw):
            sub = df.loc[df[unit] == u].copy()
            sub[unit] = f"{u}__b{i}"  # unique id to keep random-effect logic intact
            boot_parts.append(sub)
        boot_df = pd.concat(boot_parts, axis=0, ignore_index=True)
        try:
            tau_b, _, _ = _single_pass(boot_df, random_state + b + 1)
        except Exception:
            continue
        t_vals_b = []
        for t in time_vals:
            sb = boot_df.loc[boot_df[time] == t]
            if sb.empty or t not in tau_b:
                continue
            mean_t = float(tau_b[t].mean())
            boot_per_time[t].append(mean_t)
            t_vals_b.append(mean_t)
        if t_vals_b:
            boot_point.append(float(np.mean(t_vals_b)))

    if len(boot_point) < max(20, n_bootstrap // 4):
        raise RuntimeError(
            f"Only {len(boot_point)}/{n_bootstrap} bootstrap draws succeeded. "
            "Check data balance and treatment overlap."
        )

    # Per-time SE/CI
    se_col, low_col, high_col = [], [], []
    for t in per_time_ate["time"]:
        draws = np.array(boot_per_time[t])
        if draws.size < 20:
            se_col.append(np.nan)
            low_col.append(np.nan)
            high_col.append(np.nan)
            continue
        se_col.append(float(draws.std(ddof=1)))
        low_col.append(float(np.quantile(draws, alpha / 2)))
        high_col.append(float(np.quantile(draws, 1 - alpha / 2)))
    per_time_ate["se"] = se_col
    per_time_ate["ci_low"] = low_col
    per_time_ate["ci_high"] = high_col

    avg = float(np.array([r for r in boot_point]).mean())
    se = float(np.std(boot_point, ddof=1))
    ci = (
        float(np.quantile(boot_point, alpha / 2)),
        float(np.quantile(boot_point, 1 - alpha / 2)),
    )

    return BCFLongResult(
        per_time_ate=per_time_ate,
        average_ate=float(per_time_ate["ate_point"].mean()),
        average_se=se,
        average_ci=ci,
        individual_cate=individual_cate,
        model_info={
            "n_time_points": len(time_vals),
            "n_units": int(df[unit].nunique()),
            "n_obs": int(len(df)),
            "n_trees_mu": n_trees_mu,
            "n_trees_tau": n_trees_tau,
            "n_bootstrap_effective": len(boot_point),
            "reference": "Alessi, Zorzetto et al. (arXiv:2508.08418, 2025)",
        },
    )
