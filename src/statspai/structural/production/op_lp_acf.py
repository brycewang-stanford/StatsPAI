"""
Olley-Pakes / Levinsohn-Petrin / Ackerberg-Caves-Frazer production function
estimators.

Note on ``diagnostics["ar_rho"]``: this is the linear coefficient on
``omega_{t-1}`` in the productivity polynomial ``g``. It equals the AR(1)
persistence parameter only when ``productivity_degree == 1`` (the
default). For higher degrees, the full polynomial coefficients are
needed to characterize the productivity process — ``ar_rho`` then
captures only the linear component.

All three are two-step proxy-variable estimators. They share the stage-1
nonparametric control function

    y_it = Phi(free_it, state_it, proxy_it) + eta_it

and differ only in (i) which input plays the proxy role and (ii) which
moment conditions identify the structural parameters in stage 2.

This module exposes three thin wrappers on top of the shared GMM driver
``_estimate_proxy``:

* :func:`olley_pakes`            — proxy = investment, free = labor
* :func:`levinsohn_petrin`       — proxy = intermediate input (materials),
                                    free = labor
* :func:`ackerberg_caves_frazer` — proxy = intermediate input,
                                    free = labor, but lagged labor and
                                    current capital instrument the second
                                    stage (the ACF correction)

Cobb-Douglas is the only supported functional form here.  Translog and
Wooldridge (2009) one-step GMM live in sibling modules.

References
----------
Olley & Pakes (1996, Econometrica) [@olley1996dynamics]
Levinsohn & Petrin (2003, Rev. Econ. Stud.) [@levinsohn2003estimating]
Ackerberg, Caves & Frazer (2015, Econometrica) [@ackerberg2015identification]
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import optimize

from ._core import (
    elasticities_at,
    expand_inputs,
    firm_bootstrap_indices,
    gmm_objective,
    panel_lag,
    productivity_residual,
    stage_one_phi,
)
from ._result import ProductionResult


# ---------------------------------------------------------------------------
# Shared driver
# ---------------------------------------------------------------------------

def _prepare_panel(
    data: pd.DataFrame,
    output: str,
    free: Sequence[str],
    state: Sequence[str],
    proxy: str,
    panel_id: str,
    time: str,
    extra_lag_columns: Sequence[str] = (),
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Sort by (id, time), attach lags, drop pre-period rows.

    Returns the cleaned working frame plus the row-mask aligned to the
    *original* sort, so the caller can recover firm × time alignment of the
    productivity series.
    """
    cols = list({output, *free, *state, proxy, panel_id, time, *extra_lag_columns})
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")
    df = (
        data[cols]
        .dropna()
        .sort_values([panel_id, time])
        .reset_index(drop=True)
        .copy()
    )
    df["__panel_id__"] = df[panel_id].to_numpy()
    df["__time__"] = df[time].to_numpy()

    # Warn if any firm has non-consecutive time periods — the AR(1)
    # productivity assumption then mixes lags across genuine gaps.
    try:
        gaps = df.groupby(panel_id)[time].diff()
        if (gaps > 1).any():
            n_gap_firms = df.loc[gaps > 1, panel_id].nunique()
            warnings.warn(
                f"{n_gap_firms} firms have non-consecutive time periods; "
                "the lag operator will treat these gaps as 1-period lags "
                "and bias the AR productivity fit. Consider dropping "
                "incomplete firm histories.",
                UserWarning, stacklevel=3,
            )
    except (TypeError, ValueError):
        # Non-numeric time column (e.g. strings / datetimes) — skip the
        # consecutive-period check.
        pass

    for col in [*free, *state, proxy, output, *extra_lag_columns]:
        df[f"__lag1__{col}"] = panel_lag(df, col, panel_id, time, lag=1)
    return df, df["__panel_id__"].to_numpy()


def _estimate_proxy(
    df: pd.DataFrame,
    *,
    output: str,
    free: Sequence[str],
    state: Sequence[str],
    proxy: str,
    method: str,
    polynomial_degree: int = 3,
    productivity_degree: int = 3,
    functional_form: str = "cobb-douglas",
    boot_reps: int = 0,
    seed: Optional[int] = None,
) -> ProductionResult:
    """Two-step GMM driver shared by OP, LP, and ACF.

    The three methods only differ in (a) which input plays the role of
    ``proxy`` and (b) which lags appear in the stage-2 instrument matrix
    ``Z``.  ACF additionally requires that *no* free input enter the
    moment vector at the contemporaneous lag.

    ``functional_form`` controls the input expansion: ``cobb-douglas``
    keeps the raw linear inputs (one parameter per input); ``translog``
    expands to second-order polynomial (linear + 0.5 * x_j^2 + cross
    terms), so the parameter vector grows from ``p`` to ``p*(p+3)/2``.
    Stage-2 instruments are expanded in the same way to keep the
    moment system just-identified.
    """
    free = list(free)
    state = list(state)
    raw_input_names = free + state

    # ---- Stage 1 -------------------------------------------------------
    Z1_cols = raw_input_names + [proxy]
    Z1 = df[Z1_cols].to_numpy(dtype=float)
    y = df[output].to_numpy(dtype=float)
    phi_hat, eta_hat, _ = stage_one_phi(y, Z1, degree=polynomial_degree)

    # ---- Stage 2: build instruments ------------------------------------
    # OP/LP: free input enters the moment at time t (free is "freely chosen
    # given omega" but uncorrelated with the *innovation* xi).
    # ACF: only LAGGED free inputs are valid; current free input is an
    # endogenous response to the innovation. Both versions instrument
    # state inputs at the contemporaneous lag (state is predetermined).
    instr_cols: List[str] = []
    instr_names: List[str] = []
    for s in state:
        instr_cols.append(s)                       # k_it
        instr_names.append(s)
    if method == "acf":
        for f in free:
            instr_cols.append(f"__lag1__{f}")      # l_{i,t-1}
            instr_names.append(f)                   # name without lag prefix
    else:  # op / lp
        for f in free:
            instr_cols.append(f)
            instr_names.append(f)
    Z2_raw = df[instr_cols].to_numpy(dtype=float)

    # Expand inputs and instruments under the functional form.
    inputs_raw = df[raw_input_names].to_numpy(dtype=float)
    inputs_mat, expanded_names = expand_inputs(
        inputs_raw, raw_input_names, functional_form,
    )
    Z2, _ = expand_inputs(Z2_raw, instr_names, functional_form)
    panel_arr = df["__panel_id__"].to_numpy()
    time_arr = df["__time__"].to_numpy()

    # Drop rows where any instrument or input is NaN (lag operator created
    # NaNs at t = t_min for each firm).
    valid = (
        np.isfinite(phi_hat)
        & np.all(np.isfinite(inputs_mat), axis=1)
        & np.all(np.isfinite(Z2), axis=1)
    )
    phi_w = phi_hat[valid]
    eta_w = eta_hat[valid]
    inputs_w = inputs_mat[valid]
    Z2_w = Z2[valid]
    panel_w = panel_arr[valid]
    time_w = time_arr[valid]

    if Z2_w.shape[0] < 10:
        raise ValueError(
            f"Only {Z2_w.shape[0]} valid observations for stage 2; need at least 10. "
            "Check panel structure (sufficient time periods per firm)."
        )

    # ---- Stage 2: GMM minimization ------------------------------------
    def obj(beta):
        return gmm_objective(
            np.asarray(beta, dtype=float),
            phi_hat=phi_w,
            inputs=inputs_w,
            instruments=Z2_w,
            panel_id=panel_w,
            time=time_w,
            productivity_degree=productivity_degree,
        )

    # Multi-start. The OLS warm start is upward-biased (labor correlates
    # positively with omega) and reliably lands in a spurious basin where
    # the productivity AR overfits ω onto ω_lag, driving moments to
    # near-zero at economically implausible β. We deliberately avoid OLS
    # and use a small grid of plausible economic priors instead. Best
    # objective among converged starts wins.
    n_in = inputs_mat.shape[1]
    n_lin = len(raw_input_names)
    def _start(linear_vals):
        s = np.zeros(n_in)
        s[:n_lin] = linear_vals[:n_lin]
        return s
    starts: List[np.ndarray] = [
        _start([0.5] * n_lin),     # equal-weight, plausible CD
        _start([0.3] * n_lin),
        _start([0.7] * n_lin),
        _start([0.6, 0.3] + [0.0] * max(0, n_lin - 2)),  # labor-heavy
        _start([0.3, 0.6] + [0.0] * max(0, n_lin - 2)),  # capital-heavy
    ]
    best = None
    for b0 in starts:
        try:
            r = optimize.minimize(
                obj, b0, method="Nelder-Mead",
                options={"xatol": 1e-7, "fatol": 1e-12, "maxiter": 20000},
            )
            if (best is None) or (r.fun < best.fun):
                best = r
        except Exception:
            continue
    if best is None:
        raise RuntimeError("All optimizer starts failed in stage 2.")
    res = best
    beta_hat = res.x

    # ---- Recover productivity & innovations ---------------------------
    omega_w = phi_w - inputs_w @ beta_hat
    df_w = pd.DataFrame({
        "omega": omega_w,
        "panel_id": panel_w,
        "time": time_w,
    }).sort_values(["panel_id", "time"]).reset_index(drop=True)
    df_w["omega_lag"] = df_w.groupby("panel_id", sort=False)["omega"].shift(1)
    mask_ar = df_w["omega_lag"].notna().to_numpy()
    xi, theta = productivity_residual(
        df_w.loc[mask_ar, "omega"].to_numpy(),
        df_w.loc[mask_ar, "omega_lag"].to_numpy(),
        degree=productivity_degree,
    )

    # AR(1) summary (rho is the linear coefficient when productivity_degree>=1)
    rho = float(theta[1]) if len(theta) > 1 else float("nan")
    sigma_xi = float(np.std(xi, ddof=1))

    # ---- Standard errors -----------------------------------------------
    se = np.full(len(beta_hat), np.nan)
    cov = None
    boot_betas: List[np.ndarray] = []
    if boot_reps and boot_reps > 0:
        rng = np.random.default_rng(seed)
        for _ in range(int(boot_reps)):
            idx = firm_bootstrap_indices(panel_arr, rng)
            try:
                boot_df = df.iloc[idx].reset_index(drop=True)
                # Re-fit stage 1 + stage 2 on the resample
                Z1_b = boot_df[Z1_cols].to_numpy(dtype=float)
                y_b = boot_df[output].to_numpy(dtype=float)
                phi_b, _, _ = stage_one_phi(y_b, Z1_b, degree=polynomial_degree)
                Z2_b_raw = boot_df[instr_cols].to_numpy(dtype=float)
                inputs_b_raw = boot_df[raw_input_names].to_numpy(dtype=float)
                inputs_b, _ = expand_inputs(
                    inputs_b_raw, raw_input_names, functional_form,
                )
                Z2_b, _ = expand_inputs(
                    Z2_b_raw, instr_names, functional_form,
                )
                panel_b = boot_df["__panel_id__"].to_numpy()
                time_b = boot_df["__time__"].to_numpy()
                valid_b = (
                    np.isfinite(phi_b)
                    & np.all(np.isfinite(inputs_b), axis=1)
                    & np.all(np.isfinite(Z2_b), axis=1)
                )
                if valid_b.sum() < 10:
                    continue
                res_b = optimize.minimize(
                    lambda b: gmm_objective(
                        np.asarray(b, dtype=float),
                        phi_hat=phi_b[valid_b],
                        inputs=inputs_b[valid_b],
                        instruments=Z2_b[valid_b],
                        panel_id=panel_b[valid_b],
                        time=time_b[valid_b],
                        productivity_degree=productivity_degree,
                    ),
                    beta_hat,
                    method="Nelder-Mead",
                    options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 8000},
                )
                if not res_b.success:
                    continue
                boot_betas.append(res_b.x)
            except Exception:
                continue
        if len(boot_betas) > 1:
            B = np.vstack(boot_betas)
            cov = np.cov(B.T, ddof=1)
            se = np.std(B, axis=0, ddof=1)

    # ---- Pack result ---------------------------------------------------
    coef = {name: float(beta_hat[i]) for i, name in enumerate(expanded_names)}
    params = pd.Series(beta_hat, index=expanded_names, name="elasticity")
    std_errors = pd.Series(se, index=expanded_names, name="std_error")

    sample = df.loc[valid].reset_index(drop=True)
    sample["omega"] = omega_w
    sample["eta"] = eta_w

    # Firm-time output elasticities (constant for Cobb-Douglas, varying
    # for translog).  Stored on the result so markup() can read them.
    elasticities = elasticities_at(
        sample[raw_input_names].to_numpy(dtype=float),
        raw_input_names,
        coef,
        functional_form=functional_form,
    )

    diagnostics = {
        "stage1_r2": float(1.0 - np.var(eta_hat) / np.var(y)),
        "stage2_objective": float(res.fun),
        "stage2_converged": bool(res.success),
        "ar_rho": rho,
        "ar_sigma_xi": sigma_xi,
        "boot_reps_effective": len(boot_betas),
        "polynomial_degree": int(polynomial_degree),
        "productivity_degree": int(productivity_degree),
    }

    return ProductionResult(
        method=method,
        params=params,
        std_errors=std_errors,
        coef=coef,
        tfp=omega_w,
        residuals=eta_w,
        productivity_process={"rho": rho, "sigma": sigma_xi},
        sample=sample,
        diagnostics=diagnostics,
        model_info={
            "free_inputs": free,
            "state_inputs": state,
            "raw_input_names": raw_input_names,
            "proxy": proxy,
            "functional_form": functional_form,
            "elasticities": elasticities,
        },
        cov=cov,
    )


# ---------------------------------------------------------------------------
# Public wrappers
# ---------------------------------------------------------------------------

def _resolve_inputs(
    free: Optional[Sequence[str] | str],
    state: Optional[Sequence[str] | str],
    free_default: Sequence[str],
    state_default: Sequence[str],
) -> Tuple[List[str], List[str]]:
    def _to_list(x, default):
        if x is None:
            return list(default)
        if isinstance(x, str):
            return [x]
        return list(x)
    return _to_list(free, free_default), _to_list(state, state_default)


def olley_pakes(
    data: pd.DataFrame,
    output: str = "y",
    free: Sequence[str] | str | None = None,
    state: Sequence[str] | str | None = None,
    proxy: str = "i",
    panel_id: str = "id",
    time: str = "year",
    polynomial_degree: int = 3,
    productivity_degree: int = 1,
    functional_form: str = "cobb-douglas",
    boot_reps: int = 0,
    seed: Optional[int] = None,
    drop_zero_proxy: bool = True,
) -> ProductionResult:
    """Olley-Pakes (1996) production function estimator.

    Uses **investment** as the proxy for unobserved productivity. Firms with
    zero investment are dropped by default — the inversion of the
    investment policy requires a strictly positive proxy.

    Parameters
    ----------
    data : DataFrame
        Long-form panel: one row per (firm, year).
    output : str, default ``"y"``
        Log output column.
    free : str or list, default ``["l"]``
        Freely chosen inputs (e.g. labor). Multiple are allowed.
    state : str or list, default ``["k"]``
        State inputs (capital, predetermined).
    proxy : str, default ``"i"``
        Investment column (must be > 0 to invert).
    panel_id, time : str
        Firm and year identifiers.
    polynomial_degree : int, default 3
        Degree of the stage-1 polynomial in (free, state, proxy).
    productivity_degree : int, default 3
        Degree of the polynomial g in the AR productivity process.
    functional_form : {'cobb-douglas', 'translog'}, default 'cobb-douglas'
        Production function form. Translog adds quadratic and cross
        terms; ``ProductionResult.model_info["elasticities"]`` then
        carries firm-time output elasticities.
    boot_reps : int, default 0
        Firm-cluster bootstrap replications. ``0`` ⇒ NaN standard errors.
    seed : int, optional
        Bootstrap RNG seed.
    drop_zero_proxy : bool, default True
        Drop rows with non-positive proxy (required by the OP inversion).
        Note that dropping period ``t`` for a firm also forfeits period
        ``t+1`` for that firm in stage 2 (lag operator now has no
        predecessor), so firms with sporadic zero-investment years lose
        more observations than the raw drop count.

    Returns
    -------
    ProductionResult

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.olley_pakes(df, output="y", free="l", state="k",
    ...                       proxy="i", panel_id="id", time="year",
    ...                       boot_reps=200, seed=0)
    >>> res.coef           # {"l": 0.62, "k": 0.31}
    >>> res.summary()

    References
    ----------
    Olley, G.S. & Pakes, A. (1996). The dynamics of productivity in the
    telecommunications equipment industry. Econometrica, 64(6), 1263-1297.
    """
    free, state = _resolve_inputs(free, state, ["l"], ["k"])
    cols = [output, *free, *state, proxy, panel_id, time]
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")
    df = data.copy()
    if drop_zero_proxy:
        df = df.loc[df[proxy] > 0].copy()
    df, _ = _prepare_panel(df, output, free, state, proxy, panel_id, time)
    return _estimate_proxy(
        df,
        output=output,
        free=free,
        state=state,
        proxy=proxy,
        method="op",
        polynomial_degree=polynomial_degree,
        productivity_degree=productivity_degree,
        functional_form=functional_form,
        boot_reps=boot_reps,
        seed=seed,
    )


def levinsohn_petrin(
    data: pd.DataFrame,
    output: str = "y",
    free: Sequence[str] | str | None = None,
    state: Sequence[str] | str | None = None,
    proxy: str = "m",
    panel_id: str = "id",
    time: str = "year",
    polynomial_degree: int = 3,
    productivity_degree: int = 1,
    functional_form: str = "cobb-douglas",
    boot_reps: int = 0,
    seed: Optional[int] = None,
) -> ProductionResult:
    """Levinsohn-Petrin (2003) production function estimator.

    Uses **intermediate input** (materials / energy) as the productivity
    proxy. Avoids the OP zero-investment selection problem because most
    firms use materials in every period.

    Parameters
    ----------
    data : DataFrame
        Long-form panel.
    output : str
        Log output.
    free : str or list, default ``["l"]``
        Free inputs.
    state : str or list, default ``["k"]``
        State inputs.
    proxy : str, default ``"m"``
        Intermediate input (materials).
    panel_id, time : str
    polynomial_degree, productivity_degree : int
    functional_form : {'cobb-douglas', 'translog'}, default 'cobb-douglas'
    boot_reps : int
    seed : int

    Returns
    -------
    ProductionResult

    References
    ----------
    Levinsohn, J. & Petrin, A. (2003). Estimating production functions
    using inputs to control for unobservables. Review of Economic
    Studies, 70(2), 317-341.
    """
    free, state = _resolve_inputs(free, state, ["l"], ["k"])
    df, _ = _prepare_panel(data, output, free, state, proxy, panel_id, time)
    return _estimate_proxy(
        df,
        output=output,
        free=free,
        state=state,
        proxy=proxy,
        method="lp",
        polynomial_degree=polynomial_degree,
        productivity_degree=productivity_degree,
        functional_form=functional_form,
        boot_reps=boot_reps,
        seed=seed,
    )


def ackerberg_caves_frazer(
    data: pd.DataFrame,
    output: str = "y",
    free: Sequence[str] | str | None = None,
    state: Sequence[str] | str | None = None,
    proxy: str = "m",
    panel_id: str = "id",
    time: str = "year",
    polynomial_degree: int = 3,
    productivity_degree: int = 1,
    functional_form: str = "cobb-douglas",
    boot_reps: int = 0,
    seed: Optional[int] = None,
) -> ProductionResult:
    """Ackerberg-Caves-Frazer (2015) production function estimator.

    Corrects the OP / LP "functional dependence" identification problem:
    when free inputs (labor) are chosen at the same time as the proxy,
    the labor coefficient is *not* identified in the stage-1 polynomial.
    ACF moves all coefficient identification to stage 2, instrumenting
    free inputs with their *lagged* values and state inputs at the
    contemporaneous level.

    Parameters
    ----------
    data : DataFrame
    output : str
    free : str or list, default ``["l"]``
        Free inputs — instrumented with their lag in stage 2.
    state : str or list, default ``["k"]``
    proxy : str, default ``"m"``
        Intermediate input (materials).
    panel_id, time : str
    polynomial_degree, productivity_degree : int
    functional_form : {'cobb-douglas', 'translog'}, default 'cobb-douglas'
    boot_reps : int
    seed : int

    Returns
    -------
    ProductionResult

    Notes
    -----
    Requires at least two consecutive time periods per firm so that
    lagged labor exists.

    References
    ----------
    Ackerberg, D.A., Caves, K. & Frazer, G. (2015). Identification
    properties of recent production function estimators. Econometrica,
    83(6), 2411-2451.
    """
    free, state = _resolve_inputs(free, state, ["l"], ["k"])
    df, _ = _prepare_panel(data, output, free, state, proxy, panel_id, time)
    return _estimate_proxy(
        df,
        output=output,
        free=free,
        state=state,
        proxy=proxy,
        method="acf",
        polynomial_degree=polynomial_degree,
        productivity_degree=productivity_degree,
        functional_form=functional_form,
        boot_reps=boot_reps,
        seed=seed,
    )
