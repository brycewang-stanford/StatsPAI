"""
Recentered instruments for treatments built from exogenous shocks.

Many treatments and instruments combine as-good-as-random shocks ``g``
with non-random exposure ``w`` through a known formula,
``z_i = f_i(g, w)``: market access from new rail lines, simulated
eligibility from state policies, spillovers through a network. Even
when ``g`` is randomly assigned, units with high exposure get high
``z_i`` in every draw of the shocks, so ``z_i`` is correlated with
anything that drives exposure. Borusyak and Hull show that the bias
runs through one variable, the expected instrument::

    mu_i = E[ f_i(g, w) | w ]

which the researcher computes by recomputing ``z`` under counterfactual
shock draws that were as likely to occur as the realised one and
averaging them. Two estimators then identify ``beta``:

- ``method='recenter'``: instrument the treatment with
  ``z_i - mu_i`` (Table 1, column 2 of the paper);
- ``method='control'``: instrument with ``z_i`` and add ``mu_i`` as a
  control (column 3).

When the treatment is the exposure itself (``treat == exposure``) the
first is an IV regression of ``y`` on ``z`` instrumented by
``z - mu`` and the second is OLS of ``y`` on ``z`` and ``mu``.

The same draws give randomization inference. Under the sharp null
``beta = b`` the statistic ``T(b) = sum_i w_i (z_i - mu_i)(y_i - b x_i)``
(``y`` and ``x`` residualised on the controls) has a known distribution
over the counterfactual shocks; the recentered estimator is the
Hodges-Lehmann estimator that sets ``T`` to its randomization mean.
The confidence set collects every ``b`` that is not rejected and is
computed exactly, so it may be unbounded or a union of intervals when
the shocks move the treatment little.

References
----------
[@borusyak2023nonrandom]
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .._input_validation import check_columns
from ..core.results import CausalResult
from ..exceptions import (
    AssumptionViolation,
    AssumptionWarning,
    DataInsufficient,
    MethodIncompatibility,
)

__all__ = ["recentered_iv"]

_VCE_TO_ROBUST = {
    "robust": "hc1",
    "hc0": "hc0",
    "hc1": "hc1",
    "hc2": "hc2",
    "hc3": "hc3",
    "nonrobust": "nonrobust",
}


# --------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------


def _counterfactual_matrix(
    data: pd.DataFrame, counterfactual: Any
) -> Tuple[np.ndarray, List[str]]:
    """Return the (n, S) matrix of counterfactual exposures and its labels."""
    if isinstance(counterfactual, str):
        counterfactual = [counterfactual]
    if isinstance(counterfactual, pd.DataFrame):
        if len(counterfactual) != len(data):
            raise MethodIncompatibility(
                f"counterfactual has {len(counterfactual)} rows but data has "
                f"{len(data)}",
                recovery_hint=(
                    "Pass one row per observation of `data`, in the same order."
                ),
            )
        return (
            counterfactual.to_numpy(dtype=float),
            [str(c) for c in counterfactual.columns],
        )
    if isinstance(counterfactual, (list, tuple)) and all(
        isinstance(c, str) for c in counterfactual
    ):
        check_columns(
            data,
            {"counterfactual": list(counterfactual)},
            function="recentered_iv",
        )
        cols = list(counterfactual)
        return data[cols].to_numpy(dtype=float), cols
    arr = np.asarray(counterfactual, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or arr.shape[0] != len(data):
        raise MethodIncompatibility(
            f"counterfactual must be an (n, S) array with n = {len(data)} rows; "
            f"got shape {arr.shape}",
            recovery_hint=(
                "Stack the exposures recomputed under each counterfactual "
                "shock draw as columns, one row per observation."
            ),
        )
    return arr, [f"draw_{s}" for s in range(arr.shape[1])]


def _wls_residualize(V: np.ndarray, C: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Residuals of each column of ``V`` from a weighted projection on ``C``."""
    sw = np.sqrt(w)[:, None]
    coef, *_ = np.linalg.lstsq(C * sw, V * sw, rcond=None)
    return V - C @ coef


def _wvar(v: np.ndarray, w: np.ndarray) -> float:
    m = np.sum(w * v) / np.sum(w)
    return float(np.sum(w * (v - m) ** 2) / np.sum(w))


# --------------------------------------------------------------------
# Randomization inference
# --------------------------------------------------------------------


def _ri_pvalue(a: float, c: float, a_s: np.ndarray, c_s: np.ndarray, b: float) -> float:
    """Two-sided RI p-value of ``beta = b`` (the realised draw counted once)."""
    t_obs = abs(a - b * c)
    t_sim = np.abs(a_s - b * c_s)
    tol = 1e-12 * max(t_obs, float(np.max(t_sim, initial=0.0)), 1e-300)
    exceed = int(np.sum(t_sim >= t_obs - tol))
    return (1.0 + exceed) / (len(a_s) + 1.0)


def _ri_confidence_set(
    a: float,
    c: float,
    a_s: np.ndarray,
    c_s: np.ndarray,
    alpha: float,
) -> List[Tuple[float, float]]:
    """Exact inversion of the RI test over ``b``.

    Draw ``s`` counts against ``b`` when ``|a_s - b c_s| >= |a - b c|``,
    i.e. when the quadratic ``f_s(b) = A b^2 + B b + C >= 0``. The count
    is piecewise constant between the roots of the ``f_s``, so a sweep
    over the sorted roots gives the accepted set exactly.
    """
    S = len(a_s)
    A = c_s**2 - c**2
    B = -2.0 * (a_s * c_s - a * c)
    C0 = a_s**2 - a**2
    scale_a = np.maximum(c_s**2 + c**2, 1e-300)
    scale_b = np.maximum(np.abs(a_s * c_s) + abs(a * c), 1e-300)
    quad = np.abs(A) > 1e-12 * scale_a
    lin = (~quad) & (np.abs(B) > 1e-12 * scale_b)

    count0 = 0
    roots: List[float] = []
    deltas: List[int] = []
    for s in range(S):
        if quad[s]:
            disc = B[s] ** 2 - 4.0 * A[s] * C0[s]
            if A[s] > 0:
                count0 += 1
            if disc <= 0:
                continue
            sq = np.sqrt(disc)
            r1, r2 = sorted(((-B[s] - sq) / (2 * A[s]), (-B[s] + sq) / (2 * A[s])))
            sign = -1 if A[s] > 0 else 1
            roots += [r1, r2]
            deltas += [sign, -sign]
        elif lin[s]:
            if B[s] < 0:
                count0 += 1
            roots.append(-C0[s] / B[s])
            deltas.append(1 if B[s] > 0 else -1)
        elif C0[s] >= 0:
            count0 += 1

    def accepted(count: int) -> bool:
        return (1.0 + count) / (S + 1.0) > alpha

    if not roots:
        return [(-np.inf, np.inf)] if accepted(count0) else []
    order = np.argsort(roots)
    r_sorted = np.asarray(roots)[order]
    d_sorted = np.asarray(deltas)[order]
    counts = count0 + np.concatenate([[0], np.cumsum(d_sorted)])
    # Segment k runs from r_sorted[k-1] to r_sorted[k] (open ends at +-inf).
    edges = np.concatenate([[-np.inf], r_sorted, [np.inf]])
    intervals: List[Tuple[float, float]] = []
    for k, cnt in enumerate(counts):
        if not accepted(int(cnt)):
            continue
        lo, hi = float(edges[k]), float(edges[k + 1])
        if intervals and intervals[-1][1] >= lo:
            intervals[-1] = (intervals[-1][0], hi)
        else:
            intervals.append((lo, hi))
    return intervals


def _balance_tests(
    zt: np.ndarray,
    zt_s: np.ndarray,
    R: np.ndarray,
    names: Sequence[str],
    w: np.ndarray,
) -> pd.DataFrame:
    """RI falsification tests that ``z - mu`` is orthogonal to each ``r``."""
    rows = []
    S = zt_s.shape[1]
    for j, name in enumerate(names):
        r = R[:, j]
        t = float(np.sum(w * zt * r) / np.sum(w))
        t_s = (w * r) @ zt_s / np.sum(w)
        tol = 1e-12 * max(abs(t), float(np.max(np.abs(t_s), initial=0.0)), 1e-300)
        p = (1.0 + np.sum(np.abs(t_s) >= abs(t) - tol)) / (S + 1.0)
        rows.append({"variable": name, "statistic": t, "ri_pvalue": float(p)})
    if R.shape[1] > 1:
        sw = np.sqrt(w)
        Rw = R * sw[:, None]
        proj = Rw @ np.linalg.pinv(Rw.T @ Rw) @ Rw.T

        def quad(v: np.ndarray) -> float:
            vw = v * sw
            return float(vw @ proj @ vw)

        t = quad(zt)
        t_s = np.array([quad(zt_s[:, s]) for s in range(S)])
        tol = 1e-12 * max(t, float(np.max(t_s, initial=0.0)), 1e-300)
        p = (1.0 + np.sum(t_s >= t - tol)) / (S + 1.0)
        rows.append({"variable": "joint", "statistic": t, "ri_pvalue": float(p)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------


def recentered_iv(
    data: pd.DataFrame,
    y: str,
    treat: str,
    exposure: str,
    counterfactual: Union[Sequence[str], np.ndarray, pd.DataFrame],
    covariates: Optional[Sequence[str]] = None,
    weights: Optional[str] = None,
    cluster: Optional[str] = None,
    vce: Literal["robust", "hc0", "hc1", "hc2", "hc3", "nonrobust"] = "robust",
    method: Literal["recenter", "control"] = "recenter",
    ri: bool = True,
    ri_null: float = 0.0,
    balance: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
) -> CausalResult:
    """Recentered IV for treatments built from exogenous shocks (Borusyak-Hull).

    Adjusts for non-random exposure to as-good-as-random shocks by
    subtracting (or controlling for) the expected instrument, the
    average of the instrument across counterfactual shock draws, and
    reports randomization inference from the same draws.

    Parameters
    ----------
    data : DataFrame
        One row per observation.
    y : str
        Outcome column.
    treat : str
        Treatment column. May equal ``exposure`` when the treatment is
        itself built from the shocks (the paper's OLS case).
    exposure : str
        Realised instrument ``z_i = f_i(g, w)``, computed from the
        realised shocks.
    counterfactual : list of str, (n, S) array or DataFrame
        The same instrument recomputed under ``S >= 2`` counterfactual
        shock draws (for example permutations of the shocks), one column
        per draw, rows aligned with ``data``. The expected instrument is
        their row mean. The realised draw is not added; include it as a
        column if the draws enumerate the whole permutation class.
    covariates : list of str, optional
        Predetermined controls, entered linearly with an intercept.
    weights : str, optional
        Analytic weights column.
    cluster : str, optional
        Cluster column for the asymptotic standard error.
    vce : {'robust', 'hc0', 'hc1', 'hc2', 'hc3', 'nonrobust'}, default 'robust'
        Asymptotic variance when ``cluster`` is not given; ``'robust'``
        is HC1.
    method : {'recenter', 'control'}, default 'recenter'
        ``'recenter'`` instruments with ``exposure - mu``; ``'control'``
        instruments with ``exposure`` and controls for ``mu``.
    ri : bool, default True
        Compute the randomization p-value and confidence set.
    ri_null : float, default 0.0
        Value of ``beta`` under the null for the RI p-value.
    balance : list of str, optional
        Predetermined variables for RI falsification tests of
        ``E[(z - mu) r] = 0`` (one test per variable and a joint test).
    alpha : float, default 0.05
        Level of the confidence intervals.

    Returns
    -------
    CausalResult
        ``estimate`` / ``se`` / ``ci`` are the IV (or OLS) estimate and
        its asymptotic inference. ``diagnostics`` holds the RI p-value
        (``ri_pvalue``), the RI confidence set (``ri_ci``, a list of
        intervals, possibly unbounded), the unadjusted estimate, the
        first-stage F, and how much of the variance of the instrument
        the recentering removes. ``detail`` compares the unadjusted and
        adjusted estimates.

    Notes
    -----
    The RI reference set is the realised draw together with the ``S``
    counterfactual ones, all centred at their common mean. With
    ``method='recenter'`` this centring only rescales the instrument by
    ``S / (S + 1)``, so the Hodges-Lehmann estimator of the test
    (``ri_hodges_lehmann``) equals the reported estimate for either
    method. With ``method='recenter'`` the RI test is exact for the
    sharp null of a constant effect whenever the realised and the
    counterfactual draws are exchangeable (for example independent
    random permutations of the shocks). With ``method='control'`` the
    control ``mu`` excludes the realised draw, so exactness holds only
    as ``S`` grows. With heterogeneous effects the IV estimand is a
    convex average of effects and the asymptotic standard error is the
    one to report.

    The paper's spatial (Conley) standard errors are not offered here;
    use ``cluster`` or compute them with ``sp.conley`` on the
    recentered instrument.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n, S = 300, 200
    >>> expo = rng.uniform(0, 2, n)          # non-random exposure
    >>> def draw():
    ...     return expo * rng.binomial(1, 0.5, n)   # shocks hit exposed units
    >>> z = draw()
    >>> cf = np.column_stack([draw() for _ in range(S)])
    >>> y = 1.0 * z + 2.0 * expo + rng.normal(size=n)   # exposure confounds
    >>> df = pd.DataFrame({"y": y, "z": z})
    >>> res = sp.recentered_iv(df, "y", "z", "z", cf)
    >>> abs(res.estimate - 1.0) < 0.5
    True

    References
    ----------
    [@borusyak2023nonrandom]
    """
    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility(
            "data must be a pandas DataFrame",
            recovery_hint="Pass the data as a DataFrame.",
        )
    if method not in ("recenter", "control"):
        raise MethodIncompatibility(
            f"method must be 'recenter' or 'control', got {method!r}",
            recovery_hint="Use method='recenter' or method='control'.",
        )
    if vce not in _VCE_TO_ROBUST:
        raise MethodIncompatibility(
            f"vce must be one of {sorted(_VCE_TO_ROBUST)}, got {vce!r}",
            recovery_hint="Use vce='robust', or pass cluster= for clustering.",
        )
    if not 0 < alpha < 1:
        raise MethodIncompatibility(
            f"alpha must be in (0, 1), got {alpha!r}",
            recovery_hint="Use alpha=0.05 for 95% intervals.",
        )
    covariates = list(covariates or [])
    balance = list(balance or [])
    named = {"y": y, "treat": treat, "exposure": exposure}
    if covariates:
        named["covariates"] = covariates
    if balance:
        named["balance"] = balance
    if weights is not None:
        named["weights"] = weights
    if cluster is not None:
        named["cluster"] = cluster
    check_columns(data, named, function="recentered_iv")

    Zcf, cf_names = _counterfactual_matrix(data, counterfactual)
    S = Zcf.shape[1]
    if S < 2:
        raise DataInsufficient(
            f"recentered_iv needs at least 2 counterfactual draws, got {S}",
            recovery_hint=(
                "Recompute the exposure under many counterfactual shock draws "
                "(hundreds of permutations is typical)."
            ),
        )

    numeric = [y, treat, exposure] + covariates + balance
    if weights is not None:
        numeric.append(weights)
    frame = data[list(dict.fromkeys(numeric))].apply(pd.to_numeric, errors="coerce")
    keep = np.isfinite(frame.to_numpy(dtype=float)).all(axis=1)
    keep &= np.isfinite(Zcf).all(axis=1)
    if cluster is not None:
        keep &= data[cluster].notna().to_numpy()
    n_dropped = int((~keep).sum())
    if n_dropped:
        warnings.warn(
            f"recentered_iv dropped {n_dropped} rows with missing values in "
            "the outcome, treatment, exposure, controls or counterfactual draws.",
            AssumptionWarning,
            stacklevel=2,
        )
    idx = np.flatnonzero(keep)
    n = idx.size
    if n < 3:
        raise DataInsufficient(
            f"only {n} complete rows remain",
            recovery_hint="Check the missing values in the inputs.",
        )

    yv = frame[y].to_numpy(dtype=float)[idx]
    xv = frame[treat].to_numpy(dtype=float)[idx]
    zv = frame[exposure].to_numpy(dtype=float)[idx]
    Zc = Zcf[idx]
    wv = (
        frame[weights].to_numpy(dtype=float)[idx] if weights is not None else np.ones(n)
    )
    if np.any(wv < 0) or not np.any(wv > 0):
        raise MethodIncompatibility(
            "weights must be non-negative and not all zero",
            recovery_hint="Check the weights column.",
        )
    Xc = (
        frame[covariates].to_numpy(dtype=float)[idx] if covariates else np.empty((n, 0))
    )

    mu = Zc.mean(axis=1)
    zt = zv - mu
    zscale = max(float(np.max(np.abs(zv))), float(np.max(np.abs(Zc))), 1e-300)
    if float(np.max(np.abs(zt))) <= 1e-12 * zscale:
        raise AssumptionViolation(
            "the realised exposure equals its expectation for every unit, so "
            "the shocks leave no variation to use",
            recovery_hint=(
                "Check that the counterfactual draws re-randomise the shocks "
                "rather than copy the realised ones."
            ),
        )
    mu_sd = float(np.sqrt(_wvar(mu, wv)))
    mu_constant = mu_sd <= 1e-10 * zscale
    if mu_constant:
        warnings.warn(
            "The expected exposure is the same for every unit, so the "
            "intercept already absorbs it and the adjusted estimate equals "
            "the unadjusted one.",
            AssumptionWarning,
            stacklevel=2,
        )

    same = treat == exposure
    # Working frame with safe names so any column name works in a formula.
    work = pd.DataFrame({"_rc_y": yv, "_rc_x": xv, "_rc_z": zv, "_rc_zt": zt})
    work["_rc_mu"] = mu
    cov_names = [f"_rc_c{j}" for j in range(Xc.shape[1])]
    for j, nm in enumerate(cov_names):
        work[nm] = Xc[:, j]
    fit_kw: Dict[str, Any] = {"robust": _VCE_TO_ROBUST[vce]}
    if weights is not None:
        work["_rc_w"] = wv
        fit_kw["weights"] = "_rc_w"
    if cluster is not None:
        work["_rc_g"] = data[cluster].to_numpy()[idx]
        fit_kw["cluster"] = "_rc_g"

    from ..regression.iv import iv as _iv_api
    from ..regression.ols import regress as _regress

    ctrl = list(cov_names)
    use_mu = method == "control" and not mu_constant
    if use_mu:
        ctrl.append("_rc_mu")
    rhs_ctrl = "".join(f" + {c}" for c in ctrl)

    def _fit_iv(instr: str, controls: str) -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _iv_api(
                f"_rc_y ~ (_rc_x ~ {instr}){controls}",
                data=work,
                alpha=alpha,
                **fit_kw,
            )

    def _fit_ols(controls: str) -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _regress(f"_rc_y ~ _rc_x{controls}", data=work, **fit_kw)

    if method == "recenter":
        if same:
            work["_rc_x"] = zv
        main = _fit_iv("_rc_zt", rhs_ctrl)
    elif same:
        main = _fit_ols(rhs_ctrl)
    else:
        main = _fit_iv("_rc_z", rhs_ctrl)
    naive = (
        _fit_ols("".join(f" + {c}" for c in cov_names))
        if same
        else _fit_iv("_rc_z", "".join(f" + {c}" for c in cov_names))
    )

    est = float(main.params["_rc_x"])
    se = float(main.std_errors["_rc_x"])
    pval = float(main.pvalues["_rc_x"])
    ci_df = main.conf_int(alpha=alpha)
    ci = (float(ci_df.loc["_rc_x"].iloc[0]), float(ci_df.loc["_rc_x"].iloc[1]))

    model_info: Dict[str, Any] = {
        "method": method,
        "n_draws": S,
        "n_dropped": n_dropped,
        "treatment_is_exposure": bool(same),
        "vce": "cluster" if cluster is not None else vce,
        "unadjusted_estimate": float(naive.params["_rc_x"]),
        "unadjusted_se": float(naive.std_errors["_rc_x"]),
        "expected_exposure_mean": float(np.average(mu, weights=wv)),
        "expected_exposure_sd": mu_sd,
        "expected_exposure_constant": bool(mu_constant),
        "var_ratio_recentered": _wvar(zt, wv) / max(_wvar(zv, wv), 1e-300),
        "r2_exposure_on_expected": (
            0.0
            if mu_constant
            else float(np.corrcoef(np.sqrt(wv) * zv, np.sqrt(wv) * mu)[0, 1] ** 2)
        ),
        "inference_rule": main.inference_rule(),
    }
    fs_f = (
        main.model_info.get("first_stage_f")
        if not (method == "control" and same)
        else None
    )
    model_info["first_stage_f"] = None if fs_f is None else float(fs_f)
    if use_mu:
        model_info["mu_coefficient"] = float(main.params["_rc_mu"])
        model_info["mu_se"] = float(main.std_errors["_rc_mu"])

    detail = pd.DataFrame(
        [
            {
                "estimator": "unadjusted",
                "estimate": model_info["unadjusted_estimate"],
                "se": model_info["unadjusted_se"],
            },
            {"estimator": method, "estimate": est, "se": se},
        ]
    )

    # Randomization inference treats the realised draw as one of S + 1
    # equally likely draws and centres all of them at their common mean,
    # so the reference set does not depend on which draw was realised.
    mu_all = (zv + Zc.sum(axis=1)) / (S + 1.0)
    zt_all = zv - mu_all
    if ri:
        C = np.column_stack([np.ones(n), Xc] + ([mu] if use_mu else []))
        resid = _wls_residualize(np.column_stack([yv, xv]), C, wv)
        y_r, x_r = resid[:, 0], resid[:, 1]
        Zt_s = Zc - mu_all[:, None]
        a = float(np.sum(wv * zt_all * y_r))
        c = float(np.sum(wv * zt_all * x_r))
        a_s = (wv * y_r) @ Zt_s
        c_s = (wv * x_r) @ Zt_s
        sets = _ri_confidence_set(a, c, a_s, c_s, alpha)
        model_info["ri_null"] = float(ri_null)
        model_info["ri_pvalue"] = _ri_pvalue(a, c, a_s, c_s, ri_null)
        model_info["ri_ci"] = sets
        model_info["ri_ci_bounded"] = bool(
            sets and np.isfinite(sets[0][0]) and np.isfinite(sets[-1][1])
        )
        model_info["ri_hodges_lehmann"] = a / c if abs(c) > 0 else float("nan")
        if not model_info["ri_ci_bounded"]:
            warnings.warn(
                "The randomization confidence set is unbounded: the "
                "counterfactual shocks move the treatment too little to "
                "bound the effect.",
                AssumptionWarning,
                stacklevel=2,
            )

    if balance:
        Rb = frame[balance].to_numpy(dtype=float)[idx]
        model_info["balance"] = _balance_tests(
            zt_all, Zc - mu_all[:, None], Rb, balance, wv
        ).to_dict(orient="records")

    return CausalResult(
        method=(
            "Recentered IV (Borusyak-Hull)"
            if method == "recenter"
            else "Expected-instrument control (Borusyak-Hull)"
        ),
        estimand="beta",
        estimate=est,
        se=se,
        pvalue=pval,
        ci=ci,
        alpha=float(alpha),
        n_obs=int(n),
        detail=detail,
        model_info=model_info,
        _citation_key="borusyak2023nonrandom",
    )


# Kept in sync with paper.bib (key borusyak2023nonrandom).
CausalResult._CITATIONS["borusyak2023nonrandom"] = (
    "@article{borusyak2023nonrandom,\n"
    "  title={Nonrandom Exposure to Exogenous Shocks},\n"
    "  author={Borusyak, Kirill and Hull, Peter},\n"
    "  journal={Econometrica},\n"
    "  volume={91},\n"
    "  number={6},\n"
    "  pages={2155--2185},\n"
    "  year={2023},\n"
    "  doi={10.3982/ecta19367}\n"
    "}"
)
