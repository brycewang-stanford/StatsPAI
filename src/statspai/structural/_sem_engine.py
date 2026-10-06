"""Normal-theory maximum likelihood for linear structural equation models.

One engine for path models among observed variables, measurement models
with latent variables, and models with a mean structure. With ``v`` the
stacked vector of observed and latent variables,

    v = alpha + B v + zeta,    Var(zeta) = Psi,

so that, with ``A = (I - B)^{-1}``, the variables have covariance
``A Psi A'`` and mean ``A alpha``; the observed ones are a sub-block. A
loading is an entry of ``B`` like any other path.

:func:`statspai.path_analysis` parses the model and calls :func:`fit_sem`.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats

from ..exceptions import DataInsufficient, MethodIncompatibility
from ._sem_missing import _patterns, _saturated_moments

__all__ = ["fit_sem"]

_OP = {"load": "=~", "reg": "~", "psi": "~~", "int": "~1"}


def _columns(data: pd.DataFrame, names: List[str]) -> pd.DataFrame:
    frame = pd.DataFrame(index=data.index)
    for v in names:
        if v in data.columns:
            col = data[v]
        elif ":" in v and all(part in data.columns for part in v.split(":")):
            col = data[v.split(":")[0]].astype(float)
            for part in v.split(":")[1:]:
                col = col * data[part].astype(float)
        else:
            raise MethodIncompatibility(
                f"path_analysis: variable {v!r} is not a column of data.",
                recovery_hint="A latent variable needs a '=~' line that " "defines it.",
                diagnostics={"columns": [str(c) for c in data.columns][:30]},
            )
        if not (pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col)):
            raise MethodIncompatibility(
                f"path_analysis: column {v!r} is not numeric.",
                recovery_hint="Encode it as numbers (a two-level factor as 0/1).",
            )
        frame[v] = col.astype(float)
    return frame


def _ncp(chisq: float, df: int, target: float) -> float:
    """Noncentrality at which chisq sits at the given upper tail."""
    if df == 0 or stats.chi2.sf(chisq, df) >= target:
        return 0.0

    def gap(lam: float) -> float:
        return float(stats.ncx2.sf(chisq, df, lam)) - target

    hi = max(chisq, 1.0)
    while gap(hi) < 0:
        hi *= 2.0
    return float(optimize.brentq(gap, 1e-12, hi))


def fit_sem(
    spec: Dict[str, Any],
    data: pd.DataFrame,
    *,
    se: str,
    alpha: float,
    meanstructure: bool,
    std_lv: bool,
    growth: bool,
    auto_cov_y: bool,
    evaluate: Callable[[str, Dict[str, float]], float],
    missing: str = "listwise",
) -> Dict[str, Any]:
    """Fit the parsed model; returns the pieces of the result object."""
    loadings = spec["loadings"]
    regressions = spec["regressions"]
    latents: List[str] = []
    for lat, _, _, _ in loadings:
        if lat not in latents:
            latents.append(lat)
    for lat in latents:
        if lat in data.columns:
            raise MethodIncompatibility(
                f"path_analysis: the latent variable {lat!r} has the name of "
                "a column of data.",
                recovery_hint="Rename the latent variable.",
            )
    lat_set = set(latents)
    indicators = {ind for _, ind, _, _ in loadings}
    lhs_reg = {lhs for lhs, _, _, _ in regressions}
    rhs_reg = {var for _, var, _, _ in regressions}

    seen_order: List[str] = []
    for _, ind, _, _ in loadings:
        seen_order.append(ind)
    for lhs, _, _, _ in regressions:
        seen_order.append(lhs)
    for _, var, _, _ in regressions:
        seen_order.append(var)
    for a, b, _, _ in spec["covariances"]:
        seen_order.extend([a, b])
    for var, _, _ in spec["intercepts"]:
        seen_order.append(var)
    observed: List[str] = []
    for v in seen_order:
        if v not in lat_set and v not in observed:
            observed.append(v)
    ov_x = [
        v for v in observed
        if v in rhs_reg and v not in lhs_reg and v not in indicators
    ]  # fmt: skip
    ov_y = [v for v in observed if v not in ov_x]
    names = ov_y + ov_x + latents
    q, k, n_lat = len(ov_y), len(ov_x), len(latents)
    p = q + k
    m = p + n_lat
    idx = {v: i for i, v in enumerate(names)}
    is_x = np.zeros(m, dtype=bool)
    is_x[q:p] = True
    is_lat = np.zeros(m, dtype=bool)
    is_lat[p:] = True

    frame = _columns(data, ov_y + ov_x)
    n_all = len(frame)
    fiml = missing == "fiml"
    if fiml:
        # a row needs its exogenous variables, which are conditioned on, and
        # at least one endogenous value
        usable = frame[ov_y].notna().any(axis=1)
        if ov_x:
            usable &= frame[ov_x].notna().all(axis=1)
        frame = frame[usable]
        fiml = bool(frame.isna().to_numpy().any())  # else: complete data
    else:
        frame = frame.dropna()
    n = len(frame)
    if n <= p + 1:
        raise DataInsufficient(f"path_analysis: {n} usable rows for {p} variables.")
    Z = frame.to_numpy()
    patterns: List[Dict[str, Any]] = []
    if fiml:
        if se == "robust":
            raise MethodIncompatibility(
                "path_analysis: se='robust' is not available with " "missing='fiml'.",
                recovery_hint="Use se='standard' (observed information), or "
                "missing='listwise' with se='robust'.",
            )
        patterns = _patterns(Z)
        zbar, S = _saturated_moments(Z, patterns)
        Zc = np.zeros((0, p))
    else:
        zbar = Z.mean(axis=0)
        Zc = Z - zbar
        S = Zc.T @ Zc / n
    if np.linalg.matrix_rank(S) < p:
        raise DataInsufficient(
            "path_analysis: the variables are linearly dependent; the sample "
            "covariance matrix is singular."
        )
    Sxx = S[q:, q:]
    # full information implies a mean structure, missing values or not
    mean_on = bool(meanstructure or growth or spec["intercepts"] or missing == "fiml")

    # ---- parameter table ------------------------------------------------
    entries: List[Dict[str, Any]] = []
    label_to_free: Dict[str, int] = {}
    n_free = 0

    def add(
        kind: str, i: int, j: int, label: Optional[str], fixed: Optional[float]
    ) -> None:
        nonlocal n_free
        entry: Dict[str, Any] = {
            "kind": kind, "i": i, "j": j, "label": label or "", "fixed": fixed,
        }  # fmt: skip
        if fixed is not None:
            entry["free"] = None
        elif label and label in label_to_free:
            entry["free"] = label_to_free[label]
        else:
            entry["free"] = n_free
            if label:
                label_to_free[label] = n_free
            n_free += 1
        entries.append(entry)

    def free_or(fixed: Optional[float], default: Optional[float]) -> Optional[float]:
        """A modifier of NaN (lavaan's ``NA*``) frees the parameter."""
        if fixed is None:
            return default
        return None if math.isnan(fixed) else fixed

    paths: Set[Tuple[int, int]] = set()
    first_of: Set[str] = set()
    merged: Dict[Tuple[str, str], List[Any]] = {}
    for lat, ind, label, fixed in loadings:  # f =~ NA*a + l*a: one loading
        if (lat, ind) in merged:
            slot = merged[(lat, ind)]
            slot[0] = label or slot[0]
            slot[1] = fixed if fixed is not None else slot[1]
        else:
            merged[(lat, ind)] = [label, fixed]
    for (lat, ind), (label, fixed) in merged.items():
        if lat == ind:
            raise MethodIncompatibility(f"path_analysis: {lat!r} is its own indicator.")
        default = None
        if lat not in first_of:
            first_of.add(lat)
            if not std_lv:
                default = 1.0  # the marker indicator sets the scale
        paths.add((idx[ind], idx[lat]))
        add("load", idx[ind], idx[lat], label, free_or(fixed, default))
    for lhs, var, label, fixed in regressions:
        if lhs == var:
            raise MethodIncompatibility(
                f"path_analysis: {lhs!r} is regressed on itself."
            )
        if (idx[lhs], idx[var]) in paths:
            raise MethodIncompatibility(
                f"path_analysis: the path {lhs} ~ {var} appears twice."
            )
        paths.add((idx[lhs], idx[var]))
        add("reg", idx[lhs], idx[var], label, free_or(fixed, None))

    explicit_var: Dict[int, Tuple[Optional[str], Optional[float]]] = {}
    cov_pairs: Set[Tuple[int, int]] = set()
    for a, b, label, fixed in spec["covariances"]:
        for v in (a, b):
            if v not in idx:
                raise MethodIncompatibility(
                    f"path_analysis: {v!r} in '{a} ~~ {b}' is not a variable "
                    "of the model."
                )
        ia, ib = idx[a], idx[b]
        if is_x[ia] and is_x[ib]:
            continue  # exogenous (co)variances are held at the sample values
        if is_x[ia] or is_x[ib]:
            raise MethodIncompatibility(
                f"path_analysis: '{a} ~~ {b}' relates a disturbance to an "
                "exogenous variable, which would make that variable "
                "endogenous.",
                recovery_hint="Add an equation for it, or drop the covariance.",
            )
        if ia == ib:
            explicit_var[ia] = (label, fixed)
            continue
        pair = (max(ia, ib), min(ia, ib))
        if pair in cov_pairs:
            continue
        cov_pairs.add(pair)
        add("psi", ia, ib, label, free_or(fixed, None))

    # Covariances nobody wrote: exogenous latent variables covary; terminal
    # outcomes do only on request.
    dependent = {idx[v] for v in lhs_reg} | {idx[v] for v in indicators}
    lv_exo = [idx[v] for v in latents if idx[v] not in dependent]
    auto_groups = [lv_exo]
    if auto_cov_y:
        auto_groups.append(
            [
                idx[v] for v in ov_y + latents
                if v in lhs_reg and v not in rhs_reg and v not in indicators
            ]  # fmt: skip
        )
    for group in auto_groups:
        for a_, ia in enumerate(group):
            for ib in group[a_ + 1 :]:
                pair = (max(ia, ib), min(ia, ib))
                if pair not in cov_pairs:
                    cov_pairs.add(pair)
                    add("psi", ia, ib, None, None)
    for i in list(range(q)) + list(range(p, m)):
        label, fixed = explicit_var.get(i, (None, None))
        default = 1.0 if (std_lv and is_lat[i]) else None
        add("psi", i, i, label, free_or(fixed, default))

    if mean_on:
        explicit_int = {idx[v]: (lab, fx) for v, lab, fx in spec["intercepts"]
                        if v in idx}  # fmt: skip
        for v, _, _ in spec["intercepts"]:
            if v not in idx:
                raise MethodIncompatibility(
                    f"path_analysis: {v!r} in '{v} ~ 1' is not a variable of "
                    "the model."
                )
            if is_x[idx[v]]:
                raise MethodIncompatibility(
                    f"path_analysis: {v!r} is exogenous; its mean is held at "
                    "the sample value and cannot be modelled.",
                )
        for i in list(range(q)) + list(range(p, m)):
            label, fixed = explicit_int.get(i, (None, None))
            # growth curves put the means on the latent variables
            default = (
                (None if growth else 0.0) if is_lat[i] else (0.0 if growth else None)
            )
            if i in explicit_int and fixed is None:
                default = None  # 'v ~ 1' frees it
            add("int", i, i, label, free_or(fixed, default))

    n_moments = p * (p + 1) // 2 - k * (k + 1) // 2 + (q if mean_on else 0)
    df_model = n_moments - n_free
    if df_model < 0:
        raise MethodIncompatibility(
            f"path_analysis: the model has {n_free} free parameters and only "
            f"{n_moments} moments; it is not identified.",
            recovery_hint="Remove a path or a residual covariance, or add "
            "indicators.",
        )

    # ---- implied moments and their derivatives ---------------------------
    def matrices(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        B = np.zeros((m, m))
        Psi = np.zeros((m, m))
        a0 = np.zeros(m)
        Psi[q:p, q:p] = Sxx
        a0[q:p] = zbar[q:]
        for e in entries:
            val = e["fixed"] if e["free"] is None else theta[e["free"]]
            if e["kind"] in ("load", "reg"):
                B[e["i"], e["j"]] = val
            elif e["kind"] == "psi":
                Psi[e["i"], e["j"]] = val
                Psi[e["j"], e["i"]] = val
            else:
                a0[e["i"]] = val
        return B, Psi, a0

    def implied(theta: np.ndarray) -> Dict[str, np.ndarray]:
        B, Psi, a0 = matrices(theta)
        A = np.linalg.inv(np.eye(m) - B)
        full = A @ Psi @ A.T
        eta = A @ a0
        return {"A": A, "Psi": Psi, "full": full, "eta": eta,
                "Sigma": full[:p, :p], "mu": eta[:p]}  # fmt: skip

    def jacobian(mom: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """d Sigma / d theta and d mu / d theta, one slice per parameter."""
        A, full, eta = mom["A"], mom["full"], mom["eta"]
        dS = np.zeros((n_free, p, p))
        dM = np.zeros((n_free, p))
        for e in entries:
            f = e["free"]
            if f is None:
                continue
            i, j = e["i"], e["j"]
            if e["kind"] in ("load", "reg"):
                half = np.outer(A[:p, i], full[j, :p])
                dS[f] += half + half.T
                dM[f] += A[:p, i] * eta[j]
            elif e["kind"] == "psi":
                d = np.outer(A[:p, i], A[:p, j])
                dS[f] += d if i == j else d + d.T
            else:
                dM[f] += A[:p, i]
        return dS, dM

    logdet_S = float(np.linalg.slogdet(S)[1])
    bad = (1e10, np.zeros(n_free))

    def casewise(
        Sigma: np.ndarray, mu: np.ndarray
    ) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
        """-2 log-likelihood / n of the incomplete data, less its constant,
        with its derivatives with respect to Sigma and mu."""
        total = 0.0
        G = np.zeros((p, p))
        w_all = np.zeros(p)
        for g in patterns:
            o = g["idx"]
            Sg = Sigma[np.ix_(o, o)]
            sgn, logdet = np.linalg.slogdet(Sg)
            if sgn <= 0 or not np.isfinite(logdet):
                return None
            inv = np.linalg.inv(Sg)
            d = g["mean"] - mu[o]
            w = inv @ d
            total += g["n"] * (logdet + float(np.sum(g["S"] * inv)) + float(d @ w))
            G[np.ix_(o, o)] += g["n"] * (inv @ (Sg - g["S"]) @ inv - np.outer(w, w))
            w_all[o] += g["n"] * w
        return total / n, G / n, w_all / n

    # the same quantity at the saturated moments: the floor a model can reach
    F_sat = 0.0
    if fiml:
        floor = casewise(S, zbar)
        if floor is None:
            raise DataInsufficient(
                "path_analysis: the covariance matrix of the incomplete data "
                "is not positive definite."
            )
        F_sat = floor[0]

    def objective(theta: np.ndarray) -> Tuple[float, np.ndarray]:
        if fiml:
            try:
                mom = implied(theta)
                parts = casewise(mom["Sigma"], mom["mu"])
            except np.linalg.LinAlgError:
                return bad
            if parts is None:
                return bad
            dS, dM = jacobian(mom)
            grad = np.einsum("ij,kij->k", parts[1], dS) - 2.0 * dM @ parts[2]
            return float(parts[0] - F_sat), grad
        try:
            mom = implied(theta)
            Sigma = mom["Sigma"]
            sgn, logdet = np.linalg.slogdet(Sigma)
            if sgn <= 0 or not np.isfinite(logdet):
                return bad
            Sinv = np.linalg.inv(Sigma)
        except np.linalg.LinAlgError:
            return bad
        dS, dM = jacobian(mom)
        F = logdet + np.trace(S @ Sinv) - logdet_S - p
        G = Sinv @ (Sigma - S) @ Sinv
        if mean_on:
            d = zbar - mom["mu"]
            w = Sinv @ d
            F += float(d @ w)
            G = G - np.outer(w, w)
            grad = np.einsum("ij,kij->k", G, dS) - 2.0 * dM @ w
        else:
            grad = np.einsum("ij,kij->k", G, dS)
        return float(F), grad

    def information(mom: Dict[str, np.ndarray]) -> np.ndarray:
        dS, dM = jacobian(mom)
        if fiml:  # expected information, pattern by pattern
            info = np.zeros((n_free, n_free))
            for g in patterns:
                o = g["idx"]
                inv = np.linalg.inv(mom["Sigma"][np.ix_(o, o)])
                dSo = dS[:, o][:, :, o]
                dMo = dM[:, o]
                info += (g["n"] / n) * (
                    0.5 * np.einsum("aij,jk,bkl,li->ab", dSo, inv, dSo, inv)
                    + dMo @ inv @ dMo.T
                )
            return info
        Sinv = np.linalg.inv(mom["Sigma"])
        info = 0.5 * np.einsum("aij,jk,bkl,li->ab", dS, Sinv, dS, Sinv)
        if mean_on:
            info = info + dM @ Sinv @ dM.T
        return np.asarray(info, dtype=float)

    theta0 = np.zeros(n_free)
    for e in entries:
        f = e["free"]
        if f is None:
            continue
        i, j = e["i"], e["j"]
        if e["kind"] == "load":
            theta0[f] = 1.0
        elif e["kind"] == "psi" and i == j:
            theta0[f] = 0.05 if is_lat[i] else 0.5 * S[i, i]
        elif e["kind"] == "int" and not is_lat[i]:
            theta0[f] = zbar[i]
    if n_free:
        res = optimize.minimize(
            objective, theta0, jac=True, method="BFGS",
            options={"gtol": 1e-9, "maxiter": 5000},
        )  # fmt: skip
        theta = res.x
        # Fisher scoring tightens the solution to machine precision, which
        # BFGS alone does not reach.
        for _ in range(200):
            F, grad = objective(theta)
            if F >= 1e10:
                break
            try:
                step = np.linalg.solve(2.0 * information(implied(theta)), grad)
            except np.linalg.LinAlgError:
                break
            scale_ = 1.0
            moved = False
            while scale_ > 1e-6:
                cand = theta - scale_ * step
                F_new, _ = objective(cand)
                if np.isfinite(F_new) and F_new <= F + 1e-13:
                    theta, moved = cand, True
                    break
                scale_ *= 0.5
            if not moved or np.max(np.abs(scale_ * step)) < 1e-12:
                break
        F_min, grad = objective(theta)
        if F_min >= 1e10 or np.max(np.abs(grad)) > 1e-5:
            raise DataInsufficient(
                "path_analysis: the likelihood did not converge "
                f"(largest gradient {np.max(np.abs(grad)):.2e}). The model is "
                "probably not identified.",
                recovery_hint="Check for a feedback loop, a latent variable "
                "with fewer than three indicators and nothing else to "
                "identify it, or a residual covariance the data cannot "
                "separate from a path.",
            )
    else:  # every parameter fixed
        theta = theta0
        F_min, _ = objective(theta)
    F_min = max(F_min, 0.0)
    mom = implied(theta)
    Sigma, Psi, full = mom["Sigma"], mom["Psi"], mom["full"]
    Sinv = np.linalg.inv(Sigma)
    negative = [
        names[e["i"]] for e in entries
        if e["kind"] == "psi" and e["i"] == e["j"] and e["free"] is not None
        and theta[e["free"]] < 0
    ]  # fmt: skip
    if negative:
        warnings.warn(
            "path_analysis: the estimated variance of "
            + ", ".join(negative)
            + " is negative (a Heywood case). The model fits the data only "
            "with an inadmissible value; treat the solution as a sign of "
            "misspecification or of too small a sample.",
            RuntimeWarning,
            stacklevel=3,
        )

    # ---- covariance of the estimates -------------------------------------
    dS, dM = jacobian(mom) if n_free else (np.zeros((0, p, p)), np.zeros((0, p)))
    info = information(mom) if n_free else np.zeros((0, 0))
    if fiml and n_free:
        # With missing values the expected information assumes the pattern
        # of missingness carries no information; the observed information
        # (the Hessian) does not, and is what lavaan and Stata report.
        hess = np.zeros((n_free, n_free))
        for a_ in range(n_free):
            h = 1e-5 * max(1.0, abs(theta[a_]))
            up, dn = theta.copy(), theta.copy()
            up[a_] += h
            dn[a_] -= h
            hess[a_] = (objective(up)[1] - objective(dn)[1]) / (2 * h)
        info = 0.25 * (hess + hess.T)  # per observation: half of d2 F
    rows_i, cols_i = np.tril_indices(p)
    if n_free:
        try:
            info_inv = np.linalg.inv(info)
        except np.linalg.LinAlgError as exc:
            raise DataInsufficient(
                "path_analysis: the information matrix is singular; the "
                "model is not identified."
            ) from exc
        if np.linalg.cond(info) > 1e13:
            raise DataInsufficient(
                "path_analysis: the information matrix is singular to "
                "working precision; some parameters are not identified.",
                recovery_hint="A latent variable needs a scale (a fixed "
                "loading or variance) and enough indicators.",
            )
    else:
        info_inv = np.zeros((0, 0))
    fit: Dict[str, Any] = {}
    robust = se == "robust"
    if robust:
        # Fourth-moment covariance of the sample moments, with the exogenous
        # variables conditioned on: the residuals of y on x carry the
        # sampling variation, the x block carries none.
        D = np.einsum("ni,nj->nij", Zc, Zc)[:, rows_i, cols_i]
        first = Z.copy()
        if k:
            beta_yx = np.linalg.solve(Sxx, S[q:, :q])  # k x q
            R = Zc.copy()
            R[:, :q] = Zc[:, :q] - Zc[:, q:] @ beta_yx
            T = np.eye(p)
            T[:q, q:] = beta_yx.T
            Dr = np.einsum("ni,nj->nij", R, R)
            Dr[:, q:, q:] = Sxx
            D = np.einsum("ai,nij,bj->nab", T, Dr, T)[:, rows_i, cols_i]
            first[:, :q] = zbar[:q] + R[:, :q]
            first[:, q:] = zbar[q:]
        mult = np.where(rows_i == cols_i, 1.0, 2.0)
        Wv = (
            0.25
            * np.outer(mult, mult)
            * (
                Sinv[np.ix_(rows_i, rows_i)] * Sinv[np.ix_(cols_i, cols_i)]
                + Sinv[np.ix_(rows_i, cols_i)] * Sinv[np.ix_(cols_i, rows_i)]
            )
        )
        Jv = dS[:, rows_i, cols_i]
        if mean_on:
            stat = np.hstack([first, D])
            W = np.zeros((p + len(rows_i),) * 2)
            W[:p, :p] = Sinv
            W[p:, p:] = Wv
            Delta = np.hstack([dM, Jv])
        else:
            stat, W, Delta = D, Wv, Jv
        sc = stat - stat.mean(axis=0)
        Gamma = sc.T @ sc / n
        if n_free:
            vcov = info_inv @ (Delta @ W @ Gamma @ W @ Delta.T) @ info_inv / n
            U = W - W @ Delta.T @ info_inv @ Delta @ W
        else:
            vcov = np.zeros((0, 0))
            U = W
    else:
        vcov = info_inv / n

    # ---- the parameter table ----------------------------------------------
    def label_values(th: np.ndarray) -> Dict[str, float]:
        out = {lab: float(th[i]) for lab, i in label_to_free.items()}
        for e in entries:
            if e["label"] and e["free"] is None:
                out[e["label"]] = float(e["fixed"])
        return out

    sd = np.sqrt(np.clip(np.diag(full), 1e-300, None))
    z_crit = float(stats.norm.isf(alpha / 2.0))
    records: List[Dict[str, Any]] = []
    std_by_label: Dict[str, float] = {}

    def record(lhs: str, op: str, rhs: str, label: str, est: float,
               se_val: float, std_all: float) -> None:  # fmt: skip
        if not np.isfinite(se_val) or se_val <= 0:
            zval = pval = float("nan")
            se_out = 0.0 if se_val == 0 else float("nan")
            lo = hi = est if se_val == 0 else float("nan")
        else:
            se_out = float(se_val)
            zval = est / se_out
            pval = float(2 * stats.norm.sf(abs(zval)))
            lo, hi = est - z_crit * se_out, est + z_crit * se_out
        records.append(
            {
                "lhs": lhs, "op": op, "rhs": rhs, "label": label,
                "est": float(est), "se": se_out, "z": zval, "pvalue": pval,
                "ci_lower": lo, "ci_upper": hi, "std_all": float(std_all),
            }
        )  # fmt: skip

    def se_of(e: Dict[str, Any]) -> float:
        if e["free"] is None:
            return 0.0
        return float(math.sqrt(max(vcov[e["free"], e["free"]], 0.0)))

    def emit(e: Dict[str, Any]) -> None:
        est = e["fixed"] if e["free"] is None else theta[e["free"]]
        i, j = e["i"], e["j"]
        if e["kind"] == "load":
            std = est * sd[j] / sd[i]
            record(names[j], "=~", names[i], e["label"], est, se_of(e), std)
        elif e["kind"] == "reg":
            std = est * sd[j] / sd[i]
            record(names[i], "~", names[j], e["label"], est, se_of(e), std)
        elif e["kind"] == "psi":
            if i == j:
                std = est / (sd[i] * sd[i])  # share of variance unexplained
            else:  # correlation of the two disturbances
                std = est / math.sqrt(abs(Psi[i, i] * Psi[j, j]))
            record(names[i], "~~", names[j], e["label"], est, se_of(e), std)
        else:
            std = est / sd[i]
            record(names[i], "~1", "", e["label"], est, se_of(e), std)
        if e["label"]:
            # an equality-constrained label has one standardised value per
            # path; a definition uses the first, as lavaan does
            std_by_label.setdefault(e["label"], float(std))

    for kind in ("load", "reg", "psi"):
        for e in entries:
            if e["kind"] == kind:
                emit(e)
    for a_ in range(k):
        for b_ in range(a_, k):
            i, j = q + a_, q + b_
            record(names[i], "~~", names[j], "", S[i, j], 0.0,
                   S[i, j] / math.sqrt(S[i, i] * S[j, j]))  # fmt: skip
    if mean_on:
        for e in entries:
            if e["kind"] == "int":
                emit(e)
        for i in range(q, p):
            record(names[i], "~1", "", "", zbar[i], 0.0, zbar[i] / sd[i])
    for name, expr in spec["defined"]:
        est = evaluate(expr, label_values(theta))
        grad = np.zeros(n_free)
        for fi in label_to_free.values():
            h = 1e-6 * max(1.0, abs(theta[fi]))
            up, dn = theta.copy(), theta.copy()
            up[fi] += h
            dn[fi] -= h
            grad[fi] = (
                evaluate(expr, label_values(up)) - evaluate(expr, label_values(dn))
            ) / (2 * h)
        var = float(grad @ vcov @ grad) if n_free else 0.0
        std = evaluate(expr, {**label_values(theta), **std_by_label})
        record(name, ":=", expr.replace(" ", ""), name, est,
               math.sqrt(max(var, 0)), std)  # fmt: skip
    params = pd.DataFrame.from_records(records)

    # ---- fit measures -----------------------------------------------------
    chisq = n * F_min
    dev = zbar - mom["mu"] if mean_on else np.zeros(p)
    logdet_Sigma = float(np.linalg.slogdet(Sigma)[1])
    logdet_xx = float(np.linalg.slogdet(Sxx)[1]) if k else 0.0
    base_df = p * (p + 1) // 2 - k * (k + 1) // 2 - q
    if fiml:
        cells = sum(g["n"] * len(g["idx"]) for g in patterns)
        logl_joint = -0.5 * (cells * math.log(2 * math.pi) + n * (F_min + F_sat))
        # independence model: each endogenous variable on its own rows
        base = n * (logdet_xx + k)
        for j in range(q):
            col = Z[:, j][~np.isnan(Z[:, j])]
            base += len(col) * (math.log(float(np.var(col))) + 1.0)
        base_chisq = base - n * F_sat
    else:
        logl_joint = -0.5 * n * (
            p * math.log(2 * math.pi) + logdet_Sigma + float(np.trace(S @ Sinv))
            + float(dev @ Sinv @ dev)
        )  # fmt: skip
        base_chisq = n * (float(np.sum(np.log(np.diag(S)[:q]))) + logdet_xx - logdet_S)
    logl_x = -0.5 * n * (k * math.log(2 * math.pi) + logdet_xx + k) if k else 0.0
    logl = logl_joint - logl_x
    d_m = max(chisq - df_model, 0.0)
    d_b = max(base_chisq - base_df, 0.0)
    cfi = 1.0 - d_m / max(d_m, d_b) if max(d_m, d_b) > 0 else 1.0
    if df_model > 0 and base_df > 0 and base_chisq / base_df != 1.0:
        tli = (base_chisq / base_df - chisq / df_model) / (base_chisq / base_df - 1.0)
    else:
        tli = 1.0
    if df_model:
        rmsea = math.sqrt(max((chisq - df_model) / (df_model * n), 0.0))
        rm_lo = math.sqrt(_ncp(chisq, df_model, 0.05) / (df_model * n))
        rm_hi = math.sqrt(_ncp(chisq, df_model, 0.95) / (df_model * n))
    else:
        rmsea = rm_lo = rm_hi = 0.0
    s_sd = np.sqrt(np.diag(S))
    resid = ((S - Sigma) / np.outer(s_sd, s_sd))[rows_i, cols_i]
    if mean_on:
        resid = np.concatenate([resid, dev / s_sd])
    srmr = math.sqrt(float(np.mean(resid**2)))
    fit.update(
        {
            "chisq": float(chisq),
            "df": int(df_model),
            "pvalue": (
                float(stats.chi2.sf(chisq, df_model)) if df_model else float("nan")
            ),
            "baseline_chisq": float(base_chisq),
            "baseline_df": int(base_df),
            "cfi": float(cfi),
            "tli": float(tli),
            "rmsea": float(rmsea),
            "rmsea_ci_lower": float(rm_lo),
            "rmsea_ci_upper": float(rm_hi),
            "srmr": float(srmr),
            "logl": float(logl),
            "aic": float(-2 * logl + 2 * n_free),
            "bic": float(-2 * logl + math.log(n) * n_free),
            "npar": int(n_free),
        }
    )
    if robust and df_model > 0:
        scale = float(np.trace(U @ Gamma)) / df_model
        fit["scaling_factor"] = scale
        fit["chisq_scaled"] = float(chisq / scale)
        fit["pvalue_scaled"] = float(stats.chi2.sf(chisq / scale, df_model))

    free_names: List[str] = [""] * n_free
    for e in entries:
        if e["free"] is not None and not free_names[e["free"]]:
            if e["kind"] == "load":
                shown = f"{names[e['j']]}=~{names[e['i']]}"
            elif e["kind"] == "int":
                shown = f"{names[e['i']]}~1"
            else:
                shown = f"{names[e['i']]}{_OP[e['kind']]}{names[e['j']]}"
            free_names[e["free"]] = e["label"] or shown
    explained = sorted({i for i, _ in paths})
    r2 = pd.Series(
        {names[i]: 1.0 - Psi[i, i] / full[i, i] for i in explained}, name="r2"
    )
    obs_names = names[:p]
    scores = None
    if n_lat:
        # regression-method scores: E[latent | observed] under the model
        centre = mom["mu"] if mean_on else zbar
        lat_mean = mom["eta"][p:]
        values = np.empty((n, n_lat))
        if fiml:  # each row is scored from the variables it observes
            for g in patterns:
                o = g["idx"]
                w_g = full[p:][:, o] @ np.linalg.inv(Sigma[np.ix_(o, o)])
                values[g["rows"]] = (Z[np.ix_(g["rows"], o)] - centre[o]) @ w_g.T
        else:
            values = (Z - centre) @ (full[p:, :p] @ Sinv).T
        scores = pd.DataFrame(values + lat_mean, index=frame.index, columns=latents)
    return {
        "params": params,
        "fit": fit,
        "implied_cov": pd.DataFrame(Sigma, index=obs_names, columns=obs_names),
        "sample_cov": pd.DataFrame(S, index=obs_names, columns=obs_names),
        "vcov": pd.DataFrame(vcov, index=free_names, columns=free_names),
        "r2": r2,
        "n_obs": int(n),
        "n_dropped": int(n_all - n),
        "n_patterns": len(patterns) if fiml else 1,
        "latent_cov": pd.DataFrame(full[p:, p:], index=latents, columns=latents),
        "implied_mean": (pd.Series(mom["mu"], index=obs_names) if mean_on else None),
        "factor_scores": scores,
    }
