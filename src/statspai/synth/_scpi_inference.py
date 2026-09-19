"""Inference layer of ``sp.scpi`` (private; port of the body of R ``scpi``).

Given the data matrices and the donor weights, reproduces for a single treated
unit (``effect = "unit-time"``): ``local.geom`` / ``local.geom.2step`` (rho,
relaxed constraint set), ``u.des.prep`` + ``u.sigma.est`` (conditional moments
and variance ``Sigma`` of the pseudo-residuals), ``df.EST``, the in-sample
simulation of ``insampleUncertaintyGetDiag``, ``e.des.prep`` + ``scpi.out``
(out-of-sample bounds) and ``simultaneousPredGet`` (joint bounds).
"""

from __future__ import annotations

import warnings

import numpy as np

from . import _scpi_solvers as _sv

# ====================================================================== #
#  Inference (scpi.R body for a single treated unit, effect = unit-time)
# ====================================================================== #


def _regularize_w(kind: str, rho_max, res, B, T0, J, d0) -> float:
    """R ``regularize.w``."""
    sigma_u = np.sqrt(np.mean((res - res.mean()) ** 2))
    if kind == "type-1":
        sd_min = np.min(np.std(B, axis=0, ddof=1))
        if sd_min == 0:
            raise ValueError(
                "One of your donors has no variation in the pre-treatment period!"
            )
        CC = sigma_u / sd_min
    elif kind == "type-2":
        var_min = np.min(np.var(B, axis=0, ddof=1))
        if var_min == 0:
            raise ValueError(
                "One of your donors has no variation in the pre-treatment period!"
            )
        CC = np.max(np.std(B, axis=0, ddof=1)) * sigma_u / var_min
    else:
        raise ValueError("rho must be 'type-1', 'type-2' or a number.")
    d = J
    rho = CC * np.sqrt(np.log(d) * d0 * np.log(T0)) / np.sqrt(T0)
    if rho_max is not None:
        rho = min(rho, rho_max)
    return float(rho)


def _local_geometry(spec, w, rho, rho_max, res, B):
    """R ``local.geom`` + ``local.geom.2step`` (single unit, KM = 0)."""
    T0, J = B.shape
    d0 = int(np.sum(np.abs(w) >= 1e-6))
    if rho is None:
        rho = "type-2"
    if isinstance(rho, str):
        rho_val = _regularize_w(rho, rho_max, res, B, T0, J, d0)
        if rho_val < 0.001:  # regularize.check.lb
            rho_val = max(
                _regularize_w("type-1", 0.2, res, B, T0, J, d0),
                _regularize_w("type-2", 0.2, res, B, T0, J, d0),
            )
            if rho_val < 0.05:
                rho_val = rho_max
            warnings.warn(
                "rho was too low; increased it to favour shrinkage (R "
                "regularize.check.lb).",
                RuntimeWarning,
                stacklevel=4,
            )
    else:
        rho_val = float(rho)

    p, direction, name = spec["p"], spec["dir"], spec["name"]
    if p == "L2" or name == "ridge" or p == "no norm":
        index_w = np.ones(J, dtype=bool)
    else:
        index_w = np.abs(w) > rho_val
        if not index_w.any():
            index_w = np.zeros(J, dtype=bool)
            index_w[int(np.argmax(w))] = True
            warnings.warn(
                "rho was too high; kept the largest weight active (R "
                "regularize.check).",
                RuntimeWarning,
                stacklevel=4,
            )

    # local.geom.2step
    Qin = spec["Q2"] if p == "L1-L2" else spec["Q"]
    Qout = Qin
    if p == "L1":
        w_norm = float(np.sum(np.abs(w)))
        rhoj = rho_val
    elif p in ("L2", "L1-L2"):
        w_norm = float(np.sum(w**2))
        rhoj = min(2 * np.sqrt(w_norm) * rho_val, rho_max)
    if direction in ("<=", "==/<="):
        active = 1.0 if (w_norm - Qin) > -rhoj else 0.0
        Qout = active * (w_norm + rho_val - Qin) + Qin
    if spec["lb"] == 0:
        lb = np.where(w < rho_val, w, 0.0)
    else:
        lb = np.full(J, -np.inf)
    if p == "L1-L2":
        Q_star, Q2_star = spec["Q"], Qout
    else:
        Q_star, Q2_star = Qout, None
    return {
        "rho": rho_val,
        "d0": d0,
        "index_w": index_w,
        "Q_star": Q_star,
        "Q2_star": Q2_star,
        "lb": lb,
    }


def _df_est(spec, w, B) -> float:
    """R ``df.EST`` (KM = 0)."""
    name, p, direction = spec["name"], spec["p"], spec["dir"]
    if name == "ols" or p == "no norm":
        return float(B.shape[1])
    if name == "lasso" or (p == "L1" and direction == "<="):
        return float(np.sum(np.abs(w) >= 1e-6))
    if name == "simplex" or (p == "L1" and direction == "=="):
        return float(np.sum(np.abs(w) >= 1e-6) - 1)
    lam = spec["lambda"]
    if lam is None:  # R: d^2 / (d^2 + NULL) is numeric(0) -> df = 0
        return 0.0
    d = np.clip(np.linalg.svd(B, compute_uv=False), 0.0, None)
    return float(np.sum(d**2 / (d**2 + lam)))


def _hc_scale(u_sigma, Z, TT, df):
    if u_sigma == "HC0":
        return np.ones(TT)
    if u_sigma == "HC1":
        return np.full(TT, TT / (TT - df))
    lev = np.einsum("ij,ji->i", Z, np.linalg.solve(Z.T @ Z, Z.T))
    if u_sigma == "HC2":
        return 1.0 / (1.0 - lev)
    if u_sigma == "HC3":
        return 1.0 / (1.0 - lev) ** 2
    if u_sigma == "HC4":
        dd = np.minimum(4.0, TT * lev / df)
        return 1.0 / (1.0 - lev) ** dd
    raise ValueError("u_sigma must be one of 'HC0','HC1','HC2','HC3','HC4'.")


def _sqrtm_psd(S: np.ndarray) -> np.ndarray:
    """R ``sqrtm``: U diag(sqrt(d)) U' from the SVD."""
    U, d, _ = np.linalg.svd(S)
    d = np.clip(d, 0.0, None)
    return (U * np.sqrt(d)) @ U.T


def _proj_fitted(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return X @ np.linalg.lstsq(X, y, rcond=None)[0]


def _e_design(B, P_rows, index_w, e_order, T0):
    """``e.des.prep`` (out.feat = TRUE, constant = FALSE) + scpi.R fallbacks."""
    if e_order == 0:
        X0 = np.ones((T0, 1))
        X1 = np.ones((P_rows.shape[0], 1))
    else:
        X0 = np.column_stack([B[:, index_w], np.ones(T0)])
        X1 = np.column_stack([P_rows[:, index_w], np.ones(P_rows.shape[0])])
    if (T0 - 10) <= X0.shape[1]:
        X0 = np.ones((T0, 1))
        X1 = np.ones((P_rows.shape[0], 1))
        e_order = 0

    def detect_constant(x):
        keep = (np.sum(x == 1, axis=0) != x.shape[0]) & (np.sum(x, axis=0) != 0)
        return np.column_stack([x[:, keep], np.ones(x.shape[0])])

    X0 = detect_constant(X0)
    X1 = detect_constant(X1)
    if X0.shape[1] >= X0.shape[0] - 1:  # avoidCollin
        X0 = np.ones((T0, 1))
        X1 = np.ones((P_rows.shape[0], 1))
    return X0, X1, e_order


def _scpi_out(res, x, ev, e_method, alpha):
    """R ``scpi.out`` (effect = unit-time, out.feat = TRUE)."""
    n_eval = ev.shape[0]
    if e_method in ("gaussian", "ls"):
        bhat = np.linalg.lstsq(x, res, rcond=None)[0]
        e_mean = ev @ bhat
        res_fit = x @ bhat
        u = res - res_fit
        ghat = np.linalg.lstsq(x, np.log(u**2), rcond=None)[0]
        var_eval = ev @ ghat
        qcoef = _sv.rrq(x, u, [0.25, 0.75])
        qp = ev @ qcoef
        iq = np.abs(qp[:, 1] - qp[:, 0])
        if e_method == "gaussian":
            e_sig = np.minimum(np.sqrt(np.exp(var_eval)), iq / 1.34)
            eps = np.sqrt(-np.log(alpha) * 2) * e_sig
            return e_mean - eps, e_mean + eps, e_mean, e_sig**2
        res_var = x @ ghat
        res_st = u / np.sqrt(np.exp(res_var))
        e_sig = np.minimum(np.sqrt(np.exp(var_eval)), iq / 1.34)
        lo = e_mean + e_sig * np.quantile(res_st, alpha)
        hi = e_mean + e_sig * np.quantile(res_st, 1 - alpha)
        return lo, hi, e_mean, e_sig**2
    qcoef = _sv.rrq(x, res, [alpha, 1 - alpha])
    qp = ev @ qcoef
    return qp[:, 0], qp[:, 1], np.full(n_eval, np.nan), np.full(n_eval, np.nan)


def _insample_problem(spec, geom, beta, Qm):
    """Map R's (p, dir) constraint set to the active-set solver's arguments."""
    p, direction = spec["p"], spec["dir"]
    J = beta.shape[0]
    if p == "no norm":
        return {"kind": "y", "ell": np.full(J, -np.inf), "sum_mode": None, "ball": None}
    if p == "L1" and direction == "==":
        return {
            "kind": "y",
            "ell": geom["lb"] - beta,
            "sum_mode": "eq",
            "sum_val": geom["Q_star"] - beta.sum(),
            "ball": None,
        }
    if p == "L1" and direction == "<=":
        T = np.hstack([np.eye(J), -np.eye(J)])
        z0 = np.concatenate([np.clip(beta, 0, None), np.clip(-beta, 0, None)])
        return {
            "kind": "z",
            "T": T,
            "Qz": T.T @ Qm @ T,
            "ell": np.zeros(2 * J),
            "sum_mode": "le",
            "sum_val": geom["Q_star"],
            "ball": None,
            "y0": z0,
        }
    if p == "L2":
        return {
            "kind": "y",
            "ell": np.full(J, -np.inf),
            "sum_mode": None,
            "ball": (beta, geom["Q_star"]),
        }
    if p == "L1-L2":
        return {
            "kind": "y",
            "ell": geom["lb"] - beta,
            "sum_mode": "eq",
            "sum_val": geom["Q_star"] - beta.sum(),
            "ball": (beta, geom["Q2_star"]),
        }
    raise ValueError(
        f"Unsupported constraint p={p!r}, dir={direction!r}."
    )  # pragma: no cover


def _insample_sims(Z, P_rows, beta, spec, geom, zeta):
    """Per-draw bounds of ``insampleUncertaintyGetDiag`` (lb, ub columns).

    Each (draw, horizon, side) problem is solved by the exact active-set
    QCQP of ``_scpi_solvers``; SLSQP is the fallback when the active-set
    search cannot certify a KKT point.
    """
    TT, J = Z.shape
    Qm = Z.T @ Z / TT
    sims = zeta.shape[1]
    H = P_rows.shape[0]
    vlb = np.full((sims, H), np.nan)
    vub = np.full((sims, H), np.nan)
    prob = _insample_problem(spec, geom, beta, Qm)
    n_fallback = 0

    def solve(cc, Qq, Gg, kk):
        return _sv.insample_active_set(
            cc,
            Qq,
            Gg,
            kk,
            prob["ell"],
            prob["sum_mode"],
            prob.get("sum_val", 0.0),
            prob["ball"],
            prob.get("y0"),
        )

    for s in range(sims):
        G = zeta[:, s]
        if prob["kind"] == "z":
            T = prob["T"]
            Gz = T.T @ (Qm @ beta + G)
            k0 = beta @ Qm @ beta + 2 * G @ beta
        for h in range(H):
            xt = P_rows[h]
            for sign, store in ((-1.0, vlb), (1.0, vub)):
                c = sign * xt
                if prob["kind"] == "z":
                    zsol = solve(T.T @ c, prob["Qz"], Gz, k0)
                    y = (
                        zsol
                        if zsol is None or isinstance(zsol, str)
                        else T @ zsol - beta
                    )
                else:
                    y = solve(c, Qm, G, 0.0)
                if isinstance(y, str):  # unbounded: R/ECOS reports a failed draw
                    continue
                if y is None:
                    n_fallback += 1
                    y = _sv.insample_slsqp(
                        c,
                        Qm,
                        G,
                        beta,
                        {"p": spec["p"], "dir": spec["dir"]},
                        geom["lb"],
                        geom["Q_star"],
                        geom["Q2_star"],
                    )
                if y is not None:
                    store[s, h] = -xt @ y
    return vlb, vub, n_fallback


def scpi_inference(
    A,
    B,
    P,
    w,
    spec,
    sims=200,
    draws=None,
    rng=None,
    u_missp=True,
    u_sigma="HC1",
    u_order=1,
    u_alpha=0.05,
    e_order=1,
    e_alpha=0.05,
    rho=None,
    rho_max=0.2,
    aggregate=False,
):
    """Inference layer of R ``scpi`` given data matrices and weights.

    Separated from :func:`scpi` so the reference-parity test can hold it
    against R on R's own weights (the weights themselves are compared
    separately at the conic solver's tolerance).
    """
    if u_order not in (0, 1) or e_order not in (0, 1):
        raise NotImplementedError(
            "u_order / e_order > 1 are not ported (R u.order/e.order)."
        )
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    P = np.asarray(P, float)
    w = np.asarray(w, float)
    T0, J = B.shape
    T1 = P.shape[0]
    res = A - B @ w
    geom = _local_geometry(spec, w, rho, rho_max, res, B)
    index_w = geom["index_w"]

    # ---- pseudo-residual conditional moments (u.missp) ----
    if u_missp:
        if u_order == 0:
            u_des = np.ones((T0, 1))
        else:
            u_des = np.column_stack([B[:, index_w], np.ones(T0)])
        if T0 - 10 <= u_des.shape[1]:
            u_des = np.ones((T0, 1))
            u_order = 0
        u_mean = _proj_fitted(u_des, res)
        u_T, u_params = T0, u_des.shape[1]
    else:
        u_mean = np.zeros(T0)
        u_T, u_params = 0, 0

    df = _df_est(spec, w, B)
    if df >= T0:
        df = T0 - 1
        warnings.warn(
            "The specification uses more degrees of freedom than observations.",
            RuntimeWarning,
            stacklevel=2,
        )
    vc = _hc_scale(u_sigma, B, T0, df)
    omega = (res - u_mean) ** 2 * vc
    Sigma = (B.T * omega) @ B / T0**2
    Sigma_root = _sqrtm_psd(Sigma)

    # ---- in-sample simulation ----
    if draws is None:
        if rng is None:
            rng = np.random.default_rng()
        zraw = rng.standard_normal((J, sims))
    else:
        zraw = np.asarray(draws, float)
        if zraw.shape != (J, sims):
            raise ValueError(f"draws must have shape ({J}, {sims}), got {zraw.shape}.")
    zeta = Sigma_root @ zraw
    P_rows = np.vstack([P, P.mean(axis=0)]) if aggregate else P
    vlb, vub, n_fallback = _insample_sims(B, P_rows, w, spec, geom, zeta)
    vsig_all = np.hstack([vlb, vub])
    keep = np.sum(np.isnan(vsig_all), axis=1) < vsig_all.shape[1]
    vlb_k, vub_k = vlb[keep], vub[keep]
    n_rows = P_rows.shape[0]
    if keep.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns -> NaN
            w_lb = np.nanquantile(vlb_k, u_alpha / 2, axis=0)
            w_ub = np.nanquantile(vub_k, 1 - u_alpha / 2, axis=0)
    else:
        w_lb = w_ub = np.full(n_rows, np.nan)
        warnings.warn(
            "Every in-sample simulation problem was unbounded or failed (for "
            "w_constr='ols' this happens when the donors outnumber the "
            "pre-treatment periods: the weights are not identified). The "
            "in-sample bounds are NaN, as R scpi's are.",
            RuntimeWarning,
            stacklevel=2,
        )
    failed = np.vstack(
        [np.isnan(vlb).sum(axis=0) / sims * 100, np.isnan(vub).sum(axis=0) / sims * 100]
    )
    _sv.warn_failed(int(np.isnan(vsig_all).sum()), vsig_all.size)

    # ---- out-of-sample ----
    X0, X1, e_order_eff = _e_design(B, P_rows, index_w, e_order, T0)
    ea = e_alpha / 2
    g_lo, g_hi, e_mean, e_var = _scpi_out(res, X0, X1, "gaussian", ea)
    l_lo, l_hi, _, _ = _scpi_out(res, X0, X1, "ls", ea)
    if e_order_eff == 0:
        q_lo = np.full(P_rows.shape[0], np.quantile(res, ea))
        q_hi = np.full(P_rows.shape[0], np.quantile(res, 1 - ea))
    else:
        q_lo, q_hi, _, _ = _scpi_out(res, X0, X1, "qreg", ea)

    # ---- joint bounds (simultaneousPredGet, I = 1) ----
    if keep.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lb_joint = np.quantile(np.nanmin(vlb_k[:, :T1], axis=1), u_alpha / 2)
            ub_joint = np.quantile(np.nanmax(vub_k[:, :T1], axis=1), 1 - u_alpha / 2)
    else:
        lb_joint = ub_joint = np.nan
    eps = np.sqrt(np.log(T1 + 1)) if T1 > 1 else 1.0
    ML = g_lo[:T1] * eps + lb_joint
    MU = g_hi[:T1] * eps + ub_joint

    sl = slice(0, T1)
    bounds = {
        "insample": np.column_stack([w_lb[sl], w_ub[sl]]),
        "subgaussian": np.column_stack([w_lb[sl] + g_lo[sl], w_ub[sl] + g_hi[sl]]),
        "ls": np.column_stack([w_lb[sl] + l_lo[sl], w_ub[sl] + l_hi[sl]]),
        "qreg": np.column_stack([w_lb[sl] + q_lo[sl], w_ub[sl] + q_hi[sl]]),
        "joint": np.column_stack([ML, MU]),
    }
    out = {
        "bounds": bounds,
        "e_bounds": {
            "subgaussian": np.column_stack([g_lo[sl], g_hi[sl]]),
            "ls": np.column_stack([l_lo[sl], l_hi[sl]]),
            "qreg": np.column_stack([q_lo[sl], q_hi[sl]]),
        },
        "CI": None,
        "rho": geom["rho"],
        "Q_star": geom["Q_star"],
        "Q2_star": geom["Q2_star"],
        "lb": geom["lb"],
        "index_w": index_w,
        "df": df,
        "u_mean": u_mean,
        "u_var": omega,
        "u_T": u_T,
        "u_params": u_params,
        "u_order": u_order,
        "Sigma": Sigma,
        "e_mean": e_mean[sl],
        "e_var": e_var[sl],
        "e_T": T0,
        "e_params": X0.shape[1],
        "e_order": e_order_eff,
        "failed_sims": failed[:, sl],
        "sims": sims,
        "n_slsqp_fallback": n_fallback,
        "vsig": np.hstack([vlb[:, sl], vub[:, sl]]),
    }
    if aggregate:
        k = T1
        out["aggregate"] = {
            "insample": (float(w_lb[k]), float(w_ub[k])),
            "subgaussian": (float(w_lb[k] + g_lo[k]), float(w_ub[k] + g_hi[k])),
            "ls": (float(w_lb[k] + l_lo[k]), float(w_ub[k] + l_hi[k])),
            "qreg": (float(w_lb[k] + q_lo[k]), float(w_ub[k] + q_hi[k])),
            "e_subgaussian": (float(g_lo[k]), float(g_hi[k])),
            "e_ls": (float(l_lo[k]), float(l_hi[k])),
            "e_qreg": (float(q_lo[k]), float(q_hi[k])),
        }
    return out
