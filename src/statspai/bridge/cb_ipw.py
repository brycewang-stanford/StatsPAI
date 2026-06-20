"""
Bridge: Covariate Balancing ≡ IPW × DR (Słoczyński, Uysal & Wooldridge 2023,
arXiv 2310.18563).

Entropy Balancing / CBPS / IPW / AIPW share a common moment-equation
representation. Under correct propensity specification, IPW and CB
return the same ATE; AIPW adds the outcome-model augmentation for
double robustness. Reporting both lets the user check if balancing
adds value beyond IPW.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ..core._bootstrap import bootstrap_se as _bootstrap_se
from .core import BridgeResult, _agreement_test, _dr_combine, _register


@_register("cb_ipw")
def cb_ipw_bridge(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    alpha: float = 0.05,
    n_boot: int = 200,
    seed: int = 0,
) -> BridgeResult:
    """
    Compare IPW (path A) against entropy-balancing weighted DIM
    (path B). Under correct propensity, both target the same ATE.
    """
    df = data[[y, treat] + list(covariates)].dropna().reset_index(drop=True)
    if df[treat].nunique() != 2:
        raise ValueError(
            f"cb_ipw bridge requires binary treat; got "
            f"{df[treat].nunique()} values."
        )
    Y = df[y].to_numpy(float)
    D = df[treat].to_numpy(int)
    X = df[covariates].to_numpy(float)
    n = len(df)
    rng = np.random.default_rng(seed)

    # ---------- Path A: classic IPW ATE ---------- #
    def _ipw_ate(Yi: np.ndarray, Di: np.ndarray, Xi: np.ndarray) -> float:
        from sklearn.linear_model import LogisticRegression

        # Add intercept implicitly via sklearn
        ps = LogisticRegression(max_iter=1000).fit(Xi, Di).predict_proba(Xi)[:, 1]
        ps = np.clip(ps, 0.02, 0.98)
        w = np.where(Di == 1, 1.0 / ps, 1.0 / (1.0 - ps))
        # Hájek-normalised IPW (more stable than Horvitz-Thompson)
        w1 = (Di == 1) * w
        w0 = (Di == 0) * w
        m1 = float(np.sum(w1 * Yi) / max(np.sum(w1), 1e-12))
        m0 = float(np.sum(w0 * Yi) / max(np.sum(w0), 1e-12))
        return m1 - m0

    ate_ipw = _ipw_ate(Y, D, X)

    # ---------- Path B: entropy balancing weighted DIM ---------- #
    def _ebalance_ate(Yi: np.ndarray, Di: np.ndarray, Xi: np.ndarray) -> float:
        # Entropy balancing on the treated mean: solve for weights w_0
        # such that sum(w_0 * X_control) = mean(X_treated), maximising
        # entropy. Reweight controls; treated keep weight 1/n_t.
        Xt = Xi[Di == 1]
        Xc = Xi[Di == 0]
        Yt = Yi[Di == 1]
        Yc = Yi[Di == 0]
        target = Xt.mean(axis=0)
        # Center controls
        Xc_centered = Xc - target
        # Newton iterations on the dual (Hainmueller 2012 algorithm)
        lam = np.zeros(Xc_centered.shape[1])
        for _ in range(100):
            log_w = -Xc_centered @ lam
            log_w -= log_w.max()
            w = np.exp(log_w)
            w /= w.sum()
            grad = Xc_centered.T @ w
            if np.linalg.norm(grad) < 1e-7:
                break
            H = (Xc_centered * w[:, None]).T @ Xc_centered - np.outer(grad, grad)
            try:
                step = np.linalg.solve(H + 1e-6 * np.eye(H.shape[0]), grad)
            except np.linalg.LinAlgError:
                break
            lam = lam + step
        # ATE on the treated under balanced controls
        m1 = float(Yt.mean())
        m0 = float(np.sum(w * Yc))
        return m1 - m0

    cb_failed = False
    cb_error = None
    try:
        ate_cb = _ebalance_ate(Y, D, X)
    except Exception as exc:
        # Do NOT report IPW twice: that would make the agreement test
        # (path A vs path B) pass trivially and the DR combine meaningless,
        # silently (CLAUDE.md §7). Mark the CB path invalid and surface it.
        cb_failed = True
        cb_error = f"{type(exc).__name__}: {exc}"
        ate_cb = float("nan")
        import warnings

        warnings.warn(
            f"cb_ipw bridge: entropy-balancing (path B) failed on the main "
            f"sample ({cb_error}). The CB estimate, the IPW-vs-CB agreement "
            f"test, and the DR combine are reported as NaN; only the IPW "
            f"estimate (path A) is valid.",
            RuntimeWarning,
            stacklevel=2,
        )

    # ---------- Bootstrap SEs ---------- #
    boot_ipw = np.full(n_boot, np.nan)
    boot_cb = np.full(n_boot, np.nan)
    n_ipw_fail = 0
    n_cb_fail = 0
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            boot_ipw[b] = _ipw_ate(Y[idx], D[idx], X[idx])
        except Exception:
            n_ipw_fail += 1
        if not cb_failed:
            try:
                boot_cb[b] = _ebalance_ate(Y[idx], D[idx], X[idx])
            except Exception:
                n_cb_fail += 1

    se_ipw = _bootstrap_se(boot_ipw, n_boot, "cb_ipw IPW path")
    if cb_failed:
        se_cb = float("nan")
        diff = diff_se = diff_p = float("nan")
        est_dr = se_dr = float("nan")
    else:
        se_cb = _bootstrap_se(boot_cb, n_boot, "cb_ipw entropy-balancing path")
        diff, diff_se, diff_p = _agreement_test(ate_ipw, se_ipw, ate_cb, se_cb)
        est_dr, se_dr = _dr_combine(ate_ipw, se_ipw, ate_cb, se_cb, diff_p)

    _result = BridgeResult(
        kind="cb_ipw",
        path_a_name="IPW (Hájek-normalised)",
        path_b_name="Entropy Balancing weighted DIM",
        estimate_a=float(ate_ipw),
        estimate_b=float(ate_cb),
        se_a=se_ipw,
        se_b=se_cb,
        diff=diff,
        diff_se=diff_se,
        diff_p=diff_p,
        estimate_dr=est_dr,
        se_dr=se_dr,
        n_obs=n,
        detail={
            "cb_failed": cb_failed,
            "cb_error": cb_error,
            "n_boot_ipw_failed": n_ipw_fail,
            "n_boot_cb_failed": n_cb_fail,
        },
        reference="Słoczyński, Uysal & Wooldridge (2023), arXiv 2310.18563",
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.bridge.cb_ipw_bridge",
            params={
                "y": y,
                "treat": treat,
                "covariates": list(covariates),
                "alpha": alpha,
                "n_boot": n_boot,
                "seed": seed,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result
