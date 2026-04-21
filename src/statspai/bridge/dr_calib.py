"""
Bridge: Doubly Robust via Calibration (Zhang et al. 2025,
arXiv 2411.02771).

Standard AIPW is asymptotically doubly robust. Zhang et al. show that
finite-sample double robustness requires *jointly calibrating* the
outcome regression and the Riesz representer (propensity score). The
"bridge" here is between vanilla AIPW (path A) and calibrated AIPW
(path B); large differences indicate that miscalibration is biting.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .core import BridgeResult, _agreement_test, _dr_combine, _register


def _isotonic_calibrate(scores: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Pool-Adjacent-Violators isotonic regression — calibration."""
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(scores, target)
    return iso.predict(scores)


@_register("dr_calib")
def dr_calib_bridge(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    alpha: float = 0.05,
    n_boot: int = 200,
    seed: int = 0,
) -> BridgeResult:
    """
    Compare vanilla AIPW (path A) against calibrated AIPW (path B).
    Both target the ATE; agreement implies miscalibration is not biting.
    """
    df = data[[y, treat] + list(covariates)].dropna().reset_index(drop=True)
    Y = df[y].to_numpy(float)
    D = df[treat].to_numpy(int)
    X = df[covariates].to_numpy(float)
    n = len(df)
    rng = np.random.default_rng(seed)

    def _aipw_one(Yi, Di, Xi, calibrate=False):
        from sklearn.linear_model import LinearRegression, LogisticRegression
        # Outcome models
        m1 = LinearRegression().fit(Xi[Di == 1], Yi[Di == 1])
        m0 = LinearRegression().fit(Xi[Di == 0], Yi[Di == 0])
        mu1 = m1.predict(Xi)
        mu0 = m0.predict(Xi)
        # Propensity
        ps = (LogisticRegression(max_iter=1000)
              .fit(Xi, Di).predict_proba(Xi)[:, 1])
        ps = np.clip(ps, 0.02, 0.98)
        if calibrate:
            # Isotonic calibration of (mu1 + (Y-mu1)/ps) on Y for treated
            # and same for controls — Zhang 2025 calibration step.
            mu1_cal = mu1.copy()
            mu0_cal = mu0.copy()
            try:
                mu1_cal[Di == 1] = _isotonic_calibrate(mu1[Di == 1], Yi[Di == 1])
                mu0_cal[Di == 0] = _isotonic_calibrate(mu0[Di == 0], Yi[Di == 0])
                ps_cal = _isotonic_calibrate(ps, Di.astype(float))
                ps_cal = np.clip(ps_cal, 0.02, 0.98)
            except Exception:
                mu1_cal, mu0_cal, ps_cal = mu1, mu0, ps
            mu1, mu0, ps = mu1_cal, mu0_cal, ps_cal
        # AIPW score
        score = (
            mu1 - mu0
            + Di * (Yi - mu1) / ps
            - (1 - Di) * (Yi - mu0) / (1 - ps)
        )
        return float(np.mean(score))

    ate_aipw = _aipw_one(Y, D, X, calibrate=False)
    ate_cal = _aipw_one(Y, D, X, calibrate=True)

    # Bootstrap SE
    boot_a = np.full(n_boot, np.nan)
    boot_c = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            boot_a[b] = _aipw_one(Y[idx], D[idx], X[idx], calibrate=False)
        except Exception:
            pass
        try:
            boot_c[b] = _aipw_one(Y[idx], D[idx], X[idx], calibrate=True)
        except Exception:
            pass
    se_a = float(np.nanstd(boot_a, ddof=1)) or 1e-6
    se_c = float(np.nanstd(boot_c, ddof=1)) or 1e-6

    diff, diff_se, diff_p = _agreement_test(ate_aipw, se_a, ate_cal, se_c)
    est_dr, se_dr = _dr_combine(ate_aipw, se_a, ate_cal, se_c, diff_p)

    return BridgeResult(
        kind="dr_calib",
        path_a_name="AIPW (vanilla)",
        path_b_name="AIPW (calibrated)",
        estimate_a=float(ate_aipw),
        estimate_b=float(ate_cal),
        se_a=se_a,
        se_b=se_c,
        diff=diff,
        diff_se=diff_se,
        diff_p=diff_p,
        estimate_dr=est_dr,
        se_dr=se_dr,
        n_obs=n,
        detail={},
        reference="Zhang et al. (2025), arXiv 2411.02771",
    )
