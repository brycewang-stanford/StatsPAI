"""Correctness tests for local-polynomial (sp.lpoly) standard errors.

Regression guard for the fix that added the kernel sandwich meat (X'W^2 X) to
the conditional-variance estimator. The previous code reported
``sigma^2 (X'WX)^-1`` (valid only if the kernel weights were inverse-variance
weights), which understated the SE by up to ~25% (kernel dependent) and gave
CIs that under-covered.

Reference for the conditional variance:
    Fan, J. & Gijbels, I. (1996). "Local Polynomial Modelling and Its
    Applications." Chapman & Hall/CRC. [paper.bib: fan1996local]
"""

import numpy as np
import pandas as pd

from statspai.nonparametric.lpoly import lpoly, _local_poly_fit


def _mc_ratio_and_coverage(kernel, h=0.15, degree=1, x0=0.5, n=1500,
                           reps=400, seed=1):
    """Return (reported_SE / MC_std, empirical 95% CI coverage).

    DGP is linear so the local-linear fit is unbiased at x0 -> the Monte-Carlo
    standard deviation of the estimator is a clean target for the reported SE,
    and coverage isolates the SE (no bias confound).
    """
    rng = np.random.default_rng(seed)
    z = 1.959963984540054
    ests, ses = [], []
    cover = 0
    true = 1.0 + 2.0 * x0
    for _ in range(reps):
        x = rng.uniform(0, 1, n)
        y = 1.0 + 2.0 * x + rng.normal(0, 1, n)
        r = lpoly(data=pd.DataFrame({"y": y, "x": x}), y="y", x="x",
                  bandwidth=h, degree=degree, kernel=kernel,
                  grid=np.array([x0]), ci=False)
        f, s = r.fitted[0], r.se[0]
        ests.append(f)
        ses.append(s)
        if abs(f - true) <= z * s:
            cover += 1
    mc_std = np.std(ests, ddof=1)
    return float(np.nanmean(ses) / mc_std), cover / reps


def test_lpoly_se_matches_mc_variance_gaussian():
    # The Gaussian kernel exposed the bug most clearly (old ratio ~0.75).
    ratio, _ = _mc_ratio_and_coverage("gaussian")
    assert 0.88 < ratio < 1.15, f"SE/MC-std ratio {ratio:.3f} off nominal"


def test_lpoly_se_matches_mc_variance_epanechnikov():
    ratio, _ = _mc_ratio_and_coverage("epanechnikov")
    assert 0.88 < ratio < 1.15, f"SE/MC-std ratio {ratio:.3f} off nominal"


def test_lpoly_ci_coverage_is_nominal():
    _, cov = _mc_ratio_and_coverage("gaussian")
    # Old (no sandwich meat) under-covered; nominal is 0.95.
    assert cov >= 0.90, f"95% CI coverage {cov:.3f} too low"


def test_lpoly_se_includes_sandwich_meat():
    """On a fixed sample the reported SE must exceed the no-meat SE.

    Reproduces the old (incorrect) ``sigma^2 (X'WX)^-1`` SE by hand and checks
    the corrected SE is strictly larger -- i.e. the X'W^2 X meat is present.
    """
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 800)
    y = 1.0 + 2.0 * x + rng.normal(0, 1, 800)
    x0, h, degree, kernel = 0.5, 0.2, 1, "gaussian"

    _, se_reported = _local_poly_fit(x0, x, y, h, degree, kernel)

    # Hand-rebuild the OLD no-meat estimator.
    u = (x - x0) / h
    w = np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi) / h
    m = w > 0
    xl, yl, wl = x[m] - x0, y[m], w[m]
    X = np.column_stack([xl**j for j in range(degree + 1)])
    XtWX = X.T @ (wl[:, None] * X)
    beta = np.linalg.solve(XtWX, X.T @ (wl * yl))
    resid = yl - X @ beta
    sigma2_old = np.sum(wl * resid**2) / max(m.sum() - degree - 1, 1)
    se_old = np.sqrt(sigma2_old * np.linalg.inv(XtWX)[0, 0])

    assert np.isfinite(se_reported)
    assert se_reported > se_old
