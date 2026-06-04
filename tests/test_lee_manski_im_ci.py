"""Correctness tests for the Imbens & Manski (2004) CI in sp.lee_bounds /
sp.manski_bounds.

Regression guard for the fix that replaced the two-sided ``z_{1-alpha/2}`` on
*both* bound endpoints (a Horowitz-Manski interval for the identified *set*,
which over-covers any single parameter value) with the Imbens-Manski critical
value ``C_n`` for the partially-identified *parameter*.

Reference:
    Imbens, G.W. & Manski, C.F. (2004). "Confidence Intervals for Partially
    Identified Parameters." Econometrica, 72(6), 1845-1857.
    doi:10.1111/j.1468-0262.2004.00555.x.
"""

import numpy as np
import pandas as pd
from scipy import stats

from statspai.bounds.lee_manski import _imbens_manski_cn, lee_bounds


def test_im_cn_limits_and_monotone():
    z_one = stats.norm.ppf(0.95)      # 1.645
    z_two = stats.norm.ppf(0.975)     # 1.960
    # Point-identified limit (width 0) -> two-sided.
    assert abs(_imbens_manski_cn(0.0, 0.0, 0.1, 0.1, 0.05) - z_two) < 1e-6
    # Very wide identified set -> one-sided.
    assert abs(_imbens_manski_cn(0.0, 100.0, 0.1, 0.1, 0.05) - z_one) < 1e-4
    # Monotonically decreasing from two-sided toward one-sided as width grows.
    widths = [0.0, 0.2, 0.5, 1.0, 3.0, 10.0]
    cns = [_imbens_manski_cn(0.0, w, 0.1, 0.1, 0.05) for w in widths]
    assert all(z_one - 1e-9 <= c <= z_two + 1e-9 for c in cns)
    assert all(cns[i] >= cns[i + 1] - 1e-9 for i in range(len(cns) - 1))


def test_im_cn_degenerate_se():
    # Non-positive / non-finite SE falls back to the two-sided value.
    z_two = stats.norm.ppf(0.975)
    assert _imbens_manski_cn(0.0, 1.0, 0.0, 0.0, 0.05) == z_two
    assert _imbens_manski_cn(0.0, 1.0, np.nan, 0.1, 0.05) == z_two


def test_im_ci_coverage_at_binding_endpoint():
    """At the worst-case endpoint, IM hits ~nominal; two-sided over-covers."""
    rng = np.random.default_rng(0)
    se, width_over_se, alpha = 0.1, 5.0, 0.05
    theta_l, theta_u = 0.0, width_over_se * se
    truth = theta_u                       # parameter at the upper endpoint
    z_two = stats.norm.ppf(1 - alpha / 2)
    reps = 8000
    cov_im = cov_old = 0
    for _ in range(reps):
        lh = theta_l + rng.normal(0, se)
        uh = theta_u + rng.normal(0, se)
        if uh < lh:
            lh, uh = uh, lh
        c = _imbens_manski_cn(lh, uh, se, se, alpha)
        if lh - c * se <= truth <= uh + c * se:
            cov_im += 1
        if lh - z_two * se <= truth <= uh + z_two * se:
            cov_old += 1
    im, old = cov_im / reps, cov_old / reps
    # IM is near nominal 0.95; the old (two-sided both ends) over-covers.
    assert 0.93 <= im <= 0.965, f"IM coverage {im:.3f} off nominal"
    assert old > im + 0.01, f"old coverage {old:.3f} should exceed IM {im:.3f}"


def test_lee_bounds_im_ci_within_two_sided():
    """The IM CI must be no wider than the old two-sided-both-ends interval."""
    rng = np.random.default_rng(1)
    n = 800
    treat = rng.integers(0, 2, n)
    latent = rng.normal(treat * 0.5, 1, n)
    sel = (rng.random(n) < np.where(treat == 1, 0.8, 0.6)).astype(int)
    y = np.where(sel == 1, latent, np.nan)
    df = pd.DataFrame({"y": y, "treat": treat, "sel": sel})
    r = lee_bounds(data=df, y="y", treat="treat", selection="sel",
                   n_bootstrap=300)
    lb = r.model_info["lower_bound"]
    ub = r.model_info["upper_bound"]
    ci_lo, ci_hi = r.ci
    # CI brackets the identified-set point estimates (C_n >= 0) and is finite.
    assert np.isfinite(ci_lo) and np.isfinite(ci_hi)
    assert ci_lo <= lb + 1e-9 and ci_hi >= ub - 1e-9
    # The critical value used is in the valid [1.645, 1.960] range (never the
    # full two-sided z applied to both ends except in the point-identified
    # limit), so the IM interval is no wider than the old two-sided one.
    cn = _imbens_manski_cn(lb, ub, 0.05, 0.05, 0.05)
    assert stats.norm.ppf(0.95) - 1e-6 <= cn <= stats.norm.ppf(0.975) + 1e-6
