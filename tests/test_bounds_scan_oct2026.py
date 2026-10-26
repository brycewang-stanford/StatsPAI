"""Option scan (2026-10-27): partial-identification bounds.

Each test states the correct behaviour; every one failed before the fix
landed with it (the defect is named in the test's comment or docstring).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _selection_frame(seed: int = 3, n: int = 1203, cells: bool = False) -> pd.DataFrame:
    """Treatment raises retention in every cell (homogeneous direction)."""
    rng = np.random.default_rng(seed)
    d = rng.integers(0, 2, n)
    g = rng.integers(0, 3, n) if cells else np.zeros(n, dtype=int)
    keep = np.where(d == 1, 0.85, 0.6) - 0.1 * g
    s = (rng.uniform(size=n) < keep).astype(int)
    y = 1.0 + 0.5 * d + 0.4 * g + rng.normal(size=n)
    return pd.DataFrame({"y": np.where(s == 1, y, np.nan), "d": d, "s": s, "g": g})


# ----------------------------------------------------------------------
# 1. selection_bounds(unconditional) == lee_bounds(trimming='quantile')
# ----------------------------------------------------------------------
def test_selection_bounds_unconditional_equals_lee_bounds():
    df = _selection_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = sp.lee_bounds(
            df, y="y", treat="d", selection="s", trimming="quantile", n_bootstrap=20
        )
        got = sp.selection_bounds(
            df, y="y", treatment="d", selection="s", method="unconditional", n_boot=5
        )
    assert got.lower == pytest.approx(ref.model_info["lower_bound"], abs=1e-12)
    assert got.upper == pytest.approx(ref.model_info["upper_bound"], abs=1e-12)


# ----------------------------------------------------------------------
# 2. selection_bounds(conditional) == lee_bounds(covariates=)
# ----------------------------------------------------------------------
def test_selection_bounds_conditional_equals_tightened_lee_bounds():
    df = _selection_frame(cells=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = sp.lee_bounds(
            df,
            y="y",
            treat="d",
            selection="s",
            covariates=["g"],
            trimming="quantile",
            n_bootstrap=30,
        )
        got = sp.selection_bounds(
            df,
            y="y",
            treatment="d",
            selection="s",
            covariates=["g"],
            method="conditional",
            n_boot=5,
        )
    assert got.lower == pytest.approx(ref.model_info["lower_bound"], abs=1e-12)
    assert got.upper == pytest.approx(ref.model_info["upper_bound"], abs=1e-12)


# ----------------------------------------------------------------------
# 3. iv_bounds: the identified set must contain the truth under its own
#    assumption (population-scale n; bias gap >> sampling noise)
# ----------------------------------------------------------------------
def _imperfect_iv_frame(c_zu: float, c_vu: float, n: int = 200_000) -> pd.DataFrame:
    """y = 1.0 * x + u, x = z + v, u = c_zu z + c_vu v + 0.5 e.

    cov(z, u) = c_zu > 0 (direct effect of z on y, same sign as the indirect
    effect z -> x -> y = 1); cov(x, u) = c_zu + c_vu > 0. True beta = 1.
    """
    rng = np.random.default_rng(11)
    z, v, e = rng.normal(size=(3, n))
    u = c_zu * z + c_vu * v + 0.5 * e
    x = z + v
    return pd.DataFrame({"y": x + u, "x": x, "z": z})


def test_iv_bounds_monotone_iv_contains_truth():
    """Positive cov(z, x): OLS (1.5) and IV (1.2) both overstate beta = 1.

    The old interval [min(OLS, IV), max(OLS, IV)] = [1.2, 1.5] excluded the
    truth. Nevo-Rosen (A3 + A4, corr(x, u) > 0): (-inf, min(IV_z, IV_V1)].
    """
    df = _imperfect_iv_frame(c_zu=0.2, c_vu=0.8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.iv_bounds(
            df, y="y", treatment="x", instrument="z", assumption="monotone_iv", n_boot=2
        )
    assert res.lower == -np.inf
    assert res.model_info["closed"] is False
    assert res.upper == pytest.approx(res.model_info["beta_iv_z"], abs=0)
    assert res.lower <= 1.0 <= res.upper
    assert np.isnan(res.se_lower) and res.ci_lower == (-np.inf, -np.inf)


def _negative_iv_frame(n: int = 200_000, flip_u: bool = False) -> pd.DataFrame:
    """x = -z + v: cov(z, x) < 0, so the set is closed (Proposition 2).

    cov(z, u) = 0.2, cov(x, u) = 0.6, both of the sign of corr(x, u).
    """
    rng = np.random.default_rng(12)
    z, v, e = rng.normal(size=(3, n))
    u = 0.2 * z + 0.8 * v + 0.5 * e
    if flip_u:
        u = -u
    x = -z + v
    return pd.DataFrame({"y": x + u, "x": x, "z": z})


@pytest.mark.parametrize("flip_u", [False, True])
def test_iv_bounds_closed_set_contains_truth(flip_u):
    df = _negative_iv_frame(flip_u=flip_u)
    res = sp.iv_bounds(
        df,
        y="y",
        treatment="x",
        instrument="z",
        n_boot=2,
        xu_sign="negative" if flip_u else "positive",
    )
    assert res.model_info["closed"] is True
    assert res.lower < 1.0 < res.upper
    # population endpoints: IV_z = 1 -+ 0.2, IV_V1 = 1 +- 0.317 / 3.414
    lo, hi = (0.8, 1.0 + 0.3172 / 3.4142) if not flip_u else (1 - 0.3172 / 3.4142, 1.2)
    assert res.lower == pytest.approx(lo, abs=0.02)
    assert res.upper == pytest.approx(hi, abs=0.02)


def test_iv_bounds_v1_endpoint_obeys_corollary_2():
    """(IV_V1 - IV_z) / (OLS - IV_z) = 1 / (1 - corr(x, z)), an exact identity
    of the sample moments in the model without controls."""
    df = _negative_iv_frame(n=500)
    res = sp.iv_bounds(df, y="y", treatment="x", instrument="z", n_boot=2)
    mi = res.model_info
    rho = np.corrcoef(df["x"], df["z"])[0, 1]
    ratio = (mi["beta_iv_v1"] - mi["beta_iv_z"]) / (
        mi["ols_estimate"] - mi["beta_iv_z"]
    )
    assert ratio == pytest.approx(1.0 / (1.0 - rho), rel=1e-10)


def test_iv_bounds_controls_equal_partialled_model():
    """Exogenous controls are partialled out (Proposition 3): the endpoints
    equal 2SLS of y on (x, w) with z, and with V(1) = sd(z) x - sd(x) z."""
    rng = np.random.default_rng(3)
    n = 600
    w = rng.normal(size=n)
    z = rng.normal(size=n) + 0.5 * w
    v = rng.normal(size=n)
    x = -z + v + 0.3 * w
    y = x + 0.4 * w + 0.2 * z + 0.8 * v + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "x": x, "z": z, "w": w})
    res = sp.iv_bounds(
        df, y="y", treatment="x", instrument="z", controls=["w"], n_boot=2
    )
    W = np.column_stack([np.ones(n), w])

    def iv(inst):
        Xd = np.column_stack([x, W])
        Zd = np.column_stack([inst, W])
        return np.linalg.solve(Zd.T @ Xd, Zd.T @ y)[0]

    v1 = z.std(ddof=1) * x - x.std(ddof=1) * z
    assert res.model_info["beta_iv_z"] == pytest.approx(iv(z), rel=1e-10)
    assert res.model_info["beta_iv_v1"] == pytest.approx(iv(v1), rel=1e-10)


def test_iv_bounds_less_than_late_is_deprecated_and_unchanged():
    """'less_than_late' (Wald +/- |OLS - Wald|) has no source; it warns and
    keeps its old numbers until removal."""
    df = _imperfect_iv_frame(c_zu=0.2, c_vu=0.1, n=2000)
    with pytest.warns(DeprecationWarning, match="less_than_late"):
        res = sp.iv_bounds(
            df,
            y="y",
            treatment="x",
            instrument="z",
            assumption="less_than_late",
            n_boot=2,
        )
    x, y_, z = df["x"].to_numpy(), df["y"].to_numpy(), df["z"].to_numpy()
    wald = np.cov(z, y_)[0, 1] / np.cov(z, x)[0, 1]
    ols = np.cov(x, y_)[0, 1] / np.var(x, ddof=1)
    assert res.lower == pytest.approx(wald - abs(ols - wald), rel=1e-10)
    assert res.upper == pytest.approx(wald + abs(ols - wald), rel=1e-10)


def test_iv_bounds_first_stage_f_is_the_regression_F():
    rng = np.random.default_rng(5)
    n = 400
    z = rng.normal(size=n)
    x = 0.3 * z + rng.normal(size=n)
    df = pd.DataFrame({"y": x + rng.normal(size=n), "x": x, "z": z})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.iv_bounds(df, y="y", treatment="x", instrument="z", n_boot=2)
    r2 = np.corrcoef(z, x)[0, 1] ** 2
    f_ref = (n - 2) * r2 / (1 - r2)
    assert res.model_info["first_stage_f"] == pytest.approx(f_ref, rel=1e-10)


# ----------------------------------------------------------------------
# 7. BoundsResult CIs are Wald intervals around the point bounds
# ----------------------------------------------------------------------
def test_bounds_ci_is_centred_on_the_point_bound():
    rng = np.random.default_rng(2)
    n = 300
    d = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {
            "y": np.exp(rng.normal(size=n)) + d,  # skewed: median != point
            "d": d,
            "g": rng.integers(0, 3, n),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.horowitz_manski(
            df, y="y", treatment="d", covariates=["g"], n_boot=200, random_state=1
        )
    mid_lo = 0.5 * (res.ci_lower[0] + res.ci_lower[1])
    mid_hi = 0.5 * (res.ci_upper[0] + res.ci_upper[1])
    scale = abs(res.upper - res.lower)
    assert mid_lo == pytest.approx(res.lower, abs=1e-12 * scale)
    assert mid_hi == pytest.approx(res.upper, abs=1e-12 * scale)


# ----------------------------------------------------------------------
# 8. balke_pearl monotone bounds
# ----------------------------------------------------------------------
_COMPLIANCE = {"never": (0, 0), "always": (1, 1), "complier": (0, 1)}  # (D(0), D(1))


def _exact_iv_frame(q: dict) -> pd.DataFrame:
    """Rows replicated in exact proportion to a response-type mixture.

    ``q`` maps (compliance, y0, y1) -> integer count per arm of Z.
    """
    rows = []
    for (ctype, y0, y1), k in q.items():
        d0, d1 = _COMPLIANCE[ctype]
        for zv, dv in ((0, d0), (1, d1)):
            yv = y1 if dv == 1 else y0
            rows += [(yv, dv, zv)] * k
    return pd.DataFrame(rows, columns=["y", "d", "z"])


_MIX = {
    ("never", 0, 0): 30,
    ("never", 0, 1): 10,
    ("never", 1, 1): 15,
    ("always", 0, 1): 12,
    ("always", 1, 1): 18,
    ("complier", 0, 1): 40,
    ("complier", 0, 0): 25,
    ("complier", 1, 1): 20,
    ("complier", 1, 0): 5,
}


def _lp_monotone_bounds(df: pd.DataFrame) -> tuple:
    """Sharp ATE bounds under exogeneity + monotonicity (no defiers) by LP."""
    from scipy.optimize import linprog

    types = [(c, y0, y1) for c in _COMPLIANCE for y0 in (0, 1) for y1 in (0, 1)]
    A, b = [], []
    for zv in (0, 1):
        sub = df[df.z == zv]
        for dv in (0, 1):
            for yv in (0, 1):
                row = []
                for c, y0, y1 in types:
                    d_of_z = _COMPLIANCE[c][zv]
                    y_obs = y1 if d_of_z == 1 else y0
                    row.append(float(d_of_z == dv and y_obs == yv))
                A.append(row)
                b.append(float(((sub.d == dv) & (sub.y == yv)).mean()))
    cost = np.array([y1 - y0 for _, y0, y1 in types], dtype=float)
    lo = linprog(cost, A_eq=A, b_eq=b, bounds=(0, 1), method="highs")
    hi = linprog(-cost, A_eq=A, b_eq=b, bounds=(0, 1), method="highs")
    assert lo.status == 0 and hi.status == 0
    return lo.fun, -hi.fun


def test_balke_pearl_monotone_bounds_match_the_linear_program():
    df = _exact_iv_frame(_MIX)
    res = sp.balke_pearl(df, y="y", treat="d", instrument="z")
    lo, hi = _lp_monotone_bounds(df)
    assert res.lower_monotone == pytest.approx(lo, abs=1e-12)
    assert res.upper_monotone == pytest.approx(hi, abs=1e-12)


def test_balke_pearl_monotone_bounds_invariant_to_relabelling_the_instrument():
    df = _exact_iv_frame(_MIX)
    flipped = df.assign(z=1 - df["z"])
    a = sp.balke_pearl(df, y="y", treat="d", instrument="z")
    try:
        b = sp.balke_pearl(flipped, y="y", treat="d", instrument="z")
    except ValueError:
        # refusing a negative first stage (and saying so) is also correct
        return
    assert (b.lower, b.upper) == pytest.approx((a.lower, a.upper), abs=1e-12)
    assert (b.lower_monotone, b.upper_monotone) == pytest.approx(
        (a.lower_monotone, a.upper_monotone), abs=1e-12
    )


# ----------------------------------------------------------------------
# 9. manski_bounds validation
# ----------------------------------------------------------------------
def test_manski_bounds_rejects_non_binary_treatment():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"y": rng.uniform(size=50), "d": rng.integers(1, 3, 50)})
    with pytest.raises(ValueError):
        sp.manski_bounds(df, y="y", treat="d", y_lower=0, y_upper=1, n_bootstrap=5)


def test_manski_bounds_rejects_support_inside_observed_range():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"y": rng.uniform(size=50) * 2, "d": rng.integers(0, 2, 50)})
    with pytest.raises(ValueError):
        sp.manski_bounds(df, y="y", treat="d", y_lower=0, y_upper=1, n_bootstrap=5)
