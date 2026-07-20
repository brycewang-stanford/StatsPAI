"""Coverage gap-fill round 2 — ``statspai.rd`` remaining branches.

Targets (from the uncovered-line report):

- ``rdrobust.py`` argument-validation helpers (empty data, non-string column
  names, bad kernel / c / alpha / deriv / covs / cluster inputs);
- secondary estimation options: bare-string ``covs``, explicit ``q``,
  ``bwselect='msetwo'`` combined with ``rho`` (two-sided b = h / rho),
  ``cluster`` + ``donut`` filtering of cluster values;
- CCT (official rdrobust) delegation: kink (``deriv=1``) labeling and the
  donut + fuzzy + covs filtering path;
- ``_parse_data`` errors: missing fuzzy column, all-NaN outcome;
- rbc bootstrap with two-sided bandwidths, and its DataInsufficient guard
  when the bandwidth leaves too few observations;
- ``rdplot`` small-sample branches (IMSE bins with n<10, <3-point polynomial
  fit, singleton-bin SE, show_bw fallback when internal rdrobust fails) and
  the zero-curvature IMSE branch; ``rdplotdensity`` with a too-sparse side;
- ``rd2d`` / ``rd2d_bw`` / ``rd2d_plot`` validation and option branches.
"""

from __future__ import annotations

import importlib.util
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import statspai as sp  # noqa: E402
from statspai.exceptions import DataInsufficient, MethodIncompatibility  # noqa: E402

_HAS_CCT = importlib.util.find_spec("rdrobust") is not None
_skip_cct = pytest.mark.skipif(
    not _HAS_CCT, reason="official rdrobust package not installed"
)


def _sharp(n=1200, tau=3.0, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, n)
    Z = rng.normal(0, 1, n)
    Y = 0.5 * X + tau * (X >= 0) + 0.3 * Z + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"y": Y, "x": X, "z": Z})
    df["g"] = np.arange(n) // 30
    return df


def _fuzzy(n=1500, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, n)
    prob = 0.15 + 0.7 * (X >= 0)
    D = (rng.uniform(0, 1, n) < prob).astype(float)
    Y = 0.5 * X + 2.0 * D + rng.normal(0, 0.4, n)
    return pd.DataFrame({"y": Y, "x": X, "d": D, "w": rng.normal(0, 1, n)})


# ── rdrobust argument validation ──────────────────────────────────────────


@pytest.mark.parametrize(
    "kwargs, exc, match",
    [
        (dict(data=pd.DataFrame()), DataInsufficient, "is empty"),
        (dict(y=123), MethodIncompatibility, "non-empty column name"),
        (dict(kernel=5), MethodIncompatibility, "must be a string option"),
        (dict(kernel="   "), MethodIncompatibility, "non-empty string option"),
        (dict(c=True), MethodIncompatibility, "must be a finite number"),
        (dict(c="abc"), MethodIncompatibility, "must be a finite number"),
        (dict(c=np.inf), MethodIncompatibility, "must be finite"),
        (dict(alpha=1.5), MethodIncompatibility, r"in \(0, 1\)"),
        (dict(deriv=True), MethodIncompatibility, "integer >= 0"),
        (dict(deriv="x"), MethodIncompatibility, "integer >= 0"),
        (dict(covs=123), MethodIncompatibility, "column name or list"),
        (dict(covs=[""]), MethodIncompatibility, "non-empty string column names"),
        (dict(cluster="nope"), MethodIncompatibility, "not found in data"),
    ],
)
def test_rdrobust_argument_validation(kwargs, exc, match):
    df = _sharp(n=300)
    call = dict(data=df, y="y", x="x")
    call.update(kwargs)
    with pytest.raises(exc, match=match):
        sp.rdrobust(**call)


def test_rdrobust_fuzzy_column_missing():
    df = _sharp(n=300)
    with pytest.raises(MethodIncompatibility, match="Fuzzy variable 'nope'"):
        sp.rdrobust(df, y="y", x="x", fuzzy="nope")


def test_rdrobust_all_nan_outcome():
    df = _sharp(n=300)
    df["y"] = np.nan
    with pytest.raises(DataInsufficient, match="No finite RD observations"):
        sp.rdrobust(df, y="y", x="x")


# ── rdrobust secondary options ────────────────────────────────────────────


def test_rdrobust_covs_as_bare_string():
    # `covs="z"` (scalar string) must be promoted to ["z"] and adjust for z.
    df = _sharp()
    res = sp.rdrobust(df, y="y", x="x", covs="z")
    assert res.se > 0
    assert abs(res.estimate - 3.0) < 0.5


def test_rdrobust_explicit_q():
    df = _sharp()
    res = sp.rdrobust(df, y="y", x="x", q=3)
    assert res.model_info["polynomial_q"] == 3
    assert abs(res.estimate - 3.0) < 0.5


def test_rdrobust_msetwo_with_rho_mirrors_bandwidths():
    # Two-sided bandwidths + rho=1 must give b = h side-by-side (CCT 2018).
    df = _sharp()
    res = sp.rdrobust(df, y="y", x="x", bwselect="msetwo", rho=1.0)
    h = res.model_info["bandwidth_h"]
    b = res.model_info["bandwidth_b"]
    assert isinstance(h, tuple) and isinstance(b, tuple)
    assert h[0] != h[1]  # genuinely two-sided
    assert b == pytest.approx(h)  # rho=1 reproduces the default b=h


def test_rdrobust_cluster_with_donut_filters_cluster_values():
    df = _sharp()
    res = sp.rdrobust(df, y="y", x="x", cluster="g", donut=0.03)
    assert res.se > 0
    assert res.model_info["donut"] == 0.03
    assert abs(res.estimate - 3.0) < 0.6


# ── CCT delegation branches ───────────────────────────────────────────────


@_skip_cct
def test_cct_kink_labels_rkd():
    df = _sharp()
    res = sp.rdrobust(df, y="y", x="x", bwselect="cct", deriv=1)
    assert res.model_info["rd_type"] == "Kink"
    assert "RKD" in res.estimand
    assert res.se > 0


@_skip_cct
def test_cct_fuzzy_covs_with_donut():
    df = _fuzzy()
    res = sp.rdrobust(
        df, y="y", x="x", bwselect="cct", fuzzy="d", covs=["w"], donut=0.02
    )
    assert res.model_info["rd_type"] == "Fuzzy"
    assert abs(res.estimate - 2.0) < 1.0
    assert res.se > 0


# ── rbc bootstrap branches ────────────────────────────────────────────────


def test_rbc_bootstrap_two_sided_bandwidths():
    df = _sharp()
    res = sp.rdrobust(
        df, y="y", x="x", bwselect="msetwo", bootstrap="rbc", n_boot=99, random_state=0
    )
    boot = res.model_info["rbc_bootstrap"]
    lo, hi = boot["ci"]
    assert lo < hi
    assert lo < 3.0 < hi  # CI covers the seeded jump


def test_rbc_bootstrap_tiny_bandwidth_raises():
    df = _sharp(n=500)
    with pytest.raises(DataInsufficient, match="rbc bootstrap"):
        sp.rdrobust(df, y="y", x="x", h=0.004, bootstrap="rbc", n_boot=99)


# ── rdplot / rdplotdensity small-sample branches ──────────────────────────


def test_rdplot_tiny_sides_show_bw_fallback():
    # 8 left / 2 right points: n<10 IMSE branch, <3-point poly fit -> NaN
    # curves, singleton-bin SE=0, and show_bw's internal rdrobust failure
    # (right side < p+2) silently falls back to no shading.
    rng = np.random.default_rng(5)
    xs = np.concatenate([rng.uniform(-1, 0, 8), rng.uniform(0, 1, 2)])
    df = pd.DataFrame(
        {"y": 0.2 * xs + 1.0 * (xs >= 0) + rng.normal(0, 0.1, 10), "x": xs}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig, ax = sp.rdplot(df, y="y", x="x", c=0, show_bw=True)
    assert fig is not None and ax is not None
    plt.close("all")


def test_rdplot_exactly_linear_outcome_zero_curvature():
    # Zero curvature (m2 ~ 0): IMSE bin selector must fall back to J_base.
    rng = np.random.default_rng(11)
    df = pd.DataFrame({"x": rng.uniform(-1, 1, 200)})
    df["y"] = 2.0 * df["x"] + 1.0
    fig, ax = sp.rdplot(df, y="y", x="x", c=0)
    assert ax is not None
    plt.close("all")


def test_rdplotdensity_sparse_right_side():
    # Right side has 4 points < max(p+2, 5): CJM density returns NaN there,
    # the plot must still be produced from the left-side curve.
    rng = np.random.default_rng(13)
    xd = np.concatenate([rng.uniform(-1, 0, 300), rng.uniform(0, 1, 4)])
    fig, ax = sp.rdplotdensity(pd.DataFrame({"x": xd}), x="x", c=0)
    assert fig is not None and ax is not None
    plt.close("all")


# ── rd2d ──────────────────────────────────────────────────────────────────


def _rd2d_data(n=500, tau=1.5, seed=21):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-1, 1, n)
    x2 = rng.uniform(-1, 1, n)
    tr = (x2 >= 0.3 * x1).astype(int)
    y = tau * tr + 0.5 * x1 + 0.3 * x2 + rng.normal(0, 0.4, n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "t": tr})


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(data=[1, 2]), "must be a pandas DataFrame"),
        (dict(y=None), "non-empty column-name string"),
        (dict(approach="weird"), "'distance' or 'location'"),
        (dict(kernel="gauss"), "kernel must be"),
        (dict(p=-1), "non-negative integer"),
        (dict(h=-1.0), "h must be positive"),
        (dict(alpha=2.0), "alpha must be between 0 and 1"),
        (dict(boundary=5), "boundary must be callable"),
        (dict(n_eval=0), "n_eval must be a positive integer"),
    ],
)
def test_rd2d_argument_validation(kwargs, match):
    df = _rd2d_data(n=200)
    call = dict(data=df, y="y", x1="x1", x2="x2", treatment="t")
    call.update(kwargs)
    with pytest.raises(MethodIncompatibility, match=match):
        sp.rd2d(**call)


def test_rd2d_non_numeric_column_raises():
    df = _rd2d_data(n=200)
    df["y"] = "text"
    with pytest.raises(DataInsufficient, match="must be numeric"):
        sp.rd2d(df, y="y", x1="x1", x2="x2", treatment="t")


def test_rd2d_bw_location_approach_with_boundary():
    df = _rd2d_data()
    bw = sp.rd2d_bw(
        df,
        y="y",
        x1="x1",
        x2="x2",
        treatment="t",
        approach="location",
        boundary=lambda v: 0.3 * v,
    )
    assert np.isfinite(bw) and bw > 0


def test_rd2d_bw_boundary_not_callable_raises():
    df = _rd2d_data(n=200)
    with pytest.raises(MethodIncompatibility, match="boundary must be callable"):
        sp.rd2d_bw(df, y="y", x1="x1", x2="x2", treatment="t", boundary=7)


def test_rd2d_plot_heatmap_with_callable_boundary():
    df = _rd2d_data()
    fig, ax = sp.rd2d_plot(
        df,
        y="y",
        x1="x1",
        x2="x2",
        treatment="t",
        boundary=lambda v: 0.3 * v,
        plot_type="heatmap",
    )
    assert fig is not None and ax is not None
    plt.close("all")
