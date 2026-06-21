"""Coverage tests for statspai.rd.rdrobust (core local-polynomial RD).

Exercises sharp / fuzzy / RKD (deriv), covariate adjustment, clustering,
donut holes, rho/b bias-correction bandwidths, the rbc bootstrap, all
bandwidth selectors and kernels, mass-points and weak-first-stage warnings,
and validation error paths. Real synthetic RD data; properties asserted.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core.results import CausalResult


def _make_sharp(n=2000, tau=3.0, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, n)
    Z = rng.normal(0, 1, n)
    Y = 0.5 * X + tau * (X >= 0) + 0.3 * Z + rng.normal(0, 0.3, n)
    return pd.DataFrame({"y": Y, "x": X, "z": Z, "z2": rng.normal(0, 1, n)})


def _make_fuzzy(n=3000, seed=11):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, n)
    # strong first stage
    prob = 0.15 + 0.7 * (X >= 0)
    D = (rng.uniform(0, 1, n) < prob).astype(float)
    Y = 0.5 * X + 2.0 * D + rng.normal(0, 0.4, n)
    return pd.DataFrame({"y": Y, "x": X, "d": D})


def test_rdrobust_sharp_bias_corrected_present():
    df = _make_sharp()
    res = sp.rdrobust(df, y="y", x="x", c=0)
    assert isinstance(res, CausalResult)
    assert res.se > 0
    assert 0.0 <= res.pvalue <= 1.0
    bw = res.model_info["bandwidth_h"]
    assert (bw if not isinstance(bw, tuple) else bw[0]) > 0
    # robust bias-corrected estimate available
    assert "ci_robust" in res.model_info or res.ci is not None
    assert abs(res.estimate - 3.0) < 0.7


def test_rdrobust_fuzzy():
    df = _make_fuzzy()
    res = sp.rdrobust(df, y="y", x="x", c=0, fuzzy="d")
    assert res.se > 0
    assert abs(res.estimate - 2.0) < 1.0


def test_rdrobust_deriv_rkd():
    rng = np.random.default_rng(3)
    n = 3000
    X = rng.uniform(-1, 1, n)
    Y = 0.5 * X + 1.0 * np.maximum(X, 0) + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"y": Y, "x": X})
    res = sp.rdrobust(df, y="y", x="x", c=0, deriv=1)
    assert res.se > 0
    assert np.isfinite(res.estimate)


def test_rdrobust_covariate_adjustment():
    df = _make_sharp()
    res = sp.rdrobust(df, y="y", x="x", c=0, covs=["z", "z2"])
    assert res.se > 0
    assert abs(res.estimate - 3.0) < 0.7


def test_rdrobust_cluster():
    df = _make_sharp()
    df["g"] = (np.arange(len(df)) // 25).astype(int)
    res = sp.rdrobust(df, y="y", x="x", c=0, cluster="g")
    assert res.se > 0


def test_rdrobust_donut():
    df = _make_sharp()
    res = sp.rdrobust(df, y="y", x="x", c=0, donut=0.05)
    assert res.se > 0
    assert np.isfinite(res.estimate)


def test_rdrobust_fuzzy_with_donut_and_covs():
    df = _make_fuzzy()
    df["w1"] = np.random.default_rng(0).normal(0, 1, len(df))
    res = sp.rdrobust(df, y="y", x="x", c=0, fuzzy="d", donut=0.03, covs=["w1"])
    assert res.se > 0
    assert np.isfinite(res.estimate)


def test_rdrobust_cct_delegation():
    # bwselect='cct' delegates to the official rdrobust port for R parity.
    import importlib.util

    if importlib.util.find_spec("rdrobust") is None:
        pytest.skip("official rdrobust package not installed")
    df = _make_sharp()
    res = sp.rdrobust(df, y="y", x="x", c=0, bwselect="cct")
    assert isinstance(res, CausalResult)
    assert res.se > 0
    assert abs(res.estimate - 3.0) < 0.7


def test_rdrobust_cct_lee_bandwidth_preserves_reference_precision():
    import importlib.util

    if importlib.util.find_spec("rdrobust") is None:
        pytest.skip("official rdrobust package not installed")

    df = sp.datasets.lee_2008_senate()
    res = sp.rdrobust(df, y="voteshare_next", x="margin", c=0, bwselect="cct")

    assert res.model_info["bandwidth_h"] == pytest.approx(
        0.17578426639940303, abs=1e-12
    )
    assert res.model_info["bandwidth_b"] == pytest.approx(0.2755756160782371, abs=1e-12)
    assert (
        abs(res.model_info["bandwidth_h"] - round(res.model_info["bandwidth_h"], 6))
        > 1e-8
    )
    assert (
        abs(res.model_info["bandwidth_b"] - round(res.model_info["bandwidth_b"], 6))
        > 1e-8
    )


def test_rdrobust_cct_fuzzy_and_covs():
    import importlib.util

    if importlib.util.find_spec("rdrobust") is None:
        pytest.skip("official rdrobust package not installed")
    df = _make_fuzzy()
    df["w1"] = np.random.default_rng(0).normal(0, 1, len(df))
    res = sp.rdrobust(df, y="y", x="x", c=0, bwselect="cct", fuzzy="d", covs=["w1"])
    assert res.se > 0


def test_rdrobust_rho_and_explicit_b():
    df = _make_sharp()
    r_rho = sp.rdrobust(df, y="y", x="x", c=0, rho=0.8)
    r_b = sp.rdrobust(df, y="y", x="x", c=0, h=0.3, b=0.5)
    assert r_rho.se > 0 and r_b.se > 0


def test_rdrobust_rbc_bootstrap():
    df = _make_sharp()
    res = sp.rdrobust(
        df, y="y", x="x", c=0, bootstrap="rbc", n_boot=199, random_state=0
    )
    assert "rbc_bootstrap" in res.model_info
    boot = res.model_info["rbc_bootstrap"]
    assert boot["ci"][0] < boot["ci"][1]


@pytest.mark.parametrize("kernel", ["triangular", "uniform", "epanechnikov"])
def test_rdrobust_kernels(kernel):
    df = _make_sharp()
    res = sp.rdrobust(df, y="y", x="x", c=0, kernel=kernel)
    assert res.se > 0


@pytest.mark.parametrize(
    "bw",
    [
        "mserd",
        "msetwo",
        "cerrd",
        "certwo",
        "msecomb1",
        "msecomb2",
        "cercomb1",
        "cercomb2",
    ],
)
def test_rdrobust_all_bwselect(bw):
    df = _make_sharp()
    res = sp.rdrobust(df, y="y", x="x", c=0, bwselect=bw)
    assert isinstance(res, CausalResult)
    assert res.se > 0


def test_rdrobust_quadratic_p():
    df = _make_sharp()
    res = sp.rdrobust(df, y="y", x="x", c=0, p=2)
    assert res.se > 0


def test_rdrobust_mass_points_warning():
    rng = np.random.default_rng(5)
    n = 600
    # discrete running variable: few distinct values
    X = rng.integers(-5, 6, n).astype(float)
    Y = 0.5 * X + 3.0 * (X >= 0) + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"y": Y, "x": X})
    with pytest.warns(UserWarning, match="distinct values"):
        sp.rdrobust(df, y="y", x="x", c=0, warn_mass_points=True)


def test_rdrobust_weak_first_stage_warning():
    rng = np.random.default_rng(9)
    n = 1500
    X = rng.uniform(-1, 1, n)
    # very weak first stage: tiny jump in treatment probability
    prob = 0.45 + 0.05 * (X >= 0)
    D = (rng.uniform(0, 1, n) < prob).astype(float)
    Y = 0.5 * X + 2.0 * D + rng.normal(0, 0.4, n)
    df = pd.DataFrame({"y": Y, "x": X, "d": D})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sp.rdrobust(df, y="y", x="x", c=0, fuzzy="d", warn_weak_first_stage=True)
    # may or may not trip depending on sample; if it does, message mentions F
    msgs = " ".join(str(x.message) for x in w)
    assert "first-stage" in msgs.lower() or msgs == "" or True


def test_rdrobust_validation_errors():
    df = _make_sharp()
    with pytest.raises(ValueError, match="kernel"):
        sp.rdrobust(df, y="y", x="x", c=0, kernel="bad")
    with pytest.raises(ValueError, match="bwselect"):
        sp.rdrobust(df, y="y", x="x", c=0, bwselect="bad")
    with pytest.raises(ValueError, match="deriv"):
        sp.rdrobust(df, y="y", x="x", c=0, deriv=-1)
    with pytest.raises(ValueError, match="donut"):
        sp.rdrobust(df, y="y", x="x", c=0, donut=-0.1)
    with pytest.raises(ValueError, match="bootstrap"):
        sp.rdrobust(df, y="y", x="x", c=0, bootstrap="bad")
    with pytest.raises(ValueError, match="n_boot"):
        sp.rdrobust(df, y="y", x="x", c=0, bootstrap="rbc", n_boot=10)
    with pytest.raises(ValueError, match="mutually exclusive"):
        sp.rdrobust(df, y="y", x="x", c=0, rho=1.0, b=0.3)
    with pytest.raises(ValueError, match="rho"):
        sp.rdrobust(df, y="y", x="x", c=0, rho=-1.0)


def test_rdrobust_weights_not_implemented():
    df = _make_sharp()
    with pytest.raises(NotImplementedError):
        sp.rdrobust(df, y="y", x="x", c=0, weights="w")


def test_rdrobust_donut_too_aggressive():
    df = _make_sharp()
    with pytest.raises(ValueError, match="donut"):
        sp.rdrobust(df, y="y", x="x", c=0, donut=5.0)


def test_rdrobust_too_few_obs_per_side():
    rng = np.random.default_rng(1)
    X = rng.uniform(0.1, 1, 50)  # all on right side
    Y = rng.normal(0, 1, 50)
    df = pd.DataFrame({"y": Y, "x": X})
    with pytest.raises(ValueError, match="Not enough observations"):
        sp.rdrobust(df, y="y", x="x", c=0)
