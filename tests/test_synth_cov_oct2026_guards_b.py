"""Coverage round (Oct 2026): guards in the multi-unit generalized
synthetic control, distributional synthetic control and SDID modules,
plus the nonparametric gsynth bootstrap.
"""

from __future__ import annotations

import builtins
import importlib
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import statspai as sp  # noqa: E402
from statspai.exceptions import DataInsufficient, MethodIncompatibility  # noqa: E402

discos_mod = importlib.import_module("statspai.synth.discos")
sdid_mod = importlib.import_module("statspai.synth.sdid")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# --------------------------------------------------------------------- #
#  sp.gsynth(treat=...)
# --------------------------------------------------------------------- #


def _multi(seed=0, N=14, T=14, starts=None, noise=0.2, eff=2.0):
    """``starts`` maps unit index -> first treated period (1-based)."""
    starts = {0: 10, 1: 10, 2: 12} if starts is None else starts
    rng = np.random.default_rng(seed)
    f = rng.normal(size=T)
    lam = rng.normal(size=N)
    alpha = rng.normal(size=N)
    xi = rng.normal(size=T)
    rows = []
    for i in range(N):
        for t in range(1, T + 1):
            d = int(i in starts and t >= starts[i])
            y = alpha[i] + xi[t - 1] + lam[i] * f[t - 1] + eff * d
            rows.append({"unit": i, "time": t, "y": y + noise * rng.normal(), "d": d})
    return pd.DataFrame(rows)


def _gs(df, **kw):
    args = dict(outcome="y", unit="unit", time="time", treat="d", inference="none")
    args.update(kw)
    return sp.gsynth(df, **args)


def test_gsynth_multi_rejects_malformed_requests():
    df = _multi()
    with pytest.raises(MethodIncompatibility, match="column.s. not in data"):
        _gs(df, covariates=["nope"])
    with pytest.raises(MethodIncompatibility, match="unknown inference"):
        _gs(df, inference="jackknife")
    with pytest.raises(MethodIncompatibility, match="duplicate .unit, time. rows"):
        _gs(pd.concat([df, df.iloc[:1]]))
    with pytest.raises(MethodIncompatibility, match="0/1 indicator"):
        _gs(df.assign(d=df["d"] * 2))
    with pytest.raises(MethodIncompatibility, match="n_factors=9 needs at least"):
        _gs(df, n_factors=9)  # nine pre-periods allow at most seven


def test_gsynth_multi_reports_insufficient_data():
    with pytest.raises(DataInsufficient, match="treated in every period"):
        _gs(_multi(starts={0: 1, 1: 10}))
    with pytest.raises(DataInsufficient, match="no treated unit with enough"):
        _gs(_multi(starts={0: 3}))  # two pre-periods < min_T0 = 5
    few = _multi(N=4, starts={0: 10, 1: 10})
    with pytest.raises(DataInsufficient, match="2 never-treated unit"):
        _gs(few)


def test_gsynth_multi_drops_a_short_history_unit_and_matches_the_fit_without_it():
    df = _multi(starts={0: 10, 1: 10, 2: 3})
    with pytest.warns(RuntimeWarning, match="dropped 1 treated unit"):
        res = _gs(df, n_factors=1)
    ref = _gs(df[df.unit != 2], n_factors=1)
    # the dropped unit enters neither the control fit nor the ATT
    assert res.estimate == pytest.approx(ref.estimate, rel=1e-10)


def test_gsynth_nonparametric_bootstrap_warns_with_few_treated_units():
    df = _multi()
    point = _gs(df, n_factors=1)
    with pytest.warns(RuntimeWarning, match="resamples the 3 treated units"):
        res = _gs(df, n_factors=1, inference="nonparametric", n_boot=20, seed=0)
    # inference does not move the point estimate
    assert res.estimate == pytest.approx(point.estimate, rel=1e-12)
    assert np.isfinite(res.se) and res.se > 0
    # recovers the constant effect of 2: 0.5 is several bootstrap SEs
    # for this 3-treated-unit, noise-0.2 design
    assert res.estimate == pytest.approx(2.0, abs=0.5)


def test_gsynth_single_unit_interface_rejects_an_unknown_treated_unit():
    df = _multi()
    with pytest.raises(DataInsufficient, match="not found in column 'unit'"):
        sp.gsynth(
            df,
            outcome="y",
            unit="unit",
            time="time",
            treated_unit=99,
            treatment_time=10,
        )


# --------------------------------------------------------------------- #
#  sp.discos on individual-level data
# --------------------------------------------------------------------- #


def _micro(seed=0, J=3, T=4, n=30, shift=1.0):
    """Unit 0 treated in the last period; ``n`` individuals per cell."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(J + 1):
        for t in range(1, T + 1):
            y = rng.normal(loc=0.2 * t, size=n)
            if u == 0 and t == T:
                y = y + shift
            rows += [{"unit": u, "time": t, "y": v} for v in y]
    return pd.DataFrame(rows)


def _dc(df, **kw):
    args = dict(
        outcome="y",
        unit="unit",
        time="time",
        treated_unit=0,
        treatment_time=4,
        n_quantiles=20,
        M=50,
        placebo=False,
    )
    args.update(kw)
    return sp.discos(df, **args)


def test_discos_rejects_malformed_requests():
    df = _micro()
    with pytest.raises(MethodIncompatibility, match="column 'nope' not found"):
        _dc(df, outcome="nope")
    with pytest.raises(MethodIncompatibility, match="must be >= 2"):
        _dc(df, n_quantiles=1)
    with pytest.raises(MethodIncompatibility, match="M must be >= 1"):
        _dc(df, M=0)
    nan = df.copy()
    nan.loc[0, "y"] = np.nan
    with pytest.raises(MethodIncompatibility, match="contains missing values"):
        _dc(nan)
    with pytest.raises(MethodIncompatibility, match="treated unit 99 not found"):
        _dc(df, treated_unit=99)
    with pytest.raises(MethodIncompatibility, match="one node array per pre-period"):
        _dc(df, q_nodes=[np.linspace(0, 1, 5)] * 2)  # three pre-periods
    with pytest.raises(MethodIncompatibility, match=r"q_nodes must lie in \[0, 1\]"):
        _dc(df, q_nodes=np.array([0.1, 0.5, 1.5]))
    with pytest.raises(MethodIncompatibility, match="cdf_grid must hold one grid"):
        _dc(df, method="mixture", cdf_grid=[np.linspace(-3, 4, 10)] * 2)


def test_discos_reports_insufficient_data():
    df = _micro()
    with pytest.raises(DataInsufficient, match="at least 1 post-treatment"):
        _dc(df, treatment_time=99)
    with pytest.raises(DataInsufficient, match="at least 2 control"):
        _dc(df[df.unit <= 1])
    gone = df[~((df.unit == 0) & (df.time == 2))]
    with pytest.raises(DataInsufficient, match="treated unit has no observations"):
        _dc(gone)
    gone = df[~((df.unit == 2) & (df.time == 2))]
    with pytest.raises(DataInsufficient, match="donor 2 has no observations"):
        _dc(gone)


def test_discos_accepts_one_node_vector_for_every_pre_period():
    df = _micro()
    nodes = np.linspace(0.05, 0.95, 19)
    a = _dc(df, q_nodes=nodes)
    b = _dc(df, q_nodes=[nodes] * 3)
    # a single vector is used in each of the three pre-periods
    assert a.estimate == b.estimate
    w = np.array(list(a.model_info["weights"].values()), dtype=float)
    assert w.sum() == pytest.approx(1.0, abs=1e-8)
    # location shift of 1 in N(., 1) cells of 30 draws: sd of a difference
    # of cell means is about 0.26, so 0.8 is three of those
    assert a.estimate == pytest.approx(1.0, abs=0.8)


def test_discos_quantile_helpers_on_degenerate_samples():
    probs = np.array([0.1, 0.5, 0.9])
    # a single observation is every quantile of its sample
    np.testing.assert_array_equal(
        discos_mod._quant7_sorted(np.array([3.5]), probs), np.full(3, 3.5)
    )
    # type-7 quantiles of (0, 10) interpolate linearly
    np.testing.assert_allclose(
        discos_mod._quant7_sorted(np.array([0.0, 10.0]), probs), [1.0, 5.0, 9.0]
    )
    # fewer than two non-missing values: no quantile function
    out = discos_mod._empirical_quantile_function(np.array([1.0, np.nan]), probs)
    assert np.isnan(out).all() and out.shape == probs.shape


# --------------------------------------------------------------------- #
#  SDID
# --------------------------------------------------------------------- #


def _block(seed=0, N=8, T=10, T0=7, eff=2.0):
    rng = np.random.default_rng(seed)
    f = rng.normal(size=T).cumsum()
    lam = rng.uniform(0.5, 1.5, N)
    rows = []
    for i in range(N):
        for t in range(1, T + 1):
            d = int(i == 0 and t > T0)
            y = 5 + lam[i] * f[t - 1] + rng.normal(0, 0.2) + eff * d
            rows.append({"unit": i, "time": t, "y": y, "d": d})
    return pd.DataFrame(rows)


def test_sdid_treat_interface_runs_the_native_backend_only():
    with pytest.raises(MethodIncompatibility, match="native estimator only"):
        sp.sdid(_block(), outcome="y", unit="unit", time="time", treat="d", backend="r")


def test_sdid_time_placebo_guards():
    df = _block()
    kw = dict(y="y", unit="unit", time="time", treat_unit=0, kind="time")
    with pytest.raises(DataInsufficient, match="periods on both sides"):
        sp.synthdid_placebo(df, treat_time=99, **kw)
    with pytest.raises(DataInsufficient, match="pre-treatment periods out"):
        # one pre-period of ten: floor(1 * 1 / 10) = 0 placebo pre-periods
        sp.synthdid_placebo(df, treat_time=2, **kw)
    hole = df[~((df.unit == 3) & (df.time == 4))]
    with pytest.raises(DataInsufficient, match="balanced panel"):
        sp.synthdid_placebo(hole, treat_time=8, **kw)


def test_sdid_small_helpers():
    # one pre-period has no first differences: the noise scale defaults to 1
    assert sdid_mod._sdid_noise_level(np.array([[1.0], [2.0]])) == 1.0
    # sd of the pooled first differences (2, 2, 4, 4), ddof = 1
    two = np.array([[0.0, 2.0, 4.0], [0.0, 4.0, 8.0]])
    assert sdid_mod._sdid_noise_level(two) == pytest.approx(
        np.std([2.0, 2.0, 4.0, 4.0], ddof=1)
    )
    # nothing above a quarter of the maximum -> uniform weights
    np.testing.assert_array_equal(
        sdid_mod._sparsify_function(np.zeros(4)), np.full(4, 0.25)
    )
    # weights at or below max/4 are zeroed and the rest renormalised
    np.testing.assert_allclose(
        sdid_mod._sparsify_function(np.array([0.8, 0.2, 0.1])), [1.0, 0.0, 0.0]
    )


def test_sdid_cohort_helper_avoids_a_clashing_indicator_column_name():
    rng = np.random.default_rng(1)
    rows = []
    cohort = {0: 7, 1: 9}
    for i in range(10):
        for t in range(1, 11):
            g = cohort.get(i, 0)
            y = rng.normal() + 0.3 * i + 0.2 * t + 2.0 * (g > 0 and t >= g)
            rows.append({"unit": i, "time": t, "y": y, "g": g})
    df = pd.DataFrame(rows)
    kw = dict(y="y", unit="unit", time="time", cohort="g", se_method="noinference")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = sdid_mod._sdid_on_cohort_column(df, **kw)
        clash = sdid_mod._sdid_on_cohort_column(df.assign(__sdid_treated__=123), **kw)
    # the user's column of the same name is neither used nor overwritten
    assert clash.estimate == pytest.approx(plain.estimate, rel=1e-12)


@pytest.mark.parametrize(
    "plot", ["synthdid_plot", "synthdid_units_plot", "synthdid_rmse_plot"]
)
def test_sdid_plots_need_matplotlib(monkeypatch, plot):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.sdid(
            _block(),
            outcome="y",
            unit="unit",
            time="time",
            treated_unit=0,
            treatment_time=8,
            se_method="jackknife",
        )
    real_import = builtins.__import__

    def no_matplotlib(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("No module named 'matplotlib'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_matplotlib)
    with pytest.raises(ImportError, match="matplotlib required"):
        getattr(sp, plot)(res)
