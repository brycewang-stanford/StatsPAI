"""Additional coverage: datasets, discos plot/helpers, sensitivity/power plots,
SDID native SE branches, report sensitivity sections, exports helpers."""

import importlib
import os
import tempfile

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import statspai as sp  # noqa: E402

datasets = importlib.import_module("statspai.synth.datasets")
discos_mod = importlib.import_module("statspai.synth.discos")
sensitivity = importlib.import_module("statspai.synth.sensitivity")
power_mod = importlib.import_module("statspai.synth.power")
report = importlib.import_module("statspai.synth.report")
exports = importlib.import_module("statspai.synth.exports")
sdid_mod = importlib.import_module("statspai.synth.sdid")


def _panel(n_units=9, n_periods=18, treatment_time=12, effect=4.0, seed=61):
    rng = np.random.default_rng(seed)
    alphas = rng.normal(10, 2, n_units)
    betas = rng.normal(0.5, 0.1, n_units)
    common = rng.normal(0, 0.4, n_periods)
    records = []
    for i in range(n_units):
        for ti, t in enumerate(range(1, n_periods + 1)):
            y = alphas[i] + betas[i] * t + common[ti] + rng.normal(0, 0.2)
            if i == 0 and t >= treatment_time:
                y += effect
            records.append({"unit": f"u{i}", "time": t, "outcome": y})
    return pd.DataFrame(records)


COMMON = dict(
    outcome="outcome", unit="unit", time="time", treated_unit="u0", treatment_time=12
)


@pytest.fixture
def panel():
    return _panel()


# ---------------- datasets ----------------
@pytest.mark.parametrize(
    "loader", ["german_reunification", "basque_terrorism", "california_tobacco"]
)
def test_dataset_loaders(loader):
    df = getattr(datasets, loader)()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert df.shape[1] >= 3


def test_reference_dataset_moments():
    german = datasets.german_reunification()
    california = datasets.california_tobacco()
    assert german.shape == (748, 4)
    assert california.shape == (1209, 8)
    np.testing.assert_allclose(
        [german["year"].sum(), german["gdppc"].sum(), german["treated"].sum()],
        [1482162.0, 15348124.3, 14.0],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        [
            california["year"].sum(),
            california["cigsale"].sum(),
            california["retprice"].sum(),
        ],
        [2399865.0, 120916.8, 99589.1],
        atol=1e-6,
    )


# ---------------- discos plot + helpers ----------------
def _dist_panel(n_units=8, n_periods=14, treatment_time=9, seed=71):
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n_units):
        mu = rng.normal(5, 1)
        sd = rng.uniform(0.8, 1.5)
        for t in range(1, n_periods + 1):
            for _ in range(30):
                y = rng.normal(mu + 0.1 * t, sd)
                if i == 0 and t >= treatment_time:
                    y += 2.0
                records.append({"unit": f"u{i}", "time": t, "outcome": y})
    return pd.DataFrame(records)


@pytest.mark.parametrize(
    "ptype", ["quantile_effect", "quantile_comparison", "gap", "weights"]
)
def test_discos_plot_types(ptype):
    df = _dist_panel()
    res = discos_mod.discos(
        df,
        outcome="outcome",
        unit="unit",
        time="time",
        treated_unit="u0",
        treatment_time=9,
        method="mixture",
        n_quantiles=40,
        placebo=True,
        seed=0,
    )
    out = discos_mod.discos_plot(res, type=ptype)
    assert out is not None


def test_stochastic_dominance_second_order():
    df = _dist_panel()
    res = discos_mod.discos(
        df,
        outcome="outcome",
        unit="unit",
        time="time",
        treated_unit="u0",
        treatment_time=9,
        method="quantile",
        n_quantiles=40,
        placebo=False,
        seed=1,
    )
    out2 = discos_mod.stochastic_dominance(res, order=2)
    assert isinstance(out2, dict) and len(out2) > 0
    out1 = discos_mod.stochastic_dominance(res, order=1)
    assert isinstance(out1, dict)


# ---------------- sensitivity plot ----------------
def test_synth_sensitivity_plot(panel):
    sens = sensitivity.synth_sensitivity(panel, **COMMON, n_donor_samples=20, seed=0)
    fig = sensitivity.synth_sensitivity_plot(sens, title="Sens")
    assert fig is not None


# ---------------- power plot ----------------
def test_synth_power_plot(panel):
    df = power_mod.synth_power(
        panel, **COMMON, effect_sizes=[0.0, 2.0, 5.0], n_simulations=20, seed=0
    )
    out = power_mod.synth_power_plot(df)
    assert out is not None
    np.testing.assert_allclose(
        df[["effect_size", "power", "n_rejections", "n_simulations"]].to_numpy(),
        np.array(
            [[0.0, 1.0, 20.0, 20.0], [2.0, 1.0, 20.0, 20.0], [5.0, 1.0, 20.0, 20.0]]
        ),
        atol=1e-12,
    )


# ---------------- report sensitivity sections (markdown + latex) ----------------
def test_synth_report_markdown_with_sensitivity(panel):
    rep = report.synth_report(
        panel, **COMMON, method="classic", output="markdown", sensitivity=True
    )
    assert isinstance(rep, str)
    assert "Sensitivity" in rep or "sensitivity" in rep


def test_synth_report_latex_with_sensitivity(panel):
    rep = report.synth_report(
        panel, **COMMON, method="classic", output="latex", sensitivity=True
    )
    assert isinstance(rep, str) and len(rep) > 0


# ---------------- exports helpers via different model_info shapes ----------------
def test_exports_with_sdid_result(panel):
    res = sdid_mod.sdid(
        panel, **COMMON, method="sdid", n_reps=15, backend="native", seed=0
    )
    tex = exports.synth_to_latex(res, show_weights=True)
    assert "tabular" in tex
    md = exports.synth_to_markdown(res, show_weights=True)
    assert "|" in md


def test_exports_mixed_list(panel):
    r1 = sp.synth(panel, **COMMON, method="classic", placebo=False)
    r2 = sdid_mod.sdid(
        panel, **COMMON, method="sdid", n_reps=15, backend="native", seed=1
    )
    tex = exports.synth_to_latex([r1, r2], show_ci=True, show_weights=True)
    assert "tabular" in tex
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "multi.xlsx")
        out = exports.synth_to_excel([r1, r2], path)
        assert os.path.exists(out)


# ---------------- sdid native SE branches ----------------
def test_sdid_bootstrap_jackknife_native(panel):
    for se in ("bootstrap", "jackknife"):
        res = sdid_mod.sdid(
            panel,
            **COMMON,
            method="sdid",
            se_method=se,
            n_reps=15,
            backend="native",
            seed=0,
        )
        assert np.isfinite(res.estimate)


def test_sdid_sc_did_with_se(panel):
    for method in ("sc", "did"):
        res = sdid_mod.sdid(
            panel,
            **COMMON,
            method=method,
            se_method="placebo",
            n_reps=15,
            backend="native",
            seed=2,
        )
        assert np.isfinite(res.estimate)


def test_sdid_summary(panel):
    res = sdid_mod.sdid(
        panel, **COMMON, method="sdid", n_reps=15, backend="native", seed=3
    )
    assert isinstance(res.summary(), str)
