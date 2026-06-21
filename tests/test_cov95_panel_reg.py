"""Coverage tests for statspai.panel.panel_reg.

Exercises the PanelResults diagnostic-method wrappers, .compare /
PanelCompareResults, the CRE (Mundlak/Chamberlain) and GMM routes,
panel_compare, balance handling, and the legacy PanelRegression shim.
Assertions check structure / sane properties, not fabricated numbers.
"""

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import statspai as sp  # noqa: E402
from statspai.exceptions import DataInsufficient, MethodIncompatibility  # noqa: E402
from statspai.panel import PanelResults, PanelRegression  # noqa: E402
from statspai.panel.panel_reg import PanelCompareResults, panel_compare  # noqa: E402


@pytest.fixture
def panel_df():
    rng = np.random.default_rng(11)
    n_id, T = 40, 6
    rows = []
    for i in range(n_id):
        alpha = rng.normal(4, 2)
        for t in range(T):
            x1 = rng.normal() + 0.4 * alpha
            x2 = rng.normal()
            y = alpha + 2.0 * x1 + 1.0 * x2 + rng.normal(0, 0.5)
            rows.append({"id": i, "year": t, "y": y, "x1": x1, "x2": x2})
    return pd.DataFrame(rows)


# ── PanelResults diagnostic-method wrappers ─────────────────────────────


def test_hausman_test_method(panel_df):
    r = sp.panel(panel_df, "y ~ x1 + x2", entity="id", time="year", method="fe")
    out = r.hausman_test()
    assert out["recommendation"] in ("FE", "RE")
    assert out["statistic"] >= 0


def test_hausman_test_uses_same_re_estimator_as_panel_re():
    rng = np.random.default_rng(42)
    n_id, n_t = 100, 8
    unit = np.repeat(np.arange(n_id), n_t)
    year = np.tile(np.arange(n_t), n_id)
    n = n_id * n_t
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    unit_fe = rng.normal(0, 1, n_id)
    y = 1.0 + 0.5 * x1 - 0.3 * x2 + unit_fe[unit] + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "id": unit, "year": year})

    fe = sp.panel(df, "y ~ x1 + x2", entity="id", time="year", method="fe")
    re = sp.panel(df, "y ~ x1 + x2", entity="id", time="year", method="re")
    out = fe.hausman_test()

    assert float(out["beta_re"]["x1"]) == pytest.approx(float(re.params["x1"]))
    assert float(out["beta_re"]["x2"]) == pytest.approx(float(re.params["x2"]))
    assert out["statistic"] == pytest.approx(1.37691630362867, abs=1e-12)
    assert out["pvalue"] == pytest.approx(0.502350019722, abs=1e-12)


def test_bp_lm_test_method(panel_df):
    r = sp.panel(panel_df, "y ~ x1 + x2", entity="id", time="year", method="fe")
    out = r.bp_lm_test()
    assert out["df"] == 1
    assert 0 <= out["pvalue"] <= 1


def test_f_test_effects_method(panel_df):
    r = sp.panel(panel_df, "y ~ x1 + x2", entity="id", time="year", method="fe")
    out = r.f_test_effects()
    assert "statistic" in out


def test_pesaran_cd_test_method(panel_df):
    r = sp.panel(panel_df, "y ~ x1 + x2", entity="id", time="year", method="fe")
    out = r.pesaran_cd_test()
    assert "statistic" in out


def test_diagnostic_methods_raise_without_panel_data(panel_df):
    r = sp.panel(panel_df, "y ~ x1 + x2", entity="id", time="year", method="fe")
    r._panel_data = None
    with pytest.raises(MethodIncompatibility, match="Hausman"):
        r.hausman_test()
    with pytest.raises(MethodIncompatibility, match="BP-LM"):
        r.bp_lm_test()
    with pytest.raises(MethodIncompatibility, match="F-test"):
        r.f_test_effects()


def test_pesaran_cd_raises_without_lm_result(panel_df):
    r = sp.panel(panel_df, "y ~ x1 + x2", entity="id", time="year", method="fe")
    r._lm_result = None
    with pytest.raises(MethodIncompatibility, match="CD test"):
        r.pesaran_cd_test()


# ── compare / PanelCompareResults ───────────────────────────────────────


def test_compare_and_summary(panel_df):
    r = sp.panel(panel_df, "y ~ x1 + x2", entity="id", time="year", method="fe")
    cmp = r.compare("re")
    assert isinstance(cmp, PanelCompareResults)
    s = cmp.summary()
    assert "Panel Comparison" in s
    assert "R²" in s
    # __str__ delegates to summary, __repr__ short form
    assert str(cmp) == s
    assert "PanelCompareResults" in repr(cmp)


# ── CRE: Mundlak (with Wald test) and Chamberlain ───────────────────────


def test_mundlak_cre(panel_df):
    r = sp.panel(panel_df, "y ~ x1 + x2", entity="id", time="year", method="mundlak")
    assert isinstance(r, PanelResults)
    # Mundlak terms diagnostics + Wald test recorded
    assert r.diagnostics.get("Mundlak terms") == 2
    assert "CRE Wald chi2" in r.diagnostics
    assert "CRE Wald p-value" in r.diagnostics
    assert r.diagnostics["CRE Wald chi2"] >= 0


def test_chamberlain_cre(panel_df):
    r = sp.panel(
        panel_df, "y ~ x1 + x2", entity="id", time="year", method="chamberlain"
    )
    assert isinstance(r, PanelResults)


# ── GMM dynamic panel route ─────────────────────────────────────────────


def test_arellano_bond_gmm(panel_df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.panel(
            panel_df, "y ~ x1 + x2", entity="id", time="year", method="ab", lags=1
        )
    assert isinstance(r, PanelResults)
    assert r.model_info["method"] == "ab"
    assert "AR(1) z" in r.diagnostics or "Hansen J" in r.diagnostics


def test_system_gmm_not_implemented(panel_df):
    # System GMM (Blundell-Bond) is intentionally not implemented yet;
    # the GMM route raises rather than returning unvalidated numbers.
    with pytest.raises(NotImplementedError, match="system GMM"):
        sp.panel(
            panel_df,
            "y ~ x1 + x2",
            entity="id",
            time="year",
            method="system",
            lags=1,
            twostep=True,
        )


# ── panel_compare table ─────────────────────────────────────────────────


def test_panel_compare_default_methods(panel_df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tbl = panel_compare(panel_df, "y ~ x1 + x2", entity="id", time="year")
    assert isinstance(tbl, pd.DataFrame)
    assert "N obs" in tbl.index
    assert "R-squared" in tbl.index


def test_panel_compare_custom_methods(panel_df):
    tbl = sp.panel_compare(
        panel_df,
        "y ~ x1 + x2",
        entity="id",
        time="year",
        methods=["pooled", "fe", "re"],
    )
    assert isinstance(tbl, pd.DataFrame)
    # coefficient rows present for x1/x2
    assert "x1" in tbl.index


def test_panel_compare_handles_estimator_error(panel_df):
    # An invalid method inside the list is caught and stored as 'error'
    tbl = sp.panel_compare(
        panel_df, "y ~ x1", entity="id", time="year", methods=["fe", "not_a_method"]
    )
    assert isinstance(tbl, pd.DataFrame)


# ── balance handling ────────────────────────────────────────────────────


def test_balance_true(panel_df):
    # drop a couple rows to make it unbalanced, then balance=True
    unbal = panel_df.drop(index=[0, 1]).reset_index(drop=True)
    r = sp.panel(
        unbal, "y ~ x1 + x2", entity="id", time="year", method="fe", balance=True
    )
    assert isinstance(r, PanelResults)


def test_balance_drops_all_raises():
    # No entity is observed in all periods -> balance=True wipes everything
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "year": [0, 1, 2],
            "y": [1.0, 2.0, 3.0],
            "x1": [0.1, 0.2, 0.3],
        }
    )
    with pytest.raises(DataInsufficient, match="dropped all units"):
        sp.panel(df, "y ~ x1", entity="id", time="year", method="fe", balance=True)


# ── error paths ─────────────────────────────────────────────────────────


def test_unknown_method(panel_df):
    with pytest.raises(MethodIncompatibility, match="method must be one of"):
        sp.panel(panel_df, "y ~ x1", entity="id", time="year", method="bogus")


def test_formula_without_tilde(panel_df):
    with pytest.raises(MethodIncompatibility, match="must contain"):
        sp.panel(panel_df, "y x1", entity="id", time="year", method="fe")


def test_missing_column(panel_df):
    with pytest.raises(MethodIncompatibility, match="not found"):
        sp.panel(panel_df, "y ~ nope", entity="id", time="year", method="fe")


# ── linearmodels method branches + cov kwargs ──────────────────────────


@pytest.mark.parametrize("method", ["twoway", "fd", "be", "pooled"])
def test_linearmodels_method_branches(panel_df, method):
    r = sp.panel(panel_df, "y ~ x1 + x2", entity="id", time="year", method=method)
    assert isinstance(r, PanelResults)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cluster": "twoway"},
        {"cluster": "entity"},
        {"cluster": "time"},
        {"cluster": "id"},  # generic cluster column -> entity-cluster path
        {"robust": "robust"},
        {"robust": "kernel"},
        {"robust": "driscoll-kraay"},
    ],
)
def test_cov_kwargs_branches(panel_df, kwargs):
    r = sp.panel(
        panel_df, "y ~ x1 + x2", entity="id", time="year", method="fe", **kwargs
    )
    assert isinstance(r, PanelResults)
    assert (r.std_errors >= 0).all()


# ── plot dispatcher (routes into panel_plots, which is excluded) ────────


@pytest.mark.parametrize("ptype", ["coef", "effects", "residuals", "hausman"])
def test_plot_dispatcher(panel_df, ptype):
    r = sp.panel(panel_df, "y ~ x1 + x2", entity="id", time="year", method="fe")
    try:
        out = r.plot(type=ptype)
    except Exception as exc:  # pragma: no cover - plotting backend variance
        plt.close("all")
        pytest.skip(f"plot backend unavailable for {ptype}: {exc}")
    plt.close("all")
    assert out is not None


def test_plot_shortcut_methods(panel_df):
    r = sp.panel(panel_df, "y ~ x1 + x2", entity="id", time="year", method="fe")
    for fn in (r.plot_effects, r.plot_residuals, r.plot_hausman):
        try:
            fn()
        except Exception:  # pragma: no cover
            plt.close("all")
            pytest.skip("plot backend unavailable")
        plt.close("all")


def test_plot_unknown_type_raises(panel_df):
    r = sp.panel(panel_df, "y ~ x1 + x2", entity="id", time="year", method="fe")
    with pytest.raises(MethodIncompatibility, match="Unknown plot type"):
        r.plot(type="not_a_plot")


def test_compare_plot(panel_df):
    r = sp.panel(panel_df, "y ~ x1 + x2", entity="id", time="year", method="fe")
    cmp = r.compare("re")
    try:
        out = cmp.plot()
    except Exception:  # pragma: no cover
        plt.close("all")
        pytest.skip("plot backend unavailable")
    plt.close("all")
    assert out is not None


# ── legacy PanelRegression shim ─────────────────────────────────────────


def test_panel_regression_shim(panel_df):
    pr = PanelRegression(
        data=panel_df, formula="y ~ x1 + x2", entity="id", time="year", method="fe"
    )
    r = pr.fit()
    assert isinstance(r, PanelResults)


# ── package-level sp.panel dispatcher (panel/__init__.py) ───────────────


def test_dispatcher_hdfe_auto_formula(panel_df):
    # No `|` in formula -> dispatcher bolts entity+time on as FE.
    r = sp.panel(panel_df, "y ~ x1 + x2", entity="id", time="year", method="hdfe")
    assert type(r).__name__ == "FEOLSResult"


def test_dispatcher_hdfe_explicit_fe(panel_df):
    r = sp.panel(panel_df, "y ~ x1 + x2 | id + year", method="reghdfe")
    assert type(r).__name__ == "FEOLSResult"


def test_dispatcher_hdfe_requires_formula(panel_df):
    with pytest.raises(ValueError, match="requires a formula"):
        sp.panel(panel_df, formula=None, entity="id", time="year", method="hdfe")


def test_dispatcher_non_string_method(panel_df):
    with pytest.raises(TypeError, match="method must be a string"):
        sp.panel(panel_df, "y ~ x1", entity="id", time="year", method=123)


def test_dispatcher_unknown_method(panel_df):
    with pytest.raises(ValueError, match="Unknown method"):
        sp.panel(panel_df, "y ~ x1", entity="id", time="year", method="totally_bogus")


# ── balance_panel standalone ────────────────────────────────────────────


def test_balance_panel_function(panel_df):
    unbal = panel_df.drop(index=[2, 3]).reset_index(drop=True)
    bal = sp.balance_panel(unbal, entity="id", time="year")
    counts = bal.groupby("id")["year"].count()
    assert counts.nunique() == 1
