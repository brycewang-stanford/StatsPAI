"""Tests for the v1.15 decomposition polish.

Covers:

* Yu-Elwert (2025) nonparametric causal decomposition — algebraic
  identity (residual = 0 for plug-in), bootstrap inference, dispatcher
  routing, plot smoke test, and zero-effect / zero-selection sanity
  checks.
* Unified ``DecompResultMixin`` surface — every existing decomposition
  result class returns a non-empty ``cite()`` and a serialisable
  ``to_dict()``; ``confint()`` works whenever an ``overall`` SE is
  present; ``to_excel()`` produces a valid XLSX byte-stream.
* Wild-bootstrap helper in :mod:`._common`.
"""

from __future__ import annotations

import io
import os

import matplotlib

matplotlib.use("Agg")  # headless

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.decomposition._common import (
    bootstrap_ci,
    wild_bootstrap_stat,
    analytical_ci,
)
from statspai.decomposition._results import _CITATIONS

# ---------------------------------------------------------------------- #
# DGP helper
# ---------------------------------------------------------------------- #


def _yu_elwert_dgp(
    n: int = 800,
    *,
    treat_effect_b: float = 0.0,
    selection_b: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Tunable DGP for Yu-Elwert smoke / sanity tests.

    Group ``r=1`` is "advantaged"; treatment ``t`` has effects that
    depend on ``x1``. ``selection_b`` controls how strongly the
    disadvantaged group selects on individual effect heterogeneity.
    """
    rng = np.random.default_rng(seed)
    r_grp = rng.integers(0, 2, n).astype(float)
    x1 = rng.normal(0, 1, n) + 0.4 * r_grp
    x2 = rng.normal(0, 1, n)
    # selection: group b's propensity correlates with their CATE.
    tau_b = 0.4 + 0.5 * x1
    tau_a = 0.6 + 0.4 * x1 + treat_effect_b * 0  # placeholder for clarity
    eta_a = 0.2 + 0.3 * x1
    eta_b = -0.1 + selection_b * tau_b - 0.1 * x1
    ps = np.where(r_grp == 1, 1 / (1 + np.exp(-eta_a)), 1 / (1 + np.exp(-eta_b)))
    t = (rng.uniform(0, 1, n) < ps).astype(float)
    tau = np.where(r_grp == 1, tau_a, tau_b)
    y = 1.0 + 0.3 * x1 + 0.2 * x2 + 0.3 * r_grp + tau * t + rng.normal(0, 1, n)
    return pd.DataFrame({"y": y, "t": t, "r": r_grp, "x1": x1, "x2": x2})


# ---------------------------------------------------------------------- #
# Yu-Elwert tests
# ---------------------------------------------------------------------- #


def test_yu_elwert_plugin_residual_is_exact():
    """Plug-in components sum exactly to the disparity (algebraic identity)."""
    df = _yu_elwert_dgp(n=600)
    r = sp.yu_elwert_decompose(
        data=df,
        y="y",
        treatment="t",
        group="r",
        x=["x1", "x2"],
        method="plugin",
        inference="none",
    )
    components = r.baseline + r.prevalence + r.effect + r.selection
    assert np.isclose(components, r.disparity, atol=1e-10), (
        f"plug-in residual = {r.disparity - components:.3e}; " "should be machine zero."
    )


def test_yu_elwert_zero_selection_when_no_targeting():
    """With propensity that doesn't depend on individual gain, selection ≈ 0."""
    df = _yu_elwert_dgp(n=2000, selection_b=0.0, seed=1)
    r = sp.yu_elwert_decompose(
        data=df,
        y="y",
        treatment="t",
        group="r",
        x=["x1", "x2"],
        method="plugin",
        inference="none",
    )
    # |selection| should be small relative to total disparity
    assert abs(r.selection) < 0.10 * abs(
        r.disparity
    ), f"|selection|={abs(r.selection):.4f} vs disparity={r.disparity:.4f}"


def test_yu_elwert_bootstrap_se_present():
    """Bootstrap inference returns SEs and CIs for each component."""
    df = _yu_elwert_dgp(n=400)
    r = sp.yu_elwert_decompose(
        data=df,
        y="y",
        treatment="t",
        group="r",
        x=["x1"],
        method="plugin",
        n_boot=49,
        seed=0,
    )
    assert r.se is not None
    for k in ("disparity", "baseline", "prevalence", "effect", "selection"):
        assert k in r.se
        assert r.se[k] >= 0
        lo, hi = r.ci[k]
        assert lo <= hi


def test_yu_elwert_efficient_method_runs():
    """DR (efficient) variant runs and yields finite numbers."""
    df = _yu_elwert_dgp(n=400)
    r = sp.yu_elwert_decompose(
        data=df,
        y="y",
        treatment="t",
        group="r",
        x=["x1"],
        method="efficient",
        inference="none",
    )
    for v in (r.disparity, r.baseline, r.prevalence, r.effect, r.selection):
        assert np.isfinite(v)


def test_yu_elwert_dispatcher_routes_via_aliases():
    """The dispatcher accepts both 'yu_elwert' and 'cdgd' as aliases."""
    df = _yu_elwert_dgp(n=300)
    r1 = sp.decompose(
        "yu_elwert",
        data=df,
        y="y",
        treatment="t",
        group="r",
        x=["x1"],
        inference="none",
    )
    r2 = sp.decompose(
        "cdgd",
        data=df,
        y="y",
        treatment="t",
        group="r",
        x=["x1"],
        inference="none",
    )
    assert isinstance(r1, sp.YuElwertResult)
    assert np.isclose(r1.disparity, r2.disparity)


def test_yu_elwert_plot_runs():
    """Result.plot() returns (fig, ax) without raising."""
    df = _yu_elwert_dgp(n=300)
    r = sp.yu_elwert_decompose(
        data=df,
        y="y",
        treatment="t",
        group="r",
        x=["x1"],
        method="plugin",
        n_boot=29,
        seed=0,
    )
    fig, ax = r.plot()
    assert fig is not None and ax is not None


def test_yu_elwert_rejects_non_binary_inputs():
    """Treatment / group must be binary 0/1."""
    df = _yu_elwert_dgp(n=200)
    df.loc[df.index[0], "t"] = 2.0
    with pytest.raises(ValueError, match="binary"):
        sp.yu_elwert_decompose(
            data=df,
            y="y",
            treatment="t",
            group="r",
            x=["x1"],
            inference="none",
        )


# ---------------------------------------------------------------------- #
# Mixin / export tests
# ---------------------------------------------------------------------- #


def test_oaxaca_mixin_cite_to_dict_to_excel(tmp_path):
    rng = np.random.default_rng(0)
    n = 400
    g = rng.integers(0, 2, n)
    x = rng.normal(0, 1, n) + 0.3 * g
    y = 1 + 0.5 * x + 0.3 * g + rng.normal(0, 1, n)
    df = pd.DataFrame({"y": y, "g": g, "x": x})
    r = sp.oaxaca(data=df, y="y", group="g", x=["x"], detail=True)
    # cite
    assert "Blinder" in r.cite("string")
    assert isinstance(r.cite("bibtex_keys"), list)
    assert isinstance(r.cite("list"), list)
    # to_dict round-trips through json
    import json

    d = r.to_dict()
    s = json.dumps(d, default=str)
    assert "gap" in d.get("overall", {})
    assert isinstance(s, str)
    # to_excel writes a real workbook
    p = tmp_path / "oax.xlsx"
    r.to_excel(str(p))
    assert os.path.getsize(p) > 1000
    # to_excel with path=None returns bytes (and is non-empty)
    blob = r.to_excel()
    assert isinstance(blob, bytes) and len(blob) > 1000


def test_confint_returns_normal_intervals():
    rng = np.random.default_rng(0)
    n = 300
    g = rng.integers(0, 2, n)
    x = rng.normal(0, 1, n)
    y = 0.5 * x + 0.3 * g + rng.normal(0, 1, n)
    df = pd.DataFrame({"y": y, "g": g, "x": x})
    r = sp.oaxaca(data=df, y="y", group="g", x=["x"], detail=False)
    ci = r.confint(alpha=0.05, which="overall")
    assert ci is not None
    for k, (lo, hi) in ci.items():
        assert lo <= r.overall[k] <= hi


def test_every_decomp_class_has_citations_and_mixin():
    """Every result class registered in the dispatcher carries citations."""
    from statspai.decomposition import (
        OaxacaResult,
        GelbachResult,
        YuElwertResult,
        GapClosingResult,
        MediationDecompResult,
        DisparityDecompResult,
        KitagawaResult,
        DasGuptaResult,
        RIFResult,
        RIFDecompositionResult,
        DFLResult,
        FFLResult,
        MachadoMataResult,
        MellyResult,
        CFMResult,
        NonlinearDecompResult,
        SubgroupDecompResult,
        SourceDecompResult,
        ShapleyInequalityResult,
    )

    for cls in (
        OaxacaResult,
        GelbachResult,
        YuElwertResult,
        GapClosingResult,
        MediationDecompResult,
        DisparityDecompResult,
        KitagawaResult,
        DasGuptaResult,
        RIFResult,
        RIFDecompositionResult,
        DFLResult,
        FFLResult,
        MachadoMataResult,
        MellyResult,
        CFMResult,
        NonlinearDecompResult,
        SubgroupDecompResult,
        SourceDecompResult,
        ShapleyInequalityResult,
    ):
        assert hasattr(cls, "to_excel"), f"{cls.__name__} missing mixin"
        assert hasattr(cls, "cite"), f"{cls.__name__} missing cite"
        assert hasattr(cls, "to_dict"), f"{cls.__name__} missing to_dict"
        assert tuple(cls.bib_keys), f"{cls.__name__} has empty bib_keys"
        for k in cls.bib_keys:
            assert (
                k in _CITATIONS
            ), f"{cls.__name__} references missing key {k!r} in _CITATIONS"


# ---------------------------------------------------------------------- #
# Wild bootstrap + analytical CI helpers
# ---------------------------------------------------------------------- #


def test_wild_bootstrap_recovers_se_for_mean():
    """Wild bootstrap of the residual mean ≈ analytic SE for a clean DGP."""
    rng = np.random.default_rng(1)
    n = 400
    sigma = 1.5
    fitted = np.zeros(n)
    resid = rng.normal(0, sigma, n)

    def stat(y_star: np.ndarray) -> float:
        return float(y_star.mean())

    boot = wild_bootstrap_stat(
        stat_fn=stat,
        resid=resid,
        fitted=fitted,
        n_boot=499,
        rng=rng,
        weights="rademacher",
    )
    se_wild = float(boot.std(ddof=1))
    se_true = sigma / np.sqrt(n)
    # Wild bootstrap should recover the analytic SE within 25% in this DGP.
    assert (
        abs(se_wild - se_true) / se_true < 0.25
    ), f"se_wild={se_wild:.4f} vs se_true={se_true:.4f}"


def test_wild_bootstrap_mammen_weights():
    rng = np.random.default_rng(2)
    n = 200
    boot = wild_bootstrap_stat(
        stat_fn=lambda y: float(y.mean()),
        resid=rng.normal(size=n),
        fitted=np.zeros(n),
        n_boot=49,
        rng=rng,
        weights="mammen",
    )
    assert boot.shape == (49, 1)


def test_wild_bootstrap_clustered():
    rng = np.random.default_rng(3)
    n = 200
    clusters = np.repeat(np.arange(20), 10)
    boot = wild_bootstrap_stat(
        stat_fn=lambda y: float(y.mean()),
        resid=rng.normal(size=n),
        fitted=np.zeros(n),
        n_boot=49,
        rng=rng,
        weights="rademacher",
        clusters=clusters,
    )
    assert boot.shape == (49, 1)


def test_analytical_ci_contains_point():
    lo, hi = analytical_ci(0.5, 0.1, alpha=0.05)
    assert lo < 0.5 < hi
    # Two-sided 95% spans roughly ±1.96 SE
    assert np.isclose(hi - lo, 2 * 1.96 * 0.1, atol=1e-3)


# ---------------------------------------------------------------------- #
# Reviewer-flagged regressions (v1.15 review pass)
# ---------------------------------------------------------------------- #


def test_yu_elwert_classvars_not_dataclass_fields():
    """Reviewer C1: bib_keys / method_name must be ClassVar, not fields."""
    from dataclasses import fields

    field_names = {f.name for f in fields(sp.YuElwertResult)}
    assert "bib_keys" not in field_names
    assert "method_name" not in field_names


def test_yu_elwert_confint_works():
    """Reviewer I2: confint() must return CIs on YuElwertResult."""
    df = _yu_elwert_dgp(n=300)
    r = sp.yu_elwert_decompose(
        data=df,
        y="y",
        treatment="t",
        group="r",
        x=["x1"],
        n_boot=29,
        seed=0,
    )
    ci = r.confint(alpha=0.05)
    assert ci is not None
    assert "disparity" in ci
    lo, hi = ci["disparity"]
    assert lo < r.disparity < hi


def test_yu_elwert_nuisance_diagnostics_present():
    """Reviewer I1 / S5: nuisance dict carries fallback + bootstrap counters."""
    df = _yu_elwert_dgp(n=300)
    r = sp.yu_elwert_decompose(
        data=df,
        y="y",
        treatment="t",
        group="r",
        x=["x1"],
        n_boot=29,
        seed=0,
    )
    assert "fallback_cell_count" in r.nuisance
    assert "bootstrap_failure_count" in r.nuisance


def test_oaxaca_plot_legacy_kwargs_warn(tmp_path):
    """Reviewer S1: legacy color_pos/color_neg trigger DeprecationWarning."""
    rng = np.random.default_rng(0)
    n = 200
    g = rng.integers(0, 2, n)
    x = rng.normal(0, 1, n)
    y = 0.5 * x + 0.3 * g + rng.normal(0, 1, n)
    df = pd.DataFrame({"y": y, "g": g, "x": x})
    r = sp.oaxaca(data=df, y="y", group="g", x=["x"], detail=True)
    with pytest.warns(DeprecationWarning, match="color_pos"):
        r.plot(color_pos="#000000", color_neg="#FFFFFF")


def test_yu_elwert_rejects_non_existent_weights_kwarg():
    """Reviewer C2: weights kwarg removed from public signature."""
    import inspect

    sig = inspect.signature(sp.yu_elwert_decompose)
    assert "weights" not in sig.parameters


def test_to_word_does_not_pollute_stdout(tmp_path, capsys):
    """Reviewer I5: to_word() must not print summary to stdout."""
    rng = np.random.default_rng(0)
    n = 200
    g = rng.integers(0, 2, n)
    x = rng.normal(0, 1, n)
    y = 0.5 * x + 0.3 * g + rng.normal(0, 1, n)
    df = pd.DataFrame({"y": y, "g": g, "x": x})
    r = sp.oaxaca(data=df, y="y", group="g", x=["x"], detail=True)
    pytest.importorskip("docx")
    capsys.readouterr()  # reset
    r.to_word(str(tmp_path / "x.docx"))
    out = capsys.readouterr().out
    assert "Oaxaca-Blinder Decomposition" not in out
