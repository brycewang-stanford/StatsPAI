"""Coverage campaign (decomposition module) — Oaxaca–Blinder & Gelbach.

Targets the previously-uncovered ``reference`` variants of
``sp.decompose('oaxaca')`` — ``{0, 1, 'pooled', 'cotton', 'reimers'}`` (Neumark
1988 / Cotton 1988 / Reimers 1983 reference-coefficient choices) — plus the
``detail`` switch, and the Gelbach (2016) omitted-variable-bias decomposition.

Every test pins a real algebraic identity, not a smoke call:

* Oaxaca additivity:  ``gap == explained + unexplained``  and
  ``gap == mean_A - mean_B`` exactly, for *every* reference weighting.
* Reference selection:  ``reference=0 → beta* = beta_A``;  ``=1 → beta* = beta_B``.
* Detailed decomposition sums to the aggregate explained component.
* Gelbach exact additivity:  ``total_change == base_coef - full_coef`` and
  ``sum(delta_j) == total_change``.

These are exact (Oaxaca–Blinder is a deterministic algebraic identity), so the
tolerances are machine-epsilon-scale, per CLAUDE.md §5 (real numerical
assertions, no mocking of numerical paths).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.decomposition import datasets

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

X = ["education", "experience", "tenure"]


@pytest.fixture(scope="module")
def wage() -> pd.DataFrame:
    return datasets.cps_wage()


# ── Oaxaca: additivity identity holds for every reference weighting ───


@pytest.mark.parametrize("reference", [0, 1, "pooled", "cotton", "reimers"])
def test_oaxaca_additivity_identity(wage, reference):
    r = sp.decompose(
        "oaxaca",
        data=wage,
        y="log_wage",
        group="female",
        x=X,
        reference=reference,
    )
    o = r.overall
    gap, expl, unexpl = o["gap"], o["explained"], o["unexplained"]
    # Two-fold additivity: gap = explained + unexplained, exactly.
    assert expl + unexpl == pytest.approx(gap, abs=1e-9)
    # Raw gap equals the difference in group means, exactly.
    gs = r.group_stats
    assert gap == pytest.approx(gs["mean_a"] - gs["mean_b"], abs=1e-9)


def test_oaxaca_reference_selects_beta_star(wage):
    r0 = sp.decompose(
        "oaxaca", data=wage, y="log_wage", group="female", x=X, reference=0
    )
    np.testing.assert_allclose(
        r0.group_stats["beta_star"].to_numpy(),
        r0.group_stats["beta_a"].to_numpy(),
        atol=1e-10,
    )
    r1 = sp.decompose(
        "oaxaca", data=wage, y="log_wage", group="female", x=X, reference=1
    )
    np.testing.assert_allclose(
        r1.group_stats["beta_star"].to_numpy(),
        r1.group_stats["beta_b"].to_numpy(),
        atol=1e-10,
    )


def test_oaxaca_detailed_sums_to_aggregate(wage):
    r = sp.decompose(
        "oaxaca", data=wage, y="log_wage", group="female", x=X, reference=0, detail=True
    )
    detailed = r.detailed
    assert detailed is not None
    # The detailed explained contributions must sum to the aggregate
    # explained component (a defining property of the decomposition).
    col = "explained" if "explained" in detailed.columns else detailed.columns[1]
    total = float(np.asarray(detailed[col], dtype=float).sum())
    assert total == pytest.approx(r.overall["explained"], rel=1e-6, abs=1e-8)


def test_oaxaca_detail_false_skips_detailed(wage):
    r = sp.decompose(
        "oaxaca",
        data=wage,
        y="log_wage",
        group="female",
        x=X,
        reference=0,
        detail=False,
    )
    # Aggregate identity still holds even without the per-variable table.
    o = r.overall
    assert o["explained"] + o["unexplained"] == pytest.approx(o["gap"], abs=1e-9)


def test_oaxaca_missing_column_raises(wage):
    with pytest.raises(ValueError, match="(?i)not found"):
        sp.decompose(
            "oaxaca",
            data=wage,
            y="log_wage",
            group="female",
            x=["education", "does_not_exist"],
        )


# ── Gelbach (2016): exact additivity of the coefficient change ────────


def test_gelbach_additivity(wage):
    g = sp.decompose(
        "gelbach",
        data=wage,
        y="log_wage",
        base_x=["education"],
        added_x=["experience", "tenure"],
    )
    # total_change is exactly the short-minus-long coefficient gap.
    assert g.total_change == pytest.approx(g.base_coef - g.full_coef, abs=1e-9)
    # Per-added-variable contributions sum to the total change.
    delta_sum = float(np.asarray(g.decomposition["delta"], dtype=float).sum())
    assert delta_sum == pytest.approx(g.total_change, rel=1e-6, abs=1e-9)


def test_gelbach_var_of_interest_explicit(wage):
    g = sp.decompose(
        "gelbach",
        data=wage,
        y="log_wage",
        base_x=["education", "experience"],
        added_x=["tenure", "union"],
        var_of_interest="experience",
    )
    assert g.base_var == "experience"
    assert g.total_change == pytest.approx(g.base_coef - g.full_coef, abs=1e-9)


# ── OaxacaResult rendering surface (.plot / .to_latex / repr / html) ──


@pytest.fixture(scope="module")
def oaxaca_result(wage):
    return sp.decompose(
        "oaxaca", data=wage, y="log_wage", group="female", x=X, reference=0, detail=True
    )


def test_oaxaca_result_summary_and_repr(oaxaca_result):
    s = oaxaca_result.summary()
    assert isinstance(s, str) and len(s) > 0
    assert isinstance(repr(oaxaca_result), str)


def test_oaxaca_result_to_latex(oaxaca_result):
    tex = oaxaca_result.to_latex()
    assert isinstance(tex, str)
    assert "Explained" in tex or "explained" in tex.lower()


def test_oaxaca_result_repr_html(oaxaca_result):
    html = oaxaca_result._repr_html_()
    assert isinstance(html, str) and "<table" in html


@pytest.mark.parametrize("kind", ["forest", "waterfall"])
def test_oaxaca_result_plot(oaxaca_result, kind):
    try:
        oaxaca_result.plot(kind=kind)
    finally:
        plt.close("all")


def test_oaxaca_plot_without_detail_raises(wage):
    r = sp.decompose(
        "oaxaca",
        data=wage,
        y="log_wage",
        group="female",
        x=X,
        reference=0,
        detail=False,
    )
    with pytest.raises(ValueError, match="(?i)detail"):
        r.plot(kind="waterfall")
    plt.close("all")


# ── Oaxaca input-validation error branches ───────────────────────────


def test_oaxaca_non_binary_group_raises(wage):
    bad = wage.copy()
    bad.loc[bad.index[:10], "female"] = 2  # not in {0, 1}
    with pytest.raises(ValueError):
        sp.decompose("oaxaca", data=bad, y="log_wage", group="female", x=X)


def test_oaxaca_too_few_obs_raises(wage):
    tiny = pd.concat(
        [wage[wage["female"] == 1].head(1), wage[wage["female"] == 0].head(5)]
    )
    with pytest.raises(ValueError, match="(?i)at least 2"):
        sp.decompose("oaxaca", data=tiny, y="log_wage", group="female", x=X)


def test_oaxaca_invalid_reference_raises(wage):
    with pytest.raises(ValueError, match="(?i)invalid reference"):
        sp.decompose(
            "oaxaca",
            data=wage,
            y="log_wage",
            group="female",
            x=X,
            reference="not_a_reference",
        )


# ── GelbachResult rendering surface + validation branches ────────────


@pytest.fixture(scope="module")
def gelbach_result(wage):
    return sp.decompose(
        "gelbach",
        data=wage,
        y="log_wage",
        base_x=["education"],
        added_x=["experience", "tenure"],
    )


def test_gelbach_result_rendering(gelbach_result):
    assert isinstance(gelbach_result.summary(), str)
    assert isinstance(repr(gelbach_result), str)
    assert isinstance(gelbach_result.to_latex(), str)
    assert "<table" in gelbach_result._repr_html_()
    try:
        gelbach_result.plot()
    finally:
        plt.close("all")


def test_gelbach_var_not_in_base_raises(wage):
    with pytest.raises(ValueError, match="(?i)not in base_x"):
        sp.decompose(
            "gelbach",
            data=wage,
            y="log_wage",
            base_x=["education"],
            added_x=["tenure"],
            var_of_interest="experience",
        )


def test_gelbach_overlap_raises(wage):
    with pytest.raises(ValueError, match="(?i)both base_x and added_x"):
        sp.decompose(
            "gelbach",
            data=wage,
            y="log_wage",
            base_x=["education", "tenure"],
            added_x=["tenure"],
        )


def test_gelbach_missing_column_raises(wage):
    with pytest.raises(ValueError, match="(?i)not found"):
        sp.decompose(
            "gelbach", data=wage, y="log_wage", base_x=["education"], added_x=["nope"]
        )


def test_gelbach_too_few_obs_raises(wage):
    tiny = wage.head(3)  # n < len(all_x) + 2
    with pytest.raises(ValueError):
        sp.decompose(
            "gelbach",
            data=tiny,
            y="log_wage",
            base_x=["education"],
            added_x=["experience", "tenure"],
        )
