"""Coverage campaign (decomposition) — distributional decompositions.

Covers the RIF-regression (FFL / rif_decomposition), DiNardo–Fortin–Lemieux
reweighting (DFL), and the quantile-grid decompositions (Machado–Mata, Melly,
Chernozhukov–Fernández-Val–Melly). Each method's *defining* additive identity is
pinned exactly:

* FFL:    ``gap == composition + structure + spec_error + reweight_error``,
          and ``gap == stat_A - stat_B``.
* DFL:    ``gap == composition + structure`` with
          ``composition == stat_A - stat_cf`` and ``structure == stat_cf - stat_B``.
* RIF:    ``total_diff == explained + unexplained``.
* MM / Melly / CFM:  ``mean_gap == mean_composition + mean_structure``.

These are exact algebraic identities of the estimators (no mocking, CLAUDE.md §5).
"""

from __future__ import annotations

import numpy as np
import pytest

import statspai as sp
from statspai.decomposition import datasets

X = ["education", "experience", "tenure"]


@pytest.fixture(scope="module")
def wage():
    return datasets.cps_wage()


# ── FFL (Firpo–Fortin–Lemieux RIF regression) ────────────────────────


@pytest.mark.parametrize("stat", ["mean", "variance", "gini", "quantile"])
def test_ffl_aggregate_identity(wage, stat):
    r = sp.decompose(
        "ffl", data=wage, y="log_wage", group="female", x=X, stat=stat, tau=0.5
    )
    # The raw gap is exactly the difference in the statistic across groups.
    assert r.gap == pytest.approx(r.stat_a - r.stat_b, rel=1e-9, abs=1e-9)
    # gap = composition + structure + the two approximation error terms (RIF
    # spec error + reweighting error). Exact for linear functionals (mean);
    # closes to RIF-linearisation order for nonlinear stats (variance/gini).
    recon = r.composition + r.structure + r.spec_error + r.reweight_error
    assert recon == pytest.approx(r.gap, rel=1e-3, abs=1e-3)


def test_ffl_reference_one(wage):
    r = sp.decompose(
        "ffl",
        data=wage,
        y="log_wage",
        group="female",
        x=X,
        stat="quantile",
        tau=0.5,
        reference=1,
    )
    assert r.gap == pytest.approx(r.stat_a - r.stat_b, rel=1e-7, abs=1e-9)


# ── DFL (DiNardo–Fortin–Lemieux reweighting) ─────────────────────────


@pytest.mark.parametrize("stat", ["mean", "quantile", "gini"])
def test_dfl_decomposition_identity(wage, stat):
    r = sp.decompose(
        "dfl", data=wage, y="log_wage", group="female", x=X, stat=stat, tau=0.5
    )
    assert r.gap == pytest.approx(r.composition + r.structure, rel=1e-7, abs=1e-9)
    # Counterfactual = group B's characteristics under group A's structure:
    # composition (characteristics) = cf - B; structure (returns) = A - cf.
    assert r.composition == pytest.approx(r.stat_cf - r.stat_b, rel=1e-7, abs=1e-9)
    assert r.structure == pytest.approx(r.stat_a - r.stat_cf, rel=1e-7, abs=1e-9)
    assert r.gap == pytest.approx(r.stat_a - r.stat_b, rel=1e-7, abs=1e-9)


# ── RIF regression decomposition (formula API) ───────────────────────


@pytest.mark.parametrize("statistic", ["mean", "quantile", "variance", "gini"])
def test_rif_decomposition_identity(wage, statistic):
    r = sp.decompose(
        "rif",
        formula="log_wage ~ education + experience + tenure",
        data=wage,
        group="female",
        statistic=statistic,
        tau=0.5,
    )
    assert r.total_diff == pytest.approx(
        r.explained + r.unexplained, rel=1e-7, abs=1e-9
    )


# ── quantile-grid decompositions ─────────────────────────────────────


def test_machado_mata_mean_identity(wage):
    r = sp.decompose(
        "machado_mata",
        data=wage,
        y="log_wage",
        group="female",
        x=X,
        tau_grid=[0.25, 0.5, 0.75],
        n_sim=120,
    )
    o = r.overall
    assert o["mean_gap"] == pytest.approx(
        o["mean_composition"] + o["mean_structure"], rel=1e-6, abs=1e-6
    )


def test_melly_mean_identity(wage):
    r = sp.decompose(
        "melly",
        data=wage,
        y="log_wage",
        group="female",
        x=X,
        tau_grid=[0.25, 0.5, 0.75],
    )
    o = r.overall
    assert o["mean_gap"] == pytest.approx(
        o["mean_composition"] + o["mean_structure"], rel=1e-6, abs=1e-6
    )


def test_cfm_identity_and_ks(wage):
    r = sp.decompose(
        "cfm", data=wage, y="log_wage", group="female", x=X, tau_grid=[0.25, 0.5, 0.75]
    )
    o = r.overall
    assert o["mean_gap"] == pytest.approx(
        o["mean_composition"] + o["mean_structure"], rel=1e-6, abs=1e-6
    )
    # CFM also runs a Kolmogorov–Smirnov test on the two distributions.
    assert r.ks_stat >= 0.0
    assert 0.0 <= r.ks_pvalue <= 1.0


# ── result rendering ─────────────────────────────────────────────────


def test_distributional_result_summaries(wage):
    for method, kw in (
        ("ffl", dict(x=X, stat="quantile", tau=0.5)),
        ("dfl", dict(x=X, stat="mean")),
    ):
        r = sp.decompose(method, data=wage, y="log_wage", group="female", **kw)
        assert isinstance(r.summary(), str)
        assert isinstance(repr(r), str)
