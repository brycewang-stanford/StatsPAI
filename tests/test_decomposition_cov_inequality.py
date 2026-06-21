"""Coverage campaign (decomposition) — inequality indices & decompositions.

Targets ``decomposition/inequality.py``: the ``inequality_index`` family, the
subgroup (within/between) decomposition, the Lerman–Yitzhaki Gini source
decomposition, and the Shorrocks–Shapley covariate decomposition.

Each test pins the *defining* algebraic identity of the method, not a smoke
call (CLAUDE.md §5):

* Additive decomposability:  ``total == within + between`` (Theil-T, Theil-L /
  MLD) exactly;  ``total == within + between + overlap`` for the Gini.
* Source decomposition:  ``sum_k contribution_k == total_gini`` exactly.
* Shapley internal consistency:  ``pct_of_total == 100 * contribution / total``.
* Indices vanish at perfect equality and match the shared weighted-Gini kernel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.decomposition import datasets
from statspai.decomposition.inequality import inequality_index
from statspai.decomposition._common import weighted_gini


@pytest.fixture(scope="module")
def income() -> pd.DataFrame:
    df = datasets.cps_wage()
    df = df.copy()
    df["wage"] = np.exp(df["log_wage"])
    df["region"] = np.random.default_rng(0).integers(0, 4, len(df))
    return df


# ── inequality_index ─────────────────────────────────────────────────


@pytest.mark.parametrize("index", ["theil_t", "theil_l", "gini", "atkinson"])
def test_index_zero_at_perfect_equality(index):
    y = np.full(100, 7.0)
    assert inequality_index(y, index=index) == pytest.approx(0.0, abs=1e-9)


def test_index_positive_under_inequality():
    y = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    for index in ("theil_t", "theil_l", "gini", "atkinson"):
        assert inequality_index(y, index=index) > 0.0


def test_index_gini_matches_kernel():
    rng = np.random.default_rng(1)
    y = np.abs(rng.lognormal(0, 0.6, 400)) + 0.1
    assert inequality_index(y, index="gini") == pytest.approx(
        weighted_gini(y, np.ones_like(y)), rel=1e-9
    )


def test_generalized_entropy_alpha():
    rng = np.random.default_rng(2)
    y = np.abs(rng.lognormal(0, 0.5, 300)) + 0.1
    ge2 = inequality_index(y, alpha=2.0)  # GE(2) = half squared CV
    assert ge2 > 0.0


# ── subgroup (within / between) decomposition ────────────────────────


@pytest.mark.parametrize("index", ["theil_t", "theil_l"])
def test_subgroup_additive_decomposability(income, index):
    r = sp.decompose("inequality", data=income, y="wage", by="region", index=index)
    # Theil indices are perfectly additively decomposable.
    assert r.total == pytest.approx(r.within + r.between, rel=1e-9)
    assert len(r.per_group) >= 2


def test_subgroup_gini_has_overlap_term(income):
    r = sp.decompose("inequality", data=income, y="wage", by="region", index="gini")
    # The Gini is not additively decomposable: total = within + between + overlap.
    overlap = r.overlap if r.overlap is not None else 0.0
    assert r.total == pytest.approx(r.within + r.between + overlap, rel=1e-6)


# ── Gini source decomposition (Lerman–Yitzhaki) ──────────────────────


def test_source_decomposition_sums_to_total_gini(income):
    inc = income.copy()
    inc["s1"] = inc["wage"] * 0.6
    inc["s2"] = inc["wage"] * 0.4
    r = sp.decompose("gini_source", data=inc, sources=["s1", "s2"])
    contrib = np.asarray(r.sources["contribution"], dtype=float)
    assert contrib.sum() == pytest.approx(r.total_gini, rel=1e-9)
    assert np.asarray(r.sources["pct_of_gini"], dtype=float).sum() == pytest.approx(
        100.0, abs=1e-6
    )


# ── Shorrocks–Shapley covariate decomposition ────────────────────────


def test_shapley_internal_consistency(income):
    r = sp.decompose(
        "shapley_inequality",
        data=income,
        y="wage",
        x=["education", "experience"],
        index="theil_t",
    )
    sh = r.shapley
    contrib = np.asarray(sh["contribution"], dtype=float)
    pct = np.asarray(sh["pct_of_total"], dtype=float)
    # Each reported percentage is exactly contribution / total.
    np.testing.assert_allclose(pct, 100.0 * contrib / r.total, rtol=1e-9)
    # Covariates explain a non-negative share not exceeding the total.
    assert 0.0 <= contrib.sum() <= r.total + 1e-9


# ── result rendering surface ─────────────────────────────────────────


def test_inequality_result_rendering(income):
    r = sp.decompose("inequality", data=income, y="wage", by="region", index="theil_t")
    assert isinstance(r.summary(), str)
    assert isinstance(repr(r), str)


# ── additional index variants + GE equivalences ──────────────────────


@pytest.mark.parametrize("index", ["mld", "ge0", "ge1", "ge2"])
def test_more_indices_zero_and_positive(index):
    assert inequality_index(np.full(50, 4.0), index=index) == pytest.approx(
        0.0, abs=1e-9
    )
    y = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    assert inequality_index(y, index=index) > 0.0


def test_generalized_entropy_limit_equivalences():
    rng = np.random.default_rng(3)
    y = np.abs(rng.lognormal(0, 0.4, 400)) + 0.1
    # GE(0) == MLD == Theil-L,  GE(1) == Theil-T.
    assert inequality_index(y, index="ge0") == pytest.approx(
        inequality_index(y, index="theil_l"), rel=1e-9
    )
    assert inequality_index(y, index="ge1") == pytest.approx(
        inequality_index(y, index="theil_t"), rel=1e-9
    )


@pytest.mark.parametrize("eps", [0.5, 2.0])
def test_atkinson_inequality_aversion(income, eps):
    rng = np.random.default_rng(4)
    y = np.abs(rng.lognormal(0, 0.5, 300)) + 0.1
    a = inequality_index(y, index="atkinson", eps=eps)
    assert 0.0 < a < 1.0


def test_unknown_index_raises():
    with pytest.raises(ValueError, match="(?i)unknown index"):
        inequality_index(np.array([1.0, 2.0]), index="not_an_index")


# ── source / shapley result rendering ────────────────────────────────


def test_source_and_shapley_rendering(income):
    inc = income.copy()
    inc["s1"] = inc["wage"] * 0.7
    inc["s2"] = inc["wage"] * 0.3
    src = sp.decompose("gini_source", data=inc, sources=["s1", "s2"])
    assert isinstance(src.summary(), str)
    assert isinstance(src.to_latex(), str)
    assert "<table" in src._repr_html_()

    sh = sp.decompose(
        "shapley_inequality",
        data=income,
        y="wage",
        x=["education", "experience"],
        index="theil_t",
    )
    assert isinstance(sh.summary(), str)
    assert isinstance(sh.to_latex(), str)
    assert "<table" in sh._repr_html_()
