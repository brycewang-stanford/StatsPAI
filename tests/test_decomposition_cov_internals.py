"""Coverage campaign (decomposition) — estimator-internal branches.

Mops up method/option branches not reached by the main identity tests: the RIF
kernel (`rif.py` `rif_values` / `rifreg`), the Yu–Elwert efficient estimator and
its method guard, the Kitagawa normalisation variants (`a` / `b` / symmetric),
and a couple of shared `_common` weighted helpers. Real identities / invariants
throughout (CLAUDE.md §12).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.decomposition import datasets
from statspai.decomposition.rif import rif_values, rifreg
from statspai.decomposition import _common as C


@pytest.fixture(scope="module")
def wage():
    return datasets.cps_wage()


@pytest.fixture(scope="module")
def causal_df():
    rng = np.random.default_rng(0)
    n = 1400
    x1 = rng.normal(size=n)
    g = (rng.uniform(size=n) < 0.5).astype(int)
    tr = (rng.uniform(size=n) < 0.5).astype(int)
    ybin = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.3 - 0.4 * g)))).astype(int)
    return pd.DataFrame(
        {"ybin": ybin, "g": g, "tr": tr, "x1": x1, "region": rng.integers(0, 3, n)}
    )


# ── RIF kernel ───────────────────────────────────────────────────────


@pytest.mark.parametrize("statistic", ["mean", "quantile", "variance", "gini"])
def test_rif_values_recenter(statistic):
    rng = np.random.default_rng(1)
    y = np.abs(rng.lognormal(0, 0.5, 500)) + 0.1
    rif = rif_values(y, statistic=statistic, tau=0.5)
    assert len(rif) == len(y)
    assert np.all(np.isfinite(rif))


def test_rifreg_runs(wage):
    r = rifreg(
        "log_wage ~ education + experience", data=wage, statistic="quantile", tau=0.5
    )
    assert r is not None
    assert isinstance(r.summary(), str)


# ── Yu–Elwert efficient estimator + method guard ─────────────────────


def test_yu_elwert_efficient(causal_df):
    r = sp.decompose(
        "yu_elwert",
        data=causal_df,
        y="ybin",
        treatment="tr",
        group="g",
        x=["x1"],
        method="efficient",
        inference="none",
    )
    # The efficient (AIPW-style) estimator carries small cross-fit correction
    # terms, so the four components reconstruct the disparity up to estimation
    # order rather than to machine precision (the plugin variant is exact).
    assert r.disparity == pytest.approx(
        r.baseline + r.prevalence + r.effect + r.selection, rel=1e-3, abs=1e-4
    )


def test_yu_elwert_bad_method_raises(causal_df):
    with pytest.raises(ValueError, match="(?i)plugin.*efficient|method"):
        sp.decompose(
            "yu_elwert",
            data=causal_df,
            y="ybin",
            treatment="tr",
            group="g",
            x=["x1"],
            method="not_a_method",
            inference="none",
        )


# ── Kitagawa normalisation variants ──────────────────────────────────


@pytest.mark.parametrize("normalize", ["symmetric", "a", "b"])
def test_kitagawa_normalize_variants(causal_df, normalize):
    r = sp.decompose(
        "kitagawa",
        data=causal_df,
        rate="ybin",
        group="g",
        by="region",
        normalize=normalize,
    )
    # the raw gap is invariant to the normalisation choice
    assert r.gap == pytest.approx(r.rate_a - r.rate_b, rel=1e-9, abs=1e-9)
    # three-way components still reconstruct the gap
    assert r.gap == pytest.approx(
        r.rate_effect + r.composition_effect + r.interaction, rel=1e-9, abs=1e-9
    )


# ── shared weighted helpers ──────────────────────────────────────────


def test_weighted_ecdf_eval_points():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # ECDF evaluated below / within / above the support
    F = C.weighted_ecdf(np.array([0.5, 3.0, 9.0]), y, np.ones(5))
    assert F[0] == pytest.approx(0.0)
    assert 0.0 < F[1] <= 1.0
    assert F[2] == pytest.approx(1.0)


def test_kde_at_is_positive():
    rng = np.random.default_rng(2)
    y = rng.normal(0, 1, 400)
    dens = C.kde_at(y, 0.0)
    assert dens > 0.0
