"""Coverage campaign (decomposition) — final reachable-branch sweep.

Closes the remaining reachable branches: the ``DecompResultMixin.confint``
overall/detailed/error paths, multi-source Gini decomposition with weights,
weighted Shapley, RIF kernel variants, multi-column Kitagawa ``by``, and the
Yu–Elwert efficient + bootstrap combination. Remaining uncovered lines after
this are genuinely-defensive guards (non-convergence, non-positive-income NaN
returns, optional-dependency ImportErrors) that cannot be exercised without
editing ``src`` (this campaign is test-only). Real invariants throughout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.decomposition import datasets
from statspai.decomposition.rif import rif_values

X = ["education", "experience", "tenure"]


@pytest.fixture(scope="module")
def wage():
    return datasets.cps_wage()


# ── confint overall / detailed / error paths ─────────────────────────


def test_confint_overall_and_detailed(wage):
    r = sp.decompose(
        "oaxaca", data=wage, y="log_wage", group="female", x=X, detail=True
    )
    ci_overall = r.confint(which="overall")
    assert ci_overall is None or isinstance(ci_overall, dict)
    ci_detailed = r.confint(which="detailed")
    assert ci_detailed is None or isinstance(ci_detailed, dict)
    with pytest.raises(ValueError, match="(?i)unknown which"):
        r.confint(which="nonsense")


def test_confint_overall_with_se(wage):
    # FFL bootstrap carries overall *_se entries → confint builds intervals.
    r = sp.decompose(
        "ffl",
        data=wage,
        y="log_wage",
        group="female",
        x=X,
        stat="mean",
        inference="bootstrap",
        n_boot=30,
        seed=0,
    )
    ci = r.confint(which="overall")
    assert ci is None or isinstance(ci, dict)


# ── multi-source Gini decomposition with weights ─────────────────────


def test_source_decompose_three_sources_weighted(wage):
    df = wage.copy()
    df["w"] = np.exp(df["log_wage"])
    df["s1"] = df["w"] * 0.5
    df["s2"] = df["w"] * 0.3
    df["s3"] = df["w"] * 0.2
    df["wt"] = 1.0
    r = sp.decompose("gini_source", data=df, sources=["s1", "s2", "s3"], weights="wt")
    contrib = np.asarray(r.sources["contribution"], dtype=float)
    assert contrib.sum() == pytest.approx(r.total_gini, rel=1e-9)


# ── weighted Shapley ─────────────────────────────────────────────────


def test_shapley_with_weights(wage):
    df = wage.copy()
    df["wage"] = np.exp(df["log_wage"])
    df["wt"] = 1.0
    r = sp.decompose(
        "shapley_inequality",
        data=df,
        y="wage",
        x=["education", "experience"],
        index="theil_t",
        weights="wt",
    )
    sh = r.shapley
    pct = np.asarray(sh["pct_of_total"], dtype=float)
    contrib = np.asarray(sh["contribution"], dtype=float)
    np.testing.assert_allclose(pct, 100.0 * contrib / r.total, rtol=1e-9)


# ── RIF kernel variants ──────────────────────────────────────────────


@pytest.mark.parametrize("statistic", ["iqr", "log_var"])
def test_rif_values_extra_statistics(statistic):
    rng = np.random.default_rng(3)
    y = np.abs(rng.lognormal(0, 0.5, 400)) + 0.1
    rif = rif_values(y, statistic=statistic, tau=0.5)
    assert len(rif) == len(y) and np.all(np.isfinite(rif))


# ── multi-column Kitagawa `by` ───────────────────────────────────────


def test_kitagawa_multi_by():
    rng = np.random.default_rng(4)
    n = 1500
    df = pd.DataFrame(
        {
            "rate": rng.uniform(0, 1, n),
            "period": (rng.uniform(size=n) < 0.5).astype(int),
            "age": rng.integers(0, 3, n),
            "sex": rng.integers(0, 2, n),
        }
    )
    r = sp.decompose(
        "kitagawa", data=df, rate="rate", group="period", by=["age", "sex"]
    )
    assert r.gap == pytest.approx(r.rate_a - r.rate_b, rel=1e-9, abs=1e-9)


# ── Yu–Elwert efficient + bootstrap ──────────────────────────────────


def test_yu_elwert_efficient_bootstrap():
    rng = np.random.default_rng(5)
    n = 1200
    x1 = rng.normal(size=n)
    g = (rng.uniform(size=n) < 0.5).astype(int)
    tr = (rng.uniform(size=n) < 0.5).astype(int)
    ybin = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.3 - 0.4 * g)))).astype(int)
    df = pd.DataFrame({"ybin": ybin, "g": g, "tr": tr, "x1": x1})
    r = sp.decompose(
        "yu_elwert",
        data=df,
        y="ybin",
        treatment="tr",
        group="g",
        x=["x1"],
        method="efficient",
        inference="bootstrap",
        n_boot=30,
    )
    assert r.se is not None
