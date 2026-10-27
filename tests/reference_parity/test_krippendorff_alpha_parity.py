"""sp.krippendorff_alpha against Stata kalpha / kappaetc and R irr.

References (same CSV bytes, ``_fixtures/krippendorff_{small,data}.csv``;
``small`` is the textbook example of the kalpha help file, ``data`` 80
units x 5 coders with 22% missing codings):

* Stata 18: kalpha 1.3.0 (Klein and Reckmann, SSC) for all four levels,
  returning alpha, D_o, D_e, the number of pairable values and units;
  kappaetc 2.1.0 (Klein, SSC) with identity / quadratic / ratio weights
  and Krippendorff's ordinal weights.  ``_fixtures/_generate_krippendorff_Stata.do``.
* R 4.5.2, irr 0.85 ``kripp.alpha`` (GPL, run as a black box).
  ``_fixtures/_generate_krippendorff_R.R``.

Point estimates: rtol 1e-10 (deterministic closed form; observed gaps
are at machine precision, ~2e-16).  The bootstrap row is a stochastic
screen (S), not parity: our generator is numpy's, not Stata's, so only
the distribution is compared, with tolerances of about six Monte Carlo
standard errors of a 20000-replication quantile.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility

FIX = Path(__file__).resolve().parent / "_fixtures"
LEVELS = ["nominal", "ordinal", "interval", "ratio"]


def _data(ds):
    return pd.read_csv(FIX / f"krippendorff_{ds}.csv")


@pytest.fixture(scope="module")
def ref_stata():
    return json.loads((FIX / "krippendorff_Stata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ref_r():
    return json.loads((FIX / "krippendorff_R.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("ds", ["small", "data"])
@pytest.mark.parametrize("level", LEVELS)
def test_kalpha_parity(ds, level, ref_stata):
    res = sp.krippendorff_alpha(_data(ds), level=level)
    ref = ref_stata[ds][f"kalpha_{level}"]
    np.testing.assert_allclose(res.estimate, ref["alpha"], rtol=1e-10)
    assert res.n_pairable == ref["n"]
    assert res.n_units == ref["units"]
    # kalpha reports D_o = (n-1) sum_{c<=k} o_ck d2 and D_e = sum_{c<=k} n_c n_k d2
    n = res.n_pairable
    np.testing.assert_allclose(
        res.observed_disagreement * n * (n - 1) / 2, ref["Do"], rtol=1e-10
    )
    np.testing.assert_allclose(
        res.expected_disagreement * n * (n - 1) / 2, ref["De"], rtol=1e-10
    )


@pytest.mark.parametrize("ds", ["small", "data"])
@pytest.mark.parametrize(
    "level, key",
    [
        ("nominal", "kappaetc_identity"),
        ("interval", "kappaetc_quadratic"),
        ("ratio", "kappaetc_ratio"),
        ("ordinal", "kappaetc_ordinal_krippendorff"),
    ],
)
def test_kappaetc_parity(ds, level, key, ref_stata):
    res = sp.krippendorff_alpha(_data(ds), level=level)
    np.testing.assert_allclose(res.estimate, ref_stata[ds][key], rtol=1e-10)


@pytest.mark.parametrize("ds", ["small", "data"])
@pytest.mark.parametrize("level", LEVELS)
def test_irr_parity(ds, level, ref_r):
    d = _data(ds)
    res = sp.krippendorff_alpha(d.T.to_numpy(), level=level, layout="coders_by_units")
    np.testing.assert_allclose(res.estimate, ref_r[ds][level]["alpha"], rtol=1e-10)
    assert res.n_pairable == ref_r[ds][level]["nmatchval"]


def test_textbook_values():
    # Values quoted with the kalpha help-file example (3 decimals).
    d = _data("small")
    got = [round(sp.krippendorff_alpha(d, level=lv).estimate, 3) for lv in LEVELS]
    assert got == [0.743, 0.815, 0.849, 0.797]


@pytest.mark.parametrize("level", ["nominal", "interval"])
def test_bootstrap_screen_vs_kalpha(level, ref_stata):
    """S (stochastic screen): kalpha's pair-drawing bootstrap, 20000 reps."""
    res = sp.krippendorff_alpha(
        _data("data"), level=level, bootstrap=True, n_boot=20000, seed=20261028
    )
    ref = ref_stata[f"boot_{level}"]
    assert res.ci[0] == pytest.approx(ref["ci_lb"], abs=3e-3)
    assert res.ci[1] == pytest.approx(ref["ci_ub"], abs=3e-3)
    assert res.prob_below[0.667] == pytest.approx(ref["q667"], abs=6e-3)
    assert res.prob_below[0.8] == pytest.approx(ref["q800"], abs=6e-3)
    assert res.diagnostics["draws"] == 600  # min(25 x nonzero cells, n(m-1)/2)


# ------------------------------------------------------- analytic checks --


def test_two_coders_complete_equals_scott_pi_identity():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 4, 150)
    b = np.where(rng.uniform(size=150) < 0.7, a, rng.integers(0, 4, 150))
    res = sp.krippendorff_alpha(np.column_stack([a, b]), level="nominal")
    po = np.mean(a == b)
    p = np.bincount(np.concatenate([a, b]), minlength=4) / 300
    pe = np.sum(p**2)
    n = 300
    alpha = 1 - (n - 1) * (1 - po) / (n * (1 - pe))
    assert res.estimate == pytest.approx(alpha, abs=1e-14)


def test_invariances_and_layout():
    d = _data("data")
    base = sp.krippendorff_alpha(d, level="interval").estimate
    assert sp.krippendorff_alpha(3 * d - 7, level="interval").estimate == pytest.approx(
        base, abs=1e-13
    )
    r = sp.krippendorff_alpha(d, level="ratio").estimate
    assert sp.krippendorff_alpha(2.5 * d, level="ratio").estimate == pytest.approx(
        r, abs=1e-13
    )
    o = sp.krippendorff_alpha(d, level="ordinal").estimate
    # ordinal depends only on the order of the values
    assert sp.krippendorff_alpha(np.exp(d), level="ordinal").estimate == pytest.approx(
        o, abs=1e-13
    )
    # string labels = numeric labels for nominal; layout transpose
    labels = d.apply(lambda c: c.map(lambda v: None if pd.isna(v) else f"cat{int(v)}"))
    nom = sp.krippendorff_alpha(d).estimate
    assert sp.krippendorff_alpha(labels).estimate == pytest.approx(nom, abs=1e-15)
    assert sp.krippendorff_alpha(
        d.T, layout="coders_by_units"
    ).estimate == pytest.approx(nom, abs=1e-15)
    perfect = sp.krippendorff_alpha([[1, 1, None], [2, 2, 2], [3, None, 3]])
    assert perfect.estimate == 1.0


# ----------------------------------------------------------------- edges --


def test_undefined_and_bad_inputs():
    with pytest.raises(DataInsufficient):
        sp.krippendorff_alpha([[1, 1], [1, 1], [1, None]])
    with pytest.raises(DataInsufficient):
        sp.krippendorff_alpha([[1, 2], [3, None], [None, 1]])
    with pytest.raises(MethodIncompatibility):
        sp.krippendorff_alpha([[1, -2], [3, 3]], level="ratio")
    with pytest.raises(MethodIncompatibility):
        sp.krippendorff_alpha([["a", "b"], ["a", "a"]], level="interval")
    with pytest.raises(MethodIncompatibility):
        sp.krippendorff_alpha([[1, 2], [2, 2]], level="cardinal")
    with pytest.raises(MethodIncompatibility):
        sp.krippendorff_alpha([[1, 2], [2, 2]], ci_method="jackknife")


def test_bootstrap_seeded_and_unit_resampling():
    d = _data("data")
    a = sp.krippendorff_alpha(d, level="ordinal", bootstrap=True, n_boot=2000, seed=7)
    b = sp.krippendorff_alpha(d, level="ordinal", bootstrap=True, n_boot=2000, seed=7)
    assert a.ci == b.ci and np.array_equal(a.boot_dist, b.boot_dist)
    assert a.ci[0] < a.estimate < a.ci[1]
    u = sp.krippendorff_alpha(
        d, level="ordinal", bootstrap=True, n_boot=500, seed=7, ci_method="units"
    )
    assert u.ci[0] < u.estimate < u.ci[1]
    with pytest.warns(UserWarning, match="degenerate"):
        p = sp.krippendorff_alpha([[1, 1], [2, 2], [3, 3]], bootstrap=True, seed=1)
    assert p.ci is None and "bootstrap_skipped" in p.diagnostics


def test_result_surface():
    res = sp.krippendorff_alpha(_data("small"), level="interval")
    assert res.kalpha == res.estimate
    assert "reliable" in res.interpretation() or "tentative" in res.interpretation()
    assert res.to_frame().loc[0, "n_units"] == 11
    assert res.coincidence.to_numpy().sum() == pytest.approx(res.n_pairable)
    assert "Krippendorff" in res.summary()
    assert json.dumps(res.to_dict(), default=str)
