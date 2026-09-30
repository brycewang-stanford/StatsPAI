"""``sp.gelbach(absorb=, cluster=, shapley=)`` against Gelbach's ``b1x2``.

``absorb='fe'`` sweeps the fixed effects out of every variable; the reference
is ``b1x2`` with the FE dummies in ``x1all()``, the same model in Stata's
grammar. Reference: b1x2 4.1.0 on ``_fixtures/gelbach_fe_cluster.csv``
(``_fixtures/_generate_gelbach_fe_cluster_Stata.do``), 17 significant digits.
Tolerance rel 1e-8 (the absorber's convergence; observed <= 1e-10).

Origin: Zheng, Huang & Zhu (2026) Appendix Table 5 decomposes a coefficient
from a regression with four absorbed FEs and industry clusters; before 1.33
``sp.gelbach`` took neither, and the Shapley shares had to be assembled from
four separate regressions.
"""

import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

DATA = pd.read_csv(
    pathlib.Path(__file__).parent / "_fixtures" / "gelbach_fe_cluster.csv"
)
REL = 1e-8


def _fit(**kw):
    return sp.gelbach(DATA, "y", ["x"], ["a1", "a2"], **kw)


def _check(r, b, V):
    np.testing.assert_allclose(r.decomposition["delta"], b[:2], rtol=REL)
    assert r.total_change == pytest.approx(b[2], rel=REL)
    np.testing.assert_allclose(r.vcov.to_numpy(), np.array(V)[:2, :2], rtol=REL)
    assert r.total_se**2 == pytest.approx(V[2][2], rel=REL)


def test_cluster_without_fe():
    _check(
        _fit(cluster="cl"),
        [0.2867822352544705, -0.086024505119350558, 0.20075773013511994],
        [
            [0.0028890586468349657, -0.00032084736240876237, 0],
            [-0.00032084736240876237, 0.00044036326973505063, 0],
            [0, 0, 0.0026877271917524919],
        ],
    )


B_FE = [0.30305368777190844, -0.12499112184773423, 0.17806256592417422]


def test_absorb_with_cluster_equals_dummies():
    _check(
        _fit(absorb="fe", cluster="cl"),
        B_FE,
        [
            [0.0016470563553512142, -0.000071262223525770074, 0],
            [-0.000071262223525770074, 0.00074687598816179971, 0],
            [0, 0, 0.002251407896461474],
        ],
    )


def test_absorb_robust_equals_dummies():
    _check(
        _fit(absorb="fe"),
        B_FE,
        [
            [0.0022380814342796681, -0.000097551000526944402, 0],
            [-0.000097551000526944402, 0.00073614013627608594, 0],
            [0, 0, 0.0027791195695018654],
        ],
    )


def test_absorb_homoskedastic_equals_dummies():
    _check(
        _fit(absorb="fe", vce="nonrobust"),
        B_FE,
        [
            [0.0019097464566781277, 0.000010732571635404703, 0],
            [0.000010732571635404703, 0.00082531806747160276, 0],
            [0, 0, 0.0027565296674205396],
        ],
    )


def test_shapley_shares_average_over_entry_orders():
    r = _fit(absorb="fe", shapley=True)
    sh = r.decomposition.set_index("variable")["shapley"]
    assert sh.sum() == pytest.approx(r.total_change, rel=1e-12)

    def b(cols):
        f = "y ~ x" + "".join(f" + {c}" for c in cols) + " | fe"
        return float(sp.hdfe_ols(f, DATA, tol=1e-12).coef["x"])

    b0, b1, b2, b12 = b([]), b(["a1"]), b(["a2"]), b(["a1", "a2"])
    assert sh["a1"] == pytest.approx(0.5 * (b0 - b1) + 0.5 * (b2 - b12), rel=1e-8)
    assert sh["a2"] == pytest.approx(0.5 * (b0 - b2) + 0.5 * (b1 - b12), rel=1e-8)


def test_boundaries():
    with pytest.raises(ValueError, match="vce='robust'"):
        _fit(cluster="cl", vce="nonrobust")
    with pytest.raises(ValueError, match="not found"):
        _fit(absorb="nope")
