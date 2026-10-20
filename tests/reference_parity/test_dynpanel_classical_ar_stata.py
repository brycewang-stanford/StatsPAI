"""Classical (one-step, non-robust) dynamic-panel GMM against Stata xtabond2.

Black-box values from ``_fixtures/_generate_dynpanel_classical_ar_stata.do``
(Stata 18 MP, xtabond2 from SSC) on ``_fixtures/dynpanel_abdata.csv``.

xtabond2 divides ``e'e`` by ``N*`` where StatsPAI (like ``xtabond``) divides
by ``N* - k``; every classical statistic therefore differs by the known
factor ``sqrt(N* / (N* - k))`` (z statistics) or its inverse (SEs). With
that factor applied the values agree to 1e-9. Before 2026-10-20 the
classical AR statistics for system GMM and after orthogonal deviations did
not (FOD AR(1) -2.76 against -6.44), and the classical SEs after
orthogonal deviations were too small by about sqrt(2) (sigma^2 was halved).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from statspai.gmm._dynpanel import fit_dynamic_panel

DATA = Path(__file__).parent / "_fixtures" / "dynpanel_abdata.csv"

#: spec -> (fit_dynamic_panel kwargs, xtabond2 e(ar1), e(ar2))
STATA = {
    "sys_nox_cls": (dict(method="system"), -8.08908719645296, -1.72224128212807),
    "sys_nox_cls_h2": (
        dict(method="system", h=2),
        -8.77920033116838,
        -1.68734741303522,
    ),
    "sys_wk_cls": (
        dict(method="system", x=["w", "k"]),
        -9.53489635399684,
        -0.624969401006101,
    ),
    "fod_diff_nox_cls": (dict(transform="fod"), -6.43981587936347, -1.338075443384),
    "fod_diff_wk_cls": (
        dict(transform="fod", x=["w", "k"]),
        -2.85691507097662,
        -0.482354255489437,
    ),
    "fod_sys_nox_cls": (
        dict(transform="fod", method="system"),
        -7.26866304589011,
        -1.56469816448354,
    ),
    "diff_nox_cls": ({}, -7.2099423355829, -1.48243995163872),
}


@pytest.fixture(scope="module")
def abdata() -> pd.DataFrame:
    return pd.read_csv(DATA)


def _fit(df, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_dynamic_panel(df, y="n", id="id", time="year", robust=False, **kw)


def _df_factor(fit) -> float:
    d = fit["design"]
    n_star = int((d.row_eq == 0).sum())
    return float(np.sqrt(n_star / (n_star - d.n_params)))


@pytest.mark.parametrize("spec", sorted(STATA))
def test_classical_ar_matches_xtabond2(abdata, spec):
    kw, ar1, ar2 = STATA[spec]
    fit = _fit(abdata, **kw)
    scale = _df_factor(fit)
    np.testing.assert_allclose(fit["ar1"]["z"] * scale, ar1, rtol=1e-9)
    np.testing.assert_allclose(fit["ar2"]["z"] * scale, ar2, rtol=1e-9)


def test_fod_classical_se_matches_xtabond2(abdata):
    fit = _fit(abdata, transform="fod", x=["w", "k"])
    np.testing.assert_allclose(fit["beta"][0], 0.397106788174169, rtol=1e-9)
    np.testing.assert_allclose(
        fit["se"][0] / _df_factor(fit), 0.0906500167086847, rtol=1e-9
    )


def test_h1_sigma2_is_on_the_scale_of_its_h(abdata):
    """With h=1 (H = I) and with orthogonal deviations sigma^2 is e'e/(N*-k);
    with the first-difference H it is e'e/(2(N*-k))."""
    for kw, factor in (
        (dict(method="system", h=1), 1.0),
        (dict(transform="fod"), 1.0),
        (dict(), 2.0),
    ):
        fit = _fit(abdata, **kw)
        d = fit["design"]
        rows = d.row_eq == 0
        r = (d.dy - d.W @ fit["beta"])[rows]
        expected = float(r @ r) / (factor * (rows.sum() - d.n_params))
        assert fit["sigma2"] == pytest.approx(expected, rel=1e-12)
