"""``sp.hdfe_ols`` fit statistics and reference df against Stata ``reghdfe``.

``e(r2)``, ``e(r2_a)``, ``e(r2_within)``, ``e(r2_a_within)``, ``e(rmse)`` and
``e(df_r)`` from reghdfe 6.13.1 on ``_fixtures/hdfe_fit_stats.csv``
(``_fixtures/_generate_hdfe_fit_stats_Stata.do``). The adjusted statistics
charge the fixed effects nested in a cluster in full (reghdfe's
``df_a_nested``), and under clustering the t reference distribution has
``min(G) - 1`` degrees of freedom. Printed at 16 significant digits; tolerance
1e-12 (observed <= 5e-16).

Origin: Zheng, Huang & Zhu (2026) report adjusted R² in every column; before
1.33 ``sp.hdfe_ols`` had only ``r2_within`` and used ``t(N - K - df_a)`` for
clustered p-values (p = 0.088 against reghdfe's 0.091 with 100 clusters).
"""

import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp

DATA = pd.read_csv(pathlib.Path(__file__).parent / "_fixtures" / "hdfe_fit_stats.csv")

# fe, cluster, (r2, r2_a, r2_within, r2_a_within, rmse, df_r, se_x)
CASES = [
    (
        "firm + occ + city^q + ind^q",
        "ind",
        (
            0.7971705606274802,
            0.7058236460432461,
            0.2542831839382557,
            0.2506719644900148,
            1.021387637158659,
            11,
            0.03441906825852581,
        ),
    ),
    (
        "firm + occ + city^q",
        "occ",
        (
            0.7816847820622498,
            0.7235289311951112,
            0.2727765773227774,
            0.269701636846341,
            0.9901740995598752,
            14,
            0.04724734531203784,
        ),
    ),
    (
        "firm + occ + city^q + ind^q",
        None,
        (
            0.7971705606274802,
            0.714129801919672,
            0.2542831839382557,
            0.2507739283332593,
            1.00686479880474,
            425,
            0.04887508063195171,
        ),
    ),
    (
        "occ + city^q",
        ["ind", "city"],
        (
            0.4422545928343178,
            0.3858648917421992,
            0.1797471052907776,
            0.1767314696484644,
            1.475770005010013,
            7,
            0.1201399927543633,
        ),
    ),
]


def _fit(fe, cl, **kw):
    kw = dict(kw, **({"cluster": cl} if cl is not None else {"vce": "robust"}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.hdfe_ols(f"y ~ x + z | {fe}", DATA, **kw)


@pytest.mark.parametrize("fe,cl,ref", CASES)
def test_fit_statistics_match_reghdfe(fe, cl, ref):
    r = _fit(fe, cl)
    got = (r.r2, r.r2_a, r.r2_within, r.r2_a_within, r.rmse)
    np.testing.assert_allclose(got, ref[:5], rtol=0, atol=1e-12)
    assert r.df_inference == ref[5]


@pytest.mark.parametrize("fe,cl,ref", CASES[:3])
def test_p_values_and_cis_use_reghdfe_df(fe, cl, ref):
    r = _fit(fe, cl)
    assert float(r.se["x"]) == pytest.approx(ref[6], rel=1e-8)
    t = float(r.tvalues["x"])
    assert float(r.pvalues["x"]) == pytest.approx(
        2 * stats.t.sf(abs(t), ref[5]), rel=1e-12
    )
    half = float(r.conf_int_upper["x"] - r.coef["x"])
    assert half == pytest.approx(
        stats.t.ppf(0.975, ref[5]) * float(r.se["x"]), rel=1e-12
    )
    assert "t(" in r.summary()


def test_df_inference_options():
    fe, cl, ref = CASES[0]
    base = _fit(fe, cl)
    old = _fit(fe, cl, df_inference="resid")
    z = _fit(fe, cl, df_inference="normal")
    fixed = _fit(fe, cl, df_inference=30)
    assert old.df_inference == base.df_resid
    assert np.isinf(z.df_inference)
    assert fixed.df_inference == 30
    for r in (old, z, fixed):  # the SEs never move
        assert float(r.se["x"]) == float(base.se["x"])
    assert float(z.pvalues["x"]) < float(fixed.pvalues["x"]) < float(base.pvalues["x"])
    with pytest.raises(ValueError):
        _fit(fe, cl, df_inference="bogus")
    with pytest.raises(ValueError):
        _fit(fe, cl, df_inference=0)


def test_result_card_reports_the_t_reference():
    r = _fit(*CASES[0][:2])
    card = sp.result_card(r)
    assert card["inference"]["reference_distribution"] == "t(11)"


def test_unicode_column_names_parse():
    d = DATA.rename(columns={"y": "工资", "x": "暴露度", "occ": "职业"})
    r = sp.hdfe_ols("工资 ~ 暴露度 + z | firm + 职业", d, cluster="ind")
    ref = sp.hdfe_ols("y ~ x + z | firm + occ", DATA, cluster="ind")
    assert float(r.coef["暴露度"]) == float(ref.coef["x"])
