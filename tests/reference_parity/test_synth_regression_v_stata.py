"""``sp.synth(v_method='regression')`` against Stata ``synth``.

Stata ``synth`` without its ``nested`` option does not search for the
predictor weights V: it takes them from a regression of the pre-treatment
outcomes on the predictors. That is the command's default, it is
deterministic and it runs in milliseconds, so it is the variant a textbook
do-file runs. StatsPAI only had the nested search.

Data: ``sp.california_prop99()``, written to a .dta with states numbered
alphabetically (California = 3), and in Stata 18.0::

    tsset sid year
    synth packspercapita packspercapita(1975) packspercapita(1980) ///
          packspercapita(1988) packspercapita(1970(1)1974),        ///
          trunit(3) trperiod(1989)

Evidence tier. V and the pre-treatment RMSPE agree to 1e-9. The donor
weights Stata *stores* are rounded to three decimals, and its synthetic
path ``e(Y_synthetic)`` is built from those rounded weights, which here sum
to 0.999. So Stata's reported gaps are off by about 0.1% of the outcome
level and its post-treatment average is -18.70 where the exact weights give
-18.81. ``test_stata_synthetic_path_uses_rounded_weights`` shows that
rounding our weights reproduces Stata's path to 1e-9, which is the evidence
that the two solve the same problem.
"""

import time

import numpy as np
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility

SPEC = [
    ("packspercapita", 1975, "mean"),
    ("packspercapita", 1980, "mean"),
    ("packspercapita", 1988, "mean"),
    ("packspercapita", slice(1970, 1974), "mean"),
]

# e(V_matrix) diagonal, %16.12f
STATA_V = [0.101956683041, 0.149656598334, 0.165047264026, 0.583339454600]
# e(RMSPE), %18.12f
STATA_RMSPE = 3.505852292371
# e(W_weights): Stata's unit number -> weight (three decimals)
STATA_W = {
    "Arkansas": 0.124,
    "Missouri": 0.26,
    "North Dakota": 0.239,
    "Oklahoma": 0.376,
}
# e(Y_synthetic) for 1970, 1989 and 2000, %16.10f
STATA_Y_SYNTH = {1970: 122.7217743437, 1989: 98.1242620895, 2000: 89.0707330886}
# mean post-treatment gap from e(Y_treated) - e(Y_synthetic)
STATA_ATT = -18.703879676715


@pytest.fixture(scope="module")
def df():
    return sp.california_prop99()


@pytest.fixture(scope="module")
def fit(df):
    return sp.synth(
        df,
        "packspercapita",
        "state",
        "year",
        "California",
        1989,
        method="classic",
        special_predictors=SPEC,
        v_method="regression",
        placebo=False,
    )


def test_predictor_weights_match_stata(fit):
    v = fit.model_info["v_weights"]["v_weight"].to_numpy()
    # StatsPAI normalises tr(V) = K, Stata to a sum of one.
    assert v.sum() == pytest.approx(len(SPEC), rel=1e-12)
    np.testing.assert_allclose(v / v.sum(), STATA_V, rtol=1e-10)
    assert fit.model_info["v_method"] == "regression"


def test_pre_treatment_rmspe_matches_stata(fit):
    gaps = fit.model_info["gap_table"]
    pre = gaps.loc[gaps["time"] < 1989, "gap"].to_numpy()
    # rtol: Stata's quadratic-programming plugin stops at its own
    # tolerance; 2e-10 observed.
    assert np.sqrt(np.mean(pre**2)) == pytest.approx(STATA_RMSPE, rel=1e-8)


def test_donor_weights_round_to_stata(fit):
    w = fit.model_info["weights"].set_index("unit")["weight"]
    assert set(w.index) == set(STATA_W)
    for unit, ref in STATA_W.items():
        assert round(float(w[unit]), 3) == pytest.approx(ref, abs=1e-12)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)


def test_stata_synthetic_path_uses_rounded_weights(df, fit):
    wide = df.pivot(index="year", columns="state", values="packspercapita")
    w = fit.model_info["weights"].set_index("unit")["weight"]
    rounded = w.round(3)
    assert rounded.sum() == pytest.approx(0.999, abs=1e-12)
    path_rounded = wide[rounded.index].to_numpy() @ rounded.to_numpy()
    path_exact = wide[w.index].to_numpy() @ w.to_numpy()
    years = wide.index.to_numpy()
    for year, ref in STATA_Y_SYNTH.items():
        i = int(np.flatnonzero(years == year)[0])
        assert path_rounded[i] == pytest.approx(ref, rel=1e-9)
        # the exact path differs by the missing 0.1% of weight
        assert abs(path_exact[i] - ref) > 0.05
    gap_rounded = wide["California"].to_numpy() - path_rounded
    assert gap_rounded[years >= 1989].mean() == pytest.approx(STATA_ATT, rel=1e-9)
    assert fit.estimate == pytest.approx(
        (wide["California"].to_numpy() - path_exact)[years >= 1989].mean(), rel=1e-12
    )


def test_weights_satisfy_the_optimality_conditions(df, fit):
    # No donor outside the support would lower the V-weighted predictor
    # distance: the weights are the exact minimiser, not a solver's stop.
    mi = fit.model_info
    wide = df.pivot(index="year", columns="state", values="packspercapita")
    donors = [s for s in wide.columns if s != "California"]

    def predictors(states):
        rows = [
            wide.loc[1975, states],
            wide.loc[1980, states],
            wide.loc[1988, states],
            wide.loc[1970:1974, states].mean(),
        ]
        return np.vstack([np.atleast_1d(np.asarray(r, dtype=float)) for r in rows])

    X0 = predictors(donors)
    X1 = predictors(["California"]).ravel()
    scale = np.column_stack([X1[:, None], X0]).std(axis=1, ddof=1)
    v = mi["v_weights"]["v_weight"].to_numpy()
    w = mi["weights"].set_index("unit")["weight"].reindex(donors).fillna(0).to_numpy()
    resid = (X1 - X0 @ w) / scale
    grad = -2.0 * (X0 / scale[:, None]).T @ (v * resid)
    lam = grad[w > 1e-9].mean()
    assert np.allclose(grad[w > 1e-9], lam, rtol=0, atol=1e-9 * abs(grad).max())
    assert (grad - lam > -1e-9 * abs(grad).max()).all()


def test_fast_and_deterministic_with_placebos(df):
    start = time.perf_counter()
    kwargs = dict(method="classic", special_predictors=SPEC, v_method="regression")
    a = sp.synth(df, "packspercapita", "state", "year", "California", 1989, **kwargs)
    b = sp.synth(df, "packspercapita", "state", "year", "California", 1989, **kwargs)
    elapsed = time.perf_counter() - start
    assert a.estimate == b.estimate and a.pvalue == b.pvalue
    assert len(a.model_info["placebo_atts"]) == 38
    assert 1 / 39 <= a.pvalue <= 1.0
    # 39 closed-form fits, twice. The nested search takes minutes for the
    # same job; a generous bound still separates the two.
    assert elapsed < 30


def test_requires_predictors(df):
    with pytest.raises(MethodIncompatibility, match="needs predictors"):
        sp.synth(
            df,
            "packspercapita",
            "state",
            "year",
            "California",
            1989,
            method="classic",
            v_method="regression",
            placebo=False,
        )


# No more units than coefficients: the regression behind V is not identified.
# Stata 18, synth 0.0.8, on _fixtures/textbook_rcm.csv:
#   synth y y(2) y(5) y(8) y(11) y(14) y(17) y(20) y(23) y(26) y(29)
#         y(1(1)15) y(16(1)30), trunit(6) trperiod(31) [counit(1 2 3 4 5 7 8 9)]
# vecdiag(e(V_matrix)) %14.11f and e(RMSPE) %18.12f. With 12 units the
# constant is the column that cannot be swept; with 9 units three predictors
# and the constant are left out, and those predictors get weight zero.
UNDER_IDENTIFIED = {
    12: (
        [0.00587784254, 0.09342564234, 0.01402255917, 0.00731886058,
         0.02219887455, 0.19212400170, 0.32787194210, 0.00361166371,
         0.09293378796, 0.14777398055, 0.06661408628, 0.02622675853],
        0.462222007743,
    ),
    9: (
        [0.0, 0.04636015382, 0.19946801331, 0.09113036274, 0.0,
         0.17430375679, 0.0, 0.04981422633, 0.14375430443, 0.15185117841,
         0.08557924025, 0.05773876393],
        0.469186743369,
    ),
}  # fmt: skip


@pytest.mark.parametrize("n_units", [12, 9])
def test_under_identified_v_follows_statas_sweep(n_units):
    from pathlib import Path

    import pandas as pd

    data = pd.read_csv(Path(__file__).parent / "_fixtures" / "textbook_rcm.csv")
    if n_units == 9:
        data = data[data["unit"] <= 9]
    spec = [("y", t, "mean") for t in range(2, 30, 3)] + [
        ("y", list(range(1, 16)), "mean"),
        ("y", list(range(16, 31)), "mean"),
    ]
    fit = sp.synth(
        data, "y", "unit", "t", treated_unit=6, treatment_time=31,
        method="classic", special_predictors=spec, v_method="regression",
        placebo=False,
    )  # fmt: skip
    stata_v, stata_rmspe = UNDER_IDENTIFIED[n_units]
    v = fit.model_info["v_weights"]["v_weight"].to_numpy()
    np.testing.assert_allclose(v / v.sum(), stata_v, atol=5e-11)
    gaps = fit.model_info["gap_table"]
    pre = gaps.loc[~gaps["post_treatment"], "gap"].to_numpy()
    assert np.sqrt((pre**2).mean()) == pytest.approx(stata_rmspe, rel=1e-8)


def test_unknown_v_method_lists_regression(df):
    with pytest.raises(MethodIncompatibility, match="regression"):
        sp.synth(
            df,
            "packspercapita",
            "state",
            "year",
            "California",
            1989,
            method="classic",
            special_predictors=SPEC,
            v_method="stata",
            placebo=False,
        )
