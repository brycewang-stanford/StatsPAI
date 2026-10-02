"""``sp.match(method='nnmatch')`` against Stata ``teffects nnmatch``.

Data: ``sp.datasets.nsw_dw()`` (185 NSW treated, 2,490 PSID controls),
written to a .dta and read by Stata 18.0 MP, so both sides compute on the
same numbers. Matching covariates throughout::

    age education black hispanic married nodegree re74 re75

Each row of ``CASES`` is one Stata command; the reference is
``_b[r1vs0.treat]`` and ``_se[r1vs0.treat]`` printed with ``%20.12f``.

Two details of Stata's variance are not in its manual and were found by
rebuilding ``e(V)`` term by term (see ``matching/nnmatch.py``): the same-arm
set behind the robust conditional variance holds the unit plus ``nn(h)``
neighbours, and those neighbours must satisfy ``ematch()`` too. The
``vce_nn=4`` and ``exact=`` rows below are the ones that distinguish them.
"""

import numpy as np
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility

X = ["age", "education", "black", "hispanic", "married", "nodegree", "re74", "re75"]

# rtol: deterministic estimator, same bytes on both sides. The loosest
# agreement observed is 5e-13 (the bias-adjustment least squares).
RTOL = 1e-10

# label: (kwargs, estimate, standard error)
CASES = {
    # teffects nnmatch (re78 $X)(treat), atet
    "att": ({}, 367.727172023548, 1165.576724477436),
    # teffects nnmatch (re78 $X)(treat)
    "ate": ({"estimand": "ATE"}, -8048.540150511234, 1145.460906301221),
    # ..., atet biasadj($X)
    "att_biasadj_all": ({"bias_adjust": True}, 568.373267898747, 1160.307963109378),
    # ..., atet metric(ivariance) nn(3)
    "att_ivariance_nn3": (
        {"metric": "ivariance", "n_matches": 3},
        -359.892578304576,
        1175.831351008482,
    ),
    # ..., atet metric(euclidean)
    "att_euclidean": ({"metric": "euclidean"}, 189.596500603699, 1287.388173524406),
    # ..., atet vce(iid)
    "att_iid": ({"vce": "iid"}, 367.727172023548, 1610.429926881280),
    # ..., atet vce(robust, nn(4))
    "att_robust_nn4": ({"vce_nn": 4}, 367.727172023548, 1306.385305024590),
    # ..., atet ematch(black) biasadj(age re74)
    "att_ematch_biasadj": (
        {"exact": ["black"], "bias_adjust": ["age", "re74"]},
        1968.905259373524,
        1206.497325950330,
    ),
    # ..., ate biasadj(age education re74 re75) nn(2)
    "ate_biasadj_nn2": (
        {
            "estimand": "ATE",
            "bias_adjust": ["age", "education", "re74", "re75"],
            "n_matches": 2,
        },
        6841.535166148521,
        866.319641242936,
    ),
}

# tebalance summarize after the "att" fit: std diff raw, matched; ratio raw,
# matched (%16.12f).
BALANCE = {
    "age": (-1.203188854847, -0.425462125415, 0.356965335660, 0.629524790602),
    "education": (-0.770409310737, -0.160968432473, 0.433489397326, 0.753598041429),
    "black": (1.348539366892, 0.0, 0.847856309262, 1.0),
    "hispanic": (0.318814042126, 0.0, 3.498278774934, 1.0),
    "married": (-1.832439820430, -0.013409766574, 1.463171944401, 0.980307912639),
    "nodegree": (0.782512115519, 0.0, 0.576021147144, 1.0),
    "re74": (-2.105246744550, -1.266095058104, 0.030754442676, 0.105469752745),
    "re75": (-2.170266419756, -0.737057274902, 0.014502682485, 0.080974802569),
}


@pytest.fixture(scope="module")
def df():
    return sp.datasets.nsw_dw()


def _fit(df, **kwargs):
    return sp.match(
        df, y="re78", treat="treat", covariates=X, method="nnmatch", **kwargs
    )


@pytest.mark.parametrize("label", sorted(CASES))
def test_estimate_and_se_match_stata(df, label):
    kwargs, est, se = CASES[label]
    res = _fit(df, **kwargs)
    assert res.estimate == pytest.approx(est, rel=RTOL)
    assert res.se == pytest.approx(se, rel=RTOL)


def test_balance_table_matches_tebalance(df):
    table = _fit(df).detail.set_index("variable")
    for name, (raw, matched, ratio_raw, ratio_matched) in BALANCE.items():
        row = table.loc[name]
        # atol: Stata prints 12 decimals, and exact matches give exact zeros.
        assert row["std_diff_raw"] == pytest.approx(raw, abs=1e-11)
        assert row["std_diff_matched"] == pytest.approx(matched, abs=1e-11)
        assert row["var_ratio_raw"] == pytest.approx(ratio_raw, abs=1e-11)
        assert row["var_ratio_matched"] == pytest.approx(ratio_matched, abs=1e-11)


def test_att_matched_sample_is_one_control_per_treated(df):
    info = _fit(df).model_info
    # Stata's tebalance header: 185 treated, 185 control in the matched sample.
    assert info["n_matched_treated"] == pytest.approx(185.0)
    assert info["n_matched_control"] == pytest.approx(185.0)
    assert info["match_weights"].sum() == pytest.approx(185.0)
    assert info["matches_min"] >= 1


def test_inference_is_normal(df):
    res = _fit(df)
    z = res.estimate / res.se
    from scipy import stats

    assert res.pvalue == pytest.approx(2 * stats.norm.sf(abs(z)), rel=1e-12)
    assert res.ci[1] - res.ci[0] == pytest.approx(
        2 * stats.norm.ppf(0.975) * res.se, rel=1e-12
    )


def test_atet_is_an_alias_for_att(df):
    assert _fit(df, estimand="ATET").estimate == _fit(df).estimate


def test_row_order_does_not_matter(df):
    # Ties are kept, so no result may depend on the order of the rows.
    shuffled = df.sample(frac=1.0, random_state=11)
    a, b = _fit(df, n_matches=2), _fit(shuffled, n_matches=2)
    assert a.estimate == pytest.approx(b.estimate, rel=1e-12)
    assert a.se == pytest.approx(b.se, rel=1e-12)


@pytest.mark.parametrize(
    "kwargs, n_unmatched",
    [
        # Stata refuses both as well ("no nearest-neighbor matches within
        # caliper", "no exact matches"). Its osample() also flags controls
        # that no treated unit can use; for the ATT only the treated need a
        # match, so only they are flagged here.
        ({"caliper": 0.25}, 183),
        ({"exact": ["black", "hispanic", "married", "nodegree", "education"]}, 9),
    ],
)
def test_units_without_an_admissible_match_raise(df, kwargs, n_unmatched):
    with pytest.raises(DataInsufficient) as err:
        _fit(df, **kwargs)
    flagged = err.value.unmatched
    assert int(flagged.sum()) == n_unmatched
    assert df.loc[flagged[flagged].index, "treat"].eq(1).all()
    # Dropping the flagged units is the documented way forward.
    res = _fit(df.loc[~flagged.reindex(df.index, fill_value=False)], **kwargs)
    assert np.isfinite(res.estimate) and res.se > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"estimand": "LATE"},
        {"metric": "manhattan"},
        {"vce": "bootstrap"},
        {"n_matches": 0},
        {"vce_nn": 0},
        {"caliper": -1.0},
        {"exact": ["no_such_column"]},
    ],
)
def test_bad_arguments_raise(df, kwargs):
    with pytest.raises(MethodIncompatibility):
        _fit(df, **kwargs)


def test_non_binary_treatment_raises(df):
    bad = df.assign(treat=df["treat"] * 2)
    with pytest.raises(MethodIncompatibility, match="0/1"):
        _fit(bad)
