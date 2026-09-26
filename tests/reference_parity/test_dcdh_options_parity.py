"""dCDH option parity: controls, trends_nonparam, normalized, continuous.

Every number here is the authors' own ``DIDmultiplegtDYN`` 2.3.4 on the
same panel bytes, so the tolerance is machine
precision rather than a budget: these are the same estimators, not merely
the same estimand. The panel and the R call that produced these numbers are
``_generate_dcdh_options_data.py`` and ``_generate_dcdh_options_R.R``.

``trends_lin`` and ``predict_het`` are deliberately absent from the public
surface -- see the module docstring for what is known about the
``trends_lin`` arithmetic. (``continuous`` was added later; its pins are at
the end of this file.)
"""

import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

DATA = pathlib.Path(__file__).resolve().parent / "_fixtures" / "dcdh_options_panel.csv"

# DIDmultiplegtDYN 2.3.4, effects = 4, placebo = 2, on the same CSV bytes.
R_REFERENCE = {
    "plain": {
        "effects": [
            1.025753116654915,
            1.845267144202061,
            2.503789848234482,
            3.393674545139583,
        ],
        "placebos": [0.417271710306834, 1.051664529004348],
    },
    "controls": {
        "effects": [
            0.8421407477496585,
            1.745188847752125,
            2.477024381572633,
            3.465787292881416,
        ],
        "placebos": [0.02952383143028328, 0.3734383242897782],
    },
    "trends_nonparam": {
        "effects": [
            1.056741422341508,
            1.804441232036998,
            2.400768850383448,
            3.39846079434625,
        ],
        "placebos": [0.490365313646355, 1.086767636431232],
    },
    "normalized": {
        "effects": [
            1.025753116654915,
            0.9226335721010306,
            0.8345966160781608,
            0.8484186362848958,
        ],
        "placebos": [0.417271710306834, 0.5258322645021739],
    },
}


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    if not DATA.exists():  # pragma: no cover - fixture ships with the repo
        pytest.skip(f"missing panel fixture: {DATA}")
    return pd.read_csv(DATA)


def _fit(df, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.did_multiplegt_dyn(
            df,
            y="y",
            group="id",
            time="t",
            treatment="d",
            dynamic=3,
            placebo=2,
            se_method="analytic",
            n_boot=0,
            **kwargs,
        )


@pytest.mark.parametrize(
    "label, kwargs",
    [
        ("plain", {}),
        ("controls", {"controls": ["x1", "x2"]}),
        ("trends_nonparam", {"trends_nonparam": ["reg"]}),
        ("normalized", {"normalized": True}),
    ],
)
def test_matches_didmultiplegtdyn(panel, label, kwargs):
    res = _fit(panel, **kwargs)
    delta = res.detail.set_index("horizon")["delta_l"]
    got = np.array([float(delta[h]) for h in range(4)])
    ref = np.array(R_REFERENCE[label]["effects"])
    np.testing.assert_allclose(got, ref, rtol=1e-12, err_msg=f"{label} effects")
    got_p = np.array([float(delta[-1]), float(delta[-2])])
    ref_p = np.array(R_REFERENCE[label]["placebos"])
    np.testing.assert_allclose(got_p, ref_p, rtol=1e-12, err_msg=f"{label} placebos")


def test_controls_shrink_the_placebos_on_a_covariate_driven_trend(panel):
    """The design plants a differential trend the covariates explain."""
    plain = _fit(panel)
    adj = _fit(panel, controls=["x1", "x2"])
    p_plain = abs(float(plain.detail.set_index("horizon")["delta_l"][-1]))
    p_adj = abs(float(adj.detail.set_index("horizon")["delta_l"][-1]))
    assert p_adj < p_plain / 5


def test_normalized_is_the_effect_per_period_of_exposure(panel):
    """Binary absorbing switch: the divisor is the exposure length."""
    plain = _fit(panel).detail.set_index("horizon")["delta_l"]
    norm = _fit(panel, normalized=True).detail.set_index("horizon")["delta_l"]
    for h in range(4):
        assert float(norm[h]) == pytest.approx(float(plain[h]) / (h + 1), rel=1e-12)
    # ... and the placebo at lag l is divided by |l|.
    assert float(norm[-2]) == pytest.approx(float(plain[-2]) / 2, rel=1e-12)


def test_trends_nonparam_only_compares_within_a_cell(panel):
    """A cell with no controls contributes nothing rather than borrowing."""
    df = panel.copy()
    # Give every group its own cell: no two groups can be compared.
    df["solo"] = df["id"].astype(str)
    with pytest.raises(sp.DataInsufficient):
        _fit(df, trends_nonparam=["solo"])


def test_trends_nonparam_rejects_a_time_varying_variable(panel):
    df = panel.copy()
    df["moving"] = df["t"] % 2
    with pytest.raises(sp.MethodIncompatibility, match="time-invariant"):
        _fit(df, trends_nonparam=["moving"])


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"controls": ["nope"]}, "controls column"),
        ({"trends_nonparam": ["nope"]}, "trends_nonparam column"),
    ],
)
def test_unknown_columns_raise(panel, kwargs, match):
    with pytest.raises(sp.MethodIncompatibility, match=match):
        _fit(panel, **kwargs)


def test_options_leave_the_default_path_untouched(panel):
    """No option given: the estimate is the one the module always gave."""
    res = _fit(panel)
    delta = res.detail.set_index("horizon")["delta_l"]
    assert float(delta[0]) == pytest.approx(
        R_REFERENCE["plain"]["effects"][0], rel=1e-12
    )


# Same package, same bytes, on the companion panel whose period-one
# treatments are all distinct (Design Restriction 1(i) fails by design):
#   did_multiplegt_dyn(..., effects = 3, placebo = 1, continuous = k)
CONT_DATA = (
    pathlib.Path(__file__).resolve().parent / "_fixtures" / "dcdh_continuous_panel.csv"
)
R_CONTINUOUS = {
    1: [0.8542098191688396, 1.703596124713585, 2.727003514560116],
    2: [0.8111434125838702, 1.867598544412068, 2.77300749685663],
}


@pytest.fixture(scope="module")
def continuous_panel() -> pd.DataFrame:
    if not CONT_DATA.exists():  # pragma: no cover - fixture ships with the repo
        pytest.skip(f"missing panel fixture: {CONT_DATA}")
    return pd.read_csv(CONT_DATA)


@pytest.mark.parametrize("degree", [1, 2])
def test_continuous_matches_didmultiplegtdyn(continuous_panel, degree):
    """The polynomial is fitted per period, which is what reproduces it.

    A single pooled polynomial with time effects -- the other reading of
    "the status-quo outcome evolution is a polynomial in the period-one
    treatment" -- lands 8 to 16 percent away.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.did_multiplegt_dyn(
            continuous_panel,
            y="y",
            group="id",
            time="t",
            treatment="d",
            dynamic=2,
            placebo=1,
            se_method="analytic",
            n_boot=0,
            continuous=degree,
        )
    delta = res.detail.set_index("horizon")["delta_l"]
    got = np.array([float(delta[h]) for h in range(3)])
    np.testing.assert_allclose(got, np.array(R_CONTINUOUS[degree]), rtol=1e-12)


def test_continuous_accepts_a_non_binary_treatment(continuous_panel):
    """That check is exactly what continuous= exists to relax."""
    with pytest.raises(ValueError, match="binary"):
        sp.did_multiplegt_dyn(
            continuous_panel, y="y", group="id", time="t", treatment="d", dynamic=1
        )


@pytest.mark.parametrize("degree", [0, -1, 1.5, True])
def test_continuous_degree_must_be_a_positive_integer(continuous_panel, degree):
    with pytest.raises(sp.MethodIncompatibility, match="positive integer"):
        sp.did_multiplegt_dyn(
            continuous_panel,
            y="y",
            group="id",
            time="t",
            treatment="d",
            dynamic=1,
            continuous=degree,
        )


def test_group_effects_average_to_the_reported_effect(panel):
    """The dependent variable a heterogeneity regression would need.

    predict_het is not implemented -- see the module docstring for what the
    reference reports and what these give -- but the group-level effects it
    would regress are exposed, and they are not approximate: each horizon's
    effects average to that horizon's delta_l exactly, over the switcher
    count the reference also reports.
    """
    res = _fit(panel)
    delta = res.detail.set_index("horizon")["delta_l"]
    effects = res.model_info["group_effects"]
    for h, n in zip(range(4), (51, 39, 29, 16)):
        assert len(effects[h]) == n
        assert float(effects[h].mean()) == pytest.approx(float(delta[h]), rel=1e-12)
