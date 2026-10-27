"""``sp.fuzzy_did``: identities, known-truth designs and edge cases.

Stata parity lives in ``tests/reference_parity/test_fuzzy_did_stata_parity.py``.
Here:

* the Wald-DID is the 2SLS coefficient of the treatment instrumented by
  group x period with group and period dummies (exact identity, also on a
  grouped panel with group fixed effects as in the wealth-tax design), and
  its analytic standard error is the 2SLS HC0 / CR0 one;
* three data-generating processes with known LATE show which estimator is
  consistent under which assumption (de Chaisemartin and D'Haultfoeuille
  2018): all three when effects are stable and homogeneous and the control
  group's treatment rate does not move; Wald-TC and Wald-CIC, not Wald-DID,
  when the effect changes over time; Wald-DID, not Wald-TC, when the control
  group's new takers are selected on the untreated outcome level;
* input errors, coding equivalences, reproducibility.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import IdentificationFailure, MethodIncompatibility

FX = Path(__file__).resolve().parent / "reference_parity" / "_fixtures"
fd_mod = importlib.import_module("statspai.did.fuzzy_did")


@pytest.fixture(scope="module")
def d2():
    df = pd.read_csv(FX / "fuzzydid_2x2.csv")
    df["gxt"] = df["g"] * df["t"]
    return df


# ---------------------------------------------------------------------------
# Wald-DID = 2SLS
# ---------------------------------------------------------------------------


def test_wald_did_is_the_2sls_coefficient_and_hc0_se(d2):
    fz = sp.fuzzy_did(d2, "y", "g", "t", "d", estimators=("did",), se_method="analytic")
    iv = sp.iv("y ~ g + t + (d ~ gxt)", data=d2, robust="hc0")
    # just-identified, saturated in the four cells: the IV coefficient is
    # the ratio of the two DiDs and its residuals are the influence function
    assert fz.estimate == pytest.approx(iv.params["d"], abs=1e-12)
    assert fz.se == pytest.approx(iv.std_errors["d"], rel=1e-10)


def test_wald_did_cluster_se_is_cr0_2sls(d2):
    fz = sp.fuzzy_did(
        d2,
        "y",
        "g",
        "t",
        "d",
        estimators=("did",),
        se_method="analytic",
        cluster="cl",
    )
    iv = sp.iv("y ~ g + t + (d ~ gxt)", data=d2, cluster="cl")
    n, k, g = len(d2), 4, d2["cl"].nunique()
    # sp.iv's clustered SE carries the Stata factor G/(G-1) (N-1)/(N-K)
    factor = np.sqrt(g / (g - 1) * (n - 1) / (n - k))
    assert fz.estimate == pytest.approx(iv.params["d"], abs=1e-12)
    assert fz.se * factor == pytest.approx(iv.std_errors["d"], rel=1e-10)


def test_wald_did_equals_grouped_panel_2sls_with_group_fixed_effects():
    """The paper's group-level 2SLS: rate on log net-of-tax rate, instrument
    treat x post, group and post dummies. On a balanced grouped panel the
    coefficient is the Wald-DID of the group-period rows."""
    rng = np.random.default_rng(5)
    groups = 24
    rows = []
    for j in range(groups):
        top = int(j >= 18)  # top groups lose the tax
        for post in (0, 1):
            tau = 0.015 * (1 - top * post) + rng.uniform(0, 0.004)
            lnnet = np.log(1 - tau)
            rate = 0.3 + 0.01 * j + 0.02 * post + 1.5 * lnnet + rng.normal(0, 0.01)
            rows.append((j, top, post, lnnet, rate))
    gp = pd.DataFrame(rows, columns=["grp", "top", "post", "lnnet", "rate"])
    gp["top_post"] = gp["top"] * gp["post"]
    dummies = pd.get_dummies(gp["grp"], prefix="g", drop_first=True, dtype=float)
    gpx = pd.concat([gp, dummies], axis=1)
    rhs = " + ".join(["post"] + list(dummies.columns))
    iv = sp.iv(f"rate ~ {rhs} + (lnnet ~ top_post)", data=gpx)
    fz = sp.fuzzy_did(
        gp, "rate", "top", "post", "lnnet", estimators=("did",), se_method="none"
    )
    assert fz.estimate == pytest.approx(iv.params["lnnet"], rel=1e-10)


# ---------------------------------------------------------------------------
# Known truth: which estimator is consistent when
# ---------------------------------------------------------------------------


def _types_dgp(shares, beta, level_shift, n_per_cell, seed):
    """Repeated cross-sections with exact type shares in each cell.

    ``shares[g]`` maps type -> (share, D at t=0, D at t=1). Untreated
    outcome alpha_g + 0.5 t + level_shift[type] + N(0, 1); effect beta[t]
    on everybody treated in period t. Only the noise is random.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for g, types in shares.items():
        for t in (0, 1):
            for typ, (share, d0, d1) in types.items():
                n = int(round(share * n_per_cell))
                d = d1 if t == 1 else d0
                y0 = 0.4 * g + 0.5 * t + level_shift.get(typ, 0.0)
                y = y0 + beta[t] * d + rng.normal(size=n)
                frames.append(pd.DataFrame({"g": g, "t": t, "d": d, "y": y}))
    return pd.concat(frames, ignore_index=True)


#: control: never-takers 0.8, always-takers 0.2 (rate stable at 0.2);
#: treated group: never 0.3, always 0.25, switchers 0.45 (0.25 -> 0.7)
STABLE = {
    0: {"N": (0.8, 0, 0), "A": (0.2, 1, 1)},
    1: {"N": (0.3, 0, 0), "A": (0.25, 1, 1), "S": (0.45, 0, 1)},
}
#: control rate moves 0.2 -> 0.35 through new takers C
MOVING = {
    0: {"N": (0.65, 0, 0), "A": (0.2, 1, 1), "C": (0.15, 0, 1)},
    1: {"N": (0.3, 0, 0), "A": (0.25, 1, 1), "S": (0.45, 0, 1)},
}
N_CELL = 200_000


def _fit(df):
    r = sp.fuzzy_did(
        df, "y", "g", "t", "d", estimators=("did", "tc"), se_method="analytic"
    )
    cic = sp.fuzzy_did(
        df, "y", "g", "t", "d", estimators=("cic",), se_method="none"
    ).estimate
    est = dict(zip(r.detail["estimator"], r.detail["estimate"]))
    se = dict(zip(r.detail["estimator"], r.detail["se"]))
    return est, se, cic


def test_all_three_consistent_with_stable_homogeneous_effects():
    df = _types_dgp(STABLE, {0: 2.0, 1: 2.0}, {}, N_CELL, 1)
    est, se, cic = _fit(df)
    for k in ("W_DID", "W_TC"):
        # 4 analytic SEs (se ~ 0.007 here)
        assert abs(est[k] - 2.0) < 4 * se[k], k
    # CIC: same LATE; its sampling SE is of the TC's order
    assert abs(cic - 2.0) < 4 * se["W_TC"]


def test_effect_changing_over_time_biases_wald_did_not_tc_cic():
    # effect 1 in period 0, 2 in period 1: LATE of the switchers = 2.
    # Wald-DID limit (paper, Theorem 3.1 fails without stable effects):
    # [(0.7*2 - 0.25*1) - (0.35*2 - 0.2*1)] / (0.45 - 0.15) = 0.65 / 0.3
    df = _types_dgp(MOVING, {0: 1.0, 1: 2.0}, {}, N_CELL, 2)
    est, se, cic = _fit(df)
    assert abs(est["W_TC"] - 2.0) < 4 * se["W_TC"]
    assert abs(cic - 2.0) < 4 * se["W_TC"]
    did_limit = 0.65 / 0.3
    assert abs(est["W_DID"] - did_limit) < 4 * se["W_DID"]
    assert abs(did_limit - 2.0) > 8 * se["W_DID"]  # the bias is detectable


def test_selected_new_takers_in_control_bias_tc_not_did():
    # effect 2 always; the control group's new takers C have untreated
    # outcomes 3 higher, so E(Y|G=0, D=d) moves with the composition of
    # each treatment arm. Wald-DID (group means) is unaffected; Wald-TC's
    # delta_d inherits the composition change:
    # delta_0 = 0.5 - 3 * 0.15 / 0.8, delta_1 = 0.5 + 3 * 0.15 / 0.35,
    # TC limit = (0.5 + 2 * 0.45 - (0.75 delta_0 + 0.25 delta_1)) / 0.45
    df = _types_dgp(MOVING, {0: 2.0, 1: 2.0}, {"C": 3.0}, N_CELL, 3)
    est, se, _ = _fit(df)
    assert abs(est["W_DID"] - 2.0) < 4 * se["W_DID"]
    d0 = 0.5 - 3 * 0.15 / 0.8
    d1 = 0.5 + 3 * 0.15 / 0.35
    tc_limit = (0.5 + 2 * 0.45 - (0.75 * d0 + 0.25 * d1)) / 0.45
    assert abs(est["W_TC"] - tc_limit) < 4 * se["W_TC"]
    assert abs(tc_limit - 2.0) > 8 * se["W_TC"]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def test_analytic_tc_se_matches_the_bootstrap(d2):
    an = sp.fuzzy_did(
        d2, "y", "g", "t", "d", estimators=("did", "tc"), se_method="analytic"
    )
    bs = sp.fuzzy_did(
        d2, "y", "g", "t", "d", estimators=("did", "tc"), n_boot=400, seed=4
    )
    # same estimand's delta-method and bootstrap SDs; 400 draws carry ~3.5%
    # Monte Carlo error, the screen allows 15%
    np.testing.assert_allclose(an.detail["se"], bs.detail["se"], rtol=0.15)


def test_analytic_eqtest_uses_the_joint_influence_function(d2):
    r = sp.fuzzy_did(
        d2,
        "y",
        "g",
        "t",
        "d",
        estimators=("did", "tc"),
        se_method="analytic",
        eqtest=True,
    )
    eq = r.model_info["eqtest"].iloc[0]
    est = r.detail.set_index("estimator")["estimate"]
    assert eq["difference"] == pytest.approx(est["W_DID"] - est["W_TC"], abs=1e-15)
    # var(a - b) = var a + var b - 2 cov(a, b) and |cov| <= sd_a sd_b
    se = r.detail.set_index("estimator")["se"]
    assert abs(se["W_DID"] - se["W_TC"]) - 1e-12 <= eq["se"]
    assert eq["se"] <= se["W_DID"] + se["W_TC"] + 1e-12


def test_bootstrap_is_reproducible_with_a_seed(d2):
    a = sp.fuzzy_did(d2, "y", "g", "t", "d", n_boot=30, seed=9)
    b = sp.fuzzy_did(d2, "y", "g", "t", "d", n_boot=30, seed=9)
    pd.testing.assert_frame_equal(a.detail, b.detail)
    assert a.model_info["ci_type"] == "percentile"


# ---------------------------------------------------------------------------
# Coding equivalences
# ---------------------------------------------------------------------------


def test_one_group_variable_equals_the_two_variable_coding(d2):
    one = sp.fuzzy_did(d2, "y", "g", "t", "d", se_method="none")
    two_g = d2.assign(
        gt=np.where(d2.t == 1, d2.g, np.nan), gf=np.where(d2.t == 0, d2.g, np.nan)
    )
    two = sp.fuzzy_did(
        two_g, "y", "gt", "t", "d", group_change_next="gf", se_method="none"
    )
    np.testing.assert_allclose(
        one.detail["estimate"], two.detail["estimate"], rtol=0, atol=1e-13
    )


def test_row_order_and_period_labels_do_not_matter(d2):
    base = sp.fuzzy_did(d2, "y", "g", "t", "d", se_method="none")
    shuffled = d2.sample(frac=1.0, random_state=1).reset_index(drop=True)
    relabel = shuffled.assign(t=shuffled["t"].map({0: 2007, 1: 2009}))
    other = sp.fuzzy_did(relabel, "y", "g", "t", "d", se_method="none")
    np.testing.assert_allclose(
        base.detail["estimate"], other.detail["estimate"], rtol=0, atol=1e-12
    )


def test_decreasing_group_is_the_time_reversed_increasing_group(d2):
    """A group whose treatment falls, coded -1, gives the Wald-DID of the
    same data read backwards in time (numerator and denominator flip sign)."""
    fwd = sp.fuzzy_did(
        d2, "y", "g", "t", "d", estimators=("did",), se_method="none"
    ).estimate
    rev = d2.assign(t=1 - d2["t"])
    rev = rev.assign(
        gt=np.where(rev.t == 1, -rev.g, np.nan),
        gf=np.where(rev.t == 0, -rev.g, np.nan),
    )
    back = sp.fuzzy_did(
        rev,
        "y",
        "gt",
        "t",
        "d",
        group_change_next="gf",
        estimators=("did",),
        se_method="none",
    ).estimate
    assert back == pytest.approx(fwd, abs=1e-12)


def test_categorical_covariate_column_equals_c_notation(d2):
    a = sp.fuzzy_did(
        d2,
        "y",
        "g",
        "t",
        "d",
        covariates=["x1", "C(x2)"],
        estimators=("did", "tc"),
        se_method="none",
    )
    b = sp.fuzzy_did(
        d2.assign(x2=d2["x2"].astype(str)),
        "y",
        "g",
        "t",
        "d",
        covariates=["x1", "x2"],
        estimators=("did", "tc"),
        se_method="none",
    )
    np.testing.assert_allclose(a.detail["estimate"], b.detail["estimate"], atol=1e-12)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_no_stable_group_is_refused(d2):
    df = d2.assign(gt=np.where(d2.t == 1, 1, np.nan), gf=np.where(d2.t == 0, 1, np.nan))
    with pytest.raises(IdentificationFailure, match="stable"):
        sp.fuzzy_did(df, "y", "gt", "t", "d", group_change_next="gf", se_method="none")


def test_tc_treatment_value_absent_from_controls_points_to_newcateg():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"g": np.repeat([0, 1], n), "t": np.tile([0, 1], n)})
    # value 2 appears only in the switching group
    df["d"] = np.where(
        df.g == 1, rng.integers(0, 3, len(df)), rng.integers(0, 2, len(df))
    )
    df["y"] = df["d"] + rng.normal(size=len(df))
    with pytest.raises(IdentificationFailure, match="newcateg") as ei:
        sp.fuzzy_did(df, "y", "g", "t", "d", estimators=("tc",), se_method="none")
    assert "newcateg" in ei.value.recovery_hint
    # pooling 1 and 2 makes it estimable
    r = sp.fuzzy_did(
        df, "y", "g", "t", "d", estimators=("tc",), se_method="none", newcateg=[0, 2]
    )
    assert np.isfinite(r.estimate)


def test_one_group_variable_needs_two_periods(d2):
    df = pd.concat([d2, d2.assign(t=2)], ignore_index=True)
    with pytest.raises(MethodIncompatibility, match="exactly two periods"):
        sp.fuzzy_did(df, "y", "g", "t", "d", se_method="none")


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"estimators": ("did", "lqte")}, "estimators"),
        ({"covariates": ["x1"]}, "covariates"),  # cic requested by default
        ({"se_method": "analytic"}, "analytic"),  # cic requested by default
        ({"se_method": "delta"}, "se_method"),
        ({"modelx": ("ols", "tobit")}, "modelx"),
        ({"cic_quantile": "round"}, "cic_quantile"),
        ({"eqtest": True, "numerator": True}, "eqtest"),
        ({"newcateg": [0]}, "newcateg"),
    ],
)
def test_bad_options_are_refused(d2, kwargs, match):
    with pytest.raises(MethodIncompatibility, match=match):
        sp.fuzzy_did(d2, "y", "g", "t", "d", **kwargs)


def test_bad_group_codes_are_refused(d2):
    with pytest.raises(MethodIncompatibility, match="values in"):
        sp.fuzzy_did(d2.assign(g=d2["g"] * 2), "y", "g", "t", "d", se_method="none")


def test_numerator_needs_one_comparison():
    df = pd.read_csv(FX / "fuzzydid_multi.csv")
    with pytest.raises(MethodIncompatibility, match="single comparison"):
        sp.fuzzy_did(
            df,
            "y",
            "gt",
            "t",
            "d",
            group_change_next="gf",
            numerator=True,
            se_method="none",
        )


def test_logit_on_a_non_binary_outcome_is_refused(d2):
    with pytest.raises(MethodIncompatibility, match="0/1"):
        sp.fuzzy_did(
            d2,
            "y",
            "g",
            "t",
            "d",
            covariates=["x1"],
            modelx=("logit", "ols"),
            estimators=("did",),
            se_method="none",
        )


def test_cite_returns_the_paper(d2):
    r = sp.fuzzy_did(d2, "y", "g", "t", "d", estimators=("did",), se_method="none")
    assert "dechaisemartin2018fuzzy" in r.cite()
    assert isinstance(r, fd_mod.FuzzyDiDResult)


def test_no_first_stage_is_refused(d2):
    # the treatment rate moves identically in both groups: DID of D is 0
    df = d2.assign(d=d2["t"])
    with pytest.raises(IdentificationFailure, match="denominator is zero"):
        sp.fuzzy_did(df, "y", "g", "t", "d", estimators=("did",), se_method="none")


def test_switching_group_coded_in_one_period_only_is_refused(d2):
    # gf says "increases next period" but no row says "increased" after
    df = d2.assign(
        gt=np.where(d2.t == 1, 0, np.nan), gf=np.where(d2.t == 0, d2.g, np.nan)
    )
    with pytest.raises(IdentificationFailure, match="only one period"):
        sp.fuzzy_did(df, "y", "gt", "t", "d", group_change_next="gf", se_method="none")
