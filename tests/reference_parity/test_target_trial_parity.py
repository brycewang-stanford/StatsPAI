"""Frozen-formula parity: target-trial-emulation family analytical identities.

Analytical evidence tier — no external reference implementation exists for
this family, so each test asserts an exact hand-derivable identity:

* ``sp.target_trial_protocol`` is a spec constructor: every field must
  round-trip unchanged through the dataclass and ``to_dict()``, and the
  documented validation rules (>= 2 arms, valid causal contrast) must raise.
* ``sp.target_trial_emulate`` with unit weights reduces to the difference of
  outcome means among the eligible subset: estimate, SE, and 95% CI follow
  the closed-form IPW formulas m1 - m0, sqrt(v1 + v0), est +/- 1.96*SE, which
  we recompute by hand to machine precision. DGP: randomized binary treatment
  with a linear outcome Y = 1 + 2*A + eps, eps ~ N(0,1), so the ITT truth is
  2.0 and the diff-in-means estimator has SE ~ sqrt(2/n_arm) ~ 0.05; a 0.2
  tolerance (~4 SE) on recovery is conservative.
* ``sp.clone_censor_weight`` on fully deterministic treatment histories has
  exactly countable clone rows: a clone is dropped if inconsistent at the
  first period, censored at the first deviation otherwise. With an empty
  ``censor_covariates`` list the IPC model is skipped, so every weight is
  exactly 1.0 and the weighted per-protocol contrast of a deterministic
  outcome (Y = 10 + 3*treat) equals 3.0 exactly.
* ``sp.immortal_time_check`` flags exactly the ids with
  treatment_start < eligibility_time (NaN never-treated ids are not
  flagged) — an exact set identity on constructed data.
* ``sp.target_trial_checklist`` / ``sp.target_trial_report`` are
  deterministic string builders: we assert content identities (exact
  estimate string, 21 checklist rows, AUTO/TODO split, and that
  ``fmt="target"`` reproduces the checklist verbatim).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def proto():
    return sp.target_trial_protocol(
        eligibility="age >= 40",
        treatment_strategies=["initiate statin at t0", "no statin"],
        assignment="observational emulation",
        time_zero="first eligible visit",
        followup_end="min(event, loss, 5 years)",
        outcome="incident MI within 5y",
        causal_contrast="ITT",
        analysis_plan="IPW difference in means",
        baseline_covariates=["age"],
        notes="parity test protocol",
    )


@pytest.fixture(scope="module")
def rct_df():
    # RCT-like DGP with KNOWN ITT effect 2.0: randomized A, linear outcome.
    rng = np.random.default_rng(42)
    n = 2000
    df = pd.DataFrame(
        {
            "age": rng.uniform(30, 80, n),
            "statin": rng.integers(0, 2, n),
        }
    )
    df["mi"] = 1.0 + 2.0 * df["statin"] + rng.normal(0.0, 1.0, n)
    return df


@pytest.fixture(scope="module")
def emulated(proto, rct_df):
    return sp.target_trial_emulate(
        proto, rct_df, outcome_col="mi", treatment_col="statin"
    )


@pytest.fixture(scope="module")
def ccw_panel():
    # Deterministic treatment histories over T=4 periods:
    #   ids 0-3: always treated  (1,1,1,1)
    #   ids 4-6: never treated   (0,0,0,0)
    #   ids 7-9: deviate at t=2  (1,1,0,0)
    rng = np.random.default_rng(7)
    paths = {}
    for sid in range(4):
        paths[sid] = [1, 1, 1, 1]
    for sid in range(4, 7):
        paths[sid] = [0, 0, 0, 0]
    for sid in range(7, 10):
        paths[sid] = [1, 1, 0, 0]
    rows = []
    for sid, path in paths.items():
        for t, a in enumerate(path):
            rows.append(
                {
                    "id": sid,
                    "time": t,
                    "treat": a,
                    "age": 50.0 + rng.normal(),  # pure-noise covariate
                    "y_out": 10.0 + 3.0 * a,  # deterministic outcome
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def ccw_strategies():
    return {
        "always_treat": lambda g: g["treat"].to_numpy() == 1,
        "never_treat": lambda g: g["treat"].to_numpy() == 0,
    }


@pytest.fixture(scope="module")
def ccw_unit_weights(ccw_panel, ccw_strategies):
    # Empty censor_covariates -> IPC model skipped -> weights exactly 1.
    return sp.clone_censor_weight(
        ccw_panel,
        id_col="id",
        time_col="time",
        treatment_col="treat",
        strategies=ccw_strategies,
        censor_covariates=[],
    )


# --------------------------------------------------------------------------
# sp.target_trial_protocol — spec round-trip identity
# --------------------------------------------------------------------------


def test_protocol_fields_round_trip(proto):
    assert proto.eligibility == "age >= 40"
    assert proto.treatment_strategies == ["initiate statin at t0", "no statin"]
    assert proto.assignment == "observational emulation"
    assert proto.time_zero == "first eligible visit"
    assert proto.followup_end == "min(event, loss, 5 years)"
    assert proto.outcome == "incident MI within 5y"
    assert proto.causal_contrast == "ITT"
    assert proto.analysis_plan == "IPW difference in means"
    assert proto.baseline_covariates == ["age"]
    assert proto.time_varying_covariates == []
    assert proto.notes == "parity test protocol"


def test_protocol_to_dict_round_trip(proto):
    d = proto.to_dict()
    assert d["eligibility"] == "age >= 40"
    assert d["treatment_strategies"] == ["initiate statin at t0", "no statin"]
    assert d["causal_contrast"] == "ITT"
    # to_dict must expose exactly the 11 protocol fields.
    assert set(d) == {
        "eligibility",
        "treatment_strategies",
        "assignment",
        "time_zero",
        "followup_end",
        "outcome",
        "causal_contrast",
        "analysis_plan",
        "baseline_covariates",
        "time_varying_covariates",
        "notes",
    }


def test_protocol_summary_header_and_content(proto):
    s = proto.summary()
    assert s.splitlines()[0] == "Target Trial Protocol"
    assert "6. Causal contrast: ITT" in s
    assert "4a. Time zero: first eligible visit" in s


def test_protocol_validation_raises():
    with pytest.raises(ValueError):
        sp.target_trial_protocol(
            eligibility="x > 0",
            treatment_strategies=["only one arm"],
            assignment="randomization",
            time_zero="t0",
            followup_end="end",
            outcome="y",
        )
    with pytest.raises(ValueError):
        sp.target_trial_protocol(
            eligibility="x > 0",
            treatment_strategies=["a", "b"],
            assignment="randomization",
            time_zero="t0",
            followup_end="end",
            outcome="y",
            causal_contrast="not-a-contrast",
        )


# --------------------------------------------------------------------------
# sp.target_trial_emulate — exact diff-in-means identity + ITT recovery
# --------------------------------------------------------------------------


def test_emulate_eligibility_counts_exact(emulated, rct_df):
    n_eligible_hand = int((rct_df["age"] >= 40).sum())
    assert emulated.n_eligible == n_eligible_hand
    assert emulated.n_eligible + emulated.n_excluded_immortal == len(rct_df)


def test_emulate_estimate_equals_diff_in_means_exactly(emulated, rct_df):
    # With unit weights the IPW estimator collapses to the difference of
    # outcome means among the eligible subset — machine-precision identity.
    elig = rct_df.query("age >= 40")
    a = elig["statin"].to_numpy(dtype=float)
    y = elig["mi"].to_numpy(dtype=float)
    m1 = y[a == 1].mean()
    m0 = y[a == 0].mean()
    assert emulated.estimate == pytest.approx(m1 - m0, rel=1e-12)


def test_emulate_se_and_ci_follow_frozen_formula(emulated, rct_df):
    # SE^2 = sum(w a (y-m1)^2)/sum(w a)^2 + sum(w (1-a) (y-m0)^2)/sum(w (1-a))^2
    # with w = 1; CI = est +/- 1.96 SE.
    elig = rct_df.query("age >= 40")
    a = elig["statin"].to_numpy(dtype=float)
    y = elig["mi"].to_numpy(dtype=float)
    m1 = y[a == 1].mean()
    m0 = y[a == 0].mean()
    v1 = ((y[a == 1] - m1) ** 2).sum() / (a == 1).sum() ** 2
    v0 = ((y[a == 0] - m0) ** 2).sum() / (a == 0).sum() ** 2
    se_hand = float(np.sqrt(v1 + v0))
    assert emulated.se == pytest.approx(se_hand, rel=1e-12)
    assert emulated.ci[0] == pytest.approx(emulated.estimate - 1.96 * se_hand, rel=1e-12)
    assert emulated.ci[1] == pytest.approx(emulated.estimate + 1.96 * se_hand, rel=1e-12)


def test_emulate_recovers_known_itt_effect(emulated):
    # True ITT effect is 2.0 by construction. Per-arm n ~ 800, outcome sd = 1
    # -> SE(diff) ~ sqrt(2/800) ~ 0.05; 0.2 is a ~4-sigma tolerance.
    assert emulated.estimate == pytest.approx(2.0, abs=0.2)
    assert emulated.ci[0] <= emulated.estimate <= emulated.ci[1]


def test_emulate_unit_weights_by_default(emulated):
    w = np.asarray(emulated.weights, dtype=float)
    assert len(w) == emulated.n_eligible
    assert np.all(w == 1.0)


# --------------------------------------------------------------------------
# sp.clone_censor_weight — exact clone/censor bookkeeping identities
# --------------------------------------------------------------------------


def test_ccw_counts_exact(ccw_unit_weights):
    # Hand count of surviving clone-rows:
    #   always_treat: ids 0-3 keep 4 rows each (16); ids 4-6 inconsistent at
    #     t=0 -> dropped; ids 7-9 keep rows t=0,1 then censored (6). Total 22.
    #   never_treat:  ids 4-6 keep 4 rows each (12); all others inconsistent
    #     at t=0 -> dropped. Total 12.
    res = ccw_unit_weights
    assert res.n_originals == 10
    assert res.strategies == ["always_treat", "never_treat"]
    assert res.n_clones == 34
    counts = res.cloned_data["_strategy"].value_counts().to_dict()
    assert counts == {"always_treat": 22, "never_treat": 12}


def test_ccw_censoring_flags_exact(ccw_unit_weights):
    df = ccw_unit_weights.cloned_data
    always = df[df["_strategy"] == "always_treat"]
    never = df[df["_strategy"] == "never_treat"]
    # Compliant histories are never artificially censored.
    assert set(always.loc[always["_censored"] == 0, "id"]) == {0, 1, 2, 3}
    # Deviators (ids 7-9) are censored, keeping exactly t = 0, 1.
    censored = always[always["_censored"] == 1]
    assert set(censored["id"]) == {7, 8, 9}
    assert sorted(censored["time"].unique()) == [0, 1]
    assert (never["_censored"] == 0).all()
    assert set(never["id"]) == {4, 5, 6}


def test_ccw_weights_exactly_one_without_censor_model(ccw_unit_weights):
    # censor_covariates=[] skips the IPC model, so every weight is 1.0.
    w = ccw_unit_weights.cloned_data["_ipcw"].to_numpy(dtype=float)
    assert np.all(w == 1.0)
    assert ccw_unit_weights.weights_summary == {"mean": 1.0, "max": 1.0, "min": 1.0}


def test_ccw_weighted_per_protocol_contrast_exact(ccw_unit_weights):
    # Deterministic outcome Y = 10 + 3*treat: every surviving always_treat
    # row has Y = 13, every never_treat row has Y = 10, so the weighted
    # per-protocol contrast is exactly 3.0.
    df = ccw_unit_weights.cloned_data
    means = df.groupby("_strategy").apply(
        lambda g: np.average(g["y_out"], weights=g["_ipcw"]),
        include_groups=False,
    )
    assert means["always_treat"] == pytest.approx(13.0, rel=1e-12)
    assert means["never_treat"] == pytest.approx(10.0, rel=1e-12)
    assert means["always_treat"] - means["never_treat"] == pytest.approx(
        3.0, rel=1e-12
    )


def test_ccw_fitted_weights_positive_finite(ccw_panel, ccw_strategies):
    # With a noise covariate the pooled-logistic IPC model actually fits;
    # the censoring structure (and thus n_clones) is unchanged, and all
    # stabilized weights must be strictly positive and finite.
    res = sp.clone_censor_weight(
        ccw_panel,
        id_col="id",
        time_col="time",
        treatment_col="treat",
        strategies=ccw_strategies,
        censor_covariates=["age"],
        stabilize=True,
    )
    assert res.n_clones == 34
    w = res.cloned_data["_ipcw"].to_numpy(dtype=float)
    assert np.all(np.isfinite(w))
    assert np.all(w > 0)


# --------------------------------------------------------------------------
# sp.immortal_time_check — exact flagged-id identity
# --------------------------------------------------------------------------


def test_immortal_time_check_flags_exact_ids():
    # Flag rule: treatment_start < eligibility_time. Constructed truth:
    # id 2 (0 < 1) and id 5 (1 < 4) flagged; id 3 has tx == elig (not
    # strict, unflagged); id 4 never treated (NaN, unflagged).
    data = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "fu_time": [12.0, 8.0, 24.0, 6.0, 10.0, 9.0],
            "tx_start": [5.0, 0.0, 2.0, np.nan, 1.0, 3.0],
            "elig_time": [0.0, 1.0, 2.0, 0.0, 4.0, 0.0],
        }
    )
    diag = sp.immortal_time_check(
        data,
        id_col="id",
        time_col="fu_time",
        treatment_start_col="tx_start",
        eligibility_time_col="elig_time",
    )
    assert diag.n_total == 6
    assert diag.n_flagged == 2
    assert diag.flagged_ids == [2, 5]
    assert diag.fraction_flagged == pytest.approx(2.0 / 6.0, rel=1e-12)
    assert "immortal time bias" in diag.explanation


def test_immortal_time_check_clean_data_flags_none():
    data = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "fu_time": [10.0, 10.0, 10.0],
            "tx_start": [0.0, 2.0, np.nan],
            "elig_time": [0.0, 0.0, 0.0],
        }
    )
    diag = sp.immortal_time_check(
        data,
        id_col="id",
        time_col="fu_time",
        treatment_start_col="tx_start",
        eligibility_time_col="elig_time",
    )
    assert diag.n_flagged == 0
    assert diag.flagged_ids == []
    assert diag.fraction_flagged == 0.0


# --------------------------------------------------------------------------
# sp.target_trial_checklist / sp.target_trial_report — content identities
# --------------------------------------------------------------------------


def test_checklist_structure_and_estimate_string(emulated):
    chk = sp.target_trial_checklist(emulated)
    lines = chk.splitlines()
    assert lines[0] == "# TARGET Statement — 21-item Reporting Checklist"
    # Exactly 21 numbered rows in the markdown table.
    n_rows = sum(
        1 for ln in lines if ln.startswith("| ") and ln.split("|")[1].strip().isdigit()
    )
    assert n_rows == 21
    # 12 items are auto-fillable from protocol+result; the other 9 are TODO.
    assert chk.count("`[AUTO]`") == 12
    assert chk.count("`[TODO]`") == 9
    # The item-18 estimate string is a frozen format of the result fields.
    lo, hi = emulated.ci
    est = (
        f"{emulated.estimate:+.4f} (95% CI [{lo:+.4f}, {hi:+.4f}], "
        f"SE {emulated.se:.4f})"
    )
    assert est in chk
    # Protocol fields propagate verbatim.
    assert "age >= 40" in chk
    assert f"n eligible = {emulated.n_eligible}" in chk


def test_report_markdown_content_identity(emulated):
    rep = sp.target_trial_report(emulated, fmt="markdown")
    assert "## Methods: Target Trial Specification" in rep
    assert "## Results" in rep
    assert "| **Eligibility** | age >= 40 |" in rep
    assert f"estimate = {emulated.estimate:+.4f}" in rep
    assert f"{emulated.n_eligible} met eligibility at time zero" in rep
    assert f"SE = {emulated.se:.4f}" in rep


def test_report_target_fmt_equals_checklist(emulated):
    # fmt="target" must reproduce the markdown checklist verbatim.
    assert sp.target_trial_report(emulated, fmt="target") == sp.target_trial_checklist(
        emulated, fmt="markdown"
    )


def test_report_invalid_fmt_raises(emulated):
    with pytest.raises(ValueError):
        sp.target_trial_report(emulated, fmt="html")
