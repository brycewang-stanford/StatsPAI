"""Sprint-1 tests: Target Trial Emulation + IPCW."""
import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ---------- protocol ----------

def test_protocol_basic():
    p = sp.target_trial.protocol(
        eligibility="age >= 50",
        treatment_strategies=["statin", "no statin"],
        assignment="observational emulation",
        time_zero="date of diagnosis",
        followup_end="min(death, loss, 5y)",
        outcome="MI",
        causal_contrast="per-protocol",
        baseline_covariates=["age", "sex"],
    )
    assert p.outcome == "MI"
    assert p.causal_contrast == "per-protocol"
    assert "age >= 50" in p.summary()
    assert "Target Trial Protocol" in p.summary()


def test_protocol_validates_strategies():
    with pytest.raises(ValueError, match="at least 2 arms"):
        sp.target_trial.protocol(
            eligibility="x > 0",
            treatment_strategies=["only-one"],
            assignment="randomization",
            time_zero="t0",
            followup_end="end",
            outcome="y",
        )


def test_protocol_rejects_unknown_contrast():
    with pytest.raises(ValueError, match="causal_contrast"):
        sp.target_trial.protocol(
            eligibility="x > 0",
            treatment_strategies=["a", "b"],
            assignment="randomization",
            time_zero="t0",
            followup_end="end",
            outcome="y",
            causal_contrast="magic",
        )


# ---------- emulate ----------

def test_emulate_query_eligibility_recovers_treatment_effect():
    rng = np.random.default_rng(0)
    n = 2000
    age = rng.normal(55, 8, n)
    tx = rng.binomial(1, 0.5, n)
    y = 2.0 * tx + 0.05 * age + rng.normal(0, 1, n)
    data = pd.DataFrame({"age": age, "tx": tx, "y": y})

    p = sp.target_trial.protocol(
        eligibility="age >= 50",
        treatment_strategies=["tx=1", "tx=0"],
        assignment="randomization",
        time_zero="baseline",
        followup_end="1y",
        outcome="y",
        causal_contrast="ITT",
    )
    res = sp.target_trial.emulate(p, data, outcome_col="y", treatment_col="tx")
    assert isinstance(res, sp.target_trial.TargetTrialResult)
    assert abs(res.estimate - 2.0) < 0.2
    assert res.ci[0] < res.estimate < res.ci[1]
    assert res.n_excluded_immortal >= 0
    assert res.n_eligible + res.n_excluded_immortal == n
    assert "Target Trial Emulation" in res.summary()


def test_emulate_with_callable_eligibility():
    rng = np.random.default_rng(1)
    n = 500
    data = pd.DataFrame({
        "x": rng.normal(0, 1, n),
        "a": rng.binomial(1, 0.5, n),
        "y": rng.normal(0, 1, n),
    })
    p = sp.target_trial.protocol(
        eligibility=lambda row: row["x"] > 0,
        treatment_strategies=["A", "B"],
        assignment="randomization",
        time_zero="t0",
        followup_end="1y",
        outcome="y",
    )
    res = sp.target_trial.emulate(p, data, outcome_col="y", treatment_col="a")
    assert res.n_eligible == int((data["x"] > 0).sum())


# ---------- immortal time diagnostic ----------

def test_immortal_time_check_flags_backdated_treatment():
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "t": [5, 8, 10, 15],
        "tx_start": [3, 9, 10, 20],  # id=1 starts BEFORE elig -> immortal
        "elig_time": [5, 8, 10, 15],
    })
    diag = sp.target_trial.immortal_time_check(
        df, id_col="id", time_col="t",
        treatment_start_col="tx_start",
        eligibility_time_col="elig_time",
    )
    assert 1 in diag.flagged_ids
    assert diag.n_flagged == 1


# ---------- clone-censor-weight ----------

def test_clone_censor_weight_runs():
    rng = np.random.default_rng(2)
    n, T = 200, 3
    rows = []
    for i in range(n):
        for t in range(T):
            rows.append({
                "id": i,
                "t": t,
                "tx": 1 if rng.random() < 0.6 else 0,
                "x": rng.normal(),
            })
    df = pd.DataFrame(rows)
    strategies = {
        "always": lambda block: block["tx"].to_numpy() == 1,
        "never": lambda block: block["tx"].to_numpy() == 0,
    }
    res = sp.target_trial.clone_censor_weight(
        df, id_col="id", time_col="t", treatment_col="tx",
        strategies=strategies, censor_covariates=["x"],
    )
    assert set(res.strategies) == {"always", "never"}
    assert res.n_clones > 0
    assert "_ipcw" in res.cloned_data.columns
    assert (res.cloned_data["_ipcw"] > 0).all()


# ---------- IPCW ----------

def test_ipcw_recovers_uniform_weights_under_no_dependent_censoring():
    rng = np.random.default_rng(3)
    n = 500
    df = pd.DataFrame({
        "t": rng.uniform(0, 5, n),
        "d": rng.binomial(1, 0.5, n),
        "x": rng.normal(0, 1, n),
    })
    res = sp.ipcw(df, time="t", event="d", censor_covariates=["x"],
                  stabilize=True)
    assert isinstance(res, sp.IPCWResult)
    diag = res.diagnose()
    assert set(diag["metric"]) >= {"mean", "max", "min"}
    w = res.weights
    assert np.isfinite(w).all()
    assert (w > 0).all()
    # stabilized weights should average near 1 under independent censoring
    assert 0.5 < w.mean() < 2.0


def test_ipcw_truncation_bounds_weights():
    rng = np.random.default_rng(4)
    n = 300
    x = rng.normal(0, 3, n)
    # censoring probability strongly dependent on x -> extreme weights
    pc = 1 / (1 + np.exp(-2 * x))
    d = (rng.random(n) > pc).astype(int)
    df = pd.DataFrame({"t": rng.uniform(0, 1, n), "d": d, "x": x})
    res = sp.ipcw(df, time="t", event="d", censor_covariates=["x"],
                  truncate=(0.05, 0.95))
    w = res.weights
    # truncation must keep weights finite
    assert np.isfinite(w).all()


def test_ipcw_rejects_missing_columns():
    df = pd.DataFrame({"t": [1.0], "d": [1]})
    with pytest.raises(KeyError):
        sp.ipcw(df, time="t", event="d", censor_covariates=["nonexistent"])
