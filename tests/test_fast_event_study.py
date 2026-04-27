"""Tests for ``sp.fast.event_study`` (Phase 6)."""
from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _staggered_panel(n_units=80, n_periods=20, treatment_effect=0.5, seed=0):
    """Staggered DGP: half units treated at t=10, half never treated.

    Treatment effect is constant after treatment.
    """
    rng = np.random.default_rng(seed)
    units = np.arange(n_units)
    times = np.arange(n_periods)
    rows = []
    for u in units:
        treat_time = 10 if u < n_units // 2 else None
        for t in times:
            if treat_time is None:
                et = np.nan
                d = 0.0
            else:
                et = t - treat_time
                d = 1.0 if et >= 0 else 0.0
            unit_eff = rng.normal(0, 0.4)
            time_eff = 0.05 * t
            y = unit_eff + time_eff + treatment_effect * d + rng.normal()
            rows.append((u, t, y, et))
    df = pd.DataFrame(rows, columns=["unit", "time", "y", "event_time"])
    return df


def test_event_study_recovers_constant_effect():
    """Post-treatment coefs should average ~ true treatment effect."""
    df = _staggered_panel(seed=1, treatment_effect=0.6)
    res = sp.fast.event_study(
        df, y="y", unit="unit", time="time", event_time="event_time",
        window=(-5, 5),
    )
    td = res.tidy()
    # Post-treatment coefs (event_time >= 0) should be near +0.6
    post = td[td["event_time"] >= 0]["Estimate"]
    assert abs(post.mean() - 0.6) < 0.15, f"got {post.mean()}"


def test_event_study_pre_trend_near_zero():
    """Pre-treatment coefs (event_time <= -2) should be near zero."""
    df = _staggered_panel(seed=2, treatment_effect=0.5)
    res = sp.fast.event_study(
        df, y="y", unit="unit", time="time", event_time="event_time",
        window=(-5, 5),
    )
    td = res.tidy()
    pre = td[td["event_time"] <= -2]["Estimate"]
    # Loose bound — finite-sample noise. We just check there's no systematic
    # pre-trend.
    assert pre.abs().mean() < 0.25


def test_event_study_returns_correct_shape():
    df = _staggered_panel(seed=3)
    res = sp.fast.event_study(
        df, y="y", unit="unit", time="time", event_time="event_time",
        window=(-3, 3),
    )
    # 7 event-times in window minus the reference (-1) = 6 coefs
    assert len(res.event_times) == 6
    assert res.coefs.shape == (6,)
    assert res.ses.shape == (6,)
    assert (res.ses > 0).all()


def test_event_study_custom_reference():
    df = _staggered_panel(seed=4)
    res = sp.fast.event_study(
        df, y="y", unit="unit", time="time", event_time="event_time",
        window=(-3, 3), reference=-2,
    )
    # -2 should be excluded as the reference
    assert -2 not in res.event_times


def test_event_study_clustered_se():
    df = _staggered_panel(seed=5)
    res_default = sp.fast.event_study(
        df, y="y", unit="unit", time="time", event_time="event_time",
        window=(-3, 3),
    )
    # Default clusters on unit
    assert res_default.cluster_var == "unit"
    assert res_default.n_clusters == df["unit"].nunique()


def test_event_study_missing_column_raises():
    df = _staggered_panel(seed=6)
    with pytest.raises(KeyError):
        sp.fast.event_study(
            df, y="nope", unit="unit", time="time",
            event_time="event_time",
        )


def test_event_study_summary_string():
    df = _staggered_panel(seed=7)
    res = sp.fast.event_study(
        df, y="y", unit="unit", time="time", event_time="event_time",
        window=(-2, 2),
    )
    s = res.summary()
    assert "event_study" in s
    assert "event_time" in s


# ---------------------------------------------------------------------------
# Cluster-SE small-sample factor must charge absorbed FE rank
# (regression test for the silent bug fixed alongside this commit; the
# pre-fix factor was (n-1)/(n-k_dummies) only — i.e. event_study SEs were
# systematically too small. reghdfe / fixest both subtract Σ(G_k - 1).)
# ---------------------------------------------------------------------------

def test_event_study_se_includes_fe_rank_correction():
    """Pin event_study CR1 SE against the hand-computed sandwich.

    Reconstructs the FE-residualised system, computes the cluster-score
    meat, and applies the small-sample factor with ``extra_df = sum(G_k
    - 1)`` over the absorbed FE dimensions. The post-fix SE must equal
    this; the pre-fix (buggy) SE — same formula with ``extra_df=0`` —
    is asserted to be strictly smaller, locking the fix's direction.
    """
    df = _staggered_panel(seed=42, n_units=60, n_periods=15)
    res = sp.fast.event_study(
        df, y="y", unit="unit", time="time", event_time="event_time",
        window=(-3, 3), reference=-1,
    )

    # --- Reconstruct the system the same way event_study does ---
    wt = sp.fast.within(df, fe=["unit", "time"], drop_singletons=True)
    et = df["event_time"].to_numpy()
    finite = np.isfinite(et)
    et_int = np.where(finite, et, np.iinfo(np.int64).min).astype(np.int64)
    in_window = finite & (et_int >= -3) & (et_int <= 3)
    et_int = np.where(in_window, et_int, np.iinfo(np.int64).min)
    levels = sorted({v for v in et_int[in_window] if v != -1})
    dummies = pd.DataFrame({
        f"et_{lv}": ((et_int == lv) & in_window).astype(np.float64)
        for lv in levels
    }, index=df.index)
    df_aug = pd.concat([df, dummies], axis=1)

    y_dem, _ = wt.transform(df_aug["y"].to_numpy(dtype=np.float64))
    X_dem = wt.transform_columns(df_aug, list(dummies.columns)).to_numpy()

    XtX = X_dem.T @ X_dem
    bread = np.linalg.inv(XtX)
    beta = bread @ X_dem.T @ y_dem
    resid = y_dem - X_dem @ beta

    cluster_arr = df_aug.loc[wt.keep_mask, "unit"].to_numpy()
    cluster_codes, _ = pd.factorize(cluster_arr, sort=False)
    G = int(cluster_codes.max()) + 1
    n, k = X_dem.shape

    score = resid[:, None] * X_dem
    cluster_score = np.zeros((G, k))
    np.add.at(cluster_score, cluster_codes, score)
    meat = cluster_score.T @ cluster_score
    V_unscaled = bread @ meat @ bread

    fe_dof = sum(int(g) - 1 for g in wt.n_fe)

    # Post-fix factor (matches the new code)
    factor_fixed = (G / (G - 1.0)) * ((n - 1.0) / (n - k - fe_dof))
    se_fixed_expected = np.sqrt(np.diag(V_unscaled * factor_fixed))

    # Pre-fix (buggy) factor — same formula minus the FE-rank charge
    factor_buggy = (G / (G - 1.0)) * ((n - 1.0) / (n - k))
    se_buggy_would_have = np.sqrt(np.diag(V_unscaled * factor_buggy))

    np.testing.assert_allclose(res.ses, se_fixed_expected, rtol=1e-12, atol=1e-12)
    # Direction: post-fix SE strictly larger than the buggy one — the
    # whole point of the FE-rank correction.
    assert (res.ses > se_buggy_would_have).all(), (
        "event_study FE-rank fix did not increase SEs as expected"
    )
    # Math identity: the only thing that changes between buggy and fixed
    # factor is the denominator, so the ratio must equal sqrt((n-k)/(n-k-fe_dof))
    # uniformly across coefficients. Pin it tight — this also guards against
    # accidental factor changes in the meat / bread that wouldn't shift this
    # ratio away from the closed form.
    ratio_expected = np.sqrt((n - k) / (n - k - fe_dof))
    ratio_actual = res.ses / se_buggy_would_have
    np.testing.assert_allclose(
        ratio_actual,
        np.full_like(ratio_actual, ratio_expected),
        rtol=1e-12, atol=1e-12,
    )
    # Sanity: with this n/k/fe_dof, the bump must be at least 1% — if not,
    # the test panel is too large for the bug to have been observable and
    # we should pick a smaller one.
    assert ratio_expected > 1.01, (
        f"test panel too large to expose bug: ratio={ratio_expected:.6f}; "
        "shrink n_units / n_periods to amplify FE-rank share of DOF"
    )


def _truly_staggered_panel(n_units=80, n_periods=20, treatment_effect=0.5, seed=0):
    """Variant of ``_staggered_panel`` with per-unit treat_times so that
    event_time is NOT collinear with the time FE — required to get fixest
    past its FE-collinearity check on the dummies.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        # Random treatment timing in the middle of the panel; ~25% never treated
        if rng.random() < 0.25:
            treat_time = None
        else:
            treat_time = int(rng.integers(5, n_periods - 4))
        unit_eff = rng.normal(0, 0.4)
        for t in range(n_periods):
            if treat_time is None:
                et = np.nan
                d = 0.0
            else:
                et = t - treat_time
                d = 1.0 if et >= 0 else 0.0
            time_eff = 0.05 * t
            y = unit_eff + time_eff + treatment_effect * d + rng.normal()
            rows.append((u, t, y, et))
    return pd.DataFrame(rows, columns=["unit", "time", "y", "event_time"])


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript not on PATH")
def test_event_study_se_close_to_r_fixest(tmp_path):
    """Tight parity vs R ``fixest::feols(... | unit + time, cluster=~unit, ssc(fixef.K='full'))``.

    Empirically the SE drift is ~0.045% on this 80-unit / 15-period
    panel — the only difference is the FE-rank convention: StatsPAI
    charges ``Σ (G_k - 1)`` (matches ``sp.fast.fepois``); fixest's
    ``fixef.K='full'`` charges the true matrix rank ``Σ G_k - 1`` for
    K=2. That's a 1-DOF gap, negligible vs the ~1100 residual DOF here.
    Tolerance is set to 1% — comfortably above the 0.045% drift, well
    below the 4–5% the pre-fix bug would have produced.
    """
    df = _truly_staggered_panel(seed=11, n_units=80, n_periods=15)

    res = sp.fast.event_study(
        df, y="y", unit="unit", time="time", event_time="event_time",
        window=(-3, 3), reference=-1,
    )

    # Use ``etm{n}`` (m=minus) / ``etp{n}`` (p=plus) names to dodge R's
    # parsing of ``-`` in formula identifiers.
    def _r_name(lv: int) -> str:
        return f"etm{abs(lv)}" if lv < 0 else f"etp{lv}"
    levels = [-3, -2, 0, 1, 2, 3]
    rhs_terms = " + ".join(_r_name(lv) for lv in levels)

    csv_path = tmp_path / "es.csv"
    df.to_csv(csv_path, index=False)
    dummy_assign = "\n".join(
        f"d[['{_r_name(lv)}']] <- as.numeric(!is.na(d$event_time) & d$event_time == {lv})"
        for lv in levels
    )
    r_script = (
        "suppressMessages({library(data.table); library(fixest); library(jsonlite)})\n"
        f"d <- fread('{csv_path}')\n"
        # Mirror the window=(-3, 3) truncation: never-treated stay (NA),
        # treated outside window get NA so they're absorbed by FEs only.
        "d[, event_time := ifelse(!is.na(event_time) & event_time >= -3 & event_time <= 3,\n"
        "                          event_time, NA_real_)]\n"
        # Build dummies manually so fixest's parser doesn't drop NA rows;
        # NA event_time → all dummies = 0 (matches StatsPAI semantics).
        f"{dummy_assign}\n"
        f"f <- feols(y ~ {rhs_terms} | unit + time,\n"
        "           data=d, cluster=~unit, ssc=ssc(fixef.K='full'))\n"
        "co <- coef(f); s <- se(f)\n"
        "out <- list(names = names(co), coefs = as.list(co), se = as.list(s))\n"
        "cat(toJSON(out, auto_unbox=TRUE, digits=14))\n"
    )
    proc = subprocess.run(
        ["Rscript", "-e", r_script], capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        pytest.skip(f"Rscript failed: {proc.stderr[:300]}")
    r_out = json.loads(proc.stdout.strip().splitlines()[-1])

    py_se = {int(t): float(s) for t, s in zip(res.event_times, res.ses)}
    # ``r_out["se"]`` is a dict keyed by name (jsonlite auto_unbox keeps
    # names). Reverse the etm/etp encoding to recover signed event-times.
    r_se = {}
    for nm, s in r_out["se"].items():
        nm = str(nm)
        if nm.startswith("etm"):
            r_se[-int(nm[3:])] = float(s)
        elif nm.startswith("etp"):
            r_se[int(nm[3:])] = float(s)

    for et_val, se_py in py_se.items():
        assert et_val in r_se, f"event_time {et_val} missing from R output"
        rel = abs(se_py - r_se[et_val]) / max(abs(r_se[et_val]), 1e-12)
        assert rel < 0.01, (
            f"SE drift too large at et={et_val}: py={se_py:.6e} r={r_se[et_val]:.6e} "
            f"rel={rel:.3e}"
        )
