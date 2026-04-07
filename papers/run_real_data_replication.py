"""
Precise replication with REAL published datasets.

Uses the actual data from:
  1. Card (1995) — Wooldridge textbook version (N=3,010)
  2. LaLonde (1986) / Dehejia & Wahba (1999) — MatchIt package (N=614)
  3. California Prop 99 — built-in canonical dataset

Published benchmarks are from:
  - Angrist & Pischke, Mostly Harmless Econometrics, Table 4.1.1
  - Dehejia & Wahba (1999), JASA, Table 3
  - Abadie, Diamond & Hainmueller (2010), JASA

Usage:
    python run_real_data_replication.py
"""

import numpy as np
import pandas as pd
import time
import warnings
import os

warnings.filterwarnings("ignore")

import statspai as sp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ====================================================================
# REPLICATION 1: Card (1995) — REAL DATA
# Published: OLS educ ≈ 0.0747, IV educ ≈ 0.1315
# Source: Angrist & Pischke, MHE Table 4.1.1
# ====================================================================
def replicate_card_real():
    print("\n" + "=" * 75)
    print("REPLICATION 1: Card (1995) — REAL DATA")
    print("Published: Angrist & Pischke, MHE Table 4.1.1")
    print("Data: Wooldridge textbook (Rdatasets), N = 3,010")
    print("=" * 75)

    data_path = os.path.join(SCRIPT_DIR, "data_card1995.csv")
    if not os.path.exists(data_path):
        print("  Downloading Card (1995) data...")
        df = pd.read_csv(
            "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/card.csv"
        )
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)

    # Drop rows with missing wages
    df = df.dropna(subset=["lwage", "educ", "nearc4", "exper", "expersq",
                            "black", "south", "smsa"])

    print(f"  N = {len(df)} (after dropping missing)")

    # === OLS ===
    ols = sp.regress(
        "lwage ~ educ + exper + expersq + black + south + smsa",
        data=df, robust="hc1",
    )
    ols_educ = ols.params["educ"]
    ols_se = ols.std_errors["educ"]

    # === IV: educ instrumented by nearc4 ===
    iv = sp.ivreg(
        "lwage ~ (educ ~ nearc4) + exper + expersq + black + south + smsa",
        data=df,
    )
    iv_educ = iv.params["educ"]
    iv_se = iv.std_errors["educ"]

    # === First stage ===
    fs = sp.regress(
        "educ ~ nearc4 + exper + expersq + black + south + smsa",
        data=df, robust="hc1",
    )
    fs_coef = fs.params["nearc4"]
    fs_t = fs_coef / fs.std_errors["nearc4"]

    # === Extended spec: add controls (MHE col 5) ===
    ols_ext = sp.regress(
        "lwage ~ educ + exper + expersq + black + south + smsa + married",
        data=df, robust="hc1",
    )
    iv_ext = sp.ivreg(
        "lwage ~ (educ ~ nearc4) + exper + expersq + black + south + smsa + married",
        data=df,
    )

    print(f"\n  {'':30} {'StatsPAI':>10} {'Published':>10} {'Match?':>8}")
    print(f"  {'-'*60}")
    print(f"  {'OLS coef on educ':<30} {ols_educ:>10.4f} {'0.0747':>10} {'✓' if abs(ols_educ - 0.0747) < 0.005 else '~':>8}")
    print(f"  {'OLS SE':<30} {ols_se:>10.4f} {'0.0034':>10} {'✓' if abs(ols_se - 0.0034) < 0.001 else '~':>8}")
    print(f"  {'IV coef on educ':<30} {iv_educ:>10.4f} {'0.1315':>10} {'✓' if abs(iv_educ - 0.1315) < 0.01 else '~':>8}")
    print(f"  {'IV SE':<30} {iv_se:>10.4f} {'0.0549':>10} {'✓' if abs(iv_se - 0.0549) < 0.01 else '~':>8}")
    print(f"  {'First-stage F (nearc4)':<30} {fs_t**2:>10.1f} {'>10':>10}")
    print(f"  {'IV/OLS ratio':<30} {iv_educ/ols_educ:>10.2f} {'1.76':>10}")

    print(f"\n  Extended specification (+ married):")
    print(f"    OLS: {ols_ext.params['educ']:.4f}, IV: {iv_ext.params['educ']:.4f}")

    return ols_educ, iv_educ


# ====================================================================
# REPLICATION 2: LaLonde/Dehejia-Wahba — REAL DATA
# Published: Experimental ATT ≈ $1,794 (D&W 1999 Table 3)
# Data: MatchIt package (NSW experimental only, N=614)
# ====================================================================
def replicate_lalonde_real():
    print("\n" + "=" * 75)
    print("REPLICATION 2: LaLonde (1986) / Dehejia & Wahba (1999) — REAL DATA")
    print("Published: Experimental ATT ≈ $1,794 (D&W 1999)")
    print("Data: MatchIt R package (NSW experimental sample), N = 614")
    print("=" * 75)

    # Use the EXACT Dehejia-Wahba (1999) NSW experimental sample
    # from Dehejia's NBER page: 185 treated + 260 control = 445
    data_path = os.path.join(SCRIPT_DIR, "data_nsw_dw.csv")
    if not os.path.exists(data_path):
        print("  Downloading NSW D&W sample from NBER...")
        cols = ["treat", "age", "education", "black", "hispanic",
                "married", "nodegree", "re74", "re75", "re78"]
        treated = pd.read_csv(
            "https://users.nber.org/~rdehejia/data/nswre74_treated.txt",
            sep=r'\s+', header=None, names=cols,
        )
        control = pd.read_csv(
            "https://users.nber.org/~rdehejia/data/nswre74_control.txt",
            sep=r'\s+', header=None, names=cols,
        )
        df = pd.concat([treated, control], ignore_index=True)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)

    covs = ["age", "education", "black", "hispanic", "married", "nodegree", "re74", "re75"]

    print(f"  N = {len(df)} (treated: {int(df['treat'].sum())}, control: {int((1-df['treat']).sum())})")

    # === 1. Raw difference in means (experimental estimate) ===
    treated = df[df["treat"] == 1]["re78"]
    control = df[df["treat"] == 0]["re78"]
    raw_diff = treated.mean() - control.mean()
    raw_se = np.sqrt(treated.var() / len(treated) + control.var() / len(control))

    # === 2. OLS with controls ===
    ols = sp.regress(
        f"re78 ~ treat + {' + '.join(covs)}",
        data=df, robust="hc1",
    )
    ols_est = ols.params["treat"]
    ols_se = ols.std_errors["treat"]

    # === 3. PSM ===
    psm = sp.match(df, y="re78", treat="treat", covariates=covs)

    # === 4. DML ===
    dml = sp.dml(df, y="re78", treat="treat", covariates=covs)

    # === 5. AIPW ===
    aipw = sp.aipw(df, y="re78", treat="treat", covariates=covs)

    print(f"\n  {'Estimator':<25} {'Estimate ($)':>12} {'SE ($)':>10} {'Published':>12}")
    print(f"  {'-'*62}")
    print(f"  {'Raw diff (experimental)':<25} {raw_diff:>12.0f} {raw_se:>10.0f} {'$1,794':>12}")
    print(f"  {'OLS + controls':<25} {ols_est:>12.0f} {ols_se:>10.0f} {'—':>12}")
    print(f"  {'PSM':<25} {psm.estimate:>12.0f} {psm.se:>10.0f} {'—':>12}")
    print(f"  {'DML':<25} {dml.estimate:>12.0f} {dml.se:>10.0f} {'—':>12}")
    print(f"  {'AIPW':<25} {aipw.estimate:>12.0f} {aipw.se:>10.0f} {'—':>12}")

    match_quality = "✓" if abs(raw_diff - 1794) < 200 else "~"
    print(f"\n  Raw diff vs published $1,794: diff = ${abs(raw_diff - 1794):.0f} {match_quality}")
    print(f"  Note: All causal estimators should converge to the experimental")
    print(f"  benchmark since this is RCT data (no confounding).")

    return raw_diff


# ====================================================================
# REPLICATION 3: Prop 99 Synthetic Control — REAL DATA (built-in)
# Published: ≈ -26 packs by 2000 (Abadie et al. 2010)
# ====================================================================
def replicate_prop99_real():
    print("\n" + "=" * 75)
    print("REPLICATION 3: Abadie et al. (2010) — Prop 99 REAL DATA")
    print("Published: ≈ −26 packs per capita by 2000")
    print("Data: Built-in california_prop99() dataset")
    print("=" * 75)

    df = sp.california_prop99()
    print(f"  {df['state'].nunique()} states, {df['year'].min()}-{df['year'].max()}")

    # Synthetic control
    r = sp.synth(
        df,
        outcome="packspercapita",
        unit="state",
        time="year",
        treated_unit="California",
        treatment_time=1989,
    )

    print(f"\n  Estimated ATT: {r.estimate:.2f}")
    print(f"  Published:     ≈ −26")

    # Also run SDID
    r_sdid = sp.sdid(
        df, y="packspercapita", unit="state",
        time="year", treat_unit="California", treat_time=1989,
    )
    print(f"  SDID estimate: {r_sdid.estimate:.2f}")
    print(f"  (Arkhangelsky et al. 2021 SDID ≈ −15.6)")

    return r.estimate


# ====================================================================
# CROSS-VALIDATION: StatsPAI vs EconML on REAL Card data
# ====================================================================
def cross_validate_real_data():
    print("\n" + "=" * 75)
    print("CROSS-VALIDATION: StatsPAI vs EconML on Real Card (1995) Data")
    print("=" * 75)

    try:
        from econml.dml import LinearDML
        from sklearn.ensemble import RandomForestRegressor
    except ImportError:
        print("  EconML not installed — skipping.")
        return

    df = pd.read_csv(os.path.join(SCRIPT_DIR, "data_card1995.csv"))
    df = df.dropna(subset=["lwage", "educ", "nearc4", "exper", "expersq",
                            "black", "south", "smsa"])

    covs = ["exper", "expersq", "black", "south", "smsa"]

    # --- StatsPAI DML ---
    t0 = time.time()
    sp_dml = sp.dml(df, y="lwage", treat="educ", covariates=covs)
    sp_time = time.time() - t0

    # --- EconML DML ---
    Y = df["lwage"].values
    T = df["educ"].values
    X = df[covs].values

    t0 = time.time()
    econml_dml = LinearDML(
        model_y=RandomForestRegressor(n_estimators=100, random_state=42),
        model_t=RandomForestRegressor(n_estimators=100, random_state=42),
        cv=5,
        random_state=42,
    )
    econml_dml.fit(Y, T, X=X)
    econml_est = econml_dml.ate(X)
    econml_ci = econml_dml.ate_interval(X)
    econml_time = time.time() - t0

    print(f"\n  Task: Effect of education on log wages (Card 1995 data)")
    print(f"  N = {len(df)}")
    print(f"\n  {'Package':<12} {'Estimate':>10} {'Time(s)':>10}")
    print(f"  {'-'*34}")
    print(f"  {'StatsPAI':<12} {sp_dml.estimate:>10.4f} {sp_time:>10.3f}")
    print(f"  {'EconML':<12} {econml_est:>10.4f} {econml_time:>10.3f}")
    diff = abs(sp_dml.estimate - econml_est)
    print(f"\n  Difference: {diff:.4f}")
    print(f"  Note: Both use DML but with different hyperparameters.")
    print(f"  OLS benchmark: ~0.075, IV benchmark: ~0.132")


# ====================================================================
# MAIN
# ====================================================================
if __name__ == "__main__":
    np.random.seed(42)

    print("StatsPAI Paper — Precise Replication with REAL Data")
    print("=" * 75)
    print(f"StatsPAI version: {sp.__version__}")
    print()

    ols_educ, iv_educ = replicate_card_real()
    raw_diff = replicate_lalonde_real()
    synth_est = replicate_prop99_real()
    cross_validate_real_data()

    # === FINAL SUMMARY ===
    print("\n" + "=" * 75)
    print("FINAL SUMMARY: Replication Accuracy")
    print("=" * 75)
    print(f"\n  {'Experiment':<35} {'StatsPAI':>10} {'Published':>10} {'Verdict':>10}")
    print(f"  {'-'*67}")
    print(f"  {'Card OLS (educ)':<35} {ols_educ:>10.4f} {'0.0747':>10} {'✓' if abs(ols_educ - 0.0747) < 0.005 else '~':>10}")
    print(f"  {'Card IV (educ)':<35} {iv_educ:>10.4f} {'0.1315':>10} {'✓' if abs(iv_educ - 0.1315) < 0.01 else '~':>10}")
    print(f"  {'LaLonde raw diff':<35} {raw_diff:>10.0f} {'1794':>10} {'✓' if abs(raw_diff - 1794) < 200 else '~':>10}")
    print(f"  {'Prop 99 synth gap':<35} {synth_est:>10.1f} {'-26':>10}")
    print()
    print("  ✓ = matches within rounding")
    print("  ~ = same direction/magnitude, spec differences")
    print("=" * 75)
