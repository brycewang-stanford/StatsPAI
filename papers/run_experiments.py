"""
Numerical validation experiments for StatsPAI arXiv/JOSS paper.

Generates all tables reported in the paper:
  - Table 1: Single-run validation with known DGPs
  - Table 2: IV vs OLS bias correction
  - Table 3: Multiple estimators on observational data
  - Table 4: Meta-learner comparison
  - Table 5: Monte Carlo coverage study

Usage:
    python run_experiments.py
"""

import statspai as sp
import numpy as np
import time
import warnings

warnings.filterwarnings("ignore")


def get_ci(r):
    """Extract 95% CI from a CausalResult."""
    if hasattr(r, "ci") and r.ci:
        return r.ci[0], r.ci[1]
    return r.estimate - 1.96 * r.se, r.estimate + 1.96 * r.se


def run_table1():
    """Table 1: Point estimation with known DGPs."""
    print("\n" + "=" * 75)
    print("TABLE 1: Point Estimation with Known DGPs")
    print("=" * 75)

    results = []

    # --- DID (2x2) ---
    true = 2.0
    df = sp.dgp_did(n_units=200, n_periods=10, effect=true, seed=42)
    df["post"] = (df["time"] >= 5).astype(int)
    t0 = time.time()
    r = sp.did_2x2(df, y="y", treat="group", time="post")
    dt = time.time() - t0
    ci_l, ci_u = get_ci(r)
    results.append(("DID (2x2)", true, r.estimate, r.se, ci_l <= true <= ci_u, dt))

    # --- RD (Sharp) ---
    true = 0.5
    df_rd = sp.dgp_rd(n=3000, effect=true, seed=42)
    t0 = time.time()
    r_rd = sp.rdrobust(df_rd, y="y", x="x", c=0.0)
    dt = time.time() - t0
    ci_l, ci_u = get_ci(r_rd)
    results.append(("RD (Sharp)", true, r_rd.estimate, r_rd.se, ci_l <= true <= ci_u, dt))

    # --- IV (2SLS) ---
    true = 0.5
    df_iv = sp.dgp_iv(n=2000, effect=true, first_stage=0.4, seed=42)
    t0 = time.time()
    r_iv = sp.ivreg("y ~ (treatment ~ instrument) + x1 + x2", data=df_iv)
    dt = time.time() - t0
    est = r_iv.params["treatment"]
    se = r_iv.std_errors["treatment"]
    ci_l, ci_u = est - 1.96 * se, est + 1.96 * se
    results.append(("IV (2SLS)", true, est, se, ci_l <= true <= ci_u, dt))

    # --- DML ---
    true = 0.5
    df_obs = sp.dgp_observational(n=2000, effect=true, confounding=0.3, seed=42)
    t0 = time.time()
    r_dml = sp.dml(df_obs, y="y", treat="treatment", covariates=["x1", "x2"])
    dt = time.time() - t0
    ci_l, ci_u = get_ci(r_dml)
    results.append(("DML", true, r_dml.estimate, r_dml.se, ci_l <= true <= ci_u, dt))

    # --- PSM ---
    t0 = time.time()
    r_match = sp.match(df_obs, y="y", treat="treatment", covariates=["x1", "x2"])
    dt = time.time() - t0
    ci_l, ci_u = get_ci(r_match)
    results.append(("PSM", true, r_match.estimate, r_match.se, ci_l <= true <= ci_u, dt))

    # --- AIPW ---
    t0 = time.time()
    r_aipw = sp.aipw(df_obs, y="y", treat="treatment", covariates=["x1", "x2"])
    dt = time.time() - t0
    ci_l, ci_u = get_ci(r_aipw)
    results.append(("AIPW", true, r_aipw.estimate, r_aipw.se, ci_l <= true <= ci_u, dt))

    # Print
    print(f"{'Method':<18} {'True':>8} {'Estimate':>10} {'SE':>8} {'Covers':>8} {'Time(s)':>8}")
    print("-" * 62)
    for name, true_v, est, se, cov, t in results:
        print(f"{name:<18} {true_v:>8.3f} {est:>10.4f} {se:>8.4f} {'Yes' if cov else 'No':>8} {t:>8.3f}")

    return results, df_iv, df_obs


def run_table2(df_iv):
    """Table 2: OLS vs IV under endogeneity."""
    print("\n" + "=" * 75)
    print("TABLE 2: OLS vs. IV Under Endogeneity (True theta = 0.500, N = 2,000)")
    print("=" * 75)

    true = 0.5
    r_ols = sp.regress("y ~ treatment + x1 + x2", data=df_iv)
    r_iv = sp.ivreg("y ~ (treatment ~ instrument) + x1 + x2", data=df_iv)

    ols_est = r_ols.params["treatment"]
    iv_est = r_iv.params["treatment"]

    print(f"  OLS (biased):  {ols_est:.4f}  (bias = {abs(ols_est - true):.4f})")
    print(f"  IV  (2SLS):    {iv_est:.4f}  (bias = {abs(iv_est - true):.4f})")
    print(f"  Bias reduction: {abs(ols_est - true) / abs(iv_est - true):.1f}x")

    return ols_est, iv_est


def run_table3(df_obs, dml_est, psm_est, aipw_est):
    """Table 3: Multiple estimators on observational data."""
    print("\n" + "=" * 75)
    print("TABLE 3: Estimator Comparison on Observational Data (True theta = 0.500)")
    print("=" * 75)

    true = 0.5
    r_ols = sp.regress("y ~ treatment + x1 + x2", data=df_obs)
    ols_est = r_ols.params["treatment"]

    print(f"  OLS (naive):   {ols_est:.4f}  (bias = {abs(ols_est - true):.4f})")
    print(f"  DML:           {dml_est:.4f}  (bias = {abs(dml_est - true):.4f})")
    print(f"  PSM:           {psm_est:.4f}  (bias = {abs(psm_est - true):.4f})")
    print(f"  AIPW:          {aipw_est:.4f}  (bias = {abs(aipw_est - true):.4f})")


def run_table4():
    """Table 4: Meta-learner comparison."""
    print("\n" + "=" * 75)
    print("TABLE 4: Meta-Learner ATE Estimates (RCT Data, True ATE = 1.000)")
    print("=" * 75)

    df_rct = sp.dgp_rct(n=2000, effect=1.0, heterogeneous=False, seed=42)

    print(f"{'Learner':<14} {'ATE':>10} {'SE':>8} {'Time(s)':>8}")
    print("-" * 42)
    for lrn in ["s", "t", "x", "r", "dr"]:
        t0 = time.time()
        r = sp.metalearner(
            df_rct, y="y", treat="treatment",
            covariates=["x1", "x2", "x3"], learner=lrn,
        )
        dt = time.time() - t0
        print(f"{lrn.upper()}-Learner     {r.estimate:>10.4f} {r.se:>8.4f} {dt:>8.3f}")


def run_table5(n_sims=200):
    """Table 5: Monte Carlo coverage study."""
    print("\n" + "=" * 75)
    print(f"TABLE 5: Monte Carlo Simulation Results ({n_sims} Replications)")
    print("=" * 75)

    configs = [
        ("DID (2x2)", 2.0, "did"),
        ("RD (Sharp)", 0.5, "rd"),
        ("IV (2SLS)", 0.5, "iv"),
    ]

    print(f"{'Method':<18} {'True':>8} {'Mean Bias':>10} {'RMSE':>8} {'Coverage':>10}")
    print("-" * 58)

    for name, true, method in configs:
        covers = 0
        biases = []

        for i in range(n_sims):
            if method == "did":
                df = sp.dgp_did(n_units=100, n_periods=10, effect=true, seed=i)
                df["post"] = (df["time"] >= 5).astype(int)
                r = sp.did_2x2(df, y="y", treat="group", time="post")
                ci_l, ci_u = get_ci(r)
                biases.append(r.estimate - true)
            elif method == "rd":
                df = sp.dgp_rd(n=1000, effect=true, seed=i)
                r = sp.rdrobust(df, y="y", x="x", c=0.0)
                ci_l, ci_u = get_ci(r)
                biases.append(r.estimate - true)
            elif method == "iv":
                df = sp.dgp_iv(n=1000, effect=true, first_stage=0.4, seed=i)
                r = sp.ivreg("y ~ (treatment ~ instrument) + x1 + x2", data=df)
                est = r.params["treatment"]
                se = r.std_errors["treatment"]
                ci_l, ci_u = est - 1.96 * se, est + 1.96 * se
                biases.append(est - true)

            if ci_l <= true <= ci_u:
                covers += 1

        biases = np.array(biases)
        rmse = np.sqrt(np.mean(biases**2))
        coverage = covers / n_sims * 100
        print(f"{name:<18} {true:>8.3f} {np.mean(biases):>10.4f} {rmse:>8.4f} {coverage:>9.1f}%")


if __name__ == "__main__":
    np.random.seed(42)

    print("StatsPAI Paper — Numerical Validation Experiments")
    print("=" * 75)
    print(f"StatsPAI version: {sp.__version__}")
    print(f"NumPy version: {np.__version__}")
    print()

    # Table 1
    t1_results, df_iv, df_obs = run_table1()

    # Table 2
    run_table2(df_iv)

    # Table 3
    dml_est = t1_results[3][2]   # DML estimate
    psm_est = t1_results[4][2]   # PSM estimate
    aipw_est = t1_results[5][2]  # AIPW estimate
    run_table3(df_obs, dml_est, psm_est, aipw_est)

    # Table 4
    run_table4()

    # Table 5
    run_table5(n_sims=200)

    print("\n" + "=" * 75)
    print("All experiments completed.")
    print("=" * 75)
