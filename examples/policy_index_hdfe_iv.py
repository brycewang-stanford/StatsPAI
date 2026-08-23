"""End-to-end HDFE-IV workflow on a county-month policy-intensity panel.

Mirrors the empirical shape of Zhang et al. (2026, *Science* 393:831-836,
doi:10.1126/science.aee0747): a county-level policy-intensity index, an
outcome built from species-sighting records, county and year-month fixed
effects, an interaction instrument, and inference that has to survive
clustering and spatial correlation.

The data are simulated here so the script runs offline, but every StatsPAI
call is the one a real replication would make. See
``docs/guides/policy_index_hdfe_iv.md`` for the narrative version.

Run with::

    python examples/policy_index_hdfe_iv.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import statspai as sp

N_COUNTY, N_MONTH, SEED = 200, 36, 20260823


def simulate() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (sighting records, county-month covariate panel)."""
    rng = np.random.default_rng(SEED)
    county = np.repeat(np.arange(N_COUNTY), N_MONTH)
    ym = np.tile(np.arange(N_MONTH), N_COUNTY)
    n = county.size

    sun = rng.normal(size=N_COUNTY)  # historical solar resource
    cpu_inv = rng.normal(size=N_MONTH)  # inverse policy uncertainty
    alpha = rng.normal(scale=1.2, size=N_COUNTY)
    delta = rng.normal(scale=0.6, size=N_MONTH)

    z = sun[county] * cpu_inv[ym]  # the interaction instrument
    u = rng.normal(size=n)  # confounder: land quality
    temp = rng.normal(size=n)
    effort = rng.gamma(4.0, 1.0, size=n)  # birding hours

    policy = 1.0 * z + 0.7 * u + alpha[county] + delta[ym] + rng.normal(size=n)

    panel = pd.DataFrame(
        {
            "county": county,
            "ym": ym,
            "z": z,
            "policy": policy,
            "temp": temp,
            "log_effort": np.log(effort),
            "lat": np.repeat(rng.uniform(22, 48, N_COUNTY), N_MONTH),
            "lon": np.repeat(rng.uniform(80, 128, N_COUNTY), N_MONTH),
            "poor": np.repeat((rng.uniform(size=N_COUNTY) < 0.35).astype(int), N_MONTH),
            # Mechanism outcomes: vegetation cover down, leaf area up.
            "ndvi": -0.03 * policy + 0.3 * u + alpha[county] + rng.normal(size=n),
            "lai": 0.04 * policy + 0.3 * u + alpha[county] + rng.normal(size=n),
        }
    )

    # Sighting records: a richer community where policy intensity is low.
    n_species = 24
    rows = []
    for i in range(n):
        lam = np.exp(1.6 - 0.05 * policy[i] + 0.4 * u[i])
        weights = np.exp(-np.arange(n_species) / max(lam, 0.3))
        weights /= weights.sum()
        k = 1 + rng.poisson(6 * effort[i] / effort.mean())
        drawn = rng.choice(n_species, size=k, p=weights)
        rows.append(
            pd.DataFrame(
                {"county": county[i], "ym": ym[i], "species": [f"s{s}" for s in drawn]}
            )
        )
    return pd.concat(rows, ignore_index=True), panel


def main() -> None:
    records, panel = simulate()

    # ── 1. Build the outcome from raw records ─────────────────────────
    diversity = sp.diversity_index(
        records,
        species="species",
        by=["county", "ym"],
        index=["shannon", "richness", "pielou"],
        min_records=5,
    )
    panel = panel.merge(diversity, on=["county", "ym"], how="left").dropna(
        subset=["shannon"]
    )
    print(
        f"panel: {len(panel):,} county-months, " f"{panel.county.nunique()} counties\n"
    )

    formula = "shannon ~ (policy ~ z) + temp + log_effort"

    # ── 2. Baseline: two-way FE, clustered by county ──────────────────
    res = sp.iv(formula, data=panel, absorb=["county", "ym"], cluster="county")
    print("=== Baseline HDFE-IV ===")
    print(f"  beta(policy)      = {res.params['policy']: .5f}")
    print(f"  cluster SE        = {res.std_errors['policy']: .5f}")
    print(
        f"  absorbed DOF      = {res.model_info['fe_dof_charged']} "
        f"(nested and dropped: {res.model_info['fe_nested_in_cluster']})"
    )
    ols = sp.feols(
        "shannon ~ policy + temp + log_effort | county + ym",
        data=panel,
        vcov={"CRV1": "county"},
    )
    print(
        f"  OLS comparator    = {ols.params['policy']: .5f}  "
        "(confounded; sign flip is the point)\n"
    )

    # ── 3. Weak identification, in the vcov actually estimated ────────
    diag = sp.iv_diag(
        panel,
        y="shannon",
        endog="policy",
        instruments=["z"],
        exog=["temp", "log_effort"],
        absorb=["county", "ym"],
        cluster="county",
        n_boot=200,
        random_state=0,
    )
    print("=== Identification ===")
    print(f"  Olea-Pflueger F   = {diag.effective_F:,.1f}")
    print(f"  KP rk LM          = {diag.kp_rk_lm:,.2f}  (underidentification)")
    print(f"  KP rk Wald F      = {diag.kp_rk_f:,.1f}  (weak identification)")
    print(f"  AR 95% set        = [{diag.ar_ci[0]: .5f}, {diag.ar_ci[1]: .5f}]")
    print(
        f"  Wald 95% CI       = [{diag.ci_analytic_2sls[0]: .5f}, "
        f"{diag.ci_analytic_2sls[1]: .5f}]\n"
    )

    # ── 4. Spatial and serial correlation ─────────────────────────────
    spatial = sp.conley(res, panel, lat="lat", lon="lon", dist_cutoff=200)
    spacetime = sp.conley(
        res,
        panel,
        lat="lat",
        lon="lon",
        dist_cutoff=200,
        time="ym",
        lag_cutoff=12,
        unit="county",
    )
    print("=== Robust variance menu ===")
    print(f"  cluster(county)          SE = {res.std_errors['policy']:.5f}")
    print(f"  Conley 200km             SE = {spatial.std_errors['policy']:.5f}")
    print(f"  Conley 200km + 12m HAC   SE = {spacetime.std_errors['policy']:.5f}")
    twoway = sp.iv(
        formula, data=panel, absorb=["county", "ym"], cluster=["county", "ym"]
    )
    print(f"  cluster(county, ym)      SE = {twoway.std_errors['policy']:.5f}\n")

    # ── 5. Heterogeneity ──────────────────────────────────────────────
    print("=== Heterogeneity ===")
    for label, mask in {
        "non-poor counties": panel.poor == 0,
        "poor counties": panel.poor == 1,
    }.items():
        sub = sp.iv(
            formula,
            data=panel[mask],
            absorb=["county", "ym"],
            cluster="county",
        )
        lo = sub.params["policy"] - 1.96 * sub.std_errors["policy"]
        hi = sub.params["policy"] + 1.96 * sub.std_errors["policy"]
        print(f"  {label:<20s} {sub.params['policy']: .5f}  " f"[{lo: .5f}, {hi: .5f}]")
    print()

    # ── 6. Mechanisms: same RHS, several LHS ──────────────────────────
    print("=== Mechanisms (2SLS, one per outcome) ===")
    for outcome in ("ndvi", "lai"):
        m = sp.iv(
            f"{outcome} ~ (policy ~ z) + temp + log_effort",
            data=panel,
            absorb=["county", "ym"],
            cluster="county",
        )
        star = "*" if abs(m.params["policy"] / m.std_errors["policy"]) > 1.96 else " "
        print(
            f"  {outcome:<6s} {m.params['policy']: .5f} "
            f"({m.std_errors['policy']:.5f}){star}"
        )
    print("\n  NDVI down with LAI up is the 'low-quality greening' signature:")
    print("  structurally complex vegetation replaced by dense monoculture.")


if __name__ == "__main__":
    main()
