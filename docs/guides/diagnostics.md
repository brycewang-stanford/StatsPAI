# Automatic diagnostics — what StatsPAI checks for you

Every fitted StatsPAI result carries a self-audit. You do not have to remember
which assumption each estimator leans on — the result knows, and will tell you:

```python
import statspai as sp

res = sp.ivreg("y ~ (d ~ z)", data=df)

res.violations()          # structured list of flagged concerns (may be empty)
sp.audit(res)             # reviewer checklist: what's checked, passed, missing
```

`res.violations()` inspects diagnostics the estimator **already computed** — it
never re-runs a test or touches your data, so it is instant. Each entry is a
dict an agent can branch on:

```python
{"kind": "assumption", "severity": "warning", "test": "weak_instrument",
 "value": 4.2, "threshold": 10.0,
 "message": "First-stage F = 4.20 < 10 (Stock-Yogo 5% bias) — weak instrument …",
 "recovery_hint": "Use sp.anderson_rubin_ci …",
 "alternatives": ["sp.anderson_rubin_ci", "sp.iv"]}
```

`sp.audit(res)` is a **superset** of `violations()`: it adds the robustness /
sensitivity checks a referee would ask for (present, failed, or still missing)
and folds in every live violation, so one call gives the full picture.

Two design commitments make these signals trustworthy:

- **Fit-time and structured API agree.** Where an estimator warns at fit time
  (weak IV, few clusters, separation), the same concern appears in
  `violations()` — never one without the other.
- **Calibrated not to cry wolf.** Thresholds are set so the field's canonical
  *good* examples stay silent (e.g. California Prop-99 does not trip the
  synthetic-control pre-fit check). A diagnostic that fires on the textbook
  example would erode trust, not build it.

## The checklist

| Family | Check | Fires when | Points to |
| --- | --- | --- | --- |
| **DID** | parallel trends | pre-trend joint test p < 0.10 (Roth 2022) | `sp.sensitivity_rr`, `sp.callaway_santanna`, `sp.did_imputation` |
| **IV** | weak instrument | first-stage F < 10 (Stock-Yogo) | `sp.anderson_rubin_ci`, `sp.iv(method='liml')` |
| **Panel / OLS** | few clusters | # clusters < 30 (Cameron-Gelbach-Miller 2008) | `sp.wild_cluster_bootstrap`, `sp.wild_cluster_ci_inv` |
| **Synthetic control** | poor pre-fit | pre-RMSPE / pre-period SD > 0.6 | `sp.synth_compare`, `sp.augsynth`, `sp.synth_sensitivity` |
| **RD** | manipulation | density test (`sp.rddensity`) p < 0.05 | `sp.rddensity`, `sp.rdplotdensity`, `sp.rdrandinf` |
| **Matching** | residual imbalance | max post-match SMD > 0.25 (Stuart 2010) | `sp.ebalance`, `sp.cbps`, `sp.love_plot` |
| **Matching / IPW** | overlap | min propensity weight share < 0.05 | `sp.trimming` |
| **DML / AIPW** | weak overlap | > 5% of units at the trimming bound | `sp.trimming`, `sp.overlap_weights`, `sp.cbps` |
| **Logit / probit** | (quasi-)separation | \|slope coef\| > 15 (Albert-Anderson 1984) | penalised logit / drop the separating predictor |
| **Poisson** | over-dispersion | Pearson dispersion > 1.5 | `sp.nbreg`, robust SEs (quasi-Poisson) |
| **Count** | excess zeros | observed − predicted zeros > 0.05 | `sp.zip_model`, `sp.zinb` |
| **Heckman** | no selection | inverse-Mills p > 0.10 | `sp.regress` (more efficient) |
| **Heckman** | unstable rho | \|rho\| > 0.99 (weak exclusion restriction) | strengthen the exclusion; compare `sp.regress` |
| **Tobit** | extreme censoring | > 90% of observations censored | `sp.heckman`, report bounds |
| **Cox** | non-proportional hazards | Schoenfeld PH test p < 0.05 | time interaction / stratify, `sp.aft` |
| **Bayesian** | non-convergence | r̂ > 1.01, bulk-ESS < 400, or divergences > 0 | more draws / chains, reparameterise |
| **Any** | numerical | non-positive or non-finite standard errors | check collinearity (`sp.vif`), sandwich setup |

The thresholds live in one place (`statspai.core._agent_summary`) and are shared
by `violations()` and `audit()`, so the two can never disagree, and a future
correctness fix that moves a cutoff moves both at once.

## Using it in a workflow

```python
res = sp.dml(df, y="y", treat="d", covariates=X, model="irm")

for v in res.violations():
    if v["severity"] in ("warning", "error"):
        print(v["message"])
        print("  → try:", ", ".join(v["alternatives"]))

# Or get the full reviewer view, sorted by how ready the result is:
report = sp.audit(res)
report["coverage"]        # passed / total, in [0, 1]
report["summary"]         # {passed, failed, missing, n_total}
```

For the checks that *re-run* against the data (rather than inspecting stored
diagnostics), see `sp.assumption_audit`, and for the design-level robustness
sweep see the [robustness workflow](robustness_workflow.md) guide.
