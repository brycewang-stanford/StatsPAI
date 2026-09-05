# R-parity tolerance audit

`tests/r_parity/compare.py::TOLERANCES` is the single pre-registered
tolerance budget for the StatsPAI ↔ R (and, where materialized,
↔ Stata) parity harness. Every number in that table is a *claim*: "on
the committed golden fixtures, StatsPAI and the canonical reference
agree to within this relative difference." A reviewer who sees a 500%
entry with no justification is right to discount the whole table, so
this document grades every loose entry, records the gap actually
observed in the committed `results/*.json` artifacts, and lists —
honestly — the entries we cannot yet justify.

Audited 2026-06-10 at 64 materialized modules; re-audited 2026-09-05 at 87 modules after the JSS parity-closure pass (Sun–Abraham, GLMM, PPML-HDFE, panel SFA, VAR, TWFE event study) and the addition of modules 86 (fect) and 87 (interflex). The budget is enforced by
`tests/test_parity_harness_contract.py::test_headline_passes_are_inside_registered_r_tolerance`
(R side) and
`test_stata_headline_over_budget_modules_are_explicitly_registered`
(Stata side); the golden JSONs themselves are hash-locked by
`TIER_A_FIXTURE_LOCK.json`, so neither side of any comparison can drift
silently.

## The three tolerance tiers

Every entry belongs to one of three regimes. Conflating them in a
single column is what makes a tolerance table look arbitrary, so name
the regime explicitly:

1. **Machine precision (`1e-6` and below).** Same estimand, same
   convention, closed form or tightly converged optimizer. The
   residual is floating-point noise (typically `1e-15` to `1e-9`;
   cross-BLAS reassociation in sandwich "meat" sums can reach `~1e-8`,
   see `verify_reproduce.py::REPRO_TOL_OVERRIDE`). 78 of 87 modules
   register here on the point estimate.
2. **Convention gap (`1e-4` to `5e-2`).** Both implementations are
   correct, but they compute a *documented* different quantity:
   degrees-of-freedom divisors (`T` vs `T−k`), small-sample cluster
   corrections (`ssc`), expected vs observed information, analytic vs
   influence-function SEs. The budget bounds the size of the named
   convention difference, and the mechanism must be stated next to the
   entry.
3. **Methodological (T3/T4, above `5e-2`).** The two sides *cannot*
   agree deterministically: independent forest RNG (combined Monte
   Carlo error), bootstrap vs delta-method inference, non-unique SCM
   donor weights. The budget bounds a documented methodological
   disagreement, the verdict is graded T3/T4 rather than treated as an
   ordinary deterministic pass, and the residual-noise source is
   recorded in the module's `extra` block.

Orthogonal to all three is the *reproducibility* tolerance
(`verify_reproduce.py`, `1e-9`): same code, same data, same packages
must reproduce the committed golden values nearly bit-exactly. A parity
tolerance never excuses a reproducibility drift.

## Grading scheme

Each entry at or above `5e-2` (plus the formerly loose entries) gets a
grade:

- **A — mechanistic.** The two sides compute different, individually
  documented quantities by construction. The specific method on each
  side is named (from our source and the R function's documented
  method).
- **B — empirical.** Like-for-like comparison with residual numerical
  or Monte Carlo noise; we report the observed gap and the margin
  (tolerance ÷ observed gap).
- **C — unjustified.** Flagged for future work; no mechanism pinned
  and/or the budget does not actually bound the rows it appears to
  cover.

**Audit rules.** Never loosen a value without re-registering it here.
Tighten when the observed gap (recomputed from the committed JSONs,
worst across the R *and* Stata sides) is more than 5× smaller than the
budget, to ≈3× the observed gap rounded to a clean number, floored at
the harness-wide `1e-6` machine tier.

*Scope of the 2026-06 audit round:* the 5× rule was first applied to
the legacy loose tier (budgets ≥ `1e-2`). A follow-up pass then swept
the mid-tier modules whose margin also exceeded 5×, recomputing the
worst observed `rel_se` across **both** reference sides from the
committed JSONs and tightening each to ≈3× that gap (floored at the
`1e-6` machine tier):

| module | old `rel_se` | obs worst (both sides) | new `rel_se` |
| --- | --- | --- | --- |
| `42_nbreg` | `1e-2` | `1.4e-3` | `5e-3` |
| `43_heckman` | `1e-3` | `8.6e-5` | `5e-4` |
| `28_frontier` | `1e-4` | `1.3e-5` | `5e-5` |
| `44_mlogit` | `1e-4` | `1.2e-5` | `5e-5` |
| `41_tobit` | `1e-3` | `2.0e-6` | `1e-5` |
| `14_ols_cluster` | `1e-3` | `6.1e-9` | `1e-6` |
| `24_coxph` | `1e-3` | `2.6e-15` | `1e-6` |
| `46_clogit` | `1e-3` | `2.7e-9` | `1e-6` |
| `49_oprobit` | `1e-3` | `3.0e-7` | `1e-6` |

`45_ologit` (margin 5.1×, obs `2.0e-6`) sits at the rule boundary and
is left at `1e-5`. No value was loosened; the harness contract test
and the offline render both pass at the new budgets.

## Reproducing the audit

Run from the repository root. This recomputes, for every module, the
worst joined SE gap across both reference sides and flags sentinel
(no-joined-SE-row) budgets and over-budget non-headline rows:

```python
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "compare", Path("tests/r_parity/compare.py")
)
compare = importlib.util.module_from_spec(spec)
sys.modules["compare"] = compare
spec.loader.exec_module(compare)

for module, tol in sorted(compare.TOLERANCES.items()):
    rows = compare.collect(module)
    worst_se = max(
        (v for d in rows for v in (d.rel_se, d.rel_se_st) if v is not None),
        default=None,
    )
    budget = tol.get("rel_se", tol.get("abs_se"))
    if worst_se is None:
        print(f"{module:22s} rel_se budget {budget:8.2g}  no joined SE row (sentinel)")
    elif worst_se > budget:
        print(f"{module:22s} rel_se budget {budget:8.2g}  OVER: observed {worst_se:.3g}")
```

The two enforcement commands (both must pass after any tolerance
change):

```bash
# from the repository root
python3 tests/r_parity/compare.py          # re-render parity tables (idempotent)
python3 -m pytest tests/test_parity_harness_contract.py -q -o addopts=''
```

## Entries at or above 5e-2 — full justification

Observed gaps are recomputed from the committed
`tests/r_parity/results/<module>_{py,R}.json` and
`tests/stata_parity/results/<module>_Stata.json` artifacts (worst row;
R side / Stata side). "Margin" is tolerance ÷ worst observed gap.

| Module | Quantity | Tolerance | Grade | Observed gap (R / Stata) | Justification |
|---|---|---:|:---:|---|---|
| `05_sunab` | `rel_se` | 3e-2 | A | 0.0093 / 8e-12 | Sun–Abraham (`sun2021estimating`) event-time IW SEs. The default `att_rel_<e>` rows carry the Prop. 3 cohort-share term (Stata `eventstudyinteract` convention; py == Stata to 8e-12 on every row), while `fixest::sunab` treats the shares as fixed, so R differs by exactly that positive term at multi-cohort relative times (worst 0.93% at e=1) and by 8e-12 at single-cohort times and on the `agg="att"` ATT. The `att_rel_<e>_fixedshare` rows (`share_variance=False`) pin the fixest convention itself to 8e-12. Both sides use the fixest/reghdfe nested-K rule (12 observed cells + 5 year effects = 17 on mpdta); before 1.24.0 StatsPAI counted 9 unobserved cells and no time effects, which was the whole of the old "unpinned" 0.08% base gap. Margin 3.2×. |
| `06_rd` | `rel_se` | 0.10 | A | 0.0671 / 0.0671 | Default-bandwidth rows delegate to the official CCT port (`calonico2014robust`) and match R/Stata `rdrobust` at ~1e-12. The budget is bound only by the deliberately retained *legacy internal* SE rows at a forced common `h` (`forced_h*` diagnostics, observed 6.7%), kept to document the pre-delegation convention. Margin 1.5×. |
| `07_scm` | `rel_est` | 1.0 | A (T4) | headline 0.0239 vs R; donor weights up to 1.78 vs R, 0.0072 vs Stata | Classical SCM on the Basque fixture (`abadie2003economic`; method `abadie2010synthetic`). The donor-weight solution is *not unique* under the ADH nested-V specification: multi-start diagnostics find multiple near-best weight classes, and R `Synth::synth` and Stata `synth` land on measurably different local optima. StatsPAI's native solver tracks Stata. Verdict is GAP (T4 reference-disagreement disclosure), not PASS; module `52_scm_unique` certifies exact recovery on an identified DGP. |
| `13_causal_forest` | `rel_est` | 0.01 | B (T3) | 0.0047 / — | AIPW doubly robust ATE/ATT on both sides (`sp.causal_forest` vs `grf::causal_forest` + `average_treatment_effect`; `wager2018estimation`, `athey2019generalized`). The two forests cannot share RNG, so the row is graded against *combined Monte Carlo error* (~0.05 combined SE on the clean-overlap DGP); a multi-seed truth-recovery pytest guard backs it. Widened from 0.005 when the ATT row moved onto grf's own plug-in + Hajek-correction estimator: both sides now run the *same* estimator, so the residual 0.47% is pure forest MC, and 0.005 left only 1.07× margin against machine-to-machine drift. The ATE headline sits at 0.19%. Margin 2.1×. |
| `13_causal_forest` | `rel_se` | 0.25 | B | 0.077 / — | The AIPW *operator* is now pinned exactly — fed grf's own forest outputs, StatsPAI reproduces `grf::get_scores` elementwise to 2.3e-14 and grf's ATE/ATT estimate and `std.err` to 1e-15 (`tests/reference_parity/test_grf_aipw_operator_parity.py`) — so this budget covers forest RNG only, not an unresolved formula difference. Tightened from 0.50 after the ATT convention fix removed the historical 14.6% ATT row (now 0.087%); the remaining worst is the ATE row at 7.7%, whose SE depends on the forest's own `tau.hat` dispersion. Margin 3.2×. |
| `29_panel_sfa` | `rel_se` | 5e-2 | B | 0.0184 / 2.9e-6 | Half-normal Pitt–Lee panel SFA (`pitt1981measurement`). All three sides now fit the same likelihood: the Stata do-file constrains `xtfrontier, ti`'s truncated-normal `mu` to 0 (before 2026-09 the Stata rows were a different model, mislabelled as a "scale" difference; the intercept and `sigma_u` now agree to 1e-6). Stata's analytic-Hessian OIM (`ml` method `d2`) matches StatsPAI's central-difference OIM to 3e-6 on every SE row, which pins the Python Hessian as exact; `frontier::sfa`'s `mleCov` (Coelli's FRONTIER 4.1 routine) is 0.1%–1.8% off, worst on the intercept. Budget = 3× the worst R-side row; margin 2.7×. |
| `30_oaxaca` | `rel_se` | 0.05 | A | 0.0125 / 0.0122 | Blinder–Oaxaca (`blinder1973wage`, `oaxaca1973male`; cf. `jann2008blinder`). StatsPAI reports closed-form delta-method SEs (`src/statspai/decomposition/oaxaca.py`); `oaxaca::oaxaca` reports seeded bootstrap SEs with `R=100` replications, whose own Monte Carlo noise is ~`(2R)^{-1/2}` ≈ 7% of the SE. Tightened 2026-06-10 from 1.0 (4× margin); a future regeneration that changes the bootstrap RNG stream may legitimately require re-registration. |
| `36_mediation` | `rel_se` | 0.10 | A | 0.0701 / 0.0321 | Causal mediation (`imai2010general`). StatsPAI uses bootstrap inference (B=1000); `mediation::mediate` uses quasi-Bayesian Monte Carlo with `sims=200` (~5% MC noise by itself); the Stata bridge uses delta-method SEs. Different inference algorithms by construction; point effects match at 1e-15. Margin 1.4×. Frozen by the contract test. |
| `40_qreg` | `rel_se` | 0.10 | A | 0.0734 / 0.0302 | Median regression (`koenker2005quantile`). StatsPAI uses the Powell-type iid kernel sandwich (`src/statspai/regression/quantile.py`, kernel estimate of the residual density at zero); the R fixture deliberately reports `summary(rq, se="nid")` — the Hendricks–Koenker difference-quotient sandwich — chosen to match Stata `qreg`'s default. Different sparsity estimators by construction. Margin 1.4×. |

All remaining entries are at `3e-2` or tighter and are graded inline in
`compare.py`. Modules `26_glmm_logit`, `27_glmm_aghq` and
`47_ppmlhdfe_3fe` left this table on 2026-09-05: their 1.9% / 1.9% /
1.8% "convention" gaps were StatsPAI defects (fixed-effect covariance
conditional on the variance components; a small-sample factor matching
neither reference) and are now at `1e-2` / `2e-5` / `1e-6`.

## Tightenings applied 2026-06-10

Rule applied: observed gap (worst over R and Stata sides, all joined
rows — stricter than the headline-only enforcement) more than 5×
smaller than the budget → tighten to ≈3× observed, floored at the
harness machine tier `1e-6`. **No value was loosened.**

| Module | Quantity | Old | New | Observed worst gap | Basis |
|---|---|---:|---:|---|---|
| `03_hdfe` | `rel_se` | 1e-2 | 1e-6 | 8.4e-15 | The "1-df convention gap" comment was stale: with `ssc='fixest'`, IID SEs match `fixest::feols`/`reghdfe` at machine level on both sides. |
| `15_hdfe_cluster` | `rel_se` | 5e-2 | 1e-6 | 1.25e-11 | Stale "ssc convention": CR1 nested-FE cluster SEs now match on both sides. |
| `30_oaxaca` | `rel_se` | 1.0 | 0.05 | 1.25e-2 | Grade-A delta-vs-bootstrap gap; 3× observed ≈ 0.0375, rounded to 0.05. |
| `11_psm` | `rel_se` | 5.0 | 1e-6 | sentinel (no joined SE row) | `att_psm` carries `se=None` on all three sides *by design*; see "Sentinel entries". |
| `12_sdid` | `rel_se` | 5e-2 | 1e-6 | sentinel | Point-only ATT row (`arkhangelsky2021synthetic`); placebo SEs are backend-native diagnostics under distinct names. |
| `16_bjs` | `rel_se` | 0.25 | 1e-6 | sentinel | BJS imputation (`borusyak2024revisiting`); SE rows are side-specific (`se_cluster_if` / `se_didimputation` / `se_stata_did_imputation`). |
| `07_scm` | `rel_se` | 1.0 | 1e-6 | sentinel | All SCM rows are point-only. |
| `18_augsynth` | `rel_se` | 1.0 | 1e-6 | sentinel | `augsynth` fixture (`benmichael2021augmented`) emits no joinable SE. |
| `19_gsynth` | `rel_se` | 1.0 | 1e-6 | sentinel | `gsynth` fixture (`xu2017generalized`) emits no joinable SE. |
| `20_bacon` | `rel_se` | 1.0 | 1e-6 | sentinel | The Goodman–Bacon decomposition (`goodmanbacon2021difference`, `goodmanbacon2019bacondecomp`) has no SEs on any side. |
| `31_dfl` | `rel_se` | 1.0 | 1e-6 | sentinel | Point-only decomposition rows. |
| `39_arima` | `rel_se` | 1e-2 | 1e-6 | sentinel | No SE row joins on this fixture. |
| `52_scm_unique` | `rel_se` | 1.0 | 1e-6 | sentinel | All rows point-only. |

Verification: after these edits,
`python3 tests/r_parity/compare.py` re-rendered all parity tables
byte-identically (the budget does not enter any rendered artifact for
`rel_se`-only changes, and no `rel_est` was touched, so the strictness
tiers and the JSS Appendix B tables are unchanged), and
`python3 -m pytest tests/test_parity_harness_contract.py -q -o addopts=''`
passes every tolerance-related contract. A row-level checker asserting
the new values against *all* joined SE rows on both reference sides
(not just headline rows) also passes for every tightened entry.

## Sentinel entries

Several modules are *point-only* by design: their SE estimators differ
by construction, so the harness stores them as side-specific diagnostic
rows with distinct statistic names that never join, and the headline
row carries `se=None`. Example — module `11_psm`: StatsPAI reports the
matched-pair effect dispersion (`se_pair_effect`); `MatchIt::matchit`
documents no canonical analytic SE for nearest-neighbor matching with
replacement, so the R fixture records a weighted-`lm`-on-matched-data
diagnostic (`se_matchit_lm`); Stata `teffects psmatch` reports the
Abadie–Imbens robust SE (`abadie2006large`, `abadie2011bias`,
`se_teffects_ai`).

For such modules the `rel_se` budget is **vacuous** — no value, loose
or tight, is ever exercised. The previous loose values (up to 5.0 for
`11_psm`) were leftovers from the original 2026-05-04 harness commit,
before the SE rows were split into side-specific diagnostics, and read
as if we tolerated a 500% SE gap. They are now pinned at the `1e-6`
machine floor as **sentinels**: if a future fixture regeneration ever
makes an SE row join, the contract fails loudly and the maintainer must
consciously register a justified budget instead of inheriting a stale
loose one.

## Demonstrating a convention gap: the VAR df divisor

Module `33_var`'s SE budget illustrates why "both correct, different
convention" must be stated mechanistically. StatsPAI's default matches
Stata `var` (conditional-MLE divisor `T`); `vars::VAR` runs
per-equation `lm()` (divisor `T−k`). The entire R-side SE gap is the
deterministic ratio `sqrt(T/(T−k))`:

```python
import numpy as np
import pandas as pd
import statspai as sp

rng = np.random.default_rng(42)
n = 200
y1, y2 = np.zeros(n), np.zeros(n)
for t in range(2, n):
    y1[t] = 0.5 * y1[t - 1] - 0.2 * y2[t - 2] + rng.standard_normal()
    y2[t] = 0.3 * y2[t - 1] + 0.1 * y1[t - 1] + rng.standard_normal()
df = pd.DataFrame({"y1": y1, "y2": y2})

stata_side = sp.var(df, lags=2, se_df="stata")  # divisor T   (Stata var)
r_side = sp.var(df, lags=2, se_df="r")          # divisor T-k (vars::VAR)

T = stata_side.n_obs
k = 2 * 2 + 1  # 2 lags x 2 variables + constant per equation
ratio = np.asarray(r_side.se["y1"]) / np.asarray(stata_side.se["y1"])
print(f"observed SE ratio = {ratio.flat[0]:.6f} "
      f"(identical for every coefficient: {np.allclose(ratio, ratio.flat[0])})")
print(f"sqrt(T / (T - k)) = {np.sqrt(T / (T - k)):.6f}")
```

Output: `observed SE ratio = 1.012871 …` — the ratio−1 form (1.287%)
of the max-normalised 0.0127 gap recorded for every `33_var` SE row in
`parity_table.md` (the committed fixture has `T=198`, `k=5`,
`sqrt(198/193) − 1 = 1.287%`; the table normalises by the larger side).

## Known weak spots

The honest list, re-audited 2026-09-05. Every entry that used to sit
here as grade C or "budget bounds nothing" has been closed by locating
the mechanism on the StatsPAI side and, where a like-for-like row
exists, pinning both references at machine level. The closures are kept
below (struck through) because the *pattern* is the lesson; the open
items follow.

1. ~~**`29_panel_sfa` (`rel_se` 1e-3) — grade C.**~~ **Closed.** The Stata
   reference was a different likelihood (truncated-normal
   Battese–Coelli `ti` with free `mu`); constrained to `mu = 0` it agrees
   with StatsPAI to 3e-6 on every SE and 1e-6 on every point row, so the
   registered 5e-2 now bounds only `frontier`'s approximate covariance
   (graded B above).
2. ~~**`33_var` (`rel_se` 1e-3) — budget mis-keyed.**~~ **Closed.** The
   Python side emits `eq_*` (divisor `T`, Stata) and `eq_*__Tk`
   (divisor `T-k`, R `vars::VAR`) rows and each reference side emits
   only its own convention; every compared SE is at 1e-15 and the budget
   is 1e-6.
3. ~~**`05_sunab` — fixest-side mechanism unpinned.**~~ **Closed.** The
   17.1% figure came from the old R script re-aggregating cohort×period
   cells by hand without their covariance; the native `fixest`
   aggregation is now used. The residual 0.08% at single-cohort times
   was a StatsPAI degrees-of-freedom defect (nine all-zero
   cohort×relative-time columns counted in K, time effects not counted),
   fixed in 1.24.0. What remains is the documented Sun–Abraham Prop. 3
   share term (mechanism A), demonstrated by the `att_rel_<e>_fixedshare`
   rows matching `fixest` to 8e-12 and the default rows matching Stata
   `eventstudyinteract` to 8e-12.
4. ~~**`26_glmm_logit` / `27_glmm_aghq` — SE convention not derived.**~~
   **Closed.** Not a convention: the fixed-effect covariance omitted the
   variance-component uncertainty. The full observed-information
   Hessian of the marginal log-likelihood (Stata `vce(oim)`) puts AGHQ at
   5e-7 (R) / 4e-6 (Stata) and Laplace at 4e-6 (Stata); the remaining
   0.27% on the Laplace R side is `lme4`'s own optimum sitting 1.5e-4
   away in β (its `logLik` is 2.8e-7 lower).
5. ~~**`47_ppmlhdfe_3fe` — R-side residual unpinned.**~~ **Closed.**
   StatsPAI applied `(N-1)/(N-k)` with `k` = slopes only, matching
   neither reference. `ssc="stata"` (`N/(N-1)`) matches `ppmlhdfe` to
   1e-11 and `ssc="fixest"` (`N/(N-K)`, K counting absorbed FE levels)
   matches `fixest::fepois` to 1e-8; modules 37 and 47 emit one row per
   convention.
6. **`13_causal_forest` `rel_se` 0.25 — open by construction (T3).** The
   residual is forest-RNG dispersion in `tau.hat` and is not
   tightenable; the binding evidence is the exact AIPW-operator pin plus
   the Track B coverage sweep, not the relative band.
7. ~~**Headline-only enforcement.**~~ **Closed.**
   `test_every_r_se_row_is_inside_budget` gates the registered `rel_se`
   on *every* R-joined SE row of every PASS module (all 87 pass), and
   `test_every_stata_se_row_is_inside_budget_or_registered` requires any
   Stata SE row over budget to carry a mechanism in
   `compare.py::STATA_SE_GAP_NOTES` (three entries: `04_csdid`
   `group_overall` 0.27%, fixed-share aggregation; `71_dml_family`
   PLIV, `ivreg`'s `N/(N-K)`; `83_lpdid` horizons, `reghdfe`'s K).
8. ~~**`04_csdid` `group_overall` Stata SE (0.27%) — open.**~~ **Closed
   (2026-09-05).** `csdid`'s `estat group` GAverage aggregates the
   per-cohort influence functions with the cohort shares held fixed;
   `did::aggte(type = "group")` and `sp.aggte` add the share-estimation
   term (`did:::wif`). Rebuilding the fixed-share aggregate from
   StatsPAI's *joint* cell influence functions reproduces `csdid`'s SE to
   1e-14 (the earlier independent-cell attempt overshot because it
   dropped the cross-cell covariance). Mechanism A, registered in
   `STATA_SE_GAP_NOTES` and pinned by
   `test_csdid_group_overall_stata_se_is_the_fixed_share_aggregation`.
9. **Contract-frozen values.** `tests/test_parity_harness_contract.py`
   asserts exact equality for several budgets (e.g. `26_glmm_logit`,
   `27_glmm_aghq`, `36_mediation`, `38_drdid`, `10_honest_did`,
   `11_psm` `rel_est`). Any future change must touch both files in the
   same commit, deliberately.

---

*Last audited: 2026-09-05 (StatsPAI 1.24.0 source snapshot, 87 parity modules). Re-run
the snippet above and refresh this document whenever a `TOLERANCES`
entry changes.*
