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

Audited 2026-06-10 at 64 materialized modules; re-audited 2026-09-05 at 87 modules after the JSS parity-closure pass (Sun–Abraham, GLMM, PPML-HDFE, panel SFA, VAR, TWFE event study) and the addition of modules 86 (fect) and 87 (interflex). Re-audited 2026-09-22 at 89 modules (88 `rdbwselect` and 89 `rdms` joined the machine tier; 27 `glmm_aghq` moved to the iterative tier, see the loosening record below). The budget is enforced by
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
   see `verify_reproduce.py::REPRO_TOL_OVERRIDE`). 79 of 89 modules
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

*2026-10:* `43_heckman` `rel_se` `5e-4` → `1e-6`. The `8.6e-5` gap the
budget covered was not optimiser noise. The two-step variance used the
expected information of the first-step probit, while R `sampleSelection`
and Stata `heckman, twostep` use the observed information. With the
observed information the worst gap is `5.6e-11` against R and `4.3e-9`
against Stata. This is the decision tree's step 1: a gap that had been
given a tolerance was bisected and turned out to be on our side.

*1.32:* `40_qreg` `rel_se` `0.10` → `1e-6` (removed from the ≥ `5e-2`
table). The old row was graded A as "different sparsity estimators by
construction": StatsPAI's only SE was a Silverman-bandwidth Gaussian-kernel
iid sandwich (7.3% / 3.0% off), and it described quantreg's `se="nid"` as
Stata `qreg`'s default, which it is not (Stata's default is `vce(iid)` with
the fitted-quantile sparsity). `sp.qreg` now offers `vce="iid"` (Stata's
default, and the new default), `"robust"` (Stata `vce(robust)`) and
`"nid"` (quantreg); the module runs `vce="nid"` against quantreg `se="nid"`
(`1e-15`) and Stata `qreg, vce(robust)` (`2e-8`: Stata zeroes fitted
differences below `sqrt(eps)` where quantreg subtracts `sqrt(eps)`).

*1.32:* `41_tobit` `rel_se` `1e-5` → `1e-6`. The `2.0e-6` gap above was
the second-difference Hessian and BFGS's `gtol=1e-6` stop; `sp.tobit` now
Newton-polishes on complex-step scores (the shared ML path of `truncreg` /
`biprobit`), and the worst joined gap is `3.4e-11` (R) / `1.1e-11` (Stata).

## Row-level gap register (2026-10-27)

A module budget says how far its rows may drift. It does not say why a row
is not at machine level, and a budget justified by one row can cover
another. The Oaxaca Stata SE sat 1.2% away for months under a `5e-2` budget
written for the R bootstrap, and nobody had asked why Stata's delta method
disagreed with ours. Since 2026-10-27 every joined row of a T2 module whose
gap exceeds `1e-6` (relative, and `1e-8` absolute) must match an entry of
`compare.py::ROW_GAP_NOTES`. This applies on either reference side, for the
estimate and for the SE. Each entry is one of three kinds:

- **convention**: the reference computes a documented different quantity.
  The entry names `pinned_by` rows that run StatsPAI under the reference's
  convention, and these rows must join at machine level.
- **reference_numerics**: the reference's optimiser, root finder,
  numerical derivative or approximate covariance stops short. Where a
  second reference or a converged run exists, it is the pin.
- **stochastic_reference**: a simulated or resampled quantity. Such a row
  is a stochastic screen and is never a parity fact.

Three contract tests hold the table to the artifacts. Every row over
machine level must be registered, no entry may be stale, and every pin must
join at machine level. The SE budget tests skip registered rows, so 18
modules' SE budgets could drop to `1e-6`.

Building the register found the following:

| Module | Before | Mechanism found | Pin |
|---|---|---|---|
| `30_oaxaca` Stata SE | 1.2% under R's bootstrap budget | classical vs HC1 group regressions, and stochastic vs fixed covariate means | `oaxaca, vce(robust) fixed`: 2e-14 on the three threefold components |
| `76_pretrends` | 4e-4, attributed to `pmvnorm` noise; iterative tier | `pretrends` fixes the cutoff at 1.96, `pmvnorm` runs at abseps 1e-3, `uniroot` at its default tolerance | `threshold=1.96` against an exact integrator and a 1e-14 root: 1.6e-8; machine tier |
| `48_probit` R SE | 0.3% | `glm` reports expected information | `sampleSelection::probit` (observed information): 1e-12 |
| `42_nbreg` R SE | 0.14% | `glm.nb` holds θ at its estimate | `model_info['se_conditional_on_dispersion']`: 3e-13 |
| `61_betareg` R SE | 0.7% | `betareg` reports expected information | `hessian = TRUE`: 1.2e-7 (accuracy of `optim`'s numerical Hessian) |
| `51_newey` Stata SE | 0.5% under a `1e-2` budget | `newey` applies N/(N−K) | `hac_small=True` / `adjust = TRUE`: 4e-16 |
| `67_panel_glm` R SE | 1e-5 | `fixest` stops IWLS at `glm.tol = 1e-8` | `glm.tol = 1e-12`: 1e-15 (logit), 1e-7 (Poisson) |
| `28_frontier` | 1.3e-5 off R *and* Stata | StatsPAI: round-off of a fixed-step Hessian | fixed in the package (Richardson Hessian + Newton finish): 1e-8 |
| `29_panel_sfa` Stata | 3e-6 | StatsPAI stopped short; fixed-step Hessian | same fix: 2e-10 / 8e-10 |
| `07_scm` Stata | 4.2e-4, read as solver tolerance | `synth` builds `e(Y_synthetic)` from weights rounded to 0.001 | StatsPAI weights rounded the same way: 5e-16 (`*_w3dp`) |
| `12_sdid` Stata | 7.2e-8 | `sdid` stores `e(ATT)` via `strofreal()` (7 digits) | full value read from Mata: 1e-15 |

What the fixtures themselves turned up: `%.16g` does not round-trip every
double. The Python side used to fit the in-memory frame, while R and Stata
fitted the parsed text. `dump_csv` now swaps in the correctly rounded parse,
and Stata's `import delimited` returns exactly those doubles. R's `read.csv`
(and `readr::parse_double`) differs in the last bit on 10–36% of values on
the fixtures measured (03, 13, 28, 89). R therefore sees an input
perturbation of about 1e-16 relative. That is the floor below which a py–R
agreement says nothing, and it is far below every budget.

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
| `07_scm` | `rel_est` | 1.0 | A (T4) | headline 0.0239 vs R; donor weights up to 1.78 vs R, 0.0072 vs Stata | Classical SCM on the Basque fixture (`abadie2003economic`; method `abadie2010synthetic`). The donor-weight solution is *not unique* under the ADH nested-V specification: multi-start diagnostics find multiple near-best weight classes, and R `Synth::synth` and Stata `synth` land on measurably different local optima. StatsPAI's native solver tracks Stata. Verdict is GAP (T4 reference-disagreement disclosure), not PASS; module `52_scm_unique` certifies exact recovery on an identified DGP. |
| `13_causal_forest` | `rel_est` | 0.01 | B (T3) | 0.0026 / — | AIPW doubly robust ATE/ATT on both sides (`sp.causal_forest` vs `grf::causal_forest` + `average_treatment_effect`; `wager2018estimation`, `athey2019generalized`). The two forests cannot share RNG, so this row is one draw per engine; the T3 grade rests on refitting both engines under 50 seeds on fixed data (`tests/reference_parity/test_grf_seed_mc_equivalence.py`): their gap is below 0.1 sampling SE (TOST) at 500, 2,000 and 8,000 trees on this design and on a second fixture, and the two engines' seed-to-seed SDs agree within a factor of two (seeds spaced 1e5 apart; grf's consecutive seeds share most of their draws). (Before 1.31 the row was graded by dividing the single-draw gap by the combined *sampling* SE, which is not the forest's Monte Carlo error.) A multi-seed truth-recovery pytest guard backs it. Widened from 0.005 when the ATT row moved onto grf's own plug-in + Hajek-correction estimator: both sides now run the *same* estimator, so the residual is pure forest MC, and 0.005 left only 1.07× margin against machine-to-machine drift at the time. Since 1.31 (independent per-group seeds and per-nuisance random streams in the engine) the ATE row sits at 0.03% and the ATT row at 0.26%. Margin 3.9×. |
| `13_causal_forest` | `rel_se` | 0.05 | B | 0.0068 / — | The AIPW *operator* is now pinned exactly — fed grf's own forest outputs, StatsPAI reproduces `grf::get_scores` elementwise to 2.3e-14 and grf's ATE/ATT estimate and `std.err` to 1e-15 (`tests/reference_parity/test_grf_aipw_operator_parity.py`) — so this budget covers forest RNG only, not an unresolved formula difference. Tightened from 0.50 to 0.25 after the ATT convention fix removed the historical 14.6% ATT row, and to 0.05 in 1.29 when the GRF engine brought the worst row to 0.63%. Since 1.31's seeding fix the ATE SE is 0.16% and the ATT SE 0.68% from grf's. Margin 7.3×. |

Only the T3 forest and the T4 SCM remain in this table. On 2026-10-27
`05_sunab`, `29_panel_sfa`, `30_oaxaca` and `36_mediation` left it: their
SE budgets are now `1e-6`, and the rows that are not at machine level are
registered one by one in `compare.py::ROW_GAP_NOTES` (see "Row-level gap
register" below). `06_rd`'s `0.10` entry was already stale; its SE budget has
been `1e-6` since the native selector replaced the port. All remaining
entries are at `3e-2` or tighter and are graded inline in `compare.py`. Modules `26_glmm_logit`, `27_glmm_aghq` and
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
| `11_psm` | `rel_se` | 5.0 | 1e-6 | sentinel (no joined `se` field) | `att_psm` carries `se=None` on all three sides *by design*; see "Sentinel entries". Since 2026-09-22 the point-valued `se_teffects_ai` row joins Stata under `rel_est` at 7e-8 (`se_method='abadie_imbens_2016'`, the Abadie-Imbens 2016 estimated-score variance `teffects psmatch` reports); the default `se_abadie_imbens` (psmatch2 `ai(1) ties`, 640.41 vs 621.79) is a documented convention -- sample- vs population-ATT target and no estimated-score term. |
| `12_sdid` | `rel_se` | 5e-2 | 1e-6 | sentinel | Point-only ATT row (`arkhangelsky2021synthetic`); placebo SEs are backend-native diagnostics under distinct names. |
| `16_bjs` | `rel_se` | 0.25 | 1e-6 | sentinel | BJS imputation (`borusyak2024revisiting`); SE rows are side-specific (`se_cluster_if` / `se_didimputation` / `se_stata_did_imputation`). |
| `07_scm` | `rel_se` | 1.0 | 1e-6 | sentinel | All SCM rows are point-only. |
| `18_augsynth` | `rel_se` | 1.0 | 1e-6 | sentinel | `augsynth` fixture (`benmichael2021augmented`) emits no joinable SE. The `rel_est` budget (2e-5) is now grade **A**: the 7.9e-6 R residual is `augsynth::synth_qp`'s OSQP stopping tolerance (eps 1e-8); with OSQP at 1e-13 augsynth agrees with StatsPAI's exact simplex QP and the audited Stata/Mata bridge (`18_augsynth.do`) to 1e-11. |
| `19_gsynth` | `rel_se` | 1.0 | 1e-6 | sentinel | `gsynth` fixture (`xu2017generalized`) emits no joinable SE. An audited Stata/Mata bridge (`19_gsynth.do`, gsynth's control-only factor convention with r fixed at the R `r.cv`) joins the ATT at 2.2e-15; `fect_stata method(ife)` is a different estimator (IFEct) and is kept as an unjoined diagnostic. |
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

## Closed 2026-09-22 (no tolerance change)

- **`73_did2s` SE promoted to T2.** `sp.gardner_did`'s default
  `vce='analytic'` now computes the did2s corrected two-stage clustered
  variance (stage-1 estimation error propagated through
  `γ = (X10'X10)^{-1} X1'X2`), matching R `did2s` at 2.7e-10 and Stata
  `did2s` at 1.3e-14 on the static ATT and all 26 event-study horizons
  at 2.1e-14. `rel_se` registered at the 1e-6 machine tier; the
  pre-1.29 stage-2-only SE (25.9% low) is reachable as `vce='stage2'`
  and kept as the unjoined `static_ATT_stage2_se` row.
- **`04_csdid` Stata SE convention pinned as a row.**
  `group_overall_fixedshare` = `sp.aggte(share_variance=False)` joins
  `csdid`'s `estat group` GAverage at 6.2e-15; `STATA_SE_GAP_NOTES`
  keeps the 0.27% note for the default row only.
- **`18_augsynth` / `19_gsynth` Stata bridges materialised** (see the
  sentinel table above); `STATA_SKIP_REASON` now lists four modules
  (13, 77, 79, 80). Stata-side reproducibility: 85/85 modules, zero drift.

## Loosening applied 2026-09-22: `27_glmm_aghq` point estimate 1e-6 → 5e-6 (grade B)

The only budget ever loosened under the audit rule, registered here
with the measurement that justifies it.

- **The likelihood is shared.** The `logLik` row joins R `lme4::glmer`
  (`nAGQ = 8`) at rel 1e-13, and `lme4`'s deviance function
  (`devFunOnly = TRUE`) evaluated at StatsPAI's `(θ, β)` is within
  4.7e-11 of its value at `lme4`'s own optimum. Both sides compute the
  same 8-point adaptive Gauss–Hermite approximation.
- **The objective is flat.** Along the (intercept, √var) direction a
  5e-11 deviance step moves the intercept by 5e-7, i.e. the argmax is
  numerically undetermined at the 1e-6 level for a 1500-observation
  log-likelihood evaluated in double precision.
- **The reference's own optimisers disagree at that level.** On the
  committed fixture `lme4` with `bobyqa` (the script's setting, `rhoend
  = 1e-12`), `nloptwrap` (`xtol_abs = ftol_abs = 1e-14`),
  `optimx/nlminb` and `Nelder_Mead` return intercepts 1.1e-6, 1.5e-6 and
  1.4e-5 apart from one another while all reproduce the deviance to
  1e-10. Stata `melogit, intpoints(8)` (default `mvaghermite`, tight
  `tolerance(1e-10) ltolerance(1e-13)`) lands 2e-8 from StatsPAI on the
  intercept; its `logLik` differs by 5e-6 because Stata's mean–variance
  adaptive grid is a different 8-point approximation, and all three
  sides agree to 3e-8 at 30 points (`intmethod(mcaghermite)` gives the
  same 30-point numbers to 1e-15).
- **Budget.** 5e-6 ≈ 3× the widest `lme4`-vs-`lme4` spread that still
  reproduces the deviance to 1e-10. The module therefore reports in the
  *iterative* strictness tier (79 machine / 8 iterative / 1 moderate /
  1 T4 as of this change), not the machine tier. The SE budget (2e-5)
  is unchanged; observed 4.8e-7 (R) / 4.2e-6 (Stata).
- **Guard.** `test_glmm_parity_uses_tight_optimizer_solution_with_se_convention_guard`
  additionally asserts `logLik` rel < 1e-9, so a future drift in the
  quadrature itself cannot hide behind the optimiser-noise budget.

## Sentinel entries

Several modules are *point-only* by design: their SE estimators differ
by construction, so the harness stores them as side-specific diagnostic
rows with distinct statistic names that never join, and the headline
row carries `se=None`. Example — module `11_psm`: StatsPAI reports the
Abadie–Imbens (2006, eq. 14) sample-ATT SE (`se_abadie_imbens` = 640.41),
the estimator Stata `psmatch2, ai(J) ties` reports; `MatchIt::matchit`
documents no canonical analytic SE for nearest-neighbor matching with
replacement, so the R fixture records a weighted-`lm`-on-matched-data
diagnostic (`se_matchit_lm`); Stata `teffects psmatch` reports its own
Abadie–Imbens variant (`abadie2016matching`, `se_teffects_ai` = 621.79),
3.4% below ours. That gap was localised on 2026-09-22 (previously an
open item under the §5 rule): `teffects psmatch` reports the Abadie &
Imbens (2016) variance for matching on the *estimated* propensity score,
`σ²_δ − c'V_γ c + d'V_γ d`, whose two extra terms (−74 905 + 8 036 on
NSW-DW) are negative here, and whose base term targets the population
ATT (it includes the `(τ_i − τ)²` heterogeneity term) rather than the
sample ATT of eq. 14. `sp.match(se_method='abadie_imbens_2016')` rebuilds
the number from StatsPAI's own matched frame, within-arm conditional
variances and logit information matrix to rel 6.4e-8 (the residual is
Stata's ML stopping rule), so the point-valued `se_teffects_ai` row now
*joins* Stata under the module's `rel_est` budget of 1e-6. The default
`se_abadie_imbens` (psmatch2 `ai(1)`) is unchanged and documented as the
sample-ATT, known-score convention.

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
6. **`13_causal_forest` `rel_se` 0.05 — T3 by construction.** The
   residual is forest-RNG dispersion in `tau.hat` and cannot reach T2;
   the binding evidence is the exact AIPW-operator pin plus the Track B
   coverage sweep, not the relative band (observed worst 0.68% since
   1.31).
7. ~~**Headline-only enforcement.**~~ **Closed.**
   `test_every_r_se_row_is_inside_budget` gates the registered `rel_se`
   on *every* R-joined SE row of every PASS module (all 87 pass), and
   `test_every_stata_se_row_is_inside_budget_or_registered` requires any
   Stata SE row over budget to carry a mechanism in
   `compare.py::STATA_SE_GAP_NOTES` (two entries: `04_csdid`
   `group_overall` 0.27%, fixed-share aggregation; `71_dml_family`
   PLIV, `ivreg`'s `N/(N-K)`. The `83_lpdid` lead-row entry, 3.9e-4 and
   attributed to `reghdfe`'s K, was removed on 2026-09-07: the gap was a
   StatsPAI lead-sample rule one period stricter than `lpdid`'s, and the
   rows now agree to 2e-16).
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

*Last audited: 2026-10-27 (row-level register; 89 parity modules). Previously 2026-09-22 (StatsPAI 1.29.0 source snapshot). Re-run
the snippet above and refresh this document whenever a `TOLERANCES`
entry changes.*
