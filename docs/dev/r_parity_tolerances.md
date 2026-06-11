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

Audited 2026-06-10 at 64 materialized modules. The budget is enforced by
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
   see `verify_reproduce.py::REPRO_TOL_OVERRIDE`). 50 of 64 modules
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
| `05_sunab` | `rel_se` | 0.25 | B | 0.171 / 0.0084 | Sun–Abraham (`sun2021estimating`) event-time IW SEs. StatsPAI tracks the Stata `eventstudyinteract` clustered convention (≤0.9%); `fixest::sunab` with `agg="att"` aggregates cohort×period cells via a clustered delta method that differs by up to 17.1% on the sparse `mpdta` cohorts. Margin 1.5×. The fixest-side variance construction is not yet pinned line-by-line — also listed under weak spots. |
| `06_rd` | `rel_se` | 0.10 | A | 0.0671 / 0.0671 | Default-bandwidth rows delegate to the official CCT port (`calonico2014robust`) and match R/Stata `rdrobust` at ~1e-12. The budget is bound only by the deliberately retained *legacy internal* SE rows at a forced common `h` (`forced_h*` diagnostics, observed 6.7%), kept to document the pre-delegation convention. Margin 1.5×. |
| `07_scm` | `rel_est` | 1.0 | A (T4) | headline 0.0239 vs R; donor weights up to 1.78 vs R, 0.0072 vs Stata | Classical SCM on the Basque fixture (`abadie2003economic`; method `abadie2010synthetic`). The donor-weight solution is *not unique* under the ADH nested-V specification: multi-start diagnostics find multiple near-best weight classes, and R `Synth::synth` and Stata `synth` land on measurably different local optima. StatsPAI's native solver tracks Stata. Verdict is GAP (T4 reference-disagreement disclosure), not PASS; module `52_scm_unique` certifies exact recovery on an identified DGP. |
| `13_causal_forest` | `rel_est` | 0.005 | B (T3) | 0.0028 / — | AIPW doubly robust ATE/ATT on both sides (`sp.causal_forest` vs `grf::causal_forest` + `average_treatment_effect`; `wager2018estimation`, `athey2019generalized`). The two forests cannot share RNG, so the row is graded against *combined Monte Carlo error* (~0.05 combined SE on the clean-overlap DGP); a multi-seed truth-recovery pytest guard backs it. Margin 1.8× — not tightenable under the 5× rule. |
| `13_causal_forest` | `rel_se` | 0.50 | B | 0.146 / — | The AIPW variance estimate depends on the implementation-specific subsampled nuisance fits; with independent forests the SEs differ by up to 14.6% on this fixture. Margin 3.4× — below the 5× tightening threshold; the loosest live SE budget in the table (see weak spots). |
| `26_glmm_logit` | `rel_se` | 5e-2 | B | 0.0164 / 0.0190 | GLMM logit (Laplace), tight optimizer (`tol=1e-8`) so all sides sit on the same optimum (point gap ≤2e-4). SE gap ≤1.9% from differing information-matrix conventions at the optimum across `sp.melogit`, `lme4::glmer`, and Stata `melogit`. Margin 2.6×. Value frozen by the contract test. Mechanism not fully derived — see weak spots. |
| `27_glmm_aghq` | `rel_se` | 5e-2 | B | 0.0187 / 0.0187 | Same as module 26 with AGHQ (nAGQ=8) and the reference optimizer budget (`tol=1e-12`, point gap ≤2.4e-7). Margin 2.7×. Frozen by the contract test. |
| `30_oaxaca` | `rel_se` | 0.05 | A | 0.0125 / 0.0122 | Blinder–Oaxaca (`blinder1973wage`, `oaxaca1973male`; cf. `jann2008blinder`). StatsPAI reports closed-form delta-method SEs (`src/statspai/decomposition/oaxaca.py`); `oaxaca::oaxaca` reports seeded bootstrap SEs with `R=100` replications, whose own Monte Carlo noise is ~`(2R)^{-1/2}` ≈ 7% of the SE. Tightened 2026-06-10 from 1.0 (4× margin); a future regeneration that changes the bootstrap RNG stream may legitimately require re-registration. |
| `36_mediation` | `rel_se` | 0.10 | A | 0.0701 / 0.0321 | Causal mediation (`imai2010general`). StatsPAI uses bootstrap inference (B=1000); `mediation::mediate` uses quasi-Bayesian Monte Carlo with `sims=200` (~5% MC noise by itself); the Stata bridge uses delta-method SEs. Different inference algorithms by construction; point effects match at 1e-15. Margin 1.4×. Frozen by the contract test. |
| `40_qreg` | `rel_se` | 0.10 | A | 0.0734 / 0.0302 | Median regression (`koenker2005quantile`). StatsPAI uses the Powell-type iid kernel sandwich (`src/statspai/regression/quantile.py`, kernel estimate of the residual density at zero); the R fixture deliberately reports `summary(rq, se="nid")` — the Hendricks–Koenker difference-quotient sandwich — chosen to match Stata `qreg`'s default. Different sparsity estimators by construction. Margin 1.4×. |
| `47_ppmlhdfe_3fe` | `rel_se` | 5e-2 | B | 0.0182 / 0.0010 | PPML + 3-way HDFE, HC1 sandwich after the Gauss–Seidel multi-FE fix (point estimates at 1e-15). StatsPAI agrees with Stata `ppmlhdfe` to 0.10%; the residual 1.8% gap vs `fixest::fepois` HC1 is unpinned (weak spot). Margin 2.7×. |

All remaining entries are at `1e-2` or tighter; the larger ones are
either graded inline in `compare.py` (`04_csdid` analytic
influence-function SE, observed 0.32%, 3.1× margin,
`callaway2021difference`; `17_etwfe` observed 6.0e-4 on the Stata side,
1.7× margin; `61_betareg` expected- vs observed-information SEs) or are
flagged below as weak spots (`29_panel_sfa`, `33_var`).

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

Output: `observed SE ratio = 1.012871 …` — exactly the 1.287% gap
recorded for every `33_var` SE row in `parity_table.md` (the committed
fixture has `T=198`, `k=5`, `sqrt(198/193) − 1 = 1.287%`).

## Known weak spots

The honest list. Everything here either carries grade C, exceeds its
registered budget on rows the headline check does not gate, or rests on
a mechanism we have not pinned.

1. **`29_panel_sfa` (`rel_se` 1e-3) — grade C.** The budget is
   exceeded by every non-headline SE row: slope SEs differ by up to
   0.98% vs `frontier::sfa`, the intercept SE by 1.8% (R) and 19.5%
   (Stata, a documented `xtfrontier`-scale diagnostic;
   `pitt1981measurement`), and the `sigma_u` point row differs by 28.6%
   vs Stata. The headline (slope `rel_est`) passes, but the registered
   `rel_se` number bounds nothing as written. Needs a re-registered
   per-row budget or an explicit point-only restructure.
2. **`33_var` (`rel_se` 1e-3) — mechanism A, budget mis-keyed.** The
   `T` vs `T−k` divisor gap is fully explained (see above) and the
   Stata side matches at machine level, but every R-side SE row sits at
   1.287% — above the registered budget. The budget is implicitly keyed
   to the Stata convention only; it should be re-registered per side.
3. **`05_sunab` — fixest-side mechanism unpinned.** The 17.1% gap vs
   `fixest::sunab` per-event-time SEs is attributed to the clustered
   delta-method aggregation but has not been reproduced term-by-term.
   Margin is only 1.5×.
4. **`26_glmm_logit` / `27_glmm_aghq` — SE convention not derived.**
   The ≤1.9% SE gap at a tight common optimum is labelled an
   information-matrix convention difference across `sp.melogit`,
   `lme4::glmer`, and Stata `melogit`, but the precise difference has
   not been written down.
5. **`47_ppmlhdfe_3fe` — R-side residual unpinned.** StatsPAI matches
   Stata `ppmlhdfe` HC1 to 0.10% but `fixest::fepois` HC1 differs by
   1.8%; the score/df detail responsible has not been identified.
6. **`13_causal_forest` `rel_se` 0.50.** Now the loosest live SE
   budget (3.4× margin). Not tightenable under the 5× rule today;
   revisit after the next fixture regeneration.
7. **Headline-only enforcement.** The contract test gates each
   module's *headline* rows and metric. Non-headline rows are reported
   in the Markdown tables but not gated — e.g. `07_scm` donor-weight
   rows reach rel 1.78 (above even its T4 budget; that is the
   documented reference disagreement itself) and `52_scm_unique`'s
   distractor-weight rows show `rel_est = 1` purely because
   `Synth::synth` leaves ~1e-9 residual weights against StatsPAI's
   exact zeros (a near-zero-denominator artifact of the relative-diff
   definition, documented in the module's `certification_note`).
8. **Contract-frozen values.** `tests/test_parity_harness_contract.py`
   asserts exact equality for several budgets (e.g. `26_glmm_logit`,
   `27_glmm_aghq`, `36_mediation`, `38_drdid`, `10_honest_did`,
   `11_psm` `rel_est`). Any future change must touch both files in the
   same commit, deliberately.

---

*Last audited: 2026-06-10 (StatsPAI 1.16.1, 64 parity modules). Re-run
the snippet above and refresh this document whenever a `TOLERANCES`
entry changes.*
