# Recommendation Hit-Rate Benchmark — Findings Log

Concrete correctness gaps the benchmark surfaces, most-actionable first. Each
finding is a candidate fix for the recommend/detect_design/audit engines. Keep
this in sync with the scorecard; close findings by linking the fixing commit.

---

## F-001 · recommend is blind to the synthetic-control design  ⚠ HIGH

**Symptom.** On all three single-treated-unit comparative case studies in the
seed corpus — California Prop 99 (Abadie-Diamond-Hainmueller 2010), Basque
Country (Abadie-Gardeazabal 2003), West Germany reunification (ADH 2015) —
`detect_design` classifies the panel as `did` and `recommend` leads with
**Callaway-Sant'Anna**. CS is degenerate with a single treated unit; the
correct tool is synthetic control.

**Evidence.**
- `grep -niE "synth|synthetic|scm" src/statspai/smart/recommend.py` → no matches.
- `detect_design.py` has no single-treated-unit / synth branch.
- Scorecard: `abadie_2010_prop99`, `abadie_2003_basque`, `abadie_2015_german`
  → `MISS`, detected `did`, top-1 `callaway_santanna`.

**Why it matters.** StatsPAI ships 20+ synthetic-control estimators behind
`sp.synth(method=...)`; the recommender cannot reach a single one. An agent
that hands a comparative case study to `sp.recommend` gets a plausible-but-wrong
staggered-DiD estimator with full authority — the exact moat-reversal risk.

**Proposed fix.** Add a synth branch:
1. `detect_design` flags `synth` when a balanced panel has exactly one (or very
   few) ever-treated units and a long pre-period (T_pre ≥ ~ a few).
2. `recommend` emits a ranked synth block: `sp.synth` (classic) →
   `sp.augsynth` / `sp.synthdid` → `sp.gsynth` / `sp.mc_panel`, with
   pre-treatment-fit and placebo-inference as the attached robustness steps.
3. When timing is staggered AND multiple treated units exist, keep CS/SA top —
   do not regress that behavior.

**Status.** ✅ **FIXED** (Phase 2). `_detect_design` now routes a panel with
≤2 ever-treated units, a donor pool ≥5, and ≥4 periods to `synth`; `recommend`
emits a ranked synth block (classic SCM → Augmented SCM → Synthetic DiD) with
pre-fit + placebo robustness. Verified: Prop99/Basque/German flip MISS→HIT
(top-1 = `synth`, detected = `synth`, runnable: Prop99 ATT ≈ −19.76 ≈ ADH 2010
headline) with **zero regression** on mpdta (still `did` → Callaway-Sant'Anna).
Locked by `tests/test_recommend_synth_detection.py` (5 tests). Hit-rate top-1
0.625 → 1.0.

---

## F-002 · audit misses overlap/balance on selection-on-observables  ◆ MEDIUM

**Symptom.** On the Dehejia-Wahba (1999) NSW selection-on-observables design,
`recommend` leads (top-1) with **OLS with robust SE (baseline)**. When the
top-1 is fitted and audited, `sp.audit` classifies it as method family
`regression` and applies regression checks (robust_se, Oster) — it **never asks
for overlap / common-support, post-matching balance, or OVB sensitivity**, the
checks a referee on an observational design demands. Dynamic audit recall = 0.0
for this entry (vs. 1.0 for every other design).

**Evidence.** Scorecard "audit recall (dynamic)" row `dehejia_wahba_1999_nsw`:
fitted family `regression`, recall `0.0`. The static catalog *does* contain
these checks under the `matching` family (static recall 1.0) — the gap is that
the recommended-and-fitted top-1 (OLS) never triggers them.

**Why it matters.** DW (1999) is the canonical demonstration that *naive OLS is
biased* and propensity-score methods recover the experimental benchmark.
Leading with OLS and getting only robust-SE + Oster as the realized audit is
the textbook plausible-but-wrong loop on an observational design.

**Proposed fix (needs a design decision).**
- Option A — on an `observational` design, demote bare OLS from top-1 to a
  labelled baseline reference and lead with a balancing/PS estimator
  (PSM / DML / entropy balancing), so the realized audit asks for
  overlap + balance.
- Option B — make `sp.audit` ask for overlap / common-support / OVB-sensitivity
  on *any* observational-causal regression, independent of the estimator.
- (A) changes recommend's headline ordering for many users; (B) is more local.
  Leaning A+B, but this is a judgment call — flagged for the user.

**Status.** ✅ **FIXED** (Phase 2, A+B per user decision).
- **A** — `recommend`'s `observational` branch now leads with PSM, then DML, and
  demotes OLS to a labelled "naive baseline — biased under confounding" card.
  nsw_dw top-1 is now `psm` (still a HIT); the fitted-and-audited result is the
  `matching` family, so the realized audit asks overlap + balance + OVB.
- **B** — `sp.audit(result, treatment=...)` is now treatment-aware: a regression
  the caller declares to be a causal-adjustment regression also gets
  overlap / balance_after / ovb_sensitivity (reused from the causal catalog,
  single source of truth). Gated on the explicit treatment, so descriptive OLS
  is never flagged.
- Verified: dynamic audit mean recall 0.875 → **1.0**; nsw_dw dynamic recall
  0.0 → 1.0. Locked by `tests/test_audit_observational_treatment.py` (3 tests).

---

## F-003 · recommend's IV/RD paths under-specify two design refinements  ○ LOW

**Symptom.** Two minor expressiveness gaps surfaced while building the Tier-B
adversarial corpus (neither is a hit-rate miss; both are refinements):
1. **Multiple instruments / over-identification.** `recommend(instrument=...)`
   takes a single string, so an over-identified design (e.g. AK-1991's four
   quarter-of-birth dummies, or `dgp_iv(n_instruments=3)` →
   `instrument_1..3`) cannot be expressed end-to-end, and the engine never
   prompts the Hansen-J over-identification test on its own.
2. **Fuzzy vs sharp RD.** `recommend` returns `rdrobust` for both fuzzy and
   sharp RD without distinguishing them; a fuzzy design is IV-at-the-cutoff and
   the referee's extra ask (first-stage compliance jump / fuzzy estimand) is
   not surfaced.

**Evidence.** Scorecard rows `trap_weak_instrument` (single instrument only),
`archetype_strong_instrument`, `trap_fuzzy_rd` (detected `rd`, top-1
`local_polynomial_rd` — correct family, no fuzzy-specific guidance).

**Why it matters.** Low severity — the recommended estimator family is correct
in every case (no hit-rate impact). These are completeness gaps: the audit
should prompt over-identification for multi-instrument IV and a first-stage /
fuzzy-estimand check for fuzzy RD.

**Proposed fix.** Accept `instrument` as `str | list[str]`; when >1, add the
over-id test to the IV recommendation and audit. Add a `fuzzy` signal to the RD
path (or detect a non-deterministic treatment jump) and attach the fuzzy-RD
first-stage check.

**Status.** OPEN (low priority; tracked for a later phase).

---

## (template for future findings)
## F-00X · <one-line symptom>  <severity>
**Symptom.** …
**Evidence.** scorecard rows / grep / repro …
**Why it matters.** …
**Proposed fix.** …
**Status.** OPEN | FIXED (commit) | WONTFIX (rationale)
