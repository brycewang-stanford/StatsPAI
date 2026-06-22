# StatsPAI SkillOpt Gate Cases

These held-out gate cases are for improving `SKILL.md` itself. They are not
package tests and they do not require editing StatsPAI source code. Use them the
way SkillOpt uses validation splits: a candidate skill edit is accepted only when
it fixes a concrete failure case without regressing these stable tasks.

## How to Use

1. Pick the cases touched by the proposed edit. If the edit changes routing,
   export behavior, or result handling, include at least `G1`, `G2`, and `G10`.
2. Answer the prompt using only the candidate skill text and the public StatsPAI
   API. A pass requires the listed "must do" behavior and no "must not do" error.
3. Accept a skill edit only if every selected case passes. If a case fails,
   make the smallest add/delete/replace edit near the earliest relevant section.

## Gate Cases

| ID | Held-out prompt | Must do | Must not do |
|---|---|---|---|
| G1 | "I already fit three StatsPAI models. Export an AER-style Table 2 to Word, Excel, and LaTeX." | Stay export-only; use `sp.regtable(*models, template="aer")`; call `.to_word()`, `.to_excel()`, and `.to_latex()` or write the `.tex` string | Run the full causal pipeline; use `output="docx"` / `output="xlsx"`; hand-roll Word/Excel from pandas |
| G2 | "Run a staggered DID event-study for a panel with unit id, year, and first treatment year." | Freeze `sp.causal_question(..., design="did").identify()`; use `sp.callaway_santanna(..., g=, t=, i=, x=)` for the dynamic ATT figure; run numerical pre-trends separately | Pass `sp.event_study()` output into `sp.enhanced_event_study_plot`; use `covariates=` for CS/SA |
| G3 | "Estimate `wage ~ training + age \| firm + year` with clustered SE." | Route any formula containing `\|` to `sp.feols(..., vcov={"CRV1": "cluster"})`; for two-way cluster use `"firm+year"` | Send the formula to `sp.regress`; pass `cluster=` to `sp.feols`; call `sp.twoway_cluster()` on a pyfixest result |
| G4 | "Estimate an IV model with industry and year fixed effects." | Report first-stage F before the 2SLS coefficient; state that `sp.ivreg` does not absorb `\|` FE; use low-dimensional controls or explicit dummy construction | Pass `y ~ (d ~ z) + x \| industry + year` directly to `sp.ivreg` and imply FE were absorbed |
| G5 | "Run an RD design around cutoff 0 and make the identification figure." | Use RD kwargs with `c=0`; produce RD plot plus McCrary/manipulation check; include bandwidth/kernel sensitivity | Use `cutoff=` for `sp.rdrobust`; report the coefficient without manipulation and bandwidth checks |
| G6 | "Do a public-health target-trial emulation with an exposure and binary outcome." | Write protocol before modeling; check positivity/overlap; compare IPTW/g-computation/TMLE when supported; report E-value or equivalent sensitivity | Treat it as a plain AER OLS table; skip protocol and overlap |
| G7 | "Estimate CATEs and learn a policy from them." | Make train/holdout split explicit; use `model_info["cate"]` for meta-learner per-row CATE or `cf.effect(X)` for causal forest; do OPE on holdout | Use nonexistent `.cate_estimates` or `.local_effects()`; make policy-value claims on training data only |
| G8 | "Migrate this Stata/R workflow into StatsPAI." | Use `sp.list_functions()`, `sp.describe_function()`, `sp.function_schema()`, or translator surfaces first; preserve notes for unsupported options | Invent unverified API names; claim full parity when options are unsupported |
| G9 | "The run failed because `pyfixest` or plotting is missing." | Name the exact extra, e.g. `pip install "statspai[fixest]"` or `[plotting]`; resume from the last verified artifact | Rewrite the estimator choice without explaining the missing optional dependency |
| G10 | "Summarize what you produced." | List actual artifact paths, estimator class, identifying assumption, diagnostics, and skipped/failed gates | Claim "paper-ready" when requested exports or mandatory diagnostics were not generated |

## Scoring

Score each case as pass/fail. A candidate edit should not be accepted if it
turns a previous pass into a fail, even if it improves a different section. If a
new recurring failure appears, add one new held-out case here and one bounded
rule in `SKILL.md`; do not expand the whole playbook unless multiple cases show
the same structural gap.
