# Parity coverage gap inventory (dev / planning)

> Snapshot as of **2026-06-30**, StatsPAI 1.20.0. The *live* numbers are
> always available via `sp.parity_summary()` and the auto-generated
> [parity matrix](../parity.md); this page adds the human judgment
> (prioritization + candidate references) that the machine index deliberately
> does not assert. Candidate reference ecosystems below are **leads to verify
> before alignment**, not parity claims (CLAUDE.md §10).

## Honest denominators

The headline "111 / 1135 verified ≈ 9.8%" understates coverage because the
1135 denominator includes ~171 infrastructure functions (`output`, `plots`,
`utils`, `agent`, `workflow`, `smart`, `core`, `datasets`, `validation`,
`experimental`) for which cross-language parity is **not applicable** — they
render tables, build agent schemas, or load data; they are not estimators.

| denominator | verified | total | fraction |
| --- | ---: | ---: | ---: |
| **estimator functions** (parity-applicable) | 146 | 964 | **15.1%** |
| infra / non-estimator (parity N/A) | — | 171 | — |
| all registered | 143 | 1139 | 12.6% |

> Recent coverage gains (vs R): +`kaplan_meier`, +`logrank_test`
> (`survival::survfit`/`survdiff`, bit-exact); +`bonferroni`, +`holm`,
> +`benjamini_hochberg`, +`adjust_pvalues` (base R `stats::p.adjust`, bit-exact);
> +`het_test`, +`reset_test` (`lmtest::bptest`/`resettest`, bit-exact);
> +`survreg`, +`aft` (`survival::survreg` Weibull AFT, aligned ~1e-5).
> Probed but excluded (convention mismatch, kept honest): `johansen`
> (lag/sample convention vs `urca::ca.jo`), `granger_causality` (VAR-based vs
> pairwise `lmtest::grangertest`), `vif` (rounded output). See the closing loop.

So the real coverage metric to drive to is **verified / 964 estimators**, and
the north-star is to raise it release over release.

## Coverage by estimator family (the gap map)

`EMPTY` = a whole family with zero parity rows — the highest-leverage targets,
because one new Track A module + a `reference_parity` file can move many
functions at once.

| family | verified / total | note |
| --- | ---: | --- |
| causal | 58 / 407 | mega-bucket: DiD / IV / RD / synth / matching / DML / mediation / sensitivity — many sub-families already bit-exact; biggest *absolute* gap but heterogeneous |
| regression | 24 / 40 | GLM / count / quantile / limited-dependent + fracreg/hurdle/cloglog vs R |
| panel | 7 / 36 | FE/RE/HDFE/GMM core covered; dynamic & spatial panels open |
| mendelian | 6 / 37 | MR core has analytical recovery; cross-package MR open |
| decomposition | 5 / 31 | Oaxaca/DFL/RIF + inequality_index (Gini/Theil/Atkinson) bit-exact; Gelbach/Das-Gupta open |
| spatial | 0 / 35 | **EMPTY** |
| network | 0 / 33 | **EMPTY** |
| inference | 7 / 26 | cluster/HAC/multiway + MHT (Bonferroni/Holm/BH vs base R) covered; bootstrap open |
| diagnostics | 5 / 25 | Breusch-Pagan + RESET bit-exact (vs lmtest); rest analytical-feasible |
| dag | 0 / 23 | **EMPTY** |
| epi | 9 / 20 | OR/RR/RD/MH/PR/NNT/IRR + cohen_kappa/attributable_risk bit-exact (base-R closed form); standardization open |
| timeseries | 3 / 20 | VAR/LP/ARIMA covered; cointegration/GARCH open |
| bayes | 0 / 19 | **EMPTY** (convergence-diagnostic, not numeric-parity, ceiling) |
| conformal_causal | 0 / 17 | **EMPTY** (frontier) |
| neural_causal | 0 / 16 | **EMPTY** (frontier) |
| interference | 0 / 16 | **EMPTY** |
| structural | 0 / 12 | **EMPTY** |
| postestimation | 3 / 12 | lincom/test(Wald)/margins analytical-only (closed-form identities); contrast/pwcompare open |
| power | 5 / 12 | power_rct/two_proportions/logrank/cluster_rct + mde bit-exact (base-R z-approx); DiD/RD/IV open |
| survival | 5 / 12 | Cox/KM/log-rank bit-exact + Weibull AFT (survreg/aft) aligned, all vs R `survival`; competing-risks open |
| frontier | 2 / 12 | SFA core covered; panel SFA variants open |
| robustness | 0 / 11 | sensitivity bounds — analytical-feasible |
| target_trial / transport / survey / longitudinal / bartik | 0 each | alignable against established packages |
| neural / llm / rl / text / fairness / ope / surrogate / bridge | 0 each | frontier; analytical/simulation ceiling |

## Prioritization — where to spend alignment effort

**Tier 1 — high leverage, clear cross-language sibling, large family.**
One module here verifies many functions and closes an `EMPTY` row.
- **spatial** (35) — candidate refs to verify: R `spdep` / `splm`, Stata
  `spreg` / `spxtregress`.
- **panel** (29 gap) — extend the existing Track A panel module: dynamic
  (`xtdpdgmm`, `plm::pgmm` beyond `xtabond`), spatial panels.
- **epi** (20) — candidate refs: R `epiR` / `survival` / `metafor`, Stata
  `epitab` / `st` suite (several already have `external_parity` via NHEFS).
- **survival** (11 gap) — R `survival` (KM/AFT), `cmprsk` (Fine-Gray), Stata
  `stcox` / `streg` / `stcrreg`.
- **timeseries** (17 gap) — R `vars` / `urca` / `rugarch`, Stata `var` /
  `vec` / `arch`.

**Tier 2 — alignable, partial families to finish.**
- **decomposition** (27 gap) — extend the `_common.py`-backed family
  (Gelbach, Das-Gupta, inequality) against R `oaxaca` / `dineq` / `ddecompose`.
- **inference** (23 gap) — bootstrap / wild-cluster / MHT vs R `fwildclusterboot`,
  `sandwich`, `multcomp` (CR2/CR3/multiway already bit-exact).
- **mendelian** (31 gap) — R `MendelianRandomization` / `TwoSampleMR`
  (MR core already has analytical recovery).
- **frontier / structural / transport / survey / bartik** — established R/Stata
  siblings exist for most; verify per-method.

**Tier 3 — frontier methods, analytical/simulation is the honest ceiling.**
`neural_causal`, `conformal_causal`, `causal_llm`, `causal_rl`, `causal_text`,
`fairness`, `ope`, `surrogate`, `bridge`, most of `bayes` (where the right
evidence is convergence diagnostics + Monte-Carlo coverage, not bit-for-bit
parity). For these the target is a documented **`analytical-only`** record
(DGP recovery / closed-form / MC calibration), not a cross-package grade —
and that is the honest top grade, stated as such.

## The closing loop

1. Pick a Tier-1 family; add a Track A module (`tests/r_parity/NN_*.{py,R}` +
   Stata `.do`) **or** a `reference_parity` frozen fixture.
2. Regenerate: `python scripts/build_parity_index.py` — the new function(s)
   flip from `unverified` to a graded record automatically; the matrix,
   summary, and `docs/parity.md` update; the drift gate stays green.
3. `sp.parity_summary()`'s estimator-verified fraction is the metric of record;
   report it per release in `CHANGELOG.md`.
