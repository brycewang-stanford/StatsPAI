# Parity coverage gap inventory (dev / planning)

> Snapshot as of **2026-07-01**, StatsPAI 1.20.0. The *live* numbers are
> always available via `sp.parity_summary()` and the auto-generated
> [parity matrix](../parity.md); this page adds the human judgment
> (prioritization + candidate references) that the machine index deliberately
> does not assert. Candidate reference ecosystems below are **leads to verify
> before alignment**, not parity claims (CLAUDE.md §10).

## Honest denominators

The all-registered fraction understates coverage because the ~171
infrastructure functions (`output`, `plots`, `utils`, `agent`, `workflow`,
`smart`, `core`, `datasets`, `validation`, `experimental`) are **not
parity-applicable** — they render tables, build agent schemas, or load data;
they are not estimators.

| denominator | verified | total | fraction |
| --- | ---: | ---: | ---: |
| **estimator functions** (parity-applicable) | 213 | 964 | **22.1%** |
| infra / non-estimator (parity N/A) | — | 171 | — |
| all registered | 213 | 1139 | 18.7% |

By grade (live `sp.parity_summary()`): **121 bit-exact**, 7 aligned,
4 external-replication, 81 analytical-only, 926 unverified.

> Recent coverage gains: decomposition closed-form identities
> (`gelbach`/`das_gupta`/`subgroup_decompose`/`kitagawa_decompose`/
> `source_decompose`/`mediation_decompose`, bit-exact); robustness
> (`sensemakr`/`oster_delta`/`breakdown_frontier`, bit-exact closed forms);
> `lrtest`/`icc`/`mr`(ivw)/`margins_at`/`lee_bounds` bit-exact; frontier
> DGP-recovery (`conformal_ite`/`conformal_fair_ite`/`front_door`/`pate`/
> `fci`/`policy_value`/`policy_tree`, analytical-only). **Correctness fixes
> shipped this pass** (labeled ⚠️ in CHANGELOG/MIGRATION): `contrast`/
> `pwcompare` `C(var)` all-zeros, `dist_iv`/`kan_dlate` binary-instrument NaN,
> `ges` spurious collider edge. Probed but excluded (convention mismatch, kept
> honest): `johansen`, `granger_causality`, `vif`, `trimming` (Crump rule),
> `partial_corr_pvalue` (p-value convention), `msm` (estimand ambiguity).

So the real coverage metric to drive to is **verified / 964 estimators**, and
the north-star is to raise it release over release.

## Coverage by estimator family (the gap map)

`EMPTY` = a whole family with zero parity rows — the highest-leverage targets,
because one new Track A module + a `reference_parity` file can move many
functions at once.

| family | verified / total | note |
| --- | ---: | --- |
| causal | 85 / 407 | mega-bucket: DiD / IV / RD / synth / matching / DML / mediation / sensitivity / bounds / policy / conformal — many sub-families now bit-exact or analytical-only; biggest *absolute* gap but heterogeneous |
| regression | 25 / 40 | GLM / count / quantile / limited-dependent + fracreg/hurdle/cloglog vs R |
| panel | 13 / 36 | FE/RE/HDFE/GMM core + absorbed-FE GLM (`feglm`/`fepois`) + within transformation (`demean`) + balance filter (`balance_panel`) covered; dynamic system-GMM and spatial panels open |
| mendelian | 6 / 37 | MR core has analytical recovery + `mr`(ivw) bit-exact (IVW closed form); cross-package MR-Egger/median open |
| decomposition | 11 / 31 | Oaxaca/DFL/RIF + inequality_index + Gelbach + Das-Gupta + subgroup(Theil) + Kitagawa + Lerman-Yitzhaki source + mediation all bit-exact closed-form; Machado-Mata / RIF-regression open |
| spatial | 5 / 35 | SAR/SEM/SDM ML + SAR-2SLS/SEM-GMM bit-exact vs `spatialreg`; spatial-panel / GWR / SARAR-GMM open |
| network | 0 / 33 | **EMPTY** |
| inference | 14 / 26 | cluster/HAC/multiway + MHT + `fisher_exact` (randomization) + `lrtest`/`icc` bit-exact; wild-cluster bootstrap variants open |
| diagnostics | 6 / 25 | Breusch-Pagan + RESET + `structural_break`/`cusum_test`/`engle_granger` (known-truth recovery); rest analytical-feasible |
| dag | 1 / 23 | `pc_algorithm`/`notears`/`lingam`/`fci`/`ges` recover known CPDAG/PAG structures (analytical); remaining discovery variants open |
| epi | 14 / 20 | 2x2 measures + kappa/AR + sens/spec + standardization + auc/roc bit-exact (base-R/Mann-Whitney closed form) |
| timeseries | 6 / 20 | VAR/LP/ARIMA + cointegration (`engle_granger`) + break tests covered; GARCH / Johansen open |
| bayes | 0 / 19 | **EMPTY** (convergence-diagnostic, not numeric-parity, ceiling) |
| conformal_causal | 2 / 17 | `conformal_ite`/`conformal_fair_ite` coverage-guarantee MC (analytical); `conformal_synth` noisy, open |
| neural_causal | 0 / 16 | **EMPTY** (frontier) |
| interference | 1 / 16 | `cluster_cross_interference` DGP recovery + spillover/network_exposure analytical; partial-interference designs open |
| structural | 0 / 12 | front_door/frontdoor (identification recovery) live under causal; simultaneous-eq (`three_sls`/`sureg`) open |
| postestimation | 6 / 12 | lincom/test(Wald)/margins/margins_at + contrast/pwcompare (bit-exact after C(var) fix); pwcompare adjust-methods open |
| power | 7 / 12 | rct/two_proportions/logrank/cluster_rct/case_control + mde bit-exact (base-R z-approx); DiD/RD/IV open |
| survival | 5 / 12 | Cox/KM/log-rank bit-exact + Weibull AFT (survreg/aft) aligned, all vs R `survival`; competing-risks open |
| frontier | 2 / 12 | SFA core covered; panel SFA variants open |
| robustness | 3 / 11 | `sensemakr`/`oster_delta`/`breakdown_frontier` bit-exact closed forms; Rosenbaum/copula bounds open |
| survey | 4 / 7 | svymean/svytotal/svyglm bit-exact (vs R `survey`, HT/Hajek + linearization SE) + svydesign; calibration/rake open |
| selection / censoring | ~2 | `lee_bounds` bit-exact + `selection_bounds` analytical (Lee trimming); heckman/tobit vs R open |
| policy_learning | 2 | `policy_value` bit-exact (Athey-Wager value) + `policy_tree` oracle recovery |
| qte | 4 / 15 | `ivqreg`/`beyond_average_late`/`continuous_iv_late`/`dist_iv` LATE recovery (analytical) |
| target_trial / transport / longitudinal / bartik | pate + transport partial | `pate` transportability recovery; rest alignable against established packages |
| neural / llm / rl / text / fairness / ope / surrogate / bridge | 0 each | frontier; analytical/simulation ceiling |

## Prioritization — where to spend alignment effort

**Tier 1 — high leverage, clear cross-language sibling, large family.**
One module here verifies many functions and closes an `EMPTY` row.
- **spatial** (30 gap) — SAR/SEM/SDM ML and SAR-2SLS/SEM-GMM now bit-exact vs
  `spatialreg` (modules 65--66). Remaining leads to verify: spatial panels
  (R `splm`, Stata `spxtregress`), GWR (`GWmodel`), and the SARAR GMM /
  heteroskedastic-GM estimators (reconcile the joint moment sequence against
  `spatialreg::gstsls` / `sphet`).
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
