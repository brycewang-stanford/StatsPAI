# Cross-language parity matrix

> **Auto-generated — do not hand-edit.** Regenerate with `python scripts/build_parity_index.py`. Every row traces to a committed test artifact; nothing here is asserted from memory.

StatsPAI's promise is that every number it reports is either *aligned with an external reference implementation*, *recovered against a known truth*, or *honestly marked as neither*. This page makes that promise auditable function-by-function, and keeps the three cases apart — a method with no Stata or R sibling can reach the second and never the first, and saying so is the point. Query any function programmatically:

```python
import statspai as sp
sp.parity_status("feols")     # one function
sp.parity_matrix()            # the whole matrix
sp.parity_summary()           # honest coverage counts
```

## Taxonomy

| grade | meaning |
| --- | --- |
| `bit-exact` | matches a named R/Stata reference to machine tolerance (headline relative error ≤ 1e-6) |
| `aligned` | matches a named reference within a documented, pre-registered looser tolerance (cross-fit / convention disagreement) |
| `analytical-only` | recovers a known population parameter on a deterministic DGP, or a closed-form identity (no cross-package reference) |
| `external-replication` | reproduces published-paper numbers on a calibrated replica |
| `unverified` | registered public API, no qualifying numerical-parity evidence attached yet — **the honest gap** |

## Coverage at a glance

Read the two evidence kinds separately. Only the first answers "does StatsPAI agree with Stata/R"; the second answers "does StatsPAI recover the right answer", which is a different — and for methods with no Stata/R sibling, the only available — question. Summing them into one "verified" figure would let the smaller claim borrow the authority of the larger one, so this page does not print that total.

| evidence kind | grade | functions |
| --- | --- | ---: |
| **Compared against R/Stata** (T2) | bit-exact | 194 |
| | aligned | 21 |
| | **subtotal** | **215** |
| **No external software reference** | analytical-only (T1) | 210 |
| | external-replication (published numbers) | 4 |
| | **subtotal** | **214** |
| No numerical evidence yet | unverified | 753 |

### Honest denominators

The all-registered denominator understates coverage: it counts result and exception classes, which can never carry a parity grade, and infrastructure functions that render tables, draw plots, build agent schemas or load data. The estimator denominator is the number to drive release over release.

| denominator | cross-language | any evidence | total | cross-lang share |
| --- | ---: | ---: | ---: | ---: |
| estimator callables | 215 | 428 | 773 | 27.8% |
| infrastructure (parity N/A) | 0 | 0 | 124 | 0.0% |
| result / exception classes | 0 | 1 | 285 | 0.0% |
| **all registered** | 215 | 429 | 1182 | 18.2% |

### Coverage by estimator family

Families with zero cross-language rows are the highest-leverage targets when a reference implementation exists, and the honest ceiling when one does not — a method with no Stata/R sibling can reach `analytical-only` and no further. This table is generated from the same records as the rest of the page, so it cannot drift from them.

| family | cross-language | any evidence | estimator callables |
| --- | ---: | ---: | ---: |
| causal | 64 | 149 | 331 |
| regression | 30 | 35 | 37 |
| spatial | 18 | 22 | 34 |
| panel | 13 | 20 | 30 |
| decomposition | 11 | 17 | 29 |
| network | 23 | 24 | 25 |
| inference | 8 | 20 | 23 |
| mendelian | 14 | 17 | 23 |
| diagnostics | 7 | 10 | 22 |
| epi | 9 | 16 | 17 |
| dag | 0 | 0 | 15 |
| bayes | 0 | 7 | 14 |
| postestimation | 0 | 6 | 12 |
| timeseries | 3 | 12 | 12 |
| neural_causal | 0 | 0 | 11 |
| power | 5 | 7 | 11 |
| conformal_causal | 0 | 3 | 10 |
| structural | 0 | 7 | 10 |
| frontier | 2 | 5 | 9 |
| survival | 5 | 8 | 8 |
| robustness | 0 | 0 | 7 |
| interference | 0 | 1 | 7 |
| survey | 3 | 4 | 6 |
| target_trial | 0 | 6 | 6 |
| transport | 0 | 1 | 6 |
| fairness | 0 | 6 | 6 |
| other | 0 | 1 | 6 |
| longitudinal | 0 | 5 | 5 |
| experimental | 0 | 1 | 5 |
| causal_llm | 0 | 0 | 4 |
| bartik | 0 | 1 | 4 |
| nonparametric | 0 | 3 | 4 |
| causal_discovery | 0 | 0 | 3 |
| surrogate | 0 | 3 | 3 |
| causal_rl | 0 | 0 | 3 |
| assimilation | 0 | 3 | 3 |
| gformula | 0 | 2 | 2 |
| ope | 0 | 2 | 2 |
| causal_text | 0 | 0 | 2 |
| missing | 0 | 2 | 2 |
| mediation | 0 | 1 | 2 |
| censoring | 0 | 1 | 1 |
| synth | 0 | 0 | 1 |

## bit-exact — 194 functions

Machine-tolerance agreement with a named R/Stata reference.

| function | reference | versions | tolerance | rel err (R / Stata) | test |
| --- | --- | --- | --- | --- | --- |
| `adjust_pvalues` | base R stats::p.adjust (bonferroni/holm/BH) | R 4.5.2 | exact (atol 1e-15; observed 0) | — / — | [`test_mht_parity.py`](../tests/reference_parity/test_mht_parity.py) (+1) |
| `anderson_rubin_ci` | R ivmodel::AR.test confidence set | ivmodel 1.9.1; car 3.1.5; metafor 5.0.1 | Both endpoints at 5e-15 after the boundary bisection replaced the grid-point endpoints (previously 8.1e-3 / 4.9e-3). A reference-free test also asserts the two AR entry points agree with each other and that neither endpoint lands exactly on a grid node. | — / — | [`test_weakiv_meta_parity.py`](../tests/reference_parity/test_weakiv_meta_parity.py) |
| `anderson_rubin_test` | R ivmodel::AR.test | ivmodel 1.9.1; car 3.1.5; metafor 5.0.1 | Statistic 1.1e-15, p-value 1.2e-13, degrees of freedom exact, and the analytic AR confidence set 1.2e-14. | — / — | [`test_weakiv_meta_parity.py`](../tests/reference_parity/test_weakiv_meta_parity.py) |
| `arima` | stats::arima | R 4.5.2; stats 4.5.2 | rel_est<=1e-06, rel_se<=1e-06 | 7.4e-07 / 9.3e-09 | [`39_arima.py`](../tests/r_parity/39_arima.py) (+2) |
| `assortativity` | R igraph::assortativity_degree | igraph 2.3.3; sna 2.8; ergm 4.12.0; dyadRobust 0.0.1.0001 | 1.2e-16 on karate. | — / — | [`test_network_parity.py`](../tests/reference_parity/test_network_parity.py) |
| `attributable_risk` | base-R closed form (attributable fraction exposed + PAF) | R 4.5.2 | AFE + PAF point estimates 1e-12 abs (observed 0); CI not pinned | — / — | [`test_epi_extra_parity.py`](../tests/reference_parity/test_epi_extra_parity.py) (+1) |
| `bacon_decomposition` | bacondecomp::bacon | R 4.5.2; bacondecomp 0.1.1 | rel_est<=1e-06, rel_se<=1e-06 | 5.6e-16 / 9.6e-09 | [`20_bacon.py`](../tests/r_parity/20_bacon.py) (+2) |
| `balance_panel` | base R counts == n_periods | R 4.5.2 | rel_est<=1e-06, rel_se<=1e-06 | 0 / 0 | [`69_balance_panel.py`](../tests/r_parity/69_balance_panel.py) (+2) |
| `benjamini_hochberg` | base R stats::p.adjust(method='BH') | R 4.5.2 | exact (atol 1e-15; observed 0) | — / — | [`test_mht_parity.py`](../tests/reference_parity/test_mht_parity.py) (+1) |
| `betareg` | betareg::betareg(link.phi="log") | R 4.5.2; betareg 3.2.4 | rel_est<=1e-06, rel_se<=0.01 | 2.2e-08 / 3.1e-08 | [`61_betareg.py`](../tests/r_parity/61_betareg.py) (+2) |
| `betweenness_centrality` | R igraph::betweenness | igraph 2.3.3 | Normalised and raw on karate, raw on the directed graph, all at 1e-10. | — / — | [`test_network_parity.py`](../tests/reference_parity/test_network_parity.py) (+1) |
| `biprobit` | R VGAM::vglm(binom2.rho) bivariate probit | R 4.5.2; VGAM 1.1.14 | coef / rho 1e-6 abs (observed <= 2e-7); logLik 1e-6 rel | — / — | [`test_biprobit_parity.py`](../tests/reference_parity/test_biprobit_parity.py) (+1) |
| `bonacich_power` | R igraph::power_centrality and sna::bonpow | igraph 2.3.3; sna 2.8; ergm 4.12.0; dyadRobust 0.0.1.0001 | 3.9e-16 against igraph and 5.8e-16 against sna at beta = 0.1. | — / — | [`test_network_parity.py`](../tests/reference_parity/test_network_parity.py) |
| `bonferroni` | base R stats::p.adjust(method='bonferroni') | R 4.5.2 | exact (atol 1e-15; observed 0) | — / — | [`test_mht_parity.py`](../tests/reference_parity/test_mht_parity.py) (+1) |
| `callaway_santanna` | did::att_gt + aggte | R 4.5.2; did 2.3.0 | rel_est<=1e-06, rel_se<=1e-09 | 1.3e-15 / 1.3e-15 | [`04_csdid.py`](../tests/r_parity/04_csdid.py) (+2) |
| `cgs_continuous_did` | contdid::cont_did | R 4.5.2 | rel_est<=1e-06 | 2.4e-14 / — | [`80_contdid.py`](../tests/r_parity/80_contdid.py) (+1) |
| `clogit` | survival::clogit | R 4.5.2; survival 3.8.3 | rel_est<=1e-06, rel_se<=1e-06 | 1.3e-08 / 1.3e-08 | [`46_clogit.py`](../tests/r_parity/46_clogit.py) (+2) |
| `closeness_centrality` | Wasserman-Faust closeness from R igraph::distances | igraph 2.3.3; sna 2.8; ergm 4.12.0; dyadRobust 0.0.1.0001 | Exact (0.0) on a disconnected graph with three blocks and three isolates -- the case the correction exists for; the connected-graph values also match igraph::closeness(normalized = TRUE) to 1e-10. | — / — | [`test_network_parity.py`](../tests/reference_parity/test_network_parity.py) |
| `clustering` | R igraph::transitivity(type = 'local', isolates = 'zero') | igraph 2.3.3 | Exact on karate and on a disconnected graph whose isolates and degree-1 nodes score 0 on both sides. | — / — | [`test_network_parity.py`](../tests/reference_parity/test_network_parity.py) (+1) |
| `cohen_kappa` | base-R closed form (Cohen's kappa point estimate) | R 4.5.2 | kappa + agreements 1e-12 abs (observed ~1e-16); SE not pinned | — / — | [`test_epi_extra_parity.py`](../tests/reference_parity/test_epi_extra_parity.py) (+1) |
| `cox` | survival::coxph | R 4.5.2; survival 3.8.3 | rel_est<=1e-06, rel_se<=1e-06 | 8.4e-16 / 2.1e-10 | [`24_coxph.py`](../tests/r_parity/24_coxph.py) (+2) |
| `cr2_se` | clubSandwich::vcovCR(type="CR2"/"CR3") | R 4.5.2; clubSandwich 0.6.2 | rel_est<=1e-06, rel_se<=1e-06 | 1.8e-08 / 2.2e-08 | [`53_cr2.py`](../tests/r_parity/53_cr2.py) (+2) |
| `das_gupta` | R DasGuptR::dgnpop (product rate function, summed over strata) | DasGuptR 2.2.0; ddecompose 1.0.0; cdgd 1.0.1 | Das Gupta's Table 2.1 (two factors) and Table 6.5 (four factors x six age groups), every factor effect and both crude rates at 1e-10. | — / — | [`test_decomp_R_parity.py`](../tests/reference_parity/test_decomp_R_parity.py) |
| `ddd` | Stata 18 MP regress [aw=w], robust (aweight HC1) | Stata 18 MP | b / se 1e-12 abs (observed <= 3e-15) | — / — | [`test_did2x2_ddd_weighted_robust_parity.py`](../tests/reference_parity/test_did2x2_ddd_weighted_robust_parity.py) |
| `ddd_heterogeneous` | triplediff::ddd + agg_ddd | R 4.5.2 | rel_est<=1e-06, rel_se<=1e-06 | 5.7e-15 / — | [`77_ddd.py`](../tests/r_parity/77_ddd.py) (+1) |
| `decompose` | oaxaca::oaxaca | R 4.5.2; oaxaca 0.1.5 | rel_est<=1e-06, rel_se<=0.05 | 6.3e-16 / 1.3e-16 | [`30_oaxaca.py`](../tests/r_parity/30_oaxaca.py) (+2) |
| `degree_centrality` | R igraph::degree | igraph 2.3.3 | Normalised on karate and raw in / out / all modes on a 40-node directed graph, exact. | — / — | [`test_network_parity.py`](../tests/reference_parity/test_network_parity.py) (+1) |
| `demean` | textbook mean-within (algorithmic) | R 4.5.2 | rel_est<=1e-06, rel_se<=1e-06 | 3.5e-15 / 8.8e-15 | [`68_demean_within.py`](../tests/r_parity/68_demean_within.py) (+2) |
| `dfl_decompose` | ddecompose::dfl_decompose | R 4.5.2; ddecompose 1.0.0 | rel_est<=1e-06, rel_se<=1e-06 | 1.2e-09 / 1.8e-13 | [`31_dfl.py`](../tests/r_parity/31_dfl.py) (+3) |
| `did_2x2` | Stata 18 MP regress [aw=w], robust (aweight HC1) | Stata 18 MP | b / se 1e-12 abs (observed <= 3e-16) | — / — | [`test_did2x2_ddd_weighted_robust_parity.py`](../tests/reference_parity/test_did2x2_ddd_weighted_robust_parity.py) |
| `did_imputation` | didimputation::did_imputation | R 4.5.2; didimputation 0.5.1 | rel_est<=1e-06, rel_se<=1e-06 | 4.8e-08 / 3.5e-07 | [`16_bjs.py`](../tests/r_parity/16_bjs.py) (+2) |
| `did_multiplegt` | DIDmultiplegt::did_multiplegt (archived 0.1.4) | R 4.5.2 | rel_est<=1e-06 | 3.9e-15 / 3.2e-15 | [`81_didm.py`](../tests/r_parity/81_didm.py) (+2) |
| `did_multiplegt_dyn` | DIDmultiplegtDYN::did_multiplegt_dyn | R 4.5.2 | rel_est<=1e-06 | 3.3e-15 / 2.1e-15 | [`78_multiplegt_dyn.py`](../tests/r_parity/78_multiplegt_dyn.py) (+2) |
| `distance_band` | R spdep::dnearneigh | spdep 1.4.2; spatialreg 1.4.3 | Neighbour sets identical for all 120 points at a 0.25 radius. | — / — | [`test_spdep_parity.py`](../tests/reference_parity/test_spdep_parity.py) |
| `dml` | DoubleML::DoubleMLPLR | R 4.5.2; DoubleML 1.0.2 | rel_est<=1e-10, rel_se<=1e-10 | 0 / 3.7e-15 | [`08_dml.py`](../tests/r_parity/08_dml.py) (+2) |
| `dml_sensitivity` | doubleml (Python) DoubleML.sensitivity_analysis | — | bias_bound and adjusted theta bounds 1e-12 (observed 2.5e-15); RV 1e-6 (observed 9.2e-8); RVa is a documented convention gap (<5e-3, observed 1.4e-3) because StatsPAI exhausts |theta|-z*se with the unadjusted SE while doubleml lets the SE move with the confounding scenario | — / — | [`test_dml_sensitivity_parity.py`](../tests/external_parity/test_dml_sensitivity_parity.py) |
| `drdid` | DRDID::drdid_imp_panel | R 4.5.2; DRDID 1.2.3 | rel_est<=1e-06, rel_se<=1e-06 | 2.6e-15 / 2.2e-16 | [`38_drdid.py`](../tests/r_parity/38_drdid.py) (+2) |
| `dyadic_regression` | R dyadRobust (Aronow-Samii-Assenova dyadic-robust variance) | igraph 2.3.3; sna 2.8; ergm 4.12.0; dyadRobust 0.0.1.0001 | Coefficients 7e-16 and standard errors 2e-15 on undirected and directed dyads, after the 1.28.0 fix to the shared-member weighting; also asserted against a brute-force construction of the definition. | — / — | [`test_network_parity.py`](../tests/reference_parity/test_network_parity.py) |
| `ebalance` | ebal::ebalance 0.2.1 (Hainmueller 2012) | — | ATT rel <= 1e-5 (observed 3.2e-7); moment gap <= 1e-10 | — / — | [`test_matching_r_parity.py`](../tests/reference_parity/test_matching_r_parity.py) (+1) |
| `eigenvector_centrality` | R sna::evcent (unit L2 norm, as here) | sna 2.8; igraph 2.3.3 | 5e-11 undirected, 8e-11 directed -- power-iteration tolerance on both sides. igraph::eigen_centrality max-scales instead (and 2.x ignores scale = FALSE), so it agrees only up to one scalar; the earlier note claiming the igraph convention was wrong. | — / — | [`test_network_parity.py`](../tests/reference_parity/test_network_parity.py) (+1) |
| `ergm` | R ergm::ergm(estimate = 'MPLE') | igraph 2.3.3; sna 2.8; ergm 4.12.0; dyadRobust 0.0.1.0001 | Coefficients 2e-16 to 7e-13 for edges + triangle + nodematch + nodecov + absdiff (undirected) and edges + mutual (directed); standard errors 2e-8 (directed) and <= 3.2e-7 (undirected), inside the 1e-6 budget. | — / — | [`test_network_parity.py`](../tests/reference_parity/test_network_parity.py) |
| `etregress` | Stata 18 MP official `etregress` (Maddala 1983 model) | Stata 18 MP | Two-step: 5e-9 on every coefficient and every standard error, including the Heckman correction for the estimated first stage. ML: the likelihood, score and observed information are pinned at 9e-11 -- our Hessian reproduces Stata's reported standard errors when evaluated at Stata's own parameter vector, which is independent of either optimiser. At our own optimum the parameters sit within 2e-5 of Stata's; that gap is the two optimisers' stopping points, not a formula difference, and StatsPAI's stops at the HIGHER log-likelihood with a gradient ~300x smaller (asserted, so a regression that makes our optimum worse fails even though the 1e-4 parity assertions would still pass). vce(robust) carries Stata's N/(N-1) meat factor and vce(cluster) its g/(g-1). | — / — | [`test_etregress_stata_parity.py`](../tests/reference_parity/test_etregress_stata_parity.py) |
| `etwfe` | etwfe::etwfe + emfx | R 4.5.2; etwfe 0.6.2 | rel_est<=1e-06, rel_se<=0.001 | 1.8e-13 / 3.9e-14 | [`17_etwfe.py`](../tests/r_parity/17_etwfe.py) (+2) |
| `etwfe_emfx` | etwfe::etwfe + emfx | R 4.5.2; etwfe 0.6.2 | rel_est<=1e-06, rel_se<=0.001 | 1.8e-13 / 3.9e-14 | [`17_etwfe.py`](../tests/r_parity/17_etwfe.py) (+2) |
| `evalue` | EValue::evalues.RR | R 4.2.3; EValue 4.1.4 | rel_est<=1e-06, rel_se<=1e-06 | 5.8e-14 / 1.2e-16 | [`23_evalue.py`](../tests/r_parity/23_evalue.py) (+2) |
| `evalue_rr` | R EValue::evalues.RR | EValue 4.1.4 | Point and CI E-values at 1e-12 across ten cases, including RR < 1 and CIs crossing the null. | — / — | [`test_evalue_rr_parity.py`](../tests/reference_parity/test_evalue_rr_parity.py) |
| `event_study` | fixest::feols(y ~ i(rel, treat, ref=-1) | R 4.5.2; fixest 0.14.0 | rel_est<=1e-09, rel_se<=1e-09 | 3.2e-13 / 1.5e-14 | [`85_twfe_event_study.py`](../tests/r_parity/85_twfe_event_study.py) (+2) |
| `fect` | fect::fect(Y ~ D + X1 + X2, method=, force="two-way", se=FALSE, CV=FALSE, tol=1e-12, max.iteration=20000); Stata side uses the authors' fect_stata (GitHub, installed into a local ado path) | R 4.5.2; fect 2.4.1 | rel_est<=1e-06, rel_se<=1e-06 | 1.8e-13 / 9.8e-10 | [`86_fect.py`](../tests/r_parity/86_fect.py) (+2) |
| `feglm` | fixest::feglm (family="logit") / fixest::fepois | R 4.5.2; fixest 0.14.0 | rel_est<=1e-06, rel_se<=5e-05 | 9.7e-09 / 1.8e-09 | [`67_panel_glm.py`](../tests/r_parity/67_panel_glm.py) (+2) |
| `feols` | fixest::feols | R 4.5.2; fixest 0.14.0 | rel_est<=1e-06, rel_se<=1e-06 | 5.2e-15 / 2.9e-15 | [`03_hdfe.py`](../tests/r_parity/03_hdfe.py) (+2) |
| `fepois` | fixest::feglm (family="logit") / fixest::fepois | R 4.5.2; fixest 0.14.0 | rel_est<=1e-06, rel_se<=5e-05 | 9.7e-09 / 1.8e-09 | [`67_panel_glm.py`](../tests/r_parity/67_panel_glm.py) (+2) |
| `ffl_decompose` | R ddecompose::ob_decompose(reweighting = TRUE) | rifreg 1.1.0; dineq 0.1.0; ddecompose 1.0.0 | Observed difference, composition, structure, specification and reweighting errors at 1e-9 (1e-12 absolute floor; logit MLE in the path), both reference directions, for the variance, Gini (exact RIF supplied as custom_rif_function) and 10th/50th/90th percentiles. | — / — | [`test_decomp_R_parity.py`](../tests/reference_parity/test_decomp_R_parity.py) |
| `florentine_families` | R ergm flomarriage | igraph 2.3.3; sna 2.8; ergm 4.12.0; dyadRobust 0.0.1.0001 | The 20 marriage ties are identical edge for edge; the Pucci isolate is omitted (15 nodes against 16), which is documented. | — / — | [`test_network_parity.py`](../tests/reference_parity/test_network_parity.py) |
| `fracreg` | stats::glm(quasibinomial('logit')) [fractional response] | R 4.5.2 | coefficients 1e-10 abs (observed ~8e-15) | — / — | [`test_glm_ext_parity.py`](../tests/reference_parity/test_glm_ext_parity.py) (+1) |
| `frontier` | sfaR::sfacross | R 4.5.2; sfaR 1.0.1 | rel_est<=1e-06, rel_se<=5e-05 | 4.1e-08 / 4.0e-08 | [`28_frontier.py`](../tests/r_parity/28_frontier.py) (+2) |
| `g_computation` | base R stats::lm g-formula standardization (Robins 1986) | — | psi 1e-8 (observed <= 7e-16; bootstrap SE pinned loosely +/-25%) | — / — | [`test_gformula_parity.py`](../tests/reference_parity/test_gformula_parity.py) (+1) |
| `gap_closing` | R ddecompose::dfl_decompose (method='ipw') and ob_decompose (method='regression') | DasGuptR 2.2.0; ddecompose 1.0.0; cdgd 1.0.1 | Observed, counterfactual and closed gaps at 1e-9 for IPW in both directions (logit MLE in the path) and 1e-10 for regression. method='aipw' has no reference and is checked for double robustness on a known-truth DGP (T1). | — / — | [`test_decomp_R_parity.py`](../tests/reference_parity/test_decomp_R_parity.py) |
| `gardner_did` | did2s::did2s | R 4.5.2 | rel_est<=1e-06 | 4.8e-08 / 2.4e-12 | [`73_did2s.py`](../tests/r_parity/73_did2s.py) (+2) |
| `geary` | R spdep::geary.test | spdep 1.4.2; spatialreg 1.4.3 | C 2.0e-15. The closed-form variance and z are new in 1.27.0 (they were NaN whenever permutations=0) and match both spdep nulls at 1.5e-14: randomisation (with the m4/m2^2 term) and normality. | — / — | [`test_spdep_parity.py`](../tests/reference_parity/test_spdep_parity.py) |
| `getis_ord_g` | R spdep::globalG.test (binary weights) | spdep 1.4.2; spatialreg 1.4.3 | G 5.4e-16. Binary weights, which is what spdep recommends for this statistic. | — / — | [`test_spdep_parity.py`](../tests/reference_parity/test_spdep_parity.py) |
| `getis_ord_local` | R spdep::localG | spdep 1.4.2; spatialreg 1.4.3 | Gi* 1.7e-14; Gi 4.3e-13 after the star=False branch stopped borrowing Gi*'s standardisation. | — / — | [`test_spdep_parity.py`](../tests/reference_parity/test_spdep_parity.py) |
| `glm` | base R stats::glm (binomial logit + Poisson log) | R 4.5.2 | coef / logLik / AIC 1e-8 abs (observed <= 5e-13); SE ~1e-3 rel | — / — | [`test_glm_parity.py`](../tests/reference_parity/test_glm_parity.py) (+1) |
| `gsynth` | gsynth::gsynth | R 4.5.2; gsynth 1.4.0 | rel_est<=1e-06, rel_se<=1e-06 | 7.7e-14 / — | [`19_gsynth.py`](../tests/r_parity/19_gsynth.py) (+1) |
| `hdfe_ols` | fixest::feols | R 4.5.2; fixest 0.14.0 | rel_est<=1e-06, rel_se<=1e-06 | 5.2e-15 / 2.9e-15 | [`03_hdfe.py`](../tests/r_parity/03_hdfe.py) (+3) |
| `heckman` | sampleSelection::heckit | R 4.5.2; sampleSelection 1.2.14 | rel_est<=1e-06, rel_se<=0.0005 | 1.0e-11 / 1.0e-11 | [`43_heckman.py`](../tests/r_parity/43_heckman.py) (+2) |
| `het_test` | lmtest::bptest (studentized Breusch-Pagan) | R 4.5.2; lmtest 0.9.40 | statistic & p-value 1e-10 rel (observed ~1e-13) | — / — | [`test_diagnostics_parity.py`](../tests/reference_parity/test_diagnostics_parity.py) (+1) |
| `holm` | base R stats::p.adjust(method='holm') | R 4.5.2 | exact (atol 1e-15; observed 0) | — / — | [`test_mht_parity.py`](../tests/reference_parity/test_mht_parity.py) (+1) |
| `honest_did` | HonestDiD::createSensitivityResults_relativeMagnitudes | R 4.5.2; HonestDiD 0.2.8 | abs_est<=1e-06, abs_se<=1e-06 | 4.4e-16 / 5.6e-17 | [`21_honest_relmags.py`](../tests/r_parity/21_honest_relmags.py) (+2) |
| `hurdle` | pscl::hurdle(dist='poisson', zero.dist='binomial') | R 4.5.2; pscl 1.5.9 | count + zero coefficients 1e-6 abs (observed ~2e-8) | — / — | [`test_glm_ext_parity.py`](../tests/reference_parity/test_glm_ext_parity.py) (+1) |
| `impacts` | R spatialreg::impacts on a lagsarlm fit | spdep 1.4.2; spatialreg 1.4.3 | Direct, indirect and total at 1e-6, inheriting the SAR rho's own agreement. | — / — | [`test_spdep_parity.py`](../tests/reference_parity/test_spdep_parity.py) |
| `incidence_rate_ratio` | base-R closed form (rate ratio + conditional-binomial exact CI) | R 4.5.2 | estimate 1e-12; exact CI 1e-10 abs (observed ~3e-15) | — / — | [`test_epi_parity.py`](../tests/reference_parity/test_epi_parity.py) (+1) |
| `inequality_index` | base-R closed form (Gini/Theil-T/Theil-L/Atkinson; = ineq) | R 4.5.2 | all indices 1e-12 abs (observed ~2e-16) | — / — | [`test_inequality_parity.py`](../tests/reference_parity/test_inequality_parity.py) (+1) |
| `interflex` | interflex::interflex(vartype="delta", vcov.type="robust", neval=5, nbins=3, bw=1); Stata side uses the SSC interflex command | R 4.5.2 | rel_est<=1e-06, rel_se<=1e-06 | 4.0e-15 / 1.5e-14 | [`87_interflex.py`](../tests/r_parity/87_interflex.py) (+2) |
| `ipw` | base R stats::glm(binomial) + hand-rolled Hajek weighted means | — | Hajek ATE/ATT estimate 1e-9 (observed <= 2e-15; SE not pinned) | — / — | [`test_ipw_parity.py`](../tests/reference_parity/test_ipw_parity.py) (+1) |
| `iv` | AER::ivreg | R 4.5.2; AER 1.2.16 | rel_est<=1e-06, rel_se<=1e-06 | 1.1e-11 / 1.1e-11 | [`02_iv.py`](../tests/r_parity/02_iv.py) (+3) |
| `ivreg` | AER::ivreg | R 4.5.2; AER 1.2.16 | rel_est<=1e-06, rel_se<=1e-06 | 1.1e-11 / 1.1e-11 | [`02_iv.py`](../tests/r_parity/02_iv.py) (+2) |
| `join_counts` | R spdep::joincount.multi (binary weights) | spdep 1.4.2; spatialreg 1.4.3 | BB, WW and BW all exact. A reference-free guard also asserts BB + WW + BW = S0/2, the identity the BW defect violated (70.75 against 50). | — / — | [`test_spdep_parity.py`](../tests/reference_parity/test_spdep_parity.py) |
| `kaplan_meier` | survival::survfit | R 4.5.2; survival 3.8.3 | S(t) at every event time 1e-12 (observed ~3e-17); median exact | — / — | [`test_survival_km_parity.py`](../tests/reference_parity/test_survival_km_parity.py) (+1) |
| `karate_club` | R igraph::make_graph('Zachary') | igraph 2.3.3; sna 2.8; ergm 4.12.0; dyadRobust 0.0.1.0001 | Adjacency matrix identical. | — / — | [`test_network_parity.py`](../tests/reference_parity/test_network_parity.py) |
| `katz_centrality` | R igraph::alpha_centrality | igraph 2.3.3; sna 2.8; ergm 4.12.0; dyadRobust 0.0.1.0001 | 7.1e-16 with normalized = False (normalized = True L2-scales the same vector). | — / — | [`test_network_parity.py`](../tests/reference_parity/test_network_parity.py) |
| `kitagawa_decompose` | R DasGuptR::dgnpop with ratefunction sum(size*rate)/sum(size) | DasGuptR 2.2.0; ddecompose 1.0.0; cdgd 1.0.1 | Rate and composition effects on Das Gupta's Table 5.1 at 1e-10; interaction exactly 0. | — / — | [`test_decomp_R_parity.py`](../tests/reference_parity/test_decomp_R_parity.py) |
| `knn_weights` | R spdep::knearneigh + knn2nb | spdep 1.4.2; spatialreg 1.4.3 | Neighbour sets identical for all 120 points, k=4, on a random point set chosen so no distance ties make the answer non-unique. | — / — | [`test_spdep_parity.py`](../tests/reference_parity/test_spdep_parity.py) |
| `liml` | ivmodel::LIML | R 4.5.2; ivmodel 1.9.1 | rel_est<=1e-06, rel_se<=1e-06 | 1.7e-15 / 9.7e-16 | [`59_liml.py`](../tests/r_parity/59_liml.py) (+2) |
| `lm_tests` | R spdep::lm.RStests | spdep 1.4.2; spatialreg 1.4.3 | All five statistics and their p-values at 1e-9. Before the fix: LM_err 39.47 against 19.58, and Robust_LM_err 20.49 (p=6e-6) against 0.0397 (p=0.84) -- the Anselin lag-vs-error decision rule, reversed. | — / — | [`test_spdep_parity.py`](../tests/reference_parity/test_spdep_parity.py) |
| `local_projections` | lpirfs::lp_lin | R 4.5.2; lpirfs 0.2.5 | rel_est<=1e-06, rel_se<=1e-06 | 5.0e-15 / 4.4e-15 | [`34_lp.py`](../tests/r_parity/34_lp.py) (+2) |
| `logit` | stats::glm(family=binomial("logit")) | R 4.5.2; stats 4.5.2 | rel_est<=1e-06, rel_se<=1e-06 | 2.7e-11 / 2.7e-11 | [`57_logit.py`](../tests/r_parity/57_logit.py) (+2) |
| `logrank_test` | survival::survdiff | R 4.5.2; survival 3.8.3 | chi-square 1e-10 rel (observed ~8e-16); p-value 1e-10 abs | — / — | [`test_survival_km_parity.py`](../tests/reference_parity/test_survival_km_parity.py) (+1) |
| `lp_did` | direct transcription (no LP-DiD R package installed); Stata side uses the authors' lpdid | R 4.5.2 | rel_est<=1e-10, rel_se<=1e-10 | 5.0e-15 / 2.5e-15 | [`83_lpdid.py`](../tests/r_parity/83_lpdid.py) (+2) |
| `mantel_haenszel` | base-R closed form (Robins-Breslow-Greenland MH; = epiR) | R 4.5.2 | estimate, se_log, CI 1e-12 abs (observed 0) | — / — | [`test_epi_parity.py`](../tests/reference_parity/test_epi_parity.py) (+1) |
| `match` | MatchIt::matchit 4.7.2 (nearest, glm/logit distance) | — | 1:1 and 2:1 PS matching without replacement: rel <= 1e-9. Mahalanobis metric pinned against MatchIt:::mahalanobis_dist (rel <= 1e-10); greedy m_order='data'/'closest' rel <= 1e-9. | — / — | [`test_matching_r_parity.py`](../tests/reference_parity/test_matching_r_parity.py) (+1) |
| `mde` | base-R closed form (RCT minimum detectable effect) | R 4.5.2 | effect size 1e-6 abs (output rounded to 6 dp; observed ~2e-8) | — / — | [`test_power_extra_parity.py`](../tests/reference_parity/test_power_extra_parity.py) (+1) |
| `mediate` | mediation::mediate | R 4.5.2; mediation 4.5.1 | rel_est<=1e-06, rel_se<=0.1 | 6.7e-15 / 3.6e-15 | [`36_mediation.py`](../tests/r_parity/36_mediation.py) (+3) |
| `mediation` | mediation::mediate | R 4.5.2; mediation 4.5.1 | rel_est<=1e-06, rel_se<=0.1 | 6.7e-15 / 3.6e-15 | [`36_mediation.py`](../tests/r_parity/36_mediation.py) (+2) |
| `melogit` | lme4::glmer(nAGQ=8) | R 4.5.2; lme4 2.0.1 | rel_est<=1e-06, rel_se<=2e-05 | 2.4e-07 / 8.4e-07 | [`27_glmm_aghq.py`](../tests/r_parity/27_glmm_aghq.py) (+2) |
| `meta_analysis` | R metafor::rma (method='FE' and 'DL') | ivmodel 1.9.1; car 3.1.5; metafor 5.0.1 | 6.2e-16 across all nine reported quantities: fixed and random pooled effect and standard error, tau^2, Cochran Q and its p-value, I^2 and H^2. REML is not implemented here, which is a capability gap rather than a disagreement. | — / — | [`test_weakiv_meta_parity.py`](../tests/reference_parity/test_weakiv_meta_parity.py) |
| `metalearner` | econml.metalearners SLearner / TLearner / XLearner | — | S / T / X conditional-average-treatment-effect vectors match econml elementwise to 1e-12 absolute (observed <= 1.1e-15) when both fitting stages use the same base learner | — / — | [`test_metalearner_econml_parity.py`](../tests/external_parity/test_metalearner_econml_parity.py) |
| `mixed` | lme4::lmer | R 4.5.2; lme4 2.0.1 | rel_est<=1e-06, rel_se<=1e-06 | 1.3e-10 / 4.9e-11 | [`25_lmm.py`](../tests/r_parity/25_lmm.py) (+2) |
| `mlogit` | nnet::multinom | R 4.5.2; nnet 7.3.20 | rel_est<=1e-06, rel_se<=5e-05 | 2.6e-07 / 7.4e-09 | [`44_mlogit.py`](../tests/r_parity/44_mlogit.py) (+2) |
| `moran` | R spdep::moran.test (randomisation null) | spdep 1.4.2; spatialreg 1.4.3 | I 1.9e-15, expectation, variance and z all at 1e-15 on the row-standardised lattice. | — / — | [`test_spdep_parity.py`](../tests/reference_parity/test_spdep_parity.py) |
| `moran_local` | R spdep::localmoran | spdep 1.4.2; spatialreg 1.4.3 | Every Ii at 8.1e-15. | — / — | [`test_spdep_parity.py`](../tests/reference_parity/test_spdep_parity.py) |
| `moran_residuals` | R spdep::lm.morantest | spdep 1.4.2; spatialreg 1.4.3 | Statistic 5e-16; the p-value at 1e-7 once X is supplied so the Cliff-Ord regression-residual null can be formed. Both spdep alternatives are recorded because lm.morantest defaults to one-sided. | — / — | [`test_spdep_parity.py`](../tests/reference_parity/test_spdep_parity.py) |
| `mr` | R MendelianRandomization::mr_ivw through the sp.mr dispatcher | MendelianRandomization 0.10.0; TwoSampleMR 0.7.9; RadialMR 1.2.4; MRPRESSO 1.0; mr.raps 0.4.3 | IVW estimate and default random-effects SE, 1e-10. | — / — | [`test_mr_R_parity.py`](../tests/reference_parity/test_mr_R_parity.py) |
| `mr_cml` | R MendelianRandomization::mr_cML (DP = FALSE, n = 17723) | MendelianRandomization 0.10.0; TwoSampleMR 0.7.9; RadialMR 1.2.4; MRPRESSO 1.0; mr.raps 0.4.3 | Estimate and SE for every K = 0..6, the BIC-selected fit with its invalid set {12, 14}, and the MA-BIC average, all at 1e-9 (both sides iterate to |d theta| <= 1e-7). | — / — | [`test_mr_R_parity.py`](../tests/reference_parity/test_mr_R_parity.py) |
| `mr_egger` | R MendelianRandomization::mr_egger, TwoSampleMR::mr_egger_regression | MendelianRandomization 0.10.0; TwoSampleMR 0.7.9; RadialMR 1.2.4; MRPRESSO 1.0; mr.raps 0.4.3 | Slope, intercept and both SEs at 1e-10. | — / — | [`test_mr_R_parity.py`](../tests/reference_parity/test_mr_R_parity.py) |
| `mr_f_statistic` | R MendelianRandomization::mr_ivw @Fstat | MendelianRandomization 0.10.0; TwoSampleMR 0.7.9; RadialMR 1.2.4; MRPRESSO 1.0; mr.raps 0.4.3 | Mean F statistic at 1e-10. | — / — | [`test_mr_R_parity.py`](../tests/reference_parity/test_mr_R_parity.py) |
| `mr_heterogeneity` | R TwoSampleMR::mr_ivw / mr_egger_regression (Q, Q_df, Q_pval) | MendelianRandomization 0.10.0; TwoSampleMR 0.7.9; RadialMR 1.2.4; MRPRESSO 1.0; mr.raps 0.4.3 | IVW and Egger (Ruecker) Q, degrees of freedom and p-values at 1e-10. | — / — | [`test_mr_R_parity.py`](../tests/reference_parity/test_mr_R_parity.py) |
| `mr_ivw` | R MendelianRandomization::mr_ivw (default / fixed / random), TwoSampleMR::mr_ivw | MendelianRandomization 0.10.0; TwoSampleMR 0.7.9; RadialMR 1.2.4; MRPRESSO 1.0; mr.raps 0.4.3 | Estimate, SE under all three models, RSE and Cochran's Q at 1e-10 (observed <= 1e-15). | — / — | [`test_mr_R_parity.py`](../tests/reference_parity/test_mr_R_parity.py) |
| `mr_leave_one_out` | R MendelianRandomization::mr_ivw on each leave-one-out subset | MendelianRandomization 0.10.0; TwoSampleMR 0.7.9; RadialMR 1.2.4; MRPRESSO 1.0; mr.raps 0.4.3 | All 28 leave-one-out estimates and default-model SEs at 1e-10. | — / — | [`test_mr_R_parity.py`](../tests/reference_parity/test_mr_R_parity.py) |
| `mr_median` | R MendelianRandomization::mr_median (weighted / simple / penalized) | MendelianRandomization 0.10.0; TwoSampleMR 0.7.9; RadialMR 1.2.4; MRPRESSO 1.0; mr.raps 0.4.3 | Point estimates for all three weightings at 1e-10. The bootstrap SE is Monte Carlo on both sides and is not compared (T3). | — / — | [`test_mr_R_parity.py`](../tests/reference_parity/test_mr_R_parity.py) |
| `mr_mode` | R MendelianRandomization::mr_mbe (weighted / unweighted, stderror = simple) | MendelianRandomization 0.10.0; TwoSampleMR 0.7.9; RadialMR 1.2.4; MRPRESSO 1.0; mr.raps 0.4.3 | Point estimates for both weightings at 1e-10 -- the same point of the same 512-point density grid. The bootstrap SE is Monte Carlo on both sides and is not compared (T3). | — / — | [`test_mr_R_parity.py`](../tests/reference_parity/test_mr_R_parity.py) |
| `mr_multivariable` | R MendelianRandomization::mr_mvivw (default random effects) | MendelianRandomization 0.10.0; TwoSampleMR 0.7.9; RadialMR 1.2.4; MRPRESSO 1.0; mr.raps 0.4.3 | Three direct effects and SEs at 1e-10. | — / — | [`test_mr_R_parity.py`](../tests/reference_parity/test_mr_R_parity.py) |
| `mr_pleiotropy_egger` | R TwoSampleMR::mr_egger_regression (intercept test) | MendelianRandomization 0.10.0; TwoSampleMR 0.7.9; RadialMR 1.2.4; MRPRESSO 1.0; mr.raps 0.4.3 | Intercept, SE and t(n - 2) p-value at 1e-10. | — / — | [`test_mr_R_parity.py`](../tests/reference_parity/test_mr_R_parity.py) |
| `mr_presso` | R MRPRESSO::mr_presso (NbDistribution = 2000) | MendelianRandomization 0.10.0; TwoSampleMR 0.7.9; RadialMR 1.2.4; MRPRESSO 1.0; mr.raps 0.4.3 | Raw estimate and SE, observed RSS, and the outlier-corrected estimate and SE at 1e-10; outlier set {12, 14} identical. The simulated p-values are Monte Carlo on both sides (T3) and follow the reference's k / B convention. | — / — | [`test_mr_R_parity.py`](../tests/reference_parity/test_mr_R_parity.py) |
| `mr_radial` | R RadialMR::ivw_radial (alpha = 0.05, no Bonferroni) | MendelianRandomization 0.10.0; TwoSampleMR 0.7.9; RadialMR 1.2.4; MRPRESSO 1.0; mr.raps 0.4.3 | Square-root weights, per-variant Q contributions and total Q at 1e-10; the outlier set is identical with bonferroni=False (StatsPAI's default applies Bonferroni). | — / — | [`test_mr_R_parity.py`](../tests/reference_parity/test_mr_R_parity.py) |
| `mr_steiger` | R TwoSampleMR::mr_steiger with r from get_r_from_bsen | MendelianRandomization 0.10.0; TwoSampleMR 0.7.9; RadialMR 1.2.4; MRPRESSO 1.0; mr.raps 0.4.3 | R^2 on both traits and the direction at 1e-10; the p-value (1.8e-73) at 1e-12. | — / — | [`test_mr_R_parity.py`](../tests/reference_parity/test_mr_R_parity.py) |
| `multiway_cluster_vcov` | sandwich::vcovCL(cluster=~g1+g2+g3) | R 4.5.2; sandwich 3.1.1 | rel_est<=1e-06, rel_se<=1e-06 | 2.1e-15 / 2.1e-15 | [`56_multiway_cluster.py`](../tests/r_parity/56_multiway_cluster.py) (+2) |
| `nbreg` | MASS::glm.nb | R 4.5.2; MASS 7.3.65 | rel_est<=1e-06, rel_se<=0.005 | 6.0e-10 / 1.3e-10 | [`42_nbreg.py`](../tests/r_parity/42_nbreg.py) (+2) |
| `netlm` | R sna::netlm | igraph 2.3.3; sna 2.8; ergm 4.12.0; dyadRobust 0.0.1.0001 | Coefficients 2e-15 directed and undirected. QAP p-values are permutation draws and are not compared. | — / — | [`test_network_parity.py`](../tests/reference_parity/test_network_parity.py) |
| `netlogit` | R sna::netlogit | igraph 2.3.3; sna 2.8; ergm 4.12.0; dyadRobust 0.0.1.0001 | Coefficients 1.4e-9 (IRLS on both sides). QAP p-values are permutation draws and are not compared. | — / — | [`test_network_parity.py`](../tests/reference_parity/test_network_parity.py) |
| `network_components` | R igraph::components | igraph 2.3.3; sna 2.8; ergm 4.12.0; dyadRobust 0.0.1.0001 | Counts and sizes exact on a disconnected graph; weak and strong counts on the directed graph. | — / — | [`test_network_parity.py`](../tests/reference_parity/test_network_parity.py) |
| `network_modularity` | R igraph::modularity | igraph 2.3.3; sna 2.8; ergm 4.12.0; dyadRobust 0.0.1.0001 | Exact for a fixed split and for igraph's own fast-greedy partition. | — / — | [`test_network_parity.py`](../tests/reference_parity/test_network_parity.py) |
| `network_summary` | R igraph edge_density / diameter / mean_distance / transitivity | igraph 2.3.3; sna 2.8; ergm 4.12.0; dyadRobust 0.0.1.0001 | Density, diameter, mean path length, transitivity and assortativity exact; average clustering matches igraph::transitivity(type = 'average', isolates = 'zero'), the convention used here. | — / — | [`test_network_parity.py`](../tests/reference_parity/test_network_parity.py) |
| `number_needed_to_treat` | base-R closed form (NNT = 1/risk difference) | R 4.5.2 | estimate 1e-12 abs (observed 0); CI not pinned | — / — | [`test_epi_parity.py`](../tests/reference_parity/test_epi_parity.py) (+1) |
| `oaxaca` | oaxaca::oaxaca | R 4.5.2; oaxaca 0.1.5 | rel_est<=1e-06, rel_se<=0.05 | 6.3e-16 / 1.3e-16 | [`30_oaxaca.py`](../tests/r_parity/30_oaxaca.py) (+3) |
| `odds_ratio` | base-R closed form (Woolf logit; = epiR::epi.2by2) | R 4.5.2 | estimate, se_log, CI 1e-12 abs (observed 0) | — / — | [`test_epi_parity.py`](../tests/reference_parity/test_epi_parity.py) (+1) |
| `ologit` | MASS::polr(method="logistic") | R 4.5.2; MASS 7.3.65 | rel_est<=1e-06, rel_se<=1e-05 | 1.8e-07 / 3.5e-07 | [`45_ologit.py`](../tests/r_parity/45_ologit.py) (+2) |
| `oprobit` | MASS::polr(method="probit") | R 4.5.2; MASS 7.3.65 | rel_est<=1e-06, rel_se<=1e-06 | 3.4e-07 / 2.8e-08 | [`49_oprobit.py`](../tests/r_parity/49_oprobit.py) (+2) |
| `overlap_weights` | WeightIt::weightit 1.7.0 (method='glm'), R 4.5.2 | R 4.5.2; WeightIt 1.7.0 | All four estimands of the shared-propensity family (Li, Li & Li 2019 Table 1) relative to WeightIt: ATO 2.5e-14, ATE 4.1e-14, ATT 2.3e-14, ATC 4.1e-14. The propensity score itself matches R glm(family=binomial) to 2.6e-14 absolute. | — / — | [`test_overlap_weights_r_parity.py`](../tests/reference_parity/test_overlap_weights_r_parity.py) (+1) |
| `pagerank` | R igraph::page_rank (damping 0.85) | igraph 2.3.3; sna 2.8; ergm 4.12.0; dyadRobust 0.0.1.0001 | 5e-12 undirected, 1e-12 directed. | — / — | [`test_network_parity.py`](../tests/reference_parity/test_network_parity.py) |
| `panel` | plm::plm + plm::phtest | R 4.5.2; plm 2.6.7 | rel_est<=1e-06, rel_se<=0.001 | 4.7e-14 / 1.5e-15 | [`35_panel.py`](../tests/r_parity/35_panel.py) (+2) |
| `panel_fgls` | Stata 18 xtgls, panels(hetero) | Stata 18 MP | 6.3e-16 on every coefficient and 7.0e-16 on every standard error against the two-step default, and 4.7e-08 against `xtgls, igls` for the iterated variant. A reference-free test also asserts the two are distinct estimators, so a silent return to iterating fails. | — / — | [`test_panel_stata_parity.py`](../tests/reference_parity/test_panel_stata_parity.py) |
| `panel_logit` | Stata 18 xtlogit, re | Stata 18 MP | Graded by CONVERGENCE rather than a fixed tolerance: Stata integrates adaptively and StatsPAI does not, so the honest claim is that agreement improves as the Gauss-Hermite rule is refined. Observed 3.8e-04 at 12 points and 2.4e-07 at 60, with the log-likelihood at 2.1e-08 -- the sharpest single check, since it is the same objective evaluated at the same optimum. sigma_u and rho at 1e-4. | — / — | [`test_panel_stata_parity.py`](../tests/reference_parity/test_panel_stata_parity.py) |
| `panel_probit` | Stata 18 xtprobit, re | Stata 18 MP | Same convergence grading: 8.0e-05 at 12 quadrature points and 4.0e-08 at 60, log-likelihood at 1.7e-09, sigma_u and rho at 1e-4. | — / — | [`test_panel_stata_parity.py`](../tests/reference_parity/test_panel_stata_parity.py) |
| `panel_qtet` | qte::panel.qtet 1.3.1 (Callaway & Li 2019) | — | all 19 quantiles: abs < 1e-8 (observed 6.8e-12); ATT abs < 1e-6. panel.qtet composes ordinary ecdf evaluations and type-7 quantiles, both of which have exact numpy equivalents, so this is machine-precision agreement rather than a tolerance band. | — / — | [`test_panel_qtet_parity.py`](../tests/reference_parity/test_panel_qtet_parity.py) (+1) |
| `poisson` | stats::glm(family=poisson()) | R 4.5.2; stats 4.5.2 | rel_est<=1e-06, rel_se<=1e-06 | 9.2e-15 / 8.7e-12 | [`58_poisson.py`](../tests/r_parity/58_poisson.py) (+2) |
| `policy_tree` | policytree::policy_tree | R 4.5.2; policytree 1.2.4 | rel_est<=1e-06, rel_se<=1e-06 | 9.6e-16 / 1.4e-16 | [`70_policy_tree.py`](../tests/r_parity/70_policy_tree.py) (+2) |
| `power_cluster_rct` | base-R closed form (design-effect-inflated z-approx power) | R 4.5.2 | power 1e-12 abs (observed ~2e-16) | — / — | [`test_power_extra_parity.py`](../tests/reference_parity/test_power_extra_parity.py) (+1) |
| `power_logrank` | base-R closed form (Schoenfeld log-rank power) | R 4.5.2 | power 1e-12 abs (observed ~2e-16) | — / — | [`test_power_parity.py`](../tests/reference_parity/test_power_parity.py) (+1) |
| `power_rct` | base-R closed form (two-sample pooled-sigma z-approx power) | R 4.5.2 | power 1e-12 abs (observed ~2e-16) | — / — | [`test_power_parity.py`](../tests/reference_parity/test_power_parity.py) (+1) |
| `power_two_proportions` | base-R closed form (unpooled Wald two-proportion z-approx) | R 4.5.2 | power 1e-12 abs (observed ~2e-16) | — / — | [`test_power_parity.py`](../tests/reference_parity/test_power_parity.py) (+1) |
| `ppmlhdfe` | fixest::fepois | R 4.5.2; fixest 0.14.0 | rel_est<=1e-06, rel_se<=2e-06 | 4.9e-13 / 2.2e-15 | [`37_ppmlhdfe.py`](../tests/r_parity/37_ppmlhdfe.py) (+2) |
| `prevalence_ratio` | base-R closed form (Katz-log; = epiR::epi.2by2) | R 4.5.2 | estimate, se_log, CI 1e-12 abs (observed ~2e-16) | — / — | [`test_epi_parity.py`](../tests/reference_parity/test_epi_parity.py) (+1) |
| `probit` | stats::glm(family=binomial("probit")) | R 4.5.2; stats 4.5.2 | rel_est<=1e-06, rel_se<=0.01 | 3.1e-07 / 1.6e-08 | [`48_probit.py`](../tests/r_parity/48_probit.py) (+2) |
| `psm` | MatchIt::matchit | R 4.5.2; MatchIt 4.7.2 | rel_est<=1e-06, rel_se<=1e-06 | 1.2e-15 / 2.0e-16 | [`11_psm.py`](../tests/r_parity/11_psm.py) (+2) |
| `psmatch2` | Stata 18 MP + psmatch2 4.0.12 / pstest 4.2.2 (Leuven & Sianesi 2003) | Stata 18 MP; psmatch2 4.0.12 | Observed py<->Stata relative gaps on the committed fixtures: nearest-neighbour ATT 1.2e-16 and its analytic SE exactly 0 (as is _weight per row); radius ATT 2.8e-16; Abadie-Imbens ai(1)/ai(2) SE 3.0e-15/1.7e-15; PSM-DID across all five weight regimes 2.0e-14; pstest per-covariate rows 1.3e-14; Mahalanobis ATT 1.2e-13; llr ATT 2.1e-10 (worst of four kernels); kernel ATT 9.1e-10; pstest summary block 1.3e-9. The two loosest rows are bounded by fixture precision, not by the estimator: pstest accumulates MeanBias/MedBias in a Stata float (2.6e-9) and r(seatt) for the llr reroute was captured at 8 significant digits (9.9e-9). | — / — | [`test_psmatch2_parity.py`](../tests/reference_parity/test_psmatch2_parity.py) (+3) |
| `qreg` | quantreg::rq | R 4.5.2; quantreg 6.1 | rel_est<=1e-06, rel_se<=0.1 | 3.3e-15 / 4.4e-15 | [`40_qreg.py`](../tests/r_parity/40_qreg.py) (+2) |
| `rd_honest` | RDHonest::RDHonest 1.0.1.9000 (Armstrong & Kolesar) | R 4.5.2; RDHonest 1.0.1.9000 | estimate / std.error / maximum.bias / conf.low / conf.high 1e-9 rel at fixed bandwidth; 1e-6 rel when the bandwidth and M are selected | — / — | [`test_rdhonest_parity.py`](../tests/reference_parity/test_rdhonest_parity.py) (+1) |
| `rdbwselect` | rdrobust::rdbwselect; Stata side uses the authors' rdbwselect ado. certwo is R-only: Stata rdbwselect 10.0.0 exits r(3200) on it, including on the package's own rdrobust_senate.dta | R 4.5.2; rdrobust 3.0.0 | rel_est<=1e-06 | 9.4e-13 / 3.5e-09 | [`88_rdbwselect.py`](../tests/r_parity/88_rdbwselect.py) (+2) |
| `rddensity` | rddensity::rddensity | R 4.5.2; rddensity 2.6 | rel_est<=1e-06, rel_se<=1e-06 | 9.3e-12 / 1.8e-11 | [`09_rddensity.py`](../tests/r_parity/09_rddensity.py) (+2) |
| `rdmc` | rdmulti::rdmc 2.0.0 (Cattaneo, Titiunik, Vazquez-Bare & Keele) | R 4.5.2; rdmulti 2.0.0 | per-cutoff coefficients, robust coefficients, robust SEs and the pooled weighted estimate 1e-9 rel; selected bandwidths 1e-5 rel | — / — | [`test_rdmulti_parity.py`](../tests/reference_parity/test_rdmulti_parity.py) (+1) |
| `rdms` | rdmulti::rdms; Stata side uses the rdms ado from the rdpackages GitHub mirror (rdmulti is not on SSC: ssc describe rdmulti returns r(601)), installed into a local gitignored ado path | R 4.5.2 | rel_est<=1e-06, rel_se<=1e-06 | 1.2e-12 / 5.4e-10 | [`89_rdms.py`](../tests/r_parity/89_rdms.py) (+2) |
| `rdpower` | rdpower::rdpower 3.0 (Cattaneo, Titiunik & Vazquez-Bare) | R 4.5.2; rdpower 3.0 | robust bias-corrected SE & power 1e-8 rel (observed 4.2e-14) | — / — | [`test_rdlocrand_parity.py`](../tests/reference_parity/test_rdlocrand_parity.py) (+1) |
| `rdrandinf` | rdlocrand::rdrandinf 2.0 (Cattaneo, Titiunik & Vazquez-Bare) | R 4.5.2; rdlocrand 2.0 | observed statistic & asymptotic p-value 1e-8 rel (observed 2.3e-15) | — / — | [`test_rdlocrand_parity.py`](../tests/reference_parity/test_rdlocrand_parity.py) (+1) |
| `rdrobust` | rdrobust::rdrobust | R 4.5.2; rdrobust 3.0.0 | rel_est<=1e-06, rel_se<=0.1 | 2.5e-14 / 9.4e-11 | [`06_rd.py`](../tests/r_parity/06_rd.py) (+2) |
| `rdsampsi` | rdpower::rdsampsi 3.0 (Cattaneo, Titiunik & Vazquez-Bare) | R 4.5.2; rdpower 3.0 | required sample sizes n_left / n_right / n_total asserted as exact integer equality (no tolerance) | — / — | [`test_rdlocrand_parity.py`](../tests/reference_parity/test_rdlocrand_parity.py) (+1) |
| `rdwinselect` | rdlocrand::rdwinselect 2.0 (Cattaneo, Titiunik & Vazquez-Bare) | R 4.5.2; rdlocrand 2.0 | window grid 1e-12 rel (observed 0); per-window counts Nl / Nr asserted as exact integer equality | — / — | [`test_rdlocrand_parity.py`](../tests/reference_parity/test_rdlocrand_parity.py) (+1) |
| `reciprocity` | R igraph::reciprocity | igraph 2.3.3; sna 2.8; ergm 4.12.0; dyadRobust 0.0.1.0001 | Exact on a 40-node directed graph. | — / — | [`test_network_parity.py`](../tests/reference_parity/test_network_parity.py) |
| `regress` | lm + sandwich::vcovHC | R 4.5.2; sandwich 3.1.1 | rel_est<=1e-06, rel_se<=1e-06 | 1.1e-12 / 1.3e-12 | [`01_ols.py`](../tests/r_parity/01_ols.py) (+2) |
| `relative_risk` | base-R closed form (Katz-log; = epiR::epi.2by2 / Stata epitab) | R 4.5.2 | estimate, se_log, CI 1e-12 abs (observed 0) | — / — | [`test_epi_parity.py`](../tests/reference_parity/test_epi_parity.py) (+1) |
| `reset_test` | lmtest::resettest(power=2:3, type='fitted') | R 4.5.2; lmtest 0.9.40 | F-statistic & p-value 1e-10 rel (observed ~1e-13) | — / — | [`test_diagnostics_parity.py`](../tests/reference_parity/test_diagnostics_parity.py) (+1) |
| `rif_decomposition` | dineq::rif + manual OLS | R 4.5.2; dineq 0.1.0 | rel_est<=1e-06, rel_se<=1e-06 | 2.2e-15 / 1.4e-16 | [`32_rif.py`](../tests/r_parity/32_rif.py) (+2) |
| `rifreg` | R rifreg::rifreg (variance, quantiles) and dineq::rif + lm (Gini) | rifreg 1.1.0; dineq 0.1.0; ddecompose 1.0.0 | Coefficients at 1e-10 for the variance and the 10th/50th/90th percentiles (quantile_convention='rifreg') and for the Gini against dineq's exact RIF; the stock rifreg Gini, which integrates the Lorenz curve numerically, agrees to 1e-4. | — / — | [`test_decomp_R_parity.py`](../tests/reference_parity/test_decomp_R_parity.py) |
| `risk_difference` | base-R closed form (Wald; = epiR::epi.2by2 / Stata epitab) | R 4.5.2 | estimate, se, CI 1e-12 abs (observed 0) | — / — | [`test_epi_parity.py`](../tests/reference_parity/test_epi_parity.py) (+1) |
| `sac` | R spatialreg::sacsarlm | spdep 1.4.2; spatialreg 1.4.3 | rho and lambda at 1e-5, slope coefficients at 1e-6 -- a bounded two-parameter ML line search on both sides. | — / — | [`test_spdep_parity.py`](../tests/reference_parity/test_spdep_parity.py) |
| `sar` | spatialreg::lagsarlm / spatialreg::errorsarlm / spatialreg::lagsarlm(Durbin=TRUE) | R 4.5.2; spatialreg 1.4.3 | rel_est<=1e-06, rel_se<=1e-06 | 8.3e-08 / 5.1e-08 | [`65_spatial.py`](../tests/r_parity/65_spatial.py) (+2) |
| `sar_gmm` | spatialreg::stsls(W2X=FALSE) / spatialreg::GMerrorsar | R 4.5.2; spatialreg 1.4.3 | rel_est<=1e-06, rel_se<=1e-06 | 4.6e-08 / 7.3e-16 | [`66_spatial_gmm.py`](../tests/r_parity/66_spatial_gmm.py) (+2) |
| `sbw` | sbw::sbw 1.2 (Zubizarreta 2015), quadprog solver | — | ATT rel <= 1e-8 (observed <= 4e-10) under both standardisation conventions: tolerance_scale='target' == bal_std='target' and 'group' == bal_std='group', at bal_tol 0.05 and 0.02. | — / — | [`test_matching_r_parity.py`](../tests/reference_parity/test_matching_r_parity.py) (+1) |
| `sdid` | synthdid::synthdid_estimate | R 4.5.2; synthdid 0.0.9 | rel_est<=1e-06, rel_se<=1e-06 | 2.6e-15 / 7.2e-08 | [`12_sdid.py`](../tests/r_parity/12_sdid.py) (+2) |
| `sdm` | spatialreg::lagsarlm / spatialreg::errorsarlm / spatialreg::lagsarlm(Durbin=TRUE) | R 4.5.2; spatialreg 1.4.3 | rel_est<=1e-06, rel_se<=1e-06 | 8.3e-08 / 5.1e-08 | [`65_spatial.py`](../tests/r_parity/65_spatial.py) (+2) |
| `sem` | spatialreg::lagsarlm / spatialreg::errorsarlm / spatialreg::lagsarlm(Durbin=TRUE) | R 4.5.2; spatialreg 1.4.3 | rel_est<=1e-06, rel_se<=1e-06 | 8.3e-08 / 5.1e-08 | [`65_spatial.py`](../tests/r_parity/65_spatial.py) (+2) |
| `sem_gmm` | spatialreg::stsls(W2X=FALSE) / spatialreg::GMerrorsar | R 4.5.2; spatialreg 1.4.3 | rel_est<=1e-06, rel_se<=1e-06 | 4.6e-08 / 7.3e-16 | [`66_spatial_gmm.py`](../tests/r_parity/66_spatial_gmm.py) (+2) |
| `sensemakr` | sensemakr::sensemakr | R 4.5.2; sensemakr 0.1.6 | rel_est<=1e-06, rel_se<=1e-06 | 5.0e-08 / 5.0e-08 | [`22_sensemakr.py`](../tests/r_parity/22_sensemakr.py) (+2) |
| `slx` | R spatialreg::lmSLX | spdep 1.4.2; spatialreg 1.4.3 | Every coefficient at 1e-10. | — / — | [`test_spdep_parity.py`](../tests/reference_parity/test_spdep_parity.py) |
| `sqreg` | R quantreg::rq (Barrodale-Roberts), Koenker 2005 | quantreg see sqreg_R.json provenance | Coefficients 3.5e-14 against quantreg::rq at tau = 0.25 / 0.50 / 0.75 -- both sides minimise the same pinball loss with the same simplex. Standard errors differ from R's se='iid' by ONE SCALAR PER QUANTILE, constant across coefficients to 6e-16: the sandwich is identical and only the sparsity estimate 1/f(0) differs (Powell kernel here, Koenker-Bassett with a Siddiqui/Hall-Sheather bandwidth there). The test asserts the ratio's constancy rather than a numerical band, which a structural difference could not satisfy. R's default se='nid' (Hendricks-Koenker, also Stata qreg's) is a third convention and is recorded as one. | — / — | [`test_sqreg_parity.py`](../tests/reference_parity/test_sqreg_parity.py) |
| `stacked_did` | hand-written stack + fixest::feols | R 4.5.2; fixest 0.14.0 | rel_est<=1e-06 | 3.9e-13 / 7.1e-13 | [`75_stacked.py`](../tests/r_parity/75_stacked.py) (+2) |
| `staggered_rollout` | staggered::staggered / staggered_cs / staggered_sa (1.2.2) | R 4.5.2 | rel_est<=1e-10 | 9.1e-16 / 5.7e-16 | [`82_staggered.py`](../tests/r_parity/82_staggered.py) (+2) |
| `sun_abraham` | fixest::sunab | R 4.5.2; fixest 0.14.0 | rel_est<=1e-06, rel_se<=0.03 | 2.8e-11 / 2.7e-11 | [`05_sunab.py`](../tests/r_parity/05_sunab.py) (+2) |
| `sureg` | systemfit::systemfit(method="SUR", noDfCor) | R 4.5.2; systemfit 1.1.30 | rel_est<=1e-06, rel_se<=1e-06 | 1.5e-14 / 1.5e-15 | [`60_sureg.py`](../tests/r_parity/60_sureg.py) (+2) |
| `svyglm` | survey::svyglm (design-based GLM + linearization SE) | R 4.5.2 | coefficients + SE 1e-10 abs (observed ~2e-15 / 6e-15) | — / — | [`test_survey_parity.py`](../tests/reference_parity/test_survey_parity.py) (+1) |
| `svymean` | survey::svymean (Horvitz-Thompson/Hajek + Taylor SE) | R 4.5.2 | estimate + SE 1e-10 abs (observed ~5e-15 / 8e-17) | — / — | [`test_survey_parity.py`](../tests/reference_parity/test_survey_parity.py) (+1) |
| `svytotal` | survey::svytotal (Horvitz-Thompson + Taylor SE) | R 4.5.2 | estimate 1e-12 rel; SE 1e-10 rel (observed ~2e-12 / 1e-14) | — / — | [`test_survey_parity.py`](../tests/reference_parity/test_survey_parity.py) (+1) |
| `synth` | Synth::synth | R 4.5.2; Synth 1.1.10 | rel_est<=1e-06, rel_se<=1e-06 | 7.8e-08 / 7.7e-08 | [`52_scm_unique.py`](../tests/r_parity/52_scm_unique.py) (+2) |
| `three_sls` | R systemfit::systemfit(method='3SLS') | R 4.5.2; systemfit 1.1.30 | coef 1e-9 abs (observed <= 1e-15); SE ~5e-3 rel | — / — | [`test_threesls_parity.py`](../tests/reference_parity/test_threesls_parity.py) (+1) |
| `tmle` | tmle::tmle | R 4.5.2; tmle 2.1.1 | rel_est<=1e-06, rel_se<=1e-06 | 1.9e-09 / 1.9e-09 | [`72_tmle.py`](../tests/r_parity/72_tmle.py) (+2) |
| `tobit` | censReg::censReg | R 4.5.2; censReg 0.5.38 | rel_est<=1e-06, rel_se<=1e-05 | 2.8e-08 / 2.8e-08 | [`41_tobit.py`](../tests/r_parity/41_tobit.py) (+2) |
| `transitivity` | R igraph::transitivity(type = 'global') | igraph 2.3.3; sna 2.8; ergm 4.12.0; dyadRobust 0.0.1.0001 | Exact on karate. | — / — | [`test_network_parity.py`](../tests/reference_parity/test_network_parity.py) |
| `truncreg` | truncreg::truncreg(method="NR") | R 4.5.2; truncreg 0.2.5 | rel_est<=1e-06, rel_se<=0.0001 | 3.3e-08 / 9.5e-08 | [`62_truncreg.py`](../tests/r_parity/62_truncreg.py) (+2) |
| `twoway_cluster` | sandwich::vcovCL(cluster=~g1+g2) | R 4.5.2; sandwich 3.1.1 | rel_est<=1e-06, rel_se<=1e-06 | 7.8e-16 / 7.8e-16 | [`54_twoway_cluster.py`](../tests/r_parity/54_twoway_cluster.py) (+2) |
| `var` | vars::VAR | R 4.5.2; vars 1.6.1 | rel_est<=1e-06, rel_se<=1e-06 | 3.1e-15 / 6.6e-15 | [`33_var.py`](../tests/r_parity/33_var.py) (+2) |
| `vif` | R car::vif | ivmodel 1.9.1; car 3.1.5; metafor 5.0.1 | 2.0e-16 on every variance inflation factor, once the returned values stopped being rounded to two decimals. | — / — | [`test_weakiv_meta_parity.py`](../tests/reference_parity/test_weakiv_meta_parity.py) |
| `wooldridge_did` | etwfe::etwfe + emfx | R 4.5.2; etwfe 0.6.2 | rel_est<=1e-06, rel_se<=0.001 | 1.8e-13 / 3.9e-14 | [`17_etwfe.py`](../tests/r_parity/17_etwfe.py) (+2) |
| `xtabond` | plm::pgmm | R 4.5.2; plm 2.6.7 | rel_est<=1e-06, rel_se<=1e-06 | 9.0e-16 / 1.4e-15 | [`50_xtabond.py`](../tests/r_parity/50_xtabond.py) (+2) |
| `yu_elwert_decompose` | R cdgd::cdgd0_manual on independently fitted within-cell lm / within-group glm nuisances | DasGuptR 2.2.0; ddecompose 1.0.0; cdgd 1.0.1 | method='efficient': disparity, baseline, prevalence, effect, selection and their EIF standard errors at 1e-9. method='plugin' has no reference implementation and is covered by its exact additivity identity. | — / — | [`test_decomp_R_parity.py`](../tests/reference_parity/test_decomp_R_parity.py) |
| `zip_model` | pscl::zeroinfl(dist="poisson") | R 4.5.2; pscl 1.5.9 | rel_est<=1e-06, rel_se<=0.0001 | 7.7e-08 / 1.1e-07 | [`63_zip.py`](../tests/r_parity/63_zip.py) (+2) |

## aligned — 21 functions

Agreement within a documented, pre-registered looser tolerance.

| function | reference | versions | tolerance | rel err (R / Stata) | test |
| --- | --- | --- | --- | --- | --- |
| `aft` | survival::survreg (Weibull AFT) | R 4.5.2; survival 3.8.3 | coefficients & log-scale 5e-5 abs (observed ~1e-5) | — / — | [`test_aft_parity.py`](../tests/reference_parity/test_aft_parity.py) (+1) |
| `augsynth` | augsynth::augsynth | R 4.5.2; augsynth 0.2.0 | rel_est<=2e-05, rel_se<=1e-06 | 7.9e-06 / — | [`18_augsynth.py`](../tests/r_parity/18_augsynth.py) (+1) |
| `causal_forest` | grf::causal_forest | R 4.5.2; grf 2.6.1 | rel_est<=0.01, rel_se<=0.25 | 1.9e-03 / — | [`13_causal_forest.py`](../tests/r_parity/13_causal_forest.py) (+1) |
| `cbps` | CBPS::CBPS 0.24 (Imai & Ratkovic 2014) | — | ATE over/exact and ATT exact: rel <= 5e-3 (R's optimiser slack). ATT over is NOT pinned to R -- CBPS's ATT gradient mis-scales the balance block by n/n_1 and stops off-stationarity; StatsPAI is asserted to attain strictly better covariate balance instead. | — / — | [`test_matching_r_parity.py`](../tests/reference_parity/test_matching_r_parity.py) (+1) |
| `centrality` | R igraph degree / betweenness / closeness / page_rank / eigen_centrality | igraph 2.3.3; sna 2.8; ergm 4.12.0; dyadRobust 0.0.1.0001 | degree, betweenness (normalised and raw), closeness and PageRank at 1e-10 on Zachary's karate club. The eigenvector column is L2-normalised (networkx) where igraph max-scales it: the ratio is constant across nodes to 1e-14, so it is the same vector under a documented normalisation. | — / — | [`test_network_parity.py`](../tests/reference_parity/test_network_parity.py) |
| `cic` | qte::CiC | R 4.5.2 | rel_est<=1e-06 | 2.4e-15 / 6.0e-03 | [`74_cic.py`](../tests/r_parity/74_cic.py) (+2) |
| `cloglog` | stats::glm(binomial('cloglog')) | R 4.5.2 | coefficients 5e-5 abs (observed ~1e-5; IRLS convergence) | — / — | [`test_glm_ext_parity.py`](../tests/reference_parity/test_glm_ext_parity.py) (+1) |
| `community_detection` | R igraph::cluster_louvain (T3: randomised on both sides) | igraph 2.3.3; sna 2.8; ergm 4.12.0; dyadRobust 0.0.1.0001 | T3, not T2: 200 seeded runs on karate have mean modularity within four combined standard errors of igraph's 200 runs (0.4157 vs 0.4145), and both reach the same maximum, 0.41979, the known optimum for this graph. | — / — | [`test_network_parity.py`](../tests/reference_parity/test_network_parity.py) |
| `conditional_lr_ci` | R ivmodel::CLR | ivmodel 1.9.1 | Monte Carlo by construction and graded as such: ivmodel integrates Moreira's conditional distribution while StatsPAI simulates it, so the two cannot agree deterministically. The test asserts the error SHRINKS with n_sim rather than pinning a number -- 3.8e-3 at n_sim=5,000 and 1.7e-4 at 200,000 -- which is the only honest statement about a simulated critical value. The endpoints themselves are bisected off-grid, so the residual is the critical value and not the grid. | — / — | [`test_weakiv_meta_parity.py`](../tests/reference_parity/test_weakiv_meta_parity.py) |
| `functional_form_test` | didFF::didFF | R 4.5.2 | rel_est<=0.001 | 1.3e-14 / — | [`79_didff.py`](../tests/r_parity/79_didff.py) (+1) |
| `genmatch` | Matching::Match 4.10-15 (Weight = 3, Weight.matrix) | — | Deterministic kernel only: given the same diagonal W, the 1-NN assignment agrees with Matching::Match on all 163 uniquely matched treated units on MatchIt::lalonde. | — / — | [`test_matching_r_parity.py`](../tests/reference_parity/test_matching_r_parity.py) (+1) |
| `hits` | R igraph::hits_scores | igraph 2.3.3; sna 2.8; ergm 4.12.0; dyadRobust 0.0.1.0001 | Hub and authority vectors are igraph's up to normalisation: L1 here (documented), max = 1 in igraph; the ratio is constant across nodes to 1e-11. | — / — | [`test_network_parity.py`](../tests/reference_parity/test_network_parity.py) |
| `mr_raps` | R mr.raps 0.4.3 (simple / overdispersed / overdispersed.robust) | MendelianRandomization 0.10.0; TwoSampleMR 0.7.9; RadialMR 1.2.4; MRPRESSO 1.0; mr.raps 0.4.3 | Simple and L2-overdispersed fits (beta, SE, tau2) at 1e-8. Robust Huber / Tukey: the sandwich reproduces R's SEs at 1e-10 when evaluated at R's own (beta, tau2) and integrate() moments; the fitted values agree to 5e-5 (beta), 5e-4 (SE), 1.5e-3 (tau2) because R stops uniroot at its default tolerance and integrate() at 1.2e-4 -- a test shows StatsPAI's root satisfies the estimating equation more tightly than R's. | — / — | [`test_mr_R_parity.py`](../tests/reference_parity/test_mr_R_parity.py) |
| `optimal_match` | optmatch::pairmatch 0.10.8 on a logit propensity score | — | Total matched distance <= optmatch's (1 + 1e-6). The matched pairs are not pinned: the assignment problem is degenerate on this data, so equally optimal solutions report different ATTs. | — / — | [`test_matching_r_parity.py`](../tests/reference_parity/test_matching_r_parity.py) (+1) |
| `pretrends_power` | pretrends::pretrends / pretrends::slope_for_power (GitHub, not CRAN) | R 4.5.2 | rel_est<=0.001 | 4.0e-05 / 1.4e-04 | [`76_pretrends.py`](../tests/r_parity/76_pretrends.py) (+2) |
| `pretrends_slope_for_power` | pretrends::pretrends / pretrends::slope_for_power (GitHub, not CRAN) | R 4.5.2 | rel_est<=0.001 | 4.0e-05 / 1.4e-04 | [`76_pretrends.py`](../tests/r_parity/76_pretrends.py) (+2) |
| `qdid` | qte::QDiD 1.3.1 | — | max deviation / scale < 0.08, sign agreement on the large effects and correlation > 0.999. R's quantiles come from BMisc::weighted_quantile (stats::optimize on a piecewise-linear check function, which has plateaus where every point is a minimiser) while sp.qdid interpolates the empirical inverse CDF; on this fixture the gap is at most ~152 currency units against effects running to ~8900. | — / — | [`test_qdid_parity.py`](../tests/reference_parity/test_qdid_parity.py) (+1) |
| `qte` | qte::ci.qte / qte::ci.qtet 1.3.1 (Firpo 2007) | — | max relative deviation < 0.01 on lalonde.exp / lalonde.psid. Both sides minimise the same weighted check function; R's BMisc::weighted_quantile uses a golden-section search whose answer on a plateau is an optimiser artifact, so point-value equality is not asserted. | — / — | [`test_firpo_qte_parity.py`](../tests/reference_parity/test_firpo_qte_parity.py) (+1) |
| `survreg` | survival::survreg (Weibull AFT) | R 4.5.2; survival 3.8.3 | coefficients & log-scale 5e-5 abs (observed ~1e-5) | — / — | [`test_aft_parity.py`](../tests/reference_parity/test_aft_parity.py) (+1) |
| `xtfrontier` | frontier::sfa | R 4.5.2; frontier 1.1.8 | rel_est<=0.001, rel_se<=0.05 | 2.8e-06 / 1.8e-06 | [`29_panel_sfa.py`](../tests/r_parity/29_panel_sfa.py) (+2) |
| `zinb` | pscl::zeroinfl(dist="negbin") | R 4.5.2; pscl 1.5.9 | rel_est<=1e-05, rel_se<=0.001 | 1.1e-06 / 2.1e-07 | [`64_zinb.py`](../tests/r_parity/64_zinb.py) (+2) |

## external-replication — 4 functions

Reproduces published-paper numbers; sources in `tests/external_parity/PUBLISHED_REFERENCE_VALUES.md`.

| function | test |
| --- | --- |
| `aggte` | [`test_honest_did_paper_parity.py`](../tests/external_parity/test_honest_did_paper_parity.py) (+1) |
| `breakdown_m` | [`test_honest_did_paper_parity.py`](../tests/external_parity/test_honest_did_paper_parity.py) |
| `g_estimation` | [`test_whatif_nhefs.py`](../tests/external_parity/test_whatif_nhefs.py) |
| `parallel_trends_robustness` | [`test_rebel_canal_published.py`](../tests/external_parity/test_rebel_canal_published.py) |

## analytical-only — 210 functions

Recovers a known DGP truth / closed-form identity within tolerance; no cross-package reference. See `tests/reference_parity/REFERENCES.md`.

| function | test |
| --- | --- |
| `W` | [`test_spdep_parity.py`](../tests/reference_parity/test_spdep_parity.py) |
| `ackerberg_caves_frazer` | [`test_structural_parity.py`](../tests/reference_parity/test_structural_parity.py) |
| `aggte_from_influence` | [`test_aggte_r_did_parity.py`](../tests/reference_parity/test_aggte_r_did_parity.py) |
| `aipw` | [`test_paper_parity.py`](../tests/reference_parity/test_paper_parity.py) (+1) |
| `always_treat` | [`test_longitudinal_parity.py`](../tests/reference_parity/test_longitudinal_parity.py) |
| `assimilative_causal` | [`test_assimilation_parity.py`](../tests/reference_parity/test_assimilation_parity.py) |
| `attrition_test` | [`test_attrition_test_parity.py`](../tests/reference_parity/test_attrition_test_parity.py) |
| `auc` | [`test_auc_parity.py`](../tests/reference_parity/test_auc_parity.py) |
| `auto_cate` | [`test_ml_causal_recovery_parity.py`](../tests/reference_parity/test_ml_causal_recovery_parity.py) |
| `auto_cate_tuned` | [`test_ml_causal_recovery_parity_round2.py`](../tests/reference_parity/test_ml_causal_recovery_parity_round2.py) |
| `average_treatment_effect` | [`test_forest_ate_parity.py`](../tests/reference_parity/test_forest_ate_parity.py) |
| `bayes_iv` | [`test_bayes_diagnostics_parity.py`](../tests/reference_parity/test_bayes_diagnostics_parity.py) |
| `bayes_rd` | [`test_bayes_diagnostics_parity.py`](../tests/reference_parity/test_bayes_diagnostics_parity.py) |
| `bayes_synth` | [`test_bayes_synth_parity.py`](../tests/reference_parity/test_bayes_synth_parity.py) |
| `bcf` | [`test_bcf_parity.py`](../tests/reference_parity/test_bcf_parity.py) |
| `bcf_factor_exposure` | [`test_bcf_factor_exposure_parity.py`](../tests/reference_parity/test_bcf_factor_exposure_parity.py) |
| `beyond_average_late` | [`test_beyond_average_late_parity.py`](../tests/reference_parity/test_beyond_average_late_parity.py) (+1) |
| `bidirectional_pci` | [`test_proximal_parity.py`](../tests/reference_parity/test_proximal_parity.py) |
| `blp` | [`test_structural_parity.py`](../tests/reference_parity/test_structural_parity.py) |
| `bootstrap` | [`test_bootstrap_parity.py`](../tests/reference_parity/test_bootstrap_parity.py) |
| `bradford_hill` | [`test_bradford_hill_parity.py`](../tests/reference_parity/test_bradford_hill_parity.py) |
| `breakdown_frontier` | [`test_breakdown_frontier_parity.py`](../tests/reference_parity/test_breakdown_frontier_parity.py) |
| `breslow_day_test` | [`test_breslow_day_parity.py`](../tests/reference_parity/test_breslow_day_parity.py) |
| `bunching` | [`test_bunching_parity.py`](../tests/reference_parity/test_bunching_parity.py) |
| `bvar` | [`test_timeseries_parity.py`](../tests/reference_parity/test_timeseries_parity.py) |
| `calibration_test` | [`test_calibration_test_parity.py`](../tests/reference_parity/test_calibration_test_parity.py) |
| `cardinality_match` | [`test_cardinality_match_parity.py`](../tests/reference_parity/test_cardinality_match_parity.py) |
| `cate_eval` | [`test_ml_causal_recovery_parity.py`](../tests/reference_parity/test_ml_causal_recovery_parity.py) |
| `causal_kalman` | [`test_assimilation_parity.py`](../tests/reference_parity/test_assimilation_parity.py) |
| `causal_policy_forest` | [`test_ope_parity.py`](../tests/reference_parity/test_ope_parity.py) |
| `check_absorbing` | [`test_absorbing_reference.py`](../tests/reference_parity/test_absorbing_reference.py) |
| `clone_censor_weight` | [`test_target_trial_parity.py`](../tests/reference_parity/test_target_trial_parity.py) |
| `cluster_cate` | [`test_ml_causal_recovery_parity_round2.py`](../tests/reference_parity/test_ml_causal_recovery_parity_round2.py) |
| `cluster_cross_interference` | [`test_cluster_cross_interference_parity.py`](../tests/reference_parity/test_cluster_cross_interference_parity.py) |
| `cluster_robust_se` | [`test_cluster_robust_se_parity.py`](../tests/reference_parity/test_cluster_robust_se_parity.py) |
| `conformal_cate` | [`test_conformal_causal_parity.py`](../tests/reference_parity/test_conformal_causal_parity.py) |
| `conformal_fair_ite` | [`test_conformal_fair_ite_parity.py`](../tests/reference_parity/test_conformal_fair_ite_parity.py) |
| `conformal_ite` | [`test_conformal_ite_parity.py`](../tests/reference_parity/test_conformal_ite_parity.py) |
| `conformal_ite_interval` | [`test_conformal_causal_parity.py`](../tests/reference_parity/test_conformal_causal_parity.py) |
| `conley` | [`test_conley_acreg_spacetime_parity.py`](../tests/reference_parity/test_conley_acreg_spacetime_parity.py) (+2) |
| `continuous_did` | [`test_dose_response_parity.py`](../tests/reference_parity/test_dose_response_parity.py) |
| `continuous_iv_late` | [`test_continuous_iv_late_parity.py`](../tests/reference_parity/test_continuous_iv_late_parity.py) |
| `contrast` | [`test_contrast_pwcompare_parity.py`](../tests/reference_parity/test_contrast_pwcompare_parity.py) |
| `counterfactual_fairness` | [`test_fairness_parity.py`](../tests/reference_parity/test_fairness_parity.py) |
| `cox_frailty` | [`test_competing_risks_parity.py`](../tests/reference_parity/test_competing_risks_parity.py) |
| `cr3_jackknife_vcov` | [`test_recovery_batch2_parity.py`](../tests/reference_parity/test_recovery_batch2_parity.py) |
| `cuminc` | [`test_competing_risks_parity.py`](../tests/reference_parity/test_competing_risks_parity.py) |
| `cusum_test` | [`test_cusum_test_parity.py`](../tests/reference_parity/test_cusum_test_parity.py) |
| `demographic_parity` | [`test_fairness_parity.py`](../tests/reference_parity/test_fairness_parity.py) |
| `did` | [`test_cs_weighted_parity.py`](../tests/reference_parity/test_cs_weighted_parity.py) (+2) |
| `did_balance` | [`test_did_balance_parity.py`](../tests/reference_parity/test_did_balance_parity.py) |
| `did_had` | [`test_did_had_parity.py`](../tests/reference_parity/test_did_had_parity.py) |
| `direct_standardize` | [`test_standardize_parity.py`](../tests/reference_parity/test_standardize_parity.py) |
| `discos` | [`test_distributional_te_parity.py`](../tests/reference_parity/test_distributional_te_parity.py) |
| `dist_iv` | [`test_dist_iv_parity.py`](../tests/reference_parity/test_dist_iv_parity.py) |
| `distributional_did` | [`test_functional_form_extended_parity.py`](../tests/reference_parity/test_functional_form_extended_parity.py) |
| `distributional_te` | [`test_distributional_te_inference.py`](../tests/reference_parity/test_distributional_te_inference.py) (+1) |
| `dml_panel` | [`test_ml_causal_recovery_parity.py`](../tests/reference_parity/test_ml_causal_recovery_parity.py) |
| `dose_response` | [`test_dose_response_parity.py`](../tests/reference_parity/test_dose_response_parity.py) |
| `effective_f_test` | [`test_anderson_rubin_parity.py`](../tests/reference_parity/test_anderson_rubin_parity.py) (+1) |
| `engle_granger` | [`test_engle_granger_parity.py`](../tests/reference_parity/test_engle_granger_parity.py) |
| `equalized_odds` | [`test_fairness_parity.py`](../tests/reference_parity/test_fairness_parity.py) |
| `evidence_without_injustice` | [`test_fairness_parity.py`](../tests/reference_parity/test_fairness_parity.py) |
| `fairlie` | [`test_decomposition_family_parity.py`](../tests/reference_parity/test_decomposition_family_parity.py) |
| `fairness_audit` | [`test_fairness_parity.py`](../tests/reference_parity/test_fairness_parity.py) |
| `fci` | [`test_fci_parity.py`](../tests/reference_parity/test_fci_parity.py) |
| `finegray` | [`test_competing_risks_parity.py`](../tests/reference_parity/test_competing_risks_parity.py) |
| `fisher_exact` | [`test_fisher_exact_parity.py`](../tests/reference_parity/test_fisher_exact_parity.py) |
| `focal_cate` | [`test_ml_causal_recovery_parity_round2.py`](../tests/reference_parity/test_ml_causal_recovery_parity_round2.py) |
| `fortified_pci` | [`test_proximal_parity.py`](../tests/reference_parity/test_proximal_parity.py) |
| `four_way_decomposition` | [`test_four_way_decomposition_parity.py`](../tests/reference_parity/test_four_way_decomposition_parity.py) |
| `front_door` | [`test_front_door_parity.py`](../tests/reference_parity/test_front_door_parity.py) |
| `frontdoor` | [`test_frontdoor_parity.py`](../tests/reference_parity/test_frontdoor_parity.py) |
| `garch` | [`test_timeseries_parity.py`](../tests/reference_parity/test_timeseries_parity.py) |
| `gelbach` | [`test_gelbach_parity.py`](../tests/reference_parity/test_gelbach_parity.py) |
| `general_bunching` | [`test_bunching_parity.py`](../tests/reference_parity/test_bunching_parity.py) |
| `geolift` | [`test_geolift_parity.py`](../tests/reference_parity/test_geolift_parity.py) |
| `ges` | [`test_ges_parity.py`](../tests/reference_parity/test_ges_parity.py) |
| `gformula_ice_fn` | [`test_gformula_family_parity.py`](../tests/reference_parity/test_gformula_family_parity.py) |
| `gformula_mc` | [`test_gformula_family_parity.py`](../tests/reference_parity/test_gformula_family_parity.py) |
| `gmm` | [`test_general_gmm_parity.py`](../tests/reference_parity/test_general_gmm_parity.py) (+1) |
| `granger_causality` | [`test_timeseries_parity.py`](../tests/reference_parity/test_timeseries_parity.py) |
| `grapple` | [`test_grapple_parity.py`](../tests/reference_parity/test_grapple_parity.py) |
| `hal_tmle` | [`test_ml_causal_recovery_parity_round2.py`](../tests/reference_parity/test_ml_causal_recovery_parity_round2.py) |
| `hausman_test` | [`test_diag_recovery_parity.py`](../tests/reference_parity/test_diag_recovery_parity.py) |
| `honest_variance` | [`test_forest_rate_honest_parity.py`](../tests/reference_parity/test_forest_rate_honest_parity.py) |
| `horowitz_manski` | [`test_horowitz_manski_parity.py`](../tests/reference_parity/test_horowitz_manski_parity.py) |
| `icc` | [`test_icc_parity.py`](../tests/reference_parity/test_icc_parity.py) |
| `immortal_time_check` | [`test_target_trial_parity.py`](../tests/reference_parity/test_target_trial_parity.py) |
| `indirect_standardize` | [`test_standardize_parity.py`](../tests/reference_parity/test_standardize_parity.py) |
| `influence_functions` | [`test_aggte_r_did_parity.py`](../tests/reference_parity/test_aggte_r_did_parity.py) |
| `interactive_fe` | [`test_panel_estimators_parity.py`](../tests/reference_parity/test_panel_estimators_parity.py) |
| `interference` | [`test_interference_parity.py`](../tests/reference_parity/test_interference_parity.py) |
| `ipcw` | [`test_ipcw_parity.py`](../tests/reference_parity/test_ipcw_parity.py) |
| `irf` | [`test_timeseries_parity.py`](../tests/reference_parity/test_timeseries_parity.py) |
| `its` | [`test_timeseries_parity.py`](../tests/reference_parity/test_timeseries_parity.py) |
| `iv_diag` | [`test_diag_recovery_parity.py`](../tests/reference_parity/test_diag_recovery_parity.py) (+1) |
| `ivqreg` | [`test_ivqreg_parity.py`](../tests/reference_parity/test_ivqreg_parity.py) |
| `jackknife_se` | [`test_recovery_batch2_parity.py`](../tests/reference_parity/test_recovery_batch2_parity.py) |
| `jive` | [`test_jive_parity.py`](../tests/reference_parity/test_jive_parity.py) |
| `johansen` | [`test_timeseries_parity.py`](../tests/reference_parity/test_timeseries_parity.py) |
| `kan_dlate` | [`test_dist_iv_parity.py`](../tests/reference_parity/test_dist_iv_parity.py) |
| `kdensity` | [`test_kdensity_parity.py`](../tests/reference_parity/test_kdensity_parity.py) |
| `lasso_iv` | [`test_lasso_iv_parity.py`](../tests/reference_parity/test_lasso_iv_parity.py) |
| `lasso_select` | [`test_lasso_select_parity.py`](../tests/reference_parity/test_lasso_select_parity.py) |
| `lee_bounds` | [`test_lee_bounds_parity.py`](../tests/reference_parity/test_lee_bounds_parity.py) |
| `levinsohn_petrin` | [`test_structural_parity.py`](../tests/reference_parity/test_structural_parity.py) |
| `lincom` | [`test_postestimation_parity.py`](../tests/reference_parity/test_postestimation_parity.py) |
| `lingam` | [`test_causal_discovery_parity.py`](../tests/reference_parity/test_causal_discovery_parity.py) |
| `long_term_from_short` | [`test_surrogate_parity.py`](../tests/reference_parity/test_surrogate_parity.py) |
| `longitudinal_analyze` | [`test_longitudinal_parity.py`](../tests/reference_parity/test_longitudinal_parity.py) |
| `longitudinal_contrast` | [`test_longitudinal_parity.py`](../tests/reference_parity/test_longitudinal_parity.py) |
| `lpbwselect_mse_dpi` | [`test_did_had_parity.py`](../tests/reference_parity/test_did_had_parity.py) (+1) |
| `lprobust_at_point` | [`test_lprobust_parity.py`](../tests/reference_parity/test_lprobust_parity.py) |
| `lrtest` | [`test_lrtest_parity.py`](../tests/reference_parity/test_lrtest_parity.py) |
| `ltmle` | [`test_ml_causal_recovery_parity_round2.py`](../tests/reference_parity/test_ml_causal_recovery_parity_round2.py) |
| `ltmle_survival` | [`test_ml_causal_recovery_parity_round2.py`](../tests/reference_parity/test_ml_causal_recovery_parity_round2.py) |
| `malmquist` | [`test_frontier_efficiency_parity.py`](../tests/reference_parity/test_frontier_efficiency_parity.py) |
| `manski_bounds` | [`test_manski_bounds_parity.py`](../tests/reference_parity/test_manski_bounds_parity.py) |
| `margins` | [`test_postestimation_parity.py`](../tests/reference_parity/test_postestimation_parity.py) |
| `margins_at` | [`test_margins_at_parity.py`](../tests/reference_parity/test_margins_at_parity.py) |
| `markup` | [`test_structural_parity.py`](../tests/reference_parity/test_structural_parity.py) |
| `matrix_completion` | [`test_matrix_completion_parity.py`](../tests/reference_parity/test_matrix_completion_parity.py) |
| `mc_panel` | [`test_matrix_completion_parity.py`](../tests/reference_parity/test_matrix_completion_parity.py) |
| `mediate_interventional` | [`test_mediate_interventional_parity.py`](../tests/reference_parity/test_mediate_interventional_parity.py) |
| `mediation_decompose` | [`test_mediation_decompose_parity.py`](../tests/reference_parity/test_mediation_decompose_parity.py) |
| `metafrontier` | [`test_frontier_efficiency_parity.py`](../tests/reference_parity/test_frontier_efficiency_parity.py) |
| `mi_estimate` | [`test_imputation_parity.py`](../tests/reference_parity/test_imputation_parity.py) |
| `mice` | [`test_imputation_parity.py`](../tests/reference_parity/test_imputation_parity.py) |
| `model_averaging_dml` | [`test_ml_causal_recovery_parity.py`](../tests/reference_parity/test_ml_causal_recovery_parity.py) |
| `mr_lap` | [`test_mr_lap_parity.py`](../tests/reference_parity/test_mr_lap_parity.py) |
| `mr_mediation` | [`test_mr_mediation_parity.py`](../tests/reference_parity/test_mr_mediation_parity.py) |
| `msm` | [`test_msm_family_parity.py`](../tests/reference_parity/test_msm_family_parity.py) |
| `multi_treatment` | [`test_multi_treatment_parity.py`](../tests/reference_parity/test_multi_treatment_parity.py) |
| `network_exposure` | [`test_interference_parity.py`](../tests/reference_parity/test_interference_parity.py) |
| `network_graph` | [`test_network_parity.py`](../tests/reference_parity/test_network_parity.py) |
| `never_treat` | [`test_longitudinal_parity.py`](../tests/reference_parity/test_longitudinal_parity.py) |
| `notch` | [`test_notch_parity.py`](../tests/reference_parity/test_notch_parity.py) |
| `notears` | [`test_causal_discovery_parity.py`](../tests/reference_parity/test_causal_discovery_parity.py) |
| `olley_pakes` | [`test_structural_parity.py`](../tests/reference_parity/test_structural_parity.py) |
| `orthogonal_to_bias` | [`test_fairness_parity.py`](../tests/reference_parity/test_fairness_parity.py) |
| `oster_delta` | [`test_oster_delta_parity.py`](../tests/reference_parity/test_oster_delta_parity.py) |
| `panel_unitroot` | [`test_timeseries_parity.py`](../tests/reference_parity/test_timeseries_parity.py) |
| `particle_filter` | [`test_assimilation_parity.py`](../tests/reference_parity/test_assimilation_parity.py) |
| `pate` | [`test_pate_parity.py`](../tests/reference_parity/test_pate_parity.py) |
| `pc_algorithm` | [`test_causal_discovery_parity.py`](../tests/reference_parity/test_causal_discovery_parity.py) |
| `peer_effects` | [`test_peer_effects_parity.py`](../tests/reference_parity/test_peer_effects_parity.py) |
| `policy_value` | [`test_policy_tree_parity.py`](../tests/reference_parity/test_policy_tree_parity.py) (+1) |
| `policy_weight_ate` | [`test_policy_weight_parity.py`](../tests/reference_parity/test_policy_weight_parity.py) |
| `policy_weight_marginal` | [`test_policy_weight_parity.py`](../tests/reference_parity/test_policy_weight_parity.py) |
| `policy_weight_observed_prte` | [`test_policy_weight_parity.py`](../tests/reference_parity/test_policy_weight_parity.py) |
| `policy_weight_subsidy` | [`test_policy_weight_parity.py`](../tests/reference_parity/test_policy_weight_parity.py) |
| `power_case_control` | [`test_epi_diag_parity.py`](../tests/reference_parity/test_epi_diag_parity.py) |
| `power_ols` | [`test_recovery_batch_parity.py`](../tests/reference_parity/test_recovery_batch_parity.py) |
| `pretrends_equivalence` | [`test_fect_equivalence_parity.py`](../tests/reference_parity/test_fect_equivalence_parity.py) |
| `principal_strat` | [`test_principal_strat_parity.py`](../tests/reference_parity/test_principal_strat_parity.py) |
| `prod_fn` | [`test_structural_parity.py`](../tests/reference_parity/test_structural_parity.py) |
| `proximal` | [`test_proximal_parity.py`](../tests/reference_parity/test_proximal_parity.py) |
| `proximal_surrogate_index` | [`test_surrogate_parity.py`](../tests/reference_parity/test_surrogate_parity.py) |
| `pwcompare` | [`test_contrast_pwcompare_parity.py`](../tests/reference_parity/test_contrast_pwcompare_parity.py) |
| `qte_hd_panel` | [`test_hd_panel_qte.py`](../tests/reference_parity/test_hd_panel_qte.py) |
| `quasi_untreated_test` | [`test_did_had_parity.py`](../tests/reference_parity/test_did_had_parity.py) |
| `rate` | [`test_forest_rate_honest_parity.py`](../tests/reference_parity/test_forest_rate_honest_parity.py) |
| `regime` | [`test_longitudinal_parity.py`](../tests/reference_parity/test_longitudinal_parity.py) |
| `ri_test` | [`test_recovery_batch2_parity.py`](../tests/reference_parity/test_recovery_batch2_parity.py) (+1) |
| `rlasso_effects` | [`test_rlasso_parity.py`](../tests/reference_parity/test_rlasso_parity.py) |
| `roc_curve` | [`test_auc_parity.py`](../tests/reference_parity/test_auc_parity.py) |
| `romano_wolf` | [`test_romano_wolf_parity.py`](../tests/reference_parity/test_romano_wolf_parity.py) |
| `sarar_gmm` | [`test_spatial_models_parity.py`](../tests/reference_parity/test_spatial_models_parity.py) |
| `selection_bounds` | [`test_selection_bounds_parity.py`](../tests/reference_parity/test_selection_bounds_parity.py) |
| `sensitivity_specificity` | [`test_epi_diag_parity.py`](../tests/reference_parity/test_epi_diag_parity.py) |
| `shapley_inequality` | [`test_decomposition_family_parity.py`](../tests/reference_parity/test_decomposition_family_parity.py) |
| `sharp_ope_unobserved` | [`test_ope_parity.py`](../tests/reference_parity/test_ope_parity.py) |
| `source_decompose` | [`test_source_decompose_parity.py`](../tests/reference_parity/test_source_decompose_parity.py) |
| `spatial_did` | [`test_spatial_models_parity.py`](../tests/reference_parity/test_spatial_models_parity.py) (+1) |
| `spatial_iv` | [`test_spatial_models_parity.py`](../tests/reference_parity/test_spatial_models_parity.py) |
| `spatial_panel` | [`test_spatial_models_parity.py`](../tests/reference_parity/test_spatial_models_parity.py) |
| `spillover` | [`test_interference_parity.py`](../tests/reference_parity/test_interference_parity.py) |
| `spillover_did` | [`test_spillover_rings.py`](../tests/reference_parity/test_spillover_rings.py) |
| `ssaggregate` | [`test_bartik_ssagg_parity.py`](../tests/reference_parity/test_bartik_ssagg_parity.py) |
| `stabilized_weights` | [`test_stabilized_weights_parity.py`](../tests/reference_parity/test_stabilized_weights_parity.py) |
| `staggered_cs` | [`test_staggered_extended_parity.py`](../tests/reference_parity/test_staggered_extended_parity.py) |
| `staggered_sa` | [`test_staggered_extended_parity.py`](../tests/reference_parity/test_staggered_extended_parity.py) |
| `stepwise` | [`test_stepwise_parity.py`](../tests/reference_parity/test_stepwise_parity.py) |
| `stochastic_dominance` | [`test_distributional_te_parity.py`](../tests/reference_parity/test_distributional_te_parity.py) |
| `structural_break` | [`test_structural_break_parity.py`](../tests/reference_parity/test_structural_break_parity.py) |
| `subcluster_wild_bootstrap` | [`test_wcb_recovery_parity.py`](../tests/reference_parity/test_wcb_recovery_parity.py) |
| `subgroup_decompose` | [`test_subgroup_decompose_parity.py`](../tests/reference_parity/test_subgroup_decompose_parity.py) |
| `super_learner` | [`test_ml_causal_recovery_parity.py`](../tests/reference_parity/test_ml_causal_recovery_parity.py) |
| `surrogate_index` | [`test_surrogate_parity.py`](../tests/reference_parity/test_surrogate_parity.py) |
| `survivor_average_causal_effect` | [`test_principal_strat_parity.py`](../tests/reference_parity/test_principal_strat_parity.py) |
| `svydesign` | [`test_survey_parity.py`](../tests/reference_parity/test_survey_parity.py) |
| `synthdid_estimate` | [`test_texas_synth_parity.py`](../tests/reference_parity/test_texas_synth_parity.py) |
| `tF_critical_value` | [`test_tf_critical_value_parity.py`](../tests/reference_parity/test_tf_critical_value_parity.py) |
| `target_trial_checklist` | [`test_target_trial_parity.py`](../tests/reference_parity/test_target_trial_parity.py) |
| `target_trial_emulate` | [`test_target_trial_parity.py`](../tests/reference_parity/test_target_trial_parity.py) |
| `target_trial_protocol` | [`test_target_trial_parity.py`](../tests/reference_parity/test_target_trial_parity.py) |
| `target_trial_report` | [`test_target_trial_parity.py`](../tests/reference_parity/test_target_trial_parity.py) |
| `test` | [`test_postestimation_parity.py`](../tests/reference_parity/test_postestimation_parity.py) |
| `test_calibration` | [`test_calibration_test_parity.py`](../tests/reference_parity/test_calibration_test_parity.py) |
| `translog_design` | [`test_translog_design_parity.py`](../tests/reference_parity/test_translog_design_parity.py) |
| `transport_generalize` | [`test_transport_parity.py`](../tests/reference_parity/test_transport_parity.py) |
| `weighted_conformal_prediction` | [`test_conformal_causal_parity.py`](../tests/reference_parity/test_conformal_causal_parity.py) |
| `wild_cluster_bootstrap` | [`test_wcb_recovery_parity.py`](../tests/reference_parity/test_wcb_recovery_parity.py) (+1) |
| `wild_cluster_ci_inv` | [`test_wild_cluster_ci_inv_parity.py`](../tests/reference_parity/test_wild_cluster_ci_inv_parity.py) |
| `wooldridge_prod` | [`test_structural_parity.py`](../tests/reference_parity/test_structural_parity.py) |
| `xlearner` | [`test_ml_causal_recovery_parity.py`](../tests/reference_parity/test_ml_causal_recovery_parity.py) |
| `xtdpdsys` | [`test_dynpanel_abdata_parity.py`](../tests/reference_parity/test_dynpanel_abdata_parity.py) |
| `xtlsdvc` | [`test_lsdvc_parity.py`](../tests/reference_parity/test_lsdvc_parity.py) |
| `yatchew_linearity_test` | [`test_did_had_parity.py`](../tests/reference_parity/test_did_had_parity.py) |

## unverified — 753 functions

These are registered public functions with no cross-language or published-reference parity evidence attached **yet**. This is the honest coverage gap, not a claim of incorrectness — many are frontier methods with no Stata/R sibling to align against. Query any of them with `sp.parity_status(name)`; the closing roadmap lives in [`docs/dev/parity_status_roadmap.md`](dev/parity_status_roadmap.md).
