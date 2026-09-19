<a id="parity-campaign-phase3"></a>

## Unreleased — ⚠️ Cross-language campaign phase 3: 125 defects fixed against R / Stata references

**Who is affected.** Anyone who reported numbers from the functions below.
Phase 3 compared survival / epidemiology, time series, inference and
sensitivity, panel and GLMM, treatment effects, DiD and synthetic control,
RD and IV, and spatial / survey / structural estimators with their R or
Stata reference on the same data. Where a row below changes a default,
rerun the analysis; most old numbers were not a documented quantity and
cannot be reproduced. Where an option restores the old number, the last
column names it. "—" means no route back is offered.

### Survival and epidemiology

| Function | What changes | Old → new (example) | Old number, if you need it |
| --- | --- | --- | --- |
| `sp.cuminc` | delta-method SE / CI (`F(t) - F(t_j)`); Gray's test | cause-1 SE at t = 2.0: 0.027944 → 0.030489; Gray chi2 2.3950 → 1.8499 | — (old values were wrong) |
| `sp.finegray` | censoring weights at left limits; default SE is Fine–Gray's sandwich | coef 0.440740 → 0.440603; SE 0.084511 → 0.081090 | `vce="model"` for the old SE; old weights not kept |
| `sp.cox_frailty` | all outputs; `.theta` is now the frailty variance | beta 0.480666 → 0.528795; theta 0.500004 (precision) → 0.240201 (variance) | — (old fit ignored the frailty); for a precision use `1/res.theta` |
| `sp.roc_curve` / `sp.auc` | AUC with tied scores; `thresholds` / `tpr` / `fpr` have one entry per distinct score | 0.68376068 → 0.68389966 | — |
| `sp.kdensity` | default width is R `bw.nrd0`; `"sheather-jones"` is Sheather–Jones | 0.522176 → 0.525683; SJ 0.5222 → 0.5592 | `bw_method="stata"` gives Stata's 1.349 rule with Stata's percentiles |
| `sp.cox(ties="breslow")` | runs Breslow | x1 0.355720 → 0.335603 | `ties="efron"` is what it used to run |

### Time series

| Function | What changes | Old → new (example) | Old number, if you need it |
| --- | --- | --- | --- |
| `sp.panel_unitroot` | LLC / IPS / Fisher are the named tests (Stata `xtunitroot` by default) | LLC −7.80 → −4.656; Fisher P 239.1 → 117.09 | — (old numbers were not these tests) |
| `sp.engle_granger` | step-2 ADF without a constant; `trend` acts in step 1; MacKinnon (2010) critical values | Z −9.6167 → −9.6364; 5% CV −3.34 → −3.3608 | — (old statistic had no valid critical values) |
| `sp.johansen` | critical values (max-eig k−r = 3, 4; separate 'n' / 'ct' tables) | max-eig 5% k−r=3: 21.12 → 20.97 | statistics unchanged |
| `sp.bvar` | Minnesota prior per equation; no column-order dependence | gdp own-lag 0.901 → 0.6382 | — |
| `sp.garch` | pre-sample `mean(eps²)`; converged optimiser; `forecast()` uses all lags | log L −2144.43397943 → −2144.43398728 | — (differences ~1e-5) |
| `sp.cusum_test` | exact boundary coefficient | 0.948 → 0.9478982 | n/a (1e-4) |
| `sp.structural_break` | sup-F grid includes strucchange's last point | changes only when the maximum is at that point | — |

### Inference and sensitivity

| Function | What changes | Old → new (example) | Old number, if you need it |
| --- | --- | --- | --- |
| `sp.wild_cluster_bootstrap`, `sp.wild_cluster_boot`, `sp.subcluster_wild_bootstrap`, `sp.panel` feols `wild` | exact (enumerated) p with strict `>` when `2^G <= n_boot` | p 0.92569 → 0.9306640625 (G = 12) | — (MC estimate of the same quantity; `>=` was wrong) |
| `sp.wild_cluster_ci_inv` | endpoints are the jumps of p(h0) | lower −1.114049 → −1.121498336078 | — |
| `sp.fisher_exact`, `sp.ri_test` | exact p on designs with at most `n_perm(s)` assignments | 0.1443 → 132/924 | pass `n_perm(s)` below the number of assignments |
| `sp.rosenbaum_bounds` / `sp.rosenbaum_gamma` | no continuity correction; zeros ranked; correct tail | Γ = 2 upper bound 0.1077 → 0.149128 | `zero_method="wilcox"` reproduces the zero handling only |
| `sp.oster_bounds` | exact solution from data; corrected approximation from summaries | δ* 1.20492 → 0.904199433 | — (wrong formula) |
| `sp.oster_delta` | default `r_max=1.3` means min(1, 1.3 R_full); exact solution | δ* 0.16189 → 0.90420 | pass an explicit `r_max` in (R_full, 1] |
| `sp.pate(method="aipw")` | per-arm Hájek augmentation | bias −0.168 → −0.001 (simulation) | — |

### Panel and GLMM

| Function | What changes | Old → new (example) | Old number, if you need it |
| --- | --- | --- | --- |
| `sp.xtnbreg(model="fe")` | Hausman–Hall–Griliches conditional FE (Stata `xtnbreg, fe`) | z1 0.418 → 0.3783114 | `model="ufe"` |
| `sp.xtnbreg(model="re")` | beta-dispersion RE (Stata `xtnbreg, re`) | — | `model="normal_re"` |
| `sp.menbreg` / `sp.megamma` / `sp.meologit` | observed-curvature Laplace / AGHQ | gamma `_cons` 0.6196 → 0.6372 | `curvature="expected"` |
| `sp.meglm(family="gaussian")` | σ² estimated (was fixed at 1) | SE(x1) 0.0428 → 0.0335 (Stata) | — (old was not a Gaussian mixed model) |
| GLMMs, `sp.mixed` | optimum finished with Newton steps | `mepoisson` `_cons` 0.3451384 → 0.3450937699 | — |
| `sp.meologit` | SEs from the full observed information; threshold SEs | — | — |
| `sp.icc` | Stata `estat icc` SE / CI; latent ICC for `melogit` / `meologit` (was NaN) | mixed ML SE heuristic → 0.0533785 | — |
| `MixedResult.aic` / `.bic` | residual variance counted once | AIC was 2 too high, BIC log n too high | — |
| `sp.interactive_fe` | Bai's SE (M_Λ X M_F), CR1 cluster factor, `tol` 1e-10 | xa SE 0.03194 → 0.03304 | — |
| `sp.gmm` (nonlinear) | Gauss–Newton finish | estimates move ~1e-8 | — |
| `sp.mixlogit` | positive SDs; no ε perturbation; robust × N/(N−1) | — | `small_sample=False` for the robust factor |

### Treatment effects

| Function | What changes | Old → new (example) | Old number, if you need it |
| --- | --- | --- | --- |
| `sp.ltmle` | R `ltmle`'s algorithm; `propensity_bounds` bounds the cumulative g, default `(0.01, 1.0)` | binary ATE 0.757 → 0.373 | — (invalid) |
| `sp.gformula_ice_fn(bootstrap=0)` | SE is the M-estimation sandwich | 0.0591 → 0.0683 | — (not an SE) |
| `sp.aipw(estimand="ATT")` | SE includes the ratio term | 0.1018 → 0.0899 | — |
| `sp.principal_strat` | "Complier (LATE)" is the Wald LATE | 6.395 → 7.355 | E[Y(1)\|c] = `(mu_11 p11 − mu_01 p01)/pi_c` |
| `sp.survivor_average_causal_effect` | bounds keep rows with missing outcomes | (1.684, 1.684) → (0.725, 2.612) | — |
| `sp.lee_bounds` | Lee's sample-quantile trimming | (0.7176, 2.6187) → (0.7245, 2.6120) | —; `trimming="exact"` for fractional trimming |
| `sp.manski_bounds(assumption="mts")` | MTS alone | lower bound 0 → −0.336 | `assumption="mts_mtr"` |
| `sp.horowitz_manski` | strata missing an arm kept | [−0.332, 0.629] → [−0.337, 0.663] | — |
| `sp.mediate_interventional(tv_confounders=)` | D→L→Y path included | IDE 0.574 → 1.106 | — |
| `sp.multi_treatment` | unpenalised mlogit GPS; bootstrap refits the same outcome model | ATE 1.163538 → 1.163558 | — |
| `sp.g_estimation` | logistic propensity by default | psi 1.0564 → 1.0552 | `propensity_model="linear"` |
| `sp.ipcw` | censored rows weighted 0; summaries over uncensored rows | censored weights ≥ 0.77 → 0 | none needed (observed-row weights unchanged) |

### DiD and synthetic control

| Function | What changes | Old → new (example) | Old number, if you need it |
| --- | --- | --- | --- |
| `sp.sdid`, `sp.synthdid_estimate`, `sp.sc_estimate`, `sp.did_estimate` | placebo / bootstrap / jackknife SE follow `synthdid::vcov`; undefined SEs are NaN | Prop. 99 placebo SE (seed 42) 2.6041 → 2.5630; five-treated jackknife 0.1248 → 0.4382 | — (old SEs were not a documented quantity) |
| `sp.breakdown_m` | inverts the FLCI; honours `method` | mpdta CS e=0 0.01925 → 0.00799 | the closed form is still returned, with a warning, when no covariance is available |
| `sp.honest_did` (native smoothness) | worst-case bias fixed; no rounding | mpdta CS e=1 M=0.02 lower −0.12136 → −0.10740 | — |
| `sp.continuous_did(method="twfe")` | SE degrees of freedom; unbalanced-panel slope | iid SE 0.042912 → 0.047212; unbalanced slope 0.403496 → 0.402988 | — |
| `sp.continuous_did(method="att_gt")` | bootstrap with multiplicity; bins bootstrapped jointly | pooled SE 0.10639 → 0.18500 | — (old bootstrap was a subsample) |
| `sp.continuous_did(method="dose_response")` | SE is a bootstrap of the average slope | 0.24308 → 0.15439 | `mean(model_info["dose_response_pointwise_se"])` |
| `sp.did_timevarying_covariates` | outcome-regression ATT(g,t); group aggregation | 1.273390 → 1.257705 | —; `aggregation="simple"` gives the treated-count-weighted aggregate of the corrected cells |
| `sp.mc_panel`, `sp.matrix_completion` | `fixed_effects="two-way"`; `tol` 1e-5 → 1e-10, `max_iter` 1000 → 5000 | ATT at lambda 8: 2.934 → 2.385 | `fixed_effects="none", tol=1e-5, max_iter=1000` |
| `sp.mc_synth`, `sp.synth(method="mc")` | `fixed_effects="two-way"`; `tol` 1e-6 → 1e-10, `max_iter` 500 → 5000; CV grid from the centred matrix | `mc_synth(seed=0)` 1.66626 → 1.32783 | `fixed_effects="none", tol=1e-6, max_iter=500` |
| `sp.spillover_did` (staggered only) | ring effects by exposure onset; SE adds the weight term | ring 1 0.7912 → 1.2376; direct SE 0.0779 → 0.0914 | — |
| `sp.harvest_did` | own cohort excluded from placebo controls; joint-covariance inference; IF cell SE | headline SE 0.0621 → 0.1015; cohort-5 e=−4 −0.3550 → −0.5104 | — |
| `sp.causal_impact` | SE uses correlated forecast errors | 0.2348 → 0.2852 | `sqrt(mean(detail.predicted_se[post]**2)/n_post)` |
| `sp.scpi` | intervals from R `scpi`'s procedure; `se` / `pvalue` NaN; `period_results` loses `in_sample_var` / `out_sample_var` | Germany 1991 [0.230, 0.774] → [−0.731, 1.316] | — |
| `sp.scest(w_constr="lasso")` | L1 ball ‖w‖₁ ≤ 1 | Germany max \|Δw\| 0.227 | — |
| `sp.scest(w_constr="ridge")` | L2 ball, Q from `shrinkage.EST` | Germany max \|Δw\| 0.194 | — |
| `lasso_lambda`, `ridge_lambda` (`scest`, `scpi`) | deprecated and ignored | — | use `Q=` / `Q2=` |
| `sp.ssaggregate` | AKM SE of `x`; z instead of t(n−k); HC1 diagnostic from the 2SLS sandwich | SE 0.06646 → 0.29044 | — |
| `sp.shift_share_se` | AKM on the recorded instrument; raises for results not from `sp.bartik` / `sp.ssaggregate` | 0.017595 → 0.290442 | — |
| `sp.bartik(robust=<other>)` | raises `ValueError` | silently HC1 → error | `robust="hc1"` |
| `sp.staggered_synth` | donors untreated through the effect window; unit-level ATT | 2.6649 → 2.5971 | — (contaminated donors) |
| `sp.staggered_synth(penalization=)` | multisynth's `lambda` on the normalised objective | — | — (the old scale had no reference) |
| `sp.discos` (individual-level data) | Gunsilius estimator; default `method="quantile"` | 1.2491 → 0.9245 | aggregate to unit-period means first (the fallback then runs, with a warning) |
| `sp.demeaned_synth` / `sp.robust_synth` | consistent MSPE-ratio placebo p-value | null DGP 0.111 → 0.222 / 0.444 | — (old statistic was inconsistent) |
| `sp.robust_synth(l1_penalty>0)` | unpenalised intercept; `l1` multiplies ‖w‖₁ in `RSS + l2‖w‖² + l1‖w‖₁` | intercept 0.53 → 20.74 at `l2=5, l1=1e-12` | old `l1` ≈ new `l1/2`, except that the intercept is no longer penalised |
| `sp.robust_synth` / `sp.demeaned_synth(covariates=)` | raises `NotImplementedError` | — | drop the argument (it was ignored) |

### RD and IV

| Function | What changes | Old → new (example) | Old number, if you need it |
| --- | --- | --- | --- |
| `sp.tF_critical_value`, `sp.tF_adjustment` | every value between F = 4 and 106 | c(10) 3.16 → 3.4353 | — (old table was wrong) |
| `sp.iv_diag` | `tF_critical_value`, `tF_adjusted_ci`, `se_ols`, `ci_ols`, `p_ols`; tF NaN for k > 1 | tF c.v. 18.66 → 12.238 (HC1); `se_ols` 0.030699 → 0.035494 | — |
| `sp.weakrobust` | `clr_stat`, `clr_pvalue`, `clr_ci`, `k_stat`, `k_pvalue`, `k_ci`, `tF_critical_value` | CLR 0.0094457 → 0.0090722 | `clr_method="simulate"` restores the simulation only |
| `sp.jive` | `variant="jive2"` estimates; default SEs | jive2 0.18519 → −3.38682; SE 3.3019 → 7.0217 | — |
| `sp.rdrobust` | explicit `p <= deriv` honoured; `deriv > p` raises | `deriv=1, p=1`: 1.3843 → 0.9933 | pass `p=deriv+1` |
| `sp.rkd` | default bandwidth; fuzzy SE | h 0.21760 → 0.21801 | pass `h=` explicitly |
| `sp.rdplot` | bin count / positions, CI bars; default kernel string | esmv J (17, 16) → (38, 41) | `nbins=` to fix the bins |
| `sp.rdplotdensity` | every curve | left density 0.6045 → 0.2325 | — |
| `sp.mccrary_test` | estimate / SE / p; `model_info["n_bins"]` is the number of histogram cells | θ 1.3036 → 1.2707 | — |
| `sp.rdhte`, `sp.rdbwhte`, `sp.rdhte_lincom` | robust SE / CI / p, default bandwidth, binary-z output, `rdbwhte` return type for subgroups | SE 0.0697 → 0.0993; h 0.1506 → 0.3893 | `vce="hc1"` for the HC1 flavour; the conventional SE is not exposed |
| `sp.rd_bias_aware_fuzzy` | CI, M defaults, h default, nearest-neighbour SE | default set (0.62, 3.92) → (0.941, 3.143) | — |
| `sp.rdrbounds` | both bounds | Γ = 2 upper 0.159 → 0.346 (R) | — |

### Spatial, survey and structural

| Function | What changes | Old → new (example) | Old number, if you need it |
| --- | --- | --- | --- |
| `W.transform` on non-binary weights | styles re-weight the constructed weights; `"V"` sums to n | Columbus `"V"` sum 105.2985 → 49 | set `transform = "B"` first, then `"R"` |
| `sp.gwr` (exponential kernel, adaptive Gaussian / exponential) | GWmodel kernels | AICc 896.3499 → 894.1283726 | — (old kernel was not a documented quantity) |
| `sp.gwr_bandwidth` | CV is leave-one-out; GWmodel `bw.gwr` search | CV adaptive bisquare 7 → 147 | —; for the old search bounds pass `bw_min` / `bw_max` |
| `sp.mgwr` | β kept directly (was 0 where x = 0) | — | — |
| `sp.sarar_gmm` | GS2SLS (`spatialreg::gstsls`) | const 43.97302 → 43.54044357 | `w_lags=1` for PySAL `GM_Combo`; the old estimator is not reproducible |
| `sp.spatial_panel` | joint-information SEs; exact ρ; two-way SDM lags | SE(lpcap) 0.02535161 → 0.02544250 | `vce="oim"` for Stata `xsmle`; the old conditional β SEs are not available |
| `sp.svydesign` (+ `svymean` / `svytotal` SE) with `fpc=` given as population counts and clusters | Sampling fraction per stratum is #PSU sampled / N_h (R `as.fpc`), not #elements / N_h | `full` design SE: NaN → 0.49514353151367907 (R 0.4951435315136789) | Not reachable; the element-based fraction is not a documented quantity (it gives 1 − f < 0 → NaN) |
| `sp.svydesign` design df (CIs, p-values) when PSU ids repeat across strata and `nest=False` | PSUs are keyed by (stratum, PSU), so df = #PSU − #strata, as in R and Stata | df 1, CI [6.7114, 20.2148] → df 17, CI [12.342014912296227, 14.584186736399586] (R/Stata [12.34201491229616, 14.58418673639966]) | Not reachable; the old df counted raw ids globally, inconsistent with the nested variance it was paired with |
| `sp.svyglm(family="binomial" \| "poisson")` SEs | Sandwich bread is (X′ diag(w·V(μ)) X)⁻¹, not the Gaussian (X′WX)⁻¹ | Logit SE [0.1179, 0.0385] → [0.5098, 0.1731] (R; match 3.9e-14); Poisson SE [0.4900, 0.1605] → [0.2138, 0.0502]; coefficients move ~1e-9 (tighter IRLS) | Not reachable; the Gaussian bread is wrong for non-Gaussian families |
| `sp.svymean` / `sp.svytotal` DEFF | Denominator is svyvar·(N − n)/(N·n), not svyvar/Σw, so it no longer depends on weight scale | Mean DEFF 204.877 → 3.9032222795694005; total DEFF 1511.21 → 28.790859422197673 (R `deff=TRUE`) | Not reachable; the scale-dependent value is not a documented quantity (`deff="replace"` gives Stata's no-fpc DEFF, not the old number) |
| `sp.rake` | Stops when every category share is within `tol` of its target (relative), default `tol=1e-10`; was an absolute change on O(1/n) weights at 1e-6 | n = 100 000: margin error 4.3e-4 → 8.1e-11; n = 200: gap to R's fixed point 2.4e-7 (design start) / 1.9e-6 (equal start) → 1.04e-10 | Not reachable; `tol` now means a relative margin gap, so old `tol=1e-6` runs but stops on a different rule |
| `sp.lcsf` / `sp.zisf` | Newton polish after L-BFGS-B, and a Richardson-extrapolated Hessian for OIM SEs | lcsf vs sfaR (two fixture cases): estimates max rel err 1.15e-3 / 1.43e-3 → 1.1e-8 / 6.2e-9; SEs 1.44e-3 / 2.13e-3 → 8.6e-8 / 6.7e-8; log-likelihood −324.1954416 → reaches sfaR's −324.1954399 | Not reachable; the old optimum was a premature stop, not a different estimand |
| `sp.metafrontier(cost=True)` | Sign of the technology gap flipped for cost frontiers; TE_meta changes with it | TGR min / mean / max 1.0 / 1.0 / 1.0 → 0.170 / 0.445 / 1.0 | Not reachable; TGR ≡ 1 came from clipping a wrong-signed gap |
| `sp.blp` standard errors | Joint (β, σ) robust GMM sandwich with 1/N-scaled Jacobian and analytic dδ/dσ | Linear SEs (0.00434, 0.00070, 0.00187) → (0.1075, 0.0201, 0.0474); σ SE 4.99e-5 → 0.0304 (fixture, N = 609; pyblp match 5.6e-9) | Not reachable; old SEs were too small by √N (linear) and N (σ) |
| `sp.blp` elasticities when price is in `x_random` | Each simulated consumer's price coefficient is α + σ_p·ν (`sigma_price` was unused) | Rel. error vs pyblp > 1e-2 → ~1e-8 | Not reachable; the old value ignored the estimated σ_p |
