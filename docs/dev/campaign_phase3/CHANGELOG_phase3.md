Cross-language parity campaign, phase 3. Estimator callables with a
cross-language (R / Stata) grade went from 224 to 362 of 773 (29.0% → 46.8%).
The campaign fixed 125 defects: survival / epidemiology 11, time series 7,
inference / sensitivity 7, panel / GLMM 12, treatment effects 12,
DiD / synthetic control 37, RD / IV 17, spatial / survey / structural 22
(each numbered item in the family reports counted once; docstring-only and
label-only corrections not counted). Most were found by running the named R
or Stata reference on the same CSV bytes; the rest turned up while building
those comparisons, from a known-truth DGP or from reading the code, where no
package computes the quantity. None was caught by the existing unit tests.
See [MIGRATION.md](MIGRATION.md#parity-campaign-phase3).

### ⚠️ Correctness

**Survival and epidemiology**

- **`sp.cuminc`: the delta-method variance used the wrong increment, and
  Gray's test was a different test.** The Marubini–Valsecchi variance terms
  use `F(t) - F(t_j)`; the code used `F(t) - F(t_{j-1})`. At the first event
  time, where the variance is the binomial `p(1-p)/n`, the old value was
  0.000228 against Stata `stcompet`'s 0.000282. Cause-1 SE at
  t = 0.25 / 2.0 / 7.0 goes from 0.015107 / 0.027944 / 0.032854 to
  0.016795 / 0.030489 / 0.035325 (`stcompet` to 2e-16); CIs move with it.
  Gray's test was a log-rank on unweighted subdistribution risk sets with a
  hypergeometric variance. The risk set agrees with Gray's only without
  censoring, and the variance never does. It is now Gray's score and
  asymptotic covariance, matching R `cmprsk::cuminc` to 1e-14: cause 1
  chi2 2.3950 (p 0.1217) → 1.8499 (p 0.1738), cause 2 0.5354 → 0.7835.
- **`sp.finegray`: censoring weights at the wrong limit; SEs were the
  inverse information.** The weights were `Ĝ(t)/Ĝ(T_i)`; `cmprsk::crr` and
  Stata `stcrreg` use left limits, `Ĝ(t-)/Ĝ(T_i-)`. The two differ whenever
  a censoring ties an event time. Coefficients (0.440740, 0.270487,
  0.282621) → (0.440603, 0.270658, 0.281948) (`crr` to 6e-15). The default
  SE is now Fine and Gray's sandwich with the censoring-estimation term
  (`vce="robust"`, `crr` `var`): (0.084511, 0.168711, 0.168777) →
  (0.081090, 0.159197, 0.157572). `vce="model"` returns the old inverse
  information (`crr` `invinf`); `small_sample=True` applies `stcrreg`'s
  N/(N-1).
- **`sp.cox_frailty` returned the ordinary Cox fit.** The beta step ignored
  the frailty offset, theta was searched on [0.5, 100] of a likelihood that
  was not the gamma marginal, and the SEs came from the plain Cox
  information. Rewritten: Newton fit of the gamma-penalised partial
  likelihood in (beta, w) for fixed theta, theta maximising the integrated
  likelihood (R `c.loglik`, Stata `e(ll)`), SEs from the full penalised
  information. `.theta` is now the frailty **variance**, as in R and Stata;
  it was documented as a precision. Fixture: beta (0.480666, -0.506041) →
  (0.528795, -0.504217); SE (0.099537, 0.176958) → (0.105516, 0.187037);
  theta 0.500004 (precision, at its bound) → 0.240201 (variance; Stata
  0.240199, chibar2 6.30 on both sides). Matches R
  `coxph(frailty(sparse=FALSE))` and Stata `stcox, shared()`.
- **`sp.roc_curve` / `sp.auc` were wrong with tied scores.** The curve
  stepped one observation at a time, so a case–control tie counted 0 or 1
  depending on an unstable argsort (0.6694 against the Mann–Whitney 0.6672
  on one simulated draw). The AUC is now the Mann–Whitney placement value
  with ties counted one half: 0.68376068 → 0.68389966 on the fixture's tied
  score (pROC and Stata `roctab` to 2e-16). `thresholds`, `tpr` and `fpr`
  now have one entry per distinct score, not per observation.
- **`sp.kdensity`: `bw_method="sheather-jones"` returned the Silverman
  rule, and the default Silverman width matched neither reference.** The
  Sheather–Jones branch was literally `0.9*a*n^(-1/5)`; it now solves the
  equation without binning: 0.5222 → 0.5592 (2e-7 from R
  `bw.SJ(nb=1e7, tol=1e-14)`, 1.8e-3 from R's binned default). The default
  combined R's quantile type with Stata's 1.349 constant; it is now exactly R
  `bw.nrd0` (1.34): 0.522176 → 0.525683 (+0.67%). This second change is a
  choice between two constants, made because the old value matched no
  reference; `bw_method="stata"` reproduces Stata `kdensity`'s width.
- **`sp.cox(ties="breslow")` ran Efron's approximation.** The code comment
  said "Breslow currently reuses the Efron helper". On tied data x1 goes from
  0.355720 to 0.335603 (`coxph` and `stcox` Breslow to 1e-9). `ties="efron"`
  is what the old call ran.

**Time series**

- **`sp.panel_unitroot`: three of the four tests were not the named test.**
  LLC was `sqrt(N)·mean(t_i) + sqrt(N)`, with no long-run variance, pooled
  regression or adjustment table. IPS used rounded moments that did not
  depend on T or lags, and invented moments for `trend='n'`. Fisher combined
  normal-CDF p-values instead of MacKinnon (1994) p-values. Hadri treated
  `trend='n'` as a trend. Rewritten after Stata `_xturllc` / `_xturips` /
  `_xturfisher` / `_xturhadri` and `plm::purtest`, with
  `convention='stata'` (default) or `'plm'`, `dfcor=` and `robust=`; matches
  both to 1e-10. On `ts_panel.csv` (lags=1): LLC −7.80 → −4.656, IPS
  −6.238 → −6.109, Fisher P 239.1 → 117.09 (each = `xtunitroot`). On a
  random-walk panel LLC went from −0.833 (p 0.20) to −0.0016 (p 0.50).
- **`sp.engle_granger` compared its statistic with critical values for a
  different regression.** The residual ADF regression included a constant;
  the MacKinnon Engle–Granger critical values are tabulated for step 2
  without deterministic terms. `trend` ('c' / 'ct' / 'ctt') now enters step
  1, as in `egranger`; `trend='ct'` and `alpha` used to be ignored, NaNs were
  not dropped, and the asymptotic three-decimal critical values were wrong
  for N ≥ 3 (N=3 5%: −3.78 against −3.74066). Critical values are now the
  MacKinnon (2010) response surface at T = n−1, `alpha` ∈ {0.01, 0.05,
  0.10}. y1~y2, L=0: Z −9.6167 → −9.6364 (Stata `egranger` −9.6364), 5%
  critical value −3.34 → −3.3608; with `trend='ct'`: Z −8.6168 → −8.4234,
  critical value −3.34 → −3.8190. Matches `egranger` and `urca::ur.df`.
- **`sp.johansen` critical values.** Max-eigenvalue 5% values for k−r = 3
  and 4 were 21.12 and 27.42 (Stata and Osterwald-Lenum: 20.97, 27.07);
  `trend='n'` and `'ct'` used the unrestricted-constant table (for 'n',
  trace k−r=3 should be 24.31, not 29.68); k=1 trace used 3.84 instead of
  3.76; `alpha` and an invalid `test` were ignored. Now Stata's full
  5-case table, with the restricted cases `trend='rc'` / `'rt'` and Stata's
  trend names; `alpha` ∈ {0.05, 0.01}. The statistics were already right;
  only the rank decisions change.
- **`sp.bvar` built every equation's prior from the last equation.** One
  prior-variance vector was overwritten in a loop, so the own-lag variance
  went to the wrong variable and results depended on column order (gdp
  own-lag 0.901 with columns (gdp, infl, rate), 0.637 after reordering). The
  prior variances were also scaled by s_k² a second time. Now the Minnesota
  prior with fixed Σ₀ of Stata `bayes, minnfixedcovprior: var`, solved in
  closed form; new `lambda3`, `lambda4` and `sigma='ar'|'var'`. gdp own-lag
  0.901 → 0.6382 (Stata MCMC 0.6383 ± 0.00017).
- **`sp.garch`: pre-sample variance and optimiser.** The pre-sample variance
  was `var(eps)`; Stata `arch` and `rugarch` use `mean(eps²)` at the current
  μ. Nelder–Mead stopped short; the fit now uses an analytic score, BFGS and
  a Newton polish. `forecast()` ignored every ARCH / GARCH lag beyond the
  first. Log-likelihood −2144.43397943 → −2144.43398728 (Stata `e(ll)`
  −2144.43398728); μ 0.0629375 → 0.0629347 (Stata 0.0629347).
- **`sp.cusum_test` boundary.** `alpha` values other than .01 / .05 / .10
  silently used 0.948. The boundary coefficient is now the exact root of the
  crossing probability (0.9478982 at 5%); the default boundary array moves by
  1e-4 relative.
- **`sp.structural_break` sup-F grid** stopped one candidate short of
  `strucchange::Fstats(from=0.15)`. The sup-F changes only when the maximum
  sits at the last point.

**Inference and sensitivity**

- **`sp.ivreg(vce="wild")` / the WRE IV wild bootstrap counted ties and
  never enumerated.** It now uses every Rademacher sign vector when
  `2**G <= n_boot` and a strict-inequality p-value, as Stata `boottest`
  does. On the 16-cluster weak-instrument fixture the p-value is 22361 /
  65536 = 0.3412017822, equal to `ivreg2` + `boottest` (0.34120178)
  instead of agreeing within simulation error; results report
  `enumerated` and `n_boot_requested`.
- **`sp.wild_cluster_bootstrap`, `sp.wild_cluster_boot`,
  `sp.subcluster_wild_bootstrap` (and `sp.panel` feols `wild` p-values)
  never enumerated and counted ties.** Stata `boottest` and R
  `fwildclusterboot` enumerate the Rademacher grid when `2^G <= reps` and
  count `|t*| > |t|` strictly; StatsPAI always sampled `n_boot` draws and
  counted `>=`, which under enumeration adds 2/2^G. G = 12, `n_boot=9999`:
  `d` p 0.92569 → 0.9306640625 (= 3812/4096, both references); `x2`
  0.054505 → 0.0546875; subcluster 0.90999 → 0.91259765625 (Stata). The
  change applies whenever `2^G <= n_boot` (G ≤ 9 at the default 999) and by
  at most 2/B otherwise. `n_boot` now reports the draws used.
- **`sp.wild_cluster_ci_inv` interpolated the CI endpoints.** p(h0) is a
  step function; the endpoints were linear interpolations on a 41-point
  grid. Each bracket is now bisected to the jump. `d` CI (−1.114049,
  1.103695) → (−1.121498336078, 1.103681178126), equal to
  `fwildclusterboot` to 4.8e-12. It warns when the grid does not bracket an
  endpoint (it returned the grid edge).
- **`sp.fisher_exact` / `sp.ri_test` sampled designs they could enumerate.**
  With no more assignments than `n_perm(s)` the distribution is now exact
  (R `ri2`): simple design p 0.1443 → 132/924 = 0.142857…, stratified
  0.0315 → 148/4900 = 0.0302040816.
- **`sp.rosenbaum_bounds` / `sp.rosenbaum_gamma`: continuity correction,
  zeros, and the two-sided bound in the wrong tail.** A 0.5 continuity
  correction that no reference applies is gone; zero differences are ranked
  with weight 0 (`zero_method="pratt"`, R `DOS2` and Stata `rbounds`) instead
  of dropped; "less" and two-sided bounds use the correct tail. Upper bound,
  40 pairs, Γ = 1, 1.5, 2, 2.5, 3: (0.0022, 0.0302, 0.1077, 0.2232, 0.3525)
  → (0.002835, 0.041828, 0.149128, 0.301256, 0.459566). Two-sided on the
  negated data: (4.2e-3, 7.7e-5, 1e-6, 0, 0) → (0.00567, 0.0837, 0.2983,
  0.6025, 0.9191); the old bound called a negative effect insensitive to any
  hidden bias.
- **`sp.oster_bounds` had the two R² gains swapped.** It computed
  β* = β̃ − δ(β̊−β̃)(R̃²−R̊²)/(R_max−R̃²) instead of
  … (R_max−R̃²)/(R̃²−R̊²), and only the approximation. From data it now
  solves Oster's exact problem as Stata `psacalc` does (`method="exact"`);
  from summary statistics it uses the corrected approximation. R_max = 1.3
  R̃² = 0.61269: δ* 1.20492 → 0.904199433 (`psacalc` 0.9041994327987157);
  β*(δ=1) 0.078546 → −0.071462629.
- **`sp.oster_delta` read the default `r_max=1.3` as an R² of 1.3.** Values
  above 1 are now a multiplier of R_full (default min(1, 1.3 R_full), as
  documented), and the solution is exact. Defaults: δ* 0.16189 → 0.90420;
  β*(δ=1) −2.39096 → −0.07146.
- **`sp.pate(method="aipw")` was not doubly robust.** The augmentation was
  normalised by the sum of all odds weights and rescaled by n_exp / n_tgt.
  With a correct participation model and a misspecified outcome model the
  bias was −0.168 (MC SE 0.013); with per-arm Hájek normalisation it is
  −0.001 (MC SE 0.016). No package computes this estimator; the check is on
  known truth.

**Panel and GLMM**

- **`sp.meglm(family="gaussian")` fixed the residual variance at 1.** SE of
  x1 was 0.0428 against Stata / `lmer` 0.0335, and point estimates were off
  in the third digit. σ² is now estimated; the fit equals
  `sp.mixed(method="ml")` and Stata `meglm` (4.2e-9 / 2.8e-7).
- **GLMM and `sp.mixed` optimisers stopped short.** `meglm` stopped on
  L-BFGS-B's relative function change, 2.7e-4 from the optimum: `mepoisson`
  Laplace `_cons` 0.3451384 → 0.3450937699 (Stata 0.34509377003). `sp.mixed`
  REML / ML stopped ~1e-6 short (balanced-design REML ICC 1.04e-6 from the
  ANOVA value, now 7e-11). Both now finish with Newton steps.
- **GLMMs with non-canonical families (NB-2, gamma, ordinal logit) used the
  Fisher curvature** in the Laplace approximation and AGHQ scaling (lme4's
  PIRLS convention). Stata, glmmTMB and `clmm` use the observed curvature,
  now the default `curvature="observed"`. Gamma `_cons` 0.6196 → 0.6372
  (Stata 0.63715030, glmmTMB 0.63715031); NB `_cons` 0.4480 → 0.4728; ologit
  x1 0.6816 → 0.6865. `curvature="expected"` gives the old approximation.
- **`sp.meologit` SEs held the variance components fixed**, the formula
  `meglm` dropped in 1.24.0, and there were no threshold SEs. Now the inverse
  observed information over (β, thresholds, covariance), with delta-method
  `thresholds_se`.
- **`sp.icc`** returned NaN for every GLMM, used a heuristic SE
  (var(log σ²_u) ≈ 2/n_groups) and failed on three-level fits. It now gives
  the latent ICC for `melogit` / `meologit`, raises for count / gamma GLMMs
  and random-slope models (Stata `estat icc` refuses them too), uses the
  delta method and a logit-scale CI as `estat icc` (mixed ML SE → 0.0533785
  = Stata), and returns Stata's level ICCs for three-level fits (SE NaN,
  warned).
- **`MixedResult.n_params` counted the residual variance twice**: AIC was
  +2 and BIC +log n against Stata `e(k)` / R `AIC()`.
- **`sp.xtnbreg(model="fe" / "re")` computed different estimators.** `"fe"`
  was an unconditional NB-2 with entity dummies, `"re"` a normal
  random-intercept NB-2 GLMM. Both now compute Stata's (and `pglm`'s)
  Hausman–Hall–Griliches models: conditional FE and beta-dispersion RE. z1
  0.418 → 0.3783114 (Stata 0.3783114). The old estimators are
  `model="ufe"` and `model="normal_re"`.
- **`sp.interactive_fe` SEs were not Bai's.** Regressors were projected with
  M_F only; Bai's D0 uses M_Λ X M_F. xa SE 0.03194 → 0.03304 (default dof) /
  0.0331716 (`dof="regife"`, = Stata `regife`). Cluster SEs now carry the
  CR1 factor, final residuals use the final factors, and the default `tol`
  is 1e-10 (was 1e-6).
- **`sp.gmm` with nonlinear moments stopped ~1e-8 short** and reported
  `converged=False`. A Gauss–Newton finish (Stata `gmm`'s algorithm) puts the
  estimates 1.2e-12 from Stata.
- **`sp.mixlogit`** reported negative SDs, perturbed the model with hidden
  `+1e-6` / `+1e-8` terms on the Cholesky diagonal and SDs, and a
  `H + 1e-8 I` ridge on the Hessian, and its robust variance lacked Stata's
  N/(N−1). All removed; with Stata's Halton draws the log-likelihood agrees
  with Stata `mixlogit` to 5e-13.

**Treatment effects**

- **`sp.ltmle` was a different algorithm, and the binary ATE came out
  doubled.** Pseudo-outcomes were fit by linear regression, only the current
  treatment was set to the regime, and binary outcomes were updated on
  `logit(Y)` with `Y` clipped to 1e-6. Re-implemented as R `ltmle`'s
  sequential-regression TMLE. Binary ATE / SE 0.75744 / 0.03477 → 0.37280 /
  0.04434; continuous 1.95265 / 0.09166 → 2.04340 / 0.10320 (both = R).
  `propensity_bounds` now bounds the cumulative g, default `(0.01, 1.0)`
  (was `(0.01, 0.99)` per step).
- **`sp.gformula_ice_fn(bootstrap=0)` reported `sd(Y)/sqrt(n)` as the SE.**
  It is now the sandwich of the stacked sequential OLS equations: 0.05909 →
  0.06830 (`geex` 0.06830). The point estimate does not change.
- **`sp.aipw(estimand="ATT")` SE** omitted the `-tau D / p` term of the
  ratio estimator's influence function: 0.10180 → 0.08987 with the default
  cross-fit (checked against DoubleML's ATTE score).
- **`sp.multi_treatment`**: the multinomial-logit GPS was sklearn's
  L2-penalised `LogisticRegression()`, the bootstrap refit a different
  outcome model (50 trees) from the estimate (100 trees), and arms with ≤ 2
  units silently got a zero outcome model (now an error). ATE(1 vs 0)
  1.163538 → 1.163558; bootstrap SE (B=50) 0.08920 → 0.09146.
- **`sp.lee_bounds` trimmed one observation too few** (`floor((1−p) n)`).
  The default is now Lee's sample-quantile rule (`trimming="quantile"`):
  (0.71763, 2.61867) → (0.72453, 2.61202).
- **`sp.survivor_average_causal_effect` and the `sp.principal_strat` SACE
  collapsed the bounds.** Outcomes missing for non-survivors were dropped with
  their rows, which set P(S=1|D) = 1: (1.68442, 1.68442) → (0.72453,
  2.61202), now equal to `sp.lee_bounds`.
- **`sp.principal_strat` "Complier (LATE)"** reported E[Y(1)|complier]. It
  now reports the Wald LATE: 6.39543 → 7.35517 (`AER::ivreg` 7.35517).
- **`sp.manski_bounds(assumption="mts")` also imposed MTR** and swapped the
  endpoints when the naive difference was negative. `"mts"` is MTS alone;
  the joint assumption is `"mts_mtr"`: [0, 0.32812] → [−0.336, 0.32812]
  (Stata `tebounds`).
- **`sp.horowitz_manski` dropped strata missing a treatment arm**, and with
  them their probability mass: [−0.332, 0.629] → [−0.337, 0.663] on a
  constructed case, equal to the unconditional bounds.
- **`sp.mediate_interventional(tv_confounders=)`** plugged in the marginal
  mean of L for both arms, deleting the D→L→Y path. IDE 0.57441 → 1.10616,
  total 0.81801 → 1.34976 (CMAverse `rpnde` / `te`).
- **`sp.g_estimation`** ignored `propensity_covariates` and always fitted a
  linear-probability propensity. The default is now DTRreg's logistic
  propensity: psi (1.05636, 0.94856) → (1.05524, 0.94731) = DTRreg; on NHEFS
  3.4626 → 3.4611485591 (What If Program 14.2's logistic g-estimate).
  `propensity_model="linear"` gives the old numbers.
- **`sp.ipcw`** gave censored rows the same weight as observed rows
  (`np.where(d == 1, w, w)`); they now get 0, as documented (was ≥ 0.77).
  `method="cox_ph"` crashed with more than one covariate and accumulated the
  Breslow hazard in data order.

**DiD and synthetic control**

- **`sp.sdid` / `sp.synthdid_estimate` / `sp.sc_estimate` /
  `sp.did_estimate`: none of the three SEs was synthdid's.** The placebo SE
  refit each control from scratch (one placebo unit, `ddof=1`, `n_reps` and
  `seed` ignored); the bootstrap resampled controls only; the jackknife
  dropped controls and refit the weights. They now follow `synthdid::vcov`:
  random permutations warm-started from the full-sample weights with frozen
  regularisation and divisor r, all-unit bootstrap, leave-one-unit-out
  jackknife with fixed weights. Per draw they agree with R to 2.1e-11. Prop.
  99 placebo (seed 42): sdid 2.6041 → 2.5630, sc 3.4510 → 3.4350, did 4.9324
  → 5.0013. Five-treated panel jackknife: sdid 0.1248 → 0.4382, sc 0.2378 →
  1.5116, did 0.3189 → 0.7810 (= R to 1e-14). An SE synthdid leaves
  undefined is NaN with a warning; Prop. 99 jackknife (one treated unit)
  0.7476 → NaN. Point estimates are unchanged.
- **`sp.breakdown_m` ignored `method` and did not invert the set it names.**
  It returned `(|θ̂| − z·SE)/(e+1)` for every input. With smoothness and a
  covariance it now root-finds on the FLCI that `sp.honest_did` reports
  (R HonestDiD with an exact quantile to 6e-11): e=0 / 1 / 2 on a
  hand-crafted event study 0.1512 / 0.0967 / 0.0772 → 0.15345 / 0.06460 /
  0.03829; mpdta CS e=0 0.019253 → 0.007990. `relative_magnitude` inverts the
  native RM interval (mpdta 0.019253 → 1.948) and warns; without a covariance
  the closed form is kept and warns.
- **`sp.honest_did(method="smoothness")`, native: false SLSQP convergence
  widened the FLCI.** The worst-case bias b(h) froze at its start value
  whenever the variance constraint was slack there (12% off a bound at
  M = 0.02, 123% at M = 0.05). Fixture e=1, M=0.02 lower bound 0.10497 →
  0.11913 (HonestDiD 0.11913); mpdta CS e=1 lower bounds −0.081896 /
  −0.121360 → −0.078258 / −0.107401 at M = 0.01 / 0.02. The best h is
  refined past the grid, and `ci_lower`, `ci_upper` and `M` are no longer
  rounded to 6 decimals.
- **`sp.continuous_did(method="twfe")`: SE degrees of freedom and
  unbalanced panels.** The iid SE divided by `n - 1` although the unit and
  period effects were absorbed (0.042912363 → 0.047212137, fixest
  0.0472121368); the clustered factor omitted period effects not nested in
  the cluster (0.049641066 → 0.049849497); and the one-pass double demeaning
  gave a wrong slope on unbalanced panels (0.403496461 → 0.402987816, SE
  0.044333346 → 0.049373238, both = fixest). Matches `fixest::feols` to
  1e-15.
- **`sp.continuous_did(method="att_gt")` bootstrap.** Each drawn unit was
  kept once however often it was drawn (a subsample), and the pooled SE
  assumed independent dose bins although they share one comparison arm.
  Pooled SE 0.10639 → 0.18500 (unbalanced 0.11190 → 0.18991); point
  estimates unchanged. With no zero-dose units the SE was NaN; it is now
  finite and `model_info["control_arm"]` names the arm used.
- **`sp.continuous_did(method="dose_response")` reported the average
  pointwise SE of the fitted level** as the SE of the average slope. It is
  now a unit bootstrap of the average slope: 0.243084 → 0.154392
  (unbalanced 0.255191 → 0.146908). The `linregress` fallback warns and is
  recorded in `model_info["fallback"]` instead of switching silently.
- **`sp.did_timevarying_covariates` was a different estimator.** Comparison
  units' covariates were frozen at their median period instead of the
  cohort's g−1, and the adjustment was a pooled `dY ~ D + X` regression. It
  is now the outcome-regression ATT(g,t) of `ptetools::pte_default` /
  `did::att_gt(est_method="reg")` (to 1e-14), and the headline is the group
  aggregation: 1.273390 → 1.257705 (ptetools 1.25770523749721); ATT(4,4)
  1.242550 → 1.823768.
- **`sp.mc_panel` / `sp.mc_synth` / `sp.matrix_completion` /
  `sp.synth(method="mc")` never estimated the fixed effects.** Both
  references (MCPanel, `fect`) default to unpenalised unit and time effects;
  StatsPAI fitted soft-impute with no intercept, so the nuclear norm shrank
  the outcome level towards zero. New `fixed_effects="two-way"` default. True
  ATT 2.344: `mc_panel` 2.64068 → 2.37710, at lambda 8 2.93426 → 2.38462
  (MCPanel 2.38462); `mc_synth(seed=0)` 1.66626 → 1.32783; Prop. 99
  `mc_synth(placebo=False, seed=0)` −13.119 → −17.980. The stopping rules
  (`tol` 1e-5 / 1e-6, `max_iter` 1000 / 500) left about 8e-5 relative error
  in the ATT; now `tol=1e-10`, `max_iter=5000`, and hitting the cap warns.
  `mc_panel` returns p = NaN, not 0.0, when the SE is 0.
- **`sp.spillover_did` ignored ring exposure timing under staggered
  adoption.** Every ring unit entered every cohort's comparison. Each ring is
  now measured from its exposure onset: ring 1 0.79123 → 1.23765, ring 2
  0.25243 → 0.39661 (R `did` 1.2376 / 0.3966). SEs now include the
  cohort-share weight term: direct SE 0.077918 → 0.091366 (`did`). Single-
  cohort output is unchanged.
- **`sp.harvest_did`**: pre-period placebo cells used the treated cohort as
  its own control (cohort 5, e=−4: −0.3550 → −0.5104, `did` −0.5104); the
  event-study, aggregate and pre-trend inference assumed independent cells,
  understating the headline SE by about 40% (0.06207 → 0.10148; estimate
  1.79243 → 1.78531; pre-trend p 0.303 → 0.170); and cell SEs switch from
  Welch (`ddof=1`) to `did`'s influence-function convention (divisor n).
- **`sp.causal_impact`: the SE of the average and cumulative effect treated
  post-period forecast errors as independent** although they share the AR(1)
  state. SE 0.23480 → 0.28516, `se_total` 7.0439 → 8.5547; point estimate
  unchanged. Pinned by a dense-Gaussian identity, not by R `CausalImpact`,
  which fits a different model.
- **`sp.scpi` was not the Cattaneo–Feng–Titiunik procedure.** It subsampled
  pre-periods for the in-sample part, used a residual variance for the
  out-of-sample part, built `effect ± z·sqrt(var_in + var_out)` and reported
  an SE and p-value R does not define. It is now a port of R `scpi` 4.0.1's
  single-treated-unit path. Germany, simplex: 1991 effect interval [0.230,
  0.774] → [−0.731, 1.316] (R, same weights: [−0.700, 1.302]); 2003 [−4.32,
  −2.61] → [−5.25, −1.11]; average-effect CI (−2.29, −1.05) → (−3.29,
  −0.22). The old intervals were 2–4× too narrow. `se` / `pvalue` are NaN;
  nominal coverage is `1 - (u_alpha + e_alpha)`.
- **`sp.scest(w_constr="lasso" | "ridge")` were penalised fits on
  standardised data.** R `scpi` constrains `||w||_1 ≤ Q` (Q = 1) and
  `||w||_2 ≤ Q` (Q from `shrinkage.EST`). Germany max |w − w_R| 0.227
  (lasso) / 0.194 (ridge) → 1.8e-12 / 1.4e-5. `"simplex"` and `"ols"` are
  unchanged.
- **`sp.ssaggregate` AKM SE was the wrong formula.** It summed
  `s_ik Z_i e_i` instead of AKM's `hX_k s_k'e` and was anti-conservative. SE
  of `x`: IV + controls 0.06646 → 0.29044, IV intercept only 0.04967 →
  0.27941, reduced form 0.22565 → 0.96533 (= R `ShiftShareSE` = Stata
  `ivreg_ss`). The `SE (HC1)` diagnostic dropped the first stage: 0.17750 →
  0.27585. p-values and CIs use z, as both references do. Point estimates
  unchanged.
- **`sp.shift_share_se` used the second-stage fitted values as the
  instrument**: 0.017595 → 0.290442. It now reruns AKM on the inputs that
  `sp.bartik` / `sp.ssaggregate` record, and raises `ValueError` for other
  results instead of returning a number.
- **`sp.staggered_synth` used contaminated donors and was not Ben-Michael,
  Feller & Rothstein's estimator.** Donors for cohort g included units
  adopting after g while the effect window ran to the panel end, so later
  adopters' treated outcomes entered earlier cohorts' counterfactuals; the ATT
  was weighted by post-period count. Rewritten as `augsynth::multisynth`'s QP
  (ATT to 5.6e-12, jackknife SE to 1.8e-11): `method="separate"` 2.664881 →
  2.597125, `"pooled"` 2.597259 → 2.526639. `penalization` is now
  multisynth's `lambda` on the normalised objective.
- **`sp.discos` averaged individual-level data away.** It pivoted to
  unit-period means and treated each unit's time series as its
  distribution. On individual-level data it now runs the Gunsilius estimator
  of R `DiSCos::DiSCo` 0.1.4 (default `method="quantile"`): truth 0.9,
  `"quantile"` 1.249059 → 0.924513, `"mixture"` 0.921300 → 0.967711.
  Aggregate panels keep the old heuristic with a warning; its weights are now
  solved exactly (Prop. 99 −23.0638600 → −23.0638480, 30 s → 0.7 s).
- **`sp.demeaned_synth` / `sp.robust_synth` placebo p-values compared
  different statistics**: the treated unit's mean squared post gap against
  each placebo's squared mean post gap, which is anti-conservative. Null DGP:
  p 0.111 → 0.222 (demeaned) and 0.111 → 0.444 (robust). Point estimates
  unchanged up to solver noise (4e-10).
- **`sp.robust_synth(variant="elastic_net", l1_penalty>0)`** penalised the
  intercept and thresholded at `l1` instead of `l1/2`. At `l2=5,
  l1_penalty=1e-12` the intercept jumped from 20.7388 to 0.5346 and the ATT
  from 3.0336 to 1.9978; now 20.7388 / 3.0336, matching glmnet to 1e-11. The
  default `l1_penalty=0` is unaffected.

**RD and IV**

- **`sp.tF_critical_value` / `sp.tF_adjustment`: the table was not Lee,
  McCrary, Moreira & Porter's.** c(10) was 3.16 (LMMP / ivDiag: 3.4353),
  c(15) 2.54 (2.8662), c(50) 1.98 (2.1529), and it returned 1.96 from F = 75
  on instead of from 106.09. Every value was too small. Now R `ivDiag::tF`'s table and interpolation, 0 error on 26 F
  values. Below F = 4 it returns `inf` where ivDiag clamps to 18.66.
- **`sp.iv_diag` / `sp.anderson_rubin_test` / `sp.weakrobust` looked tF up
  at the homoskedastic F** although the t-ratio uses HC1 / cluster SEs. On
  `rd_iv_ivw`: 18.66 at F=3.96 → 12.238 (HC1, F_eff 4.293) / 16.475
  (cluster). With more than one instrument tF is NaN (None in
  `anderson_rubin_test`), as in ivDiag.
- **`sp.iv_diag` `se_ols` ignored `vcov` and `cluster`**: 0.030699 →
  0.035494 (HC1) / 0.039353 (cluster), = ivDiag.
- **`sp.weakrobust`, `sp.conditional_lr_ci`, `sp.k_test_ci`,
  `conditional_lr_test`: CLR and K statistics.** With two or more
  instruments `Z L^-1` was not orthonormal, so every statistic was
  mis-scaled: CLR 0.0094457 → 0.0090722 (`ivmodel` 0.00907221605522), K
  0.0090481 → 0.0086880 (Stata 0.0086879820). K "at h0" was read off the
  nearest grid point (31.889 vs 32.144 on the docstring example); it is now
  evaluated at h0. The CLR critical value was simulated (the set's lower end
  moved −4.43 / −3.79 between seeds 0 and 1); `method="exact"` (default)
  integrates it: set −3.7680547 (`ivmodel` −3.7680086). `"simulate"` /
  `weakrobust(clr_method="simulate")` keep the simulation.
- **`sp.jive`**: `variant="jive2"` kept observation i (`fitted/(1−h)`):
  0.18519 → −3.38682 (Stata `ujive2` −3.386819). The default SE was
  `s²(X̂'X)⁻¹` without the sandwich: 3.3019 → 7.0217 (Stata 7.021736).
- **`sp.rdrobust` raised an explicit `p ≤ deriv` to `deriv+1`**:
  `rdrobust(deriv=1, p=1)` returned the p=2 estimate (1.3843 against R
  0.9933). `p` defaults to R's rule (1, or deriv+1); an explicit `p` is
  honoured; `deriv > p` raises. A window with too few observations
  (`h=1e-4`, estimate 5.4e-6, SE 6e-22, p 0.0) now raises
  `DataInsufficient`, as R errors.
- **`sp.rkd`**: the default bandwidth was a rule of thumb and the fuzzy SE
  left out the covariance of the two kinks. Now `rdrobust(deriv=1,
  vce="hc1")`: h 0.21760 → 0.21801, estimate 1.03151 → 1.03463; fuzzy SE at
  h=0.4 0.225889 → 0.217965 (R 0.217964991849126).
- **`sp.rdplot`**: an "IMSE" rule with a 0.7 fudge factor chose the bins
  (esmv J = (17, 16) → (38, 41) = R), and `kernel` never reached the fit.
  Bins, bin CIs (R's t intervals) and the polynomial are now R `rdplot`'s;
  the default kernel is R's `"uniform"`, which keeps the old curve.
- **`sp.rdplotdensity`** used a per-side ECDF, a rule-of-thumb bandwidth and
  a heuristic SE. Left density at the cutoff 0.6045 → 0.2325 (R); the old
  curves were about 2× too high.
- **`sp.mccrary_test` was a different estimator under McCrary's name**, with
  silent fallbacks (density 0.01, SE 0.1). Now R `rdd::DCdensity`: θ 1.3036
  (SE 0.2756) → 1.2707 (0.1910). The automatic check in
  `sp.rdrobust(...).model_info["mccrary"]` changes with it.
- **`sp.rdhte` / `sp.rdbwhte` / `sp.rdhte_lincom`**: SEs were the order-p
  HC1 (0.0697) instead of R `rdhte`'s robust bias-corrected HC3 (0.0993);
  default h 0.1506 → 0.3893 (R `rdbwselect`); `b` was ignored; binary
  moderators were not treated as subgroups; `bandwidth_h` was rounded to 6
  decimals; degenerate fits returned zeros with a `1e10·I` covariance.
- **`sp.rd_bias_aware_fuzzy`**: the bias bound `h²M/12` is below the
  local-linear worst case. Rebuilt on the RDHonest construction (bias from
  the realised weights, MROT for `M_y` / `M_d`, RDHonest's bandwidth,
  nearest-neighbour variance). Fixed h=0.4, M=(2, 0.5): (1.1549, 2.3799) →
  (1.1327, 2.3635); defaults (0.62, 3.92), h 0.141, M_y 89.7 → (0.941,
  3.143), h 0.1713, M_y 35.4 (= RDHonest).
- **`sp.rdrbounds` used one median split** with fixed-n assignment instead
  of the extremum over all thresholds with Bernoulli assignment. Upper bounds
  at Γ = 1.2 / 1.5 / 2 were 0.012 / 0.056 / 0.159 against R 0.021 / 0.094 /
  0.346. Now R's algorithm; agreement is within Monte Carlo error (T3).

**Spatial, survey and structural**

- **`W.transform` discarded the constructed weights.** Every style was
  rebuilt from binary weights, so kernel and inverse-distance W became binary
  under `"R"` (Georgia row 146 was uniform 0.023256; now the kernel / row-sum
  values 3.09e-5, 7.19e-5, …). `"V"` lacked the global `n / Q` rescale:
  Columbus queen weights summed to 105.2985, now 49 (= n, spdep style S).
- **`sp.gwr` kernels differed from GWmodel and mgwr.** The exponential kernel
  was truncated at u ≥ 1, adaptive Gaussian and exponential used only the k
  nearest points, fractional adaptive bw was rounded up and bw > n capped.
  AICc: Gaussian adaptive 40 896.3499 → 894.1283726, exponential adaptive 40
  896.7510 → 891.2529635, exponential fixed 60000 1105.0634 → 897.9314442
  (= GWmodel `gwr.basic`).
- **`sp.gwr_bandwidth(criterion="CV")` minimised in-sample RSS**, which is
  monotone in the bandwidth, so it always chose the smallest candidate. Now
  the leave-one-out score: adaptive bisquare 7 → 147, fixed 48990 →
  316468.4969 (GWmodel 316468.4968965). The golden-section search is now
  GWmodel's `bw.gwr` (bounds, floor / round probes, stop; `tol` 1e-3 →
  1e-4): fixed AICc bandwidth 211025.27 → 210996.3347.
- **`sp.mgwr`** recovered β as f/x (0 wherever x = 0) and silently reused the
  previous bandwidth when a search raised. β is kept directly, the exception
  propagates, and non-convergence warns. Other numbers are unchanged.
- **`sp.sarar_gmm` was not GS2SLS.** Stage 3 filtered the instruments, stage
  1 used only [X, WX], the β and ρ SEs were stage-1 SEs printed next to
  stage-3 estimates, and λ came from Nelder–Mead. Rewritten as
  `spatialreg::gstsls`: Columbus const 43.97302 → 43.54044357 (= gstsls),
  ρ 0.453550 → 0.46178656, λ −0.006961 → −0.01698126, SE(const) 11.2365 →
  10.6284221.
- **`sp.spatial_panel`**: SAR / SDM β SEs were conditional on ρ; they now
  come from the joint information matrix, as `splm`: SE(lpcap) 0.02535161 →
  0.02544250 (splm), SE(ρ) 0.02108514 → 0.02351640. ρ was found by bounded
  Brent with the default tolerance and bounds clipped to ±0.99: 0.27468821 →
  0.2746887114326. SDM with `effects="twoways"` lagged the demeaned X,
  W·(QX), instead of Q·(WX): ρ 0.367330749 → 0.368846212 (splm
  0.3688462096).
- **`sp.svydesign`: fpc given as population counts** was divided by the
  number of elements instead of PSUs; the SE was NaN, now 0.49514353151367907
  (R 0.4951435315136789).
- **`sp.svydesign`: design df collapsed to 1** when PSU ids repeat across
  strata. CI [6.7114, 20.2148] → [12.34201491229616, 14.58418673639966] with
  df 17, equal to R and Stata.
- **`sp.svyglm` binomial and Poisson used the Gaussian bread.** Logit SEs
  [0.1179, 0.0385] against R [0.5098, 0.1731]; Poisson [0.4900, 0.1605]
  against [0.2138, 0.0502]. Now to 3.9e-14.
- **The design effect of a survey mean depended on the weight scale**: DEFF
  204.877 → 3.9032222795694005 (R 3.9032).
- **`sp.rake` stopped on an absolute change in O(1/n) weights**, leaving a
  4.3e-4 margin error at n = 100 000. The rule is now relative margin
  error, `tol=1e-10`; unknown or missing categories raise.
- **`sp.lcsf` / `sp.zisf` stopped about 1e-3 short**, with a fixed-step
  finite-difference Hessian. Newton polish plus a Richardson Hessian:
  distance from sfaR 1.15e-3 → 1.1e-8 (estimates), 1.44e-3 → 8.6e-8 (SEs).
- **`sp.metafrontier(cost=True)` returned TGR ≡ 1**: the sign of the gap was
  wrong. TGR min / mean / max 1 / 1 / 1 → 0.170 / 0.445 / 1.0.
- **`sp.blp` SEs and elasticities.** SEs were too small by √N (linear) and N
  (σ): the Jacobian was not scaled and the joint (β, σ) sandwich was missing.
  Linear SEs (0.00434, 0.00070, 0.00187) → (0.1075, 0.0201, 0.0474); σ SE
  4.99e-5 → 0.0304. Elasticities ignored the random price coefficient
  (`sigma_price` was unused); the error against pyblp was over 1e-2, now
  about 1e-8.

### Fixed

**Survival and epidemiology**

- `sp.breslow_day_test` warns when it drops a stratum with a zero fitted
  cell (it dropped them silently).
- `sp.power_case_control`'s sample-size search also steps downwards, so it
  finds the minimum from above (it only stepped up from the Wald closed
  form).
- `docs/reference/survival.md` no longer advertises `ties="exact"`, which was
  never supported.

**Inference and sensitivity**

- `sp.oster_bounds` / `sp.oster_delta` warn when they fall back because
  `r_max ≤ R̃²` (the fallback was silent).

**Panel and GLMM**

- `sp.panel_compare` records a failing method with
  `WorkflowDegradedWarning` instead of swallowing the exception into an
  "error" cell.
- `sp.lrtest` warns only when the χ̄² mixture is a bound; the old warning
  also fired in the single-added-effect case, where the formula is exact.
- `sp.mixlogit` reports `converged` by the score; `sp.gmm` no longer reports
  `converged=False` at the optimum.
- The GLMM module docstring no longer calls Laplace "Stata meglm default"
  (Stata's default is `mvaghermite`, 7 points); a Notes section maps each
  Stata `intmethod` to `nAGQ`. The gamma docstring's claim that the observed
  information "would lose definiteness when y < μ" is removed.
- `sp.interactive_fe` warns when it drops units with missing cells (it
  dropped them silently).

**Treatment effects**

- `sp.lee_bounds(covariates=)` is still ignored, but now with a
  `UserWarning`.

**DiD and synthetic control**

- `sp.sdid(covariates=...)` raises `MethodIncompatibility`; the covariates
  were ignored.
- `sp.continuous_did(method="att_gt" | "dose_response")` warns that
  `controls` are ignored.
- `sp.mc_synth`: the swallowed `except Exception` in the placebo and CV
  loops is removed; NaN cells of unbalanced panels are treated as unobserved
  instead of reaching the SVD. The citation string gives the correct author
  (Khashayar Khosravi). `sp.mc_panel`'s docstring states the default-lambda
  heuristic the code uses.
- `sp.harvest_did` no longer claims to implement Abadie, Angrist, Frandsen &
  Pischke (2025), a survey chapter that defines no such estimator; the
  docstring describes the Callaway–Sant'Anna building blocks and the
  StatsPAI-specific precision aggregation.
- `sp.causal_impact` warns that `n_seasons` is ignored. Its docstrings no
  longer describe a local-level Bayesian model "equivalent to" R
  `CausalImpact`; it is a regression plus AR(1) state-space model with
  frequentist prediction intervals.
- `sp.scdata` raises on duplicate (unit, time) rows (it averaged them),
  warns when it drops donors with missing pre-period data, and raises on
  missing post-period donor values.
- `sp.ssaggregate(cluster=, alpha=)` were ignored. `cluster` now gives the
  region-cluster row and `alpha` sets the AKM / AKM0 CI level.
- `sp.bartik(robust=...)` used HC1 for any value other than `"nonrobust"`
  (e.g. `"hc3"`); unimplemented values now raise. A zero Rotemberg
  denominator warns and reports NaN instead of all-zero weights.
- `sp.robust_synth` / `sp.demeaned_synth(covariates=...)` raise
  `NotImplementedError`; the argument was ignored.
- `model_info` pre-fit MSPE / RMSE (`robust_synth`, `demeaned_synth`) and
  RMSQE (`discos`) are no longer rounded to 6 decimals.
- `sp.robust_synth(variant="penalized")` is relabelled "Ridge-penalized
  simplex SCM"; it was labelled Abadie & L'Hour (2021).
- Placebo loops in `robust_synth`, `demeaned_synth` and `discos` catch only
  `ValueError` / `LinAlgError`; the dose-response and time-varying-covariate
  bootstraps no longer use blanket `except Exception`.

**RD and IV**

- `sp.rlasso_effect(s)` warns when a target is spanned by the selected
  controls (cps2012 `female:hsd08`: hdm −4.7e13, StatsPAI 1.9e-36); no
  number changes.
- `sp.lasso_iv` is no longer attributed to BCCH (2012) in the docstring and
  `model_info["method"]`, and no longer forms an n×n matrix.
- `sp.ivqreg`'s docstring no longer claims agreement with Stata `ivqreg2`
  or "R quantreg::ivqreg".

**Spatial, survey and structural**

- `sp.spatial_iv`: `alpha` was ignored and the docstring said "Conley-style
  spatial HAC" for what is HC0 (sphet `het=TRUE`). Estimates and SEs are
  unchanged.
- `sp.svydesign`: a single-PSU stratum was handled silently and the
  docstring misstated R's behaviour; it now warns, and `lonely_psu=` offers
  R's rules. Default numbers are unchanged.
- `sp.linear_calibration`: the docstring stated the wrong objective;
  collinear calibration variables now warn. Numbers are unchanged.

### Added

**Survival and epidemiology**

- `sp.cuminc(variance="gray", conf_type="log-log", rho=)`: cmprsk's variance,
  `stcompet`'s bounds, Gray's weight.
- `sp.finegray(vce="robust" | "model", small_sample=)`.
- `sp.cox_frailty(theta=, ties=)`; results carry `lr_theta0` (chibar2),
  `log_frailties` and `loglik_cox`.
- `sp.direct_standardize(ci_method=, variance=)` and `.se`;
  `sp.indirect_standardize(ci_method=)`.
- `sp.roc_curve(se_method="delong" | "hanley-empirical")`;
  `sp.sensitivity_specificity(ci_method="exact")`.
- `sp.kdensity(bw_method="stata" | "nrd0")`; `sp.lpoly(se_method="stata",
  pwidth=)`; `sp.power_case_control(test="chi2")`.
- Parity file `tests/reference_parity/test_survival_epi_R_parity.py` against
  survival, cmprsk, epitools, DescTools, pROC, epiR and Stata.

**Time series**

- `sp.irf(cumulative=, sigma_df=)`; `sp.its(hac_small_sample=)`.
- `sp.granger_causality` returns `chi2` / `chi2_p_value`; `causing` may be a
  list (joint test).
- `sp.structural_break(method="global")` (strucchange `breakpoints` with
  BIC), and `sup_wald` / `sup_break` on sup-F results.
- `sp.cusum_test` returns `statistic`, `p_value` and `boundary_coef`.
- `sp.garch(presample="stata" | "rugarch", vce="oim" | "opg" | "robust")`
  and `garch_loglik()`.
- `sp.panel_unitroot(convention="stata" | "plm", dfcor=, robust=)`.
- Parity file `tests/reference_parity/test_timeseries_R_parity.py` against
  urca, aTSA, vars, strucchange, sandwich, rugarch, plm and Stata.

**Inference and sensitivity**

- `sp.rosenbaum_bounds(zero_method="pratt" | "wilcox")`: `"wilcox"` is
  `rbounds::psens`'s zero handling.
- Wild-bootstrap results carry `enumerated` and `n_boot_requested`;
  `sp.ri_test` carries `exact`; `sp.oster_bounds` carries `method` and
  `beta_adjusted_alternatives`.
- Parity files `test_inference_sens_R_parity.py` /
  `test_inference_sens_stata_parity.py` against sandwich, clubSandwich,
  summclust, fwildclusterboot, ri2, DOS2, rbounds, EValue, robomit and Stata
  `regress` / `jackknife` / `boottest` / `ritest` / `rbounds` / `psacalc`.

**Panel and GLMM**

- `curvature="observed" | "expected"` on `sp.meglm`, `sp.melogit`,
  `sp.mepoisson`, `sp.menbreg`, `sp.megamma`, `sp.meologit`;
  `MEGLMResult.thresholds_se`.
- `sp.xtnbreg(model="ufe" | "normal_re")` for the previous estimators.
- `sp.gmm(sandwich_weight="estimation")` (Stata's two-step weight),
  `sp.absorb_ols(cluster_df="min")` (reghdfe), `sp.interactive_fe(dof=)`,
  `sp.mixlogit(halton_burn=, halton_shift=, small_sample=)`.
- Seven parity files `tests/reference_parity/test_panel_*_parity.py` against
  lme4, glmmTMB, ordinal, performance, psych, pglm, phtt, gmm and Stata.

**Treatment effects**

- `sp.aipw(cross_fit=, se_method=)`: `cross_fit=False` fits the nuisances
  on the full sample and `se_method="sandwich"` reproduces Stata
  `teffects aipw`; `model_info["potential_outcome_means"]` and `_se`.
- `sp.multi_treatment(outcome_model="linear",
  se_method="influence" | "sandwich")`: Stata `teffects aipw` with a
  multivalued treatment.
- `sp.stabilized_weights(density_sd=)`, `sp.msm(density_sd=)`: `"ml"`
  reproduces `ipw::ipwtm(family="gaussian")`.
- `sp.tmle(q_bound=)`: 5e-4 reproduces `tmle::tmle`.
- `sp.lee_bounds(se_method="analytic", trimming="quantile" | "exact")`;
  `trimming=` on `sp.principal_strat` and
  `sp.survivor_average_causal_effect`.
- `sp.g_estimation(propensity_model="logit" | "linear")`.
- `sp.four_way_decomposition(vce="ols" | "ml")` with delta-method SEs in
  `result.se` (CMAverse / `med4way`).
- `sp.manski_bounds(assumption="mts_mtr")`.
- `sp.ltmle` supports censoring nodes as R `ltmle` does.
- Parity file `tests/reference_parity/test_teffects_R_parity.py`.

**DiD and synthetic control**

- Private `statspai.did._twfe_weights.dcdh_fe_weights`, bit-exact with
  `TwoWayFEWeights::twowayfeweights(type="feTR")`. `sp.twfe_decomposition`
  does not use it yet (see below).
- Alias proofs for `sp.bjs`, `sp.borusyak_jaravel_spiess`, `sp.did_2stage`
  and `sp.synthdid_estimate`.
- `sp.did_timevarying_covariates(aggregation="group" | "simple")`;
  `model_info` carries `att_group` and `att_simple`.
- `continuous_did` results carry `model_info["dof_K"]`, `["control_arm"]`,
  `["dose_response_pointwise_se"]` and `["se_method"]`.
- `sp.mc_panel` / `sp.mc_synth(fixed_effects="two-way" | "unit" | "time" |
  "none")`; `model_info` keys `lambda_mcpanel`, `lambda_fect`,
  `low_rank_matrix`, `fixed_effects_matrix`, `converged`, `n_iter`. Both
  share the solver in `statspai.matrix_completion._core`.
- `sp.scest(Q=, Q2=)` and `w_constr="L1-L2"`; `sp.scpi(sims, u_missp,
  u_sigma, u_order, u_alpha, e_order, e_alpha, rho, rho_max, Q, Q2, draws)`
  with `model_info` keys `bounds`, `CI`, `rho`, `Sigma`, `u_var`, `e_mean`,
  `e_var`, `df`, `vsig`, `failed_sims`; `draws` takes R's standard normals.
  `sp.scdata` also returns R-named `A`, `B`, `C`, `P`, `J`, `KM`, `T0`,
  `T1`.
- `sp.ssaggregate` returns the Borusyak–Hull–Jaravel shock-level data set in
  `data_info["shock_data"]` and the shock-level IV coefficient and HC0 SE
  (= Stata / R `ssaggregate` then `ivreg2, robust`), plus every
  `ShiftShareSE` row (Homoscedastic, EHW, region cluster, AKM, AKM0 with the
  inverted CI).
- `sp.bartik` exposes Rotemberg weights in `model_info["rotemberg_weights"]`
  with the per-industry just-identified `beta_k`.
- `sp.staggered_synth(nu=, fixedeff=, n_leads=, n_lags=,
  se_method="jackknife")` and `model_info["event_study"]`, `["weights"]`,
  `["nu"]`, `["global_l2"]`, `["ind_l2"]`, `["jackknife_atts"]`.
- `sp.discos(M=, simplex=, q_nodes=, cdf_grid=)` and DiSCo's permutation
  test in `model_info["permutation"]`.
- `model_info["mspe_ratio"]` / `["placebo_mspe_ratios"]` in
  `demeaned_synth` / `robust_synth`.
- Parity files `test_did_synth_R_parity.py` and
  `test_did_synth_{didvar,mc,misc,scpi,shiftshare,synthvar}_parity.py`.

**RD and IV**

- `sp.weakrobust(clr_method=)`, `conditional_lr_test(method=)`,
  `sp.conditional_lr_ci(method=)`.
- `sp.mccrary_test(bin_width=)`; `sp.rdhte(q=, vce=)`;
  `sp.rdbwhte(q=, bwselect=, vce=, cluster=)`; `sp.rdhte_lincom(linfct=)`.
- `sp.rkd` adds the robust bias-corrected row in `model_info["robust"]`;
  `sp.rd_bias_aware_fuzzy` returns RDHonest's own interval in
  `model_info["bias_aware"]["rdhonest"]`.
- `fig.rdplot_data` / `fig.rdplotdensity_data` carry the plotted numbers.
- Parity files `test_rd_iv_R_parity.py` and `test_rd_iv_rd_R_parity.py`.

**Spatial, survey and structural**

- `sp.mgwr(bws=)`; `sp.sarar_gmm(sig2n_k=, w_lags=)` (`w_lags=1` is PySAL
  `GM_Combo`); `sp.spatial_panel(vce="information" | "oim")` (`"oim"` =
  Stata `xsmle`).
- `sp.gwr` results carry `cv`, `influence` and `tr_StS`; the `sp.spatial_iv`
  coefficient table gains `z`, `p`, `ci_lower` and `ci_upper`.
- `sp.svydesign(lonely_psu=)`, `svymean` / `svytotal` `deff=`,
  `sp.svyglm(dof=)`.
- `sp.metafrontier(envelope="own" | "all")`; the default `"all"` is
  unchanged and `"own"` equals R `metafrontier`.
- Parity fixtures against spdep, GWmodel, spatialreg, sphet, splm, Stata
  `xsmle`, R `survey` and Stata `svy`, sfaR, `markupest`, `metafrontier` and
  pyblp.

### Changed

- `sp.panel_unitroot`: IPS and Hadri with `trend="n"` raise.
- `sp.oster_bounds` / `sp.oster_delta`: `r_max > 1` raises in
  `oster_bounds`; in `oster_delta` a value above 1 is a multiplier.
- `sp.xtnbreg` accepts `model="fe" | "re" | "pooled" | "ufe" | "normal_re"`.
- `sp.interactive_fe(method="pca")` issues a `DeprecationWarning`; it was
  identical to `"iterative"`.
- `sp.manski_bounds`: a refuted `"mts_mtr"` raises instead of reporting
  swapped endpoints.
- `sp.scest` / `sp.scpi`: `lasso_lambda` and `ridge_lambda` are ignored with
  a `DeprecationWarning`; use `Q=` / `Q2=`. `sp.scpi`'s `period_results`
  drops `in_sample_var` / `out_sample_var` and adds `joint_lower` /
  `joint_upper`.
- `sp.staggered_synth` raises on non-absorbing treatment, unbalanced
  panels, first-period adopters and cohorts without an eligible donor. Its
  QP is solved by an exact active set instead of SLSQP.
- `sp.discos` on aggregate panels (one row per unit-period) warns that it is
  not the Gunsilius estimator and labels
  `model_info["estimator"] = "time_series_quantiles_fallback"`.
- `sp.rdrobust(p=None)`; `sp.rdplot(kernel="uniform")`; `sp.rdplot` with
  fewer than 20 observations and `sp.rdplotdensity` with a side too sparse
  for rddensity raise, as R does; `sp.rdbwhte` returns a DataFrame for
  subgroups; `sp.mccrary_test`'s `model_info["n_bins"]` is the number of
  histogram cells.
- `sp.gwr_bandwidth`: default search bounds are GWmodel's (`[20, n]` /
  `[D/5000, D]`) and `tol` defaults to 1e-4.

**Not fixed in this release.** `sp.twfe_decomposition` is not the
decomposition it names: its headline (−0.02718 on mpdta) is not the TWFE β
(−0.03751), its Bacon "earlier vs later" rows use the wrong window (2004 vs
2006: −0.0148 against −0.0327), all weights are 1/9, and its "dCDH" weights
are 7 positive cohort×period rows where `TwoWayFEWeights` gives 875 cell
weights, 125 of them negative. Its treated-vs-never rows and
`model_info["twfe_beta"]` are correct. A strict xfail pins the gaps.

### Reference-implementation findings

Places where the reference is wrong or not unique. StatsPAI does not copy
them; each is reconstructed or asserted in the parity test.

- **statsmodels `tau_2010s`** (Engle–Granger critical values) has two typos
  against the paper: N=2 1% β₂ −33.527 (paper −22.527) and N=3 5% β₁ −8.5632
  (paper −8.5631). StatsPAI transcribes the paper.
- **Stata `_xturllc.ado`**, model 3: σ* = .971 at T = 40, where plm has .871
  and the column is otherwise monotone (.906, .871, .842). The test rebuilds
  Stata's t* from StatsPAI's quantities with Stata's value. `_xturips.ado`'s
  trend-mean at lag 8, T = 60 is −2.204 where plm has −2.024; that cell is
  not exercised.
- **rugarch** SEs differ by about 0.7%: at identical parameters StatsPAI's
  Hessian agrees with Stata's to 4e-6 and with rugarch's numerical Hessian
  only to about 0.7%.
- **Stata `boottest` CI**: its Chandrupatla search returns early on the step
  function; both reported endpoints are values its own test rejects
  (p = 204/4096, 202/4096).
- **Stata `leebounds` 1.5** keeps its trimming threshold in a 15-digit local
  macro, so its tie branch never runs and its lower bound drops the quantile
  observation. With the threshold held exactly (`%21x`) it matches
  `trimming="exact"` on both bounds.
- **R `AIPW` 0.6.9.3 ATT** divides its control term by P(A = 0); the test
  rebuilds that number from the same nuisances, and the DR ATT is
  cross-checked against DoubleML's ATTE score.
- **Stata `me*` with `mcaghermite`**: `mepoisson` / `menbreg` / `meglm`
  gamma estimates sit 3e-5 / 1e-6 / 1e-5 from the optimum of Stata's own
  objective. Stata's reported `e(V)` for non-canonical Laplace GLMMs is not
  the Hessian of its own objective at its own estimates (NB-2 4.5e-4, gamma
  5.5e-3, ordinal 5e-5; grade C, not reconstructed); for AGHQ it is
  reproduced by a fixed-node Hessian (grade B). SEs are certified against
  glmmTMB's AD Hessian instead.
- **Stata `regife`**: without the `require` package, `reghdfe` errors with
  r(9) and `regife` silently posts the pooled-OLS starting values as its
  estimate (the IFE estimate only in `e(bend)`).
- **phtt `sig2.hat`** demeans residuals unit by unit although the model has
  no unit effect; reconstructed in the test.
- **DiSCos 0.1.4 `DiSCo_per_iter`** fills each placebo's quantile matrix
  from column 2, so column 1 stays zero and every placebo counterfactual in
  quantile mode drops a weighted donor. Patched and unpatched distances
  differ by up to 99.6%; the p-value (1/6) happens to agree on the fixture.
- **ptetools 1.0.1** loses `control_group` when the cohort column is named
  `g` (non-standard evaluation); **did 2.3.0** drops never-treated units when
  the cohort column is an integer; **didFF 0.1.0 `distDD`** crashes when a
  bin's influence function is dropped (`balance_e = 1`).
- **HonestDiD** uses a Monte-Carlo `.qfoldednormal` quantile, and its
  derivative bisection over h stops at a step of (h0 − hMin)/100: FLCI bounds
  agree with the exact-quantile rerun only to about 2e-5 while the
  half-length agrees to 1e-8. `breakdown_m` against shipped HonestDiD is
  aligned at 4.3e-4.
- **R `scpi` 4.0.1**, simplex: CLARABEL leaves Norway at 2.0e-6, above the
  1e-6 threshold used to count active donors, so R gets df = 6 where the
  exact optimum gives 5 and R's rho is 8% larger. R's own L1-L2 fit reaches
  the same optimum and reproduces StatsPAI's rho to 3e-9.
- **ivDiag 1.0.6** effective F with more than one instrument uses the
  un-partialled Z'Z and differs by 13–16% from Stata `weakivtest` and
  StatsPAI.
- **GWmodel** local R² under an adaptive kernel applies the transposed
  weight matrix (reproduced exactly in the test). Its default C++ path
  silently ignores `bw.seled` (noted for the `mgwr` bandwidth search, which
  stays unpinned).
