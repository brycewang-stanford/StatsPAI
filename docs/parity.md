# Cross-language parity matrix

> **Auto-generated — do not hand-edit.** Regenerate with `python scripts/build_parity_index.py`. Every row traces to a committed test artifact; nothing here is asserted from memory.

StatsPAI's promise is *numerical alignment with Stata / R*. This page makes that promise auditable function-by-function. Query any function programmatically:

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

| status | functions |
| --- | ---: |
| bit-exact | 144 |
| aligned | 16 |
| analytical-only | 225 |
| external-replication | 6 |
| **verified (subtotal)** | **391** |
| unverified | 775 |
| **total registered** | **1166** |

## bit-exact — 144 functions

Machine-tolerance agreement with a named R/Stata reference.

| function | reference | versions | tolerance | rel err (R / Stata) | test |
| --- | --- | --- | --- | --- | --- |
| `adjust_pvalues` | base R stats::p.adjust (bonferroni/holm/BH) | R 4.5.2 | exact (atol 1e-15; observed 0) | — / — | [`test_mht_parity.py`](../tests/reference_parity/test_mht_parity.py) (+1) |
| `arima` | stats::arima | R 4.5.2; stats 4.5.2 | rel_est<=1e-06, rel_se<=1e-06 | 7.4e-07 / 9.3e-09 | [`39_arima.py`](../tests/r_parity/39_arima.py) (+2) |
| `attributable_risk` | base-R closed form (attributable fraction exposed + PAF) | R 4.5.2 | AFE + PAF point estimates 1e-12 abs (observed 0); CI not pinned | — / — | [`test_epi_extra_parity.py`](../tests/reference_parity/test_epi_extra_parity.py) (+1) |
| `auc` | Mann-Whitney rank AUC (= pROC::auc / sklearn) | R 4.5.2 | AUC 1e-12 abs (observed 0) | — / — | [`test_auc_parity.py`](../tests/reference_parity/test_auc_parity.py) |
| `bacon_decomposition` | bacondecomp::bacon | R 4.5.2; bacondecomp 0.1.1 | rel_est<=1e-06, rel_se<=1e-06 | 5.6e-16 / 9.6e-09 | [`20_bacon.py`](../tests/r_parity/20_bacon.py) (+2) |
| `balance_panel` | base R counts == n_periods | R 4.5.2 | rel_est<=1e-06, rel_se<=1e-06 | 0 / 0 | [`69_balance_panel.py`](../tests/r_parity/69_balance_panel.py) (+2) |
| `benjamini_hochberg` | base R stats::p.adjust(method='BH') | R 4.5.2 | exact (atol 1e-15; observed 0) | — / — | [`test_mht_parity.py`](../tests/reference_parity/test_mht_parity.py) (+1) |
| `betareg` | betareg::betareg(link.phi="log") | R 4.5.2; betareg 3.2.4 | rel_est<=1e-06, rel_se<=0.01 | 2.2e-08 / 3.1e-08 | [`61_betareg.py`](../tests/r_parity/61_betareg.py) (+2) |
| `betweenness_centrality` | Freeman betweenness centrality (shortest-path mediation) | R 4.5.2 | centrality 1e-12 abs (observed 0) | — / — | [`test_network_centrality_parity.py`](../tests/reference_parity/test_network_centrality_parity.py) |
| `biprobit` | R VGAM::vglm(binom2.rho) bivariate probit | R 4.5.2; VGAM 1.1.14 | coef / rho 1e-6 abs (observed <= 2e-7); logLik 1e-6 rel | — / — | [`test_biprobit_parity.py`](../tests/reference_parity/test_biprobit_parity.py) (+1) |
| `bonferroni` | base R stats::p.adjust(method='bonferroni') | R 4.5.2 | exact (atol 1e-15; observed 0) | — / — | [`test_mht_parity.py`](../tests/reference_parity/test_mht_parity.py) (+1) |
| `bootstrap` | nonparametric bootstrap contract (Efron 1979) | R 4.5.2 | estimate/se contract 1e-12 abs (observed 0); SE ~ analytic 10% | — / — | [`test_bootstrap_parity.py`](../tests/reference_parity/test_bootstrap_parity.py) |
| `breakdown_frontier` | additive-violation breakdown identities (Masten & Poirier 2021) | R 4.5.2 | breakdown point / CI / bounds 1e-12 abs (observed 0) | — / — | [`test_breakdown_frontier_parity.py`](../tests/reference_parity/test_breakdown_frontier_parity.py) |
| `callaway_santanna` | did::att_gt + aggte | R 4.5.2; did 2.3.0 | rel_est<=1e-06, rel_se<=0.01 | 1.3e-15 / 1.3e-15 | [`04_csdid.py`](../tests/r_parity/04_csdid.py) (+2) |
| `cgs_continuous_did` | contdid::cont_did | R 4.5.2 | rel_est<=1e-06 | 2.4e-14 / — | [`80_contdid.py`](../tests/r_parity/80_contdid.py) (+1) |
| `clogit` | survival::clogit | R 4.5.2; survival 3.8.3 | rel_est<=1e-06, rel_se<=1e-06 | 1.3e-08 / 1.3e-08 | [`46_clogit.py`](../tests/r_parity/46_clogit.py) (+2) |
| `clustering` | local clustering coefficient (Watts-Strogatz 1998) | R 4.5.2 | coefficient 1e-12 abs (observed 0) | — / — | [`test_network_centrality_parity.py`](../tests/reference_parity/test_network_centrality_parity.py) |
| `cohen_kappa` | base-R closed form (Cohen's kappa point estimate) | R 4.5.2 | kappa + agreements 1e-12 abs (observed ~1e-16); SE not pinned | — / — | [`test_epi_extra_parity.py`](../tests/reference_parity/test_epi_extra_parity.py) (+1) |
| `contrast` | treatment-contrast identity (= Stata margins, contrast(r)) | R 4.5.2 | contrast == dummy coefficient 1e-12 abs (observed <= 1e-15) | — / — | [`test_contrast_pwcompare_parity.py`](../tests/reference_parity/test_contrast_pwcompare_parity.py) |
| `cox` | survival::coxph | R 4.5.2; survival 3.8.3 | rel_est<=1e-06, rel_se<=1e-06 | 8.4e-16 / 2.1e-10 | [`24_coxph.py`](../tests/r_parity/24_coxph.py) (+2) |
| `cr2_se` | clubSandwich::vcovCR(type="CR2"/"CR3") | R 4.5.2; clubSandwich 0.6.2 | rel_est<=1e-06, rel_se<=1e-06 | 1.8e-08 / 2.2e-08 | [`53_cr2.py`](../tests/r_parity/53_cr2.py) (+2) |
| `das_gupta` | Das Gupta (1993) exact standardization decomposition identity | R 4.5.2 | factor-effect sum + pct 1e-12 abs (observed 0) | — / — | [`test_dasgupta_parity.py`](../tests/reference_parity/test_dasgupta_parity.py) |
| `ddd` | Stata 18 MP regress [aw=w], robust (aweight HC1) | Stata 18 MP | b / se 1e-12 abs (observed <= 3e-15) | — / — | [`test_did2x2_ddd_weighted_robust_parity.py`](../tests/reference_parity/test_did2x2_ddd_weighted_robust_parity.py) |
| `ddd_heterogeneous` | triplediff::ddd + agg_ddd | R 4.5.2 | rel_est<=1e-06, rel_se<=1e-06 | 5.7e-15 / — | [`77_ddd.py`](../tests/r_parity/77_ddd.py) (+1) |
| `decompose` | oaxaca::oaxaca | R 4.5.2; oaxaca 0.1.5 | rel_est<=1e-06, rel_se<=0.05 | 6.3e-16 / 1.3e-16 | [`30_oaxaca.py`](../tests/r_parity/30_oaxaca.py) (+2) |
| `degree_centrality` | Freeman normalized degree centrality (deg_i / (n-1)) | R 4.5.2 | centrality 1e-12 abs (observed 0) | — / — | [`test_network_centrality_parity.py`](../tests/reference_parity/test_network_centrality_parity.py) |
| `demean` | textbook mean-within (algorithmic) | R 4.5.2 | rel_est<=1e-06, rel_se<=1e-06 | 3.5e-15 / 8.8e-15 | [`68_demean_within.py`](../tests/r_parity/68_demean_within.py) (+2) |
| `dfl_decompose` | ddecompose::dfl_decompose | R 4.5.2; ddecompose 1.0.0 | rel_est<=1e-06, rel_se<=1e-06 | 1.2e-09 / 1.8e-13 | [`31_dfl.py`](../tests/r_parity/31_dfl.py) (+2) |
| `did_2x2` | Stata 18 MP regress [aw=w], robust (aweight HC1) | Stata 18 MP | b / se 1e-12 abs (observed <= 3e-16) | — / — | [`test_did2x2_ddd_weighted_robust_parity.py`](../tests/reference_parity/test_did2x2_ddd_weighted_robust_parity.py) |
| `did_imputation` | didimputation::did_imputation | R 4.5.2; didimputation 0.5.1 | rel_est<=1e-06, rel_se<=1e-06 | 4.8e-08 / 3.5e-07 | [`16_bjs.py`](../tests/r_parity/16_bjs.py) (+2) |
| `did_multiplegt` | DIDmultiplegt::did_multiplegt (archived 0.1.4) | R 4.5.2 | rel_est<=1e-06 | 3.9e-15 / 3.2e-15 | [`81_didm.py`](../tests/r_parity/81_didm.py) (+2) |
| `did_multiplegt_dyn` | DIDmultiplegtDYN::did_multiplegt_dyn | R 4.5.2 | rel_est<=1e-06 | 3.3e-15 / 2.1e-15 | [`78_multiplegt_dyn.py`](../tests/r_parity/78_multiplegt_dyn.py) (+2) |
| `direct_standardize` | base closed form (directly standardized rate; = Stata dstdize) | R 4.5.2 | DSR 1e-12 abs (observed 0) | — / — | [`test_standardize_parity.py`](../tests/reference_parity/test_standardize_parity.py) |
| `dml` | DoubleML::DoubleMLPLR | R 4.5.2; DoubleML 1.0.2 | rel_est<=1e-10, rel_se<=1e-10 | 0 / 3.7e-15 | [`08_dml.py`](../tests/r_parity/08_dml.py) (+2) |
| `dml_sensitivity` | doubleml (Python) DoubleML.sensitivity_analysis | — | bias_bound and adjusted theta bounds 1e-12 (observed 2.5e-15); RV 1e-6 (observed 9.2e-8); RVa is a documented convention gap (<5e-3, observed 1.4e-3) because StatsPAI exhausts |theta|-z*se with the unadjusted SE while doubleml lets the SE move with the confounding scenario | — / — | [`test_dml_sensitivity_parity.py`](../tests/external_parity/test_dml_sensitivity_parity.py) |
| `drdid` | DRDID::drdid_imp_panel | R 4.5.2; DRDID 1.2.3 | rel_est<=1e-06, rel_se<=1e-06 | 2.6e-15 / 2.2e-16 | [`38_drdid.py`](../tests/r_parity/38_drdid.py) (+2) |
| `ebalance` | ebal::ebalance 0.2.1 (Hainmueller 2012) | — | ATT rel <= 1e-5 (observed 3.2e-7); moment gap <= 1e-10 | — / — | [`test_matching_r_parity.py`](../tests/reference_parity/test_matching_r_parity.py) (+1) |
| `eigenvector_centrality` | leading adjacency eigenvector (Bonacich 1972) | R 4.5.2 | centrality 1e-9 abs (observed <= 1e-15) | — / — | [`test_network_centrality_parity.py`](../tests/reference_parity/test_network_centrality_parity.py) |
| `etwfe` | etwfe::etwfe + emfx | R 4.5.2; etwfe 0.6.2 | rel_est<=1e-06, rel_se<=0.001 | 1.8e-13 / 3.9e-14 | [`17_etwfe.py`](../tests/r_parity/17_etwfe.py) (+2) |
| `etwfe_emfx` | etwfe::etwfe + emfx | R 4.5.2; etwfe 0.6.2 | rel_est<=1e-06, rel_se<=0.001 | 1.8e-13 / 3.9e-14 | [`17_etwfe.py`](../tests/r_parity/17_etwfe.py) (+2) |
| `evalue` | EValue::evalues.RR | R 4.2.3; EValue 4.1.4 | rel_est<=1e-06, rel_se<=1e-06 | 5.8e-14 / 1.2e-16 | [`23_evalue.py`](../tests/r_parity/23_evalue.py) (+2) |
| `evalue_rr` | VanderWeele-Ding closed form (= R EValue package) | R 4.5.2 | point + CI E-value 1e-12 abs (observed 0) | — / — | [`test_evalue_rr_parity.py`](../tests/reference_parity/test_evalue_rr_parity.py) |
| `feglm` | fixest::feglm (family="logit") / fixest::fepois | R 4.5.2; fixest 0.14.0 | rel_est<=1e-06, rel_se<=5e-05 | 9.7e-09 / 1.8e-09 | [`67_panel_glm.py`](../tests/r_parity/67_panel_glm.py) (+2) |
| `feols` | fixest::feols | R 4.5.2; fixest 0.14.0 | rel_est<=1e-06, rel_se<=1e-06 | 5.2e-15 / 2.9e-15 | [`03_hdfe.py`](../tests/r_parity/03_hdfe.py) (+2) |
| `fepois` | fixest::feglm (family="logit") / fixest::fepois | R 4.5.2; fixest 0.14.0 | rel_est<=1e-06, rel_se<=5e-05 | 9.7e-09 / 1.8e-09 | [`67_panel_glm.py`](../tests/r_parity/67_panel_glm.py) (+2) |
| `fracreg` | stats::glm(quasibinomial('logit')) [fractional response] | R 4.5.2 | coefficients 1e-10 abs (observed ~8e-15) | — / — | [`test_glm_ext_parity.py`](../tests/reference_parity/test_glm_ext_parity.py) (+1) |
| `frontier` | sfaR::sfacross | R 4.5.2; sfaR 1.0.1 | rel_est<=1e-06, rel_se<=5e-05 | 4.1e-08 / 4.0e-08 | [`28_frontier.py`](../tests/r_parity/28_frontier.py) (+2) |
| `g_computation` | base R stats::lm g-formula standardization (Robins 1986) | — | psi 1e-8 (observed <= 7e-16; bootstrap SE pinned loosely +/-25%) | — / — | [`test_gformula_parity.py`](../tests/reference_parity/test_gformula_parity.py) (+1) |
| `gardner_did` | did2s::did2s | R 4.5.2 | rel_est<=1e-06 | 4.8e-08 / 2.4e-12 | [`73_did2s.py`](../tests/r_parity/73_did2s.py) (+2) |
| `gelbach` | Gelbach (2016) exact conditional decomposition identity | R 4.5.2 | total_change + contribution sum 1e-12 abs (observed 0) | — / — | [`test_gelbach_parity.py`](../tests/reference_parity/test_gelbach_parity.py) |
| `glm` | base R stats::glm (binomial logit + Poisson log) | R 4.5.2 | coef / logLik / AIC 1e-8 abs (observed <= 5e-13); SE ~1e-3 rel | — / — | [`test_glm_parity.py`](../tests/reference_parity/test_glm_parity.py) (+1) |
| `gsynth` | gsynth::gsynth | R 4.5.2; gsynth 1.4.0 | rel_est<=1e-06, rel_se<=1e-06 | 7.7e-14 / — | [`19_gsynth.py`](../tests/r_parity/19_gsynth.py) (+1) |
| `heckman` | sampleSelection::heckit | R 4.5.2; sampleSelection 1.2.14 | rel_est<=1e-06, rel_se<=0.0005 | 1.0e-11 / 1.0e-11 | [`43_heckman.py`](../tests/r_parity/43_heckman.py) (+2) |
| `het_test` | lmtest::bptest (studentized Breusch-Pagan) | R 4.5.2; lmtest 0.9.40 | statistic & p-value 1e-10 rel (observed ~1e-13) | — / — | [`test_diagnostics_parity.py`](../tests/reference_parity/test_diagnostics_parity.py) (+1) |
| `holm` | base R stats::p.adjust(method='holm') | R 4.5.2 | exact (atol 1e-15; observed 0) | — / — | [`test_mht_parity.py`](../tests/reference_parity/test_mht_parity.py) (+1) |
| `honest_did` | HonestDiD::createSensitivityResults_relativeMagnitudes | R 4.5.2; HonestDiD 0.2.8 | abs_est<=1e-06, abs_se<=1e-06 | 4.4e-16 / 5.6e-17 | [`21_honest_relmags.py`](../tests/r_parity/21_honest_relmags.py) (+2) |
| `hurdle` | pscl::hurdle(dist='poisson', zero.dist='binomial') | R 4.5.2; pscl 1.5.9 | count + zero coefficients 1e-6 abs (observed ~2e-8) | — / — | [`test_glm_ext_parity.py`](../tests/reference_parity/test_glm_ext_parity.py) (+1) |
| `icc` | variance-ratio identity ICC = var_u / (var_u + var_e) | R 4.5.2 | ICC vs model variance components 1e-12 abs (observed 0) | — / — | [`test_icc_parity.py`](../tests/reference_parity/test_icc_parity.py) |
| `incidence_rate_ratio` | base-R closed form (rate ratio + conditional-binomial exact CI) | R 4.5.2 | estimate 1e-12; exact CI 1e-10 abs (observed ~3e-15) | — / — | [`test_epi_parity.py`](../tests/reference_parity/test_epi_parity.py) (+1) |
| `indirect_standardize` | base closed form (SMR / indirect std.; = Stata istdize) | R 4.5.2 | expected + SMR 1e-12 abs (observed 0) | — / — | [`test_standardize_parity.py`](../tests/reference_parity/test_standardize_parity.py) |
| `inequality_index` | base-R closed form (Gini/Theil-T/Theil-L/Atkinson; = ineq) | R 4.5.2 | all indices 1e-12 abs (observed ~2e-16) | — / — | [`test_inequality_parity.py`](../tests/reference_parity/test_inequality_parity.py) (+1) |
| `ipw` | base R stats::glm(binomial) + hand-rolled Hajek weighted means | — | Hajek ATE/ATT estimate 1e-9 (observed <= 2e-15; SE not pinned) | — / — | [`test_ipw_parity.py`](../tests/reference_parity/test_ipw_parity.py) (+1) |
| `ivreg` | AER::ivreg | R 4.5.2; AER 1.2.16 | rel_est<=1e-06, rel_se<=1e-06 | 1.1e-11 / 1.1e-11 | [`02_iv.py`](../tests/r_parity/02_iv.py) (+2) |
| `kaplan_meier` | survival::survfit | R 4.5.2; survival 3.8.3 | S(t) at every event time 1e-12 (observed ~3e-17); median exact | — / — | [`test_survival_km_parity.py`](../tests/reference_parity/test_survival_km_parity.py) (+1) |
| `kdensity` | Gaussian KDE closed form (= stats::density / sklearn) | R 4.5.2 | density 1e-12 abs (observed ~3e-18 / 0) | — / — | [`test_kdensity_parity.py`](../tests/reference_parity/test_kdensity_parity.py) |
| `kitagawa_decompose` | Kitagawa (1955) two-factor rate decomposition identity | R 4.5.2 | gap = rate + composition + interaction 1e-12 abs (observed 0) | — / — | [`test_kitagawa_decompose_parity.py`](../tests/reference_parity/test_kitagawa_decompose_parity.py) |
| `lee_bounds` | Lee (2009) trimming-bound closed form (lee2009training) | R 4.5.2 | bounds / trim fraction / retention 1e-12 abs (observed <= 1e-16) | — / — | [`test_lee_bounds_parity.py`](../tests/reference_parity/test_lee_bounds_parity.py) |
| `liml` | ivmodel::LIML | R 4.5.2; ivmodel 1.9.1 | rel_est<=1e-06, rel_se<=1e-06 | 1.7e-15 / 3.0e-16 | [`59_liml.py`](../tests/r_parity/59_liml.py) (+2) |
| `local_projections` | lpirfs::lp_lin | R 4.5.2; lpirfs 0.2.5 | rel_est<=1e-06, rel_se<=1e-06 | 5.0e-15 / 4.4e-15 | [`34_lp.py`](../tests/r_parity/34_lp.py) (+2) |
| `logit` | stats::glm(family=binomial("logit")) | R 4.5.2; stats 4.5.2 | rel_est<=1e-06, rel_se<=1e-06 | 2.7e-11 / 2.7e-11 | [`57_logit.py`](../tests/r_parity/57_logit.py) (+2) |
| `logrank_test` | survival::survdiff | R 4.5.2; survival 3.8.3 | chi-square 1e-10 rel (observed ~8e-16); p-value 1e-10 abs | — / — | [`test_survival_km_parity.py`](../tests/reference_parity/test_survival_km_parity.py) (+1) |
| `lrtest` | likelihood-ratio identity chi2 = 2*(logL_full - logL_restricted) | R 4.5.2 | chi2 / logL fields 1e-10 abs (observed 0); p == chi2.sf exact | — / — | [`test_lrtest_parity.py`](../tests/reference_parity/test_lrtest_parity.py) |
| `manski_bounds` | Manski (1990) no-assumption worst-case ATE bound identity | R 4.5.2 | width == y_upper - y_lower 1e-12 abs (observed 0) | — / — | [`test_manski_bounds_parity.py`](../tests/reference_parity/test_manski_bounds_parity.py) |
| `mantel_haenszel` | base-R closed form (Robins-Breslow-Greenland MH; = epiR) | R 4.5.2 | estimate, se_log, CI 1e-12 abs (observed 0) | — / — | [`test_epi_parity.py`](../tests/reference_parity/test_epi_parity.py) (+1) |
| `margins_at` | predictive-margin linear form of the OLS coefficients | R 4.5.2 | margin(x=v) 1e-10 abs (observed 0); symmetric-grid SE symmetric | — / — | [`test_margins_at_parity.py`](../tests/reference_parity/test_margins_at_parity.py) |
| `match` | MatchIt::matchit 4.7.2 (nearest, glm/logit distance) | — | 1:1 and 2:1 PS matching without replacement: rel <= 1e-9. Mahalanobis metric pinned against MatchIt:::mahalanobis_dist (rel <= 1e-10); greedy m_order='data'/'closest' rel <= 1e-9. | — / — | [`test_matching_r_parity.py`](../tests/reference_parity/test_matching_r_parity.py) (+1) |
| `mde` | base-R closed form (RCT minimum detectable effect) | R 4.5.2 | effect size 1e-6 abs (output rounded to 6 dp; observed ~2e-8) | — / — | [`test_power_extra_parity.py`](../tests/reference_parity/test_power_extra_parity.py) (+1) |
| `mediate` | mediation::mediate | R 4.5.2; mediation 4.5.1 | rel_est<=1e-06, rel_se<=0.1 | 6.7e-15 / 3.6e-15 | [`36_mediation.py`](../tests/r_parity/36_mediation.py) (+2) |
| `mediate_interventional` | interventional effects telescoping identity (VanderWeele Vansteelandt Robins 2014) | R 4.5.2 | total = IIE + IDE 1e-12 abs (observed 0) | — / — | [`test_mediate_interventional_parity.py`](../tests/reference_parity/test_mediate_interventional_parity.py) |
| `mediation` | mediation::mediate | R 4.5.2; mediation 4.5.1 | rel_est<=1e-06, rel_se<=0.1 | 6.7e-15 / 3.6e-15 | [`36_mediation.py`](../tests/r_parity/36_mediation.py) (+2) |
| `mediation_decompose` | natural-effects mediation (Pearl 2001; VanderWeele 2015) | R 4.5.2 | total = NDE + NIE 1e-12 abs (observed 0) | — / — | [`test_mediation_decompose_parity.py`](../tests/reference_parity/test_mediation_decompose_parity.py) |
| `melogit` | lme4::glmer(nAGQ=8) | R 4.5.2; lme4 2.0.1 | rel_est<=1e-06, rel_se<=0.05 | 2.4e-07 / 8.4e-07 | [`27_glmm_aghq.py`](../tests/r_parity/27_glmm_aghq.py) (+2) |
| `metalearner` | econml.metalearners SLearner / TLearner / XLearner | — | S / T / X conditional-average-treatment-effect vectors match econml elementwise to 1e-12 absolute (observed <= 1.1e-15) when both fitting stages use the same base learner | — / — | [`test_metalearner_econml_parity.py`](../tests/external_parity/test_metalearner_econml_parity.py) |
| `mixed` | lme4::lmer | R 4.5.2; lme4 2.0.1 | rel_est<=1e-06, rel_se<=1e-06 | 1.3e-10 / 4.9e-11 | [`25_lmm.py`](../tests/r_parity/25_lmm.py) (+2) |
| `mlogit` | nnet::multinom | R 4.5.2; nnet 7.3.20 | rel_est<=1e-06, rel_se<=5e-05 | 2.6e-07 / 7.4e-09 | [`44_mlogit.py`](../tests/r_parity/44_mlogit.py) (+2) |
| `mr` | inverse-variance-weighted MR closed form (Burgess et al. 2013) | R 4.5.2 | estimate / se / Cochran Q 1e-10 abs (observed 0) | — / — | [`test_mr_ivw_parity.py`](../tests/reference_parity/test_mr_ivw_parity.py) |
| `multiway_cluster_vcov` | sandwich::vcovCL(cluster=~g1+g2+g3) | R 4.5.2; sandwich 3.1.1 | rel_est<=1e-06, rel_se<=1e-06 | 2.1e-15 / 2.1e-15 | [`56_multiway_cluster.py`](../tests/r_parity/56_multiway_cluster.py) (+2) |
| `nbreg` | MASS::glm.nb | R 4.5.2; MASS 7.3.65 | rel_est<=1e-06, rel_se<=0.005 | 6.0e-10 / 1.3e-10 | [`42_nbreg.py`](../tests/r_parity/42_nbreg.py) (+2) |
| `number_needed_to_treat` | base-R closed form (NNT = 1/risk difference) | R 4.5.2 | estimate 1e-12 abs (observed 0); CI not pinned | — / — | [`test_epi_parity.py`](../tests/reference_parity/test_epi_parity.py) (+1) |
| `oaxaca` | oaxaca::oaxaca | R 4.5.2; oaxaca 0.1.5 | rel_est<=1e-06, rel_se<=0.05 | 6.3e-16 / 1.3e-16 | [`30_oaxaca.py`](../tests/r_parity/30_oaxaca.py) (+2) |
| `odds_ratio` | base-R closed form (Woolf logit; = epiR::epi.2by2) | R 4.5.2 | estimate, se_log, CI 1e-12 abs (observed 0) | — / — | [`test_epi_parity.py`](../tests/reference_parity/test_epi_parity.py) (+1) |
| `ologit` | MASS::polr(method="logistic") | R 4.5.2; MASS 7.3.65 | rel_est<=1e-06, rel_se<=1e-05 | 1.8e-07 / 3.5e-07 | [`45_ologit.py`](../tests/r_parity/45_ologit.py) (+2) |
| `oprobit` | MASS::polr(method="probit") | R 4.5.2; MASS 7.3.65 | rel_est<=1e-06, rel_se<=1e-06 | 3.4e-07 / 2.8e-08 | [`49_oprobit.py`](../tests/r_parity/49_oprobit.py) (+2) |
| `oster_delta` | coefficient-stability bound identities (Oster 2019) | R 4.5.2 | OLS inputs 1e-12 abs (observed 0); beta(delta*)=0 at 1e-10 | — / — | [`test_oster_delta_parity.py`](../tests/reference_parity/test_oster_delta_parity.py) |
| `panel` | plm::plm + plm::phtest | R 4.5.2; plm 2.6.7 | rel_est<=1e-06, rel_se<=0.001 | 4.7e-14 / 1.5e-15 | [`35_panel.py`](../tests/r_parity/35_panel.py) (+2) |
| `panel_qtet` | qte::panel.qtet 1.3.1 (Callaway & Li 2019) | — | all 19 quantiles: abs < 1e-8 (observed 6.8e-12); ATT abs < 1e-6. panel.qtet composes ordinary ecdf evaluations and type-7 quantiles, both of which have exact numpy equivalents, so this is machine-precision agreement rather than a tolerance band. | — / — | [`test_panel_qtet_parity.py`](../tests/reference_parity/test_panel_qtet_parity.py) (+1) |
| `poisson` | stats::glm(family=poisson()) | R 4.5.2; stats 4.5.2 | rel_est<=1e-06, rel_se<=1e-06 | 9.2e-15 / 8.7e-12 | [`58_poisson.py`](../tests/r_parity/58_poisson.py) (+2) |
| `policy_tree` | policytree::policy_tree | R 4.5.2; policytree 1.2.4 | rel_est<=1e-06, rel_se<=1e-06 | 9.6e-16 / 1.4e-16 | [`70_policy_tree.py`](../tests/r_parity/70_policy_tree.py) (+2) |
| `policy_value` | empirical policy value V(pi) = mean(Gamma * pi) (Athey & Wager 2021) | R 4.5.2 | V(pi) == mean(scores * policy) 1e-15 abs (observed 0) | — / — | [`test_policy_value_parity.py`](../tests/reference_parity/test_policy_value_parity.py) |
| `power_case_control` | base-R closed form (case-control OR power, 2-prop z) | R 4.5.2 | power 1e-12 abs (observed 0) | — / — | [`test_epi_diag_parity.py`](../tests/reference_parity/test_epi_diag_parity.py) |
| `power_cluster_rct` | base-R closed form (design-effect-inflated z-approx power) | R 4.5.2 | power 1e-12 abs (observed ~2e-16) | — / — | [`test_power_extra_parity.py`](../tests/reference_parity/test_power_extra_parity.py) (+1) |
| `power_logrank` | base-R closed form (Schoenfeld log-rank power) | R 4.5.2 | power 1e-12 abs (observed ~2e-16) | — / — | [`test_power_parity.py`](../tests/reference_parity/test_power_parity.py) (+1) |
| `power_rct` | base-R closed form (two-sample pooled-sigma z-approx power) | R 4.5.2 | power 1e-12 abs (observed ~2e-16) | — / — | [`test_power_parity.py`](../tests/reference_parity/test_power_parity.py) (+1) |
| `power_two_proportions` | base-R closed form (unpooled Wald two-proportion z-approx) | R 4.5.2 | power 1e-12 abs (observed ~2e-16) | — / — | [`test_power_parity.py`](../tests/reference_parity/test_power_parity.py) (+1) |
| `ppmlhdfe` | fixest::fepois | R 4.5.2; fixest 0.14.0 | rel_est<=1e-06, rel_se<=0.01 | 4.9e-13 / 2.2e-15 | [`37_ppmlhdfe.py`](../tests/r_parity/37_ppmlhdfe.py) (+2) |
| `prevalence_ratio` | base-R closed form (Katz-log; = epiR::epi.2by2) | R 4.5.2 | estimate, se_log, CI 1e-12 abs (observed ~2e-16) | — / — | [`test_epi_parity.py`](../tests/reference_parity/test_epi_parity.py) (+1) |
| `probit` | stats::glm(family=binomial("probit")) | R 4.5.2; stats 4.5.2 | rel_est<=1e-06, rel_se<=0.01 | 3.1e-07 / 1.6e-08 | [`48_probit.py`](../tests/r_parity/48_probit.py) (+2) |
| `psm` | MatchIt::matchit | R 4.5.2; MatchIt 4.7.2 | rel_est<=1e-06, rel_se<=1e-06 | 1.2e-15 / 2.0e-16 | [`11_psm.py`](../tests/r_parity/11_psm.py) (+2) |
| `pwcompare` | pairwise-contrast identity (= Stata pwcompare) | R 4.5.2 | pairwise diff == coef difference 1e-12 abs (observed <= 1e-15) | — / — | [`test_contrast_pwcompare_parity.py`](../tests/reference_parity/test_contrast_pwcompare_parity.py) |
| `qreg` | quantreg::rq | R 4.5.2; quantreg 6.1 | rel_est<=1e-06, rel_se<=0.1 | 3.3e-15 / 4.4e-15 | [`40_qreg.py`](../tests/r_parity/40_qreg.py) (+2) |
| `rddensity` | rddensity::rddensity | R 4.5.2; rddensity 2.6 | rel_est<=1e-06, rel_se<=1e-06 | 3.3e-11 / 8.9e-11 | [`09_rddensity.py`](../tests/r_parity/09_rddensity.py) (+2) |
| `rdrobust` | rdrobust::rdrobust | R 4.5.2; rdrobust 3.0.0 | rel_est<=1e-06, rel_se<=0.1 | 7.9e-13 / 2.4e-10 | [`06_rd.py`](../tests/r_parity/06_rd.py) (+2) |
| `regress` | lm + sandwich::vcovHC | R 4.5.2; sandwich 3.1.1 | rel_est<=1e-06, rel_se<=1e-06 | 1.1e-12 / 1.3e-12 | [`01_ols.py`](../tests/r_parity/01_ols.py) (+2) |
| `relative_risk` | base-R closed form (Katz-log; = epiR::epi.2by2 / Stata epitab) | R 4.5.2 | estimate, se_log, CI 1e-12 abs (observed 0) | — / — | [`test_epi_parity.py`](../tests/reference_parity/test_epi_parity.py) (+1) |
| `reset_test` | lmtest::resettest(power=2:3, type='fitted') | R 4.5.2; lmtest 0.9.40 | F-statistic & p-value 1e-10 rel (observed ~1e-13) | — / — | [`test_diagnostics_parity.py`](../tests/reference_parity/test_diagnostics_parity.py) (+1) |
| `rif_decomposition` | dineq::rif + manual OLS | R 4.5.2; dineq 0.1.0 | rel_est<=1e-06, rel_se<=1e-06 | 2.2e-15 / 1.4e-16 | [`32_rif.py`](../tests/r_parity/32_rif.py) (+2) |
| `risk_difference` | base-R closed form (Wald; = epiR::epi.2by2 / Stata epitab) | R 4.5.2 | estimate, se, CI 1e-12 abs (observed 0) | — / — | [`test_epi_parity.py`](../tests/reference_parity/test_epi_parity.py) (+1) |
| `roc_curve` | Mann-Whitney rank AUC (= pROC::auc / sklearn) | R 4.5.2 | AUC 1e-12 abs (observed 0) | — / — | [`test_auc_parity.py`](../tests/reference_parity/test_auc_parity.py) |
| `sar` | spatialreg::lagsarlm / spatialreg::errorsarlm / spatialreg::lagsarlm(Durbin=TRUE) | R 4.5.2; spatialreg 1.4.3 | rel_est<=1e-06, rel_se<=1e-06 | 8.3e-08 / 5.1e-08 | [`65_spatial.py`](../tests/r_parity/65_spatial.py) (+2) |
| `sar_gmm` | spatialreg::stsls(W2X=FALSE) / spatialreg::GMerrorsar | R 4.5.2; spatialreg 1.4.3 | rel_est<=1e-06, rel_se<=1e-06 | 4.6e-08 / 7.3e-16 | [`66_spatial_gmm.py`](../tests/r_parity/66_spatial_gmm.py) (+2) |
| `sbw` | sbw::sbw 1.2 (Zubizarreta 2015), quadprog solver | — | ATT rel <= 1e-8 (observed <= 4e-10) under both standardisation conventions: tolerance_scale='target' == bal_std='target' and 'group' == bal_std='group', at bal_tol 0.05 and 0.02. | — / — | [`test_matching_r_parity.py`](../tests/reference_parity/test_matching_r_parity.py) (+1) |
| `sdid` | synthdid::synthdid_estimate | R 4.5.2; synthdid 0.0.9 | rel_est<=1e-06, rel_se<=1e-06 | 2.6e-15 / 7.2e-08 | [`12_sdid.py`](../tests/r_parity/12_sdid.py) (+2) |
| `sdm` | spatialreg::lagsarlm / spatialreg::errorsarlm / spatialreg::lagsarlm(Durbin=TRUE) | R 4.5.2; spatialreg 1.4.3 | rel_est<=1e-06, rel_se<=1e-06 | 8.3e-08 / 5.1e-08 | [`65_spatial.py`](../tests/r_parity/65_spatial.py) (+2) |
| `sem` | spatialreg::lagsarlm / spatialreg::errorsarlm / spatialreg::lagsarlm(Durbin=TRUE) | R 4.5.2; spatialreg 1.4.3 | rel_est<=1e-06, rel_se<=1e-06 | 8.3e-08 / 5.1e-08 | [`65_spatial.py`](../tests/r_parity/65_spatial.py) (+2) |
| `sem_gmm` | spatialreg::stsls(W2X=FALSE) / spatialreg::GMerrorsar | R 4.5.2; spatialreg 1.4.3 | rel_est<=1e-06, rel_se<=1e-06 | 4.6e-08 / 7.3e-16 | [`66_spatial_gmm.py`](../tests/r_parity/66_spatial_gmm.py) (+2) |
| `sensemakr` | sensemakr::sensemakr | R 4.5.2; sensemakr 0.1.6 | rel_est<=1e-06, rel_se<=1e-06 | 5.0e-08 / 5.0e-08 | [`22_sensemakr.py`](../tests/r_parity/22_sensemakr.py) (+2) |
| `sensitivity_specificity` | base closed form (2x2 diagnostic accuracy; = epiR) | R 4.5.2 | sens/spec/PPV/NPV/LR 1e-12 abs (observed 0) | — / — | [`test_epi_diag_parity.py`](../tests/reference_parity/test_epi_diag_parity.py) |
| `source_decompose` | Lerman-Yitzhaki (1985) Gini source decomposition identity | R 4.5.2 | sum(contribution) == total_gini 1e-12 abs (observed ~1e-16) | — / — | [`test_source_decompose_parity.py`](../tests/reference_parity/test_source_decompose_parity.py) |
| `stacked_did` | hand-written stack + fixest::feols | R 4.5.2; fixest 0.14.0 | rel_est<=1e-06 | 3.9e-13 / 7.1e-13 | [`75_stacked.py`](../tests/r_parity/75_stacked.py) (+2) |
| `subgroup_decompose` | Theil within+between exact additive identity (Shorrocks 1980) | R 4.5.2 | total == within + between 1e-12 abs (observed 0) | — / — | [`test_subgroup_decompose_parity.py`](../tests/reference_parity/test_subgroup_decompose_parity.py) |
| `sun_abraham` | fixest::sunab | R 4.5.2; fixest 0.14.0 | rel_est<=1e-06, rel_se<=0.25 | 2.8e-11 / 2.7e-11 | [`05_sunab.py`](../tests/r_parity/05_sunab.py) (+2) |
| `sureg` | systemfit::systemfit(method="SUR", noDfCor) | R 4.5.2; systemfit 1.1.30 | rel_est<=1e-06, rel_se<=1e-06 | 1.5e-14 / 1.5e-15 | [`60_sureg.py`](../tests/r_parity/60_sureg.py) (+2) |
| `svyglm` | survey::svyglm (design-based GLM + linearization SE) | R 4.5.2 | coefficients + SE 1e-10 abs (observed ~2e-15 / 6e-15) | — / — | [`test_survey_parity.py`](../tests/reference_parity/test_survey_parity.py) (+1) |
| `svymean` | survey::svymean (Horvitz-Thompson/Hajek + Taylor SE) | R 4.5.2 | estimate + SE 1e-10 abs (observed ~5e-15 / 8e-17) | — / — | [`test_survey_parity.py`](../tests/reference_parity/test_survey_parity.py) (+1) |
| `svytotal` | survey::svytotal (Horvitz-Thompson + Taylor SE) | R 4.5.2 | estimate 1e-12 rel; SE 1e-10 rel (observed ~2e-12 / 1e-14) | — / — | [`test_survey_parity.py`](../tests/reference_parity/test_survey_parity.py) (+1) |
| `synth` | Synth::synth | R 4.5.2; Synth 1.1.10 | rel_est<=1e-06, rel_se<=1e-06 | 7.8e-08 / 7.7e-08 | [`52_scm_unique.py`](../tests/r_parity/52_scm_unique.py) (+2) |
| `three_sls` | R systemfit::systemfit(method='3SLS') | R 4.5.2; systemfit 1.1.30 | coef 1e-9 abs (observed <= 1e-15); SE ~5e-3 rel | — / — | [`test_threesls_parity.py`](../tests/reference_parity/test_threesls_parity.py) (+1) |
| `tmle` | tmle::tmle | R 4.5.2; tmle 2.1.1 | rel_est<=1e-06, rel_se<=1e-06 | 1.9e-09 / 1.9e-09 | [`72_tmle.py`](../tests/r_parity/72_tmle.py) (+2) |
| `tobit` | censReg::censReg | R 4.5.2; censReg 0.5.38 | rel_est<=1e-06, rel_se<=1e-05 | 2.8e-08 / 2.8e-08 | [`41_tobit.py`](../tests/r_parity/41_tobit.py) (+2) |
| `truncreg` | truncreg::truncreg(method="NR") | R 4.5.2; truncreg 0.2.5 | rel_est<=1e-06, rel_se<=0.0001 | 3.3e-08 / 9.5e-08 | [`62_truncreg.py`](../tests/r_parity/62_truncreg.py) (+2) |
| `twoway_cluster` | sandwich::vcovCL(cluster=~g1+g2) | R 4.5.2; sandwich 3.1.1 | rel_est<=1e-06, rel_se<=1e-06 | 7.8e-16 / 7.8e-16 | [`54_twoway_cluster.py`](../tests/r_parity/54_twoway_cluster.py) (+2) |
| `var` | vars::VAR | R 4.5.2; vars 1.6.1 | rel_est<=1e-06, rel_se<=0.001 | 3.1e-15 / 6.6e-15 | [`33_var.py`](../tests/r_parity/33_var.py) (+2) |
| `xtabond` | plm::pgmm | R 4.5.2; plm 2.6.7 | rel_est<=1e-06, rel_se<=1e-06 | 9.0e-16 / 1.4e-15 | [`50_xtabond.py`](../tests/r_parity/50_xtabond.py) (+2) |
| `zip_model` | pscl::zeroinfl(dist="poisson") | R 4.5.2; pscl 1.5.9 | rel_est<=1e-06, rel_se<=0.0001 | 7.7e-08 / 1.1e-07 | [`63_zip.py`](../tests/r_parity/63_zip.py) (+2) |

## aligned — 16 functions

Agreement within a documented, pre-registered looser tolerance.

| function | reference | versions | tolerance | rel err (R / Stata) | test |
| --- | --- | --- | --- | --- | --- |
| `aft` | survival::survreg (Weibull AFT) | R 4.5.2; survival 3.8.3 | coefficients & log-scale 5e-5 abs (observed ~1e-5) | — / — | [`test_aft_parity.py`](../tests/reference_parity/test_aft_parity.py) (+1) |
| `augsynth` | augsynth::augsynth | R 4.5.2; augsynth 0.2.0 | rel_est<=2e-05, rel_se<=1e-06 | 7.9e-06 / — | [`18_augsynth.py`](../tests/r_parity/18_augsynth.py) (+1) |
| `causal_forest` | grf::causal_forest | R 4.5.2; grf 2.6.1 | rel_est<=0.01, rel_se<=0.25 | 1.9e-03 / — | [`13_causal_forest.py`](../tests/r_parity/13_causal_forest.py) (+1) |
| `cbps` | CBPS::CBPS 0.24 (Imai & Ratkovic 2014) | — | ATE over/exact and ATT exact: rel <= 5e-3 (R's optimiser slack). ATT over is NOT pinned to R -- CBPS's ATT gradient mis-scales the balance block by n/n_1 and stops off-stationarity; StatsPAI is asserted to attain strictly better covariate balance instead. | — / — | [`test_matching_r_parity.py`](../tests/reference_parity/test_matching_r_parity.py) (+1) |
| `cic` | qte::CiC | R 4.5.2 | rel_est<=1e-06 | 2.4e-15 / 6.0e-03 | [`74_cic.py`](../tests/r_parity/74_cic.py) (+2) |
| `cloglog` | stats::glm(binomial('cloglog')) | R 4.5.2 | coefficients 5e-5 abs (observed ~1e-5; IRLS convergence) | — / — | [`test_glm_ext_parity.py`](../tests/reference_parity/test_glm_ext_parity.py) (+1) |
| `functional_form_test` | didFF::didFF | R 4.5.2 | rel_est<=0.001 | 1.3e-14 / — | [`79_didff.py`](../tests/r_parity/79_didff.py) (+1) |
| `genmatch` | Matching::Match 4.10-15 (Weight = 3, Weight.matrix) | — | Deterministic kernel only: given the same diagonal W, the 1-NN assignment agrees with Matching::Match on all 163 uniquely matched treated units on MatchIt::lalonde. | — / — | [`test_matching_r_parity.py`](../tests/reference_parity/test_matching_r_parity.py) (+1) |
| `optimal_match` | optmatch::pairmatch 0.10.8 on a logit propensity score | — | Total matched distance <= optmatch's (1 + 1e-6). The matched pairs are not pinned: the assignment problem is degenerate on this data, so equally optimal solutions report different ATTs. | — / — | [`test_matching_r_parity.py`](../tests/reference_parity/test_matching_r_parity.py) (+1) |
| `pretrends_power` | pretrends::pretrends / pretrends::slope_for_power (GitHub, not CRAN) | R 4.5.2 | rel_est<=0.001 | 3.2e-05 / 1.5e-04 | [`76_pretrends.py`](../tests/r_parity/76_pretrends.py) (+2) |
| `pretrends_slope_for_power` | pretrends::pretrends / pretrends::slope_for_power (GitHub, not CRAN) | R 4.5.2 | rel_est<=0.001 | 3.2e-05 / 1.5e-04 | [`76_pretrends.py`](../tests/r_parity/76_pretrends.py) (+2) |
| `qdid` | qte::QDiD 1.3.1 | — | max deviation / scale < 0.08, sign agreement on the large effects and correlation > 0.999. R's quantiles come from BMisc::weighted_quantile (stats::optimize on a piecewise-linear check function, which has plateaus where every point is a minimiser) while sp.qdid interpolates the empirical inverse CDF; on this fixture the gap is at most ~152 currency units against effects running to ~8900. | — / — | [`test_qdid_parity.py`](../tests/reference_parity/test_qdid_parity.py) (+1) |
| `qte` | qte::ci.qte / qte::ci.qtet 1.3.1 (Firpo 2007) | — | max relative deviation < 0.01 on lalonde.exp / lalonde.psid. Both sides minimise the same weighted check function; R's BMisc::weighted_quantile uses a golden-section search whose answer on a plateau is an optimiser artifact, so point-value equality is not asserted. | — / — | [`test_firpo_qte_parity.py`](../tests/reference_parity/test_firpo_qte_parity.py) (+1) |
| `survreg` | survival::survreg (Weibull AFT) | R 4.5.2; survival 3.8.3 | coefficients & log-scale 5e-5 abs (observed ~1e-5) | — / — | [`test_aft_parity.py`](../tests/reference_parity/test_aft_parity.py) (+1) |
| `xtfrontier` | frontier::sfa | R 4.5.2; frontier 1.1.8 | rel_est<=0.001, rel_se<=0.001 | 2.8e-06 / 8.6e-04 | [`29_panel_sfa.py`](../tests/r_parity/29_panel_sfa.py) (+2) |
| `zinb` | pscl::zeroinfl(dist="negbin") | R 4.5.2; pscl 1.5.9 | rel_est<=1e-05, rel_se<=0.001 | 1.1e-06 / 2.1e-07 | [`64_zinb.py`](../tests/r_parity/64_zinb.py) (+2) |

## external-replication — 6 functions

Reproduces published-paper numbers; sources in `tests/external_parity/PUBLISHED_REFERENCE_VALUES.md`.

| function | test |
| --- | --- |
| `aggte` | [`test_honest_did_paper_parity.py`](../tests/external_parity/test_honest_did_paper_parity.py) (+1) |
| `bibtex` | [`test_rebel_canal_published.py`](../tests/external_parity/test_rebel_canal_published.py) |
| `breakdown_m` | [`test_honest_did_paper_parity.py`](../tests/external_parity/test_honest_did_paper_parity.py) |
| `event_study` | [`test_rebel_canal_published.py`](../tests/external_parity/test_rebel_canal_published.py) |
| `g_estimation` | [`test_whatif_nhefs.py`](../tests/external_parity/test_whatif_nhefs.py) |
| `parallel_trends_robustness` | [`test_rebel_canal_published.py`](../tests/external_parity/test_rebel_canal_published.py) |

## analytical-only — 225 functions

Recovers a known DGP truth / closed-form identity within tolerance; no cross-package reference. See `tests/reference_parity/REFERENCES.md`.

| function | test |
| --- | --- |
| `ackerberg_caves_frazer` | [`test_structural_parity.py`](../tests/reference_parity/test_structural_parity.py) |
| `aggte_from_influence` | [`test_aggte_r_did_parity.py`](../tests/reference_parity/test_aggte_r_did_parity.py) |
| `aipw` | [`test_paper_parity.py`](../tests/reference_parity/test_paper_parity.py) (+1) |
| `always_treat` | [`test_longitudinal_parity.py`](../tests/reference_parity/test_longitudinal_parity.py) |
| `anderson_rubin_test` | [`test_anderson_rubin_parity.py`](../tests/reference_parity/test_anderson_rubin_parity.py) |
| `assimilative_causal` | [`test_assimilation_parity.py`](../tests/reference_parity/test_assimilation_parity.py) |
| `attrition_test` | [`test_attrition_test_parity.py`](../tests/reference_parity/test_attrition_test_parity.py) |
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
| `bradford_hill` | [`test_bradford_hill_parity.py`](../tests/reference_parity/test_bradford_hill_parity.py) |
| `breslow_day_test` | [`test_breslow_day_parity.py`](../tests/reference_parity/test_breslow_day_parity.py) |
| `bunching` | [`test_bunching_parity.py`](../tests/reference_parity/test_bunching_parity.py) |
| `bvar` | [`test_timeseries_parity.py`](../tests/reference_parity/test_timeseries_parity.py) |
| `calibration_test` | [`test_calibration_test_parity.py`](../tests/reference_parity/test_calibration_test_parity.py) |
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
| `conley` | [`test_conley_acreg_spacetime_parity.py`](../tests/reference_parity/test_conley_acreg_spacetime_parity.py) (+1) |
| `continuous_did` | [`test_dose_response_parity.py`](../tests/reference_parity/test_dose_response_parity.py) |
| `continuous_iv_late` | [`test_continuous_iv_late_parity.py`](../tests/reference_parity/test_continuous_iv_late_parity.py) |
| `counterfactual_fairness` | [`test_fairness_parity.py`](../tests/reference_parity/test_fairness_parity.py) |
| `cox_frailty` | [`test_competing_risks_parity.py`](../tests/reference_parity/test_competing_risks_parity.py) |
| `cr3_jackknife_vcov` | [`test_recovery_batch2_parity.py`](../tests/reference_parity/test_recovery_batch2_parity.py) |
| `cuminc` | [`test_competing_risks_parity.py`](../tests/reference_parity/test_competing_risks_parity.py) |
| `cusum_test` | [`test_cusum_test_parity.py`](../tests/reference_parity/test_cusum_test_parity.py) |
| `demographic_parity` | [`test_fairness_parity.py`](../tests/reference_parity/test_fairness_parity.py) |
| `describe_function` | [`test_qdid_parity.py`](../tests/reference_parity/test_qdid_parity.py) |
| `did` | [`test_did2x2_wild_parity.py`](../tests/reference_parity/test_did2x2_wild_parity.py) |
| `did_had` | [`test_did_had_parity.py`](../tests/reference_parity/test_did_had_parity.py) |
| `discos` | [`test_distributional_te_parity.py`](../tests/reference_parity/test_distributional_te_parity.py) |
| `dist_iv` | [`test_dist_iv_parity.py`](../tests/reference_parity/test_dist_iv_parity.py) |
| `distributional_te` | [`test_distributional_te_inference.py`](../tests/reference_parity/test_distributional_te_inference.py) (+1) |
| `dml_panel` | [`test_ml_causal_recovery_parity.py`](../tests/reference_parity/test_ml_causal_recovery_parity.py) |
| `dose_response` | [`test_dose_response_parity.py`](../tests/reference_parity/test_dose_response_parity.py) |
| `effective_f_test` | [`test_anderson_rubin_parity.py`](../tests/reference_parity/test_anderson_rubin_parity.py) |
| `engle_granger` | [`test_engle_granger_parity.py`](../tests/reference_parity/test_engle_granger_parity.py) |
| `equalized_odds` | [`test_fairness_parity.py`](../tests/reference_parity/test_fairness_parity.py) |
| `etregress` | [`test_etregress_parity.py`](../tests/reference_parity/test_etregress_parity.py) |
| `evidence_without_injustice` | [`test_fairness_parity.py`](../tests/reference_parity/test_fairness_parity.py) |
| `fairlie` | [`test_decomposition_family_parity.py`](../tests/reference_parity/test_decomposition_family_parity.py) |
| `fairness_audit` | [`test_fairness_parity.py`](../tests/reference_parity/test_fairness_parity.py) |
| `fci` | [`test_fci_parity.py`](../tests/reference_parity/test_fci_parity.py) |
| `ffl_decompose` | [`test_decomposition_family_parity.py`](../tests/reference_parity/test_decomposition_family_parity.py) |
| `finegray` | [`test_competing_risks_parity.py`](../tests/reference_parity/test_competing_risks_parity.py) |
| `fisher_exact` | [`test_fisher_exact_parity.py`](../tests/reference_parity/test_fisher_exact_parity.py) |
| `focal_cate` | [`test_ml_causal_recovery_parity_round2.py`](../tests/reference_parity/test_ml_causal_recovery_parity_round2.py) |
| `fortified_pci` | [`test_proximal_parity.py`](../tests/reference_parity/test_proximal_parity.py) |
| `four_way_decomposition` | [`test_four_way_decomposition_parity.py`](../tests/reference_parity/test_four_way_decomposition_parity.py) |
| `front_door` | [`test_front_door_parity.py`](../tests/reference_parity/test_front_door_parity.py) |
| `frontdoor` | [`test_frontdoor_parity.py`](../tests/reference_parity/test_frontdoor_parity.py) |
| `garch` | [`test_timeseries_parity.py`](../tests/reference_parity/test_timeseries_parity.py) |
| `geary` | [`test_geary_parity.py`](../tests/reference_parity/test_geary_parity.py) |
| `general_bunching` | [`test_bunching_parity.py`](../tests/reference_parity/test_bunching_parity.py) |
| `geolift` | [`test_geolift_parity.py`](../tests/reference_parity/test_geolift_parity.py) |
| `ges` | [`test_ges_parity.py`](../tests/reference_parity/test_ges_parity.py) |
| `getis_ord_local` | [`test_esda_local_parity.py`](../tests/reference_parity/test_esda_local_parity.py) |
| `gformula_ice_fn` | [`test_gformula_family_parity.py`](../tests/reference_parity/test_gformula_family_parity.py) |
| `gformula_mc` | [`test_gformula_family_parity.py`](../tests/reference_parity/test_gformula_family_parity.py) |
| `gmm` | [`test_general_gmm_parity.py`](../tests/reference_parity/test_general_gmm_parity.py) (+1) |
| `granger_causality` | [`test_timeseries_parity.py`](../tests/reference_parity/test_timeseries_parity.py) |
| `grapple` | [`test_grapple_parity.py`](../tests/reference_parity/test_grapple_parity.py) |
| `hal_tmle` | [`test_ml_causal_recovery_parity_round2.py`](../tests/reference_parity/test_ml_causal_recovery_parity_round2.py) |
| `hausman_test` | [`test_diag_recovery_parity.py`](../tests/reference_parity/test_diag_recovery_parity.py) |
| `hdfe_ols` | [`test_hdfe_parity.py`](../tests/reference_parity/test_hdfe_parity.py) |
| `honest_variance` | [`test_forest_rate_honest_parity.py`](../tests/reference_parity/test_forest_rate_honest_parity.py) |
| `horowitz_manski` | [`test_horowitz_manski_parity.py`](../tests/reference_parity/test_horowitz_manski_parity.py) |
| `immortal_time_check` | [`test_target_trial_parity.py`](../tests/reference_parity/test_target_trial_parity.py) |
| `influence_functions` | [`test_aggte_r_did_parity.py`](../tests/reference_parity/test_aggte_r_did_parity.py) |
| `interactive_fe` | [`test_panel_estimators_parity.py`](../tests/reference_parity/test_panel_estimators_parity.py) |
| `interference` | [`test_interference_parity.py`](../tests/reference_parity/test_interference_parity.py) |
| `ipcw` | [`test_ipcw_parity.py`](../tests/reference_parity/test_ipcw_parity.py) |
| `irf` | [`test_timeseries_parity.py`](../tests/reference_parity/test_timeseries_parity.py) |
| `its` | [`test_timeseries_parity.py`](../tests/reference_parity/test_timeseries_parity.py) |
| `iv` | [`test_regress_weights_iv_robust_parity.py`](../tests/reference_parity/test_regress_weights_iv_robust_parity.py) |
| `iv_diag` | [`test_diag_recovery_parity.py`](../tests/reference_parity/test_diag_recovery_parity.py) |
| `ivqreg` | [`test_ivqreg_parity.py`](../tests/reference_parity/test_ivqreg_parity.py) |
| `jackknife_se` | [`test_recovery_batch2_parity.py`](../tests/reference_parity/test_recovery_batch2_parity.py) |
| `jive` | [`test_jive_parity.py`](../tests/reference_parity/test_jive_parity.py) |
| `johansen` | [`test_timeseries_parity.py`](../tests/reference_parity/test_timeseries_parity.py) |
| `join_counts` | [`test_esda_local_parity.py`](../tests/reference_parity/test_esda_local_parity.py) |
| `kan_dlate` | [`test_dist_iv_parity.py`](../tests/reference_parity/test_dist_iv_parity.py) |
| `knn_weights` | [`test_esda_local_parity.py`](../tests/reference_parity/test_esda_local_parity.py) (+4) |
| `lasso_iv` | [`test_lasso_iv_parity.py`](../tests/reference_parity/test_lasso_iv_parity.py) |
| `lasso_select` | [`test_lasso_select_parity.py`](../tests/reference_parity/test_lasso_select_parity.py) |
| `levinsohn_petrin` | [`test_structural_parity.py`](../tests/reference_parity/test_structural_parity.py) |
| `lincom` | [`test_postestimation_parity.py`](../tests/reference_parity/test_postestimation_parity.py) |
| `lingam` | [`test_causal_discovery_parity.py`](../tests/reference_parity/test_causal_discovery_parity.py) |
| `list_replications` | [`test_castle_stata_parity.py`](../tests/reference_parity/test_castle_stata_parity.py) (+1) |
| `long_term_from_short` | [`test_surrogate_parity.py`](../tests/reference_parity/test_surrogate_parity.py) |
| `longitudinal_analyze` | [`test_longitudinal_parity.py`](../tests/reference_parity/test_longitudinal_parity.py) |
| `longitudinal_contrast` | [`test_longitudinal_parity.py`](../tests/reference_parity/test_longitudinal_parity.py) |
| `lp_did` | [`test_absorbing_reference.py`](../tests/reference_parity/test_absorbing_reference.py) |
| `lpbwselect_mse_dpi` | [`test_did_had_parity.py`](../tests/reference_parity/test_did_had_parity.py) (+1) |
| `lprobust_at_point` | [`test_lprobust_parity.py`](../tests/reference_parity/test_lprobust_parity.py) |
| `ltmle` | [`test_ml_causal_recovery_parity_round2.py`](../tests/reference_parity/test_ml_causal_recovery_parity_round2.py) |
| `ltmle_survival` | [`test_ml_causal_recovery_parity_round2.py`](../tests/reference_parity/test_ml_causal_recovery_parity_round2.py) |
| `malmquist` | [`test_frontier_efficiency_parity.py`](../tests/reference_parity/test_frontier_efficiency_parity.py) |
| `margins` | [`test_postestimation_parity.py`](../tests/reference_parity/test_postestimation_parity.py) |
| `markup` | [`test_structural_parity.py`](../tests/reference_parity/test_structural_parity.py) |
| `matrix_completion` | [`test_matrix_completion_parity.py`](../tests/reference_parity/test_matrix_completion_parity.py) |
| `mc_panel` | [`test_matrix_completion_parity.py`](../tests/reference_parity/test_matrix_completion_parity.py) |
| `meta_analysis` | [`test_meta_analysis_parity.py`](../tests/reference_parity/test_meta_analysis_parity.py) |
| `metafrontier` | [`test_frontier_efficiency_parity.py`](../tests/reference_parity/test_frontier_efficiency_parity.py) |
| `mi_estimate` | [`test_imputation_parity.py`](../tests/reference_parity/test_imputation_parity.py) |
| `mice` | [`test_imputation_parity.py`](../tests/reference_parity/test_imputation_parity.py) |
| `model_averaging_dml` | [`test_ml_causal_recovery_parity.py`](../tests/reference_parity/test_ml_causal_recovery_parity.py) |
| `moran` | [`test_moran_parity.py`](../tests/reference_parity/test_moran_parity.py) |
| `moran_local` | [`test_esda_local_parity.py`](../tests/reference_parity/test_esda_local_parity.py) |
| `mr_cml` | [`test_mr_cml_parity.py`](../tests/reference_parity/test_mr_cml_parity.py) |
| `mr_egger` | [`test_mr_parity.py`](../tests/reference_parity/test_mr_parity.py) |
| `mr_f_statistic` | [`test_mr_f_steiger_parity.py`](../tests/reference_parity/test_mr_f_steiger_parity.py) |
| `mr_heterogeneity` | [`test_mr_diagnostics_parity.py`](../tests/reference_parity/test_mr_diagnostics_parity.py) |
| `mr_ivw` | [`test_mr_parity.py`](../tests/reference_parity/test_mr_parity.py) |
| `mr_lap` | [`test_mr_lap_parity.py`](../tests/reference_parity/test_mr_lap_parity.py) |
| `mr_leave_one_out` | [`test_mr_parity.py`](../tests/reference_parity/test_mr_parity.py) |
| `mr_median` | [`test_mr_parity.py`](../tests/reference_parity/test_mr_parity.py) |
| `mr_mediation` | [`test_mr_mediation_parity.py`](../tests/reference_parity/test_mr_mediation_parity.py) |
| `mr_mode` | [`test_mr_mode_parity.py`](../tests/reference_parity/test_mr_mode_parity.py) |
| `mr_multivariable` | [`test_mr_multivariable_parity.py`](../tests/reference_parity/test_mr_multivariable_parity.py) |
| `mr_pleiotropy_egger` | [`test_mr_diagnostics_parity.py`](../tests/reference_parity/test_mr_diagnostics_parity.py) |
| `mr_presso` | [`test_mr_parity.py`](../tests/reference_parity/test_mr_parity.py) |
| `mr_radial` | [`test_mr_parity.py`](../tests/reference_parity/test_mr_parity.py) |
| `mr_raps` | [`test_mr_raps_parity.py`](../tests/reference_parity/test_mr_raps_parity.py) |
| `mr_steiger` | [`test_mr_f_steiger_parity.py`](../tests/reference_parity/test_mr_f_steiger_parity.py) |
| `msm` | [`test_msm_family_parity.py`](../tests/reference_parity/test_msm_family_parity.py) |
| `multi_treatment` | [`test_multi_treatment_parity.py`](../tests/reference_parity/test_multi_treatment_parity.py) |
| `network_exposure` | [`test_interference_parity.py`](../tests/reference_parity/test_interference_parity.py) |
| `never_treat` | [`test_longitudinal_parity.py`](../tests/reference_parity/test_longitudinal_parity.py) |
| `notch` | [`test_notch_parity.py`](../tests/reference_parity/test_notch_parity.py) |
| `notears` | [`test_causal_discovery_parity.py`](../tests/reference_parity/test_causal_discovery_parity.py) |
| `olley_pakes` | [`test_structural_parity.py`](../tests/reference_parity/test_structural_parity.py) |
| `orthogonal_to_bias` | [`test_fairness_parity.py`](../tests/reference_parity/test_fairness_parity.py) |
| `overlap_weights` | [`test_matching_parity.py`](../tests/reference_parity/test_matching_parity.py) (+1) |
| `panel_fgls` | [`test_panel_estimators_parity.py`](../tests/reference_parity/test_panel_estimators_parity.py) |
| `panel_logit` | [`test_panel_estimators_parity.py`](../tests/reference_parity/test_panel_estimators_parity.py) |
| `panel_probit` | [`test_panel_estimators_parity.py`](../tests/reference_parity/test_panel_estimators_parity.py) |
| `panel_unitroot` | [`test_timeseries_parity.py`](../tests/reference_parity/test_timeseries_parity.py) |
| `particle_filter` | [`test_assimilation_parity.py`](../tests/reference_parity/test_assimilation_parity.py) |
| `pate` | [`test_pate_parity.py`](../tests/reference_parity/test_pate_parity.py) |
| `pc_algorithm` | [`test_causal_discovery_parity.py`](../tests/reference_parity/test_causal_discovery_parity.py) |
| `peer_effects` | [`test_peer_effects_parity.py`](../tests/reference_parity/test_peer_effects_parity.py) |
| `policy_weight_ate` | [`test_policy_weight_parity.py`](../tests/reference_parity/test_policy_weight_parity.py) |
| `policy_weight_marginal` | [`test_policy_weight_parity.py`](../tests/reference_parity/test_policy_weight_parity.py) |
| `policy_weight_observed_prte` | [`test_policy_weight_parity.py`](../tests/reference_parity/test_policy_weight_parity.py) |
| `policy_weight_subsidy` | [`test_policy_weight_parity.py`](../tests/reference_parity/test_policy_weight_parity.py) |
| `power_ols` | [`test_recovery_batch_parity.py`](../tests/reference_parity/test_recovery_batch_parity.py) |
| `pretrends_equivalence` | [`test_fect_equivalence_parity.py`](../tests/reference_parity/test_fect_equivalence_parity.py) |
| `principal_strat` | [`test_principal_strat_parity.py`](../tests/reference_parity/test_principal_strat_parity.py) |
| `prod_fn` | [`test_structural_parity.py`](../tests/reference_parity/test_structural_parity.py) |
| `proximal` | [`test_proximal_parity.py`](../tests/reference_parity/test_proximal_parity.py) |
| `proximal_surrogate_index` | [`test_surrogate_parity.py`](../tests/reference_parity/test_surrogate_parity.py) |
| `psmatch2` | [`test_psmatch2_parity.py`](../tests/reference_parity/test_psmatch2_parity.py) |
| `qte_hd_panel` | [`test_hd_panel_qte.py`](../tests/reference_parity/test_hd_panel_qte.py) |
| `quasi_untreated_test` | [`test_did_had_parity.py`](../tests/reference_parity/test_did_had_parity.py) |
| `rate` | [`test_forest_rate_honest_parity.py`](../tests/reference_parity/test_forest_rate_honest_parity.py) |
| `rd_honest` | [`test_rdhonest_parity.py`](../tests/reference_parity/test_rdhonest_parity.py) |
| `rdmc` | [`test_rdmulti_parity.py`](../tests/reference_parity/test_rdmulti_parity.py) |
| `rdpower` | [`test_rdlocrand_parity.py`](../tests/reference_parity/test_rdlocrand_parity.py) |
| `rdrandinf` | [`test_rdlocrand_parity.py`](../tests/reference_parity/test_rdlocrand_parity.py) |
| `rdwinselect` | [`test_rdlocrand_parity.py`](../tests/reference_parity/test_rdlocrand_parity.py) |
| `regime` | [`test_longitudinal_parity.py`](../tests/reference_parity/test_longitudinal_parity.py) |
| `replicate` | [`test_castle_stata_parity.py`](../tests/reference_parity/test_castle_stata_parity.py) (+1) |
| `ri_test` | [`test_recovery_batch2_parity.py`](../tests/reference_parity/test_recovery_batch2_parity.py) |
| `rifreg` | [`test_decomposition_family_parity.py`](../tests/reference_parity/test_decomposition_family_parity.py) |
| `rlasso_effects` | [`test_rlasso_parity.py`](../tests/reference_parity/test_rlasso_parity.py) |
| `romano_wolf` | [`test_romano_wolf_parity.py`](../tests/reference_parity/test_romano_wolf_parity.py) |
| `sac` | [`test_spatial_models_parity.py`](../tests/reference_parity/test_spatial_models_parity.py) |
| `sarar_gmm` | [`test_spatial_models_parity.py`](../tests/reference_parity/test_spatial_models_parity.py) |
| `selection_bounds` | [`test_selection_bounds_parity.py`](../tests/reference_parity/test_selection_bounds_parity.py) |
| `shapley_inequality` | [`test_decomposition_family_parity.py`](../tests/reference_parity/test_decomposition_family_parity.py) |
| `sharp_ope_unobserved` | [`test_ope_parity.py`](../tests/reference_parity/test_ope_parity.py) |
| `slx` | [`test_spatial_models_parity.py`](../tests/reference_parity/test_spatial_models_parity.py) |
| `spatial_did` | [`test_spatial_models_parity.py`](../tests/reference_parity/test_spatial_models_parity.py) (+1) |
| `spatial_iv` | [`test_spatial_models_parity.py`](../tests/reference_parity/test_spatial_models_parity.py) |
| `spatial_panel` | [`test_spatial_models_parity.py`](../tests/reference_parity/test_spatial_models_parity.py) |
| `spillover` | [`test_interference_parity.py`](../tests/reference_parity/test_interference_parity.py) |
| `spillover_did` | [`test_spillover_rings.py`](../tests/reference_parity/test_spillover_rings.py) |
| `sqreg` | [`test_sqreg_parity.py`](../tests/reference_parity/test_sqreg_parity.py) |
| `ssaggregate` | [`test_bartik_ssagg_parity.py`](../tests/reference_parity/test_bartik_ssagg_parity.py) |
| `stabilized_weights` | [`test_stabilized_weights_parity.py`](../tests/reference_parity/test_stabilized_weights_parity.py) |
| `staggered_rollout` | [`test_staggered_rollout_parity.py`](../tests/reference_parity/test_staggered_rollout_parity.py) |
| `stepwise` | [`test_stepwise_parity.py`](../tests/reference_parity/test_stepwise_parity.py) |
| `stochastic_dominance` | [`test_distributional_te_parity.py`](../tests/reference_parity/test_distributional_te_parity.py) |
| `structural_break` | [`test_structural_break_parity.py`](../tests/reference_parity/test_structural_break_parity.py) |
| `subcluster_wild_bootstrap` | [`test_wcb_recovery_parity.py`](../tests/reference_parity/test_wcb_recovery_parity.py) |
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
| `wooldridge_did` | [`test_did_variants_parity.py`](../tests/reference_parity/test_did_variants_parity.py) |
| `wooldridge_prod` | [`test_structural_parity.py`](../tests/reference_parity/test_structural_parity.py) |
| `xlearner` | [`test_ml_causal_recovery_parity.py`](../tests/reference_parity/test_ml_causal_recovery_parity.py) |
| `xtdpdsys` | [`test_dynpanel_abdata_parity.py`](../tests/reference_parity/test_dynpanel_abdata_parity.py) |
| `xtlsdvc` | [`test_lsdvc_parity.py`](../tests/reference_parity/test_lsdvc_parity.py) |

## unverified — 775 functions

These are registered public functions with no cross-language or published-reference parity evidence attached **yet**. This is the honest coverage gap, not a claim of incorrectness — many are frontier methods with no Stata/R sibling to align against. Query any of them with `sp.parity_status(name)`; the closing roadmap lives in [`docs/dev/parity_status_roadmap.md`](dev/parity_status_roadmap.md).
