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
| bit-exact | 84 |
| aligned | 7 |
| analytical-only | 51 |
| external-replication | 4 |
| **verified (subtotal)** | **146** |
| unverified | 993 |
| **total registered** | **1139** |

## bit-exact — 84 functions

Machine-tolerance agreement with a named R/Stata reference.

| function | reference | versions | tolerance | rel err (R / Stata) | test |
| --- | --- | --- | --- | --- | --- |
| `adjust_pvalues` | base R stats::p.adjust (bonferroni/holm/BH) | R 4.5.2 | exact (atol 1e-15; observed 0) | — / — | [`test_mht_parity.py`](../tests/reference_parity/test_mht_parity.py) (+1) |
| `arima` | stats::arima | R 4.5.2; stats 4.5.2 | rel_est<=1e-06, rel_se<=1e-06 | 7.4e-07 / 9.3e-09 | [`39_arima.py`](../tests/r_parity/39_arima.py) (+2) |
| `attributable_risk` | base-R closed form (attributable fraction exposed + PAF) | R 4.5.2 | AFE + PAF point estimates 1e-12 abs (observed 0); CI not pinned | — / — | [`test_epi_extra_parity.py`](../tests/reference_parity/test_epi_extra_parity.py) (+1) |
| `bacon_decomposition` | bacondecomp::bacon | R 4.5.2; bacondecomp 0.1.1 | rel_est<=1e-06, rel_se<=1e-06 | 5.6e-16 / 9.6e-09 | [`20_bacon.py`](../tests/r_parity/20_bacon.py) (+2) |
| `benjamini_hochberg` | base R stats::p.adjust(method='BH') | R 4.5.2 | exact (atol 1e-15; observed 0) | — / — | [`test_mht_parity.py`](../tests/reference_parity/test_mht_parity.py) (+1) |
| `betareg` | betareg::betareg(link.phi="log") | R 4.5.2; betareg 3.2.4 | rel_est<=1e-06, rel_se<=0.01 | 2.2e-08 / 3.1e-08 | [`61_betareg.py`](../tests/r_parity/61_betareg.py) (+2) |
| `bonferroni` | base R stats::p.adjust(method='bonferroni') | R 4.5.2 | exact (atol 1e-15; observed 0) | — / — | [`test_mht_parity.py`](../tests/reference_parity/test_mht_parity.py) (+1) |
| `callaway_santanna` | did::att_gt + aggte | R 4.5.2; did 2.3.0 | rel_est<=1e-06, rel_se<=0.01 | 1.3e-15 / 1.3e-15 | [`04_csdid.py`](../tests/r_parity/04_csdid.py) (+2) |
| `clogit` | survival::clogit | R 4.5.2; survival 3.8.3 | rel_est<=1e-06, rel_se<=1e-06 | 1.3e-08 / 1.3e-08 | [`46_clogit.py`](../tests/r_parity/46_clogit.py) (+2) |
| `cohen_kappa` | base-R closed form (Cohen's kappa point estimate) | R 4.5.2 | kappa + agreements 1e-12 abs (observed ~1e-16); SE not pinned | — / — | [`test_epi_extra_parity.py`](../tests/reference_parity/test_epi_extra_parity.py) (+1) |
| `cox` | survival::coxph | R 4.5.2; survival 3.8.3 | rel_est<=1e-06, rel_se<=1e-06 | 8.4e-16 / 2.1e-10 | [`24_coxph.py`](../tests/r_parity/24_coxph.py) (+2) |
| `cr2_se` | clubSandwich::vcovCR(type="CR2"/"CR3") | R 4.5.2; clubSandwich 0.6.2 | rel_est<=1e-06, rel_se<=1e-06 | 1.8e-08 / 2.2e-08 | [`53_cr2.py`](../tests/r_parity/53_cr2.py) (+2) |
| `decompose` | oaxaca::oaxaca | R 4.5.2; oaxaca 0.1.5 | rel_est<=1e-06, rel_se<=0.05 | 6.3e-16 / 1.3e-16 | [`30_oaxaca.py`](../tests/r_parity/30_oaxaca.py) (+2) |
| `dfl_decompose` | ddecompose::dfl_decompose | R 4.5.2; ddecompose 1.0.0 | rel_est<=1e-06, rel_se<=1e-06 | 1.2e-09 / 1.8e-13 | [`31_dfl.py`](../tests/r_parity/31_dfl.py) (+2) |
| `did_imputation` | didimputation::did_imputation | R 4.5.2; didimputation 0.5.1 | rel_est<=1e-06, rel_se<=1e-06 | 4.8e-08 / 3.5e-07 | [`16_bjs.py`](../tests/r_parity/16_bjs.py) (+2) |
| `dml` | DoubleML::DoubleMLPLR | R 4.5.2; DoubleML 1.0.2 | rel_est<=1e-10, rel_se<=1e-10 | 0 / 3.7e-15 | [`08_dml.py`](../tests/r_parity/08_dml.py) (+2) |
| `drdid` | DRDID::drdid_imp_panel | R 4.5.2; DRDID 1.2.3 | rel_est<=1e-06, rel_se<=1e-06 | 2.6e-15 / 2.2e-16 | [`38_drdid.py`](../tests/r_parity/38_drdid.py) (+2) |
| `etwfe` | etwfe::etwfe + emfx | R 4.5.2; etwfe 0.6.2 | rel_est<=1e-06, rel_se<=0.001 | 1.8e-13 / 3.9e-14 | [`17_etwfe.py`](../tests/r_parity/17_etwfe.py) (+2) |
| `etwfe_emfx` | etwfe::etwfe + emfx | R 4.5.2; etwfe 0.6.2 | rel_est<=1e-06, rel_se<=0.001 | 1.8e-13 / 3.9e-14 | [`17_etwfe.py`](../tests/r_parity/17_etwfe.py) (+2) |
| `evalue` | EValue::evalues.RR | R 4.2.3; EValue 4.1.4 | rel_est<=1e-06, rel_se<=1e-06 | 5.8e-14 / 1.2e-16 | [`23_evalue.py`](../tests/r_parity/23_evalue.py) (+2) |
| `evalue_rr` | VanderWeele-Ding closed form (= R EValue package) | R 4.5.2 | point + CI E-value 1e-12 abs (observed 0) | — / — | [`test_evalue_rr_parity.py`](../tests/reference_parity/test_evalue_rr_parity.py) |
| `feols` | fixest::feols | R 4.5.2; fixest 0.14.0 | rel_est<=1e-06, rel_se<=1e-06 | 5.2e-15 / 2.9e-15 | [`03_hdfe.py`](../tests/r_parity/03_hdfe.py) (+2) |
| `fracreg` | stats::glm(quasibinomial('logit')) [fractional response] | R 4.5.2 | coefficients 1e-10 abs (observed ~8e-15) | — / — | [`test_glm_ext_parity.py`](../tests/reference_parity/test_glm_ext_parity.py) (+1) |
| `frontier` | sfaR::sfacross | R 4.5.2; sfaR 1.0.1 | rel_est<=1e-06, rel_se<=5e-05 | 4.1e-08 / 4.0e-08 | [`28_frontier.py`](../tests/r_parity/28_frontier.py) (+2) |
| `g_computation` | base R stats::lm g-formula standardization (Robins 1986) | — | psi 1e-8 (observed <= 7e-16; bootstrap SE pinned loosely +/-25%) | — / — | [`test_gformula_parity.py`](../tests/reference_parity/test_gformula_parity.py) (+1) |
| `gsynth` | gsynth::gsynth | R 4.5.2; gsynth 1.4.0 | rel_est<=1e-06, rel_se<=1e-06 | 7.7e-14 / — | [`19_gsynth.py`](../tests/r_parity/19_gsynth.py) (+1) |
| `heckman` | sampleSelection::heckit | R 4.5.2; sampleSelection 1.2.14 | rel_est<=1e-06, rel_se<=0.0005 | 1.0e-11 / 1.0e-11 | [`43_heckman.py`](../tests/r_parity/43_heckman.py) (+2) |
| `het_test` | lmtest::bptest (studentized Breusch-Pagan) | R 4.5.2; lmtest 0.9.40 | statistic & p-value 1e-10 rel (observed ~1e-13) | — / — | [`test_diagnostics_parity.py`](../tests/reference_parity/test_diagnostics_parity.py) (+1) |
| `holm` | base R stats::p.adjust(method='holm') | R 4.5.2 | exact (atol 1e-15; observed 0) | — / — | [`test_mht_parity.py`](../tests/reference_parity/test_mht_parity.py) (+1) |
| `honest_did` | HonestDiD::createSensitivityResults_relativeMagnitudes | R 4.5.2; HonestDiD 0.2.8 | abs_est<=1e-06, abs_se<=1e-06 | 4.4e-16 / 5.6e-17 | [`21_honest_relmags.py`](../tests/r_parity/21_honest_relmags.py) (+2) |
| `hurdle` | pscl::hurdle(dist='poisson', zero.dist='binomial') | R 4.5.2; pscl 1.5.9 | count + zero coefficients 1e-6 abs (observed ~2e-8) | — / — | [`test_glm_ext_parity.py`](../tests/reference_parity/test_glm_ext_parity.py) (+1) |
| `incidence_rate_ratio` | base-R closed form (rate ratio + conditional-binomial exact CI) | R 4.5.2 | estimate 1e-12; exact CI 1e-10 abs (observed ~3e-15) | — / — | [`test_epi_parity.py`](../tests/reference_parity/test_epi_parity.py) (+1) |
| `inequality_index` | base-R closed form (Gini/Theil-T/Theil-L/Atkinson; = ineq) | R 4.5.2 | all indices 1e-12 abs (observed ~2e-16) | — / — | [`test_inequality_parity.py`](../tests/reference_parity/test_inequality_parity.py) (+1) |
| `ipw` | base R stats::glm(binomial) + hand-rolled Hajek weighted means | — | Hajek ATE/ATT estimate 1e-9 (observed <= 2e-15; SE not pinned) | — / — | [`test_ipw_parity.py`](../tests/reference_parity/test_ipw_parity.py) (+1) |
| `ivreg` | AER::ivreg | R 4.5.2; AER 1.2.16 | rel_est<=1e-06, rel_se<=1e-06 | 1.1e-11 / 1.1e-11 | [`02_iv.py`](../tests/r_parity/02_iv.py) (+2) |
| `kaplan_meier` | survival::survfit | R 4.5.2; survival 3.8.3 | S(t) at every event time 1e-12 (observed ~3e-17); median exact | — / — | [`test_survival_km_parity.py`](../tests/reference_parity/test_survival_km_parity.py) (+1) |
| `liml` | ivmodel::LIML | R 4.5.2; ivmodel 1.9.1 | rel_est<=1e-06, rel_se<=1e-06 | 1.7e-15 / 3.0e-16 | [`59_liml.py`](../tests/r_parity/59_liml.py) (+2) |
| `local_projections` | lpirfs::lp_lin | R 4.5.2; lpirfs 0.2.5 | rel_est<=1e-06, rel_se<=1e-06 | 5.0e-15 / — | [`34_lp.py`](../tests/r_parity/34_lp.py) (+1) |
| `logit` | stats::glm(family=binomial("logit")) | R 4.5.2; stats 4.5.2 | rel_est<=1e-06, rel_se<=1e-06 | 2.7e-11 / 2.7e-11 | [`57_logit.py`](../tests/r_parity/57_logit.py) (+2) |
| `logrank_test` | survival::survdiff | R 4.5.2; survival 3.8.3 | chi-square 1e-10 rel (observed ~8e-16); p-value 1e-10 abs | — / — | [`test_survival_km_parity.py`](../tests/reference_parity/test_survival_km_parity.py) (+1) |
| `mantel_haenszel` | base-R closed form (Robins-Breslow-Greenland MH; = epiR) | R 4.5.2 | estimate, se_log, CI 1e-12 abs (observed 0) | — / — | [`test_epi_parity.py`](../tests/reference_parity/test_epi_parity.py) (+1) |
| `mde` | base-R closed form (RCT minimum detectable effect) | R 4.5.2 | effect size 1e-6 abs (output rounded to 6 dp; observed ~2e-8) | — / — | [`test_power_extra_parity.py`](../tests/reference_parity/test_power_extra_parity.py) (+1) |
| `mediate` | mediation::mediate | R 4.5.2; mediation 4.5.1 | rel_est<=1e-06, rel_se<=0.1 | 6.7e-15 / 3.6e-15 | [`36_mediation.py`](../tests/r_parity/36_mediation.py) (+2) |
| `mediation` | mediation::mediate | R 4.5.2; mediation 4.5.1 | rel_est<=1e-06, rel_se<=0.1 | 6.7e-15 / 3.6e-15 | [`36_mediation.py`](../tests/r_parity/36_mediation.py) (+2) |
| `melogit` | lme4::glmer(nAGQ=8) | R 4.5.2; lme4 2.0.1 | rel_est<=1e-06, rel_se<=0.05 | 2.4e-07 / 8.4e-07 | [`27_glmm_aghq.py`](../tests/r_parity/27_glmm_aghq.py) (+2) |
| `mixed` | lme4::lmer | R 4.5.2; lme4 2.0.1 | rel_est<=1e-06, rel_se<=1e-06 | 1.3e-10 / 4.9e-11 | [`25_lmm.py`](../tests/r_parity/25_lmm.py) (+2) |
| `mlogit` | nnet::multinom | R 4.5.2; nnet 7.3.20 | rel_est<=1e-06, rel_se<=5e-05 | 2.6e-07 / 7.4e-09 | [`44_mlogit.py`](../tests/r_parity/44_mlogit.py) (+2) |
| `multiway_cluster_vcov` | sandwich::vcovCL(cluster=~g1+g2+g3) | R 4.5.2; sandwich 3.1.1 | rel_est<=1e-06, rel_se<=1e-06 | 2.1e-15 / 2.1e-15 | [`56_multiway_cluster.py`](../tests/r_parity/56_multiway_cluster.py) (+2) |
| `nbreg` | MASS::glm.nb | R 4.5.2; MASS 7.3.65 | rel_est<=1e-06, rel_se<=0.005 | 6.0e-10 / 1.3e-10 | [`42_nbreg.py`](../tests/r_parity/42_nbreg.py) (+2) |
| `number_needed_to_treat` | base-R closed form (NNT = 1/risk difference) | R 4.5.2 | estimate 1e-12 abs (observed 0); CI not pinned | — / — | [`test_epi_parity.py`](../tests/reference_parity/test_epi_parity.py) (+1) |
| `oaxaca` | oaxaca::oaxaca | R 4.5.2; oaxaca 0.1.5 | rel_est<=1e-06, rel_se<=0.05 | 6.3e-16 / 1.3e-16 | [`30_oaxaca.py`](../tests/r_parity/30_oaxaca.py) (+2) |
| `odds_ratio` | base-R closed form (Woolf logit; = epiR::epi.2by2) | R 4.5.2 | estimate, se_log, CI 1e-12 abs (observed 0) | — / — | [`test_epi_parity.py`](../tests/reference_parity/test_epi_parity.py) (+1) |
| `ologit` | MASS::polr(method="logistic") | R 4.5.2; MASS 7.3.65 | rel_est<=1e-06, rel_se<=1e-05 | 1.8e-07 / 3.5e-07 | [`45_ologit.py`](../tests/r_parity/45_ologit.py) (+2) |
| `oprobit` | MASS::polr(method="probit") | R 4.5.2; MASS 7.3.65 | rel_est<=1e-06, rel_se<=1e-06 | 3.4e-07 / 2.8e-08 | [`49_oprobit.py`](../tests/r_parity/49_oprobit.py) (+2) |
| `panel` | plm::plm + plm::phtest | R 4.5.2; plm 2.6.7 | rel_est<=1e-06, rel_se<=0.001 | 4.7e-14 / 1.5e-15 | [`35_panel.py`](../tests/r_parity/35_panel.py) (+2) |
| `poisson` | stats::glm(family=poisson()) | R 4.5.2; stats 4.5.2 | rel_est<=1e-06, rel_se<=1e-06 | 9.2e-15 / 8.7e-12 | [`58_poisson.py`](../tests/r_parity/58_poisson.py) (+2) |
| `power_cluster_rct` | base-R closed form (design-effect-inflated z-approx power) | R 4.5.2 | power 1e-12 abs (observed ~2e-16) | — / — | [`test_power_extra_parity.py`](../tests/reference_parity/test_power_extra_parity.py) (+1) |
| `power_logrank` | base-R closed form (Schoenfeld log-rank power) | R 4.5.2 | power 1e-12 abs (observed ~2e-16) | — / — | [`test_power_parity.py`](../tests/reference_parity/test_power_parity.py) (+1) |
| `power_rct` | base-R closed form (two-sample pooled-sigma z-approx power) | R 4.5.2 | power 1e-12 abs (observed ~2e-16) | — / — | [`test_power_parity.py`](../tests/reference_parity/test_power_parity.py) (+1) |
| `power_two_proportions` | base-R closed form (unpooled Wald two-proportion z-approx) | R 4.5.2 | power 1e-12 abs (observed ~2e-16) | — / — | [`test_power_parity.py`](../tests/reference_parity/test_power_parity.py) (+1) |
| `ppmlhdfe` | fixest::fepois | R 4.5.2; fixest 0.14.0 | rel_est<=1e-06, rel_se<=0.01 | 4.9e-13 / 2.2e-15 | [`37_ppmlhdfe.py`](../tests/r_parity/37_ppmlhdfe.py) (+2) |
| `prevalence_ratio` | base-R closed form (Katz-log; = epiR::epi.2by2) | R 4.5.2 | estimate, se_log, CI 1e-12 abs (observed ~2e-16) | — / — | [`test_epi_parity.py`](../tests/reference_parity/test_epi_parity.py) (+1) |
| `probit` | stats::glm(family=binomial("probit")) | R 4.5.2; stats 4.5.2 | rel_est<=1e-06, rel_se<=0.01 | 3.1e-07 / 1.6e-08 | [`48_probit.py`](../tests/r_parity/48_probit.py) (+2) |
| `psm` | MatchIt::matchit | R 4.5.2; MatchIt 4.7.2 | rel_est<=1e-06, rel_se<=1e-06 | 1.2e-15 / 2.0e-16 | [`11_psm.py`](../tests/r_parity/11_psm.py) (+2) |
| `qreg` | quantreg::rq | R 4.5.2; quantreg 6.1 | rel_est<=1e-06, rel_se<=0.1 | 3.3e-15 / 4.4e-15 | [`40_qreg.py`](../tests/r_parity/40_qreg.py) (+2) |
| `rddensity` | rddensity::rddensity | R 4.5.2; rddensity 2.6 | rel_est<=1e-06, rel_se<=1e-06 | 3.3e-11 / 8.9e-11 | [`09_rddensity.py`](../tests/r_parity/09_rddensity.py) (+2) |
| `rdrobust` | rdrobust::rdrobust | R 4.5.2; rdrobust 3.0.0 | rel_est<=1e-06, rel_se<=0.1 | 7.9e-13 / 2.4e-10 | [`06_rd.py`](../tests/r_parity/06_rd.py) (+2) |
| `regress` | lm + sandwich::vcovHC | R 4.5.2; sandwich 3.1.1 | rel_est<=1e-06, rel_se<=1e-06 | 1.1e-12 / 1.3e-12 | [`01_ols.py`](../tests/r_parity/01_ols.py) (+2) |
| `relative_risk` | base-R closed form (Katz-log; = epiR::epi.2by2 / Stata epitab) | R 4.5.2 | estimate, se_log, CI 1e-12 abs (observed 0) | — / — | [`test_epi_parity.py`](../tests/reference_parity/test_epi_parity.py) (+1) |
| `reset_test` | lmtest::resettest(power=2:3, type='fitted') | R 4.5.2; lmtest 0.9.40 | F-statistic & p-value 1e-10 rel (observed ~1e-13) | — / — | [`test_diagnostics_parity.py`](../tests/reference_parity/test_diagnostics_parity.py) (+1) |
| `rif_decomposition` | dineq::rif + manual OLS | R 4.5.2; dineq 0.1.0 | rel_est<=1e-06, rel_se<=1e-06 | 2.2e-15 / 1.4e-16 | [`32_rif.py`](../tests/r_parity/32_rif.py) (+2) |
| `risk_difference` | base-R closed form (Wald; = epiR::epi.2by2 / Stata epitab) | R 4.5.2 | estimate, se, CI 1e-12 abs (observed 0) | — / — | [`test_epi_parity.py`](../tests/reference_parity/test_epi_parity.py) (+1) |
| `sdid` | synthdid::synthdid_estimate | R 4.5.2; synthdid 0.0.9 | rel_est<=1e-06, rel_se<=1e-06 | 2.6e-15 / 7.2e-08 | [`12_sdid.py`](../tests/r_parity/12_sdid.py) (+2) |
| `sensemakr` | sensemakr::sensemakr | R 4.5.2; sensemakr 0.1.6 | rel_est<=1e-06, rel_se<=1e-06 | 5.0e-08 / 5.0e-08 | [`22_sensemakr.py`](../tests/r_parity/22_sensemakr.py) (+2) |
| `sun_abraham` | fixest::sunab | R 4.5.2; fixest 0.14.0 | rel_est<=1e-06, rel_se<=0.25 | 2.8e-11 / 2.7e-11 | [`05_sunab.py`](../tests/r_parity/05_sunab.py) (+2) |
| `sureg` | systemfit::systemfit(method="SUR", noDfCor) | R 4.5.2; systemfit 1.1.30 | rel_est<=1e-06, rel_se<=1e-06 | 1.5e-14 / 1.5e-15 | [`60_sureg.py`](../tests/r_parity/60_sureg.py) (+2) |
| `synth` | Synth::synth | R 4.5.2; Synth 1.1.10 | rel_est<=1e-06, rel_se<=1e-06 | 7.8e-08 / 7.7e-08 | [`52_scm_unique.py`](../tests/r_parity/52_scm_unique.py) (+2) |
| `tmle` | base R stats::glm TMLE (van der Laan & Rubin 2006) | — | psi 1e-9 (observed 5.6e-12), EIF SE 1e-9, epsilon 1e-8 | — / — | [`test_tmle_parity.py`](../tests/reference_parity/test_tmle_parity.py) (+1) |
| `tobit` | censReg::censReg | R 4.5.2; censReg 0.5.38 | rel_est<=1e-06, rel_se<=1e-05 | 2.8e-08 / 2.8e-08 | [`41_tobit.py`](../tests/r_parity/41_tobit.py) (+2) |
| `truncreg` | truncreg::truncreg(method="NR") | R 4.5.2; truncreg 0.2.5 | rel_est<=1e-06, rel_se<=0.0001 | 3.3e-08 / 9.5e-08 | [`62_truncreg.py`](../tests/r_parity/62_truncreg.py) (+2) |
| `twoway_cluster` | sandwich::vcovCL(cluster=~g1+g2) | R 4.5.2; sandwich 3.1.1 | rel_est<=1e-06, rel_se<=1e-06 | 7.8e-16 / 7.8e-16 | [`54_twoway_cluster.py`](../tests/r_parity/54_twoway_cluster.py) (+2) |
| `var` | vars::VAR | R 4.5.2; vars 1.6.1 | rel_est<=1e-06, rel_se<=0.001 | 3.1e-15 / 6.6e-15 | [`33_var.py`](../tests/r_parity/33_var.py) (+2) |
| `xtabond` | plm::pgmm | R 4.5.2; plm 2.6.7 | rel_est<=1e-06, rel_se<=1e-06 | 9.0e-16 / 1.4e-15 | [`50_xtabond.py`](../tests/r_parity/50_xtabond.py) (+2) |
| `zip_model` | pscl::zeroinfl(dist="poisson") | R 4.5.2; pscl 1.5.9 | rel_est<=1e-06, rel_se<=0.0001 | 7.7e-08 / 1.1e-07 | [`63_zip.py`](../tests/r_parity/63_zip.py) (+2) |

## aligned — 7 functions

Agreement within a documented, pre-registered looser tolerance.

| function | reference | versions | tolerance | rel err (R / Stata) | test |
| --- | --- | --- | --- | --- | --- |
| `aft` | survival::survreg (Weibull AFT) | R 4.5.2; survival 3.8.3 | coefficients & log-scale 5e-5 abs (observed ~1e-5) | — / — | [`test_aft_parity.py`](../tests/reference_parity/test_aft_parity.py) (+1) |
| `augsynth` | augsynth::augsynth | R 4.5.2; augsynth 0.2.0 | rel_est<=2e-05, rel_se<=1e-06 | 7.9e-06 / — | [`18_augsynth.py`](../tests/r_parity/18_augsynth.py) (+1) |
| `causal_forest` | grf::causal_forest | R 4.5.2; grf 2.6.1 | rel_est<=0.005, rel_se<=0.5 | 1.9e-03 / — | [`13_causal_forest.py`](../tests/r_parity/13_causal_forest.py) (+1) |
| `cloglog` | stats::glm(binomial('cloglog')) | R 4.5.2 | coefficients 5e-5 abs (observed ~1e-5; IRLS convergence) | — / — | [`test_glm_ext_parity.py`](../tests/reference_parity/test_glm_ext_parity.py) (+1) |
| `survreg` | survival::survreg (Weibull AFT) | R 4.5.2; survival 3.8.3 | coefficients & log-scale 5e-5 abs (observed ~1e-5) | — / — | [`test_aft_parity.py`](../tests/reference_parity/test_aft_parity.py) (+1) |
| `xtfrontier` | frontier::sfa | R 4.5.2; frontier 1.1.8 | rel_est<=0.001, rel_se<=0.001 | 2.8e-06 / 8.6e-04 | [`29_panel_sfa.py`](../tests/r_parity/29_panel_sfa.py) (+2) |
| `zinb` | pscl::zeroinfl(dist="negbin") | R 4.5.2; pscl 1.5.9 | rel_est<=1e-05, rel_se<=0.001 | 1.1e-06 / 2.1e-07 | [`64_zinb.py`](../tests/r_parity/64_zinb.py) (+2) |

## external-replication — 4 functions

Reproduces published-paper numbers; sources in `tests/external_parity/PUBLISHED_REFERENCE_VALUES.md`.

| function | test |
| --- | --- |
| `aggte` | [`test_honest_did_paper_parity.py`](../tests/external_parity/test_honest_did_paper_parity.py) (+1) |
| `breakdown_m` | [`test_honest_did_paper_parity.py`](../tests/external_parity/test_honest_did_paper_parity.py) |
| `g_estimation` | [`test_whatif_nhefs.py`](../tests/external_parity/test_whatif_nhefs.py) |
| `metalearner` | [`test_causalml_book.py`](../tests/external_parity/test_causalml_book.py) |

## analytical-only — 51 functions

Recovers a known DGP truth / closed-form identity within tolerance; no cross-package reference. See `tests/reference_parity/REFERENCES.md`.

| function | test |
| --- | --- |
| `aipw` | [`test_paper_parity.py`](../tests/reference_parity/test_paper_parity.py) (+1) |
| `assimilative_causal` | [`test_assimilation_parity.py`](../tests/reference_parity/test_assimilation_parity.py) |
| `average_treatment_effect` | [`test_forest_ate_parity.py`](../tests/reference_parity/test_forest_ate_parity.py) |
| `bcf` | [`test_bcf_parity.py`](../tests/reference_parity/test_bcf_parity.py) |
| `bidirectional_pci` | [`test_proximal_parity.py`](../tests/reference_parity/test_proximal_parity.py) |
| `bunching` | [`test_bunching_parity.py`](../tests/reference_parity/test_bunching_parity.py) |
| `causal_kalman` | [`test_assimilation_parity.py`](../tests/reference_parity/test_assimilation_parity.py) |
| `cbps` | [`test_matching_parity.py`](../tests/reference_parity/test_matching_parity.py) (+1) |
| `continuous_did` | [`test_dose_response_parity.py`](../tests/reference_parity/test_dose_response_parity.py) |
| `did_multiplegt` | [`test_did_multiplegt_parity.py`](../tests/reference_parity/test_did_multiplegt_parity.py) |
| `discos` | [`test_distributional_te_parity.py`](../tests/reference_parity/test_distributional_te_parity.py) |
| `distributional_te` | [`test_distributional_te_parity.py`](../tests/reference_parity/test_distributional_te_parity.py) |
| `dose_response` | [`test_dose_response_parity.py`](../tests/reference_parity/test_dose_response_parity.py) |
| `ebalance` | [`test_matching_parity.py`](../tests/reference_parity/test_matching_parity.py) (+1) |
| `fortified_pci` | [`test_proximal_parity.py`](../tests/reference_parity/test_proximal_parity.py) |
| `gardner_did` | [`test_did_variants_parity.py`](../tests/reference_parity/test_did_variants_parity.py) |
| `general_bunching` | [`test_bunching_parity.py`](../tests/reference_parity/test_bunching_parity.py) |
| `genmatch` | [`test_matching_recovery_parity.py`](../tests/reference_parity/test_matching_recovery_parity.py) |
| `gmm` | [`test_gmm_dynamic_panel_parity.py`](../tests/reference_parity/test_gmm_dynamic_panel_parity.py) |
| `hdfe_ols` | [`test_hdfe_parity.py`](../tests/reference_parity/test_hdfe_parity.py) |
| `interference` | [`test_interference_parity.py`](../tests/reference_parity/test_interference_parity.py) |
| `iv` | [`test_regress_weights_iv_robust_parity.py`](../tests/reference_parity/test_regress_weights_iv_robust_parity.py) |
| `lincom` | [`test_postestimation_parity.py`](../tests/reference_parity/test_postestimation_parity.py) |
| `lingam` | [`test_causal_discovery_parity.py`](../tests/reference_parity/test_causal_discovery_parity.py) |
| `margins` | [`test_postestimation_parity.py`](../tests/reference_parity/test_postestimation_parity.py) |
| `match` | [`test_matching_parity.py`](../tests/reference_parity/test_matching_parity.py) (+1) |
| `matrix_completion` | [`test_matrix_completion_parity.py`](../tests/reference_parity/test_matrix_completion_parity.py) |
| `mc_panel` | [`test_matrix_completion_parity.py`](../tests/reference_parity/test_matrix_completion_parity.py) |
| `mr_egger` | [`test_mr_parity.py`](../tests/reference_parity/test_mr_parity.py) |
| `mr_ivw` | [`test_mr_parity.py`](../tests/reference_parity/test_mr_parity.py) |
| `mr_leave_one_out` | [`test_mr_parity.py`](../tests/reference_parity/test_mr_parity.py) |
| `mr_median` | [`test_mr_parity.py`](../tests/reference_parity/test_mr_parity.py) |
| `mr_presso` | [`test_mr_parity.py`](../tests/reference_parity/test_mr_parity.py) |
| `mr_radial` | [`test_mr_parity.py`](../tests/reference_parity/test_mr_parity.py) |
| `network_exposure` | [`test_interference_parity.py`](../tests/reference_parity/test_interference_parity.py) |
| `notears` | [`test_causal_discovery_parity.py`](../tests/reference_parity/test_causal_discovery_parity.py) |
| `overlap_weights` | [`test_matching_parity.py`](../tests/reference_parity/test_matching_parity.py) |
| `particle_filter` | [`test_assimilation_parity.py`](../tests/reference_parity/test_assimilation_parity.py) |
| `pc_algorithm` | [`test_causal_discovery_parity.py`](../tests/reference_parity/test_causal_discovery_parity.py) |
| `principal_strat` | [`test_principal_strat_parity.py`](../tests/reference_parity/test_principal_strat_parity.py) |
| `proximal` | [`test_proximal_parity.py`](../tests/reference_parity/test_proximal_parity.py) |
| `psmatch2` | [`test_psmatch2_parity.py`](../tests/reference_parity/test_psmatch2_parity.py) |
| `qdid` | [`test_qte_parity.py`](../tests/reference_parity/test_qte_parity.py) |
| `qte` | [`test_qte_parity.py`](../tests/reference_parity/test_qte_parity.py) |
| `rlasso_effects` | [`test_rlasso_parity.py`](../tests/reference_parity/test_rlasso_parity.py) |
| `sbw` | [`test_matching_recovery_parity.py`](../tests/reference_parity/test_matching_recovery_parity.py) |
| `spillover` | [`test_interference_parity.py`](../tests/reference_parity/test_interference_parity.py) |
| `stochastic_dominance` | [`test_distributional_te_parity.py`](../tests/reference_parity/test_distributional_te_parity.py) |
| `survivor_average_causal_effect` | [`test_principal_strat_parity.py`](../tests/reference_parity/test_principal_strat_parity.py) |
| `test` | [`test_postestimation_parity.py`](../tests/reference_parity/test_postestimation_parity.py) |
| `wooldridge_did` | [`test_did_variants_parity.py`](../tests/reference_parity/test_did_variants_parity.py) |

## unverified — 993 functions

These are registered public functions with no cross-language or published-reference parity evidence attached **yet**. This is the honest coverage gap, not a claim of incorrectness — many are frontier methods with no Stata/R sibling to align against. Query any of them with `sp.parity_status(name)`; the closing roadmap lives in [`docs/dev/parity_status_roadmap.md`](dev/parity_status_roadmap.md).
