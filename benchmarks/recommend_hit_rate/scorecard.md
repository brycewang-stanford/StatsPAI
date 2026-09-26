# Recommendation Hit-Rate Scorecard

- corpus: `1.1.0-fifty`  |  statspai: `1.31.0`  |  entries: **50** (43 core + 7 frontier; 10 Tier-A + 40 Tier-B)
- **core top-1 hit-rate: 1.0**  |  top-k: 1.0  |  hard-miss rate: 0.0  |  errors: 0
- audit catalog mean recall (static): 1.0  |  audit dynamic mean recall (fit+audit): 1.0  |  audit errors: 0
- frontier coverage (gap-probe designs recommend is being taught): **1.0** (7/7)  |  frontier fit+audit OK: 7/7
- **end-to-end (all 50 cases: acceptable top-1 AND fitted AND audited): 50/50**

## recommend hit-rate (dynamic — runs on real / synthetic data)

_Frontier (gap-probe) designs are marked ⊕ and scored separately; they do not affect the core hit-rate or the CI ratchet._

| design | id | status | detected | top-1 tag |
| --- | --- | --- | --- | --- |
| staggered_did | `callaway_santanna_2021_mpdta` | ✓ HIT | did | `callaway_santanna` |
| iv | `angrist_krueger_1991_qob` | ✓ HIT | iv | `twosls` |
| iv | `card_1995_proximity` | ✓ HIT | iv | `twosls` |
| rd | `lee_2008_senate_rd` | ✓ HIT | rd | `local_polynomial_rd` |
| synth | `abadie_2010_prop99` | ✓ HIT | synth | `synth` |
| synth | `abadie_2003_basque` | ✓ HIT | synth | `synth` |
| synth | `abadie_2015_german` | ✓ HIT | synth | `synth` |
| observational | `dehejia_wahba_1999_nsw` | ✓ HIT | observational | `psm` |
| staggered_did | `trap_staggered_heterogeneous_twfe` | ✓ HIT | did | `callaway_santanna` |
| did | `trap_did_2x2_common_timing` | ✓ HIT | did | `classic_2x2` |
| iv | `trap_weak_instrument` | ✓ HIT | iv | `liml` |
| iv | `archetype_strong_instrument` | ✓ HIT | iv | `twosls` |
| rd | `archetype_sharp_rd` | ✓ HIT | rd | `local_polynomial_rd` |
| rd | `trap_fuzzy_rd` | ✓ HIT | rd | `local_polynomial_rd` |
| observational | `trap_observational_strong_confounding` | ✓ HIT | observational | `psm` |
| observational | `lalonde_1986_nsw_experimental` | ✓ HIT | observational | `psm` |
| did | `acemoglu_angrist_2001_ada` | ✓ HIT | did | `callaway_santanna` |
| did | `card_krueger_1994_minwage` | ✓ HIT | did | `classic_2x2` |
| iv | `angrist_1990_draft_lottery` | ✓ HIT | iv | `twosls` |
| iv | `acemoglu_2001_colonial_iv` | ✓ HIT | iv | `twosls` |
| rd | `dell_2010_mining_mita_rd` | ✓ HIT | rd | `local_polynomial_rd` |
| did | `duflo_2001_school_construction_did` | ✓ HIT | did | `callaway_santanna` |
| rd | `carpenter_dobkin_2009_mlda_rd` | ✓ HIT | rd | `local_polynomial_rd` |
| observational | `bertrand_mullainathan_2004_audit` | ✓ HIT | observational | `psm` |
| iv | `nunn_wantchekon_2011_slave_iv` | ✓ HIT | iv | `twosls` |
| did | `meyer_viscusi_durbin_1995_did` | ✓ HIT | did | `classic_2x2` |
| rd | `ludwig_miller_2007_headstart_rd` | ✓ HIT | rd | `local_polynomial_rd` |
| iv | `miguel_2004_rainfall_iv` | ✓ HIT | iv | `twosls` |
| rd | `angrist_lavy_1999_classsize_rd` | ✓ HIT | rd | `local_polynomial_rd` |
| synth | `card_1990_mariel_boatlift` | ✓ HIT | synth | `synth` |
| did | `dube_lester_reich_2010_border_did` | ✓ HIT | did | `callaway_santanna` |
| did | `ditella_schargrodsky_2004_police_did` | ✓ HIT | did | `classic_2x2` |
| iv | `angrist_evans_1998_familysize_iv` | ✓ HIT | iv | `twosls` |
| observational | `imbens_2001_lottery_obs` | ✓ HIT | observational | `psm` |
| iv ⊕ | `autor_dorn_hanson_2013_bartik` | ✓ HIT | bartik | `bartik` |
| iv | `oreopoulos_2006_schooling_iv` | ✓ HIT | iv | `twosls` |
| did | `galiani_2005_water_did` | ✓ HIT | did | `callaway_santanna` |
| rd | `card_dobkin_maestas_2008_medicare_rd` | ✓ HIT | rd | `local_polynomial_rd` |
| did | `almond_2006_flu_did` | ✓ HIT | did | `callaway_santanna` |
| did | `bleakley_2007_hookworm_did` | ✓ HIT | did | `callaway_santanna` |
| rd | `black_1999_school_boundary_rd` | ✓ HIT | rd | `local_polynomial_rd` |
| iv | `chay_greenstone_2005_airquality_iv` | ✓ HIT | iv | `twosls` |
| observational | `chetty_looney_kroft_2009_salience` | ✓ HIT | observational | `psm` |
| did | `finkelstein_2007_medicare_did` | ✓ HIT | did | `callaway_santanna` |
| bunching ⊕ | `frontier_bunching_saez2010` | ✓ HIT | bunching | `bunching` |
| rkd ⊕ | `frontier_rkd_card2015` | ✓ HIT | rkd | `rkd` |
| ddd ⊕ | `frontier_ddd_gruber1994` | ✓ HIT | ddd | `ddd` |
| bartik ⊕ | `frontier_bartik_gpss2020` | ✓ HIT | bartik | `bartik` |
| decomposition ⊕ | `frontier_decomposition_oaxaca1973` | ✓ HIT | decomposition | `oaxaca` |
| did ⊕ | `frontier_repeated_cross_sections_did` | ✓ HIT | did | `classic_2x2` |

## audit recall (dynamic — fit the estimator, run sp.audit, does it ask)

| id | stage | fitted family | recall | actionable next-steps |
| --- | --- | --- | --- | --- |
| `callaway_santanna_2021_mpdta` | ok | did | 1.0 | bacon_decomposition, honest_did |
| `angrist_krueger_1991_qob` | ok | iv | 1.0 | anderson_rubin_ci |
| `card_1995_proximity` | ok | iv | 1.0 | anderson_rubin_ci |
| `lee_2008_senate_rd` | ok | rd | 1.0 | bandwidth_sensitivity, placebo_cutoff |
| `abadie_2010_prop99` | ok | synth | 1.0 | placebo_inference |
| `abadie_2003_basque` | ok | synth | 1.0 | placebo_inference |
| `abadie_2015_german` | ok | synth | 1.0 | placebo_inference |
| `dehejia_wahba_1999_nsw` | ok | matching | 1.0 | overlap, balance_after, ovb_sensitivity |
| `trap_staggered_heterogeneous_twfe` | ok | did | 1.0 | bacon_decomposition, honest_did |
| `trap_did_2x2_common_timing` | ok | did | 1.0 | parallel_trends |
| `trap_weak_instrument` | ok | iv | 1.0 | anderson_rubin_ci |
| `archetype_strong_instrument` | ok | iv | 1.0 | — |
| `archetype_sharp_rd` | ok | rd | 1.0 | bandwidth_sensitivity, placebo_cutoff |
| `trap_fuzzy_rd` | ok | rd | 1.0 | bandwidth_sensitivity |
| `trap_observational_strong_confounding` | ok | matching | 1.0 | overlap, balance_after, ovb_sensitivity |
| `lalonde_1986_nsw_experimental` | ok | matching | 1.0 | overlap, balance_after |
| `acemoglu_angrist_2001_ada` | ok | did | 1.0 | parallel_trends |
| `card_krueger_1994_minwage` | ok | did | 1.0 | parallel_trends |
| `angrist_1990_draft_lottery` | ok | iv | 1.0 | anderson_rubin_ci |
| `acemoglu_2001_colonial_iv` | ok | iv | 1.0 | anderson_rubin_ci |
| `dell_2010_mining_mita_rd` | ok | rd | 1.0 | bandwidth_sensitivity |
| `duflo_2001_school_construction_did` | ok | did | 1.0 | parallel_trends |
| `carpenter_dobkin_2009_mlda_rd` | ok | rd | 1.0 | bandwidth_sensitivity |
| `bertrand_mullainathan_2004_audit` | ok | matching | 1.0 | balance_after |
| `nunn_wantchekon_2011_slave_iv` | ok | iv | 1.0 | anderson_rubin_ci |
| `meyer_viscusi_durbin_1995_did` | ok | did | 1.0 | parallel_trends |
| `ludwig_miller_2007_headstart_rd` | ok | rd | 1.0 | bandwidth_sensitivity |
| `miguel_2004_rainfall_iv` | ok | iv | 1.0 | anderson_rubin_ci |
| `angrist_lavy_1999_classsize_rd` | ok | rd | 1.0 | bandwidth_sensitivity |
| `card_1990_mariel_boatlift` | ok | synth | 1.0 | placebo_inference |
| `dube_lester_reich_2010_border_did` | ok | did | 1.0 | — |
| `ditella_schargrodsky_2004_police_did` | ok | did | 1.0 | parallel_trends |
| `angrist_evans_1998_familysize_iv` | ok | iv | 1.0 | anderson_rubin_ci |
| `imbens_2001_lottery_obs` | ok | matching | 1.0 | overlap, balance_after |
| `autor_dorn_hanson_2013_bartik` | ok | iv | 1.0 | — |
| `oreopoulos_2006_schooling_iv` | ok | iv | 1.0 | anderson_rubin_ci |
| `galiani_2005_water_did` | ok | did | 1.0 | — |
| `card_dobkin_maestas_2008_medicare_rd` | ok | rd | 1.0 | bandwidth_sensitivity |
| `almond_2006_flu_did` | ok | did | 1.0 | — |
| `bleakley_2007_hookworm_did` | ok | did | 1.0 | — |
| `black_1999_school_boundary_rd` | ok | rd | 1.0 | bandwidth_sensitivity |
| `chay_greenstone_2005_airquality_iv` | ok | iv | 1.0 | anderson_rubin_ci |
| `chetty_looney_kroft_2009_salience` | ok | matching | 1.0 | balance_after |
| `finkelstein_2007_medicare_did` | ok | did | 1.0 | — |
| `frontier_bunching_saez2010` | ok | rd | 0.0 | — |
| `frontier_rkd_card2015` | ok | rd | 1.0 | bandwidth_sensitivity |
| `frontier_ddd_gruber1994` | ok | did | 1.0 | parallel_trends |
| `frontier_bartik_gpss2020` | ok | iv | 0.0 | — |
| `frontier_decomposition_oaxaca1973` | ok | regression | 0.0 | — |
| `frontier_repeated_cross_sections_did` | ok | did | 1.0 | parallel_trends |

_Generated by sp.recommend_benchmark()_
