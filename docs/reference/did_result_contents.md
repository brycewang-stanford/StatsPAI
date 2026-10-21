# DiD result objects: what is where

Every DiD estimator returns a `CausalResult`. The headline is on the object
(`.estimate`, `.se`, `.ci`, `.pvalue`, `.n_obs`); the tables and settings
behind it are in `.detail` and the `.model_info` dict. This page lists those
keys for the main estimators, so scripts do not have to guess.

## Reading the tables: three accessors and `tidy()`

You rarely need `model_info` for the tables themselves:

| accessor | returns | available on |
| --- | --- | --- |
| `result.attgt` | ATT(g,t): `group, time, relative_time, att, se, ci_lower, ci_upper, pvalue` | `callaway_santanna` (and `sp.did` when it routes there), `etwfe`, `wooldridge_did` |
| `result.event_study` | one row per relative time: `relative_time, att, se, ci_lower, ci_upper, pvalue` | `callaway_santanna`, `aggte(type='dynamic')`, `sun_abraham`, `did_imputation(horizon=...)`, `gardner_did(event_study=True)`, `stacked_did`, `event_study`, `etwfe_emfx(type='event')` |
| `result.aggregations` | dict of tables: `'overall'`, plus `'dynamic'`, `'group'`, `'calendar'` when the result holds them | every staggered fit above |
| `result.tidy()` | long table: `term, estimate, std_error, statistic, p_value, conf_low, conf_high, type` | every result |

The accessors only read what the fit stored. They never refit. When a table
is not there they raise `ResultAccessorUnavailable`, which is an
`AttributeError` (so `hasattr(result, "attgt")` is `False`) and carries a
`recovery_hint` naming the estimators that provide it. Two cases to know:

- `etwfe` / `wooldridge_did` store cohort-by-relative-time cells, which are
  ATT(g,t), not an event study. `result.attgt` reads them; for the event
  study call `sp.etwfe_emfx(result, type='event')`.
- `sp.aggte` returns the aggregation, not the cells. Read `.attgt` on the
  fit you passed to `aggte`. `aggregations` does not compute the missing
  types; call `sp.aggte(result, type=...)` for those.

Intervals and p-values are each estimator's own. Where a stored table has
none (`gardner_did`'s dict, the `etwfe` cells) the accessors add the normal
interval at the fit's `alpha`, the convention those estimators use.

`tidy()` has the same first seven columns on `CausalResult` and on
`EconometricResults` (`sp.regress`, `sp.feols`, ...). `CausalResult` adds
`type` (`'main'`, `'event_study'`, `'group_time'`).

## `model_info` keys

The tables below list the keys a fit with default options carries.
Options add keys (for example `calendar_time` for date-typed time columns,
`survey_design` with `design=`, `boot_seed` with a bootstrap). Keys that
start with `_` are internal and may change without notice.
`tests/test_did_result_accessors.py` checks that every key listed in the
four tables below is present on a default fit.

### `sp.callaway_santanna`

`.detail` is the ATT(g,t) table (`group, time, att, se, ci_lower, ci_upper,
pvalue, relative_time`). `.estimate` is the simple aggregation.

| key | contents |
| --- | --- |
| `estimator` | 2x2 engine per cell (`'DR'`, `'IPW'`, `'REG'`) |
| `control_group` | `'nevertreated'` or `'notyettreated'` |
| `notyet_cutoff` | not-yet-treated cutoff rule |
| `base_period` | `'universal'` or `'varying'` |
| `anticipation` | anticipation periods |
| `parallel_trends` | dict describing the parallel-trends assumption used |
| `pscore_trim` | propensity-score trimming level |
| `n_pscore_trimmed` | units trimmed |
| `weights` | weights column or `None` |
| `weighted` | whether weights were used |
| `covariates` | covariate list or `None` |
| `n_units` | units in the panel |
| `n_treated_units` | eventually treated units |
| `n_periods` | periods |
| `n_cohorts` | treated cohorts |
| `cohorts` | list of cohorts |
| `cohort_sizes` | units per cohort |
| `comparison_only_cohorts` | cohorts used only as comparisons |
| `dropped_gt_cells` | (g,t) cells that could not be estimated |
| `event_study` | dynamic aggregation, a DataFrame (`relative_time, att, se, ci_lower, ci_upper, pvalue`) |
| `pretrend_test` | dict: `statistic, df, n_cells, rank, rank_deficient, pvalue` |
| `clustervars` | cluster variable(s) or `None` |
| `bstrap` | whether the multiplier bootstrap ran |
| `biters` | bootstrap draws (0 when analytic) |
| `cband` | uniform bands requested |
| `boot_weight_type` | multiplier weights |
| `crit_val_uniform` | uniform-band critical value (or `None`) |
| `se_method` | `'analytic'` or `'bootstrap'` |
| `inference_note` | one-line description of the inference used |
| `balance` | panel balance handling |

### `sp.aggte`

`.detail` is the aggregated table. Keys: `aggregation` (`'simple'`,
`'dynamic'`, `'group'`, `'calendar'`), `balance_e`, `min_e`, `max_e`,
`bstrap`, `n_boot`, `random_state`, `cband`, `crit_val_uniform`,
`share_variance`, `base_period`, `vcov`, `vcov_source`, `cell_labels`,
`influence_functions`, `overall_influence_function`, `vcv_pre`,
`pre_event_times`, `source_method`, `inference_note`, `n_units`; with
`type='dynamic'` also `event_study`.

### `sp.sun_abraham`

`.detail` is the event-study table.

| key | contents |
| --- | --- |
| `estimator` | `'Sun-Abraham IW'` |
| `headline` | what `.estimate` is |
| `other_aggregation` | the other headline, with its value |
| `summary_aggregation` | `'event_time'` or `'fixest_att'` |
| `att_event_time` | equal-weight mean of post-period effects |
| `se_event_time` | its SE |
| `att_fixest_att` | cell-weighted ATT (`fixest` `summary(agg = "att")`) |
| `se_fixest_att` | its SE |
| `se_fixest_att_share` | its SE with cohort-share variance |
| `event_study` | event-study DataFrame |
| `event_times` | relative times estimated |
| `vcov_event_time` | covariance of the event-time coefficients |
| `pretrend_test` | dict: `statistic, df, pvalue, relative_times` |
| `pretest` | pre-trend test type |
| `pretest_periods` | periods in the test |
| `event_window` | reported window |
| `window_rule` | `'report'`, `'bin'` or `'reference'` |
| `control_group` | comparison group |
| `control_cohort` | reference cohort |
| `n_cohorts` | treated cohorts |
| `cohorts` | list of cohorts |
| `anticipation` | anticipation periods |
| `ref_p` | reference relative time(s) |
| `dropped_unidentified_cohorts` | cohorts dropped as unidentified |
| `n_units_dropped_unidentified` | their units |
| `n_rows_dropped_unidentified` | their rows |
| `share_variance` | cohort-share variance included |
| `se_type` | variance description |
| `n_clusters` | clusters |
| `n_coeffs` | interaction coefficients |
| `dof_K` | parameters counted for the small-sample factor |
| `dof_fe_not_nested` | fixed-effect parameters not nested in clusters |
| `dof_convention` | small-sample factor rule |
| `fixed_effects` | absorbed fixed effects |
| `panel` | panel input |

### `sp.did_imputation`

`.detail` is the event-study table when `horizon=` or `pretrends=` is
given, else `None`.

| key | contents |
| --- | --- |
| `estimator` | `'BJS Imputation'` |
| `n_treated_obs` | treated observations |
| `n_control_obs` | untreated observations (the Y(0) fit) |
| `n_units` | units |
| `n_time_periods` | periods |
| `n_never_treated` | never-treated units |
| `cluster_var` | cluster column |
| `vce` | `'analytic'` or `'bootstrap'` |
| `weights` | weights column or `None` |
| `y0_fe` | fixed effects of the Y(0) model |
| `y0_covariates` | covariates of the Y(0) model |
| `horizon_requested` | whether `horizon=` was given |

With `horizon=` or `pretrends=` it adds `event_study`, `event_study_vcov`,
`pretrend_method` and `event_study_convention`; with `pretrends=` also
`pretrend_test` and `pretrends`.

### `sp.drdid`

`.detail` is a two-column summary (`statistic`, `value`).

| key | contents |
| --- | --- |
| `method` | `'improved'` or `'traditional'` |
| `est_method` | `'dr'`, `'ipw'`, `'reg'` or `'twfe'` |
| `engine` | the DRDID estimator reproduced |
| `drdid_reference` | `DRDID::<engine>` |
| `panel` | `True` with `id=`, else repeated cross-sections |
| `n_treated` | treated units / observations |
| `n_control` | comparison units / observations |
| `n_pre` | pre-period rows |
| `n_post` | post-period rows |
| `ps_mean_treated` | mean propensity score, treated |
| `ps_mean_control` | mean propensity score, comparison |
| `se_method` | `'influence_function'` |
| `covariates` | covariate list |

The panel path (`id=`) adds `id`, `n_units`, `ps_flag` and `n_boot` (unused);
the repeated cross-section path adds `variance_convention`.

### `sp.gardner_did`, `sp.stacked_did`, `sp.etwfe`

- `gardner_did`: `event_study` is a dict (`horizon`, `coef`, `se`, `ci`,
  `n_obs`, `vcov`) keyed by `'D_k+h'` labels, present with
  `event_study=True`; `vcov` and `cell_labels` hold the full covariance.
  `result.event_study` turns it into the standard table.
- `stacked_did`: `event_study` (DataFrame), `event_study_vcov`,
  `event_study_betas`, `event_study_se`, `cohorts`, `skipped_cohorts`,
  `window`, `cluster_var`, `cluster_stack`, `n_clusters`, `aggregation`
  and `aggregation_weights` (the weight of each post event time in the
  headline ATT under `aggregation='treated_obs'`, else `None`).
- `etwfe`: `event_study` holds cohort-by-relative-time cells (`cohort,
  rel_time, period, estimate, se, ...`), read through `result.attgt`;
  `event_vcov` is their covariance, `cohort_vcov` that of the per-cohort
  ATTs in `.detail`.
