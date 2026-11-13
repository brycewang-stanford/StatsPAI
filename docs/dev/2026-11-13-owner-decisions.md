# Owner decisions, 2026-11-13

Eight work lines ran on 2026-11-13 (three blind replication probes for
epidemiology, applied microeconomics and time series, one for ML-causal
methods, one agent probe over MCP, two standard-error calibration screens
and one reference-parity line). Each report ended with questions that only
the owner can settle: defaults of public functions, labels, and which
reference to follow. Bryce delegated them to the coordinator for this
week. The coordinator ruled on them the same day and line `decisions`
implemented the rulings. This file records each decision, what this
release does, what a later release does, and the items deliberately kept
as they are with the reason. The format follows
`docs/dev/2026-10-20-owner-decisions.md`. Decisions are final for this
release cycle; reopening one goes through the normal deprecation process
(CLAUDE.md §3.8).

Rule for the whole wave, the same as on 2026-10-20. No number changes in
this release except where an item is marked ⚠️ (a correctness fix, logged
in CHANGELOG and MIGRATION.md). A default that should change later is
announced now with a `FutureWarning` that fires only when the caller
relies on the default: passing the option explicitly, with either value,
is silent. The switch is listed in MIGRATION.md and happens no earlier
than the next minor release after the one that carries the announcement.

Evidence for every item below is in
`tests/test_owner_decisions_2026_11_13.py` (items 1, 3, 4, 5 and 11) and
`tests/test_owner_decisions_2026_11_13_traced.py` (items 2, 6 to 10 and
12, whose source files are on a traced Track A path) unless another file
is named.
The shared helper for the announcements is
`src/statspai/_default_change.py`.

## A. Announced now, switched later

| # | Decision | This release | Later |
| --- | --- | --- | --- |
| 1 | `sp.mr_mode` default `stderror` moves to `'delta'`, the `TwoSampleMR` convention. The `'simple'` standard error treats the exposure associations as known (NOME) and is 2.7 times smaller on that package's own example (probe-epi). | `stderror=None` in the signature. Not given: `'simple'` as before, with a `FutureWarning`. `sp.mr("mode", ...)` warns the same way. Numbers unchanged (the default equals `stderror='simple'` exactly). `stderror` is now a registered parameter. | Default becomes `'delta'`. The standard error, interval and p-value change, and the weighted-mode estimate can move because its weights use the same ratio standard errors. |
| 2 | `sp.stacked_did` reference distribution moves to t(G-1), as `sp.event_study` and `sp.sun_abraham` did in 1.41 (probe-econ: p 0.0214 against 0.0257 on the castle-doctrine stack). | New option `crit_dist={'t', 'normal'}`, the name and values the two event studies use; the rule is recorded in `model_info['conventions']` with the same keys. Not given: `'normal'` with a `FutureWarning` for `family='gaussian'`. `'t'` equals R `fixest` on the stacked regression of Track A module 75: p-values to 1e-6, interval ends to 1e-7 (`tests/reference_parity/_fixtures/stacked_crit_R.csv`). `sp.did_summary` and the calibrated simulation pin `'normal'`, so they keep their numbers and stay silent. | Default becomes `'t'`. Estimates and standard errors do not change. `family='poisson'` keeps the normal distribution, as `ppmlhdfe` and `fixest::fepois` report z statistics. The two library pins are removed at the same time. |
| 3 | `sp.rkd` headline moves to the robust bias-corrected estimate and interval, as `sp.rdrobust` reports. The conventional headline had bias 0.23 sampling SDs and SE/SD 0.79 in the calibration screen (calib-reg). | New option `headline={'conventional', 'robust'}`. Not given: `'conventional'` with a `FutureWarning`. `'robust'` equals the headline of `sp.rdrobust(deriv=1, p=1)` exactly; `summary()` then shows the conventional numbers on the secondary line. Both sets stay in `model_info['conventional']` / `['robust']`; `model_info['headline']` says which is reported. | Default becomes `'robust'`: `estimate`, `se`, `pvalue` and `ci` change. |
| 4 | `sp.iv_mediate` default `equations` moves to `'joint'`. The `ivmediate` variance drops the covariance between equations: SE/SD 1.25 or 0.87 depending on the sign of the error link, against 1.06 and 1.01 for `'joint'` (calib-te). | `equations=None` in the signature. Not given: `'independent'` with a `FutureWarning`. Numbers unchanged. The Stata translation note for `ivmediate` and the `ivmediate` parity test pass `equations='independent'` explicitly. | Default becomes `'joint'`: the standard error, interval and p-value of the indirect effect change; nothing else does. |

The earliest release that may switch items 1 to 4 is the minor release
after the one that ships the announcement. The warnings say "1.45 at the
earliest", which assumes this wave ships as 1.44.0 (see the follow-ups).

## B. Done now

| # | Decision | What was done |
| --- | --- | --- |
| 5 | ⚠️ `sp.dose_response` headline. The ruling was conditional: make it the 25th-to-75th percentile contrast if that is how the documentation describes it, otherwise keep the number and fix the label. | The released documentation described a percentile contrast: the 1.43.1 docstring calls it "the 25th-75th percentile effect", the function's own warning uses the same words, the estimand label is `E[Y(t75)] - E[Y(t25)]` and the key is `effect_25_to_75`. (The paragraph on main that described grid points was written on 2026-11-13 by the ML probe to document the mismatch while this decision was pending.) The headline is now the curve at the 75th minus the 25th percentile of the observed treatment, for both methods, independent of `n_dose_points`. The two doses are appended to the evaluation grid, so the curve in `detail` is bit-identical. NHEFS, Hirano-Imbens quadratic models: 1.0974 (`n_dose_points=10`) and 0.9702 (20) before, 1.2752 for both after; `method='dr'`: 4.2639 before, 3.6071 after. CHANGELOG ⚠️ + MIGRATION. |
| 6 | `sp.dml(model='pliv')` is not a LATE. | The estimand label is `'theta'`, the coefficient of the partially linear IV model (DoubleML's target parameter, and the label PLR already uses). `model='iivm'` keeps `'LATE'`. Label only. |
| 7 | `sp.sdid(method='sc', se_method='jackknife')` must say that the jackknife is not valid for the SC weights (4.2 times the sampling SD in the screen, the same in R `synthdid`). | `AssumptionWarning` plus `model_info['diagnostics']['sc_jackknife_se']`, on all four paths of `sp.sdid`. The number is unchanged; a test recomputes it by hand from the stored weights. |
| 8 | `sp.regress` docstring said HAC inference uses normal critical values (probe-ts). | The sentence now says t(N - K), as Stata `newey`; a test pins the p-value and interval against that distribution. |
| 9 | `sp.match` summary printed the pre-matching balance table without saying so (probe-epi, probe-agent). | `CausalResult.summary()` reads `detail.attrs['stage']`: the table is headed "Covariate Balance, before matching" and the matched table follows as "Covariate Balance, after matching". `detail is model_info['balance_before']` still holds. |
| 10 | `sp.tmle` default library floods `ConvergenceWarning` on unscaled covariates; scaling inside the library would change default numbers. | No number change. Fits of the default library are wrapped: the sklearn warnings are collected and reported as one `AssumptionWarning` naming the learner, the nuisance and the remedy, recorded in `model_info['diagnostics']['super_learner_convergence']`. A library passed by the caller is not wrapped; with the same learners it gives the same estimate and standard error bit for bit. |
| 11 | Two off-policy APIs with opposite argument order (`sp.ips(X, A, R, pi_target, pi_behavior)`, `sp.ope.ips(actions, rewards, pi_b, pi_e)`). | Both are kept. Each docstring states its own order and points at the other. Both raise `MethodIncompatibility` when the shapes show a swap (a per-row vector where a matrix belongs). Two matrices cannot be told apart by shape, and no guess is made from values. Same for `snips`. |
| 12 | Three SMD denominators for one set of weights (`sp.love_plot`, `sp.match(...).balance_after`, `sp.bal_tab`). | Not unified, no number change. Each result or docstring names its denominator: `sp.match` tables carry `attrs['smd_denominator']` (pooled SD of the raw sample, before and after), `sp.ps_balance` / `sp.love_plot` use the pooled SD of the weighted sample for the adjusted series, `sp.bal_tab` follows `cobalt` (unweighted; pooled for the ATE, the treated group's for the ATT). The table of which function uses which, and which reference each matches, is in `docs/dev/modules/treatment_effects.md`; a test reproduces all four denominators by hand and shows that `sp.bal_tab(s_d_denom='pooled')` returns the `sp.match` numbers. |

## C. Kept as they are

- **`sp.xtlsdvc` default stays Stata's plug-in.** Parity with the
  reference is the stated default for this estimator. The consistent
  plug-in (`plug_in='consistent'`) removes a bias that does not shrink
  with N and stays opt-in.
- **`sp.oaxaca` default standard errors stay fixed-means.** They equal
  Stata `oaxaca ..., fixed` to 1e-12. The stochastic-regressor standard
  error, Stata's default, is added as an option by line `gaps`.
- **Defaults that follow their documented reference or convention, with
  the alternative available as an option.** `sp.aipw` default
  cross-fitting (the full-sample fit equals `teffects aipw` and is an
  option); `sp.arima` default `vce='opg'` (Stata `arima`; `'oim'` is R's);
  `VECResult.irf` / `forecast` default `sigma_df='ml'` (`'unbiased'` is
  Stata's `T - d`); `VARResult.forecast` without parameter uncertainty
  (`param_uncertainty=True` is Stata `fcast compute`); the DF-GLS lag rule
  (AIC; the Ng-Perron MAIC rule is not implemented); `sp.its` `hac_lag=4`
  (every HAC choice under-covers at T = 200, a property of Newey-West and
  not of this lag); `sp.did_imputation` `min_n` (every estimable horizon is
  reported; `min_n=30` gives Stata's `minn(30)` on unweighted data);
  `sp.pcmci` Fisher z (`ci_test=` gives the `tigramite` numbers);
  `sp.icp` `test='anova'` (`test='normal'` is the authors' R test).
- **Matching on an estimated propensity score keeps the `psmatch2`-style
  standard error** for kernel, radius and nearest-neighbour matching. It
  ignores the estimation of the score and is conservative by 1.2 to 1.5 in
  the screen. Kept because it is what the reference reports and it errs on
  the safe side; the docstring and the matching guide say which options
  are correctly sized (`abadie_imbens_2016`, bootstrap).
- **`sp.manski_bounds` and `sp.lee_bounds` report different intervals.**
  The first covers the whole identified set, the second is the
  Imbens-Manski interval for the parameter. Different targets, both
  labelled.
- **Default seeds are not unified.** Decided on 2026-10-15: published
  seeded numbers would change. `sp.conformal_counterfactual` and
  `sp.cate_gain_curve` stay unseeded by default.
- **`sp.cate_by_group` stays a descriptive spread**, documented as such.
  A group effect with estimation error is `sp.gate_test`.
- **`sp.target_trial_emulate` does not estimate weights on its own.** It
  warns and labels the crude contrast (done by probe-epi); fitting weights
  from `baseline_covariates` would change released numbers.
- **Sup-F p-values keep the current approximation** (a re-simulated Hansen
  table, 1 to 3% from R and Stata, documented). Adopting Hansen's
  published coefficients needs the table verified from the paper, which
  was not done.
- **`notears` and `ctmle` stay graded `aligned`** with their scoped notes
  (graph and candidate sequence respectively).
- **Left open.** `sp.partial_identification` returns two result classes
  depending on the method, and the default `delta` of `sp.sbw` leaves a
  small residual-imbalance bias. Neither was decided.

Module notes carry one line per kept item where the convention lives:
`docs/dev/modules/timeseries.md`, `did.md`, `panel_multilevel.md`,
`decomposition.md`, `causal_discovery_dag.md` and `treatment_effects.md`.

## Follow-ups for the next re-anchor

- **JSS sentence on Python-reference rows.** `PYTHON_REFERENCE_ROWS` grew
  by five on 2026-11-13 (`lingam`, `pcmci`, `notears`, `predict_cate`,
  `cate_gain_curve`). The JSS sentence that enumerates Python-reference
  rows has to be revisited when the paper is next re-anchored. The paper
  is not edited between anchors.
- **"versionchanged:: 1.44" assumes the next release is 1.44.0.** The
  docstrings written on 2026-11-13 (by this line and by the probe lines)
  name 1.44 as the release that carries today's changes, and the four
  announcements say "1.45 at the earliest". If the next release gets
  another number, update both. A patch release must not carry this wave:
  it contains new parameters, `FutureWarning` announcements and ⚠️ fixes.
- **Track A.** `core/results.py`, `regression/ols.py`, `matching/match.py`,
  `did/stacked_did.py`, `synth/sdid.py`, `dml/pliv.py` and `tmle/tmle.py`
  are on traced estimation paths. Both provenance ledgers were re-recorded
  after the modules were re-run and compared with the committed
  `*_py.json`; the log is in `docs/dev/jss_review_changes.md`.
