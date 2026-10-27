# AER Vol. 116 No. 10 batch: `fuzzydid` line

Motivating design: the wealth-tax repeal paper in this AER issue (DiD of
top-2% vs top-10-20% wealth holders around the 2007 Swedish repeal; a
group-level 2SLS of an outcome rate on ln(1 - tau) instrumented by
treat x post; an event study of out-migration in which never-migrants get
random placebo event dates). This line adds the estimators and the
pre-processing that design needs, and checks the 2SLS identity.

## What was added

| | |
| --- | --- |
| `sp.fuzzy_did(data, y, group_change, time, treat, *, group_change_next=None, estimators=('did','tc','cic'), covariates=None, modelx=('ols','ols','ols'), newcateg=None, numerator=False, se_method='bootstrap', n_boot=50, cluster=None, eqtest=False, cic_quantile='exact', alpha=0.05, seed=None) -> FuzzyDiDResult` | Wald-DID / Wald-TC / Wald-CIC of de Chaisemartin and D'Haultfoeuille (2018) with the conventions of Stata `fuzzydid` |
| `sp.FuzzyDiDResult` | `CausalResult` subclass; `detail` one row per estimator, `model_info['comparisons']` one row per (period pair, direction) |
| `sp.assign_placebo_dates(data, id, event_time, *, time=None, strata=None, method='empirical', require_observed=False, seed=None, column='placebo_event_time', flag='placebo') -> DataFrame` | placebo event dates for never-treated units |
| `fuzzydid` in `sp.from_stata` / `sp.stata` | `agent/_translation/_stata_fuzzydid.py`, registered via `_stata_did.HANDLERS` |
| `paper.bib`: `dechaisemartin2019fuzzy` | the Stata Journal article of `fuzzydid` (Crossref + OpenAlex, doi.org) |

Files: `src/statspai/did/fuzzy_did.py`, `src/statspai/did/_fuzzy_did_core.py`,
`src/statspai/did/placebo_dates.py`,
`src/statspai/agent/_translation/_stata_fuzzydid.py`; wiring in
`did/__init__.py`, `statspai/__init__.py`, `registry.py`,
`agent/_translation/_stata_did.py` (two lines); tests
`tests/test_fuzzy_did.py`, `tests/test_placebo_dates.py`,
`tests/test_stata_translation_fuzzydid.py`,
`tests/reference_parity/test_fuzzy_did_stata_parity.py`, one round-trip
row in `tests/test_translation.py`; fixtures
`tests/reference_parity/_fixtures/{_generate_fuzzydid_data.py,_generate_fuzzydid_Stata.do,fuzzydid_2x2.csv,fuzzydid_ordered.csv,fuzzydid_multi.csv,fuzzydid_Stata.json}`.

No Track A estimation-path file was edited. `src/statspai/__init__.py`
(bound by both parity trace ledgers) gained three exports, so the ledgers
need the usual re-record at merge.

### Dispatcher check

`sp.did(method=...)` is not extended: its `treat` is the group / cohort
column, while a fuzzy DiD needs a group-change code *and* a separate
treatment variable. `sp.continuous_did` only cites the paper (heuristic
dose-bin estimators, no Wald-TC / CIC). The dCDH family
(`did_multiplegt*`) estimates switcher effects on group-level panels, a
different estimand. So `sp.fuzzy_did` is a new entry, not a parallel
dispatcher.

## Evidence

Reference: Stata 18 MP, `fuzzydid` (SSC, Distribution-Date 20190507),
installed into a private ado directory, run as a black box (the ado has no
licence statement; its code was not read). Fixtures are synthetic, values
rounded to three decimals, read by both sides from the same CSV bytes
(`set type double` + `import delimited, asdouble`).

| design | rows | Wald-DID | Wald-TC | Wald-CIC | max abs gap |
| --- | --- | --- | --- | --- | --- |
| 2x2, binary D, control rate moves | 1,230 | 2.26742696105217 | 2.20184132261825 | 2.23925983184161 | 2.7e-15 |
| same, numerators | | 0.729277817385019 | 0.935275225402474 | 0.951169470046083 | 8.9e-16 |
| same, `continuous(x1)` | | 2.16713350746173 | 2.05307096342341 | - | 3.1e-15 |
| same, `continuous(x1) qualitative(x2)` | | 2.35691899362992 | 2.18716283259362 | - | 8.4e-15 |
| same, `modelx(ols logit)` / `(ols probit)` / `yb, (logit logit)` | | | | | 4.4e-8 (rel 2e-8) |
| 2x2, D in {0..3} | 1,475 | 1.40457601453191 | 1.45462893139623 | 1.49882629283025 | 1.3e-15 |
| same, `newcateg(1 3)` | | 1.40457601453191 | 1.30387925797264 | 1.27367521718057 | 4.2e-15 |
| same, `continuous(x1)` (+ `modelx(ols ols logit/probit)`) | | 1.3929039601558 | 1.45319277826658 | - | 7.3e-15 (GLM: 1.5e-8) |
| 40 groups x 3 periods, up / down / stable, D in {0,1,2} | 2,799 | 1.7677486963012 | 1.8004946954915 | 1.78788524065435 | 8.9e-16 |
| same, `newcateg(0 2)` | | 1.7677486963012 | 1.79474039771242 | 1.7991381696546 | 2.2e-16 (`cic_quantile='stata'`) |
| same, `continuous(x1)` | | 1.72449548026937 | 1.7738914243158 | - | 2.0e-15 |

Tolerances in the tests: 1e-11 absolute for OLS / cell-mean / quantile
rows; 1e-6 relative for rows with a logit / probit model.

Row-level gap notes:

* **logit / probit rows, 6e-11 to 4.4e-8 (reference numerics).** Stata's
  `logit` stops at its default tolerances. On the (G=1, T=0) cell's E(D|X)
  model, Stata's default and 1e-14-tolerance coefficients differ by
  ~6e-8; `sp.fuzzy_did`'s Newton solution equals the tight one to 1e-10
  (`test_logit_gap_is_statas_stopping_rule`).
* **Wald-CIC with `newcateg(0 2)` on the multi-period design, 3.65e-5 under
  the default `cic_quantile='exact'` (reference numerics, pinned).**
  `fuzzydid` evaluates `F^{-1}(u)` as `ceil((count / n0) * n1)` in floating
  point. When `count * n1 / n0` is an integer the product can round up and
  select the next order statistic; in this fixture the pooled category's
  two control cells have 284 rows each, so every switcher is such a case.
  Located by running `fuzzydid` on each period pair and direction
  separately (only the 1 -> 2 comparisons differed) and by testing the
  rounding hypothesis on that comparison (gap 1.6e-15 after). The exact
  rule is the default; `cic_quantile='stata'` reproduces Stata to 2.2e-16
  and the translator writes it.

Bootstrap standard errors (stochastic screen S, not parity; 500 draws
each side, ~4.5% SD of the ratio, 15% screen):

| | Stata W_DID / W_TC / W_CIC | sp (seed 1) |
| --- | --- | --- |
| 2x2 | 0.4313 / 0.3235 / 0.3345 | 0.4317 / 0.3248 / 0.3392 |
| 2x2, `cluster(cl)` (150 clusters) | 0.4753 / 0.3650 / 0.3600 | 0.4933 / 0.3725 / 0.3660 |
| multi | 0.1224 / 0.1164 / 0.1198 | 0.1205 / 0.1143 / 0.1176 |

Analytic SEs (new; `fuzzydid` has none): the influence function of the
Wald-DID / Wald-TC as smooth functions of atom (period x group codes x
treatment category) counts and sums, gradient by complex step. Identity on
the 2x2 fixture: Wald-DID = `sp.iv('y ~ g + t + (d ~ gxt)')` coefficient
(1e-14) and analytic SE = its HC0 SE (0.411132373243, rel 1e-10); with
`cluster='cl'`, SE x sqrt(G/(G-1) (N-1)/(N-K)) = `sp.iv(cluster=)` SE.
Analytic vs bootstrap (400 draws) within 15% for DID and TC.

Known-truth designs (`tests/test_fuzzy_did.py`, 200,000 rows per cell, exact
type shares, LATE = 2):

| DGP | Wald-DID | Wald-TC | Wald-CIC |
| --- | --- | --- | --- |
| stable, homogeneous effect; control rate fixed | consistent | consistent | consistent |
| effect 1 in period 0, 2 in period 1; control rate 0.20 -> 0.35 | -> 0.65/0.3 = 2.167 (biased) | consistent | consistent |
| effect 2; control group's new takers have untreated outcome +3 | consistent | -> 2.223 (biased, composition of delta_d) | (not asserted) |

Each limit is asserted within 4 analytic SEs; each bias exceeds 8 SEs.

Group-level 2SLS (the paper's specification): on a balanced 24-group x
2-period panel, `sp.iv('rate ~ post + group dummies + (lnnet ~ top_post)')`
equals `sp.fuzzy_did(..., estimators=('did',))` on the same rows to 1e-10.
`sp.iv` already handles this; nothing was changed there.

`assign_placebo_dates`: row-order invariance (with strata and
`require_observed`); `method='proportional'` reproduces 3x the treated
frequencies exactly in each of two strata; independent draws (30,000
never-treated) within 4 binomial SEs of every treated share; an event study
with no effect and convex calendar effects is within 4 SEs of zero at all
ten horizons, while placebo dates from a uniform distribution give |z| > 10.

Translator: seven do-file lines run through `sp.stata` reproduce the
frozen Stata numbers (1e-11; GLM row 1e-6); bootstrap run screened at 15%.

## Proposed CHANGELOG entry (`## [Unreleased]` / `### Added`)

- **`sp.fuzzy_did`**: fuzzy difference-in-differences of de Chaisemartin
  and D'Haultfoeuille (2018), the estimators of Stata `fuzzydid`. Wald-DID,
  Wald-TC (time-corrected) and Wald-CIC (changes in changes) for binary or
  ordered treatments, two periods or several groups and periods with
  increasing, decreasing and stable groups (`group_change` /
  `group_change_next`, fuzzydid's two G variables), covariates with OLS /
  logit / probit conditional means, `newcateg`, numerators for placebo
  tests, `eqtest`, bootstrap SEs (cluster bootstrap with `cluster=`) and
  influence-function SEs for the Wald-DID / Wald-TC
  (`se_method='analytic'`, equal to the 2SLS HC0 / CR0 SE for the 2x2
  Wald-DID). Point estimates equal `fuzzydid` to 1e-14 on three fixtures
  (1e-8 with logit / probit models, Stata's ML stopping rule). The Wald-CIC
  uses the exact empirical quantile; `cic_quantile='stata'` reproduces
  fuzzydid's floating-point rounding, which differs by ~1e-5 when
  `u * n` is an integer. Not implemented: `lqte`, `partial`, `sieves`.
- **`sp.assign_placebo_dates`**: placebo event dates for never-treated
  units, drawn from the treated units' event-date distribution (within
  strata, optionally only at dates where the unit is observed;
  `method='proportional'` reproduces the frequencies exactly), so event
  studies can compare treated and never-treated units in event time.
  Reproducible for a seed whatever the row order.
- **`fuzzydid` translates** in `sp.from_stata` / `sp.stata` to
  `sp.fuzzy_did` (`lqte` / `partial` / `sieves` refused).

## MIGRATION note

None: two new functions and one new translation; no existing behaviour
changes.

## Open items

* fuzzydid `lqte` (local quantile treatment effects, binary D / G / T),
  `partial` (bounds without a stable group) and `sieves` (series
  conditional means with CV-chosen order) are not implemented; the
  translator refuses them and `sp.fuzzy_did`'s registry `limitations`
  lists them.
* Bootstrap is not stratified (Stata's resampling scheme was not
  identified beyond the SE screen above).
* Covariates with a decreasing group: matched on the multi-period fixture
  only for OLS models (`multi_x1`); logit / probit there were not run.
* Not regenerated here (per batch rules): schemas, parity index,
  `docs/stats.md`, README counts, evidence inventory, source failure modes
  (the new typed raises carry `recovery_hint`s and will add cards), trace
  ledgers (`statspai/__init__.py` changed).
