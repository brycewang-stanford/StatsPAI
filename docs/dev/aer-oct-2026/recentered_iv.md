# Line `recentered_iv`: design-based recentred formula instruments

Batch driven by AER Vol. 116 No. 10 (Oct 2026). Motivating paper: Goldman,
Gracie and Porter, "Who Marries Whom?". Their instrument is the share of
opposite-sex children in a person's childhood neighbourhood. Sex at birth is
as good as random, but the neighbourhood is not, so the regressions control
for the expected instrument given group composition. The authors also test
the realised instrument against a simulated Bernoulli sex assignment. This is
the recentring logic of Borusyak and Hull (2023) (`borusyak2023nonrandom`,
checked on 2026-10-28 against Crossref and doi.org, which resolves to
econometricsociety.org; no new bib entry was needed).

## What ships (after the merge with upstream)

Another session added `sp.recentered_iv` to main on the same day
(`src/statspai/iv/recentered.py`, its own tests in
`tests/test_recentered_iv.py`). That function is the canonical estimator and
stays. This line's duplicate estimator (`RecenteredIVResult`, the formula
interface, `iv/_recentered_ri.py`) was removed. What this line now ships:

```python
sp.expected_instrument(data, instrument_fn, shock_sampler, *, n_draws=1000,
                       seed=None, keep_draws="auto", observed_shocks=None)
    -> ExpectedInstrumentResult   # mu, mc_se, sd, draws, observed,
                                  # to_frame(), counterfactual()

# two new keyword parameters on the upstream function, both default None
sp.recentered_iv(data, y, treat, exposure, counterfactual, ...,
                 absorb=None, expected=None)
```

The hand-off is two steps:

```python
ei = sp.expected_instrument(df, instrument_fn, shock_sampler, n_draws=999,
                            seed=0, observed_shocks=g)
res = sp.recentered_iv(df, "y", "d", "z", counterfactual=ei.draws,
                       absorb="region", cluster="region")
```

Files touched: `iv/expected_instrument.py` (new module, plus
`ExpectedInstrumentResult.counterfactual()`), `iv/recentered.py` (`absorb=`,
`expected=`, the `ri_statistic` diagnostic), the two `FunctionSpec` entries
in `registry.py` (ParamSpecs for `absorb` and `expected`; each entry lists the
other under `alternatives`).

## Design decisions

- **Division of labour.** `sp.expected_instrument` only simulates: it redraws
  the shocks from their known assignment process (a callable, or the spec
  dicts `{'kind': 'bernoulli', ...}` and
  `{'kind': 'permute', 'shocks', 'strata'}`), rebuilds the instrument and
  returns `mu`, its Monte Carlo SE and the `(n, n_draws)` draws. Its `draws`
  are exactly upstream's `counterfactual=` input. Rows where the instrument is
  undefined whatever the shocks are NaN and are dropped, with a warning, by
  `recentered_iv`. The estimator, the asymptotic inference and the
  randomisation inference are upstream's.
- **`absorb=`.** Point estimate and SE come from `sp.iv(absorb=)`, so they
  inherit its ivreghdfe conventions: singletons dropped, FE nested in the
  cluster not charged. `n_obs` counts the kept rows and
  `diagnostics['n_dropped_singletons']` the dropped ones. The OLS branches
  (treatment equal to the exposure) have no `sp.regress(absorb=)`. They run
  2SLS with the regressor as its own instrument, which gives the same
  estimate, sandwich and DOF charge (tested against FE dummies to 1e-10). The
  RI statistic is residualised on the same FE (FWL) with `sp.fast.demean`
  (tol 1e-13, singletons kept: their within value is zero, so they add
  nothing to any inner product), then projected on the swept controls. The
  Hodges-Lehmann root of the test therefore still equals the estimate. A `mu`
  constant within the FE is absorbed: it warns, and both methods equal the
  unadjusted estimate. If `z - mu` has no within-FE variation the call raises
  `AssumptionViolation`. `absorb=` with `weights=` raises, because `sp.iv` has
  no weighted FE sweep on its absorbed path.
- **`expected=`.** A closed-form `mu` (column or length-n array) replaces the
  row mean of the draws. In RI, the realised draw and every counterfactual
  are centred at this fixed `mu`. It does not depend on the draws, so the
  test is exact for both methods and any S. Upstream's default centres all
  S + 1 draws at their common mean, which is also exact but is a different
  statistic. `diagnostics['expected_draw_gap']` reports the largest gap
  between `mu` and the draws' row mean. Do not pass `ei.mu` from the same
  call whose `ei.draws` you pass: `mu` would then depend on the draws it is
  compared with. For a fixed simulated `mu`, make two calls with different
  seeds.
- **Why `expected=` was needed for the ritest comparison.** ritest ranks
  `T = zs' M (y - b0 d)` with the strata in `M`. That equals the statistic
  centred at the stratum mean, the exact `mu` under within-stratum
  permutation. Upstream's common-mean centring shifts it by `mu_all' M e`
  (13.28 at `b0 = 0`, 1.53 at `b0 = 1.2`). The count at `b0 = 1.2` is then
  150, not ritest's 146. It is a different exact test, not a bug, and a test
  pins the identity. With `expected=` set to the stratum mean the counts
  agree exactly.
- **Default unchanged.** Without `absorb` and `expected` the output is
  bit-identical to upstream before this change. That was checked on 96
  configurations: both methods, treatment equal or not to the exposure,
  controls, weights with clusters, HC3, balance tests, `ri_null` and `alpha`.
  All 1,800 floats are equal. One diagnostic key was added, `ri_statistic`
  (the observed `a - ri_null * c`).

## Evidence

| Check | Where | Result | Tolerance |
| --- | --- | --- | --- |
| (b) Bernoulli share closed form, S = 400 and 6400 | `tests/test_expected_instrument.py::test_simulated_mu_converges_to_closed_form_at_mc_rate` | mean square of `(mu_hat - mu)/sqrt(Var/S)` within 5 SD of 1; reported MC SE within 15% of `sqrt(Var/S)` on every row | chi2_n / n band |
| (b) Fair coins give a constant mu, flagged | `test_fair_coin_share_has_constant_mu` | `mu_constant` true | 6 MC SE |
| (b) Within-stratum permutation: mu = stratum mean; each draw keeps the stratum multiset | `test_permutation_sampler_mu_is_stratum_mean` | | 6 MC SE; draw stratum means rel 1e-12 |
| Reproducibility and row order (seed, replay from `seed_entropy`, permuted rows) | `test_seeded_results_are_reproducible_and_row_order_free` | identical / permuted `mu` | exact / rel 1e-12 |
| Streaming moments equal the kept draws | `test_keep_draws_false_streams_same_moments` | | rel 1e-12 / 1e-10 |
| (a) End to end: `ei.draws` into `recentered_iv`, exposure correlated with the error | `test_draws_feed_recentered_iv_and_recover_known_truth` | mean within 4 MC SE of beta = 1 (R = 40); naive bias > 0.3 and > 20 MC SE; simulated mu within 6 MC SE of the closed form on every row | 4 MC SE |
| (b) `absorb=` equals FE dummies as controls: estimate, SE, unadjusted estimate and SE, RI p-value, statistic, set, HL root; both methods, IV and OLS branches, HC1 and clustered | `tests/test_recentered_iv.py::test_absorb_equals_fe_dummies_as_covariates` | | 1e-10 |
| (b) Same with two-way FE (iterative sweep), `"f + h"` and `"f^h"` spellings | `test_absorb_two_way_equals_dummies` | | rel 1e-8 / 1e-12 |
| (b) Singletons dropped without changing the results | `test_absorb_singletons_are_dropped_without_changing_results` | | rel 1e-10 |
| (a) RI exactness with absorbed strata, shocks permuted within strata | `test_absorb_ri_is_exact_under_within_stratum_permutation` | P(p <= 0.1) within 3 binomial SD of 0.1 | R = 120, S = 49 |
| (b) `expected=` statistic equals `(z - mu)' M (y - b x)` computed by hand; count gives the p-value | `test_expected_ri_statistic_is_the_fixed_mu_formula` | | rel 1e-10; p exact |
| (c) ritest, 200 **fixed** within-stratum permutations shared via `samplingsourcefile` (deterministic, not a stochastic screen) | `tests/reference_parity/test_recentered_iv_stata.py` | counts 0 and 146 of 200 at b0 = 0 and 1.2, equal to ritest; observed T 1092.7491932922692 and 11.93870531662667, matched to 5.6e-15; both methods | count exact, T rel 1e-9 |
| (d) ivreghdfe, closed-form mu (Bernoulli share), absorb(f), cluster(c) and robust, mu given as draws with mean exactly mu and as `expected=` | same file | recentre 1.5910831104429131 (SE 0.087675966038617187), control 1.5990554197786582 (0.078781504314487841), naive 1.8373777318146214 (0.066966266205030808), robust SE 0.093263586546852711; observed agreement <= 3e-15 | rel 1e-8 |

Stata 18 MP: ivreghdfe 1.1.4, ivreg2 4.1.12, reghdfe 6.13.1, ritest 1.1.7.
Fixtures and scripts are in `tests/reference_parity/_fixtures/`. The data come
from `_generate_recentered_iv_data.py` and the Stata numbers from
`_generate_recentered_iv_Stata.do`. All values have 6 decimals and are read
with `float_precision="round_trip"`. The 200 permutations are stored as
`recentered_iv_ri_perms.csv.gz` (87 KB; the plain CSV, 543 KB, is above the
500 KB pre-commit limit). The generator writes it with gzip `mtime=0`, so a
rerun is byte-identical, and the do-file decompresses it to a temporary CSV
before `import delimited`. The do-file was rerun on 2026-10-28 after that
change and printed the same reference values. Inside ritest the statistic is
computed by an e-class program (`ristat`) that posts `T = zs' ey` with
`ey = reghdfe resid(...) tol(1e-14)`. ritest counts with an absolute 1e-7
tolerance and StatsPAI with a relative 1e-12 one. No draw falls near the
observed statistic, so the counts are comparable.

Test counts: `tests/test_expected_instrument.py` 25,
`tests/test_recentered_iv.py` 38 (20 upstream plus 18 for `absorb` /
`expected`), `tests/reference_parity/test_recentered_iv_stata.py` 11. The
doctests in both modules pass (3).

## Proposed CHANGELOG entry (Unreleased, Added)

- `sp.expected_instrument`: simulates the expected instrument
  `mu_i = E[z_i | exposure]` of an instrument built from as-good-as-random
  shocks and non-random exposure [@borusyak2023nonrandom], by redrawing the
  shocks from a user callable or a built-in Bernoulli / within-strata
  permutation spec. It returns `mu` with its Monte Carlo SE and the
  `(n, n_draws)` draws. Pass the draws to `sp.recentered_iv` as
  `counterfactual=`.
- `sp.recentered_iv(absorb=)`: absorbs high-dimensional fixed effects. The
  estimate and SE go through `sp.iv(absorb=)` and match `ivreghdfe` (observed
  3e-15). The randomisation statistic is residualised on the same fixed
  effects.
- `sp.recentered_iv(expected=)`: takes a closed-form expected instrument. The
  randomisation test then centres every draw at it and stays exact. With the
  stratum mean under within-stratum permutation, the counts on 200 fixed
  permutations equal Stata `ritest` exactly. Without the new arguments the
  results are unchanged. The new diagnostic `ri_statistic` reports the
  observed statistic.

## Open issues

- Conventional SEs treat units as independent (or clustered as asked). With
  few effective shocks they understate uncertainty, and the RI result is the
  design-based answer. Shock-level, exposure-robust conventional SEs for
  general formula instruments (an AKM-style variance) are not implemented.
- `absorb=` with `weights=` is refused because `sp.iv` has no weighted FE
  sweep on its absorbed path. The workaround is FE dummies in `covariates=`.
- Balance tests (`balance=`) are not residualised on `covariates` or
  `absorb`. This is unchanged from upstream: the test stays valid under the
  design, but it is a test of the raw covariate.
- Simulation error in a simulated `mu` is not propagated into the
  conventional SE. It acts like extra instrument noise independent of the
  error term. RI stays exact.
- Derived artifacts were not regenerated, per the batch rules: schemas,
  parity index, `docs/stats.md`, README counts, evidence inventory, source
  failure modes and trace ledgers. The integrator runs them. The stub
  `src/statspai/__init__.pyi` only re-exports `recentered_iv`, so the new
  keyword parameters need no stub change.
- Track A estimation files: **not edited**.
