# Migrating between StatsPAI versions + from PyStataR

Internal version-to-version migrations are at the top; the long-form
`PyStataR → StatsPAI` migration follows below.

---

<a id="rdrobust-bandwidth-rebuild"></a>

## Unreleased — ⚠️ `sp.rdrobust` numbers change

**What changed.** The CCT bandwidth selector and the bias-correction step
were both wrong. On `rdrobust`'s own `rdrobust_RDsenate` with default
settings `sp.rdrobust` reported **12.39**; R reports **7.41**.

| | old | new | R |
| --- | ---: | ---: | ---: |
| headline effect | 12.39 | **7.5065** | 7.5065 |
| bandwidth `h` (p=1, tri) | 4.633 | **17.7544** | 17.7544 |
| bandwidth `h` (p=2, tri) | 4.633 | **22.2563** | 22.2563 |
| bias bandwidth `b` | = `h` | **28.0281** | 28.0281 |

The old `h` was **identical for p=1 and p=2** because the rate exponent was
hard-coded to `1/5`, which is CCT's `1/(2p+3)` only at `p=1`.

**Effect.** Every `sp.rdrobust` / `sp.rdbwselect` number changes, and so do
the downstream diagnostics built on them (`rdbwsensitivity`, `rdbalance`,
`rdplacebo`, `rd_multi_extrapolate`). **Re-run anything whose numbers came
from these.** There is no flag restoring the old behaviour; it was not an
alternative bandwidth convention, it was the wrong formula.

**How to check an archived figure.** If you recorded the bandwidth, the old
`h` was roughly `n^{-1/5}`-scaled off the correct one and insensitive to `p`
— an `h` that does not move when you change `p` is the signature. The
conventional estimate at a *user-supplied* `h` was always correct, so
`sp.rdrobust(..., h=<your old h>)` reproduces the old point estimate.

**Behaviour changes beyond the numbers.**

| Before | After |
| --- | --- |
| `bwselect='msesum'` / `'cersum'` raised `ValueError` | Accepted; all six R variants work |
| `b` defaulted to `h` | `b` comes from the cascade when `h` is auto-selected; still `b = h` when you supply `h` yourself, matching R |

**Not fixed.** `covs=` is a silent no-op — see the Known issues section of
the CHANGELOG. If you have been passing covariates to `sp.rdrobust`, you
have been getting unadjusted estimates, and that is still true after this
release.

---

<a id="unified-sensitivity-scale"></a>

## Unreleased — ⚠️ `unified_sensitivity`: E-value scale and Oster inputs

Two quantities in the dashboard were computed from the wrong inputs.

**E-value from an un-standardised coefficient.** The E-value is defined on
the risk-ratio scale. `unified_sensitivity` forced `measure="RR"` and
passed a raw regression coefficient through unchanged whenever it was
positive, so a $1,548 treatment effect was read as a risk ratio of 1548:

```text
before:  RR "1548.24"  ->  E-value 3095.99      (meaningless)
after:   d = 0.2072, RR = exp(0.91*d) = 1.2075  ->  E-value 1.7082
```

A mean difference must be standardised by the outcome SD first
(`vanderweele2017sensitivity`). Migration:

```python
# supply the scale — data=/y= is usually already there for Sensemakr
sp.unified_sensitivity(fit, term="treat", data=df, y="re78", controls=X)

# or give the SD directly
sp.unified_sensitivity(fit, term="treat", outcome_sd=df["re78"].std(ddof=1))

# or declare that the estimate really is a ratio
sp.unified_sensitivity(hazard_fit, term="treat", measure="RR")
```

Without a scale the E-value is now `nan` with an explanatory note instead
of a fabricated number. If you published an E-value from a linear model,
recompute it — the old one described a risk ratio you never estimated.

**Oster inputs.** `r2_treated` / `r2_controlled` read like sensemakr's
partial R^2 but were consumed as the short- and long-regression R^2. Given
sensemakr-style values they produced `delta* = -12.765` where
`sp.oster_delta` reported `-2.339` for the same specification — two
contradictory deltas in one report. They are renamed `r2_short` /
`r2_long` (old names still work, with a `DeprecationWarning`), and when
`data`, `y`, `treat` and `controls` are available the R^2 are derived from
the data so the two paths agree by construction.

---

<a id="continuous-did-cgs"></a>

## Unreleased — `sp.continuous_did(method="cgs")` is superseded

**What changed.** `method="cgs"` now emits a `DeprecationWarning` and will be
removed after one minor release. Use `sp.cgs_continuous_did` instead.

**Why.** That mode was an MVP standing in for an estimator StatsPAI did not
have: outcome regression only, a bootstrap standard error, and formula
details left as `[待核验]` in `docs/rfc/continuous_did_cgs.md`. The
replacement is the actual Callaway-Goodman-Bacon-Sant'Anna estimator, with
`ATT(d)` and `ACRT(d)` from a B-spline in the dose and an
influence-function variance — and it is pinned against the authors' own
`contdid` package: both curves at four grid points and both overall
quantities, across three spline specifications, agree to 1e-12.

**What to do.**

```python
# Before
sp.continuous_did(df, y="y", dose="d", time="t", id="i", method="cgs")

# After
sp.cgs_continuous_did(df, y="y", dose="d", time="t", unit="i", cohort="g",
                      degree=3, num_knots=0)
```

The new function needs a `cohort` column (the first-treatment period, 0 for
never-treated) rather than inferring a single pre/post split, which is what
lets it handle staggered adoption at all.

The other `continuous_did` modes — `twfe`, `att_gt`, `dose_response` — are
unchanged. They are dose-bin and local-linear heuristics, useful for a quick
look, and the docstring says so.

---

<a id="multiplegt-switch-directions"></a>

## Unreleased — ⚠️ `sp.did_multiplegt` dynamics/placebos, and switch-off in `_dyn`

**What changed.** Four numbers move, all on non-trivial designs:

| Function | What | Why |
| --- | --- | --- |
| `sp.did_multiplegt` | dynamic effect at horizon ≥ 1 | switchers who switch again inside the window are now excluded |
| `sp.did_multiplegt` | placebo value | the pre-window stability condition is applied |
| `sp.did_multiplegt_dyn` | everything, on non-absorbing panels | switch-off events are no longer dropped |

Nothing else moves. In particular the placebo's **sign is unchanged** — see
below.

**Why it went unnoticed.** All four were invisible without a working
reference, and the reference looked broken: `DIDmultiplegt` 2.x returns `NaN`
from `mode="old"` on its own bundled example. The archived **0.1.4** works,
and against it the static DID_M effect was already bit-exact while the
dynamic and placebo paths were not.

**On the placebo's sign: dCDH's own implementations disagree, and StatsPAI
now says so instead of choosing.** On `did::mpdta` the Stata and R packages
return the same three effects to six decimals and the same
`|placebo_1| = 0.024269` — with opposite signs. The new `placebo_sign`
parameter selects between them and **defaults to the Stata convention this
function has always used**, so nothing you have reported changes.

```python
sp.did_multiplegt(df, ..., placebo=1)                      # Stata sign (default)
sp.did_multiplegt(df, ..., placebo=1, placebo_sign="r")    # DIDmultiplegt sign
```

If you compare StatsPAI output against an R script, pass `placebo_sign="r"`
or the placebos will look like they disagree when only the convention does.

**What to do.** For the two rows above: nothing at the call site; re-run and
re-read. Absorbing
panels are unaffected by the `_dyn` change — with a binary treatment every
control already shares the baseline of zero, and that is verified rather than
assumed.

```python
# Both now match the reference; the counts are worth looking at too.
res = sp.did_multiplegt(df, y="y", group="i", time="t", treatment="d",
                        placebo=1, dynamic=1)
res.model_info["event_study"]      # placebo, effect, dynamic on one scale
```

---

<a id="multiplegt-dyn-placebo"></a>

## Unreleased — ⚠️ `sp.did_multiplegt_dyn` placebos are now the estimator's placebos

**What changed.** The placebo at lag ℓ was computed as
`Y_{F-1-ℓ} − Y_{F-1-ℓ-1}` — a one-period difference sliding backwards
through the pre-period. de Chaisemartin & D'Haultfœuille's placebo is the
effect window reflected about `F-1`: `Y_{F-1-ℓ} − Y_{F-1}`, a long difference
the same length as the effect it mirrors, reported with the reverse sign so
it sits on the same event-study scale. Every placebo value changes. Effects
are unaffected.

**Why.** Two reasons, and the second is the one that bites:

1. It is a different quantity. A one-period difference at lag ℓ does not
   mirror the ℓ-period effect and does not test what the paper's placebo
   tests.
2. It needed one more pre-period, so it silently used fewer cohorts. On the
   parity fixture, lag 1 ran on 96 switchers where `DIDmultiplegtDYN` uses
   146 — the earliest cohort was dropped without a word.

Since the placebos feed `model_info["joint_placebo_test"]`, the module's
parallel-trends diagnostic was testing the wrong contrast on the wrong
subsample. If you reported a passed placebo test from this function, re-run
it.

**What to do.** Nothing at the call site; re-run and re-read the placebos.
They now match `DIDmultiplegtDYN` 2.3.4 to 5e-15, switcher counts included
(`tests/reference_parity/test_multiplegt_dyn_parity.py`, Track A module
`78_multiplegt_dyn`).

**Related, not a break.** `aggregation="switchers"` is new and reproduces
the R package's `Av_tot_eff`; the default stays on the equal-weight average
over horizons, so the headline number is unchanged.

```python
sp.did_multiplegt_dyn(df, y="y", group="i", time="t", treatment="d",
                      dynamic=3, aggregation="switchers")
```

---

<a id="pretrends-power-test"></a>

## Unreleased — ⚠️ `sp.pretrends_power` defaults to the pre-test Roth (2022) analyses

**What changed.** `sp.pretrends_power(result)` returned the power of the
*joint Wald* test that all pre-period coefficients are zero. It now returns
the power of the coefficient-by-coefficient pre-test: reject if any
pre-period coefficient is individually significant at `alpha`. That is the
practice Roth (2022) analyses, and the quantity his `pretrends` R package
reports — the paper the docstring has always cited.

**Why.** The two answer different questions and are not close. On the
reference fixture at a linear violation of slope 0.02:

| | power |
| --- | --- |
| coefficient-by-coefficient (new default) | 0.332 |
| joint Wald (old default) | 0.157 |

They are not even on the same footing: the joint test has size exactly
`alpha`, while the eyeball test rejects above `alpha` under the null because
each of the K coefficients gets its own `alpha`-level look. Reporting the
Wald number under Roth's name understated how often a real trend would have
been spotted, which is the opposite of the paper's message.

**What to do.**

```python
# Previous behaviour, explicitly:
sp.pretrends_power(res, test="joint")["power"]

# Or read it off the new default call — both are always returned:
out = sp.pretrends_power(res)
out["power"]        # coefficient-by-coefficient
out["power_joint"]  # joint Wald, unchanged from before
```

No key was removed: `noncentrality` and `critical_value` are still reported
under both settings. New keys: `power_under_null`, `bayes_factor`,
`likelihood_ratio`, `test`, `threshold_tstat`, `power_joint`.

**New in the same release.** `sp.pretrends_slope_for_power(result,
target_power=0.5)` inverts the calculation — the slope of a linear pre-trend
the pre-test would catch half the time. It is the number to quote when a
reader asks what a passed pre-test actually rules out, and mirrors
`pretrends::slope_for_power`.

Both are pinned against `pretrends` 0.1.0 in
`tests/reference_parity/test_pretrends_power_parity.py` and Track A module
`76_pretrends`.

---

<a id="unified-sensitivity-term"></a>

## Unreleased — ⚠️ `sp.unified_sensitivity` analysed the intercept

**What changed.** `sp.unified_sensitivity(result)` pulled the coefficient to
analyse with `params.iloc[0]` and its standard error with
`std_errors.iloc[0]`. For a formula regression those are the **intercept**,
not the treatment. On the LaLonde baseline:

```text
what it analysed:  66.51    (Intercept)
what you meant:  1548.24    (treat)
```

The standard error came from the intercept too, so the entire dashboard —
E-value, breakdown point, Rosenbaum bounds — described a parameter nobody
asked about.

**Why you may not have noticed.** It usually raised instead of answering,
but for an unrelated reason: the intercept's CI spanned zero, the
risk-ratio conversion mapped `(-4892, 5025)` to `(4893, 5026)` via
`1 + |limit|`, that interval excludes the converted point estimate `67.5`,
and an assertion inside `evalue` fired with "Point estimate should lie
inside the CI." Designs whose intercept CI stays positive skipped that
tripwire and got a confident wrong number.

**Migration.** Name the coefficient:

```python
# before — silently analysed the Intercept
sp.unified_sensitivity(ols_fit)

# after — explicit, and the only form that still works for a multi-term fit
sp.unified_sensitivity(ols_fit, term="treat")
```

If you already pass `treat=` (it names the treatment for the Sensemakr
component), that doubles as the term — no need to name the same column
twice:

```python
sp.unified_sensitivity(fit, data=df, y="re78", treat="treat", controls=X)
```

With more than one non-intercept coefficient and neither `term=` nor
`treat=`, the function now raises `MethodIncompatibility` listing the
candidates rather than guessing. A fit
with exactly one non-intercept coefficient still needs no `term=`, and
results exposing a scalar `.estimate` / `.ate` (`CausalResult` and friends)
are unaffected — they never went through the coefficient path.

If you published a robustness claim produced by the old code path on a
multi-term regression, re-run it with `term=` — the previous output did not
describe your treatment effect.

**Related.** `sp.sensitivity_dashboard` does *not* share this defect: it
already skipped intercept-like names. It did, however, return an empty
dashboard graded `overall_stability='?'` when called without `data=`,
because most of its dimensions re-estimate on perturbed samples. That case
now emits a `RuntimeWarning` instead of looking like a pass.

---

<a id="aipw-default-seed"></a>

## Unreleased — ⚠️ `sp.aipw` was not reproducible; default `seed` is now 42

**What changed.** `sp.aipw(..., seed=...)` defaulted to `None`, which reached
`np.random.default_rng(None)` and therefore seeded the cross-fitting fold
split from OS entropy. The default is now `42`.

**Why it mattered.** Three identical calls on the same 614-row LaLonde frame:

```text
+308.87    +149.84    +905.21
```

That spread is wider than the treatment effect being estimated, so which
number reached your paper depended on when you happened to run the script.
It could not be pinned from the outside either — `np.random.default_rng`
does not consult the legacy global RNG, so `np.random.seed(7)` before every
call changed nothing.

It propagated one level up: `sp.causal_question(...).estimate()` resolves a
selection-on-observables plan to cross-fitted AIPW, so the headline estimate
of a whole estimand-first pipeline moved between runs.

**Migration.**

```python
# Reproducible (the new default) — nothing to do:
sp.aipw(df, y="re78", treat="treat", covariates=X)

# Old behaviour: a fresh random fold split on every call.
sp.aipw(df, y="re78", treat="treat", covariates=X, seed=None)
```

If you published a number produced by the old default, you cannot reproduce
it by re-running — the fold split that generated it is gone. Re-estimate
with the new default and report that number instead.

**Scope.** An audit of the stochastic surface found `sp.dml`, `sp.tmle` and
`sp.metalearner` already deterministic by default; `sp.tmle`, `sp.bcf` and
`sp.super_learner` already defaulted to `42`. `sp.aipw` was the only
offender. The seed actually used is now recorded in
`result.model_info['seed']`, and `tests/test_estimator_determinism.py` pins
the convention for the whole family.

---

<a id="nsw-lalonde-default-simulated-false"></a>

## Unreleased — bundled datasets now default to the real published data

**What changed.** Five loaders ship a real extract in
`statspai/datasets/data/`. Four of them also offer a calibrated replica
behind `simulated=`, and their defaults now all point at the real data:
`card_1995`, `lee_2008_senate`, `california_prop99` and `nsw_lalonde`.

The rule is now uniform: **if StatsPAI ships the real published data, a
bare call returns it.** Previously only `nsw_lalonde` behaved that way, so
what you got depended on which dataset you reached for.

Why the real extract is the better default: it is what reproduces the
papers. `card_1995` returns OLS 0.074 / IV 0.132 against Table 2's
0.075 / 0.132 — the replica gives 0.110 / 0.142.

`nsw_lalonde` in detail:

| | old default (`simulated=True`) | new default (`simulated=False`) |
| --- | --- | --- |
| shape | `(445, 10)` | `(614, 11)` |
| columns | no `hispanic` | adds `black`, `hispanic` |
| naive OLS ATT | ≈ **+$1,794** | ≈ **−$635** |
| `df.attrs['data_source']` | `'simulated'` | `'real'` |

**Why.** The old default was a quiet correctness trap. `sp.datasets`'s own
first example is a bare `nsw_lalonde()`, so a reader following the docs got
simulated numbers that match no published table while believing they were
looking at LaLonde's data. Defaulting to the real extract makes the honest
path the default one; the replica stays available and is still the right
choice when you want the *experimental* subset.

**Migration.**

```python
# The real extract is now the default for all five bundled datasets:
df = sp.datasets.card_1995()          # (3010, 9) real NLSYM
df = sp.datasets.lee_2008_senate()    # (1390, 2) rdrobust RDsenate
df = sp.datasets.california_prop99()  # (1209, 8) ADH smoking panel
df = sp.datasets.nsw_lalonde()        # (614, 11) MatchIt::lalonde

# The calibrated replicas remain available:
df = sp.datasets.nsw_lalonde(simulated=True)
```

**Shapes change where the two variants differ.** `california_prop99` keeps
its columns (order only) and `card_1995` gains `nearc2`, so those are
near-transparent. `lee_2008_senate` differs materially — the real
`rdrobust::rdrobust_RDsenate` extract is 1,390 rows with columns `x`
(lagged Democratic margin, percent points) and `y` (current vote share),
against the replica's 6,558 rows of `margin` / `voteshare_next` on a 0-1
scale. The two are genuinely different frames on different scales; they
are deliberately *not* forced into a shared vocabulary, because giving
them the same column names on different units would be a worse trap than
the shape change. Pass `simulated=True` to keep the replica.

The signature is a plain `simulated: bool = False` — no sentinel, no
warning to silence. If you relied on the old default, pass
`simulated=True`.

`sp.datasets.list_datasets()` now carries a `source` column saying which
variant each bare `name()` call returns, so you can see at a glance
whether you are getting a real extract or a calibrated replica.

**Offline note.** The real extract is a CSV bundled in the wheel under
`statspai/datasets/data/`, so this default needs no network. Verified by
loading it in a clean venv built from the wheel with `socket` hard-blocked.

---

<a id="ltmle-influence-curve-martingale-term"></a>

## Unreleased — ⚠️ `sp.ltmle` standard errors were 250–400× too small

**What changed.** The efficient influence curve for LTMLE is

    sum_k H_k (Q*_{k+1} - Q*_k)  +  (Q*_1 - psi)

Only the second term was being computed. What the function reported as a
standard error was therefore the dispersion of a fitted conditional mean, not
the sampling variability of the estimator. The martingale sum is now
accumulated across time points.

**Effect.** Point estimates are **unchanged**. Standard errors, confidence
intervals, p-values and any significance marks in `.summary()` all change, and
the old ones were not usable: on a two-period DGP with a known ATE the
reported SE was 0.00024 where the estimator's actual Monte-Carlo standard
deviation was 0.059 (n = 500). The discrepancy grew with sample size — 353× at
n = 2000, 405× at n = 8000 — because the reported quantity was not converging
at the √n rate at all.

**What to do.** Re-run anything that used `sp.ltmle` for inference. Any
conclusion that rested on an `sp.ltmle` confidence interval or p-value should
be treated as unsupported until recomputed; intervals will be roughly two
orders of magnitude wider.

**Remaining limitation, now quantified.** The module targets with a one-step
fluctuation rather than iterating to convergence, so the martingale sum is
near zero but not identically zero and the SE stays mildly anti-conservative.
Over 200 replications it runs ~13% below the Monte-Carlo standard deviation at
n = 1000 (nominal-95% coverage 0.905) and ~7% below at n = 4000 (coverage
0.930). This is stated with those numbers in the function's Notes. For
inference needing honest coverage with flexible ML nuisances, a full
CV-LTMLE / iterated-targeting path is still the right tool and is tracked as a
follow-up.

---

<a id="gmm-unadjusted-variance-and-conventions"></a>

## Unreleased — ⚠️ `sp.gmm` variance, closed form, and HAC conventions

**What changed.** Three things in `sp.gmm`.

1. `se='unadjusted'` returned `(D'WD)⁻¹/n` for every weighting matrix. That
   formula is the estimator's variance only at the efficient weight `W = S⁻¹`;
   under any other weight it is smaller than the truth. It now warns.
   `se='robust'` — the default — was already the sandwich and is unchanged.
2. Moment conditions affine in θ were minimised with BFGS. They have a closed
   form, which is now used; `diagnostics['n_iter'] == 0` reports it.
3. The Bartlett HAC kernel is now evaluated at `lag/bandwidth` (vanishing at
   `lag == bandwidth`), matching R `sandwich`.

**Effect.** `se='unadjusted'` results are unchanged numerically but now carry
a warning wherever they were wrong. Point estimates for linear moment
conditions change in the last digits — they are now the exact minimiser
rather than BFGS's approximation, agreeing with R's analytic two-step to
1e-12. HAC standard errors change by percent-level amounts if you were
relying on the previous bandwidth convention.

**What to do.** If you reported `se='unadjusted'` standard errors from a
one-step fit or with an explicit `W=`, re-run: those numbers were too small.
Switch to `se='robust'`, or use a weight-updating method (`'twostep'`,
`'iterative'`, `'cue'`) so the efficient formula actually applies.

**New parameters.** `jacobian=` (analytic `D(theta)`), `vcov=` (`'mds'`,
`'iid'`, `'hac'`, `'cluster'`), `cluster=`, `hac_bandwidth=`, and `center=` —
moment centring, which R `gmm` does by default and Stata does not. `center`
defaults to Stata's convention, so existing results are unchanged.

---

<a id="xtabond-twostep-ar-test"></a>

## Unreleased - `sp.xtabond` two-step AR(1)/AR(2) statistics changed

**What changed.** The Arellano-Bond serial-correlation test variance
decomposes into three terms, the last of which is
`(W'q)' Avar(beta) (W'q)`. StatsPAI always evaluated it at the uncorrected
robust sandwich. When `twostep=True` the *reported* VCE is either the
Windmeijer-corrected one (`robust=True`) or the conventional
`(W'ZA2Z'W)^-1` (`robust=False`), so the test was using a variance the
coefficient table did not.

**Who is affected.** Only `twostep=True` fits, and only the `ar1_z` /
`ar1_p` / `ar2_z` / `ar2_p` fields - coefficients and standard errors are
untouched. One-step fits are bit-identical: there the reported and naive
VCEs are the same matrix and the correction is exactly zero.

**How large was the error.** On Stata's `abdata`:

| spec | old AR(1) z | new | Stata |
| --- | --- | --- | --- |
| `lags=1, twostep=True` | -2.2438 | -2.1000 | -2.1000 |
| AB(1991) Table 4, `twostep=True` | -4.3229 | -3.1030 | -3.1030 |

**What to do.** Re-read any AR(2) conclusion drawn from a two-step fit. The
direction of the change is not uniform - it can move the statistic either
way - so a previously "passing" AR(2) test is not automatically safe.

---

<a id="gmm-unadjusted-se"></a>

## Unreleased - `sp.gmm(se='unadjusted')` now reports the efficient-GMM variance

**What changed.** `se='unadjusted'` used to return `(D'WD)^-1/n` for
whatever weight matrix `W` was in force. That expression is the variance of
the GMM estimator **only when `W` is efficient** (`W = S^-1`); with any
other weight the estimator's variance is the sandwich
`(D'WD)^-1 D'W S W D (D'WD)^-1 / n`, which is generally larger. The
reported standard errors were therefore too small in exactly the case a
user reaches for a custom `W`.

It now returns `(D' S^-1 D)^-1/n` - the efficiency bound - and warns when
the weight actually used is not efficient, pointing at `se='robust'`.

**Who is affected.** Only calls that combined `se='unadjusted'` with a
non-efficient weight: `method='onestep'` with an explicit `W=`, or the
identity default. Two-step, iterated and CUE fits are **unchanged**,
because at the efficient weight `(D'WD)^-1` and `(D'S^-1 D)^-1` are the
same matrix.

**What to do.** Nothing if you used the default `se='robust'`. If you
relied on `se='unadjusted'` with a custom weight, switch to `se='robust'`:
that is the variance of the estimator you actually computed.

---

<a id="xtabond-listwise-deletion"></a>

## Unreleased — `sp.xtabond` no longer deletes instruments on covariate `NaN`s

**What changed.** `sp.xtabond` used to start with

```python
df = data[[id, time, y] + x].dropna()
```

Listwise deletion across *all* columns means a missing value in any covariate
at period *t* removes that row entirely — and with it `y_{i,t}` as a GMM
instrument and as a lag source, not merely as an estimation observation.
Availability is now evaluated **per variable**: a covariate that is
unobserved early costs only the equations that need it.

**Who is affected.** Only fits where some covariate had missing values in the
estimation window — most commonly because the user built lagged regressors by
hand, which necessarily leaves leading `NaN`s. If every covariate was complete
over the periods used, nothing changes; `tests/test_xtabond_golden.py` locks
that.

**How large was the error.** On Stata's `abdata` panel with the Arellano-Bond
(1991) Table 4 specification (`n` on two lags of `n`, `l(0/1).w`, `l(0/2).k`):

| | old `sp.xtabond` | new | Stata `xtabond` |
| --- | --- | --- | --- |
| observations | 331 | 611 | 611 |
| instruments | 19 | 32 | 32 |
| ρ̂₁ | 0.660 | 0.849 | 0.849 |

**What to do.** Nothing, unless you have published numbers from an affected
fit; re-run those. Hand-built lag columns are no longer necessary either —
`x=["l(0/1).w", "l(0/2).k"]` is now accepted directly.

---

<a id="qte-firpo-mislabel"></a>

## Unreleased — `sp.qte` method names and default

**What changed.** `sp.qte(method='quantile_regression')` was documented,
labelled and registered as Firpo (2007). It is not. It returns the
coefficient on `D` in a quantile regression of `Y` on `D + controls` — a
**conditional** QTE, which absent rank invariance is not a treatment effect
on any quantile of the outcome distribution. Firpo (2007) is the
**unconditional** estimator, which reweights by the propensity score and
compares the marginal quantiles of `Y(1)` and `Y(0)`.

| Old | New | Numbers |
| --- | --- | --- |
| `method='quantile_regression'` | `method='conditional_qr'` | **unchanged** |
| — (did not exist) | `method='firpo_qte'` | new — the actual Firpo QTE |
| — (did not exist) | `method='firpo_qtt'` | new — Firpo QTT |
| `method='distribution'` | `method='distribution'` | **unchanged**, but now labelled QTT rather than QTE, which is what it always computed |

**The default changed** from `'quantile_regression'` to `'firpo_qte'`. A
call that relied on the default now returns a different estimand. Pass
`method='conditional_qr'` explicitly to keep the old numbers.

`method='quantile_regression'` still works and emits a
`DeprecationWarning`; it is removed in 1.23.0.

**Which should you use?** If you want "the effect on the median worker",
that is the unconditional `'firpo_qte'` (or `'firpo_qtt'` for the effect on
treated units). `'conditional_qr'` answers "holding covariates fixed, how
does the τ-th conditional quantile shift" — a within-cell statement that
does not aggregate to a distributional effect.

**`sp.qdid` reference correction.** `sp.qdid` was described as Athey &
Imbens (2006) changes-in-changes in its docstring, its method label and the
registry. It implements **QDiD** — the DiD contrast applied to quantiles —
which is the estimator Athey & Imbens propose CiC *in place of*, and
criticise directly. **No numbers change**; only the attribution. For
changes-in-changes use `sp.cic`.

---

<a id="dist-iv-quantile-wald-ratio"></a>

## Unreleased — ⚠️ `sp.dist_iv` estimated the wrong object

**What changed.** `sp.dist_iv` (and its alias `sp.kan_dlate`) computed a
*Wald ratio of quantiles*:

```text
LATE_q(τ) = [Q(τ | Z=1) − Q(τ | Z=0)] / [E(D | Z=1) − E(D | Z=0)]
```

The quantile operator is not linear, so the mean-Wald rescaling that makes
the ordinary LATE work does not carry over. That expression is inconsistent
for any quantile estimand — it is not a noisier version of the complier QTE,
it converges to something else. It now uses Abadie (2002, 2003) κ-weighted
complier CDFs and returns

```text
QTE_c(τ) = F⁻¹_{Y(1)|complier}(τ) − F⁻¹_{Y(0)|complier}(τ)
```

**Effect.** Every `sp.dist_iv` / `sp.kan_dlate` number changes. The old bias
was multiplicative in the first stage: on a design with a true complier
`QTE(τ) ≡ 2.0` and `Δp = 0.5`, the old code returned ≈ 4.0 at n = 200,000.

**Approximate back-conversion.** The old estimator was roughly

```text
old(τ)  ≈  [Q(τ|Z=1) − Q(τ|Z=0)] / Δp
```

so when the treated and control quantile curves are near-parallel you can
sanity-check an archived figure with `new(τ) · Δp ≈ Q(τ|Z=1) − Q(τ|Z=0)`,
i.e. **`old(τ) ≈ new(τ) / Δp`** only in the special case of a homogeneous
shift among compliers with no always-takers. With always-takers present
there is no exact conversion — the old quantiles mixed compliers,
always-takers and never-takers in proportions that depend on τ. **Re-run
the estimation.** There is no flag restoring the old behaviour; it was not
an alternative convention.

**Other behaviour changes in the same release.**

| Before | After |
| --- | --- |
| `covariates=` accepted, then silently discarded | Selects Frölich & Melly (2013) unconditional IV-QTE weighting; changes the estimate |
| Constant instrument → all-`NaN` result object + warning | Raises `ValueError` |
| Near-zero first stage → silent estimate | `UserWarning` naming the complier share and first-stage *t* |
| Bootstrap SE only | Analytic influence-function SE by default (`se='auto'`); bootstrap when covariates are supplied |

**`sp.kan_dlate` is deprecated** (removal in 1.23.0). It was always a pure
alias for `sp.dist_iv` and never implemented a Kolmogorov-Arnold bridge
function. Its docstrings also attributed arXiv:2506.12765 to two different
authors; verification against arXiv and the DataCite DOI registry shows the
paper is *Model Risk in Machine-Learning Distributional IV Estimation* by
**Charles Shaw** alone, and neither its title nor its v1 abstract mentions a
KAN. Call `sp.dist_iv` directly.

---

<a id="genmatch-variance-basis"></a>

## Unreleased — ⚠️ `sp.genmatch` distance uses full-sample variances

**What changed.** The genetic-matching kernel computed its generalised
distance after standardising covariates by the **control group's**
variances. Its own module docstring specified `D' S^(-1/2) W S^(-1/2) D`,
and the metric `Matching::Match(Weight = 3, Weight.matrix = W)` implements
is the diagonal of the **full-sample** variances. The kernel now uses
those.

**Effect.** Genetic-matching weights, matched pairs and ATT all change.
Given a fixed diagonal `W`, the kernel now reproduces `Matching::Match`'s
assignment on every uniquely matched treated unit of `MatchIt::lalonde`
(163/163).

**What to do.** Re-run any `sp.genmatch` analysis. Note the genetic
*search* is stochastic and was never reproducible across languages or
seeds; only the deterministic distance-and-assignment kernel is pinned, in
`tests/reference_parity/test_matching_r_parity.py`.

---

<a id="sbw-tolerance-scale"></a>

## Unreleased — `sp.sbw` balance tolerance now names its units

**What changed.** `delta` was always interpreted against the full-sample
standard deviation. `sbw::sbw` quotes the same tolerance against either
the target group (`bal_std="target"`, treated units under ATT) or the
group being reweighted (`bal_std="group"`, controls), so a tolerance alone
did not determine the estimator. `tolerance_scale` now selects among
`'sd'` (the previous behaviour, still the default), `'target'`, `'group'`
and `'raw'`.

**Effect.** None by default. But the conventions are not interchangeable:
on `MatchIt::lalonde` at `delta = 0.05` the ATT is 1330.30 under
`'target'`, 1335.87 under `'sd'` and 1342.89 under `'group'`.

**What to do.** When reconciling with `sbw::sbw`, set `tolerance_scale` to
match its `bal_std`. When reporting a tolerance, report the scale too.

---

<a id="match-with-replacement-ties"></a>

## Unreleased — `sp.match` can now pool tied controls

**What changed.** Under matching *with replacement*, `sp.match` kept only
the lowest-index control among equidistant candidates. `ties='all'` pools
them and splits the weight, and `tie_tolerance` sets how close squared
distances must be to count as tied.

**Effect.** None by default (`ties='first'`). With
`ties='all', tie_tolerance=1e-5` the ATT and the Abadie-Imbens population
standard error match `Matching::Match` exactly.

**What to do.** Nothing is required. Use `ties='all'` if you would rather
not have row order decide which of several equally good controls is used,
and add `tie_tolerance=1e-5` when reconciling with `Matching::Match`.

---

<a id="cbps-solver-rewrite"></a>

## Unreleased — ⚠️ `sp.cbps` now solves the Imai-Ratkovic problem

**What changed.** The CBPS GMM was posed in the raw covariate basis with an
empirical outer-product weighting matrix. CBPS is defined in a standardised,
orthonormalised basis with the *model-implied* moment covariance frozen at the
starting value. Neither the just-identified quadratic form nor the GMM
weighting is invariant to that change of basis, so the old code minimised a
different objective and returned a different estimator — not a less precise
version of the same one.

**Effect.** Every `sp.cbps` estimate changes. On `MatchIt::lalonde` the
old `estimand='ATE', variant='over'` result was **8.6x** off `CBPS::CBPS`
(1585.99 vs 165.88); `ATT`/`over` was 25% off; coefficients differed by up
to 170%. After the rewrite, ATE (both variants) and ATT/`exact` agree with
R to ≤5e-3 relative.

**What to do.** Re-run any analysis whose numbers came from `sp.cbps`. If
you need to reproduce an old figure, there is no flag for the previous
behaviour — it was not an alternative convention, it was the wrong problem.
For a sanity check on the new results, `variant='exact'` must now balance
covariates to |SMD| < 1e-6; that identity holds only for the corrected
solver.

**Note on `estimand='ATT', variant='over'`.** StatsPAI deliberately does
*not* reproduce `CBPS::CBPS` here. CBPS's analytic ATT gradient divides the
balance block by `n_1` where the moment's Jacobian carries `1/n`,
overstating it by `n/n_1`, and its `optim` call stops at a non-stationary
point as a result. StatsPAI uses the correct Jacobian and attains both a
lower GMM objective and better covariate balance (max |SMD| 0.037 vs 0.106
on lalonde). This is asserted in
`tests/reference_parity/test_matching_r_parity.py`.

---

<a id="ebalance-exact-balance"></a>

## Unreleased — ⚠️ `sp.ebalance` now achieves exact moment balance

**What changed.** The entropy-balancing dual was minimised with L-BFGS-B on
unscaled constraints. When covariates live on different scales the dual
Hessian is badly conditioned and the optimiser stops early, so the weights
did not match the targeted moments — which is the one property entropy
balancing is defined by. The convergence check could not catch it either:
it compared an *absolute* moment gap against 0.01, which is meaningless
when one constraint is a 0/1 indicator and the next is annual earnings in
dollars.

**Effect.** On `MatchIt::lalonde` the reweighted control mean of `re74` was
2.66 away from the treated mean (1.3e-3 relative); it is now 1e-15
relative. ATT estimates move in the 3rd significant figure (1269.45 →
1273.26, against `ebal::ebalance`'s 1273.26).

**What to do.** Re-run affected analyses. The convergence warning now fires
on a standardised gap above 1e-6, so a result that previously passed
silently may now warn — that warning is correct and means the treated
moments are likely outside the convex hull of the control moments.
`model_info['max_standardized_moment_gap']` records the achieved gap.

---

<a id="tmle-shared-nuisance-and-fluctuation"></a>

## Unreleased — `sp.tmle` gains `Q` / `g1W` / `fluctuation`

**What changed.** Three new parameters. `Q` takes an `(n, 2)` matrix of
`[Q(0,W), Q(1,W)]` and `g1W` a propensity vector; supplying either bypasses
the corresponding Super Learner stage, so the targeting step can run on
externally-estimated nuisances. `fluctuation` selects the submodel used in
the targeting step.

**Effect on existing code: none.** The default `fluctuation='single'` is the
one-clever-covariate submodel `H(A,W) = A/g - (1-A)/(1-g)` StatsPAI has
always used, and its numbers are unchanged. `model_info['epsilon']` keeps its
scalar type on that path.

**What is new.** `fluctuation='per_arm'` fits two clever covariates, `A/g`
and `-(1-A)/(1-g)`, jointly — the submodel the R `tmle` package uses. Both
are valid TMLEs solving the efficient-influence-function equation and are
asymptotically equivalent, but they differ at finite *n*: about 1.3e-3
relatively on the Track A module-72 fixture. Use `'per_arm'` when
reconciling against `tmle::tmle`, which it reproduces to ~1e-9 on a shared
initial fit.

`model_info` gains `epsilon_vec` (the full fluctuation vector in both
modes), `fluctuation`, and `nuisance_source`. Under `'per_arm'` the scalar
`model_info['epsilon']` is `None`, because no scalar fluctuation parameter
exists there; read `epsilon_vec` instead. `sl_outcome_weights` /
`sl_propensity_weights` are `None` when the corresponding nuisance was
supplied — there is no ensemble to report weights for.

**What to do.** Nothing. If you compare StatsPAI against `tmle::tmle`, pass
`fluctuation='per_arm'` and supply the same `Q` / `g1W` to both.

---

<a id="dml-panel-learner-aliases"></a>

## Unreleased — `sp.dml_panel` accepts `sp.dml`'s learner aliases

**What changed.** `sp.dml(ml_g='linear')` resolved short learner names;
`sp.dml_panel(ml_g='linear')` did not, and failed inside scikit-learn's
`clone()` with `TypeError: Cannot clone object ''linear''`, a message that
named neither the offending parameter nor the accepted values. Both now
route through the same `resolve_learner`.

**Effect.** Strictly additive — calls that passed estimator instances are
unaffected, and calls that passed strings previously raised.

**What to do.** Nothing.

---

<a id="dml-sensitivity-structural-residual"></a>

## Unreleased — ⚠️ `sp.dml_sensitivity` uses the structural outcome residual

**What changed.** The DML omitted-variable-bias bound scales by
`S = sqrt(σ²ν²)`, where for the PLR coefficient
`σ² = E[(Y − ℓ(X) − θ(D − m(X)))²]` and `ν² = 1/E[(D − m(X))²]`. StatsPAI
computed the numerator as `sd(Y − ℓ(X))`, i.e. without removing the treatment's
own contribution. Because `sd(Y − ℓ)² = σ² + θ²·sd(D − m)²`, the scaling factor
was systematically too large.

**Effect.** `rv_q`, `rv_qa`, `bias_bound`, `adjusted_estimate_low/high`, `s`,
and every row of `benchmarks` change. The direction is consistent: the old
code **overstated the bias bound and understated the robustness value**, so it
portrayed estimates as *more* fragile to unobserved confounding than the bound
warrants. On a linear-nuisance PLR fit (n = 1500) the bias bound was 0.0671
instead of 0.0529 (27% too large) and `RV_1` was 0.454 instead of 0.533.

After the fix, `bias_bound` and the adjusted `theta` bounds match
`doubleml`'s `sensitivity_analysis` to 2.5e-15 and `RV` to 9.2e-8 on a shared
fold partition (`tests/external_parity/test_dml_sensitivity_parity.py`).

`model='irm'` is **unaffected**: it stores `y_resid` as the score residual
`ψ − θ̂`, which is already centred, so subtracting `θ·d_resid` again would
double-count.

**What to do.** Re-run any reported robustness values. If you previously
concluded that a DML estimate was *not* robust on the basis of a low `RV_q`,
recheck — the corrected value is higher. `rv_qa` remains a StatsPAI
convention: it exhausts `|θ| − z·se` using the unadjusted standard error,
whereas `doubleml` lets the standard error move with the confounding scenario;
the two differ by about 0.14% on the pinned fixture and that gap is asserted
rather than hidden.

---

<a id="dml-irm-iivm-se-normalisation"></a>

## Unreleased — ⚠️ `sp.dml` IRM / IIVM standard errors normalise by `n`

**What changed.** The unweighted IRM and IIVM branches divided the
influence-function variance by `n − 1` (`ddof=1`). Nothing else in the module
did: PLR and PLIV use `n`, the weighted IRM/IIVM branches use `n`, and even
the `normalize_ipw`/ATTE branch *inside `irm.py`* uses `mean(psi**2)`, i.e.
`n`. So `sp.dml(model='plr')` and `sp.dml(model='irm')` on the same data
reported standard errors under two different conventions, and which one you
got for IRM depended on whether you passed `normalize_ipw`. `DoubleML`
normalises by `n`; all paths now agree.

**Effect.** IRM and IIVM standard errors shrink by exactly `sqrt((n−1)/n)`.
That is 0.025% at n = 2000 and about 1% at n = 50 — immaterial for most
reported results, but it is the difference between matching `DoubleML` and
not. On the Track A module-71 fixture the IRM/IIVM standard errors now agree
with `DoubleML` 1.0.2 to 1.1e-10 on a shared fold partition (they were off by
the `sqrt(n/(n−1))` factor, observed ratio 1.00025009389849 against
`sqrt(2000/1999) = 1.00025009378908`). **Point estimates are unchanged**, as
are PLR and PLIV in full.

**What to do.** Nothing. Confidence intervals narrow very slightly; if you
need the old figures, multiply the reported SE by `sqrt(n/(n−1))`.

---

<a id="dml-fold-indices-all-models"></a>

## Unreleased — `sp.dml(fold_indices=...)` now works for IRM / PLIV / IIVM

**What changed.** `fold_indices=` was accepted only for `model='plr'`; the
other three model classes raised `MethodIncompatibility` rather than silently
ignore the argument. All four now route caller-supplied folds through a
shared `_make_splits` helper, so cross-fitting can be pinned to an explicit
partition for every model.

**Effect.** No change to any existing result — the parameter previously
raised, so nothing depended on it. With folds supplied, the estimate becomes
independent of `random_state`, which is what makes a bit-exact comparison
against `DoubleML` possible (Track A module 71).

**What to do.** Nothing is required. If you supply folds for `irm` or
`iivm`, note that you are bypassing the built-in `StratifiedKFold`: each
training set must contain both classes of the binary nuisance target, and a
partition that violates this now raises `DataInsufficient` naming the
offending fold rather than fitting a degenerate classifier.

---

<a id="causal-forest-grf-att-convention"></a>

## Unreleased — ⚠️ Causal-forest ATT/ATC now use grf's estimator

**What changed.** `sp.causal_forest(...).average_treatment_effect(
target_sample='treated')` (and `'control'`) computed the mean of a single
Robins doubly-robust score divided by `p̂₁`. `grf::average_treatment_effect`
does something structurally different: it reports the **plug-in CATE average
over the target arm plus a Hájek-normalised doubly-robust correction**, and
reports the standard error as the square root of the *sum* of the two
components' variances — the plug-in dispersion
`Σ_{i:Tᵢ=1}(τ̂ᵢ - τ̄)² / n₁²` plus `n/(n-1) · Σᵢ Δᵢ² / n²` — rather than the
dispersion of one score vector. StatsPAI's docstring claimed GRF-style
ATT/ATC aggregation, so this was a documentation/implementation mismatch,
not a deliberate alternative convention.

**Effect.** ATT and ATC point estimates change slightly and their standard
errors change materially. Given `grf`'s own forest outputs — so that the
forest is held fixed and the comparison isolates the formula — the old
route matched `grf`'s ATT point estimate to 9.3e-5 but produced a standard
error **12% larger**. The new code reproduces `grf` 2.6.1's ATT estimate and
`std.err` to 1e-15 on those same inputs. **ATE and ATO are unchanged**, and
the ATE score vector was already elementwise identical to `grf::get_scores`.

On the Track A module-13 clean-overlap fixture, where the two sides grow
independent forests, the ATT standard-error gap against `grf` fell from
14.6% to 0.087%.

**What to do.** Nothing is required; the new numbers are the ones the
documentation always described. If you reported causal-forest ATT/ATC
standard errors from an earlier version, re-run — the previous figures were
conservative (too wide) rather than anti-conservative, so significance
claims do not flip in the dangerous direction, but they were not `grf`'s.
The two operators are now exposed directly as
`statspai.forest.forest_inference.aipw_scores` (the ATE influence function,
elementwise equal to `grf::get_scores`) and `grf_att_atc` (the ATT/ATC
decomposition), so the formula can be inspected and reused without going
through a fitted forest.

---

<a id="policy-tree-exact-depth2-search"></a>

## Unreleased — ⚠️ `sp.policy_tree` now solves the depth-2 problem exactly

**What changed.** The module documented an exhaustive depth-1/depth-2 search
but implemented a greedy one: each candidate root split was scored as though
both of its children were terminal leaves, and only then did the routine
recurse. That one-step lookahead is exact for a depth-1 stump, but for
`max_depth=2` — the default — the root split that scores best with terminal
children is routinely *not* the root split that admits the best pair of
depth-1 subtrees. Candidate thresholds were also subsampled to at most 50
quantiles per covariate, so the returned tree was not even the greedy optimum
over the full split grid.

Depth ≤ 2 now maximises the Athey–Wager objective
`sum_i Gamma_i * pi(X_i)` exactly, by exhaustive search over the complete
grid of distinct covariate values, matching `policytree::policy_tree`'s
`x <= t` split convention and its "smallest permitted terminal node" reading
of `min_leaf_size`.

**Effect.** For `max_depth=2`, the learned policy and every quantity derived
from it — `policy`, `value_policy`, `value_gain`, `fraction_treated`,
`rules` — can change. On the Track A module-70 fixture the old search fell
0.70% short of the welfare optimum and assigned 78 of 1200 units to the
wrong arm. `max_depth=1` is unaffected (greedy is exact for a stump), except
where the 50-quantile threshold subsample previously missed the best split.
Depth ≥ 3 is unchanged in kind — still greedy, because exhaustive search is
combinatorially infeasible — but now searches the full threshold grid.

**What to do.** Nothing is required; the new numbers are the ones the
documentation always promised, and they now agree with `policytree` 1.2.4 to
9.6e-16 with all per-row policy decisions identical. If you need to
reproduce an earlier figure, pass `search='greedy'` — but note that this
still searches the full grid, so it does not reproduce the old
50-quantile behaviour exactly. Check `result['search_mode']` to see which
search actually ran; `search='auto'` (the default) falls back to greedy with
a `UserWarning` when the exact sweep would exceed its cost budget, and
`split_step=k` thins the candidate grid the way `policytree`'s `split.step`
does.

---

<a id="mahalanobis-pooled-covariance"></a>

## Unreleased — ⚠️ Mahalanobis matching uses the pooled within-group covariance

**What changed.** `sp.match(distance='mahalanobis')` and
`sp.optimal_match(metric='mahalanobis')` built the metric from `cov(X)` over
the pooled sample. The Mahalanobis matching metric of Rubin (1980) — the
reference the module already cited, and the one `MatchIt` uses — is the
pooled *within-group* covariance `[(n₁−1)S₁ + (n₀−1)S₀] / (n₁+n₀−2)`. The
total covariance is inflated along the direction in which the group means
differ, which is exactly the direction matching needs to resolve most
finely, so it systematically under-weights the covariates that separate the
groups.

**Effect.** All Mahalanobis matching estimates change. The new default
reproduces `MatchIt:::mahalanobis_dist` to 1e-15.

**What to do.** Pass `mahalanobis_cov='total'` to restore the previous
metric if you need to reproduce an earlier figure. New work should keep the
default.

---

<a id="match-m-order"></a>

## Unreleased — `sp.match` greedy matching order is now explicit

**What changed.** Nearest-neighbour matching *without replacement* is
order-dependent: each treated unit consumes a control, so who is matched
first changes who is left. StatsPAI processed treated units
closest-pair-first without documenting it. That is now the `m_order`
parameter.

**Effect.** None by default — `m_order='smallest_min_dist'` is the previous
behaviour. But the choice is material: on `MatchIt::lalonde` with
Mahalanobis distance the ATT ranges over more than 5x across orders, so if
you are comparing against another package you should set it explicitly.
`m_order='data'` and `'closest'` reproduce the MatchIt rules of the same
name exactly.

**What to do.** Nothing is required. When reconciling with R, set
`m_order='data'` (MatchIt's default for non-propensity distances) or
`'closest'`.

---

<a id="honest-did-flci"></a>

## Unreleased — ⚠️ `sp.honest_did(method='smoothness')` now returns the real FLCI

**What changed.** The native smoothness path returned
`θ̂ ± M·(e+1) ± z·SE`: the worst-case bias added to an ordinary Wald interval.
That is not the Rambachan-Roth confidence set — it ignores the pre-period
covariance and was *narrower* than the reference at every M, overstating how
robust a result is to parallel-trends violations. It now solves the actual
fixed-length confidence interval.

Separately, `backend='r'` was building `sigma <- diag(ses^2)` before calling
`HonestDiD`, discarding the cross-period covariance. Both backends now receive
the full event-study covariance recovered from the influence functions.

**Effect.** All `method='smoothness'` intervals move. Two changes will look
surprising and are correct:

- **The interval is no longer centred on the event-study coefficient.** Its
  centre is the optimal affine estimator, which extrapolates the pre-trend.
- **`M=0` no longer equals the Wald interval.** `Δ^SD(0)` still permits an
  arbitrary *linear* pre-trend, so the M=0 FLCI prices in that extrapolation.
  R `HonestDiD` behaves identically.

**Who is affected.** Anyone reporting `sp.honest_did(method='smoothness')`.
Re-run; the new numbers agree with R `HonestDiD` to ~7e-5 on width. If the
event-study covariance cannot be recovered (a result carrying no influence
functions), the old approximation is still used and now warns.
`method='relative_magnitude'` is unchanged and still approximate.

<a id="cs-rcs-reg-covariates"></a>

## Unreleased — ⚠️ `callaway_santanna(panel=False, estimator='reg', x=[...])` changed estimator

**What changed.** Repeated cross-sections with covariates previously used a
StatsPAI-specific approximation: the outcome was residualised on the covariates
using the never-treated pool with period fixed effects, and then plain 2×2
cell-mean differences were taken. R `did` instead calls
`DRDID::reg_did_rc`, which fits period- and group-specific outcome regressions
inside each (g, t) cell. StatsPAI now does the same.

**Effect.** ATT and SE both move for `panel=False` **with covariates**. In
exchange the estimator now reproduces R `did::att_gt(panel=FALSE,
est_method="reg", xformla=...)` to ~1e-11 on `did::mpdta`
(−0.0419686124 with never-treated controls).

**Who is affected.** Only `panel=False` calls that pass `x=`. Repeated
cross-sections *without* covariates are unchanged (still the cell-mean DiD, and
still equal to the unconditional panel simple ATT). Panel calls are entirely
unaffected. Re-run and use the new values; the old path was an approximation
with no reference implementation behind it.

<a id="cs-varying-base-period-e-minus-1"></a>

## Unreleased — ⚠️ `base_period='varying'` now reports the `e = −1` placebo

**What changed.** `sp.callaway_santanna`'s (g, t) grid builder skipped
`t == g − 1 − anticipation` under *every* base-period scheme. Under
`base_period='universal'` that is correct — it is the reference period and
ATT(g, g−1) is zero by construction. Under `base_period='varying'` it is not
the reference: the base for `t = g−1` is `g−2`, so ATT(g, g−1) is an estimable
pre-treatment placebo. R `did` and Stata `csdid` both report it; StatsPAI
dropped it.

**Effect.** Under `base_period='varying'` the event study gains one row at
`e = −1`. On canonical `did::mpdta` that cell is −0.024459, matching R `did`
2.3.0 and Stata `csdid` to the printed precision — and with it restored the
*entire* varying event study now agrees with both references, where previously
only the post-treatment half did.

Post-treatment coefficients do not move, under either scheme. The default
`base_period='universal'` path is completely unchanged.

**Who is affected.** Users of `base_period='varying'`, and in particular
anything that consumes the pre-period vector:

- `sp.honest_did` / `sp.sensitivity_rr` — one more pre-period enters the
  Rambachan–Roth restriction set, so breakdown values and robust CIs shift.
- `sp.pretrends_test` / `sp.pretrends_power` — the joint pre-trend test gains a
  degree of freedom.

Re-run these if you have recorded output from `base_period='varying'`. If you
need the old event-time grid for comparison, `base_period='universal'` is
unchanged, but note it is a *different* placebo estimand, not the old buggy
one. There is no flag to restore the omission — it was a bug.

<a id="aggte-group-overall-weighting"></a>

## Unreleased — ⚠️ `sp.aggte(type='group')` overall ATT is now cohort-size weighted

**What changed.** The per-cohort effects θ(g) were correct, but collapsing them
into the single reported `.estimate` used equal `1/K` weights. R
`did::aggte(type="group")` weights each cohort by its share of treated units:
`sum_g (p_g / sum_g p_g) * θ(g)`. The `cohort_sizes` series needed for this was
already being computed by `sp.callaway_santanna` and passed into the weight
builder, where it was silently ignored.

**Effect.** Only the headline scalar (`.estimate`, `.se`, `.pvalue`, `.ci`) of
`sp.aggte(type='group')` moves. The `.detail` frame — one row per cohort — is
unchanged. The size of the shift depends on how unequal the cohorts are and how
much the effect varies across them; it is exactly zero when all treated cohorts
are the same size. On a 300-unit simulated panel the overall went from
0.4315153 to 0.4317301.

**Who is affected.** Anyone who read the overall number off
`sp.aggte(type='group')`. `type='simple'`, `'dynamic'`, and `'calendar'` are
unaffected — `simple` was already cohort-share weighted, and R reports the
dynamic and calendar overalls as unweighted means across event times and
calendar periods respectively, which is what StatsPAI already did. Re-run and
use the new value; there is no flag to restore the old behavior.

<a id="aggte-analytic-se-covariance"></a>

## Unreleased — ⚠️ `sp.aggte(bstrap=False)` standard errors were ~0.64× too small

**What changed.** With `bstrap=False`, `sp.aggte` combined the per-cell
standard errors as `sqrt(Σ wₖ² seₖ²)` — the formula for *independent* cells.
ATT(g, t) cells are not independent: they are built from overlapping sets of
control units, so the omitted covariance terms are large and positive. Both the
per-cell and the overall SE now aggregate through the influence functions,
`sqrt(mean((Ψw)²)/n)`, matching R `did` and matching what this function's own
`bstrap=True` branch already used for the overall estimate.

**Effect.** SEs on the `bstrap=False` path get larger — on simulated staggered
panels the old value averaged **0.635×** the multiplier-bootstrap SE, so a
nominal 5% Wald test of a true null was rejecting about 21–23% of the time.
Confidence intervals widen accordingly and some previously "significant"
aggregations will stop being significant. Point estimates do not move.

**Who is affected.**

- Callers who passed `bstrap=False` explicitly.
- Callers whose result carried no influence-function matrix, where `aggte`
  forces `bstrap=False` internally.
- **Not** the default path: `sp.aggte` ships `bstrap=True`.
- **Not** `sp.callaway_santanna`'s own headline SE, which already aggregated
  through the influence functions and is numerically unchanged.

If you have recorded SEs or p-values from `sp.aggte(..., bstrap=False)`,
re-run them. The new values are the correct ones; there is no flag to restore
the old behavior. As before, `bstrap=True` additionally gives you the uniform
sup-t bands, which the analytic path cannot produce.

<a id="cic-athey-imbens-step2"></a>

## Unreleased — ⚠️ `sp.cic` now reproduces the Athey-Imbens estimator

**What changed.** The step-2 counterfactual in `sp.cic` had two defects: it
composed the empirical CDFs with the control-post (`y01`) and treated-pre
(`y10`) cells transposed relative to Athey & Imbens (2006) eq. 9, and it used
linearly-interpolated CDF / quantile functions on a finite τ grid instead of
the step-function ECDF and its generalized inverse. It now computes the
counterfactual map `k(y) = F_01⁻¹(F_00(y))` on the step ECDF.

**Effect.** The unconditional ATT converged ~0.5% away from the reference
(2.8% with covariates); it now matches Kranker's Stata `cic` (a direct port of
the A&I Matlab) to the printed digits — e.g. 2.999904 on the test fixture, where the old
grid-dependent code gave 3.01792 at the default `n_grid=200` (3.01388 in the
large-grid limit). Every `sp.cic` point estimate and QTE moves
slightly.

**Who is affected.** Anyone who ran `sp.cic`. If you have recorded CIC numbers
from an earlier release, re-run them; the new values are the correct A&I
estimates. There is no flag to restore the old behavior — it was a bug.

<a id="panel-hdfe-multiway-cluster-nul"></a>

## Unreleased — ⚠️ Panel HDFE multiway cluster SEs no longer collapse

**What changed.** `sp.hdfe_ols` / `sp.feols`' native N-way cluster sandwich
formed its intersection clusters by joining the dimension labels with a `"\0"`
separator, but `pd.factorize` truncates object strings at an embedded NUL
byte. Every intersection therefore collapsed onto its first cluster variable,
so distinct specifications such as `cluster(prov, year)` and
`cluster(pref, year)` returned *identical* standard errors. Replaced with a
mixed-radix integer code combination, mirroring the fix already applied to the
standalone `sp.multiway_cluster_vcov` inference path in v1.17.0.

**Effect.** Two-way and higher cluster-robust SEs from the panel HDFE path
change, toward Stata `reghdfe`. One-way clustering and non-clustered SEs are
unaffected.

**Who is affected.** Anyone using `vcov={"CRV1": [a, b]}`-style multiway
clustering through `sp.hdfe_ols` / native `sp.feols`. Re-run affected models.

<a id="conley-non-psd-nan"></a>

## Unreleased — ⚠️ Conley non-PSD variances report `nan`, not `0`

**What changed.** Kernel-weighted spatial HAC is not positive semi-definite in
finite samples; with a uniform kernel `S'WS` routinely has negative diagonal
entries. Every Conley path used `sqrt(max(V, 0))`, which turned a negative
variance into `se = 0` — reported downstream as `t = ∞`, `p = 0`. Affected
terms now return `nan` with a loud `RuntimeWarning` (Stata `acreg` reports the
same terms as missing). Rounding-level negatives are still clamped to 0
silently.

**Effect.** Where the Conley covariance was non-PSD, SEs that used to read
`0.0` now read `nan`. The covariance itself is unchanged (and still matches
`acreg` to ~1e-12 where it is PSD), so any SE that was previously non-zero is
unchanged.

**Remedies** named in the warning: widen/narrow the distance cutoff, use
`kernel="bartlett"` (tapered kernels are far better behaved than the uniform
indicator), or check whether the coordinates are collinear with the absorbed
fixed effects.

<a id="event-study-pre-vcov-optin"></a>

## Unreleased — `sp.event_study` pre-period covariance (opt-in this release)

**What changed.** `sp.event_study` now computes the full cluster-robust
covariance of the event-time coefficients (always available in
`model_info['vcov']`). The pre-period submatrix `model_info['vcv_pre']` —
which `pretrends_test`, `pretrends_power`, `sensitivity_rr`, and `honest_did`
use in place of the historical diagonal (independent-pre-coefficients)
approximation — is written **only** when you pass `expose_pre_vcov=True`.

**Why opt-in for now.** Switching the default would move published honest-DiD
and pre-trend numbers during the live JOSS review. By default the diagonal
fallback still fires — but it now **warns loudly** that it is assuming the
pre-period coefficients are independent (they are not; they share the omitted
reference period and the fixed effects). The correct full covariance becomes
the default in a future release, at which point it will be logged as a flagged
⚠️ correctness fix.

**What to do.** For statistically correct honest-DiD / power today, pass
`expose_pre_vcov=True` to `sp.event_study`. To reproduce numbers from an
earlier release, do nothing — the default is unchanged.

<a id="event-study-headline-att-se"></a>

## Unreleased — ⚠️ Event-study headline ATT SE uses the full covariance

**What changed.** Three event-study estimators reported a headline ("overall
ATT") standard error that treated the post-period event-time coefficients as
independent:

- `sp.event_study`: `sqrt(mean(se²)/m)` → now `sqrt(w'Vw)` with `w = 1/m`
  over the post-period block of the cluster-robust `model_info["vcov"]`.
- `sp.design_robust_event_study`: same formula swap on its cluster-robust
  vcov (validated against a 400-draw cluster bootstrap of the full
  procedure: analytic 0.3065 vs bootstrap 0.3101; the old formula gave
  0.2414).
- `sp.cohort_anchored_event_study`: the per-event-time bootstrap loops were
  merged into one **joint** cluster bootstrap; the headline SE is now the
  bootstrap SD of the post-period average itself.

Event-time coefficients share a reference period and fixed effects, so their
covariance is large and positive; the independence approximation understated
the headline SE — by ~2× on a realistic staggered test panel.

Additionally, `sp.event_study`'s `model_info["pretrend_test"]` is now a
**cluster-robust Wald test** (`F(q, G-1)` on the same vcov as the printed
SEs) instead of a classical homoskedastic F-test, and the `or 1e-6`
fabricated-SE fallback in `design_robust` / `cohort_anchored` is gone (an
unavailable SE is `NaN` plus a warning).

**Effect.** Headline `se` / `pvalue` / `ci` change (typically wider /
less significant) for every call to these three functions; headline
`estimate` and the per-coefficient rows are unchanged. Downstream,
`sp.parallel_trends_robustness` breakdown values shift slightly (e.g.
0.504 → 0.485 on the covariance-export test fixture). `pretrend_test`
statistics/p-values change under clustering; the dict gains `df_denom`.

**Who is affected.** Anyone quoting the overall ATT inference or the inline
pre-trend test from these estimators. Re-run affected models; the new values
are the correct ones. There is no flag to restore the old behavior — it was
a bug. (This is separate from the `expose_pre_vcov` opt-in above, which
remains opt-in during the JOSS review: that flag governs what the
*downstream pre-trend tools* consume, whereas this fix governs the headline
aggregation, whose correct covariance was already computed unconditionally.)

<a id="did2x2-ddd-weighted-robust"></a>

## Unreleased — ⚠️ Weighted robust SEs in `sp.did_2x2` / `sp.ddd`

**What changed.** With analytic weights, the `robust=True` (HC1) branch
built the sandwich meat as `X'diag(w·e²)X`; the WLS score is `w·x·e`, so the
correct meat is `Σ w²e²xx'` (Stata aweight-robust / R `sandwich`
convention). The cluster branch always squared the score correctly.

**Effect.** `sp.did_2x2(..., weights=, robust=True)` and
`sp.ddd(..., weights=, robust=True)` SEs change (~9% on dispersed weights)
and now match Stata 18 MP `regress ..., [aw=w] robust` to machine precision
(pinned in `tests/reference_parity/test_did2x2_ddd_weighted_robust_parity.py`).
Point estimates, unweighted SEs, and clustered SEs are unchanged.

**Who is affected.** Only weighted + `robust=True` calls. Re-run them.

<a id="parallel-trends-robustness-inf-verdict"></a>

## Unreleased — ⚠️ `sp.parallel_trends_robustness` verdict at `Mbar* = ∞`

**What changed.** When the honest CI still excludes zero at the top of the
search range (`Mbar = 1e4`), the breakdown is `inf` — maximal robustness.
The verdict builder's `not np.isfinite(...)` guard routed that case into the
"NOT robust: the CI already includes zero at M = 0" sentence — the exact
opposite conclusion. `inf` now yields a "robust over the entire searched
range" verdict; failed (NaN) families are excluded from the binding-family
comparison and listed in an explicit note instead of silently (and
order-dependently) participating in the `min`.

**Effect.** Only the `verdict` string changes. The `breakdown` and
`ci_grid` tables were always correct.

**Who is affected.** Anyone (human or agent) who read `result.verdict` for
a large effect measured in raw units. Re-read those verdicts.

<a id="conley-duplicate-unit-time"></a>

## Unreleased — ⚠️ `sp.conley` rejects duplicated `(unit, time)` rows

**What changed.** The spatio-temporal path (`time=` + `unit=`) resolves the
cross-unit block through a single-valued `(unit, time) → row` lookup. With
more than one row per unit-period (e.g. plant-level rows with
`unit="county"`), the cross-unit terms silently kept only the last duplicate
row while the within-unit terms kept all rows — wrong SEs with no signal.
`sp.conley` now raises a `ValueError` naming the offending unit, matching
Stata `acreg`'s repeated-id-time restriction.

**Effect.** Previously-silent wrong answers become an immediate error.

**What to do.** Aggregate your data to one row per `(unit, time)`, or pass
the true row-level identifier as `unit=` if each row is its own location.

<a id="proximal-surrogate-index-bridge-2sls"></a>

## Unreleased — ⚠️ `sp.proximal_surrogate_index` bridge is now proper 2SLS

**What changed.** The linear bridge `h(s, x)` used to be read off a
second-stage regression of `Y` on `[1, W, S_hat, X]`. Because `S_hat` — the
first-stage projection of `S` on `[1, W, X]` — is an exact affine function of
those same columns, that design matrix is rank-deficient, and the reported
"bridge slope" was whatever minimum-norm split `np.linalg.lstsq` happened to
return. Concretely, the point estimate depended on the *units* of the proxy
`W`: on a fixed persistent-confounding DGP with true ATE 1.32, the estimate
was 0.49 with `W` as given, 0.008 with `W×10`, and 1.22 with `W×0.01`. The
second stage now excludes `W` (`Y ~ [1, S_hat, X]`), which solves the correct
bridge moment `E[(Y - h(S,X)) · (1, W, X)'] = 0` — classical 2SLS with the
proxies as excluded instruments. Estimates are now invariant to rescaling `W`
and recover the true long-term ATE in the linear model.

**Why.** A point estimate that changes by two orders of magnitude when a proxy
switches from dollars to cents is not an estimate of anything (§7 — numerical
correctness is the floor).

**Who is affected.** Every previous `sp.proximal_surrogate_index` call —
earlier point estimates, SEs, and CIs were unit-dependent artifacts and should
be discarded, not compared against the new output. `sp.surrogate_index` and
`sp.long_term_from_short` are untouched.

**Action.** Re-run affected analyses. Two calls that previously "worked" now
raise: fewer proxies than surrogates raises `MethodIncompatibility`
(under-identified order condition), and proxies whose first-stage projections
are collinear raise `DataInsufficient` (rank condition). Both used to return
minimum-norm artifacts silently.

---

<a id="callaway-santanna-nevertreated-no-control"></a>

## Unreleased — ⚠️ `sp.callaway_santanna` fails loudly with an empty never-treated control

**What changed.** `sp.callaway_santanna(control_group="nevertreated")` on a
panel where every unit is eventually treated (no `g=0` units) used to return a
silent `ATT = 0.0`: each `ATT(g,t)` had no comparison cell, returned `0.0`, and
those aggregated to a headline `0.0` with no warning. It now raises
`MethodIncompatibility`.

**Why.** `0.0` is a specific wrong number that reads as "no treatment effect,"
so a mis-specified control group produced a plausible-looking but meaningless
estimate instead of an error (§7 — fail loudly).

**Who is affected.** Only calls that requested `control_group="nevertreated"`
on a panel with zero never-treated units. Any panel with at least one
never-treated unit (including `NaN`/`inf`-coded, which are treated as
never-treated) is unchanged, and `control_group="notyettreated"` is unchanged.

**Action.** Use `control_group="notyettreated"` (later-treated cohorts serve as
controls), or add never-treated units to the panel. No previously-valid
estimate changes.

---

<a id="eigenvector-centrality-bipartite"></a>

## Unreleased — ⚠️ `sp.eigenvector_centrality` fixed on bipartite graphs

**What changed.** Eigenvector centrality was computed by naive power iteration
`x <- A x`. On a bipartite graph the adjacency spectrum is symmetric
(`lambda_max = -lambda_min`), so the iteration oscillates between the two
sign-partitions and never converges; after `max_iter` steps it returned a
near-uniform vector (a star scored ~`1/sqrt(n)` for every node). The leading
eigenvector is now obtained by direct eigendecomposition, so the dominant nodes
score correctly (star hub `1/sqrt(2)`, leaves `1/sqrt(8)`).

**Why.** The returned centralities were qualitatively wrong on any bipartite or
near-bipartite network — the whole point of the measure (ranking nodes by
recursive influence) was lost.

**Who is affected.** Any `sp.eigenvector_centrality` call on a bipartite or
near-bipartite graph. Non-bipartite connected graphs (where power iteration did
converge) are numerically unchanged up to normalization.

**Action.** Re-run; the new scores are the correct leading eigenvector. The
`max_iter` / `tol` arguments remain accepted for backward compatibility.

---

<a id="ges-collider-acyclicity"></a>

## Unreleased — ⚠️ `sp.ges` no longer adds a spurious collider-parent edge

**What changed.** Greedy Equivalence Search searched over edge additions with no
acyclicity constraint. On a v-structure `X -> Z <- Y` it could add an edge into
`Z` and another out of `Z`; scoring a parent of `X`/`Y` then conditioned on the
collider and made the two independent parents look dependent, so a false
`X -- Y` edge entered the graph. The search now rejects cycle-creating edges and
returns the DAG's CPDAG (v-structures directed, reversible edges undirected).

**Why.** The recovered skeleton was wrong — colliders came back fully connected
instead of `X -> Z <- Y`, contradicting the d-separation structure.

**Who is affected.** Any `sp.ges` result on data containing a collider (common).
Recovered graphs may lose spurious edges and gain correct v-structure
orientations; chains now read as undirected CPDAG edges rather than a single
arbitrary orientation.

**Action.** Re-run `sp.ges`; the new adjacency is the correct CPDAG. No API
change (`.edges()`, `.adjacency`, `.to_frame()` unchanged in shape).

---

<a id="dist-iv-binary-instrument-nan"></a>

## Unreleased — ⚠️ `sp.dist_iv` / `sp.kan_dlate` no longer NaN on binary instruments

**What changed.** The distributional-IV Wald estimator split the instrument at
`Z > median(Z)`. For a binary `Z` with more 1s than 0s the median is 1, so the
high group (`Z > 1`) was empty and `late_q` came back all-`NaN` with no error —
about half of ordinary data draws. The split now falls back to `Z >= median`
when the strict split is degenerate, so a binary instrument always separates
into its two levels; it returns NaN only when `Z` is constant.

**Why.** A silently all-NaN point estimate is a correctness failure — the
function ran to completion and returned a result object full of NaNs.

**Who is affected.** Any `sp.dist_iv` / `sp.kan_dlate` call whose instrument is
binary (or discrete with the median on the top support point). Draws that
already produced finite estimates are numerically unchanged.

**Action.** Re-run affected calls; previously-NaN quantiles now carry the
correct Wald LATE. No API change.

---

<a id="contrast-pwcompare-categorical"></a>

## Unreleased — ⚠️ `sp.contrast` / `sp.pwcompare` now fire `C(var)` factor dummies

**What changed.** `sp.contrast` and `sp.pwcompare` previously returned all-zero
contrasts (and zero SEs / p-values) when the model was fit with a
formula-encoded categorical such as `y ~ C(g) + x`. The predictive-margin
engine matched coefficient terms to raw data columns, so design terms named
`C(g)[T.1]` never responded to setting the raw `g` column to a level. The margin
builder now parses treatment-coded factor terms (`C(var)[T.level]`, including
string levels), so reference/adjacent/pairwise contrasts equal the
corresponding dummy coefficients exactly.

**Why.** All-zero contrasts are a silent correctness failure — the function ran
without error but every reported difference was wrong. The fix restores the
documented Stata `margins, contrast(...)` behaviour for factor-encoded models.

**Who is affected.** Any `sp.contrast` / `sp.pwcompare` call on a model fit with
`C(...)` factor notation. Models that coded the categorical as a plain numeric
column were already correct and are unchanged.

**Action.** Re-run affected contrasts; the new numbers are the correct ones
(they equal the treatment-dummy coefficients). No API change.

---

<a id="did-multiplegt-baseline-conditioning"></a>

## Unreleased — ⚠️ `sp.did_multiplegt` now baseline-conditions switcher/stayer cells

**What changed.** `sp.did_multiplegt` now computes DID_M, dynamic effects, and
placebo effects within each baseline-treatment cell `d_{t-1}`. Switchers are
compared only to stayers with the same baseline treatment; switch-off cells are
sign-flipped so the reported estimand is the effect of gaining treatment. The
dynamic path additionally uses robust stayers that keep the baseline treatment
unchanged through the full horizon `[t, t+h]`, and placebo effects use the Stata
`did_multiplegt (old)` mirror sign convention.

**Why.** Pooling all stayers in a period let already-treated stayers contaminate
untreated control trends and mixed switch-on / switch-off effects under one
majority sign. The static path is pinned to Stata reference values, and the
dynamic/placebo path is guarded by small hand-computable panels that isolate the
robust-stayer and placebo-sign requirements.

**Who is affected.** Any `sp.did_multiplegt` run with multiple baseline
treatment values, switch-off events, dynamic effects, or placebo effects can
change. Designs with only switch-on events and a single valid same-baseline
stayer set may be unchanged.

**Action required.** Re-run reported `did_multiplegt` estimates, especially if
they used `dynamic=` or `placebo=`. No call-site change is required; this is a
numerical correction. For release/JOSS notes, flag this as a correctness fix
that can change point estimates.

---

<a id="spatial-ml-fullinfo-se"></a>

## Unreleased — ⚠️ `sp.sar` / `sp.sdm` report full-information coefficient SEs

**What changed.** The coefficient standard errors from `sp.sar` (spatial lag)
and `sp.sdm` (spatial Durbin) now come from the inverse of the full
`(β, ρ, σ²)` maximum-likelihood information matrix — the same asymptotic
covariance `spatialreg::lagsarlm` reports — instead of the concentrated
`σ²(XᵀX)⁻¹`. The bounded `ρ`/`λ` line-search was also tightened to
`xatol=1e-10`.

**Why.** The concentrated formula treats the spatial parameter `ρ` as known,
dropping the `β`–`ρ` covariance and understating the coefficient SEs; on a
row-standardised `W` the intercept SE came out roughly half its correct value.
The full information matrix was already being formed and inverted to produce the
`ρ` SE, so the correct `Var(β)` is the leading block of that same inverse.
Module `65_spatial` now grades `sar`/`sem`/`sdm` **bit-exact** against
`spatialreg` (worst relative error 8.3e-8 on estimates, 2.0e-8 on SEs).

**Who is affected.** Any `sp.sar` / `sp.sdm` result whose reported coefficient
standard errors, t/z-statistics, p-values, or confidence intervals were used;
the intercept SE moves most. Point estimates move only at the ≲1e-5 level from
the tighter optimiser. `sp.sem` and `sp.slx` standard errors are unchanged.

**Action required.** Re-run any `sp.sar` / `sp.sdm` inference; coefficient point
estimates are substantively unchanged, but SEs (hence significance) can differ.
No call-site change is required — this is a numerical correction.

---

<a id="etwfe-cgroup-simple-att"></a>

## Unreleased — ⚠️ `sp.etwfe` now honors `cgroup` and reports the R/Stata simple ATT

**What changed.** The public `sp.etwfe` headline now matches R
`etwfe::emfx(type="simple")` and Stata `jwdid, estat simple`: a
treated-observation-weighted simple ATT over post-treatment cohort-time effects.
The default `cgroup="notyet"` now uses not-yet-treated comparisons, while
`cgroup="nevertreated"` matches R `etwfe(cgroup="never")`.

**Why.** The previous public default was labelled `cgroup="notyet"` but behaved
like a never-treated-style estimand under a different aggregation. On the
canonical `did::mpdta` panel, this produced about `-0.0385` for the default
instead of the R/Stata not-yet-treated simple ATT `-0.047709918`. The corrected
`cgroup="nevertreated"` path matches the R never-treated value
`-0.039951275`.

**Who is affected.** Any code using `sp.etwfe(...).estimate`, `se`, `pvalue`, or
CI can change. The lower-level `sp.wooldridge_did` helper keeps its historical
saturated-TWFE cohort headline. `sp.etwfe_emfx` now defaults to
`weighting="treated"`; pass `weighting="cohort"` when you need the historical
cohort-share aggregation for comparison.

**Migration.**

| Before | After |
|---|---|
| R/Stata-compatible simple ATT via `sp.etwfe(..., panel=False)` + `sp.etwfe_emfx(..., weighting="treated")` | `sp.etwfe(...)` directly for the default not-yet-treated panel estimand |
| Previous never-treated-style default comparison | `sp.etwfe(..., cgroup="nevertreated")` |
| Historical saturated-TWFE helper output | `sp.wooldridge_did(...)` |
| Historical cohort-share emfx aggregation | `sp.etwfe_emfx(fit, weighting="cohort")` |

For release/JOSS notes, flag this as a correctness fix because the public
default point estimate changes on staggered panels.

---

<a id="bch-post-lasso-iv-deprecation"></a>

## Unreleased — Deprecation: `iv.bch_post_lasso_iv` → `sp.rlasso_iv`

**What changed.** `statspai.iv.bch_post_lasso_iv` now emits a
`DeprecationWarning`. It was StatsPAI's original, from-memory reconstruction
of the Belloni–Chen–Chernozhukov–Hansen (2012) post-Lasso IV estimator and
does **not** agree numerically with R's `hdm`: on the canonical eminent-domain
application it returns ≈0.013 where `hdm::rlassoIV` returns 0.227 (~17× off),
because it uses the asymptotic penalty `λ = 2c√{2n log(2p/α)}` and selects
only instruments (no control selection).

**Why.** `sp.rlasso_iv` is a faithful, parity-tested port of `hdm::rlassoIV`
(verified to ~1e-6 against `hdm` 0.3.2, exact on eminent domain). It supports
all four selection regimes (instruments, controls, both, neither).

**Migration.**

| Before | After |
|---|---|
| `iv.bch_post_lasso_iv(y='y', endog='d', instruments=z_cols, data=df)` | `sp.rlasso_iv(y='y', d='d', z=z_cols, data=df, select_Z=True, select_X=False)` |
| `iv.bch_post_lasso_iv(..., exog=x_cols)` | `sp.rlasso_iv(..., x=x_cols, select_Z=True, select_X=True)` |

The result object differs (`RLassoIVResult` exposes `.coef` / `.se` / `.tstat`
/ `.pvalue` / `.conf_int()` / `.summary()` / `.cite()`). `bch_post_lasso_iv`
keeps its original numerics during the deprecation window; nothing about
existing call sites breaks, but new code should use `sp.rlasso_iv`. See
[`docs/guides/rigorous_lasso_hdm.md`](docs/guides/rigorous_lasso_hdm.md).

---

<a id="cusum-boundary"></a>

## 1.20.0 — ⚠️ `sp.cusum_test` used the wrong CUSUM boundary

**What changed.** The recursive-residual CUSUM test compared the CUSUM path
against a **constant** critical value (`1.358` at 5%). That constant is the
`sup|Brownian bridge|` quantile of the *OLS-CUSUM* (Ploberger–Krämer 1992), a
different test; the Brown–Durbin–Evans recursive CUSUM crosses a **linear**
boundary `a·[1 + 2 s/(n−k)]` (`a = 0.948` at 5%) that widens from `a` to `3a`
across the sample. The old constant over-rejected late breaks and
under-rejected early ones — empirically it rejected ≈32% of stable series at a
nominal 5% level (now ≈4%).

**Who is affected.** Anyone reading `cusum_test(...)["reject"]` or
`["critical_value"]`. **`critical_value` changed from a scalar to the boundary
array**; `reject` is now True iff the path crosses that boundary anywhere.

**Action required.** If you compared `max_cusum` to a hard-coded `1.358`, use
the returned `reject` instead. Point estimates / the CUSUM path are unchanged.

---

<a id="lee-imbens-manski"></a>

## 1.20.0 — ⚠️ `sp.lee_bounds` reported a Horowitz–Manski CI labelled "Imbens–Manski"

**What changed.** The confidence interval padded *both* bound endpoints by the
two-sided `z_{1−α/2}`. That is the Horowitz–Manski interval for the identified
**set**, which over-covers the partially identified **parameter**, yet it was
labelled "Imbens–Manski". It is now the genuine Imbens & Manski (2004) interval:
a critical value `C_n` solving `Φ(C_n + Δ/σ_max) − Φ(−C_n) = 1 − α` that
interpolates between the one-sided `z_{1−α}` (wide bounds) and the two-sided
`z_{1−α/2}` (point identification). Refs verified via Crossref + RePEc/IDEAS
(Econometrica 72(6):1845–1857, doi:10.1111/j.1468-0262.2004.00555.x).

**Who is affected.** Anyone reading the CI from `sp.lee_bounds`. The interval is
**narrower** (correct). Point bounds, midpoint estimate, and bound width are
unchanged.

**Action required.** None beyond noting that previously reported CIs were
conservative (too wide).

---

<a id="rd-hc-variance"></a>

## 1.20.0 — ⚠️ RD heteroskedasticity-robust standard errors were inflated

**What changed.** The local-polynomial HC ("conventional"/"robust") variance
built its sandwich *meat* with the kernel weight to the **first** power
(`Σ w_i x_i x_i' e_i²`) instead of **squared** (`Σ w_i² x_i x_i' e_i²`) as the
Calonico–Cattaneo–Titiunik (2014) variance requires. Every HC-robust RD
standard error was therefore inflated — ≈1.4× for a uniform kernel versus R
`rdrobust` `vce="hc0"`. Affects `sp.rdrobust`, `sp.rd2d`, RD heterogeneous-
effects, and `sp.rd_bias_aware_fuzzy`. (Cluster-robust RD SEs were already
correct.)

**Who is affected.** Anyone using RD HC-robust SEs / CIs / p-values. Point
estimates are **unchanged**; SEs/CIs are now **smaller** and match R `rdrobust`
to the documented HC1-vs-HC0 d.o.f. convention.

**Action required.** None beyond noting that prior HC-robust RD intervals were
conservative (too wide). Re-run if you reported their exact width.

---

<a id="cs-pretrend-f"></a>

## 1.20.0 — ⚠️ `sp.callaway_santanna` pre-trend Wald test over-rejected

**What changed.** The joint pre-trend test (`model_info["pretrend_test"]`)
referred its Wald statistic `W = θ̂'V̂⁻¹θ̂` to `χ²(k)`. Because the pre-period
ATT(g,t) are strongly correlated (shared base period and control group) and
`V̂` is estimated, the plug-in χ² over-rejected in finite samples — empirical
size ≈0.15 at a nominal 5% level for ~60 units. It now applies the Hotelling-T²
correction, referring `W·(G−k)/(k·(G−1))` to `F(k, G−k)` (`G` = number of
units), which is exact under normal influence functions and → χ²(k)/k as
`G → ∞` (empirical size ≈0.07).

**Who is affected.** Anyone reading the pre-trend test p-value. ATT point
estimates and SEs are unchanged. The new p-value is (weakly) larger (less
likely to spuriously reject parallel trends).

**Action required.** None beyond noting prior pre-trend p-values were too small.

---

<a id="gardner-es-weighting"></a>

## 1.20.0 — ⚠️ `gardner_did(event_study=True)` overall ATT was unweighted

**What changed.** In event-study mode the overall ATT was the *unweighted* mean
of the post-period coefficients. It is now the treated-observation-**weighted**
mean (the `did2s` aggregated-ATT convention), which equals the
non-event-study `gardner_did` ATT exactly. The unweighted mean disagreed with
the non-ES path under heterogeneous effects / unbalanced horizon support
(e.g. 1.63 vs the correct 1.75).

**Who is affected.** Anyone using `gardner_did(event_study=True).estimate` (the
headline overall ATT). The per-horizon event-study coefficients are unchanged;
only their aggregation into the overall ATT changed.

**Action required.** Re-run if you reported the event-study overall ATT; it now
matches the obs-weighted (non-event-study) value.

---

<a id="regress-weights"></a>

## 1.20.0 — ⚠️ `sp.regress` ignored `weights=` (silently fit unweighted OLS)

**What changed.** `sp.regress(..., weights=col)` accepted the `weights`
argument through `**kwargs` and then never used it — the returned fit was
plain unweighted OLS, with no warning. As of this fix it solves the weighted
least squares problem with Stata `aweight` semantics, so `weights=` changes the
coefficients, standard errors (classical / HC-robust / clustered), and R²
exactly as a weighted regression should. Verified against **Stata 18 MP**
`regress y x [aw=w]` (+ `, robust` / `, vce(cluster …)`) to machine precision.

**Who is affected.** Anyone who called `sp.regress(..., weights=w)`. Calls
*without* `weights=` are numerically identical (the unweighted code path is
byte-for-byte unchanged).

**Action required.** Re-run any weighted `sp.regress` fits — prior results were
the unweighted OLS solution. The new path also raises `ValueError` on
non-finite, non-positive, wrong-length, or unknown-column weights instead of
silently proceeding. This mirrors the `sp.feols` no-FE weights fix below; the
same fail-silently bug existed independently in the OLS estimator.

---

<a id="hdfe-cluster-nested-fe"></a>

## 1.20.0 — ⚠️ `sp.hdfe_ols` cluster-robust SE inflated when an absorbed FE was nested in the cluster

**What changed.** The native HDFE backend (`sp.hdfe_ols` / `sp.absorb_ols`,
**not** the pyfixest path) built the CRV1 finite-sample factor
`(N−1)/(N−K) · G/(G−1)` with `K` counting **every** absorbed fixed-effect level
(plus the regressors). When a fixed-effect dimension is fully nested in the
cluster variable — the canonical `absorb(unit + time) + cluster(unit)` case,
where each `unit` maps to exactly one cluster — the cluster-robust sandwich
already accounts for arbitrary within-cluster correlation, so counting that
FE's `(G−1)` levels again in `K` double-penalises the degrees of freedom and
inflates the standard error. The backend now detects nested dimensions (every
FE level maps to a single cluster level) and drops their levels from the cluster
DOF, matching Stata `reghdfe`, `pyfixest`, and `sp.feols`. Non-nested
dimensions (e.g. `time` under `cluster(unit)`) are charged exactly as before;
when *all* absorbed FEs are nested, one degree of freedom is retained for the
intercept.

**Who is affected.** Anyone reading clustered standard errors / t-stats /
p-values / CIs from `sp.hdfe_ols(..., cluster=…)` or `sp.absorb_ols(...,
cluster=…)` where an absorbed FE is nested in the cluster — the very common
`absorb(entity, time) + cluster(entity)` design. The inflation grew with the
ratio of absorbed FE levels to `N`: ≈5.4% on the reporter's MRE panel and ≈6.3%
on a 37,869-row firm-year panel. Point estimates and `iid` / `hetero` SEs are
unchanged; only clustered SEs change, and they get **smaller** (less
conservative), so results that were marginally non-significant may now cross
conventional thresholds.

**Action required.** Re-run any `sp.hdfe_ols` / `sp.absorb_ols` fits that
combined absorbed FEs with clustering on (or nested in) one of those FE
dimensions; the previous clustered SEs were systematically too large. The
corrected FE dof charged to CRV1 is exposed as `dof_fe_cluster`, and the
detected nested dimensions as `nested_fe` (in `cluster_info`) /
`nested_fe_in_cluster` (raw result), so you can confirm what was reclassified.
`sp.feols` (pyfixest backend) was already correct and is unchanged.

---

<a id="feols-nofe-weights"></a>

## 1.18.0 — ⚠️ `sp.feols` ignored `weights=` when no fixed effects were absorbed

**What changed.** Called with regressors but **no** fixed effects, `sp.feols`
took an intercept-only OLS fallback that accepted the `weights=` argument but
never used it — the fit was unweighted. As of 1.18.0 the fallback solves the
weighted least squares (WLS) normal equations, so `weights=` now changes the
coefficients, standard errors, and R² exactly as a weighted regression should.

**Who is affected.** Anyone who called `sp.feols(..., weights=w)` **without**
fixed effects. Calls *with* fixed effects were already weighted correctly and
are unaffected; calls without `weights=` are numerically identical.

**Action required.** Re-run any no-FE weighted `sp.feols` fits — prior results
were the unweighted OLS solution. The new path also raises on non-finite,
negative, or zero-total-mass weight vectors instead of silently proceeding.

---

<a id="evalue-hr-ci-parity"></a>

## 1.18.0 — ⚠️ `sp.evalue` HR / CI E-value parity with R `EValue`

**What changed.** Two numerical behaviours of `sp.evalue` (and
`sp.evalue_from_result`) changed so that StatsPAI now reproduces the R
`EValue` package exactly (#21):

1. **Hazard ratios.** `measure='HR'` was always treated as a *rare-outcome*
   ratio (`OR ≈ RR ≈ HR`). It now uses the exact common-outcome conversion
   `(1 − 0.5^√HR)/(1 − 0.5^√(1/HR))` by default, matching `EValue::evalues.HR`.
   HR E-values change for non-rare outcomes.
2. **Confidence intervals that cross the null.** The CI E-value is now exactly
   `1` whenever the interval already contains the null (or a user-supplied
   `true` value), instead of a spurious value > 1 computed from the limit.

**How to get the old numbers.** Pass `rare=True` for the rare-HR
approximation. There is **no** flag to restore the un-clamped CI E-value — the
old value was incorrect (it claimed "confounding needed" for a result already
compatible with the null).

**Parameter rename.** `rare_outcome` → `rare`. The old name still works (emits
`DeprecationWarning`) and will be removed no earlier than the next minor.

**Who is affected.** Anyone computing an E-value from a hazard ratio with a
non-rare outcome, or reading the CI E-value of a non-significant result. RR-
and OR-based point E-values are unchanged.

**JOSS / JSS.** This is parity module `23_evalue` in the JSS cross-language
table (`Paper-JSS/manuscript/tables/appendix_b_parity.tex`); the change
*increases* agreement with R `EValue` and the row remains a machine-precision
**PASS** (worst relative difference 5.8e-14 over 26 rows). No JOSS (#10604)
numeric figure uses an HR or CI E-value.

---

<a id="matching-nearest-tie-break"></a>

## 1.18.0 — ⚠️ `sp.match` nearest-neighbor tie-breaking stabilised

**What changed.** `sp.match(method='nearest')` now resolves exact equal-distance
nearest-neighbor ties by the source DataFrame index. Previously the
Euclidean/propensity nearest-neighbor path delegated tie selection to
`argpartition` and incidental row order, so ties on discrete or binary
covariates could move the ATT across environments. Lower-index control units
are now selected first; when matching without replacement and multiple treated
units have the same best distance, lower-index treated units are assigned first.

**Who is affected.** Only users whose matching data contain exact
equal-distance ties. Continuous covariates without exact ties are unchanged.
For tied designs, results are now deterministic across row order and backend as
long as the DataFrame index preserves unit identity. One caveat: distances that
are merely *near*-equal (differing at the ~1e-13 ULP level because the BLAS
build computes propensity scores slightly differently) are still resolved by
strict comparison, so a residual backend sensitivity of that magnitude remains
— on LaLonde it amounts to ~$4.5 (vs. ~$150 before the fix), and all GitHub CI
platforms (ubuntu/windows/macos) now agree bitwise.

**Action required.** None for code. If you previously recorded a nearest-match
estimate on tied discrete covariates, re-run it once and treat the new value as
the stable pin. The bundled LaLonde 1:1 NN PSM guard now pins the two observed
fixed points exactly (`1967.94` on GitHub CI, `1963.43` under Accelerate on
macOS 26) instead of allowing the old ~$300 cross-backend tie band.

---

<a id="blp-maxiter-fix"></a>

## 1.18.0 — ⚠️ `sp.blp` functionality fix (was non-functional)

**What changed.** `sp.blp` (BLP random-coefficients logit demand) now runs.
Previously its GMM inner loop called `_gmm_objective(..., maxiter=1000)` while
the parameter is named `maxiter_inner`, so **every** `sp.blp` call raised
`TypeError: _gmm_objective() got an unexpected keyword argument 'maxiter'` as
soon as the outer optimiser evaluated the objective — i.e. on every estimation
path (`contraction`, `mpec`, `gmm`).

**Who is affected.** Anyone who tried to call `sp.blp`. Because the function
produced *no* output before (it crashed), this fix cannot move any
previously-correct number — JOSS (#10604) / JSS dossier figures are unaffected.
`sp.blp` (and `BLPResult`) appears in the JSS manuscript only as a
function-inventory catalog row (`function_inventory_full.tex`), never in any
numeric or parity table, and the fix does not change that row. **Note the name
collision:** the "BLP" entries in the JSS parity-change log (`05-parity.tex`,
`05-parity-compact.tex`) refer to the **Best Linear Projection of CATE**
(`best_linear_projection` / `blp_test` / `test_calibration`,
Chernozhukov-Demirer et al.) — a different feature, not this
Berry-Levinsohn-Pakes demand estimator. A regression guard now recovers the
known linear price/characteristic coefficients on a pure-logit DGP
(`tests/test_tierD_structural_analytic.py::TestBLPAnalytic`).

**Action required.** None, beyond noting that `sp.blp` is now usable. Found by
the Tier D analytic special-case test campaign (CLAUDE.md §5).

<a id="dag-dseparation-fix"></a>

## 1.18.0 — ⚠️ d-separation corrected (forks & colliders)

**What changed.** `statspai.dag`'s d-separation engine (`_d_separated`, behind
`DAG.d_separated`, `adjustment_sets`, `backdoor_paths`, `do_rule1/2/3`,
`do_calculus_apply`, `swig`, `dag_recommend_estimator`) moralised the ancestral
graph incorrectly — it married *siblings* instead of *co-parents*. So the two
non-trivial d-separation rules were backwards: conditioning on a common cause
did **not** block a fork (`A ⊥ C | M` on `M→A, M→C` wrongly returned `False`),
and conditioning on a collider did **not** open it (`A ⊥ C | K` on `A→K←C`
wrongly returned `True`). Chains were unaffected. Moralisation now connects
every pair of a node's parents, so all three canonical structures and the
adjustment-set / do-calculus routines built on them are correct.

**Who is affected.** Anyone who used `DAG.d_separated`, `adjustment_sets`,
`backdoor_paths`, the do-calculus rule checkers, `swig`, or
`dag_recommend_estimator`. **Re-derive any adjustment sets / identification
conclusions** obtained from these — previous fork/collider answers were
unreliable. No API change. None of the package's reference-parity or
JOSS/JSS-dossier numbers come from these graph routines, and all nine
dag-touching test files pass unchanged (none had encoded the broken behaviour).

**Action required.** None code-wise; re-check any DAG-derived adjustment sets.
Found by the Tier D analytic special-case campaign (CLAUDE.md §5); guard:
`tests/test_tierD_p2_dag_dsep_analytic.py`.

<a id="granger-wald-variance-fix"></a>

## 1.18.0 — ⚠️ `sp.granger_causality` test statistic corrected

**What changed.** `sp.granger_causality` now computes the correct Wald
statistic. The coefficient covariance used in the test was a placeholder
`V = sigma2 * I` (the caused equation's residual variance, not its coefficient
covariance), which omitted the design-matrix factor `(X'X)⁻¹`. The reported
F-statistic was too small by a factor of roughly `T·Var(regressors)`, so the
test essentially never rejected — even a textbook-strong lagged link went
undetected (true F≈326 reported as ≈0.36). `VARResult` now stores `(X'X)⁻¹` and
the test forms `Var(β̂_caused) = σ²_caused·(X'X)⁻¹`; the F now equals the
standard restricted-vs-unrestricted OLS F-test.

**Who is affected.** Anyone who called `sp.granger_causality` (directly or via
`VARResult.granger_test`). **Re-run any Granger conclusions** — prior runs
almost certainly failed to reject and were not trustworthy. There is no API
change. No JOSS (#10604) / JSS table uses this function, and the previous output
was statistically meaningless, so no valid published result is invalidated.

**Action required.** None code-wise; re-run Granger tests and expect them to
detect real causal directions now. Found by the Tier D analytic special-case
campaign (CLAUDE.md §5); guard: `tests/test_tierD_p2_timeseries_analytic.py`.

<a id="ols-qr-kernel"></a>

## 1.18.0 — ⚠️ OLS kernel switched to a QR solve (numerical accuracy)

**What changed.** The core OLS kernel — `ols_fit` for coefficients and
`OLSEstimator.estimate` for the variance-covariance matrix, both in
`src/statspai/` — now solves least squares via the **QR factorisation** of the
design matrix `X = QR` (`b = R⁻¹Qᵀy`, `(X'X)⁻¹ = R⁻¹R⁻ᵀ`). The previous
implementation solved the normal equations `(X'X) b = X'y` and formed
`inv(X'X)` directly. Forming `X'X` squares the condition number of `X`, so on
ill-conditioned designs roughly half of the available digits are lost — and on
the worst cases the result is meaningless.

**Why.** The new NIST StRD certification suite
(`tests/numerical_accuracy/test_nist_strd_ols.py`) showed the normal-equations
path produced **0 correct digits** on the NIST Filippelli dataset (a degree-10
polynomial fit, `cond(X) ≈ 1e10`) and only ~6 digits on several Wampler
polynomials. The QR path tracks `cond(X)` rather than `cond(X)²` and lifts
those to ~7 and ~9–13 digits respectively, matching the published certified
values.

**Who is affected.**

- **Well-conditioned regressions (the overwhelming majority): no action.**
  Coefficients, standard errors, R², F all match the old output to ≈1e-12 —
  far below any reporting precision. The full `reference_parity` and
  `external_parity` (JOSS reproduction) suites pass unchanged.
- **Regressions on near-collinear or high-degree-polynomial designs:** you will
  now get **different — and correct — numbers**. If you previously fit, say, a
  high-order polynomial trend or a strongly collinear specification directly
  (without centring/orthogonalising) and recorded the coefficients, re-run and
  expect them to move. The old numbers were the unstable ones.

There is **no API change** and nothing to rewrite; this note exists only so the
numerical shift on ill-conditioned designs is on the record.

Separately, exact-fit OLS (`R² == 1`) now reports the F-statistic as `inf`
(matching NIST's certified "Infinity") instead of emitting a divide-by-zero
`RuntimeWarning`; non-exact fits are unaffected.

Separately, `sp.regress` now fits intercept models in mean-centred
(Frisch-Waugh-Lovell) coordinates and reconstructs the intercept afterward.
This is algebraically identical to the raw fit in exact arithmetic, but it
avoids catastrophic cancellation when `y` or a regressor has a very large
constant offset. Well-conditioned fits remain unchanged to machine precision;
the visible effect is on pathological offset designs such as the NIST StRD
ANOVA `SmLs07/08/09` cases, where F/R² accuracy improves down to the float64
input-representation floor.

---

<a id="regress-collinearity-guard"></a>

## 1.18.0 — ⚠️ `sp.regress` raises on perfect collinearity; `sp.logit`/`sp.probit` warn on separation

**What changed.** Two silent-failure corners now fail loudly:

- **Perfect collinearity** in `sp.regress` — duplicate or proportional
  regressors, the dummy-variable trap (complementary 0/1 dummies plus an
  intercept), or a constant non-intercept regressor — previously returned
  enormous unidentified coefficients (e.g. `~1e14`) with no warning. It now
  raises `statspai.exceptions.NumericalInstability`, with the offending columns
  in `error.diagnostics`.
- **Perfect / quasi-complete separation** in `sp.logit` / `sp.probit` —
  where the outcome is perfectly predicted and the maximum-likelihood estimate
  does not exist — previously returned large finite coefficients with no
  signal. It now emits a `statspai.exceptions.ConvergenceWarning`.

**Why.** The project rule is "fail loudly": returning wrong numbers silently is
the cheapest way to hide a correctness problem. Neither case has a meaningful
answer to return.

**What you need to do.**

- If you *intended* collinear regressors, drop one (or remove the intercept for
  a single constant regressor). The exception names them.
- For separation, use penalized (Firth) logistic regression, drop the
  separating predictor, or pool sparse categories.

**Scope / non-goals.** Collinearity detection is deliberately *structural*
(duplicate/proportional columns, zero-variance regressors), not based on the
condition number or matrix rank. A rank tolerance loose enough to catch real
collinearity also flags legitimately ill-conditioned but full-rank designs —
the NIST StRD Filippelli benchmark is numerically *more* singular
(`s_min/s_max ~ 6e-16`) than an exactly duplicated column yet must fit. So a
general exact linear dependence among 3+ columns that is not reducible to a
pairwise duplicate or a constant column is **not** auto-detected; inspect the
design's condition number if you suspect one.

---

<a id="drdid-traditional-normalisation"></a>

## 1.17.0 — ⚠️ `sp.drdid(method='trad')` ATT correctness fix

**What changed.** The traditional doubly-robust DiD branch of `sp.drdid`
(Sant'Anna & Zhao 2020) divided each of its four cell terms — treated/control ×
post/pre, each a weighted average of the outcome-regression residual — by the
**full sample size** `n` rather than by that cell's weight mass. On a balanced
2×2 each cell holds ~¼ of the sample, so every term was scaled down by its
sample share and the ATT was biased toward zero by ~50%. Concretely, on a 2×2
with true ATT 2.0 (raw DiD 1.96) `method='trad'` returned ≈1.04. Each term is
now normalised by its own weight total. The traditional estimator therefore now
reduces **exactly** to the raw 2×2 DiD when no covariates are supplied, and
recovers the true ATT with covariates.

Separately, `sp.drdid` now **raises `ValueError`** when `method` is neither
`'imp'` nor `'trad'`. Previously any other string (e.g. `'ipw'`, `'reg'`,
`'dr'`) fell through silently to the traditional branch; such calls were never
distinct estimators and now fail loudly.

**Who is affected.** Callers of `sp.drdid(..., method='trad')` (or any non-`imp`
string, which silently ran the traditional branch). The **default**
`method='imp'` (improved, locally efficient) already normalised correctly and
is **unchanged** — its point estimates, standard errors, the
source-audit / R-`DRDID` parity numbers (which pin
`drdid_imp_panel`), and every other default-path result are **not** affected.

**Action.** If you relied on `method='trad'` output, re-run; the corrected ATT
is ~2× the previously reported (downward-biased) value and now matches the raw
DiD / `method='imp'`. Replace any `method` value other than `'imp'`/`'trad'`
with one of those two.

---

<a id="multiway-cluster-intersection"></a>

## 1.17.0 — ⚠️ `sp.multiway_cluster_vcov` multiway-cluster SE correctness fix

**What changed.** `sp.multiway_cluster_vcov` forms the Cameron-Gelbach-Miller
(2011) variance by inclusion-exclusion over the clustering dimensions, which
requires an *intersection* cluster: the unique combinations of the dimensions'
levels (e.g. the distinct `(firm, year)` pairs). The intersection key was built
by joining the dimensions into one string with a `"\0"` separator, but NumPy
fixed-width unicode strips the embedded NUL byte, so `(1, 23)` and `(12, 3)`
both collapsed to `"123"`. On a 40×50 crossed-cluster DGP this merged 1733 true
intersection clusters into 1639, which inflated the `G/(G-1)` finite-sample
factor on the subtracted intersection term and biased the multiway SE by ~0.2%
(two-way) to ~0.5% (three-way) away from the canonical estimator. The
intersection key is now built collision-free via `np.unique(axis=0)` on
per-dimension integer codes. `sp.multiway_cluster_vcov` now reproduces
`sandwich::vcovCL(cluster = ~ g1 + g2 + ...)` and `sp.twoway_cluster` to machine
precision (two-way exact; three-way relative error ~4e-7).

**Who is affected.** Callers of `sp.multiway_cluster_vcov` with **two or more**
clustering dimensions, and the multiway-clustered standard errors of
`did.harvest` and `panel.feols`. One-way clustering is unaffected.
`sp.twoway_cluster` is **not** affected — it used a separate, collision-free
intersection key and already matched `sandwich::vcovCL` to machine precision.
Point estimates are unchanged; only multiway-cluster SEs/CIs/p-values move
(typically by tenths of a percent, always toward the canonical value).

**Action.** Re-run any analysis that reported `sp.multiway_cluster_vcov`-based
multiway-cluster SEs with ≥2 dimensions; the corrected SEs now agree with
`sandwich::vcovCL` (R) and Stata multiway-cluster conventions.

---

<a id="structural-break-supf-null"></a>

## 1.17.0 — ⚠️ `sp.structural_break` sup-F p-value null distribution correctness fix

**What changed.** The sup-F / Chow statistic in `sp.structural_break(...)` is a
*supremum* of the Chow F statistic over all candidate break points. Under the
null of no break it therefore follows the Andrews (1993) sup-F limiting law,
**not** the ordinary `F(k, n-2k)` distribution. The previous code referred the
maximised statistic to the F CDF (`1 - scipy.stats.f.cdf(best_f, k, n-2k)`),
which ignored the search over break points and produced p-values that were far
too small. Measured false-positive rate on Gaussian white noise at the 5%
level: **33–37%** (n ∈ {100, 200, 400}) — a roughly 7× inflation. P-values are
now drawn from the Andrews (1993) null (a q-vector Brownian-bridge functional,
sampled by a deterministic seeded simulation cached per `(q, grid, trimming)`),
which restores **nominal size (~5%)** with no material loss of power. The same
correct critical value now governs the Bai-Perron sequential `supF(l+1|l)`
stopping rule, so `method='bai-perron'` stops over-segmenting noise.

**Who is affected.** Anyone who relied on the `p_values` / `break_dates` of
`sp.structural_break` with `method` in `{'sup-f', 'chow', 'bai-perron'}`.
Previously-reported breaks (and their tiny p-values) were anti-conservative;
some "significant" breaks were spurious. The point estimate of the *location*
of the most likely break (`break_dates[0]` for sup-F) is unchanged — only its
significance and the break **count** change.

**What to do.** Re-run any structural-break tests and re-check significance
against the corrected p-values. A break that was marginal under the old
(inflated) test may no longer reject. `method='bai-perron'` may now return
fewer breaks. The result object additionally gained populated `f_stats` /
`p_values` for the Bai-Perron path (one entry per detected break, sorted by
date), where it previously returned `None`.

**Reference.** Andrews, D.W.K. (1993). "Tests for Parameter Instability and
Structural Change with Unknown Change Point." *Econometrica*, 61(4), 821-856.
doi:10.2307/2951764 (verified via Crossref, the Econometric Society, and
RePEc).

---

<a id="msm-singleperiod-iptw"></a>

## 1.17.0 — ⚠️ `sp.stabilized_weights` / `sp.msm` single-period IPTW correctness fix

**What changed.** On a **single-period (point-treatment) panel**,
`sp.stabilized_weights(...)` (and therefore `sp.msm(...)`) previously returned
stabilized weights that were all exactly `1.0`. The within-unit lagged-
treatment column is all-zero in that setting, which made the logistic
treatment-model design singular; the failure was silently caught and the
weights fell back to the marginal mean for both the numerator and denominator,
cancelling to `1.0`. The MSM then silently reduced to an unweighted,
**confounded** regression. The fix drops zero-variance columns before fitting,
so the confounders are now used and the weights are computed correctly.

**Who is affected.** Anyone who called `sp.stabilized_weights` / `sp.msm` on a
panel with **one period per unit** (point treatment). Multi-period panels —
the intended MSM use case — are **unaffected** (their weights already varied
correctly and are numerically identical before and after).

**What to do.** Re-run any single-period MSM analyses: the previous output was
equivalent to an unadjusted regression and should not be relied on. The fixed
weights match a textbook stabilized-IPTW computation to machine precision. If
the treatment model genuinely cannot be fit (e.g. perfect separation), you now
get a `RuntimeWarning` instead of a silent fallback.

---

<a id="sp-synth-default-classic"></a>

## 1.16.1 — ⚠️ `sp.synth()` default method restored to `'classic'`

**What changed.** A bare `sp.synth(...)` call (no `method=`) now runs
`method='classic'` — canonical Abadie–Diamond–Hainmueller (2010) synthetic
control with convex, non-negative, sum-to-one donor weights. The signature
default had silently drifted to `method='augmented'` (Augmented SCM,
Ben-Michael, Feller & Rothstein 2021), which deliberately allows negative
donor weights by extrapolating outside the donor convex hull. That
contradicted the documented default (`sp.synth` docstring: `method : str,
default 'classic'`), the migration-from-R mapping (`Synth::synth` is
classic), and the canonical Prop99 examples shipped in the docs.

**Who is affected.** Anyone calling `sp.synth(...)` **without** an explicit
`method=`. Estimated effects, donor weights, and synthetic trajectories
revert from ASCM to classic SCM. Every call that already passes `method=`
(including `method='augmented'` / `'ascm'`) is **unchanged**.

**What to do.** To keep Augmented SCM, pass it explicitly:

```python
res = sp.synth(df, outcome=..., unit=..., time=...,
               treated_unit=..., treatment_time=..., method='augmented')
```

Otherwise no action is needed — the default now matches the documentation and
the R `Synth` reference implementation.

Guarded by
`tests/test_synth.py::TestSyntheticControl::test_weights_non_negative`.

---

<a id="sp-causal-forest-aipw-fix"></a>

## 1.16.0+source.20260531 — ⚠️ Causal-forest ATE/ATT now doubly-robust (AIPW)

**What changed.** `CausalForest.average_treatment_effect(...)` previously
returned a plug-in average of the forest's CATE predictions. Forest
regularisation shrinks those predictions, so the plug-in mean is biased
(≈ 15 % high on a clean-overlap design) and is *not* the estimand
`grf::average_treatment_effect` reports. It now returns the doubly-robust
AIPW influence-function mean built from the forest's own cross-fitted
nuisances (`Γ_i = τ̂ + (T−ê)/(ê(1−ê))·(Y − m̂ − (T−ê)τ̂)`), with the
influence-function standard error `sd(Γ)/√n`.

**Who is affected.** Anyone reading
`cf.average_treatment_effect(...)['estimate']` or `['se']` (any
`target_sample`: `all`/`treated`/`control`/`overlap`). The plug-in
convenience methods `cf.ate()` / `cf.att()` are **unchanged**.

**What to do.** Re-run any analysis that reported a causal-forest ATE/ATT
from `average_treatment_effect`. The new estimate is closer to truth and
agrees with `grf` within combined Monte Carlo error.

```python
ate = cf.average_treatment_effect(target_sample="all")  # ['method']=='aipw'
ate_plugin = cf.ate()                                   # still available, plug-in
```

Guarded by `tests/reference_parity/test_causal_forest_aipw_recovery.py`
and `tests/reference_parity/test_grf_parity.py`.

---

## 1.16.0 — ⚠️ `sp.xtabond` Arellano-Bond GMM correctness fix

**What broke.** `sp.xtabond` (and `sp.panel(method='ab')`) used a flat,
fixed block of lagged-level instrument columns and then dropped every
row that was missing any of them — on a short panel this discards most
of the sample — and weighted with `W = (Z'Z)⁻¹`. The correct
Arellano-Bond estimator uses a **block-diagonal** GMM instrument matrix
(each available deeper lag `Y_{i,s}`, `s ≤ t-2`, is a period-specific
moment; missing lags are zero-filled, no rows dropped) and the one-step
weight `W = (Σᵢ Zᵢ'H Zᵢ)⁻¹`, with `H` the first-difference MA(1)
structure (2 on the diagonal, −1 on the first off-diagonals). The old
code returned `β_{y₋₁}=0.264 (se 0.224)` where Stata returns
`0.391 (se 0.046)` — a 48 % estimate gap and an 80 % SE gap.

**Who is affected.** Anyone who called `sp.xtabond(...)` or
`sp.panel(..., method='ab'|'system')` on an earlier release. **Both the
point estimates and the standard errors change** — point estimates are
*not* preserved here (unlike the qreg fix).

**What to do.**

| Surface | Pre-fix | Action |
| --- | --- | --- |
| `res.estimate`, `detail["coefficient"]` | biased (instrument set wrong) | Rerun |
| `res.se`, `detail["se"]`, `res.ci`, `res.pvalue` | wrong | Rerun |
| `gmm_lags` default | `(2, 5)` | now `(2, None)` = all deeper lags (Stata default); pass an explicit max to cap |
| `method='system'` | returned a number | now raises `NotImplementedError`; use `method='difference'` |
| `twostep=True` SEs | uncorrected | now Windmeijer (2005)-corrected when `robust=True` |

**Verification.** One-step robust `sp.xtabond` now matches Stata
`xtabond y x, lags(1) vce(robust)` to machine precision on the parity
DGP (`tests/r_parity/50_xtabond`, rel ≈ 1e-15 on both β and SE);
guarded by `tests/test_gmm.py::TestArellanoBond::test_parity_matches_stata_xtabond`.

---

<a id="sp-qreg-se-fix"></a>

## 1.16.0 — ⚠️ `sp.qreg` Powell sandwich SE correctness fix

**What broke.** The Powell (1991) kernel sandwich for quantile
regression standard errors was implemented with an extra factor of
`n` in the denominator: `V = τ(1−τ) / (n · f̂(0)²) · (X'X)⁻¹`. The
textbook formula (Koenker 2005, eq. 3.7) is
`V = τ(1−τ) / f̂(0)² · (X'X)⁻¹` — no `n`. The reported SE was
therefore the correct SE divided by √n. On the parity dataset with
n = 500 (`tests/r_parity/40_qreg`), the bug under-reported SE by
~20× and produced z-statistics in the 6–30 range for null
covariates.

**Who is affected.** Anyone who used the `se`, `pvalue`, `ci`, or
`z` columns of `sp.qreg(...).detail` (or the top-level `res.se` /
`res.pvalue` / `res.ci`) on an earlier release. Point estimates
(`res.estimate`, `detail["coefficient"]`) are **unchanged at machine
precision** and do not need to be rerun.

**What to do.** Pull the patch, then rerun any analysis that
referenced an `sp.qreg` standard error. Concretely:

| Surface | Pre-fix value | Action |
| --- | --- | --- |
| `res.se`                                       | SE / √n   | Multiply by √n to recover, or just rerun |
| `res.pvalue`                                   | ~0        | Rerun — most pre-fix p-values were spuriously zero |
| `res.ci`                                       | too narrow | Rerun |
| `res.detail["se" / "z" / "pvalue"]`            | as above  | Rerun |
| `res.estimate`, `res.detail["coefficient"]`    | correct   | No change needed |

**Verification.** The cross-language parity table in
`tests/r_parity/results/parity_table_3way.md` for module `40_qreg`
shows the post-fix SE matching `quantreg::rq` (Powell `nid` kernel)
within 1.4–6.8 % and Stata `qreg` (Koenker-Bassett) within 2.9 %.
This is the expected residual gap between three different
implementations of the same sandwich.

**Why was it not caught earlier.** No 3-way Stata parity test
existed for quantile regression before the 2026-05-28 session, and
the unit tests in `tests/test_quantile.py` checked only point
estimates and that SEs were finite — never against an external
reference value.

---

<a id="sp-rdrobust-bwselect-cct-r-parity-opt-in"></a>

## v1.15.2 → v1.15.3 — doc-only PyPI hero-banner fix

**No code changes, no migration step.** The v1.15.2 PyPI project page
rendered the hero banner as a broken image because the `<img>` tag in
`README.md` / `README_CN.md` used a repo-relative path
(`docs/logo/readme-1.png`) that PyPI's long-description renderer
cannot resolve. v1.15.3 swaps the path for the absolute raw GitHub
URL so the banner loads on PyPI / TestPyPI / off-GitHub mirrors.
Module hashes match v1.15.2 bit-for-bit; only the long-description
metadata baked into the wheel + sdist changes.

---

## v1.15.1 → v1.15.2 — strict-JSON MCP wire, dual-track replicate, packaging

**No estimator numerical path changes.** Three classes of consumers
should take note:

- **`sp.agent.mcp_server` clients** (Claude Desktop / Codex / any
  RFC 8259-strict JSON parser). v1.15.1 could leak the non-standard
  literals `NaN` / `Infinity` / `-Infinity` into responses whenever an
  estimator surfaced a degenerate float (`np.nan` standard errors on a
  singular covariate, `inf` log-likelihood on a saturated model, etc).
  v1.15.2 walks all containers before `json.dumps` and serialises with
  `allow_nan=False`, replacing those values with `null`. **Action**:
  none — strict parsers that previously failed now succeed; lenient
  parsers see `null` where they used to see `NaN`. Update your
  downstream JSON Schema if it explicitly typed those fields as
  `number` (they should be `["number", "null"]`).

- **`sp.causal_text` users.** The MVP relied on a soft import of
  `sentence-transformers`. v1.15.2 adds an explicit
  `pip install statspai[text]` extra. The lazy import path is
  preserved, but the `ImportError` message now points at the extra
  instead of suggesting a bare `pip install sentence-transformers`.

- **`sp.replicate` users.** Entries for Card (1995), Abadie-Diamond-
  Hainmueller (2010), Lalonde (1986) / DW (1999), and Lee (2008) now
  return classic + modern recipes computed on the bundled real CSVs
  instead of single-track simulated stubs. If you were pinning to the
  v1.15.1 simulated numbers in CI, switch to the published-paper
  benchmarks now exposed via `df.attrs['paper_original']` (see
  `sp.datasets.nsw_lalonde(simulated=False)` and
  `sp.datasets.lee_2008_senate(simulated=False)`).

Existing `sp.rdrobust` / `sp.nbreg` / `sp.xtnbreg` / `sp.menbreg`
call sites carry over unchanged from v1.15.1.

---

## v1.15.0 → v1.15.1 — `sp.rdrobust(bwselect='cct')` R-parity opt-in

**No breaking change.** `sp.rdrobust` keeps `bwselect='mserd'` (StatsPAI's
own MSE-optimal recipe) as the default — every existing call returns the
same numbers. A new opt-in value `bwselect='cct'` is added for users who
need bit-equal R `rdrobust::rdrobust` parity.

`sp.nbreg`, `sp.xtnbreg`, and `sp.menbreg` also get clearer README /
release-note documentation in v1.15.1. Their call signatures and
numerical paths are unchanged, so there is no migration step for
negative-binomial regression users.

### When to switch from `'mserd'` to `'cct'`

Use `bwselect='cct'` when **any** of these apply:

- You're replicating a CCT 2014 / Cattaneo-Idrobo-Titiunik (2018, 2020)
  paper and need the published numbers to the 4th decimal.
- A reviewer asks for "the same number R `rdrobust` gives".
- Your data has features that stress StatsPAI's internal pilot bandwidth
  (heavy tails, small `n`, mass points). On the canonical Lee/CCT Senate
  replication, `'mserd'` gives `Conv = 12.62 / h = 4.6` while `'cct'`
  gives `Conv = 7.41 / h = 17.75` — the latter matches R bit-equal.

Keep the default `bwselect='mserd'` when:

- You don't need exact R parity, **and**
- You don't want a soft dependency on the `rdrobust` package, **and**
- Your downstream tests / pipelines have already been calibrated against
  StatsPAI's `'mserd'` numbers.

### How to switch

```python
import statspai as sp

# Before — StatsPAI internal MSE-optimal (kept stable)
res = sp.rdrobust(data=df, y='y', x='x', c=0)
# After — R-bit-equal via official rdrobust delegation
res = sp.rdrobust(data=df, y='y', x='x', c=0, bwselect='cct')
```

Install the optional dependency once:

```bash
pip install statspai[rd-cct]   # adds rdrobust>=1.3
```

Calling `bwselect='cct'` without it raises a clear `ImportError` that
points you to the install command — no silent fallback.

### Why we didn't change `'mserd'` itself

Aligning the internal `'mserd'` to R `rdbwselect`'s recursive 3-step
recipe would shift point estimates on every dataset that exercises
StatsPAI's RD path (5+ test classes, `r_parity` scripts, downstream
docs / notebooks). The additive `'cct'` route gives anyone who wants R
parity an immediate path **and** preserves the 1.x line's numerical
stability. A future major version may flip the default.

---

## v1.11 → v1.12 — DML module hardening

`sp.dml`, `sp.dml_panel`, `sp.dml_model_averaging` keep all of their
existing call signatures (every old script imports the same way and
runs without code changes), but several internal numerical behaviours
shift on the boundaries of the input space. The full release-note
discussion lives in [`CHANGELOG.md`](CHANGELOG.md) under
`[1.12.0]`; the breaking points are summarised here.

### What can change in your numbers

| Estimator | What changed | When you'll notice |
| --- | --- | --- |
| `sp.dml(model='irm')` | `KFold` → `StratifiedKFold` (stratified by D). Empty subgroup folds were silently filled with zeros for `g(1, X)` / `g(0, X)`; they now raise `IdentificationFailure`. | Small N, imbalanced D, or small `n_folds` may give point estimates a hair different from before — folds are no longer drawn from the un-stratified KFold sequence. |
| `sp.dml(model='iivm')` | Same — `StratifiedKFold` on Z, plus empty-subgroup `IdentificationFailure`. | Small N or imbalanced Z. |
| `sp.dml(model='pliv')` | Weak-IV floor on the ML-residualised partial correlation: `1e-6 → 1e-3`. | When your instrument's first-stage corr after ML residualisation is in `[1e-6, 1e-3]`, the call now raises `RuntimeError` with a clear hint to consult `sp.weakrobust` / `sp.anderson_rubin_test`. |
| `sp.dml_model_averaging` | Default `weight_rule="inverse_risk"` → `"short_stacking"`. | Different default point estimate. To preserve the v1.11 number, pass `weight_rule="inverse_risk"` explicitly. |
| `sp.dml_model_averaging` | NaN rows in `y` / `treat` / `covariates` are now dropped instead of being passed to sklearn. | If your data had NaNs you may have been getting `RuntimeError("No candidate produced a finite estimate")` or, worse, NaN θ̂; now you'll silently lose those rows but the estimate will be finite. The dropped count is reported in `model_info["n_dropped_missing"]`. |
| `sp.dml_panel(binary_treatment=True)` | Now a deprecated no-op — the previous classifier path was incorrect. The estimator runs as `binary_treatment=False` (regressor on D̃) regardless. | Different θ̂ when you used `binary_treatment=True`; a `DeprecationWarning` fires so you see it. |

### Recovering the v1.11 default for `dml_model_averaging`

```python
# v1.11 default behaviour (inverse-MSE-weighted average of per-candidate θ̂)
result = sp.dml_model_averaging(
    df, y="y", treat="d", covariates=cov_list,
    weight_rule="inverse_risk",   # v1.12 default is "short_stacking"
)

# v1.12 default — Ahrens et al. (2025, JAE) eq. 7 short-stacking
result = sp.dml_model_averaging(
    df, y="y", treat="d", covariates=cov_list,
    # weight_rule="short_stacking" (now the default)
)
result.model_info["weights_g"]   # CLS stacking weights for E[Y|X]
result.model_info["weights_m"]   # CLS stacking weights for E[D|X]
```

### Recovering the v1.11 `dml_panel(binary_treatment=True)` semantics

There is no recovery — the v1.11 path was incorrect (classifier on
within-demeaned features but raw {0,1} labels). For DR-style ATE on
binary D in panels, prefer one of:

```python
# (a) sp.dml IRM with unit dummies as covariates
import pandas as pd
unit_dummies = pd.get_dummies(df["unit"], drop_first=True)
df_aug = pd.concat([df, unit_dummies], axis=1)
sp.dml(df_aug, y="y", treat="d",
       covariates=[*cov_list, *unit_dummies.columns.tolist()],
       model="irm")

# (b) sp.etwfe (extended TWFE for staggered binary treatment in panels)
sp.etwfe(df, yname="y", tname="t", gname="treatment_cohort",
         idname="unit", covariates=cov_list)

# (c) sp.callaway_santanna (staggered DR-DiD)
sp.callaway_santanna(df, yname="y", tname="t",
                     gname="treatment_cohort", idname="unit")
```

### New capabilities (no migration needed — purely additive)

- `sample_weight=` is now accepted on `sp.dml(model='plr' | 'irm')`,
  `sp.dml_panel`, and `sp.dml_model_averaging`. Pass a 1-D array, a
  pandas Series, or a column name. The weighted estimator uses a
  Z-estimator sandwich variance throughout. `sp.dml(model='pliv' | 'iivm')`
  raise `NotImplementedError` if a non-trivial weight is supplied.
- `random_state=` (default 42) on every `sp.dml(model=...)` call
  controls fold assignment deterministically.
- `model_info["diagnostics"]` is populated on every variant — propensity
  distribution, n clipped, subgroup-fallback counts, partial correlation,
  approximate first-stage F, etc.
- String learner aliases (already shipped in 1.11.4) still work:
  `sp.dml(..., ml_g='rf', ml_m='lasso')`.

---

## v1.11 → v1.12 — `esttab` becomes a thin facade over `regtable`

The Stata-style `esttab()` previously shipped a ~500-line
`EstimateTable` class that re-implemented the full renderer pipeline.
PR-B/5c in v1.12 collapses it to a thin facade that translates
Stata-flavoured kwargs and forwards to `sp.regtable`.

**API is unchanged**, including `eststo()` / `estclear()` global store,
`isinstance(x, EstimateTableResult)` type identity, and all
`esttab(*results, se=, t=, p=, ci=, stats=, output=, ...)` keyword
spellings. Rendered output now matches `regtable`'s book-tab style.
A `DeprecationWarning` is emitted on first use; plan to migrate to
`sp.regtable(...)` directly within the next two minor releases.

### Behaviour changes

| Old | New |
| --- | --- |
| `se=True/t=True/p=True/ci=True` exclusive flags | translated to `regtable(se_type='se' \| 't' \| 'p' \| 'ci')`. Priority `ci > p > t > se` if multiple are passed (matches legacy). |
| `output='csv'` | implemented via `result.to_dataframe().to_csv()`. |
| `output='markdown'` / `'md'` / `'tex'` aliases | unchanged, all forward to the corresponding regtable renderer. |
| `filename=` extension auto-detect | unchanged (`.tex` → latex, `.html` → html, `.md` → markdown, `.csv` → csv). |

### Side-by-side migration

```python
# Before — Stata-style stateful workflow
sp.eststo(m1, name="(1)")
sp.eststo(m2, name="(2)")
sp.esttab(stats=["N", "R2", "adj_R2"], output="latex",
          filename="table1.tex")
sp.estclear()

# After — direct regtable call (same LaTeX, no global state)
sp.regtable(
    [m1, m2],
    model_labels=["(1)", "(2)"],
    stats=["N", "R2", "adj_R2"],
    filename="table1.tex",
)
```

---

## v1.11 → v1.12 — `modelsummary` becomes a thin facade over `regtable`

The R-style `modelsummary()` previously shipped a ~700-line renderer
pipeline that re-implemented coefficient extraction, star formatting,
three-line table styling and every export format. PR-B/5b in v1.12
collapses it to a thin facade that translates R-flavoured kwargs and
forwards to `sp.regtable`.

**API is unchanged**, but rendered output now matches `regtable` (book-tab
three-line, publication-quality star legend). A `DeprecationWarning` is
emitted on first use; plan to migrate to `sp.regtable(...)` directly
within the next two minor releases.

### Behaviour changes

| Old | New |
| --- | --- |
| `stars={"*": 0.10, "**": 0.05, "***": 0.01}` | only the threshold *values* are kept; the symbol overrides are dropped (regtable's ladder is `*/**/***` by convention; use `regtable(notation='symbols')` for `†/‡/§`) |
| `se_type='brackets'` | downgraded to parens with `UserWarning`; use `show_ci=True` for `[lo, hi]` if you want brackets to convey actual information |
| `se_type='none'` | downgraded to parens with `UserWarning`; the SE row stays |
| Stat keys `nobs/r_squared/adj_r_squared/f_stat` | translated to regtable canonical (`N`/`r2`/`adj_r2`/`F`) |
| Stat keys `method`/`bandwidth`/`estimand` | silently dropped (modelsummary-only; build a custom `add_rows={}` if needed) |

`coefplot` is unchanged — independent of the table renderer.

### Side-by-side migration

```python
# Before — R-style functional API
sp.modelsummary(m1, m2, m3,
                model_names=["Base", "Mid", "Full"],
                stats=["nobs", "r_squared", "adj_r_squared"],
                output="latex")

# After — direct regtable call (same LaTeX output, full control)
sp.regtable(
    [m1, m2, m3],
    model_labels=["Base", "Mid", "Full"],
    stats=["N", "r2", "adj_r2"],
).to_latex()
```

---

## v1.11 → v1.12 — `outreg2` becomes a thin facade over `regtable`

The Stata-style `OutReg2` class and `outreg2()` function previously
shipped a bespoke 800-line renderer that re-implemented coefficient
extraction, star formatting, three-line table styling, and Excel /
Word / LaTeX export. PR-B in v1.12 collapses that to ~150 lines of
glue that translates Stata-flavoured kwargs and forwards to
`sp.regtable`.

**API is unchanged**, but rendered output now matches `regtable`'s
canonical book-tab style. The visible label changes are listed below.
A `DeprecationWarning` is emitted on first use; plan to migrate to
`sp.regtable(...)` directly within the next two minor releases.

### Label / format changes

| Legacy outreg2 output | New (regtable canonical) |
| --- | --- |
| `Variables` column header | blank (book-tab convention) |
| `R-squared` | `R²` |
| `Adj. R-squared` | `Adj. R²` |
| `Observations` | `N` |
| `F-statistic / Trees` | `F` *(bug fix: "/ Trees" only applied to causal-forest results)* |
| LaTeX missing star legend | proper `\multicolumn` legend below the rule |
| LaTeX `& None & None \\` junk row | gone *(bug fix: spurious empty ATE row)* |

### Removed parameter

| Old | New |
| --- | --- |
| `show_se=False` | no longer supported. Emits `UserWarning`; the SE row stays. Use `sp.regtable(..., se_type='t' \| 'p' \| 'ci')` directly if you need a different cell. |

### Side-by-side migration

```python
# Before — Stata-style stateful builder
o = sp.OutReg2()
o.set_title("Wage Regressions")
o.add_model(m1, "Baseline")
o.add_model(m2, "Full")
o.add_note("Robust SE in parentheses")
o.to_excel("table1.xlsx")

# After — direct regtable call (same Excel output, full control)
sp.regtable(
    [m1, m2],
    title="Wage Regressions",
    model_labels=["Baseline", "Full"],
    notes=["Robust SE in parentheses"],
).to_excel("table1.xlsx")
```

---

## Migrating from `pyreghdfe`

`pyreghdfe` (`pip install pyreghdfe`) is a Python port of Stata's
`reghdfe` maintained as a standalone package. Its scope — multi-way FE
OLS with robust / multi-way cluster SEs, singleton dropping, weighted
regression — is now a strict subset of `sp.hdfe_ols` / `sp.absorb_ols`
in StatsPAI.

### API mapping (pyreghdfe → StatsPAI)

| `pyreghdfe` | StatsPAI (`import statspai as sp`) |
| --- | --- |
| `reghdfe(data=df, y='y', x=['x'], fe=['firm','year'], cluster=['firm'])` | `sp.absorb_ols(y=df['y'].values, X=df[['x']].values, fe=df[['firm','year']], cluster=df['firm'].values, solver='lsmr')` |
| Stata-style formula via pyreghdfe is not supported | `sp.hdfe_ols("y ~ x \| firm + year", data=df, cluster="firm")` (formula interface via pyfixest backend) |
| `solver='lsmr'` / `'lsqr'` | `solver='lsmr'` / `'lsqr'` — same Krylov paths (scipy.sparse.linalg) |
| Krylov-based solvers (LSMR/LSQR) | default `solver='map'` — alternating projections + Irons-Tuck acceleration, typically faster on well-conditioned panels. LSMR/LSQR remain opt-in for pathological FE structures. |
| weighted regression | `weights=` kwarg; LSMR path uses the standard √w transformation on both the sparse design and the response |
| singleton drop | `drop_singletons=True` (default) |
| multi-way cluster SE | `cluster=[firm_arr, year_arr]` (inclusion-exclusion CGM with PSD correction) |

### What you also get

- `sp.ppmlhdfe` — Poisson pseudo-ML with HDFE (not available in `pyreghdfe`).
- Rust-accelerated mean-sweep kernel ([rust/statspai_hdfe/](rust/statspai_hdfe/)).
- Formula interface and unified result object (`summary()`, `to_latex()`, `to_excel()`).
- One-line cross-solver parity check (all three solvers exposed under the
  same API — see `tests/test_hdfe_native.py::test_demean_alt_solver_matches_map_two_way`).

### Numerical parity

Default MAP and `solver='lsmr'` / `'lsqr'` agree on identical data to
`atol=1e-6` on two-way FE OLS (with and without weights, with and
without clustering). See the cross-solver parity suite in
`tests/test_hdfe_native.py`. We do not take a runtime dependency on
`pyreghdfe`; correctness is anchored to scipy's well-established
`scipy.sparse.linalg.lsmr` / `lsqr` plus the internal MAP baseline.

### When to prefer which solver

- **Default (`solver='map'`)**: almost everything. MAP + Aitken is
  typically 2–5× faster than LSMR on canonical firm × year panels.
- **`solver='lsmr'`**: ill-conditioned / highly nested FE structures
  where MAP shows slow convergence (`converged=False`,
  `iters==maxiter`). LSMR is more robust to near-redundancy between FE
  dimensions.
- **`solver='lsqr'`**: exposed for users migrating from code that
  explicitly requested LSQR. For new work prefer LSMR, which scipy
  implements on the same interface and generally offers better
  numerical stability on sparse least-squares.

---

## v1.8.0 → v1.9.0 — Agent-native API surface (no breaking changes)

**Strictly additive release.** Twelve new agent-shaped APIs land
under ``sp.``: ``audit``, ``bib_for``, ``brief``, ``detect_design``,
``examples``, ``preflight``, ``session`` (the seven new top-level
functions), plus ``result.brief()`` / ``result.cite(format=...)``
methods, plus three MCP-server features (``statspai-mcp`` console
script, ``prompts/list``, per-function ``statspai://function/{name}``
resources). **No estimator numerical paths changed**; every
coefficient / SE / CI / p-value is byte-identical to v1.8.0. See
the v1.9.0 [CHANGELOG](CHANGELOG.md#190--agent-native-api-surface-12-modules-across-4-phases)
entry for the full surface.

### Backward-compat invariants the test suite pins

The 422 new tests include explicit regression guards on these
contracts. If your code depended on any of them, nothing changes.

- ``CausalResult.to_dict()`` with no kwargs is **byte-identical**
  to ``to_dict(detail="standard")`` — the legacy default. The new
  ``detail`` parameter is keyword-only and adds three documented
  levels (``"minimal"`` / ``"standard"`` / ``"agent"``).
- ``CausalResult.cite()`` with no kwargs still returns a BibTeX
  string. The new ``format=`` keyword adds ``"apa"`` / ``"json"``
  options without changing the default.
- ``result.for_agent()`` is now a thin alias for
  ``result.to_dict(detail="agent")`` and produces the same dict.
  Existing callers see no change; new code should prefer the
  explicit form for readability.
- ``result.to_agent_summary()`` is unchanged. Its docstring now
  cross-references ``to_dict(detail="agent")`` so future readers
  know the distinction (``to_agent_summary`` is the *nested*
  schema with a ``point`` sub-dict; ``to_dict(detail="agent")`` is
  the *flat* schema). Both round-trip through ``json.dumps``.
- ``execute_tool``'s exception envelope still carries the legacy
  ``error`` / ``tool`` / ``arguments`` / ``remediation`` fields
  unchanged. Two new fields — ``error_kind`` and ``error_payload``
  — are added **only** when the caught exception is a
  ``StatsPAIError`` subclass, so any agent that previously branched
  on ``"error_kind" in out`` to detect structured errors gets a
  clean signal.

### One subtle widening to be aware of

- ``sp.agent.execute_tool``'s default serializer now invokes
  ``r.to_dict(detail="agent")`` instead of ``r.to_dict()``. The
  result dict is a strict superset of the previous shape — every
  pre-1.9 key is still present at the same path; ``violations``,
  ``warnings``, ``next_steps``, and ``suggested_functions`` are
  added. The MCP ``tools/call`` payload is therefore ~3× larger by
  default. Agents that need the smaller form should pass
  ``detail="standard"`` (or ``"minimal"``) in the ``tools/call``
  arguments — the MCP input schema documents this.

### New entry points worth knowing about

- Agents handed unfamiliar data → ``sp.detect_design(df)``.
- Before an expensive call → ``sp.preflight(df, "did", y=..., ...)``.
- After fitting → ``result.brief()`` for dashboards,
  ``sp.audit(result)`` for the missing-evidence checklist,
  ``result.cite(format="apa")`` for prose citations.
- Reproducible RNG → ``with sp.session(seed=42): ...``.
- One-shot install for MCP clients → ``pip install statspai`` now
  exposes ``statspai-mcp`` on PATH (Claude Desktop /
  ``claude_desktop_config.json`` example in
  [agent/mcp_server.py](src/statspai/agent/mcp_server.py)).

---

## v1.6.5 → v1.6.6 — ⚠️ Heckman two-step SE correctness fix (+ HDFE solver option)

**Two-part release.** (1) Correctness fix for `sp.heckman` standard
errors — point estimates unchanged, **SE / t / p / CI change**.
(2) Additive HDFE LSMR/LSQR solver option — all HDFE MAP output is
byte-identical to v1.6.5.

### What changed numerically (Heckman two-step)

`sp.heckman(...)` previously reported an HC1-style sandwich that the
source code itself flagged as
`"Heckman SEs are complex; robust is conservative"`. This was a known
limitation, not a secret bug — but it meant reported SEs, t-stats,
p-values and CIs were off by an amount that depended on (a) how
strongly selection induced heteroskedasticity `σ²(1 − ρ² δ_i)` and
(b) how uncertain the probit first-stage estimate γ̂ was.

v1.6.6 replaces it with the textbook Heckman (1979) / Greene (2003, eq.
22-22) / Wooldridge (2010, §19.6) analytical two-step variance:

```text
V(β̂) = σ̂² (X*'X*)⁻¹ [ X*'(I − ρ̂² D_δ) X* + ρ̂² F V̂_γ F' ] (X*'X*)⁻¹
```

- `X*`: second-stage design matrix including λ̂ as its last column.
- `δ_i = λ̂_i (λ̂_i + Z_iγ̂) ≥ 0` (Mills' ratio inequality).
- `D_δ = diag(δ_i)`; `F = X*' D_δ Z` (`k × q`).
- `V̂_γ = (Z' diag(w_i) Z)⁻¹` with probit information weights
  `w_i = φ(Z_iγ̂)² / [Φ(Z_iγ̂)(1 − Φ(Z_iγ̂))]`.
- `σ̂² = RSS / n_sel + β̂_λ² · mean(δ_i)` (Greene 22-21) —
  replaces the old naive `RSS / (n_sel − k)`.
- `ρ̂² = β̂_λ² / σ̂²`.

`model_info['sigma']` / `model_info['rho']` now also use this
consistent σ̂², so downstream code reading those fields will see
slightly different numbers.

### Who is affected

- Any caller of `sp.heckman(...)` — SEs, t-stats, p-values, CIs change.
- Point estimates `β̂` **do not change** (OLS of y on [X, λ̂]
  is unaffected by the variance formula).
- Callers that pin SE values in their own test suites against a
  pre-v1.6.6 StatsPAI will need to re-baseline.

### What you should do

1. **If you cited a Heckman SE / t / p / CI from StatsPAI ≤ 1.6.5**,
   re-run and update. The direction of change depends on whether
   selection-induced heteroskedasticity (reduces SE) or
   generated-regressor uncertainty (increases SE) dominates.
2. **Cross-validation**: compare the new output against Stata
   `heckman y x, select(z) twostep` or R
   `sampleSelection::heckit(...)`. Both implement the same Heckman
   (1979) formula; agreement should be to the documented precision.
3. **If you want the old conservative HC1 sandwich** for any reason
   (e.g. replicating a legacy pipeline), there is no supported way to
   get it. The old formula was not a convention choice — it was a
   known approximation the project had not yet replaced.

### Reference formula

Same as above, with the influence-function derivation:

```text
β̂ − β = (X*'X*)⁻¹ [ X*' e − β̂_λ · X*' D_δ Z · (γ̂ − γ) ] + o_p(n^{-1/2})
```

The first term gives the heteroskedastic `X*'(I − ρ̂² D_δ) X*`
contribution; the second gives the `ρ̂² F V̂_γ F'` generated-regressor
contribution, since `∂λ / ∂γ' = −λ(λ + Zγ) Z' = −δ · Z'`.

---

## v1.6.4 → v1.6.5 — ⚠️ Standalone LIML correctness fix

**Narrow correctness follow-up to v1.6.4.** If your codebase only uses
`sp.ivreg`, `sp.iv.iv`, `sp.iv.fit`, or `sp.ivreg(method='liml')` you
are **not affected** — those paths were fixed in v1.6.4. This release
closes an orphan copy of the same bug that lived in the standalone
`sp.liml` / `sp.iv.liml` entry point.

### What changed numerically

Anything calling `sp.liml(...)` directly will see both **β̂ and SE
change** compared to ≤ v1.6.4. Two independent bugs were fixed:

1. **κ_LIML solver**: switched from the non-symmetric
   `np.linalg.eigvals(inv(A) @ B)` (which can silently return complex
   eigenvalues and a biased κ) to the proper generalized symmetric
   eigenvalue problem `scipy.linalg.eigh(S_exog, S_full)`. Point
   estimates β̂ shift to the correct κ.
2. **Sandwich meat**: the cluster / robust meat used raw `X` instead of
   the k-class transformed `AX = (I − κ M_Z) X`. Same bug family as
   v1.6.4 for 2SLS; same fix (use the influence-function regressor in
   the meat).

### Post-fix consistency checks

- `sp.liml(...)` now produces **byte-identical** output to
  `sp.ivreg(..., method='liml')`.
- β̂ agrees with `linearmodels.IVLIML` to machine precision.
- Cluster SEs differ from `linearmodels.IVLIML` by ~0.1–0.2% because
  StatsPAI uses the k-class FOC-derived meat `AX = (I − κ M_Z) X`,
  while `linearmodels` uses the 2SLS-style meat `X̂ = P_Z X`
  regardless of κ. Both estimators are asymptotically equivalent and
  coincide exactly at κ = 1 (2SLS). The convention is documented in
  the new test file `tests/reference_parity/test_liml_se_parity.py`.

### What you should do

1. **If you have published LIML results** from a version ≤ v1.6.4 via
   `sp.liml(...)`, re-run and update — the old κ could be materially
   off and the old SE was built from the wrong meat.
2. **If you want LIML and only used `sp.ivreg(method='liml')`**, no
   action needed; v1.6.4 already has the correct formula.
3. **If you pinned SE or coefficient values** against the standalone
   `sp.liml` in your test suite, re-baseline to the v1.6.5 numbers.

### Reference formula (same as v1.6.4 for the k-class meat)

```text
β̂ − β = (X' A X)⁻¹ (AX)' u ,  A = (1 − κ) I + κ P_Z
Meat (cluster):  Σ_c (Σ_{i∈c} (AX)_i u_i)(·)'
Bread         :  (X' A X)⁻¹  = (AX' X)⁻¹
```

For 2SLS (κ = 1) `AX = P_Z X = X̂`; for LIML/Fuller `AX` is the
k-class transformed regressor.

---

## v1.6.3 → v1.6.4 — ⚠️ IV SE correctness fix

**Correctness-fix release.** No API surface changes, no new functions,
no docstring renames. **Numerical output of IV cluster / robust SE
changes** — this is the whole point of the release.

### What changed numerically

`sp.iv`, `sp.ivreg`, and `sp.iv.fit(method='2sls' | 'liml' | 'fuller')`
produce different standard errors when called with `robust={'hc0',
'hc1', 'hc2', 'hc3'}` or `cluster=...`. The fix restores the textbook
Cameron–Miller (2015) / Stata `ivregress` / `linearmodels` formula —
meat uses the projected regressor `X̂ = P_W X` rather than the raw
`X = [X_exog, X_endog]`.

Concretely the sandwich is now

```text
V̂ = (X̂'X̂)⁻¹ · [ Σ_c (X̂_c' û_c)(û_c' X̂_c) ] · (X̂'X̂)⁻¹
```

for the cluster case, and analogously for HC0/HC1/HC2/HC3. Before v1.6.4
the bread used `X̂` but the meat used `X`, which is a strictly incorrect
estimator for 2SLS — it happens to coincide with the correct formula
only when the first stage is a perfect fit (never, in practice).

### Who is affected

- Any IV workflow using `robust=` or `cluster=` with 2SLS, LIML, or Fuller.
- **Not affected**: point estimates (`β̂` is algebraically unchanged by
  the projection in the meat), nonrobust default SE, `method='gmm'`,
  `method='jive'`, and `sp.iv.ujive` / `ijive` / `rjive`.

### What you should do

1. **If you have published results** citing an IV SE / t-stat / p-value
   / CI from StatsPAI ≤ 1.6.3, re-run and update. The bias in the
   reported SE can be several-fold depending on first-stage fit —
   **not a rounding issue**.
2. **If you have pinned SE values in your test suite** against an
   earlier StatsPAI version, expect a mismatch. You can verify the new
   numbers by cross-checking with `linearmodels.IV2SLS(...).fit(
   cov_type='clustered', debiased=True)` — they should now agree to
   machine precision.
3. **If you were intentionally trying to reproduce the old (wrong)
   numbers**, don't. There is no supported way to get the
   pre-v1.6.4 behaviour because it was not a convention choice — it
   was a bug.

### Reference formula

For k-class with parameter κ (2SLS → κ=1, LIML → κ=κ_LIML, Fuller →
κ_LIML − α/(n−K)):

- Bread: `(X' A X)⁻¹` with `A = (1−κ) I + κ P_W`
- Meat: uses `A X` (the k-class transformed regressor); for 2SLS
  `A X = P_W X = X̂`
- FOC: `X' A (y − X β) = 0`, so the influence function is
  `β̂ − β = (X'AX)⁻¹ (AX)' u`, and the cluster/robust variance
  plugs `(AX)_i u_i` into the moment sum.

Pre-v1.6.4 the implementation plugged `X_i u_i` instead of `(AX)_i u_i`.

---

## v1.6.2 → v1.6.3 — DiD frontier sprint

**Strictly additive** plus one docstring / label truth-up. No existing
estimator's numerical path changes.

### User-visible changes worth noting

1. **`sp.continuous_did(method='att_gt')` result labels** —
   - ``result.method`` changed from
     `"Continuous DID (Callaway et al. 2024)"` to
     `"Continuous DID (dose-bin heuristic)"`.
   - ``result.estimand`` changed from
     `"ACRT (Average Causal Response on Treated)"` to
     `"Sample-weighted mean of dose-bin 2x2 DIDs (not CGS 2024 ATT(d|g,t))"`.
   - Why: the previous labels claimed paper fidelity with CGS (2024)
     that the implementation did not deliver. Numerical output is
     unchanged. If you were parsing these strings in a pipeline, update
     the matcher.
   - If you actually want a CGS (2024)-style estimator: the new
     `method='cgs'` is an **MVP** (2-period design, OR only) with
     paper formulas flagged `[待核验]`. See
     `docs/rfc/continuous_did_cgs.md`.

2. **`sp.did_multiplegt(dynamic=H)` semantic clarification** — the
   docstring now states explicitly that this is a pair-rollup
   extension, **not** the dCDH (2024) `did_multiplegt_dyn` estimator.
   Numerical output is unchanged; if you were using `dynamic=H` and
   calling it "dCDH 2024", switch to the new `sp.did_multiplegt_dyn`
   (also MVP — see `docs/rfc/multiplegt_dyn.md`).

### New functions (no migration needed, just additive)

`sp.lp_did`, `sp.ddd_heterogeneous`, `sp.did_timevarying_covariates`,
`sp.did_multiplegt_dyn` (MVP), `sp.continuous_did(method='cgs')` (MVP).

### Bib key updates

`paper.bib` entry `dechaisemartin2022fixed` upgraded from SSRN to the
published *Econometrics Journal* 26(3):C1–C30 (2023) version. Any
downstream uses of the bib key via `[@dechaisemartin2022fixed]` are
unaffected; the expanded citation will now render to the journal
version.

---

## v1.5.x → agent-native infrastructure (Unreleased)

Pure-additive release. **No migration required** for existing code.
New agent-native surface area documented here for adopters.

### 1. Exception taxonomy (new public module)

```python
from statspai.exceptions import (
    AssumptionViolation, IdentificationFailure,
    DataInsufficient, ConvergenceFailure,
    NumericalInstability, MethodIncompatibility,
)
```

Domain errors subclass the right stdlib base (`ValueError` /
`RuntimeError`), so existing `try / except ValueError` blocks still
catch `AssumptionViolation` and `DataInsufficient`, and
`except RuntimeError` still catches `ConvergenceFailure` and
`NumericalInstability`. No call-site changes required.

New code should prefer the specific subclass + attach a
`recovery_hint`:

```python
raise AssumptionViolation(
    "Parallel trends rejected at p=0.003",
    recovery_hint="Run sp.sensitivity_rr for Rambachan-Roth honest CI.",
    diagnostics={"test": "pretrends", "pvalue": 0.003},
    alternative_functions=["sp.sensitivity_rr", "sp.callaway_santanna"],
)
```

### 2. Agent-native result methods

- `result.violations()` — structured list of assumption /
  diagnostic issues with `severity` / `recovery_hint` / `alternatives`.
- `result.to_agent_summary()` — JSON-ready structured payload.
- Complement (do not replace) existing `summary()` / `tidy()` /
  `next_steps()`.

### 3. Registry agent cards

- `sp.agent_card(name)` — full metadata including pre-conditions,
  assumptions, failure modes with recovery hints, ranked
  alternatives, typical minimum N.
- `sp.agent_cards(category=None)` — bulk export of entries that
  have at least one agent-native field populated (currently:
  `regress`, `iv`, `did`, `callaway_santanna`, `rdrobust`, `synth`).

### 4. Guide `## For Agents` blocks

Run `python scripts/sync_agent_blocks.py` after any change to a
registered spec's agent-native fields. The `--check` flag is
CI-friendly and fails non-zero on drift.

---

## v1.4.x → v1.5.0

Minor release.  Only one change requires any migration:

### `sp.mr` is now a dispatcher function, not a module alias

Before v1.5.0, `sp.mr` was a reference to the `statspai.mendelian`
submodule, and `sp.mr.mr_ivw(...)` worked as attribute access on the
module.

In v1.5.0, `sp.mr` is the new **unified dispatcher** for the MR family,
matching the pattern of `sp.synth` / `sp.decompose` / `sp.dml`:

```python
sp.mr("ivw",   beta_exposure=bx, beta_outcome=by,
       se_exposure=sx, se_outcome=sy)
sp.mr("egger", beta_exposure=bx, beta_outcome=by,
       se_exposure=sx, se_outcome=sy)
sp.mr("mvmr",  snp_associations=snp_df,
       outcome="beta_y", outcome_se="se_y",
       exposures=["beta_bmi", "beta_ldl"])
```

| Old (<= v1.4.2) | New (>= v1.5.0) |
| --- | --- |
| `sp.mr.mr_ivw(...)` | `sp.mr_ivw(...)` (already available since v0.9) or `sp.mr("ivw", ...)` |
| `sp.mr.mr_egger(...)` | `sp.mr_egger(...)` or `sp.mr("egger", ...)` |
| `sp.mr.mr_presso(...)` | `sp.mr_presso(...)` or `sp.mr("presso", ...)` |
| `sp.mr` (as module alias) | `sp.mendelian` (module access preserved under this name) |

**Rule of thumb:** if your code uses `sp.mr_*` (underscore form) it
already works unchanged in v1.5.0.  Only the uncommon
`sp.mr.<attribute>` pattern needs rewriting.

### Output numerical differences you may notice after upgrading

- `sp.mr_egger` / `sp.mendelian_randomization(..., methods=["egger"])`
  slope p-values and CIs now use `t(n − 2)` rather than `Normal`, matching
  `sp.mr_pleiotropy_egger` and R's `MendelianRandomization` package.
  Effect is invisible for `n_snps ≥ ~100`.  For very small `n_snps` (say
  5 or 6) CIs widen by ~1.6×.
- `sp.mr_presso` p-values now use the `(k + 1) / (B + 1)` MC convention,
  so they are strictly positive (floor `1 / (B + 1)`).  No change for
  non-extreme cases; fixes `-inf` propagation through `log(p)` downstream.

---

## From PyStataR to StatsPAI

`PyStataR` is deprecated. All of its functionality is now available in
[StatsPAI](https://github.com/brycewang-stanford/StatsPAI), under a
unified `sp.*` namespace.

```bash
pip install statspai
```

```python
import statspai as sp
```

## API mapping

| PyStataR | StatsPAI |
|---|---|
| `pdtab.tab1(df, 'x')` / `tab2(df, 'x', 'y')` | `sp.tab(df, 'x')` / `sp.tab(df, 'x', 'y')` |
| `pywinsor2.winsor2(df, ['x'], cuts=(1,99))` | `sp.winsor(df, ['x'], cuts=(1,99))` |
| `pywinsor2.outlier_indicator(df, ['x'])` | `sp.outlier_indicator(df, ['x'])` |
| `pyoutreg.outreg(models, 'out.xlsx')` | `sp.outreg2(models, filename='out.xlsx')` |
| `pyegen.rowmean(df, ['x1','x2'])` | `sp.rowmean(df, ['x1','x2'])` |
| `pyegen.rowtotal(df, ['x1','x2'])` | `sp.rowtotal(df, ['x1','x2'])` |
| `pyegen.rowmax/rowmin(df, [...])` | `sp.rowmax(df, [...])` / `sp.rowmin(df, [...])` |
| `pyegen.rowsd(df, [...])` | `sp.rowsd(df, [...])` |
| `pyegen.rownonmiss(df, [...])` | `sp.rowcount(df, [...])` |
| `pyegen.rank(df, 'x', by='g')` | `sp.rank(df, 'x', by='g')` |

## Why migrate

- **One package, one namespace.** `sp.*` covers everything PyStataR did,
  plus DID, RD, synthetic control, IV, matching, DML, causal forest,
  meta-learners, and more.
- **Actively maintained.** PyStataR is frozen; new features land only in
  StatsPAI.
- **Cleaner naming.** No "Stata" in the name — StatsPAI is Python-native.

## Questions

Open an issue on
[StatsPAI/issues](https://github.com/brycewang-stanford/StatsPAI/issues).
