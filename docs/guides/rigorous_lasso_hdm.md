# Rigorous (data-driven) Lasso — `sp.rlasso` and the `hdm` port

`sp.rlasso` is StatsPAI's **faithful Python port of R's
[`hdm`](https://cran.r-project.org/package=hdm) package** (Chernozhukov,
Hansen & Spindler, *The R Journal* 8(2), 2016, doi
[`10.32614/RJ-2016-040`](https://doi.org/10.32614/RJ-2016-040)). It
implements the rigorous Lasso of Belloni, Chernozhukov & Hansen (2014)
and the optimal-instrument IV of Belloni, Chen, Chernozhukov & Hansen
(2012, *Econometrica*, doi
[`10.3982/ECTA9626`](https://doi.org/10.3982/ECTA9626)).

The defining property versus a vanilla `sklearn` Lasso is the
**data-driven, theory-justified penalty** — no cross-validation:

$$\lambda_0 = 2\,c\,\sqrt{n}\,\Phi^{-1}\!\Big(1-\tfrac{\gamma}{2p}\Big),
\qquad c = 1.1,\quad \gamma = 0.1/\log n,$$

with per-coefficient, heteroskedasticity-robust penalty *loadings*
$\Psi_j$ refined by iteration, and an optional **post-Lasso** OLS refit
on the selected support to remove shrinkage bias.

## Why a faithful port (and not a reconstruction)

An earlier from-memory reconstruction (`iv.bch_post_lasso_iv`) used the
*asymptotic* penalty $\lambda = 2c\sqrt{2n\log(2p/\alpha)}$ and selected
only instruments. On the canonical BCH eminent-domain application it came
out **~17× off** `hdm` (0.013 vs 0.227). The lesson: aligning a
selection method numerically requires porting the reference algorithm
exactly, not approximating its formulas. `sp.rlasso` does that, and the
[parity tests](https://github.com/brycewang-stanford/StatsPAI/blob/main/tests/reference_parity/test_rlasso_parity.py)
pin every surface against `hdm` 0.3.2:

| Surface | Agreement with `hdm` |
| --- | --- |
| `rlasso` coefficients / `λ₀` / loadings / residuals | machine precision (~1e-13) |
| `rlasso` selected support | **exact** |
| `rlasso_effect` (partialling-out & double-selection) α, SE | ~1e-14 |
| `rlasso_iv` (all 4 selection regimes, well-conditioned design) | ~1e-14 |
| `rlasso_iv` on eminent domain (rank-deficient controls) | ~1e-9 |

> **Implementation note.** hdm's IV routines invert near-singular control
> blocks with `MASS::ginv`, whose singular-value cutoff is
> $\sqrt{\varepsilon}\approx 1.49\times10^{-8}$. numpy's `pinv` defaults
> to a far tighter `rcond = 1e-15`, which keeps spurious tiny singular
> values and silently produces a *different* pseudo-inverse. Matching the
> MASS tolerance is exactly what restores eminent-domain parity.

## `hdm` ↔ StatsPAI function map

The complete public surface of `hdm`, and where each piece lives in
StatsPAI. Two categories: **faithful ports** (bit-for-bit against `hdm`,
no cross-validation, deterministic fixtures) and **same-score
equivalents** (the high-dimensional treatment-effect estimators, which
StatsPAI exposes through `sp.dml`'s Neyman-orthogonal scores with
cross-fitting (Chernozhukov et al., *Econometrics Journal* 2018) rather
than `hdm`'s full-sample post-Lasso — same estimand and score family, not
claimed bit-for-bit).

| `hdm` | StatsPAI | Status |
| --- | --- | --- |
| `rlasso` | `sp.rlasso` | **faithful port** — coeffs/`λ₀`/loadings/residuals ~1e-13, support **exact** |
| `rlassologit` | `sp.rlassologit` | **faithful port** — support **exact**, glmnet engine ~1e-6, `post=TRUE` refit ~1e-9 |
| `rlassoEffect` | `sp.rlasso_effect` | **faithful port** — α, SE ~1e-14 (partialling-out & double-selection) |
| `rlassoEffects` | `sp.rlasso_effects` | **faithful port** — ~1e-14 (multi-target) |
| `rlassoIV`, `rlassoIVselectX`, `rlassoIVselectZ` | `sp.rlasso_iv(select_X=…, select_Z=…)` | **faithful port** — ~1e-14 well-conditioned, ~1e-9 rank-deficient |
| `tsls` | `sp.rlasso_iv(select_Z=False, select_X=False)` | robust 2SLS path (no selection) |
| `lambdaCalculation` | `sp.rlasso(penalty=…)` (internal) | homoskedastic & heteroskedastic (X-independent) bit-exact; X-dependent in-distribution only |
| `rlassoATE` | `sp.dml(model='irm', score='ATE', ml_g='rlasso', ml_m='rlassologit')` | same doubly-robust (AIPW) score family; cross-fitted, DoubleML-aligned |
| `rlassoATET` | `sp.dml(model='irm', score='ATTE', …)` | same score family; cross-fitted |
| `rlassoLATE` | `sp.dml(model='iivm', ml_g='rlasso', ml_m='rlassologit')` | same LATE Wald-ratio score family; cross-fitted |
| `rlassoLATET` | `sp.dml(model='iivm', …)` | covered by the IIVM score; a LATET-specific variant is not separately exposed |
| `rlassologitEffect`, `rlassologitEffects` | `sp.rlassologit_effect` / `sp.rlassologit_effects` | **faithful port** — coef & post SE ~1e-14, multi-target ~1e-15 |
| `print` / `summary` / `confint` / `predict` | `.summary()` / `.conf_int()` / `.predict()` | result-object methods |
| dataset `EminentDomain` | parity fixture in `tests/reference_parity/` | used to pin `rlasso_iv` |
| datasets `pension`, `GrowthData`, `AJR`, `cps2012`, `BLP` | not yet bundled as `sp.datasets` | on the roadmap (canonical-case reproduction) |

> **On the treatment-effect family.** `hdm::rlassoATE/ATET/LATE/LATET`
> estimate the same program-evaluation parameters as `sp.dml`'s IRM and
> IIVM models; StatsPAI routes them through the cross-fitted DML scores so
> the high-dimensional and generic-ML paths share one validated
> implementation, rather than maintaining a second full-sample estimator.
> The `sp.dml` path is parity-tested against `DoubleML` (Bach et al.,
> *JMLR* 2022 for Python; *Journal of Statistical Software* 2024 for R).

## The three entry points

### 1. `sp.rlasso` — rigorous (post-)Lasso

```python
import numpy as np, statspai as sp

rng = np.random.default_rng(0)
n, p = 200, 100
X = rng.normal(size=(n, p))
beta = np.zeros(p); beta[:4] = [3, -2, 1.5, 1]
y = X @ beta + rng.normal(size=n)

fit = sp.rlasso(X, y, post=True)        # post-Lasso (default)
fit.n_selected                          # -> 4 (recovers the support)
fit.beta[:4]                            # -> ~[3, -2, 1.5, 1]
fit.lambda0, fit.sigma                  # data-driven penalty, residual scale
fit.predict(X)                          # fitted / out-of-sample predictions
print(fit.summary())
```

Tuning is via the `penalty` and `control` dicts, mirroring `hdm`:

```python
sp.rlasso(X, y, penalty={"c": 1.1, "gamma": 0.05,
                         "homoscedastic": False})   # default: heteroskedastic
sp.rlasso(X, y, post=False)                          # plain Lasso (c defaults to 0.5)
```

### 2. `sp.rlasso_effect` — effect of `d` after selecting controls

The high-dimensional analogue of a partial regression coefficient —
root-`n` consistent and asymptotically normal under approximate
sparsity (Belloni–Chernozhukov–Hansen, 2014).

```python
res = sp.rlasso_effect(X, y, d, method="partialling out")   # or "double selection"
res.alpha, res.se, res.pvalue, res.conf_int()
```

### 3. `sp.rlasso_iv` — IV with rigorous selection

A faithful port of `hdm::rlassoIV`. Selection can run on the instruments
`z`, the controls `x`, both, or neither:

```python
# BCH (2012) optimal-instrument IV — select among many instruments
res = sp.rlasso_iv(y="y", d="d", z=z_cols, x=x_cols, data=df,
                   select_Z=True, select_X=False)
res.coef, res.se        # eminent domain logGDP -> 0.2274, 0.2466
```

| `select_Z` | `select_X` | method (`hdm` equivalent) |
| --- | --- | --- |
| `True` | `False` | `rlassoIVselectZ` — instrument selection (BCH 2012) |
| `False` | `True` | `rlassoIVselectX` — partial out high-dim controls |
| `True` | `True` (default) | double selection on Z and X |
| `False` | `False` | plain robust 2SLS (`tsls`) |

## Panels: `cluster=` (the cluster-Lasso)

`hdm` assumes independent observations. In a panel the scores
`x_it * e_it` are correlated within unit, the heteroskedastic loadings
are too small, and the Lasso keeps variables it should not. Belloni,
Chernozhukov, Hansen and Kozbur (2016) replace the loading by
`sqrt(sum_i (sum_t x_itj e_it)^2 / (nT))`. `sp.rlasso`,
`sp.rlasso_effect`, `sp.rlasso_effects` and `sp.rlasso_iv` take
`cluster=` for this. The final standard error is then cluster-robust.

```python
import statspai as sp

# Fixed effects: within-transform first, then no intercept is needed.
cols = ["y", "d"] + controls + instruments
w = df[cols] - df.groupby("unit")[cols].transform("mean")
w["unit"] = df["unit"]

# Effect of d on y, controls selected by double selection
res = sp.rlasso_effect(controls, "y", "d", data=w,
                       method="double selection", cluster="unit")

# Many instruments and many controls
iv = sp.rlasso_iv("y", "d", instruments, controls, data=w,
                  cluster="unit", intercept=False)
```

What it buys, in a design with a known answer (100 units, 8 periods, 60
candidate controls, AR(1) regressors and errors with coefficient 0.8,
1,167 draws): without `cluster=` double selection keeps 6.1 controls on
average and its 95% interval covers 78.6% of the time; with it, 4.6
controls and 93.3%.

This is the estimator of Stata's `rlasso, cluster()`, `pdslasso,
cluster()` and `ivlasso, cluster()` (lassopack). On the same data the
penalty level, the selected variables, the estimates and the variances
agree to 1e-9 (`tests/reference_parity/test_cluster_lasso_stata.py`).
Three conventions to know when comparing:

- `gamma` defaults to `0.1 / log(G)`, `G` the number of clusters.
- The variance has no small-sample factor, as in `pdslasso`. Multiply by
  `G / (G - 1)` for the `regress, cluster()` convention.
- The loadings depend on the residuals, and the residuals on what was
  selected, so the iteration can have more than one fixed point.
  StatsPAI follows hdm's iteration and lassopack its own; on small
  panels they sometimes stop at supports that differ in one marginal
  variable. Both are valid solutions of the same problem.

## As a Double-ML nuisance learner

The rigorous Lasso is the *theory-correct* sparse nuisance learner for
Double/Debiased ML (the DML rate conditions are stated for a plug-in,
not a cross-validated, penalty). It plugs straight into `sp.dml`:

```python
res = sp.dml(data=df, y="y", treat="d", covariates=x_cols,
             model="plr", ml_g="rlasso", ml_m="rlasso", n_folds=5)
```

`sp.RlassoRegressor` / `sp.RlassoClassifier` are scikit-learn-compatible
(`get_params` / `set_params` / clone-safe), so they survive the
cross-fitting `clone()` on every fold. The classifier is a *linear
probability* propensity (clipped to `(ε, 1−ε)`) — convenient, but prefer
a calibrated classifier when propensity calibration matters.

This path is pinned against R: with a shared fold partition (pass
`fold_indices=`), `sp.dml(model='plr', ml_g='rlasso', ml_m='rlasso')`
reproduces a manual Double-ML PLR estimator whose nuisances are
`hdm::rlasso` to **machine precision** (θ̂ and SE; see
`test_dml_rlasso_learner_matches_r_doubleml`). So the rigorous-Lasso DML
path is validated end to end, not just learner-by-learner.

## Relationship to `iv.bch_post_lasso_iv`

`iv.bch_post_lasso_iv` is the original reconstruction and is **deprecated**
(it emits a `DeprecationWarning`); it is kept only for backward
compatibility with its existing numerics. For agreement with `hdm` — and
for selection on both instruments and controls — use `sp.rlasso_iv`. The
two are not numerically interchangeable.

## What is and isn't ported

Ported and parity-tested against `hdm`: `rlasso`, `rlassoEffect` /
`rlassoEffects` (single and multi-target), `rlassoIV` (all four selection
regimes), `tsls`, `rlassologit` (the logistic rigorous Lasso), and the
data-driven `lambdaCalculation` for the homoskedastic and heteroskedastic
(X-independent) penalties.

### Not yet ported

Not yet ported as separate `hdm`-named wrappers: `rlassoLATET` is covered
through the IIVM score in `sp.dml(...)`, not exposed as its own
`sp.rlasso_latet`; exact R RNG parity for
`penalty={"X.dependent.lambda": True}` is also not claimed because StatsPAI
does not reproduce R's Mersenne-Twister stream. These are documented scope
boundaries rather than silent approximations.

### `sp.rlassologit` — the logistic rigorous Lasso

`hdm::rlassologit` is the binary-outcome analogue of `rlasso`: its
penalized fit is `glmnet(family="binomial", alpha=1, lambda=λ,
standardize=TRUE)` at a single data-driven `λ`. StatsPAI reproduces
glmnet's binomial lasso at that `λ` **directly** — an IRLS outer loop, a
weighted coordinate-descent inner loop, the `1/n`-scaled deviance
objective, population-variance standardization and glmnet's `pmin`
probability clamp — rather than substituting scikit-learn's L1 logistic
(whose objective/standardization differ at a fixed `λ`).

```python
fit = sp.rlassologit(X, y, post=True)   # y binary
fit.predict(X, type="response")          # probabilities  (or "link" = log-odds)
sp.RlassologitClassifier()               # genuine logistic propensity for sp.dml
```

Parity (vs `hdm` 0.3.2 / `glmnet` 4.1): the **selected support matches
exactly**; the glmnet engine's coefficients match to ~1e-6 (glmnet's own
convergence tolerance — no tighter ground truth exists); and `post=True`
(the default) coefficients/intercept/residuals — coming from an
*unpenalized* logistic refit on the selected set — match to ~1e-9.

`sp.RlassologitClassifier` is the principled binary nuisance learner for
Double-ML: `sp.dml(model='irm', ml_m='rlassologit')` uses a *calibrated*
logistic propensity (unlike the linear-probability `RlassoClassifier`).

### `sp.rlassologit_effect` / `rlassologit_effects` — logistic treatment effects

The high-dimensional *logistic treatment effect* (Belloni–Chernozhukov–Wei
2016 double-selection for GLMs: a `√σ²`-weighted union of a logistic and a
linear rigorous-Lasso selection, refit by a low-dimensional logit, with a
max-of-two sandwich variance) is now a **faithful port of
`hdm::rlassologitEffect(s)`**:

```python
res = sp.rlassologit_effect(X, y, d)          # binary y, treatment d, controls X
res.alpha, res.se, res.pvalue, res.conf_int()
out = sp.rlassologit_effects(X, y, index=[0, 1])   # each X column as the target
```

Parity vs `hdm` 0.3.2 (`tests/reference_parity/test_rlassologit_effect_parity.py`):
the coefficient and the post-Lasso standard error match to ~1e-14, the
multi-target effects to ~1e-15, and the non-post SE to ~1e-7 (the non-post
LassoShooting tolerance). For a plain binary-treatment causal effect you can
still use `sp.dml(model='irm', ml_m='rlassologit')`.

**X-dependent penalty simulation** (`penalty={"X.dependent.lambda": True}`)
is implemented but matches `hdm` only *in distribution* — R's
Mersenne-Twister stream is not reproduced — so it is not bit-exact. The
default (X-independent) path is.

## Reproducing the `hdm` vignette

The three headline worked examples in the `hdm` vignette reproduce
**exactly** in StatsPAI — same data, same numbers:

| `hdm` vignette example | call | StatsPAI = `hdm` |
| --- | --- | --- |
| **Growth** — conditional convergence (Barro-Lee `GrowthData`) | `sp.rlasso_effect(X, y, d)` | coef **−0.04981** (SE 0.01394) partialling-out; **−0.05001** (SE 0.01579) double-selection |
| **AJR** — institutions → GDP, settler-mortality IV (`AJR`) | `sp.rlasso_iv(y, d, z, x, select_X=True, select_Z=False)` | coef **0.84503** (SE 0.26993) |
| **cps2012** — gender wage gap (`cps2012`) | `sp.rlasso_effects(X, y, index=female_cols)` | female coef **−0.15492** (SE 0.05016); all 16 female targets match `hdm` |

```python
import statspai as sp, pandas as pd

# Growth: effect of (log) initial GDP on growth, selecting ~60 controls
g = pd.read_csv("hdm_growth_data.csv")
y, d = g["Outcome"].to_numpy(), g["gdpsh465"].to_numpy()
X = g.drop(columns=["Outcome", "intercept", "gdpsh465"]).to_numpy()
sp.rlasso_effect(X, y, d, method="partialling out").alpha   # -> -0.04981

# AJR: expropriation risk on GDP, instrumented by settler mortality
a = pd.read_csv("hdm_ajr_data.csv")
y, d, z = a["GDP"].to_numpy(), a["Exprop"].to_numpy(), a["logMort"].to_numpy()
Xa = a.drop(columns=["GDP", "Exprop", "logMort"]).to_numpy()
sp.rlasso_iv(y, d, z, x=Xa, select_Z=False, select_X=True).coef  # -> 0.84503
```

All three are pinned to `hdm` in
[`tests/reference_parity/test_rlasso_vignette_parity.py`](https://github.com/brycewang-stanford/StatsPAI/blob/main/tests/reference_parity/test_rlasso_vignette_parity.py).
The datasets are public, published economic facts (AJR: Acemoglu,
Johnson & Robinson 2001, *AER* 91(5); Growth: Barro & Lee; cps2012: U.S.
CPS 2012). Growth and AJR are bundled in full; for cps2012 the committed
fixture is a deterministic 800-row subsample (the full 27 MB expanded
design is not bundled) that pins the female main effect, while the
full-sample number above is reproduced and recorded at generation time.

## Regenerating the parity fixtures

The `hdm` reference numbers are deterministic (no cross-validation, no
RNG on the default penalty path), so the fixtures are a hard contract.
Regenerate only when the algorithm changes:

```bash
Rscript tests/reference_parity/_generate_rlasso.R           # core / effect / IV / logit
Rscript tests/reference_parity/_generate_rlasso_vignette.R  # Growth + AJR vignette
pytest tests/reference_parity/test_rlasso_parity.py tests/reference_parity/test_rlasso_vignette_parity.py -q
```

## References

- Belloni, A., Chen, D., Chernozhukov, V. & Hansen, C. (2012). Sparse
  Models and Methods for Optimal Instruments With an Application to
  Eminent Domain. *Econometrica*, 80(6), 2369–2429.
  doi [`10.3982/ECTA9626`](https://doi.org/10.3982/ECTA9626).
- Belloni, A., Chernozhukov, V., Hansen, C. & Kozbur, D. (2016).
  Inference in High-Dimensional Panel Models With an Application to Gun
  Control. *Journal of Business & Economic Statistics*, 34(4), 590–605.
  doi [`10.1080/07350015.2015.1102733`](https://doi.org/10.1080/07350015.2015.1102733).
- Belloni, A., Chernozhukov, V. & Hansen, C. (2014). Inference on
  Treatment Effects after Selection among High-Dimensional Controls.
  *The Review of Economic Studies*, 81(2), 608–650.
- Chernozhukov, V., Hansen, C. & Spindler, M. (2016). hdm:
  High-Dimensional Metrics. *The R Journal*, 8(2), 185–199.
  doi [`10.32614/RJ-2016-040`](https://doi.org/10.32614/RJ-2016-040).
