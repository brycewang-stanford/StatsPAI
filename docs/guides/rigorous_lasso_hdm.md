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
regimes), `tsls`, and the data-driven `lambdaCalculation` for the
homoskedastic and heteroskedastic (X-independent) penalties.

**Not yet ported — `rlassologit`** (the *logistic* rigorous Lasso and its
`rlassologitEffect(s)`). `hdm::rlassologit` delegates the penalized fit to
`glmnet::glmnet(family="binomial")` at a single data-driven `λ`. Reproducing
it faithfully means matching glmnet's logistic-lasso solution — its
standardization, intercept handling and objective scaling differ from
scikit-learn's L1 logistic regression at a fixed `λ` — which is a separate
parity exercise. Rather than ship an unvalidated approximation (the very
failure mode this module was built to avoid), it is intentionally left out
until it can be pinned against `glmnet`. For a binary treatment under
Double-ML, use `sp.dml(model='irm', ...)` with a genuine classifier.

**X-dependent penalty simulation** (`penalty={"X.dependent.lambda": True}`)
is implemented but matches `hdm` only *in distribution* — R's
Mersenne-Twister stream is not reproduced — so it is not bit-exact. The
default (X-independent) path is.

## Regenerating the parity fixtures

The `hdm` reference numbers are deterministic (no cross-validation, no
RNG on the default penalty path), so the fixtures are a hard contract.
Regenerate only when the algorithm changes:

```bash
Rscript tests/reference_parity/_generate_rlasso.R
pytest tests/reference_parity/test_rlasso_parity.py -q
```

## References

- Belloni, A., Chen, D., Chernozhukov, V. & Hansen, C. (2012). Sparse
  Models and Methods for Optimal Instruments With an Application to
  Eminent Domain. *Econometrica*, 80(6), 2369–2429.
  doi [`10.3982/ECTA9626`](https://doi.org/10.3982/ECTA9626).
- Belloni, A., Chernozhukov, V. & Hansen, C. (2014). Inference on
  Treatment Effects after Selection among High-Dimensional Controls.
  *The Review of Economic Studies*, 81(2), 608–650.
- Chernozhukov, V., Hansen, C. & Spindler, M. (2016). hdm:
  High-Dimensional Metrics. *The R Journal*, 8(2), 185–199.
  doi [`10.32614/RJ-2016-040`](https://doi.org/10.32614/RJ-2016-040).
