# Rigorous (data-driven) Lasso — the `hdm` port

`statspai.rlasso` is a faithful Python port of the R `hdm` package
(Chernozhukov, Hansen & Spindler, *The R Journal* 8(2), 2016): rigorous
Lasso with a data-driven, theory-justified penalty (no cross-validation),
post-double-selection treatment-effect inference, and optimal-instrument
IV. Every surface is pinned to `hdm` numerically — see
[the rigorous-Lasso guide](../guides/rigorous_lasso_hdm.md) for the full
`hdm` ↔ StatsPAI map and parity tables.

## Entry points

```python
import statspai as sp

# 1. Rigorous (post-)Lasso — hdm::rlasso
fit = sp.rlasso(X, y, post=True)          # data-driven penalty, post-Lasso refit
fit.n_selected                            # size of the selected support
fit.beta, fit.lambda0, fit.sigma          # coefficients, penalty level, residual scale
fit.predict(X)                            # fitted / out-of-sample predictions

# 2. Effect of d after selecting controls — hdm::rlassoEffect(s)
res = sp.rlasso_effect(X, y, d, method="partialling out")   # or "double selection"
res.alpha, res.se, res.pvalue, res.conf_int()
out = sp.rlasso_effects(X, y, index=[0, 1])                 # several targets at once

# 3. IV with rigorous selection — hdm::rlassoIV
res = sp.rlasso_iv(y="y", d="d", z=z_cols, x=x_cols, data=df,
                   select_Z=True, select_X=False)
res.coef, res.se

# 4. Logistic rigorous Lasso — hdm::rlassologit
fit = sp.rlassologit(X, y, post=True)     # binary y
fit.predict(X, type="response")           # probabilities ("link" = log-odds)
```

## The data-driven penalty

Unlike a cross-validated Lasso, the penalty is fixed by theory, which is
what gives the post-selection estimators their inference guarantees:

$$\lambda_0 = 2\,c\,\sqrt{n}\,\Phi^{-1}\!\Big(1-\tfrac{\gamma}{2p}\Big),
\qquad c = 1.1,\quad \gamma = 0.1/\log n,$$

with per-coefficient, heteroskedasticity-robust loadings refined by
iteration. Tune via the `penalty` / `control` dicts, mirroring `hdm`:

```python
sp.rlasso(X, y, penalty={"c": 1.1, "gamma": 0.05, "homoscedastic": False})
sp.rlasso(X, y, post=False)               # plain Lasso (c defaults to 0.5)
```

## As a Double-ML nuisance learner

The rigorous Lasso is the theory-correct sparse plug-in for DML (the rate
conditions are stated for a plug-in, not a CV-tuned, penalty). It wires
directly into `sp.dml`:

```python
r = sp.dml(df, y="y", treat="d", covariates=x_cols,
           model="plr", ml_g="rlasso", ml_m="rlasso", n_folds=5)
```

`sp.RlassoRegressor` / `sp.RlassologitClassifier` are scikit-learn-compatible
and clone-safe across cross-fitting folds.

## Numerical parity with `hdm`

Pinned against `hdm` 0.3.2 in `tests/reference_parity/test_rlasso_parity.py`,
`test_rlassologit_parity.py`, and `test_rlasso_vignette_parity.py` (29 tests):

| Surface | Agreement with `hdm` |
| --- | --- |
| `rlasso` coefficients / `λ₀` / loadings / residuals | ~1e-13; **selected support exact** |
| `rlasso_effect` / `rlasso_effects` (α, SE) | ~1e-14 |
| `rlasso_iv` eminent domain, `select_Z` (BCH 2012) | `coef 0.2274`, `SE 0.2466` (~1e-9) |
| `rlassologit` selected support / `post` refit | support exact; `post` ~1e-9 |
| `hdm` vignette: Growth, AJR, cps2012 | published worked-example coefficients pinned to `atol=1e-6` |

## References

- Chernozhukov, V., Hansen, C. & Spindler, M. (2016). hdm:
  High-Dimensional Metrics. *The R Journal*, 8(2), 185–199.
  doi [`10.32614/RJ-2016-040`](https://doi.org/10.32614/RJ-2016-040).
- Belloni, A., Chen, D., Chernozhukov, V. & Hansen, C. (2012). Sparse
  Models and Methods for Optimal Instruments With an Application to
  Eminent Domain. *Econometrica*, 80(6), 2369–2429.
  doi [`10.3982/ECTA9626`](https://doi.org/10.3982/ECTA9626).
- Belloni, A., Chernozhukov, V. & Hansen, C. (2014). Inference on
  Treatment Effects after Selection among High-Dimensional Controls.
  *The Review of Economic Studies*, 81(2), 608–650.
  doi [`10.1093/restud/rdt044`](https://doi.org/10.1093/restud/rdt044).
