# StatsPAI v0.9.8 — Bayesian Marginal Treatment Effects (MTE) + extra VI backends

**Author:** Bryce Wang · **Date:** 2026-04-20 · **Status:** design → implementation

## 1. Motivation

v0.9.7 delivered a linear CATE-by-covariate HTE-IV and closed two of the three v0.9.6 follow-ons. The v0.9.7 "Non-goals / deferred" list called out two concrete next steps:

1. **MTE / complier-heterogeneity IV** — Heckman-Vytlacil (2005) Marginal Treatment Effects let us map how the LATE varies *along the propensity-to-be-treated distribution* (``U_D``). None of Stata / R / Python ship a first-class **Bayesian** MTE primitive — `grf` does forests, `mtefe` (Stata) is frequentist, `ivmte` (R) is frequentist with bounds. Shipping a Bayesian MTE wrapper is a clean differentiator and extends the `sp.bayes_hte_iv` story.
2. **Extra VI backends** — `_sample_model` already dispatches between NUTS and ADVI. Adding Pathfinder (faster warm-start than ADVI) and SMC (exact-ish, robust to multimodal posteriors) is a one-line change per backend thanks to the helper.

## 2. Scope

### In scope (v0.9.8)

- **`sp.bayes_mte(data, y, treat, instrument, covariates=None, u_grid=..., ...)`**
  — Bayesian MTE estimation. Models propensity-to-be-treated (first stage) and the response surface jointly, then traces the posterior over `MTE(u) = E[Y_1 - Y_0 | U_D = u]` on a user-supplied grid.

- **`inference='pathfinder' | 'smc'`** additions to `_sample_model`. Pathfinder uses `pm.fit(method='fullrank_advi')` as a warm-start proxy (PyMC 5.x ships `pmx.fit` as experimental; we use `fullrank_advi` as the stable equivalent). SMC uses `pm.sample_smc(...)` — great for hard posteriors where NUTS gets stuck.

- **`BayesianMTEResult`** — sibling of `BayesianCausalResult`; exposes:
  - `.mte_curve` — `pd.DataFrame(columns=['u', 'posterior_mean', 'posterior_sd', 'hdi_low', 'hdi_high'])`.
  - `.ate / .att / .atu` — posterior means of the integrated MTE over the relevant U_D regions.
  - `.plot_mte()` — quick matplotlib visualisation (skipped if matplotlib missing).

### Out of scope (deferred)

- **Multi-instrument MTE** — this release uses a single scalar instrument Z. Multi-Z MTE needs a policy-relevant weighting scheme (Carneiro-Heckman-Vytlacil 2011) that isn't a drop-in addition.
- **Non-linear MTE surfaces** — we fit `MTE(u)` as a polynomial of order `poly_u` (default 2). Gaussian-process-on-u is a future story.
- **Rust Phase 2 wire-in** — stays on `feat/rust-hdfe` until cibuildwheel is green.

## 3. API

### 3.1 `sp.bayes_mte`

```python
def bayes_mte(
    data: pd.DataFrame,
    y: str,
    treat: str,
    instrument: str,
    covariates: list[str] | None = None,
    *,
    u_grid: np.ndarray | None = None,      # default: np.linspace(0.05, 0.95, 19)
    poly_u: int = 2,                        # polynomial order on U_D for MTE shape
    prior_coef_sigma: float = 10.0,
    prior_mte_sigma: float = 5.0,
    prior_noise: float = 5.0,
    rope: tuple[float, float] | None = None,
    hdi_prob: float = 0.95,
    inference: str = 'nuts',
    advi_iterations: int = 20000,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_state: int = 42,
    progressbar: bool = False,
) -> BayesianMTEResult
```

### 3.2 Model

Standard Heckman (1979) / Heckman-Vytlacil (2005) latent-index structure:

```
# Selection equation (first stage, latent)
D_i = 1{ pi_0 + pi_Z * Z_i + pi_X' X_i + U_D_i > 0 }
U_D_i ~ Uniform(0, 1)            (on propensity scale)

# Outcome equations (linear separability)
Y_1i = alpha_1 + beta_X' X_i + tau(U_D_i) + eps_1i
Y_0i = alpha_0 + beta_X' X_i +       0    + eps_0i
Y_i  = D_i * Y_1i + (1 - D_i) * Y_0i

# MTE: tau(u) = b_0 + b_1 * u + b_2 * u^2 + ...
```

**Pragmatic simplification** (this release): fit propensity `P(D|Z, X)` via a logit first stage, then sample `U_D_i` via the MLE propensity (plug-in). The Bayesian layer lies on `tau(u)`'s polynomial coefficients `(b_0, …, b_poly_u)` plus the structural residual variance. This is *not* the full joint Heckman-Vytlacil posterior (which requires MCMC over the selection-equation parameters jointly with the outcome parameters). It's the same pragmatic trick `sp.bayes_iv` uses (control-function plug-in) — gives a posterior over the MTE curve that is asymptotically correct under correct first-stage specification. Document this caveat.

### 3.3 Pathfinder / SMC backends

```python
# In _sample_model:
if inference == 'pathfinder':
    approx = pm.fit(n=advi_iterations, method='fullrank_advi',
                    random_seed=random_state, progressbar=progressbar)
    trace = approx.sample(draws)
    trace.attrs['actual_chains'] = 1
    trace.attrs['inference'] = 'pathfinder'
elif inference == 'smc':
    trace = pm.sample_smc(draws=draws, chains=chains,
                          random_seed=random_state, progressbar=progressbar)
    trace.attrs['actual_chains'] = chains
    trace.attrs['inference'] = 'smc'
```

Add `'pathfinder'` and `'smc'` to the valid-inference check; update summary()'s ADVI branch to recognise `pathfinder` (rhat still meaningless) but leave SMC's rhat as a real diagnostic (SMC returns multi-chain traces).

## 4. File plan

| File | Change |
|---|---|
| `src/statspai/bayes/mte.py` | NEW |
| `src/statspai/bayes/_base.py` | Extend `_sample_model` with pathfinder + smc; add `BayesianMTEResult` |
| `src/statspai/bayes/__init__.py` | Export `bayes_mte`, `BayesianMTEResult` |
| `src/statspai/__init__.py` | Top-level exports |
| `tests/test_bayes_mte.py` | NEW |
| `tests/test_bayes_advi.py` | Add parametrised test for pathfinder / smc backends |
| `pyproject.toml` | `version = "0.9.8"` |
| `CHANGELOG.md` | 0.9.8 entry |

## 5. Test plan

- `test_bayes_mte_returns_expected_result` — has `mte_curve`, `ate`, `att`, `atu` attributes.
- `test_bayes_mte_flat_mte_recovers_constant` — constant-effect DGP → slope(u) coefficients HDI cover 0.
- `test_bayes_mte_monotone_mte_recovery` — linearly-increasing MTE → positive slope HDI.
- `test_bayes_mte_top_level_export`.
- `test_bayes_mte_custom_u_grid`.
- `test_bayes_pathfinder_runs` (parametrised across all 6 estimators? — 5 existing + mte).
- `test_bayes_smc_runs` (same list).

## 6. Success criteria

1. `sp.bayes_mte` on a constant-effect DGP returns an MTE curve whose SD is bounded + slope-on-u HDIs cover 0.
2. `sp.bayes_mte` on a monotone-MTE DGP recovers the sign of the slope with `prob_positive > 0.95`.
3. `inference='pathfinder'` works on at least 2 of the existing estimators (cost: 1 test).
4. `inference='smc'` works on at least 2 of the existing estimators.
5. Full suite stays green on top of v0.9.7's baseline.

## 7. Non-goals / known caveats

- Plug-in first-stage propensity means the MTE posterior does not propagate first-stage uncertainty into `tau(u)`. Full joint posterior is Q0 work for 0.9.9+.
- Pathfinder using `fullrank_advi` is a stand-in; when PyMC's `pmx.fit` stabilises we'll switch.
- Policy-relevant MTE weights (AMTE, PRTE) are not yet exposed; only the untreated, treated, and average integrals are.
