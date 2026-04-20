# StatsPAI v0.9.7 — Heterogeneous-effect Bayesian IV + ADVI toggle

**Author:** Bryce Wang · **Date:** 2026-04-20 · **Status:** design → implementation

## 1. Motivation

v0.9.6 landed average-LATE Bayesian IV, fuzzy RD, and per-learner
Optuna tuning. The deferred list going into this release was:

1. Bayesian *heterogeneous-effect* IV — current `bayes_iv` reports a
   single LATE number; it can't answer "is the LATE different for
   subgroup X?".
2. Variational inference (ADVI) toggle — NUTS is the right default
   but for big-N / prototype mode users want a posterior in 30 s,
   not 5 min.
3. Bayesian bunching — queued for evaluation.

## 2. Scope

### In scope

- **`sp.bayes_hte_iv(data, y, treat, instrument, effect_modifiers, ...)`**
  — Bayesian linear IV where the LATE is linear in user-supplied
  effect modifiers:

  ```
  D = pi_0 + pi_Z' Z + pi_X' X + v           (first stage)
  Y = alpha + tau_0 * D + (tau_hte' M) * D + beta_X' X + rho * v_hat + eps
  ```

  where ``M = X_eff - mean(X_eff)`` is the centred effect-modifier
  matrix. Posterior over ``(tau_0, tau_hte)`` gives:
  - Average LATE = ``tau_0``
  - CATE slope posteriors (one per modifier)
  - Predicted CATE at user-supplied covariate values (optional)

  Returns a new `BayesianHTEIVResult` that extends
  `BayesianCausalResult` with a `.cate_slopes` DataFrame.

- **`inference='nuts' | 'advi'`** parameter on every existing
  Bayesian estimator (`bayes_did`, `bayes_rd`, `bayes_iv`,
  `bayes_fuzzy_rd`) *and* the new `bayes_hte_iv`. ADVI goes through
  `pm.fit(method='advi')` and draws `draws` samples from the
  fitted approximation. Documented as "fast and approximate" —
  R-hat is not meaningful for ADVI so we report `np.nan`.

### Out of scope (with explicit rationale)

- **Bayesian bunching** — after review, the Kleven/Saez/Chetty
  structural bunching machinery is fundamentally a macro/public-
  finance identification strategy that maps *behavioural elasticities
  from kinks/notches in a tax schedule*. Turning that into a Bayesian
  estimator requires:
  1. A structural utility / optimisation parameterisation that
     doesn't generalise across kink types;
  2. Priors on taste heterogeneity that are domain-specific and
     hard to set defaults for;
  3. A model fit that is only as interpretable as the underlying
     structural model, which defeats the "agent-native one-liner"
     design thesis.

  The frequentist `sp.bunching` already exists for users who need
  this tool. **We explicitly do not ship `sp.bayes_bunching`**; the
  pay-off per LOC is lower than every other item on the Bayesian
  roadmap. Revisit if a user files a clear use case.

- Multi-instrument interaction effects, marginal treatment effect
  (MTE) posteriors — 0.9.8+.

- Rust Phase 2 — separate branch work.

## 3. API

### 3.1 `sp.bayes_hte_iv`

```python
def bayes_hte_iv(
    data: pd.DataFrame,
    y: str,
    treat: str,
    instrument: str | Sequence[str],
    effect_modifiers: Sequence[str],    # ≥ 1 covariate with potential heterogeneity
    covariates: list[str] | None = None, # controls entering both stages
    *,
    prior_late: tuple[float, float] = (0.0, 10.0),
    prior_hte_sigma: float = 5.0,        # Normal(0, sigma) prior on slope of tau
    prior_coef_sigma: float = 10.0,
    prior_noise: float = 5.0,
    inference: str = 'nuts',             # 'nuts' | 'advi'
    advi_iterations: int = 20000,
    rope: tuple[float, float] | None = None,
    hdi_prob: float = 0.95,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_state: int = 42,
    progressbar: bool = False,
) -> BayesianHTEIVResult
```

### 3.2 `BayesianHTEIVResult`

Inherits from `BayesianCausalResult`; adds:

| field | type | notes |
|---|---|---|
| `cate_slopes` | `pd.DataFrame` | one row per effect modifier: `term, estimate (posterior mean), std_error, hdi_low, hdi_high, prob_positive` |
| `effect_modifiers` | `list[str]` | modifier names |
| `predict_cate(values: dict)` | method | predict posterior CATE at specific modifier values; returns `{mean, median, sd, hdi_low, hdi_high, prob_positive}` |

### 3.3 `inference='advi'`

On every Bayesian estimator (5 total after v0.9.7):

```python
if inference == 'advi':
    approx = pm.fit(n=advi_iterations, method='advi',
                    random_seed=random_state, progressbar=progressbar)
    trace = approx.sample(draws)
    # r-hat is meaningless for ADVI; set to nan
elif inference == 'nuts':
    trace = pm.sample(...)
else:
    raise ValueError(...)
```

Documented caveat: ADVI is a **mean-field** approximation (each
parameter is treated as independent Gaussian), so correlated
posteriors look tighter than they should. Users who care about
calibration should use NUTS.

## 4. File plan

| File | Change |
|---|---|
| `src/statspai/bayes/hte_iv.py` | NEW |
| `src/statspai/bayes/_base.py` | `BayesianHTEIVResult` subclass; `_sample_model(inference=...)` helper used by all estimators |
| `src/statspai/bayes/did.py` / `rd.py` / `iv.py` / `fuzzy_rd.py` | Add `inference` parameter; route through `_sample_model` |
| `src/statspai/bayes/__init__.py` | Export `bayes_hte_iv`, `BayesianHTEIVResult` |
| `src/statspai/__init__.py` | Top-level exports |
| `tests/test_bayes_hte_iv.py` | NEW |
| `tests/test_bayes_advi.py` | NEW — ADVI toggle on each estimator |
| `pyproject.toml` | `version = "0.9.7"` |
| `CHANGELOG.md` | 0.9.7 entry |

## 5. Test plan

### `test_bayes_hte_iv.py` (≥ 7 tests)

1. `test_bayes_hte_iv_imports_cleanly`.
2. `test_bayes_hte_iv_returns_extended_result` — has `cate_slopes`.
3. `test_bayes_hte_iv_average_late_recovered_on_hetero_dgp`.
4. `test_bayes_hte_iv_slopes_recovered_on_hetero_dgp`.
5. `test_bayes_hte_iv_no_heterogeneity_dgp_slopes_near_zero`.
6. `test_bayes_hte_iv_predict_cate_returns_dict`.
7. `test_bayes_hte_iv_missing_modifier_raises`.

### `test_bayes_advi.py` (≥ 5 tests)

1. `test_bayes_did_advi_runs`.
2. `test_bayes_rd_advi_runs`.
3. `test_bayes_iv_advi_runs`.
4. `test_bayes_fuzzy_rd_advi_runs`.
5. `test_bayes_invalid_inference_raises`.

## 6. Success criteria

1. `sp.bayes_hte_iv` on a DGP with true `tau(X) = 1 + 0.5 * X` recovers both `tau_0` (HDI covers 1) and the slope on `X` (HDI covers 0.5).
2. On a DGP with `tau(X) ≡ 1` (no heterogeneity) the slope HDI straddles 0 with `prob_positive ≈ 0.5`.
3. `inference='advi'` produces a usable `BayesianCausalResult` on each of the 5 estimators; R-hat is reported as `NaN`; posterior mean on the primary estimand is within 2× NUTS posterior SD.
4. Full suite stays green (no regressions on top of v0.9.6's 1911 baseline).
