# StatsPAI v0.9.5 — Bayesian causal + Optuna-tuned auto_cate + Rust HDFE spike

**Author:** Bryce Wang · **Date:** 2026-04-20 · **Status:** design → implementation

## 1. Motivation

The v0.9.4 retrospective left three items flagged as "not yet done":

1. **Bayesian causal** — `sp.bayes_did` / `sp.bayes_rd` existed only as aspiration. Stata has `bayes:` prefix, R has `brms` / `bayesDID` / `RStanARM`. Python's causal stack has no unified Bayesian DID/RD primitive; `PyMC-Econometrics` is a docs project, not a batteries-included package.
2. **Optuna-tuned auto_cate** — v0.9.4 ships the learner race but relies on sklearn GBM defaults for nuisances. `econml` pipelines normally wrap Optuna/Ray Tune around the nuisance models; our CATE pipeline needs that same escape hatch.
3. **Rust HDFE kernel** — Numba gives 3× over NumPy but loses to fixest's OpenMP C++ on >1M-row panels. A Rust + Rayon port is the long-term answer.

This spec shapes **v0.9.5** around #1 and #2, and converts #3 into a time-boxed *spike* (design + benchmark harness, no build-system change) that the next release can pick up without a cold start.

## 2. Scope

### In scope (v0.9.5)

- **`statspai/bayes/` module**
  - `sp.bayes_did(data, y, treat, post, unit=None, covariates=None, ...)` — 2x2 and panel DID via PyMC with hierarchical shrinkage on unit effects.
  - `sp.bayes_rd(data, y, running, cutoff, ...)` — sharp RD with local polynomial + normal prior.
  - `BayesianCausalResult` dataclass with posterior summary, HDI, R-hat, ESS, `.tidy()` / `.glance()` delegations.
  - PyMC is an **optional dependency**. If missing, each function raises a clear `ImportError` at call time, never at `import statspai`.

- **`sp.auto_cate_tuned(...)` (optuna-backed tuner)**
  - Wraps `sp.auto_cate` — first tunes the nuisance GBM hyperparameters via Optuna's `TPESampler` against held-out R-loss, then runs `auto_cate` with the best nuisance.
  - Optuna is optional; missing → informative `ImportError`.
  - `n_trials=25`, `timeout=None` by default (users override).

- **Top-level exports**: `sp.bayes_did`, `sp.bayes_rd`, `sp.BayesianCausalResult`, `sp.auto_cate_tuned`.

- **Tests**: ≥ 6 for bayes_did (2x2 + panel + priors + failed-convergence warning + PyMC-missing guard), ≥ 4 for bayes_rd (sharp recovery + cutoff orientation + bandwidth + PyMC-missing guard), ≥ 4 for auto_cate_tuned (API + Optuna-missing guard + custom search space + beats-vanilla on noisy DGP smoke).

### Out of scope → deferred to 1.0 (with spike artefacts shipped now)

- **Rust HDFE port**. The v0.9.5 release will ship:
  - `docs/superpowers/specs/2026-04-20-v095-rust-hdfe-spike.md` — a 2–3 page design covering the maturin layout, PyO3 FFI surface, Rayon task graph, cibuildwheel matrix, and a phased rollout plan.
  - `statspai/fast/bench.py` — a pure-Python benchmark harness that times the existing NumPy, Numba-JIT, and (future) Rust kernels on the same DGPs. Lets the next session produce apples-to-apples Speed comparisons without re-instrumenting.
  - **No Rust code or `maturin` build step in v0.9.5.** Adding a Rust build to `pip install` requires cross-platform wheel testing we can't clear in this release window. A half-built Rust dep would be the single worst thing we could ship — it silently raises install failures on Windows/Linux-musl.

## 3. API design

### 3.1 `sp.bayes_did`

```python
def bayes_did(
    data,
    y: str,
    treat: str,              # 1 if ever-treated, 0 if never
    post: str,               # 1 if post-treatment period
    unit: str | None = None, # if provided, hierarchical random effect on unit
    time: str | None = None, # if provided, hierarchical random effect on time
    covariates: list[str] | None = None,
    *,
    prior_ate: tuple[float, float] = (0.0, 10.0),     # Normal(mu, sigma)
    prior_unit_sigma: float = 5.0,                     # HalfNormal
    prior_time_sigma: float = 5.0,                     # HalfNormal
    prior_noise: float = 5.0,                          # HalfNormal on sigma_eps
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_state: int = 42,
    progressbar: bool = False,
) -> BayesianCausalResult
```

Model (panel case):

```
y[i,t] = alpha_i + gamma_t + tau * did_{i,t} + x_{i,t}' * beta + eps_{i,t}
alpha_i ~ Normal(0, sigma_unit)
gamma_t ~ Normal(0, sigma_time)
tau ~ Normal(mu_ate, sigma_ate)
eps ~ Normal(0, sigma_eps)
sigma_unit, sigma_time, sigma_eps ~ HalfNormal(.)
beta ~ Normal(0, 10)
```

Returns posterior over `tau`; exposes HDI, posterior probability `P(tau > 0)`, R-hat, ESS. If `unit`/`time` are not supplied the 2x2 model is used (no random effects).

### 3.2 `sp.bayes_rd`

```python
def bayes_rd(
    data,
    y: str,
    running: str,
    cutoff: float = 0.0,
    bandwidth: float | None = None,  # if None, use rule-of-thumb: 0.5 * std(running)
    poly: int = 1,                    # 1 = local linear (Stata-default), 2 = local quadratic
    *,
    prior_tau: tuple[float, float] = (0.0, 10.0),
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_state: int = 42,
    progressbar: bool = False,
) -> BayesianCausalResult
```

Model:

```
y = a0 + a1 * (x - c) + ... + a_p * (x - c)^p
  + tau * treated
  + b1 * treated * (x - c) + ... + b_p * treated * (x - c)^p
  + eps
tau ~ Normal(mu_tau, sigma_tau)
a, b ~ Normal(0, 10)
eps ~ Normal(0, sigma_eps);  sigma_eps ~ HalfNormal(5)
```

Only observations within `[cutoff - bw, cutoff + bw]` enter the model (sharp RD).

### 3.3 `BayesianCausalResult`

Dataclass (loose sibling of `CausalResult`):

| field | type | notes |
|---|---|---|
| `method` | `str` | e.g. `"Bayesian DID (panel)"` |
| `estimand` | `str` | `'ATT'` / `'LATE'` |
| `posterior_mean` | `float` | point summary |
| `posterior_median` | `float` | |
| `posterior_sd` | `float` | |
| `hdi_lower`, `hdi_upper` | `float` | 95 % HDI |
| `prob_positive` | `float` | `P(tau > 0 | data)` |
| `prob_rope` | `float | None` | `P(|tau| < rope)` if user supplied `rope` |
| `rhat` | `float` | convergence diagnostic; warn if >1.01 |
| `ess` | `float` | effective sample size |
| `n_obs` | `int` | |
| `trace` | `az.InferenceData` | full trace for downstream plotting |
| `.tidy()` | `DataFrame` | single row: term, estimate, std_error, hdi_low, hdi_high |
| `.glance()` | `DataFrame` | method, nobs, rhat, ess, chains, draws |
| `.summary()` | `str` | printable header + tidy + glance + convergence warning |

### 3.4 `sp.auto_cate_tuned`

```python
def auto_cate_tuned(
    data, y, treat, covariates,
    learners=('s', 't', 'x', 'r', 'dr'),
    n_trials: int = 25,
    timeout: float | None = None,
    search_space: dict | None = None,   # user override
    n_folds: int = 5,
    random_state: int = 42,
    alpha: float = 0.05,
    verbose: bool = False,
) -> AutoCATEResult
```

Default search space (over nuisance GBMs):

```python
{
    'outcome_n_estimators': [100, 200, 400, 800],
    'outcome_max_depth': [2, 3, 4, 5, 6],
    'outcome_learning_rate': [0.01, 0.03, 0.05, 0.1],
    'outcome_subsample': [0.6, 0.8, 1.0],
    'propensity_n_estimators': [100, 200, 400],
    'propensity_max_depth': [2, 3, 4, 5],
    'propensity_learning_rate': [0.03, 0.05, 0.1],
}
```

Objective: shared-nuisance **R-loss** on `n_folds` held-out splits with a shared seed. Optuna's `TPESampler` minimises this. Resulting `GradientBoostingRegressor` / `GradientBoostingClassifier` are handed to `auto_cate(...)` as `outcome_model=` / `propensity_model=`, and the final `AutoCATEResult` is returned with `model_info['tuned_params']` and `model_info['n_trials']` for reproducibility.

## 4. File plan

| File | Change |
|---|---|
| `src/statspai/bayes/__init__.py` | NEW — exports `bayes_did`, `bayes_rd`, `BayesianCausalResult` |
| `src/statspai/bayes/_base.py` | NEW — `BayesianCausalResult` + shared PyMC import guard + HDI / R-hat helpers |
| `src/statspai/bayes/did.py` | NEW — `bayes_did` implementation |
| `src/statspai/bayes/rd.py` | NEW — `bayes_rd` implementation |
| `src/statspai/metalearners/auto_cate_tuned.py` | NEW — `auto_cate_tuned` Optuna wrapper |
| `src/statspai/metalearners/__init__.py` | export `auto_cate_tuned` |
| `src/statspai/fast/__init__.py` | NEW |
| `src/statspai/fast/bench.py` | NEW — benchmark harness for HDFE kernels |
| `src/statspai/__init__.py` | top-level exports |
| `tests/test_bayes_did.py` | NEW |
| `tests/test_bayes_rd.py` | NEW |
| `tests/test_auto_cate_tuned.py` | NEW |
| `pyproject.toml` | `version = "0.9.5"`, add `pymc`, `arviz`, `optuna` to `[project.optional-dependencies]` |
| `CHANGELOG.md` | 0.9.5 entry |
| `docs/superpowers/specs/2026-04-20-v095-rust-hdfe-spike.md` | NEW — Rust spike design doc |

## 5. Test plan

### 5.1 `test_bayes_did.py`

1. `test_bayes_did_imports_cleanly` — just importing the module must not trigger a PyMC dependency scan.
2. `test_bayes_did_missing_pymc_raises_informative` — monkey-patch module to simulate missing PyMC.
3. `test_bayes_did_2x2_recovers_true_att` — 2x2 DID with true ATT = 1.5; 95% HDI covers 1.5.
4. `test_bayes_did_panel_recovers_true_att_with_unit_fe` — unit random effects, staggered DGP with known ATT.
5. `test_bayes_did_prob_positive_calibrated` — when true ATT is clearly > 0, `prob_positive` ≥ 0.99.
6. `test_bayes_did_tidy_glance_work` — `.tidy()` returns one-row DF with expected columns; `.glance()` reports rhat, ess.
7. `test_bayes_did_low_draws_raises_convergence_warning` — stress-test rhat warning path.

### 5.2 `test_bayes_rd.py`

1. `test_bayes_rd_imports_cleanly`.
2. `test_bayes_rd_missing_pymc_raises_informative`.
3. `test_bayes_rd_sharp_recovery` — sharp RD with true τ = 2, 95% HDI covers 2.
4. `test_bayes_rd_no_effect_has_zero_hdi` — null DGP, HDI straddles 0, `prob_positive` ≈ 0.5.
5. `test_bayes_rd_bandwidth_respected` — narrower bandwidth ⇒ fewer observations in model.
6. `test_bayes_rd_poly2_still_runs` — quadratic polynomial fits without error.

### 5.3 `test_auto_cate_tuned.py`

1. `test_auto_cate_tuned_missing_optuna_raises_informative`.
2. `test_auto_cate_tuned_api` — returns AutoCATEResult with `tuned_params` in model_info.
3. `test_auto_cate_tuned_small_n_trials_smoke` — n_trials=5, returns usable result.
4. `test_auto_cate_tuned_custom_search_space` — user override honoured.

## 6. Success criteria

1. `pytest tests/test_bayes_did.py tests/test_bayes_rd.py tests/test_auto_cate_tuned.py -q` → all green.
2. `pytest tests/ --ignore=tests/reference_parity` → no new failures vs v0.9.4 baseline.
3. `sp.__version__ == '0.9.5'`.
4. `import statspai as sp` must succeed even in an environment **without** PyMC or Optuna installed — these are optional extras, not core deps.
5. README-style snippet runs end-to-end:
   ```python
   import statspai as sp
   r = sp.bayes_did(df, y='y', treat='ever', post='post', unit='id')
   print(r.summary())
   ```

## 7. Non-goals / trade-offs (explicit)

- **Rust code lives on a separate branch.** Ship only the design + benchmark harness in 0.9.5. Adding a Rust build step to `pip install` without a clean CI story is the single worst thing we could ship to users.
- **Variational inference** (`pymc.fit`) is NOT wired for 0.9.5. NUTS only. VI is strictly a perf optimisation and can land in 0.9.6 behind a flag.
- **Bayesian fuzzy RD, diff-in-RD, bunching, IV-Bayes** — all deferred. Sharp RD and 2x2/panel DID are the two canonical cases; broader Bayesian causal ships as 0.9.6+.
- **Auto-tune search-space tuning** (e.g., auto-detect n/p and pick good defaults) is deferred — users pass their own `search_space` dict to override.
