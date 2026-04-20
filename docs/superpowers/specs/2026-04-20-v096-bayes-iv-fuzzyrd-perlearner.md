# StatsPAI v0.9.6 — Bayesian IV + fuzzy RD, per-learner Optuna, Rust branch

**Author:** Bryce Wang · **Date:** 2026-04-20 · **Status:** design → implementation

## 1. Motivation

v0.9.5 closed three big认怂 items but left three explicit follow-ons:

1. **Bayesian口袋深度** — sharp RD + 2×2/panel DID shipped, but fuzzy RD, Bayesian IV, and bunching were explicitly queued.
2. **Optuna粒度** — `auto_cate_tuned` only tunes nuisance models (shared across learners). A learner whose final-stage CATE model is poorly regularised will not benefit from better nuisances.
3. **Rust主干整洁** — 1.0 should carry the Rust kernel. Start the branch now to amortise the wheel-build pain instead of confronting it all at the final sprint.

## 2. Scope

### In scope for v0.9.6

- **`sp.bayes_iv(data, y, treat, instrument, covariates=None, ...)`** — Bayesian linear IV with joint structural + first-stage model; posterior over the LATE.
- **`sp.bayes_fuzzy_rd(data, y, treat, running, cutoff, ...)`** — fuzzy RD via joint Bayesian local polynomial with ratio estimator for LATE.
- **`auto_cate_tuned(..., tune='nuisance' | 'per_learner' | 'both')`** — new tuner mode that, per learner, samples final-CATE-model hyperparameters independently and picks the best configuration per learner before the race.
- **`feat/rust-hdfe` branch** — Cargo crate scaffold, PyO3 `group_demean` stub, pure-Python fallback wired via `statspai.panel.hdfe`. Branch **not** merged to `main`; `pyproject.toml` on `main` remains maturin-free.

### Out of scope (stays on the "next batch" list)

- Bunching Bayesian estimator (Kleven-style is too structural / not agent-native)
- Variational inference for DID / RD
- Multi-instrument, heterogeneous-effect Bayesian IV (LATE on compliers only for now)
- Per-learner nuisance tuning (i.e. separate nuisance HPs per learner) — always shared for now

## 3. API

### 3.1 `sp.bayes_iv`

```python
def bayes_iv(
    data: pd.DataFrame,
    y: str,
    treat: str,                     # endogenous treatment D
    instrument: str | list[str],    # single or multiple instruments Z
    covariates: list[str] | None = None,
    *,
    prior_late: tuple[float, float] = (0.0, 10.0),
    prior_first_stage_sigma: float = 5.0,
    prior_coef_sigma: float = 10.0,
    prior_noise: float = 5.0,
    rope: tuple[float, float] | None = None,
    hdi_prob: float = 0.95,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_state: int = 42,
    progressbar: bool = False,
) -> BayesianCausalResult
```

**Joint model** (continuous Y and D, single or multiple Z):

```
D = pi_0 + pi_1' * Z + pi_2' * X + v
Y = alpha + LATE * D + beta' * X + eps
corr(v, eps) = rho   (weak instrument: LATE is weakly identified)
```

We model `(v, eps)` as a bivariate Normal with a `pm.LKJCholeskyCov` prior on the correlation; this encodes endogeneity directly. The posterior over `LATE` is the object of interest; the model prices weak identification *automatically* (HDI widens as `pi_1 -> 0`).

### 3.2 `sp.bayes_fuzzy_rd`

```python
def bayes_fuzzy_rd(
    data: pd.DataFrame,
    y: str,
    treat: str,           # binary uptake D (may differ from treated side of cutoff)
    running: str,
    cutoff: float = 0.0,
    bandwidth: float | None = None,
    poly: int = 1,
    *,
    prior_late: tuple[float, float] = (0.0, 10.0),
    prior_slope_sigma: float = 10.0,
    prior_noise: float = 5.0,
    rope: tuple[float, float] | None = None,
    hdi_prob: float = 0.95,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_state: int = 42,
    progressbar: bool = False,
) -> BayesianCausalResult
```

Fits two local polynomial models inside the bandwidth: Y and D each on `(x - c)` with side-dependent slopes. The LATE is the **ratio**
`itt_Y / itt_D`. We model this as a derived deterministic so the posterior of `late` inherits both noise channels automatically (Wald-ratio posterior). If `prob(itt_D ≈ 0)` is large (weak first stage at cutoff), the posterior on `late` becomes bimodal / heavy-tailed — which is correct behaviour.

### 3.3 `auto_cate_tuned(..., tune='per_learner')`

Add a `tune` kwarg to `auto_cate_tuned`:

| value | behaviour |
|---|---|
| `'nuisance'` | **Default.** v0.9.5 behaviour: tune shared nuisance GBM by R-loss. |
| `'per_learner'` | For each learner, tune its *final CATE model* hyperparameters (if the learner has one) against held-out R-loss; keep nuisance at defaults. |
| `'both'` | First tune shared nuisance (mode `'nuisance'`), then for each learner tune its CATE model (mode `'per_learner'`) using the tuned nuisance. |

For S / T / X learners where the CATE is derived analytically from the outcome/propensity models, "per-learner tuning" reduces to "per-learner nuisance tuning" (each learner gets a separate GBM search). For R / DR, the final-stage `cate_model` is the one tuned.

Emits `result.best_result.model_info['per_learner_params']` as a dict keyed by learner short code when `tune in ('per_learner', 'both')`.

### 3.4 Rust branch

Open `feat/rust-hdfe` with the layout proposed in `docs/superpowers/specs/2026-04-20-v095-rust-hdfe-spike.md`. Minimum checked-in:

```
rust/
├── Cargo.toml
├── statspai_hdfe/
│   ├── Cargo.toml
│   └── src/
│       └── lib.rs                  # PyO3 module with group_demean stub
```

Branch-only changes:
- `pyproject.toml` gets a `[tool.maturin]` section **on the branch**, plus `build-system = "maturin"`.
- `src/statspai/panel/hdfe_rust.py` import guard that tries `statspai_hdfe` and falls back to the existing Numba kernel.

The branch is pushed but not merged. `main` stays build-tool-clean.

## 4. File plan

| File | Change |
|---|---|
| `src/statspai/bayes/iv.py` | NEW — `bayes_iv` implementation |
| `src/statspai/bayes/fuzzy_rd.py` | NEW — `bayes_fuzzy_rd` implementation |
| `src/statspai/bayes/__init__.py` | Export `bayes_iv`, `bayes_fuzzy_rd` |
| `src/statspai/metalearners/auto_cate_tuned.py` | Add `tune` param + per-learner loop |
| `src/statspai/__init__.py` | Top-level exports |
| `tests/test_bayes_iv.py` | NEW |
| `tests/test_bayes_fuzzy_rd.py` | NEW |
| `tests/test_auto_cate_tuned.py` | Add per-learner tests |
| `pyproject.toml` | `version = "0.9.6"` |
| `CHANGELOG.md` | 0.9.6 entry |

## 5. Success criteria

1. `sp.bayes_iv` recovers a known LATE at n=600 with HDI coverage and reports widening HDI as the instrument weakens.
2. `sp.bayes_fuzzy_rd` recovers a known fuzzy LATE on a sharp-equivalent DGP (full compliance) within HDI.
3. `auto_cate_tuned(tune='per_learner')` produces a non-empty `per_learner_params` dict; its ATE estimate is within 4σ of truth on the constant-effect DGP.
4. `auto_cate_tuned(tune='both')` runs without error.
5. Full suite still green: no new failures on top of v0.9.5's 1781-passed baseline.
6. `feat/rust-hdfe` exists locally with a compilable `Cargo.toml` (but *not* required to actually build — this spec just says the scaffold is present).

## 6. Test plan

### `test_bayes_iv.py` (≥ 6 tests)

1. `test_bayes_iv_imports_cleanly`
2. `test_bayes_iv_top_level_export`
3. `test_bayes_iv_recovers_late_strong_instrument` — HDI covers true LATE.
4. `test_bayes_iv_weak_instrument_hdi_widens` — weak Z has SD ≥ 2× strong-Z case.
5. `test_bayes_iv_multiple_instruments` — 2+ instruments still fit.
6. `test_bayes_iv_covariates_honoured` — adding a controls list doesn't crash.

### `test_bayes_fuzzy_rd.py` (≥ 5 tests)

1. `test_bayes_fuzzy_rd_top_level_export`
2. `test_bayes_fuzzy_rd_full_compliance_matches_sharp` — full compliance → behaves like sharp RD within margin.
3. `test_bayes_fuzzy_rd_partial_compliance_recovers_late`
4. `test_bayes_fuzzy_rd_bandwidth_shrinks_sample`
5. `test_bayes_fuzzy_rd_cutoff_validation`

### `test_auto_cate_tuned.py` (+4 tests)

1. `test_auto_cate_tuned_per_learner_mode`
2. `test_auto_cate_tuned_both_mode`
3. `test_auto_cate_tuned_per_learner_sets_params_key`
4. `test_auto_cate_tuned_invalid_tune_mode_raises`

## 7. Non-goals

- Structural bunching Bayesian estimator (Kleven-style). Deferred to 0.9.7+.
- VI sampler for IV / fuzzy RD. NUTS only.
- Merging `feat/rust-hdfe` to main. Explicitly a branch-only deliverable.
- Heterogeneous-effect Bayesian IV. LATE only.
