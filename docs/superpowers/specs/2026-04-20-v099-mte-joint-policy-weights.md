# StatsPAI v0.9.9 — Joint first-stage MTE + policy-relevant weights

**Author:** Bryce Wang · **Date:** 2026-04-20 · **Status:** design → implementation

## 1. Motivation

v0.9.8 shipped Bayesian MTE via a **plug-in** first-stage (logit MLE → fixed propensity → polynomial MTE posterior). The non-goals list called out two concrete next steps:

1. **Full-joint first stage** — model the first-stage logit coefficients *inside* the PyMC graph so first-stage uncertainty propagates into the MTE curve.
2. **Policy-relevant MTE weights** — AMTE, PRTE (Carneiro-Heckman-Vytlacil 2011). The current `BayesianMTEResult` exposes `ate / att / atu` as simple integrals over the population / treated / untreated region; policy-relevant weights generalise this to arbitrary weight functions over `U_D`.

This release closes both in one sweep.

## 2. Scope

### In scope (v0.9.9)

- **`sp.bayes_mte(..., first_stage='plugin' | 'joint')`** — new kwarg. Defaults to `'plugin'` for backward-compat.
- **`BayesianMTEResult.policy_effect(weight_fn, label)`** — returns posterior summary of `E[w(U) * MTE(U)] / E[w(U)]`.
- **`sp.bayes.policy_weight_*`** helper builders for common policies:
  - `policy_weight_prte(baseline_propensity_shift)` — PRTE under a uniform shift of propensity by `delta`.
  - `policy_weight_subsidy(u_lo, u_hi)` — uniform subsidy affecting only units whose `U_D ∈ [u_lo, u_hi]`.

### Out of scope (stays queued)

- Multi-instrument Bayesian MTE — still one scalar Z.
- GP-over-u MTE surface — still polynomial of order `poly_u`.
- Rust Phase 2 — separate branch.

## 3. API

### 3.1 Joint first-stage

```python
def bayes_mte(
    data, y, treat, instrument, covariates=None,
    *,
    first_stage: str = 'plugin',            # 'plugin' | 'joint'
    u_grid: np.ndarray | None = None,
    poly_u: int = 2,
    ...
)
```

`first_stage='joint'` switches to:

```
pi_0 ~ Normal(0, prior_coef_sigma)
pi_Z ~ Normal(0, prior_coef_sigma)
pi_X ~ Normal(0, prior_coef_sigma, shape=k_x)
p_i = sigmoid(pi_0 + pi_Z * Z_i + pi_X' X_i)
D_i ~ Bernoulli(p_i)                          # observed
```

The structural equation uses `p_i` (a Deterministic over first-stage params) in the MTE polynomial:

```
alpha, beta_X ~ Normal priors
b_mte ~ Normal priors  (shape poly_u+1)
MTE_i = sum_k b_mte[k] * p_i ** k
mu_Y_i = alpha + beta_X' X_i + D_i * MTE_i
Y_i ~ Normal(mu_Y_i, sigma_eps)
```

Because `p_i` is itself random, `MTE_i` inherits first-stage uncertainty automatically. **Trade-off**: joint sampling is slower than the plug-in variant (roughly 2×–4× wall time for comparable DGPs) because the posterior landscape is higher-dimensional. For production rigour use joint; for prototype or large-n use plugin.

### 3.2 Policy effects

```python
def policy_effect(
    self,
    weight_fn: Callable[[np.ndarray], np.ndarray],
    label: str = 'policy',
    rope: tuple[float, float] | None = None,
) -> dict:
    """Posterior summary of E[w(U) * MTE(U)] / E[w(U)] over the fit's u_grid.

    Returns a dict with keys:
        label, estimate (posterior mean), std_error, hdi_low, hdi_high,
        prob_positive, prob_rope (optional).
    """
```

This is the *policy-relevant treatment effect* for a user-specified weighting kernel. Unlike `ate / att / atu` (which are pre-computed integrals over the grid + population shares), `policy_effect` lets the user craft arbitrary policy counterfactuals after the fit.

### 3.3 Helper builders

```python
sp.bayes.policy_weight_prte(shift: float)
    -> Callable[[np.ndarray], np.ndarray]
```
Returns a weight function implementing "units whose propensity shifts across the treatment margin under a uniform index shift `shift`". Sensible when the policy counterfactual is "raise the subsidy by `delta`".

```python
sp.bayes.policy_weight_subsidy(u_lo: float, u_hi: float)
    -> Callable[[np.ndarray], np.ndarray]
```
Weight is 1 inside `[u_lo, u_hi]` and 0 outside — subsidy reaches the intended band of compliers.

## 4. File plan

| File | Change |
|---|---|
| `src/statspai/bayes/mte.py` | Add `first_stage='joint'` branch |
| `src/statspai/bayes/_base.py` | Add `policy_effect` method on `BayesianMTEResult` |
| `src/statspai/bayes/policy_weights.py` | NEW — weight builders |
| `src/statspai/bayes/__init__.py` | Export weight builders |
| `tests/test_bayes_mte.py` | Add joint-mode tests |
| `tests/test_bayes_mte_policy.py` | NEW — policy_effect + builder tests |
| `pyproject.toml` | `version = "0.9.9"` |
| `CHANGELOG.md` | 0.9.9 entry |

## 5. Test plan

- `test_bayes_mte_joint_vs_plugin_similar_posterior_mean` — on a correctly-specified DGP, joint and plug-in should roughly agree on the posterior mean (differ on HDI width, joint wider).
- `test_bayes_mte_joint_hdi_wider_than_plugin` — joint correctly propagates first-stage uncertainty; HDI width should be ≥ plug-in's width.
- `test_policy_effect_ate_matches_native` — `policy_effect(lambda u: np.ones_like(u))` ≈ `.ate`.
- `test_policy_effect_subsidy_band_recovers_local_mte`.
- `test_policy_weight_prte_returns_callable`.
- `test_policy_weight_subsidy_bounds_enforced`.

## 6. Success criteria

1. `first_stage='joint'` returns a `BayesianMTEResult` with the same field contract as `'plugin'`.
2. On a correctly-specified DGP the joint-mode posterior mean is within 1 prior SD of the plug-in value.
3. `policy_effect(lambda u: np.ones_like(u))` matches `result.ate` within 1e-6.
4. Full suite stays green vs v0.9.8 baseline.

## 7. Non-goals

- Multi-instrument MTE with per-instrument weights.
- Non-linear MTE surface (GP over `u`).
- Automatic policy-bound extrapolation beyond the grid.
