# StatsPAI v0.9.11 — Multi-instrument MTE + CHV-2011 observed-propensity PRTE

**Author:** Bryce Wang · **Date:** 2026-04-20 · **Status:** design → implementation

## 1. Motivation

v0.9.10 shipped textbook HV-latent MTE but left two threads from earlier non-goals lists:

1. **Multi-instrument MTE** — `sp.bayes_mte` took `instrument: str` (scalar), forcing users with 2+ IVs to pick one. All other Bayesian estimators (`sp.bayes_iv`, `sp.bayes_hte_iv`) already support `instrument: str | list`; MTE was the inconsistent one.

2. **True CHV-2011 PRTE builder** — v0.9.9 shipped `sp.policy_weight_prte(shift)` as a **stylised** rectangle and the docstring explicitly said the real CHV-2011 PRTE requires the observed propensity kernel. The worked example in the docstring asked the user to hand-roll a `gaussian_kde`-based weight_fn. v0.9.11 makes that one-liner: `sp.policy_weight_observed_prte(propensity_sample, shift)`.

Both are small, well-scoped additions that close open API gaps.

## 2. Scope

### In scope (v0.9.11)

- **`sp.bayes_mte(instrument: str | Sequence[str], ...)`** — accept multi-instrument:
  - Scalar path unchanged (API back-compat).
  - List path: `Z` becomes `(n, k)` matrix; first-stage logit `pi_Z ~ Normal(0, σ, shape=k)`; `logit = pi_0 + Z @ pi_Z + ...`.
- **`sp.policy_weight_observed_prte(propensity_sample, shift)`** — CHV 2011 weights: `w(u) ∝ [f_P(u) - f_{P+Δ}(u)] / Δ` where `f_P` is the kernel-density estimate of the observed propensity sample. Normalised to unit sum on the grid passed into `policy_effect`.
- Retain the stylised `sp.policy_weight_prte(shift)` unchanged — useful as a quick exploration tool, and the docstring already flags it as stylised.

### Out of scope (explicitly deferred)

- Bivariate-normal HV selection model (Heckman-style with `(U_0, U_1, V)` covariance structure). This is its own design problem — the right move is 0.9.12+.
- Policy counterfactual = "add a new instrument" — would require dedicated API.
- Rust Phase 2.

## 3. API changes

### 3.1 `bayes_mte(instrument: str | Sequence[str])`

```python
def bayes_mte(
    data: pd.DataFrame,
    y: str,
    treat: str,
    instrument: Union[str, Sequence[str]],   # CHANGED: was str
    covariates: Optional[List[str]] = None,
    ...
)
```

Inside:
- Normalise to a list `iv_cols = [instrument] if isinstance(instrument, str) else list(instrument)`.
- `Z = clean[iv_cols].to_numpy(dtype=float)` → shape `(n, k)`.
- `_logit_propensity(Z, X, D)` already handles 2-D Z (see its `W = Z.reshape(-1, 1) if Z.ndim == 1 else Z` branch — good).
- PyMC side: `pi_Z = pm.Normal('pi_Z', mu=0, sigma=prior_coef_sigma, shape=k)` then `logit = pi_intercept + pm.math.dot(Z, pi_Z) + ...`.

### 3.2 `sp.policy_weight_observed_prte(propensity_sample, shift)`

```python
def policy_weight_observed_prte(
    propensity_sample: np.ndarray,
    shift: float,
    *,
    bw_method: str | float | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """True CHV-2011 PRTE weights from the observed propensity
    distribution, via Gaussian KDE."""
```

Implementation:
- Validate `propensity_sample` lies in `[0, 1]` (common sanity check).
- Validate `shift` in `(-1, 1)` non-zero (matches `policy_weight_prte`).
- Build `kde = scipy.stats.gaussian_kde(propensity_sample, bw_method=bw_method)`.
- Return closure `w(u) -> (kde(u) - kde(u - shift)) / shift`, clipped at 0 from below (negative weights are not meaningful for integration against an MTE curve and usually indicate grid-edge artefacts).

Edge cases handled:
- `shift > 0`: positive marginal expansion of propensity (compliers shift up).
- `shift < 0`: negative marginal shrinkage (defiers / contraction).
- `u - shift` outside `[0, 1]`: kde density naturally falls off; clip avoids negative weight.

### 3.3 Export wiring

Add `policy_weight_observed_prte` to `sp.bayes.__all__` and top-level `__all__`.

## 4. File plan

| File | Change |
|---|---|
| `src/statspai/bayes/mte.py` | `instrument: str | Sequence[str]`, list normalisation, shape-k `pi_Z` prior |
| `src/statspai/bayes/policy_weights.py` | NEW func `policy_weight_observed_prte` |
| `src/statspai/bayes/__init__.py` | Export new builder |
| `src/statspai/__init__.py` | Top-level export |
| `tests/test_bayes_mte_multi_iv.py` | NEW — multi-IV recovery + scalar back-compat |
| `tests/test_bayes_mte_policy.py` | Extend with `policy_weight_observed_prte` tests |
| `pyproject.toml` | `version = "0.9.11"` |
| `CHANGELOG.md` | 0.9.11 entry |

## 5. Test plan

- `test_bayes_mte_multi_instrument_scalar_back_compat` — passing a single-element list returns same posterior as scalar within sampling noise.
- `test_bayes_mte_multi_instrument_recovery` — 2-IV DGP where both first-stage coefficients are identified; the MTE polynomial still recovers truth.
- `test_bayes_mte_multi_instrument_model_info` — `model_info['instruments']` reports the list.
- `test_policy_weight_observed_prte_returns_callable`.
- `test_policy_weight_observed_prte_input_validation` — reject out-of-bounds samples, zero shift, shift outside `(-1,1)`.
- `test_policy_weight_observed_prte_positive_shift_yields_marginal_mass` — on a uniform-propensity sample with shift=0.1, the weight peaks near the induced margin.
- `test_policy_weight_observed_prte_integrates_with_policy_effect` — end-to-end: fit MTE, call `r.policy_effect(sp.policy_weight_observed_prte(r._propensity_sample, 0.1))`, returns a finite posterior.

## 6. Success criteria

1. Multi-instrument `bayes_mte` runs on a 2-IV DGP and recovers the true MTE polynomial within HDI at n=600.
2. `policy_weight_observed_prte(uniform_sample, shift=0.1)` returns a mass concentrated around `u ∈ [0.5-0.05, 0.5+0.05]` on a uniform-propensity DGP (sanity).
3. Scalar-instrument calls remain backward-compatible (existing tests don't change behaviour).
4. Two rounds of code review — no ship-blockers.
5. Full regression stays within the existing flakiness baseline (≤ 2 pre-existing flakies on an isolated-retry-passing basis).

## 7. Non-goals

- Bivariate-normal HV (deferred).
- Per-instrument policy weights (deferred).
- IV-strength diagnostics for MTE first stage (users can call `sp.check_identification` on the input).
