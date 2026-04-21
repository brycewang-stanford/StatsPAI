# StatsPAI v0.9.13 — arviz HDI-kwarg compat + ATT/ATU uncertainty

**Author:** Bryce Wang · **Date:** 2026-04-20 · **Status:** design → implementation

## 1. Motivation

Two items were explicitly deferred across v0.9.10 / v0.9.11 / v0.9.12 code reviews:

1. **`az.hdi(..., hdi_prob=...)` kwarg renamed to `prob` in arviz ≥ 0.18.** All our Bayesian estimators call `az.hdi(samples, hdi_prob=self.hdi_prob)`. The day a user upgrades arviz past that version, every Bayesian estimator returns malformed output. We ship a tiny compatibility shim now so the jump is a no-op.

2. **`BayesianMTEResult.att` / `.atu` expose the posterior mean only.** The internal `_integrated_effect` helper already computes `(mean, sd)` but v0.9.12 discarded the SD. Without SD / HDI, users can't tell whether ATT differs from ATU meaningfully.

Both are small, surgical, and high-value-per-LOC.

## 2. Scope

### In scope (v0.9.13)

- **`_az_hdi_compat(samples, hdi_prob=...)`** in `src/statspai/bayes/_base.py` — calls `az.hdi(samples, hdi_prob=...)` first, falls back to `az.hdi(samples, prob=...)` on `TypeError`. All internal Bayesian code replaces the direct `az.hdi(...)` call with this shim.

- **ATT / ATU posterior SD + HDI** on `BayesianMTEResult`:
  - `att_sd`, `att_hdi_lower`, `att_hdi_upper`
  - `atu_sd`, `atu_hdi_lower`, `atu_hdi_upper`

- **`_integrated_effect` returns the full posterior summary** (mean, sd, hdi_low, hdi_high); caller records all four.

- **Summary output** prints ATT / ATU with uncertainty when non-NaN.

### Out of scope

- Full bivariate-normal HV.
- Rust Phase 2.
- Per-unit ATT (only population-level integrated ATT).

## 3. API changes

### 3.1 `_az_hdi_compat`

```python
def _az_hdi_compat(samples, hdi_prob=0.95):
    _, az = _require_pymc()
    try:
        return np.asarray(az.hdi(samples, hdi_prob=hdi_prob)).ravel()
    except TypeError:
        return np.asarray(az.hdi(samples, prob=hdi_prob)).ravel()
```

Every `az.hdi` call in `statspai.bayes.*` goes through this wrapper.

### 3.2 `BayesianMTEResult` new fields

Append (default NaN → preserves serialised-result backward compat):

```
att_sd, att_hdi_lower, att_hdi_upper
atu_sd, atu_hdi_lower, atu_hdi_upper
```

`ate_sd` is NOT a new field — `posterior_sd` covers the primary estimand.

### 3.3 `summary()` update

When ATT/ATU SD is finite, print `ATT: 1.234 (sd 0.567, HDI [...])`.

## 4. File plan

| File | Change |
|---|---|
| `src/statspai/bayes/_base.py` | NEW `_az_hdi_compat`; `BayesianMTEResult` new fields; `summary()` extended |
| `src/statspai/bayes/mte.py` | `_integrated_effect` returns full summary; caller wires into result |
| `src/statspai/bayes/did.py`, `rd.py`, `iv.py`, `fuzzy_rd.py`, `hte_iv.py` | Replace `az.hdi(...)` calls with shim |
| `tests/test_bayes_mte_uncertainty.py` | NEW |
| `tests/test_bayes_hdi_compat.py` | NEW |
| `pyproject.toml` | `version = "0.9.13"` |

## 5. Test plan

### `test_bayes_hdi_compat.py`

1. `test_shim_accepts_hdi_prob_kwarg` — forwards to `az.hdi`.
2. `test_shim_handles_typeerror` — monkey-patch `az.hdi` to raise on `hdi_prob`, confirm fallback to `prob`.
3. `test_shim_returns_length_2_array`.

### `test_bayes_mte_uncertainty.py`

1. `test_att_atu_sd_fields_populated` — after fit, both SDs finite.
2. `test_att_atu_hdi_brackets_posterior_mean` — `hdi_lower < mean < hdi_upper`.
3. `test_summary_shows_att_atu_uncertainty` — string contains `sd ` and `HDI [`.
4. `test_empty_population_nan_safe` — all-treated DGP → ATU SD is NaN, no crash.

## 6. Success criteria

1. Existing arviz still works.
2. Mocked future-arviz (raises on `hdi_prob`) works via fallback.
3. `r.att_sd`, `r.atu_sd` finite on a typical fit.
4. Two rounds of code review, zero ship-blockers.
