# StatsPAI v0.9.12 — Probit-scale MTE (Heckman selection frame)

**Author:** Bryce Wang · **Date:** 2026-04-20 · **Status:** design → implementation

## 1. Motivation

v0.9.11 handled multi-IV MTE and the true CHV-2011 PRTE weight. The one remaining thread in the Bayesian MTE frontier is the **Heckman (1979) selection model** with Gaussian/probit-scale errors, which is the conventional parametric frame underlying HV 2005:

```
V_i ~ N(0, 1)
D_i = 1{ logit(p_i) + V_i > 0 }   (probit / normal-tail selection)
MTE(v) = μ_1(X) - μ_0(X) + (ρ_{1V}σ_1 - ρ_{0V}σ_0) · v  (linear in V)
```

Under the strict bivariate-normal HV assumption, MTE is linear on the V scale. Our existing polynomial-in-U_D model (U_D ∈ [0,1]) is richer but parametrised on the uniform scale; a user who wants the conventional Heckman interpretation needs the V-scale formulation.

v0.9.12 adds a third orthogonal axis to `sp.bayes_mte`:

| axis | options |
|---|---|
| `first_stage` | `'plugin'` / `'joint'` |
| `mte_method` | `'polynomial'` / `'hv_latent'` |
| **`selection`** (new) | `'uniform'` (default) / `'normal'` |

Together these describe the 8-combo grid of Bayesian MTE specifications. All 8 are expected to fit without error; recovery characteristics differ by DGP.

## 2. Scope

### In scope (v0.9.12)

- **`sp.bayes_mte(..., selection='uniform' | 'normal')`** — new kwarg. `'uniform'` (default) preserves v0.9.11 behaviour. `'normal'` reinterprets the polynomial abscissa as `V_i = Φ^(-1)(U_D_i)` (probit scale, V ∈ ℝ). All combinations with existing `first_stage` / `mte_method` flags supported.
- **Mathematical identity guarantee**: `'normal'` mode fits the MTE polynomial in V. Under `poly_u=1` + `selection='normal'` + `mte_method='hv_latent'` the model exactly matches the linear Heckman-HV MTE.
- **Method label** reflects the selection scale: `"Bayesian MTE on V scale (...)"` vs `"Bayesian MTE on U_D scale (...)"`.

### Out of scope (deferred to 0.9.13+)

- **Full bivariate-normal error covariance** `(U_0, U_1, V) ~ N(0, Σ)` with free correlations `ρ_{0V}`, `ρ_{1V}`. Requires explicit mixture modelling and MvNormal over `(Y, D)` that has known convergence pathologies in PyMC. Tracked as a separate release.
- Selection-on-levels (`μ_0` ≠ `μ_1`) as a free parameter: currently absorbed into `alpha + beta_X·X`. Disentangling requires the full mixture model.
- Rust Phase 2, VI backends for MTE.

## 3. API

```python
def bayes_mte(
    data, y, treat, instrument, covariates=None,
    *,
    first_stage: str = 'plugin',
    mte_method: str = 'polynomial',
    selection: str = 'uniform',        # NEW
    u_grid: np.ndarray | None = None,
    poly_u: int = 2,
    ...
)
```

### Semantics

Under `selection='uniform'` (v0.9.11 default):
```
Abscissa a ∈ [0, 1]  (either p_i in polynomial mode or U_D_i in hv_latent)
MTE-curve: τ(a) = Σ_k b_k · a^k,  a ∈ u_grid ⊂ [0, 1]
```

Under `selection='normal'`:
```
Abscissa v ∈ ℝ  with v = Φ^{-1}(a)
MTE-curve: τ(v) = Σ_k b_k · v^k,  v ∈ Φ^{-1}(u_grid) ⊂ ℝ
```

The returned `mte_curve` DataFrame still has a `u` column (in [0,1]) for user convenience — this is the natural propensity scale. The `v` column is added when `selection='normal'` so users can see the probit coordinate.

ATE / ATT / ATU integrals: on the V scale, we integrate against the Gaussian density `φ(v) dv` rather than uniform `du`. This matches the HV identified-integrand under bivariate-normal.

### Defaults justification

`'uniform'` stays the default because:
1. No breaking change to v0.9.11 users.
2. The `[0,1]` propensity scale is the agent-native abstraction most users want.
3. `'normal'` is a purer "I'm doing Heckman textbook work" mode — users who need it will reach for it.

## 4. File plan

| File | Change |
|---|---|
| `src/statspai/bayes/mte.py` | Add `selection` kwarg; transform abscissa via `Φ^{-1}` when `'normal'`; adjust ATE integration to Gaussian measure when `'normal'`; update method label + `model_info`. |
| `src/statspai/bayes/_base.py` | `mte_curve` DataFrame adds a `v` column when fit was on V scale (empty otherwise). |
| `tests/test_bayes_mte_selection.py` | NEW — `selection='normal'` recovery + orthogonality. |
| `pyproject.toml` | `version = "0.9.12"` |
| `CHANGELOG.md` | 0.9.12 entry |

## 5. Test plan

- `test_selection_uniform_back_compat` — scalar call with default `selection` returns v0.9.11 behaviour.
- `test_selection_normal_api_surface` — new kwarg flows to `model_info['selection']`; method label mentions V scale.
- `test_selection_normal_recovers_linear_heckman` — DGP with Gaussian V and MTE linear in V; `poly_u=1 + selection='normal' + hv_latent` recovers the slope.
- `test_selection_normal_all_orthogonal_combos_run` — 4 combos (plugin/joint × polynomial/hv_latent) × selection='normal'.
- `test_selection_invalid_value_raises`.
- `test_selection_normal_mte_curve_has_v_column`.

## 6. Success criteria

1. `selection='normal'` runs on all 4 `(first_stage, mte_method)` combos.
2. On a DGP with `MTE(V) = 0.5 + 1.5·V` (Gaussian latent V), `poly_u=1 + selection='normal' + hv_latent` recovers `(0.5, 1.5)` within HDI.
3. Full suite stays green (zero-new-failures rule).
4. Two rounds of code review, zero ship-blockers.

## 7. Non-goals (explicit)

- Free cross-correlation parameters `(ρ_{0V}, ρ_{1V})` — the bivariate-normal mixture is its own release.
- Policy weights on V scale — current `policy_weight_*` builders still operate on the `[0,1]` grid; users on `selection='normal'` get `u_grid` transformed via `Φ` so policy integrals remain well-defined on the propensity scale.
- Changing the sign / direction of `V = Φ^{-1}(p)` (we use the standard convention).
