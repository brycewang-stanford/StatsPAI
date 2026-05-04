# RFC: `sp.hal_tmle(variant='projection')` — Riesz-projection HAL-TMLE

> **Status:** open. Implementation deferred until a parity test against
> a published number is available; the v1.11.x code path was a no-op
> on the point estimate (see `CHANGELOG`), and `1.13` raises
> `NotImplementedError` rather than ship a half-correct version.

## Background

`sp.hal_tmle` currently implements the standard "delta" TMLE plug-in
with HAL nuisances (Benkeser & van der Laan 2016) — see
[`src/statspai/tmle/hal_tmle.py`](../../src/statspai/tmle/hal_tmle.py).
The `variant='projection'` flag was reserved for the
**Riesz-projection targeting step** introduced by Li, Qiu, Wang & van
der Laan (2025) [`@li2025regularized`, arXiv:2506.17214] §3.2, which
re-estimates the efficient-influence-function (EIF) clever covariate
inside the HAL working model space rather than using the canonical
inverse-propensity form.

The pre-`1.13` implementation accepted `variant='projection'` but
silently shrunk the targeting `ε` by an ad-hoc factor *after* the
estimate had already been computed, so `result.estimate` was
unchanged. That was misleading enough that we now raise
`NotImplementedError` — better to refuse than to fake it.

## Why this is non-trivial

The paper's projection step is genuinely different from the standard
plug-in TMLE flow:

1. **Riesz representer** — for the ATE under binary treatment, the
   canonical representer is `r(D, X) = D/g(X) - (1-D)/(1-g(X))`, where
   `g` is the propensity score. The "projection" replaces this with
   `r̂(D, X)` defined as the L²-projection of the canonical
   representer onto the HAL working-model space. This stabilises the
   clever covariate when `g` is near 0 or 1.
2. **Targeting step** — the standard TMLE one-dimensional fluctuation
   `Q_n^*(D, X) = Q_n(D, X) + ε · H(D, X)` is replaced by
   `Q_n^*(D, X) = Q_n(D, X) + ε · r̂(D, X)`, with `ε` solving the
   modified score equation `∑ r̂(D_i, X_i) · (Y_i − Q_n^*(D_i, X_i)) = 0`.
3. **Inference** — the influence function picks up the projection
   residual; the variance estimator must use `r̂`, not `r`.

A correct implementation needs all three pieces, plus a parity-grade
test (the paper's published number is the natural target).

## Implementation roadmap

This RFC is a starting point for whoever picks this up. The
recommended path:

### Phase 1 — minimal projection variant (1–2 days)

1. After fitting `Q_n` and `g_n` via HAL (existing code in
   [`hal_tmle`](../../src/statspai/tmle/hal_tmle.py)), compute the
   canonical representer `r_n(D, X)` on every training row.
2. Re-fit `r_n` against the same HAL basis used for `Q_n` (call this
   `r̂_n`). L² projection (ridge regression on the HAL features) is
   the simplest choice; the paper allows L¹ as well.
3. Replace the clever covariate `H(D, X)` in the targeting step with
   `r̂_n(D, X)` and solve for `ε` using the standard logistic
   submodel for binary `Y` or the Gaussian submodel for continuous `Y`.
4. Recompute `result.estimate` from the targeted `Q_n^*`.
5. Recompute the influence function and SE using the projected
   representer.
6. Write a parity test in `tests/reference_parity/test_hal_tmle_parity.py`
   replicating the paper's headline number (or, if the paper publishes
   a simulation table, a representative cell from it).
7. Update the registry: drop the `variant='projection'` limitation and
   register the variant as `stable` once the parity test passes.

### Phase 2 — full HAL with subset-product basis (open)

The current implementation is "main-effects HAL" (per-feature step
functions only — see the docstring at the top of
[`hal_tmle.py`](../../src/statspai/tmle/hal_tmle.py)). The full HAL
basis (subset-product indicators) is what gives the universal
càdlàg-approximation guarantee. A separate RFC should cover the
sparse-tensor implementation needed to make the full basis tractable
on `p > 5` covariates.

## What blocks promotion to `stability='stable'`

Two gates:

1. The parity test must replicate the paper's published number to
   within a documented `atol` / `rtol`.
2. The implementation must be documented to flag when the projection
   step degenerates (e.g. when the HAL basis is too small to give a
   meaningful re-estimation, or when `g_n` saturates at 0/1 so that
   the canonical `r_n` is undefined in part of the support).

Until both gates clear, `variant='projection'` should remain
`experimental` (when implemented) or continue to raise
`NotImplementedError` (the current state).

## Pointers

- Paper: Li, Qiu, Wang & van der Laan (2025), arXiv:2506.17214,
  bib key `@li2025regularized` in `paper.bib`.
- Existing HAL nuisance code: [`src/statspai/tmle/hal_tmle.py`](../../src/statspai/tmle/hal_tmle.py).
- Standard TMLE targeting: [`src/statspai/tmle/tmle.py`](../../src/statspai/tmle/tmle.py).
- Stability tier docs: [`docs/guides/stability.md`](../guides/stability.md).
