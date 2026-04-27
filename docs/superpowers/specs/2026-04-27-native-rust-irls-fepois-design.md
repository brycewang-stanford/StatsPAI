# Native Rust IRLS for `sp.fast.fepois` — Design Spec

> **Status**: design (pre-implementation). Authored 2026-04-27. Targets
> v1.8.0 (Phase A) and v1.8.1 (Phase B). Implementation gated on user
> approval of this spec and the follow-up `writing-plans` output.

## 1. Goal

Close the wall-clock gap between `sp.fast.fepois` (StatsPAI's native
PPML-HDFE estimator, shipped in v1.8 RC under `feat(fast): native HDFE
stack`, commit `b4ba4a3`) and R `fixest::fepois` from the current ~4×
to ≤ 1.5× on the project's medium benchmark dataset (n=1M, fe1=100k,
fe2=1k), without breaking any existing public API or numeric guarantee.

## 2. Non-goals

The following are explicitly out of scope for this work and remain on
the existing tracked list in `benchmarks/hdfe/SUMMARY.md`:

- GLM families other than Poisson (logit / negbin / gamma — `feglm`).
- Anderson(m) acceleration in the demean inner loop.
- GPU / CUDA benchmarks (no hardware on the dev box).
- Cluster-robust SE inside `sp.fast.fepois` beyond the current CR1
  (CR2 / CR3 / wild bootstrap stay on the Phase 4 inference primitives
  surface; wiring `vcov="cr2"|"cr3"` into `fepois()` is a separate PR).
- IM (Imbens-Kolesar) Satterthwaite DOF correction.
- Word/DOCX export from `sp.fast.etable`.
- Wiring `sp.callaway_santanna` / `sp.sun_abraham` through `backend="fast"`.

## 3. Background — current state

### 3.1 Wall-clock measurement

Source: `benchmarks/hdfe/SUMMARY.md` and `PHASE2_VERIFY.md`. Medium
dataset, n=1M, fe1=100k, fe2=1k.

| Backend                          | Wall    | Iters | vs fixest |
|----------------------------------|--------:|------:|----------:|
| R `fixest::fepois`               | 0.64 s  | 5     | 1.0×      |
| `sp.fast.fepois` (Phase 0 ship)  | 2.61 s  | 6     | **4.08×** |
| `pyfixest.fepois`                | 4.16 s  | —     | 6.5×      |

### 3.2 Where the gap lives

The IRLS outer loop in `src/statspai/fast/fepois.py` (lines 548–601) is
in pure Python. Each of the 6 outer iters does:

1. `z = eta + (y - mu) / mu`, `w = mu * obs_weights` — O(n) NumPy.
2. **Weighted alternating-projection demean** of `z` (1 column) and
   `X` (k columns) via `_weighted_ap_demean` (line 204). This path
   uses Python `np.bincount(weights=...)` for the weighted group sum;
   the Phase 1 Rust kernel `demean_2d` is **unweighted only** and
   never gets called from inside IRLS. **This step is ~80% of wall
   time on the medium dataset.**
3. WLS solve: `(X̃ᵀ W X̃) β = X̃ᵀ W z̃`, k×k SPD system. Trivial.
4. `eta_new`, `mu_new`, deviance, step-halving. O(n) NumPy.

Closing the 4× gap therefore boils down to: get the inner weighted
demean off Python, and (further) get the entire IRLS off Python so
there is no Python ↔ Rust FFI boundary inside the hot loop.

## 4. Decomposition

Two independently-shippable phases:

| Phase | Length | Rust delta | Python delta | Target wall (medium) | Target version |
|-------|-------:|------------|--------------|---------------------:|----------------|
| **A — Weighted demean kernel**     | ~1 week  | + `demean_2d_weighted` PyO3 + `pub fn weighted_demean_*` crate-internal | `_weighted_ap_demean` rewires to Rust with NumPy fallback | ≤ 1.5 s | v1.8.0 |
| **B — Native Rust IRLS**           | ~2 weeks | + `irls.rs` with full IRLS loop + `fepois_irls` PyO3 binding; reuses Phase A weighted demean as a crate-internal call | `fepois()` becomes a thin parser / pre-pass / vcov shell; IRLS body removed | ≤ 0.95 s (≤ 1.5× fixest) | v1.8.1 |

### 4.1 Three invariants across both phases

1. **Public API frozen.** `sp.fast.fepois(formula, data, ...)` signature,
   `FePoisResult` schema, and accessor return values are unchanged. The
   only user-observable diff after Phase B is the `backend` field on
   `FePoisResult` flipping from `"statspai-native"` to
   `"statspai-native-rust"` — flagged in CHANGELOG.
2. **Rust-unavailable still works.** `_HAS_RUST = False` (CI flag
   `STATSPAI_SKIP_RUST=1` or no compiled wheel) must keep the full
   feature set running on the existing Python path. The Python
   `_weighted_ap_demean_numpy` is preserved as the fallback.
3. **Numeric continuity.** Each phase ships parity tests that gate
   the merge: vs `pyfixest.fepois` ≤ 1e-13, vs `fixest::fepois` ≤ 1e-6,
   vs the prior Python path ≤ 1e-12 on coef and ≤ 1e-10 on SE. Tolerances
   that need to widen MUST be justified inline; silent widening is a
   release blocker.

## 5. Phase A — Weighted demean kernel

### 5.1 Rust crate surface

File-level changes under `rust/statspai_hdfe/src/`:

```text
demean.rs
  + #[inline] fn weighted_group_sweep(
        x: &mut [f64], codes: &[i64],
        weights: &[f64], wsum: &[f64], scratch: &mut [f64],
    )
  + fn weighted_sweep_all_fe(
        x, fe_codes, weights, wsum_list, scratch,
    )
  + pub fn weighted_demean_column_inplace(
        x, fe_codes, weights, wsum_list, scratch,
        max_iter, tol_abs, tol_rel, accelerate, accel_period,
    ) -> DemeanInfo
  + pub fn weighted_demean_matrix_fortran_inplace(
        mat, n, p, fe_codes, weights, wsum_list, wsum_lens,
        max_iter, tol_abs, tol_rel, accelerate, accel_period,
    ) -> Vec<DemeanInfo>
        // Rayon-parallel over columns, reuses aitken_step + safeguard

lib.rs
  + #[pyfunction] fn demean_2d_weighted(
        x: PyReadwriteArray2<f64>,         // F-contig (n, p)
        fe_codes: &PyList,                  // K × i64[n]
        wsum: &PyList,                      // K × f64[G_k]
        weights: PyReadonlyArray1<f64>,     // f64[n]
        max_iter, tol_abs, tol_rel, accelerate, accel_period,
    ) -> PyList                             // List[{iters, converged, max_dx}]
  - __version__: "0.2.0" → "0.3.0"
```

The two `pub fn` are crate-internal API (not just `#[pyfunction]`
wrappers): Phase B's `irls.rs` will call them directly without going
through PyO3.

### 5.2 Inner kernel diff

The unweighted sweep (current `group_sweep`):

```rust
for i: scratch[codes[i]] += x[i];
for g: scratch[g] /= counts[g];     // unweighted: counts = group size
for i: x[i] -= scratch[codes[i]];
```

The new weighted sweep:

```rust
for i: scratch[codes[i]] += weights[i] * x[i];
for g: scratch[g] /= wsum[g];       // wsum[g] = Σ_{i ∈ g} weights[i]
for i: x[i] -= scratch[codes[i]];
```

Other components — `base_scale` precompute, double-threshold stop
(`tol_abs + tol_rel * base_scale`), Aitken / Irons-Tuck extrapolation
every `accel_period` sweeps, `max|x_acc| < 10·base_scale` safeguard,
Rayon-parallel column dispatch — are reused **bit-for-bit** from the
existing Phase 1 path.

### 5.3 Python wiring

In `src/statspai/fast/fepois.py`:

- Rename current `_weighted_ap_demean` → `_weighted_ap_demean_numpy`.
- Add new `_weighted_ap_demean_dispatcher(arr, fe_codes, counts_list,
  weights, *, max_iter, tol)`:
  - If `_HAS_RUST` and `len(fe_codes) >= 2`: precompute
    `wsum_list = [np.bincount(c, weights=weights, minlength=G) for c, G in zip(fe_codes, counts_list)]`,
    coerce `arr` to F-contiguous f64 (no-op if already), call
    `_rust.demean_2d_weighted(...)`, return view + DemeanInfo list.
  - Else: delegate to `_weighted_ap_demean_numpy`.
- The two existing call sites in `fepois()` (lines 553-560) switch
  to the dispatcher with no signature change.

`K == 1` short-circuit stays Python (closed-form one-shot demean,
already in `_weighted_ap_demean`); FFI overhead would dominate for
that case.

### 5.4 FFI ownership table

| Data           | Owner   | Copies                                       |
|----------------|---------|----------------------------------------------|
| `arr` (z or X) | Python  | 0 if F-contig, else 1 `asfortranarray`       |
| `fe_codes[k]`  | Python  | 0 — read-only view                           |
| `wsum[k]`      | Python  | 0 — read-only view; recomputed each IRLS iter |
| `weights`      | Python  | 0 — read-only view                           |
| `scratch[k]`   | Rust    | Rust owns; per-thread allocation, sized G_k  |
| Aitken `hist`  | Rust    | Rust owns; up to 3 column-sized snapshots    |

### 5.5 Error handling

**Rust-side validation (raises `PyValueError`):**

| Check                                  | Trigger                      |
|----------------------------------------|------------------------------|
| `len(fe_codes) == len(wsum)`           | mismatched K                 |
| `weights.len() == n`                   | weights/x length mismatch    |
| `fe_codes[k].len() == n`               | per-FE length mismatch       |
| `x.is_fortran_contiguous()`            | not F-contig                 |
| `x.shape[0] == n`                      | shape mismatch               |

**Rust-side runtime guards (no validation, just safe semantics):**

- `if wsum[g] > 0.0 { scratch[g] /= wsum[g]; }` — empty weighted
  groups do not update; matches Python `np.divide(..., where=wsum>0)`.

**Python-side responsibilities (already enforced upstream in `fepois()`):**

- `weights >= 0` and finite (lines 419–426).
- `fe_codes` are dense int64 in `[0, G_k)`.
- `arr` ndim ∈ {1, 2}, dtype f64.

The dispatcher does **not** re-validate these — Rust trusts the
contract on this internal boundary.

### 5.6 Tests (Phase A merge gate)

**Rust unit tests** (`rust/statspai_hdfe/src/demean.rs#tests`):

1. `weighted_oneway_exact` — single FE, hand-computed weighted means
   match to atol 1e-12.
2. `weighted_twoway_converges` — balanced 2×2 panel; with `weights ≡ 1`
   the result matches the unweighted `demean_2d` path to atol 1e-12.
3. `weighted_unequal_weights` — non-balanced weights; verify Aitken
   safeguard fires when needed and never NaNs.
4. `weighted_zero_group` — synthetic group with `wsum = 0`; assert no
   panic, no NaN, group simply not updated.

**Python parity tests** (`tests/test_fast_fepois.py` extension):

5. `test_rust_weighted_demean_matches_numpy` — 5 random seeds, atol 1e-14.
6. `test_fepois_rust_path_coef_parity` — medium dataset; coef diff vs
   prior Python path < 1e-13, vs `pyfixest.fepois` < 1e-13, vs
   `fixest::fepois` < 1e-6.
7. `test_fepois_rust_path_se_parity` — IID / HC1 / CR1; SE diff vs
   prior Python path < 1e-10.
8. `test_fepois_rust_path_with_weights` — `weights="w"` with Rust
   path; coef parity vs `pyfixest.fepois(..., weights=)` < 1e-13.
9. `test_fepois_no_rust_fallback` — monkeypatch `_HAS_RUST = False`;
   confirm Python path runs and matches the Rust path.

**Regression suite (must all stay green; 208 currently):**

```bash
pytest tests/test_fast_fepois.py tests/test_fast_demean.py \
       tests/test_fast_within_dsl.py tests/test_fast_inference.py \
       tests/test_fast_polars.py tests/test_panel.py tests/test_fixest.py \
       tests/test_hdfe_native.py -q
```

### 5.7 Benchmark gate (Phase A merge blocker)

New file `benchmarks/hdfe/run_fepois_phase_a.py` runs `sp.fast.fepois`
on the medium dataset, reports wall time, and writes a JSON next to
the existing `feols_bench.json`.

| Backend                                           |      Target | Status           |
|---------------------------------------------------|------------:|------------------|
| R `fixest::fepois`                                |      0.64 s | baseline         |
| `sp.fast.fepois` (Phase 0 — Python demean)        |      2.61 s | baseline         |
| `sp.fast.fepois` (Phase A — Rust weighted demean) | **≤ 1.5 s** | **merge blocker**|

If Phase A wall-clock ≥ 1.5 s, the PR does **not** merge:

- record findings in `benchmarks/hdfe/AUDIT.md` ("Phase A round X")
- return to brainstorming to identify what assumption broke
- do not silently widen the threshold

### 5.8 Phase A versioning

- Rust crate: `0.2.0` → `0.3.0` (new public Rust function).
- Python package: bump to `1.8.0`. CHANGELOG entry under `## Performance`:
  > `sp.fast.fepois` weighted within-transform now runs on the Rust
  > kernel, closing ~50% of the wall-clock gap to `fixest::fepois`.

## 6. Phase B — Native Rust IRLS

### 6.1 Rust crate surface

```text
Cargo.toml
  // DEFAULT: no new dependency — implement a hand-coded k×k SPD Cholesky
  // + back-solve in pure Rust (~80 LOC). k ≤ ~30 in practice.
  // ESCAPE HATCH: if hand-coded Cholesky measurably underperforms vs faer
  // on the medium benchmark by ≥ 5% wall, add ``faer = "0.19"`` and switch.
  // The decision is recorded inline in the Phase B PR description; do not
  // pre-add the dependency unless the escape hatch fires.

irls.rs (new)
  + struct FePoisIRLSConfig {
        maxiter: u32,
        tol: f64,
        fe_tol: f64,
        fe_maxiter: u32,
        eta_clip: f64,         // default 30.0
        accel_period: u32,
        max_halvings: u32,     // default 10
    }
  + struct FePoisIRLSResult {
        beta: Vec<f64>,
        x_tilde: Vec<f64>,     // F-order (n, p), returned to Python for vcov
        w: Vec<f64>,           // final IRLS working weights
        eta: Vec<f64>,
        mu: Vec<f64>,
        deviance: f64,
        log_likelihood: f64,
        iters: u32,
        converged: bool,
        n_halvings: u32,
        n_inner_max_dx: f64,   // worst inner AP convergence across iters
    }
  + pub fn fepois_loop(
        y: &[f64], x: &[f64] /* F-order n×p */, n: usize, p: usize,
        fe_codes: &[&[i64]], counts: &[&[f64]] /* group sizes, used for
                                                  base-scale heuristics */,
        obs_weights: &[f64],
        config: &FePoisIRLSConfig,
    ) -> FePoisIRLSResult
        // wsum_list = bincount(codes_k, weights = mu * obs_weights) is
        // recomputed inside the IRLS body each iter (mu changes per iter,
        // so wsum cannot be seeded from a precomputed buffer).

lib.rs
  + #[pyfunction] fn fepois_irls(
        y, x, fe_codes, obs_weights, config_dict,
    ) -> dict
  - __version__: "0.3.0" → "0.4.0"
```

`fepois_loop` calls `weighted_demean_matrix_fortran_inplace` (the
Phase A crate-internal function, reused without going through PyO3),
allocating scratch and Aitken history once outside the iteration
loop and reusing them across all outer IRLS iters.

### 6.2 Python wiring

In `src/statspai/fast/fepois.py`:

- The IRLS body (lines 542–601) is replaced with a single call:
  ```python
  result = _rust.fepois_irls(
      y, X_F, fe_codes, obs_weights,
      {"maxiter": maxiter, "tol": tol, "fe_tol": fe_tol,
       "fe_maxiter": fe_maxiter, "eta_clip": 30.0,
       "accel_period": 5, "max_halvings": 10},
  )
  beta = result["beta"]
  X_tilde = result["x_tilde"]
  w = result["w"]
  # ... existing vcov code stays
  ```
- vcov computation (IID, HC1, CR1) **stays in Python**: it's one
  matrix multiply per type, ms-scale, and keeping it in Python lets
  the existing `crve` / `boottest` primitives stay unchanged.
- Pre-passes (singleton + separation) **stay in Python**: singleton
  detection already calls Rust (`statspai_hdfe.singleton_mask`);
  separation is Poisson-specific iterative pruning that's small
  relative to the IRLS body.
- Fallback (`_HAS_RUST = False`) keeps the post-Phase-A Python IRLS
  (which itself uses the Phase A Rust weighted demean if available,
  else the pure-Python path).

### 6.3 Phase B tests

**Numeric parity gate:**

- vs Phase A: coef atol ≤ 1e-12, SE atol ≤ 1e-10, all three vcov.
- vs `pyfixest.fepois`: coef atol ≤ 1e-13.
- vs `fixest::fepois`: coef atol ≤ 1e-6.

**Fuzz suite:**

- `(n ∈ {1k, 100k, 1M}) × (FE dim ∈ {1, 2, 3}) × (weights ∈ {none, random}) × (separation ∈ {none, induced}) × seed ∈ [0..20]`.
- Compare Rust IRLS output to Python fallback path; require coef
  atol ≤ 1e-12 across all combinations.

**Regression suite:** all v1.8.0 tests stay green.

### 6.4 Phase B benchmark gate

| Target                                 | Threshold |
|----------------------------------------|----------:|
| `sp.fast.fepois` (Phase B) wall, medium| **≤ 0.95 s** (≤ 1.5× fixest) |

Same rule as Phase A: missing the gate means the PR does not merge,
and the AUDIT.md gets a write-up.

### 6.5 Phase B versioning

- Rust crate: `0.3.0` → `0.4.0`.
- Python package: `1.8.0` → `1.8.1`. CHANGELOG:
  > `sp.fast.fepois` IRLS outer loop is now native Rust. Wall-clock
  > on the medium benchmark closes from ~2.3× of `fixest::fepois`
  > (post Phase A) to ≤ 1.5× of `fixest::fepois`.
  > `FePoisResult.backend` flips from
  > `"statspai-native"` to `"statspai-native-rust"`.

## 7. Risk register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------:|-------:|------------|
| Phase A wall-clock < 50% improvement (Python overhead larger than estimated) | low | medium | benchmark gate at 1.5 s — failure triggers re-brainstorm, not silent widening |
| Numeric drift in weighted demean due to subtle sweep-order difference | low | high | bit-for-bit reuse of Phase 1 algorithm + 5 parity tests at atol 1e-14 |
| faer adds wheel-build complexity on Windows / macOS / linux | medium | medium | Phase B kickoff evaluates faer vs hand-coded Cholesky; default to hand-coded if any wheel matrix breaks |
| Phase B non-convergence on previously-converging fits | low | high | fuzz suite (seeds 0..20, all FE dims, with/without separation) gates merge |
| `STATSPAI_SKIP_RUST=1` CI path silently breaks | medium | high | dedicated `test_fepois_no_rust_fallback` test; Phase A Python path stays as canonical reference |
| Phase A gets shipped but Phase B never lands (3-week project gets descoped) | medium | low | Phase A is independently valuable — it's a reusable primitive for `feols(weights=)` and future GLM families regardless of whether Phase B happens |

## 8. Out of scope (re-stated for clarity)

- Adding `weights` parameter to public `sp.fast.demean`.
- `sp.fast.feglm` (logit / negbin / gamma).
- Anderson(m) acceleration.
- GPU / CUDA paths.
- New cluster-robust SE wiring inside `fepois()` (CR2 / CR3 / wild
  bootstrap stay on the inference primitives).
- Refactoring the unweighted Phase 1 path.

## 9. Acceptance summary

This spec is implementable when both phases pass their gates:

**Phase A (v1.8.0) ships when:**

- 9 new tests pass (4 Rust unit + 5 Python parity).
- 208 existing tests still pass.
- `STATSPAI_SKIP_RUST=1` CI path stays green.
- Medium benchmark wall-clock ≤ 1.5 s.

**Phase B (v1.8.1) ships when:**

- All Phase A gates still pass.
- Fuzz suite (seeds 0..20 × FE dim × weights × separation) passes
  at coef atol ≤ 1e-12 vs Python fallback.
- Coef parity vs `pyfixest.fepois` ≤ 1e-13, vs `fixest::fepois` ≤ 1e-6.
- Medium benchmark wall-clock ≤ 0.95 s.

## 10. References

- Correia, S., Guimarães, P., Zylkin, T. (2020). "Fast Poisson
  estimation with high-dimensional fixed effects." *Stata Journal*
  20(1). DOI: 10.1177/1536867X20909691.
  `bib key: correia2020ppmlhdfe`
- Bergé, L. (2018). "Efficient estimation of maximum likelihood
  models with multiple fixed-effects: the R package FENmlm." CREA
  Discussion Paper 2018-13.
  `bib key: berge2018fenmlm`
- Varadhan, R., Roland, C. (2008). "Simple and globally convergent
  methods for accelerating the convergence of any EM algorithm."
  *Scandinavian Journal of Statistics* 35(2), 335-353.
  DOI: 10.1111/j.1467-9469.2007.00585.x.
  `bib key: varadhan2008squarem`
- Existing project artifacts:
  `benchmarks/hdfe/SUMMARY.md`, `benchmarks/hdfe/AUDIT.md`,
  `benchmarks/hdfe/PHASE2_VERIFY.md`,
  `src/statspai/fast/fepois.py`,
  `rust/statspai_hdfe/src/demean.rs`,
  `rust/statspai_hdfe/src/lib.rs`.

> All cited references must be cross-checked against Crossref / DOI
> before they enter `paper.bib`. Per project red line (CLAUDE.md §10),
> bib keys above are **proposed** — verification happens at PR time,
> not now.
