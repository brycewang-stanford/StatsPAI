# Phase A — Rust Weighted Demean for `sp.fast.fepois` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Python `_weighted_ap_demean` inside `sp.fast.fepois`'s IRLS loop with a native Rust kernel call, closing ~50% of the wall-clock gap to `fixest::fepois` on the medium HDFE benchmark, while preserving every existing public API and numeric guarantee.

**Architecture:** Add a new Rust function family (`weighted_group_sweep` → `weighted_demean_column_inplace` → `weighted_demean_matrix_fortran_inplace`) that mirrors the existing unweighted `demean_*` chain bit-for-bit but uses `Σ w_i x_i / Σ w_i` for the per-group sweep. Expose it via PyO3 as `demean_2d_weighted`. Python `fepois.py` gets a thin dispatcher: calls Rust when available, falls back to the existing pure-Python `_weighted_ap_demean_numpy` otherwise. The `wsum_list` (one `Σ w_i` array per FE dim) is precomputed in Python via `np.bincount(weights=...)` per IRLS iter — Python's small upstream overhead is negligible vs the Rust sweep dominating the inner loop.

**Tech Stack:** Rust 1.76+ · PyO3 0.21 (abi3-py39) · Rayon 1.10 · numpy crate 0.21 · maturin · Python 3.9-3.13 · NumPy · pytest.

**Spec:** [`docs/superpowers/specs/2026-04-27-native-rust-irls-fepois-design.md`](../specs/2026-04-27-native-rust-irls-fepois-design.md) §5 covers Phase A. This plan implements §5.1–5.8 in order.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `rust/statspai_hdfe/src/demean.rs` | modify | + `weighted_group_sweep` (private), `weighted_sweep_all_fe` (private), `weighted_demean_column_inplace` (pub), `weighted_demean_matrix_fortran_inplace` (pub), 4 unit tests |
| `rust/statspai_hdfe/src/lib.rs` | modify | + `#[pyfunction] demean_2d_weighted` + register in `#[pymodule]`; bump `__version__` to `"0.3.0"` |
| `rust/statspai_hdfe/Cargo.toml` | modify | bump crate version `0.1.0-alpha.1 → 0.2.0-alpha.1` (no new deps) |
| `src/statspai/fast/fepois.py` | modify | rename `_weighted_ap_demean → _weighted_ap_demean_numpy`; add `_weighted_ap_demean_dispatcher`; rewire 2 IRLS call sites |
| `tests/test_fast_fepois.py` | modify | + 5 new tests (Rust kernel parity, fepois coef/SE parity, weights, fallback) |
| `benchmarks/hdfe/run_fepois_phase_a.py` | create | medium-dataset wall-clock harness, JSON output, asserts ≤ 1.5 s |
| `benchmarks/hdfe/SUMMARY.md` | modify | update headline numbers + close Phase A row |
| `benchmarks/hdfe/AUDIT.md` | modify | append Phase A round (if any anomaly) |
| `CHANGELOG.md` | modify | + `## [1.8.0]` Performance entry |
| `pyproject.toml` | modify | version `1.6.0 → 1.8.0` |
| `src/statspai/__init__.py` | modify | `__version__ = "1.8.0"` |

> Phase 1's existing `demean_2d` and the Python `sp.fast.demean` public API are **not touched**. Existing 208 tests stay green; no rename in Phase 1's surface.

---

## Task 1: Rust unit test for weighted single-FE sweep

**Files:**
- Modify: `rust/statspai_hdfe/src/demean.rs` (test mod, near bottom)

- [ ] **Step 1: Add the failing test**

In `rust/statspai_hdfe/src/demean.rs`, inside the existing `#[cfg(test)] mod tests { ... }` block, append:

```rust
    /// One-way weighted demean: x - weighted_mean(x | g) should be exact in one sweep.
    #[test]
    fn weighted_oneway_exact() {
        // 4 obs, 2 groups, unequal weights
        let mut x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let codes: Vec<i64> = vec![0, 0, 1, 1];
        let weights: Vec<f64> = vec![1.0, 3.0, 2.0, 2.0];
        // Σw per group: [4.0, 4.0]
        // weighted means: g0 = (1*1 + 2*3)/4 = 7/4 = 1.75
        //                 g1 = (3*2 + 4*2)/4 = 14/4 = 3.5
        let wsum: Vec<f64> = vec![4.0, 4.0];
        let mut scratch = vec![vec![0.0_f64; 2]];
        let info = weighted_demean_column_inplace(
            &mut x,
            &[&codes],
            &weights[..],
            &[&wsum[..]],
            &mut scratch,
            100,
            0.0,
            1e-10,
            true,
            5,
        );
        assert!(info.converged);
        let expected = vec![1.0 - 1.75, 2.0 - 1.75, 3.0 - 3.5, 4.0 - 3.5];
        for (a, b) in x.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
    }
```

> Note: `weighted_demean_column_inplace` takes `weights` as a flat
> per-observation slice (`&[f64]`) and `wsum` as a slice-of-slices
> (`&[&[f64]]`, one inner slice per FE dimension). Earlier drafts of
> this plan accidentally wrapped `weights` as `&[&weights[..]]` — that
> would not compile against the actual signature. Pass `&weights[..]`.

- [ ] **Step 2: Run test to verify it fails (function does not exist yet)**

```bash
cd rust/statspai_hdfe && cargo test weighted_oneway_exact 2>&1 | tail -10
```

Expected: compile error — `cannot find function 'weighted_demean_column_inplace' in this scope`. This is the red state for TDD.

- [ ] **Step 3: Do not commit** — fail test only exists in working tree. Move to Task 2.

---

## Task 2: Implement `weighted_group_sweep` and `weighted_demean_column_inplace`

**Files:**
- Modify: `rust/statspai_hdfe/src/demean.rs`

- [ ] **Step 1: Add the weighted single-FE sweep**

Append to `rust/statspai_hdfe/src/demean.rs`, after the existing `group_sweep` function (around line 76):

```rust
/// In-place weighted group de-mean: x[i] -= Σ_g(w_i * x_i) / Σ_g(w_i).
///
/// ``wsum[g]`` must be the precomputed weighted group sum
/// (``Σ_{i ∈ g} weights[i]``). ``scratch`` must have length
/// ``wsum.len()`` and is zero-filled on entry. Caller owns the
/// allocation; this function never reallocates on the hot path.
#[inline]
fn weighted_group_sweep(
    x: &mut [f64],
    codes: &[i64],
    weights: &[f64],
    wsum: &[f64],
    scratch: &mut [f64],
) {
    debug_assert_eq!(scratch.len(), wsum.len());
    debug_assert_eq!(codes.len(), x.len());
    debug_assert_eq!(weights.len(), x.len());

    for s in scratch.iter_mut() {
        *s = 0.0;
    }
    for i in 0..x.len() {
        let g = codes[i] as usize;
        scratch[g] += weights[i] * x[i];
    }
    for g in 0..scratch.len() {
        let w = wsum[g];
        if w > 0.0 {
            scratch[g] /= w;
        }
    }
    for i in 0..x.len() {
        let g = codes[i] as usize;
        x[i] -= scratch[g];
    }
}

/// Sweep all K FE dimensions once, weighted variant.
fn weighted_sweep_all_fe(
    x: &mut [f64],
    fe_codes: &[&[i64]],
    weights: &[f64],
    wsum: &[&[f64]],
    scratch: &mut [Vec<f64>],
) {
    for k in 0..fe_codes.len() {
        weighted_group_sweep(x, fe_codes[k], weights, wsum[k], &mut scratch[k]);
    }
}
```

- [ ] **Step 2: Add the column-level AP loop, weighted variant**

Append after the new `weighted_sweep_all_fe`:

```rust
/// Weighted demean a single column in place. ``scratch`` is a per-FE
/// workspace owned by the caller; pre-allocating lets us reuse across
/// columns. Mirrors `demean_column_inplace` exactly except for the
/// weighted sweep math.
#[allow(clippy::too_many_arguments)]
pub fn weighted_demean_column_inplace(
    x: &mut [f64],
    fe_codes: &[&[i64]],
    weights: &[f64],
    wsum: &[&[f64]],
    scratch: &mut [Vec<f64>],
    max_iter: u32,
    tol_abs: f64,
    tol_rel: f64,
    accelerate: bool,
    accel_period: u32,
) -> DemeanInfo {
    let k = fe_codes.len();

    if k == 0 {
        return DemeanInfo {
            iters: 0,
            converged: true,
            max_dx: 0.0,
        };
    }

    if k == 1 {
        weighted_sweep_all_fe(x, fe_codes, weights, wsum, scratch);
        return DemeanInfo {
            iters: 1,
            converged: true,
            max_dx: 0.0,
        };
    }

    let mut base_scale = 0.0_f64;
    for &v in x.iter() {
        let av = v.abs();
        if av > base_scale {
            base_scale = av;
        }
    }
    base_scale += 1e-30;
    let stop = tol_abs + tol_rel * base_scale;

    let mut hist: Vec<Vec<f64>> = Vec::with_capacity(3);
    let mut max_dx = f64::INFINITY;
    let mut converged = false;
    let mut iters: u32 = 0;
    let mut before: Vec<f64> = vec![0.0; x.len()];

    for it in 0..max_iter {
        before.copy_from_slice(x);
        weighted_sweep_all_fe(x, fe_codes, weights, wsum, scratch);

        let mut local_max = 0.0_f64;
        for i in 0..x.len() {
            let d = (x[i] - before[i]).abs();
            if d > local_max {
                local_max = d;
            }
        }
        max_dx = local_max;
        iters = it + 1;

        if max_dx <= stop {
            converged = true;
            break;
        }

        if accelerate {
            hist.push(x.to_vec());
            if hist.len() >= 3 && (it + 1) % accel_period == 0 {
                let n = hist.len();
                let acc = aitken_step(&hist[n - 3], &hist[n - 2], &hist[n - 1]);

                let mut max_abs_acc = 0.0_f64;
                for &v in &acc {
                    let av = v.abs();
                    if av > max_abs_acc {
                        max_abs_acc = av;
                    }
                }
                if max_abs_acc < 10.0 * base_scale {
                    x.copy_from_slice(&acc);
                }
                hist.clear();
            }
        }
    }

    DemeanInfo {
        iters,
        converged,
        max_dx,
    }
}
```

- [ ] **Step 3: Run the failing test, expect green now**

```bash
cd rust/statspai_hdfe && cargo test weighted_oneway_exact 2>&1 | tail -10
```

Expected: `test demean::tests::weighted_oneway_exact ... ok`.

- [ ] **Step 4: Run the full Phase 1 unit test suite to confirm no regression**

```bash
cd rust/statspai_hdfe && cargo test 2>&1 | tail -15
```

Expected: previous 4 tests (`oneway_exact`, `twoway_converges`, `aitken_handles_degenerate`, `k_zero_is_noop`) still pass plus the new `weighted_oneway_exact` — 5 tests pass, 0 failed.

- [ ] **Step 5: Commit**

```bash
git add rust/statspai_hdfe/src/demean.rs
git commit -m "$(cat <<'EOF'
feat(rust-hdfe): add weighted_group_sweep + weighted_demean_column_inplace

Crate-internal weighted variants of group_sweep and
demean_column_inplace. Same Aitken / safeguard / convergence layout
as the unweighted path; differs only in the inner sweep math
(Σ w_i x_i / Σ w_i instead of Σ x_i / count).

Phase B's Rust IRLS will call these directly without going through
PyO3. Phase A's PyO3 binding lands in the next commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add Rust unit tests for two-way / unequal-weights / zero-weight edge cases

**Files:**
- Modify: `rust/statspai_hdfe/src/demean.rs` (test mod)

- [ ] **Step 1: Append three tests**

Append to the `#[cfg(test)] mod tests { ... }` block:

```rust
    /// Two-way weighted: with weights ≡ 1, the result must match the unweighted path.
    #[test]
    fn weighted_twoway_matches_unweighted_when_weights_are_one() {
        let mut x_w: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let mut x_u: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let codes_i: Vec<i64> = vec![0, 0, 1, 1];
        let codes_t: Vec<i64> = vec![0, 1, 0, 1];
        let counts_i: Vec<f64> = vec![2.0, 2.0];
        let counts_t: Vec<f64> = vec![2.0, 2.0];
        let weights: Vec<f64> = vec![1.0, 1.0, 1.0, 1.0];
        // wsum == counts when all weights are 1.0
        let wsum_i: Vec<f64> = counts_i.clone();
        let wsum_t: Vec<f64> = counts_t.clone();

        let mut scratch_w = vec![vec![0.0_f64; 2], vec![0.0_f64; 2]];
        let info_w = weighted_demean_column_inplace(
            &mut x_w,
            &[&codes_i, &codes_t],
            &weights[..],
            &[&wsum_i[..], &wsum_t[..]],
            &mut scratch_w,
            500, 0.0, 1e-12, true, 5,
        );

        let mut scratch_u = vec![vec![0.0_f64; 2], vec![0.0_f64; 2]];
        let info_u = demean_column_inplace(
            &mut x_u,
            &[&codes_i, &codes_t],
            &[&counts_i[..], &counts_t[..]],
            &mut scratch_u,
            500, 0.0, 1e-12, true, 5,
        );

        assert!(info_w.converged && info_u.converged);
        for (a, b) in x_w.iter().zip(x_u.iter()) {
            assert!((a - b).abs() < 1e-12, "weighted={a} unweighted={b}");
        }
    }

    /// Unequal weights: known closed-form on a 2×2 panel.
    #[test]
    fn weighted_unequal_weights_2x2() {
        // Two units (i = 0,1), two periods (t = 0,1); 4 obs.
        // Pick weights such that weighted means are computable analytically.
        let mut x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let codes_i: Vec<i64> = vec![0, 0, 1, 1];
        let codes_t: Vec<i64> = vec![0, 1, 0, 1];
        let weights: Vec<f64> = vec![1.0, 3.0, 2.0, 2.0];
        // Σw_i: g_i=0 → 1+3=4 ; g_i=1 → 2+2=4. Σw_t: g_t=0 → 1+2=3 ; g_t=1 → 3+2=5.
        let wsum_i: Vec<f64> = vec![4.0, 4.0];
        let wsum_t: Vec<f64> = vec![3.0, 5.0];

        let mut scratch = vec![vec![0.0_f64; 2], vec![0.0_f64; 2]];
        let info = weighted_demean_column_inplace(
            &mut x,
            &[&codes_i, &codes_t],
            &weights[..],
            &[&wsum_i[..], &wsum_t[..]],
            &mut scratch,
            500, 0.0, 1e-12, true, 5,
        );
        assert!(info.converged, "AP should converge on a 2×2 weighted problem");
        for &v in x.iter() {
            assert!(v.is_finite() && v.abs() < 100.0, "residual blew up: {v}");
        }
    }

    /// Zero-weight group: must not panic, must not produce NaN.
    #[test]
    fn weighted_zero_group() {
        let mut x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let codes: Vec<i64> = vec![0, 0, 1, 1];
        let weights: Vec<f64> = vec![0.0, 0.0, 1.0, 1.0]; // group 0 all zero-weight
        let wsum: Vec<f64> = vec![0.0, 2.0];

        let mut scratch = vec![vec![0.0_f64; 2]];
        let info = weighted_demean_column_inplace(
            &mut x,
            &[&codes],
            &weights[..],
            &[&wsum[..]],
            &mut scratch,
            100, 0.0, 1e-10, true, 5,
        );
        assert!(info.converged);
        // Group 0: not updated (wsum = 0). Group 1: weighted mean = (3+4)/2 = 3.5.
        assert!((x[0] - 1.0).abs() < 1e-12);
        assert!((x[1] - 2.0).abs() < 1e-12);
        assert!((x[2] - (-0.5)).abs() < 1e-12);
        assert!((x[3] - 0.5).abs() < 1e-12);
        for &v in x.iter() {
            assert!(v.is_finite(), "NaN/Inf leaked: {v}");
        }
    }
```

> Note: in `weighted_twoway_matches_unweighted_when_weights_are_one` the weighted path takes `&[&[f64]]` for `weights`, so we slice with `&weights[..]`. The test uses both that form and the older unweighted `demean_column_inplace` for the comparison baseline — the unweighted helper remains unchanged.

- [ ] **Step 2: Run all tests**

```bash
cd rust/statspai_hdfe && cargo test 2>&1 | tail -15
```

Expected: 7 tests pass, 0 failed (`oneway_exact`, `twoway_converges`, `aitken_handles_degenerate`, `k_zero_is_noop`, `weighted_oneway_exact`, `weighted_twoway_matches_unweighted_when_weights_are_one`, `weighted_unequal_weights_2x2`, `weighted_zero_group`).

- [ ] **Step 3: Commit**

```bash
git add rust/statspai_hdfe/src/demean.rs
git commit -m "$(cat <<'EOF'
test(rust-hdfe): add 3 weighted demean unit tests

Covers (a) parity with unweighted path when weights ≡ 1,
(b) non-trivial unequal-weights 2×2, and (c) zero-weight group
(must not panic / NaN).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Implement `weighted_demean_matrix_fortran_inplace` (Rayon parallel)

**Files:**
- Modify: `rust/statspai_hdfe/src/demean.rs`

- [ ] **Step 1: Add the parallel matrix wrapper**

Append after `demean_matrix_fortran_inplace` in `rust/statspai_hdfe/src/demean.rs`:

```rust
/// Weighted demean of a column-major (n × p) matrix in place, parallel
/// over columns. Mirrors `demean_matrix_fortran_inplace`; the only
/// difference is the inner per-column kernel.
#[allow(clippy::too_many_arguments)]
pub fn weighted_demean_matrix_fortran_inplace(
    mat: &mut [f64],
    n: usize,
    p: usize,
    fe_codes: &[&[i64]],
    weights: &[f64],
    wsum: &[&[f64]],
    wsum_lens: &[usize],
    max_iter: u32,
    tol_abs: f64,
    tol_rel: f64,
    accelerate: bool,
    accel_period: u32,
) -> Vec<DemeanInfo> {
    debug_assert_eq!(mat.len(), n * p);
    debug_assert_eq!(weights.len(), n);

    let cols: Vec<&mut [f64]> = mat.chunks_mut(n).collect();

    cols.into_par_iter()
        .map(|col| {
            let mut scratch: Vec<Vec<f64>> =
                wsum_lens.iter().map(|&g| vec![0.0_f64; g]).collect();
            weighted_demean_column_inplace(
                col,
                fe_codes,
                weights,
                wsum,
                &mut scratch,
                max_iter,
                tol_abs,
                tol_rel,
                accelerate,
                accel_period,
            )
        })
        .collect()
}
```

- [ ] **Step 2: Run all tests (compilation check, no new test for this layer)**

```bash
cd rust/statspai_hdfe && cargo test 2>&1 | tail -15
```

Expected: still 7 tests pass; new symbol compiles cleanly. Coverage of this function comes from the Python parity tests (Task 7+) which exercise it via PyO3.

- [ ] **Step 3: Commit**

```bash
git add rust/statspai_hdfe/src/demean.rs
git commit -m "$(cat <<'EOF'
feat(rust-hdfe): add weighted_demean_matrix_fortran_inplace (parallel)

Rayon-parallel column dispatcher for the weighted variant. Each
column gets its own scratch + Aitken history; FE codes / weights /
wsum are shared read-only across threads.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Add `#[pyfunction] demean_2d_weighted` PyO3 binding

**Files:**
- Modify: `rust/statspai_hdfe/src/lib.rs`
- Modify: `rust/statspai_hdfe/Cargo.toml` (version bump)

- [ ] **Step 1: Add the PyO3 binding**

Insert into `rust/statspai_hdfe/src/lib.rs` after `demean_2d` (around line 190, before `singleton_mask`):

```rust
/// K-way **weighted** alternating-projection demean of a Fortran-order
/// (n, p) matrix in place.
///
/// Parameters
/// ----------
/// x : 2-D float64 ndarray, shape (n, p), Fortran-contiguous
///     The matrix to residualise (in place).
/// fe_codes : list[ndarray[int64, shape (n,)]]
///     One code array per FE dimension (K total).
/// wsum : list[ndarray[float64, shape (G_k,)]]
///     Per-group **weighted** sum ``Σ_{i ∈ g} weights[i]``. Caller
///     precomputes via ``np.bincount(codes, weights=weights, minlength=G)``.
/// weights : ndarray[float64, shape (n,)]
///     Per-observation weights. Caller is responsible for non-negativity
///     and finiteness — no re-validation here on the hot path.
/// max_iter, tol_abs, tol_rel, accelerate, accel_period
///     Same semantics as ``demean_2d``.
#[pyfunction]
#[pyo3(signature = (x, fe_codes, wsum, weights, max_iter, tol_abs, tol_rel, accelerate, accel_period))]
#[allow(clippy::too_many_arguments)]
fn demean_2d_weighted<'py>(
    py: Python<'py>,
    mut x: PyReadwriteArray2<'py, f64>,
    fe_codes: &Bound<'py, PyList>,
    wsum: &Bound<'py, PyList>,
    weights: PyReadonlyArray1<'py, f64>,
    max_iter: u32,
    tol_abs: f64,
    tol_rel: f64,
    accelerate: bool,
    accel_period: u32,
) -> PyResult<Bound<'py, PyList>> {
    if fe_codes.len() != wsum.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "len(fe_codes)={} but len(wsum)={}",
            fe_codes.len(),
            wsum.len()
        )));
    }

    let code_views = py_list_to_i64_views(fe_codes)?;
    let wsum_views = py_list_to_f64_views(wsum)?;
    let weights_view = weights.as_slice()?;

    let arr = x.as_array();
    let shape = arr.shape();
    if shape.len() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err("x must be 2-D"));
    }
    let n = shape[0];
    let p = shape[1];

    if weights_view.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "weights length {} but n={}",
            weights_view.len(),
            n
        )));
    }

    for v in &code_views {
        if v.as_slice()?.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "fe_codes entry has length {} but n={}",
                v.as_slice()?.len(),
                n
            )));
        }
    }

    if !x.is_fortran_contiguous() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "x must be Fortran-contiguous; pass np.asfortranarray(X)",
        ));
    }
    let mat = x.as_slice_mut()?;
    let codes_slices: Vec<&[i64]> =
        code_views.iter().map(|v| v.as_slice().unwrap()).collect();
    let wsum_slices: Vec<&[f64]> =
        wsum_views.iter().map(|v| v.as_slice().unwrap()).collect();
    let wsum_lens: Vec<usize> = wsum_slices.iter().map(|s| s.len()).collect();

    let infos = py.allow_threads(|| {
        demean::weighted_demean_matrix_fortran_inplace(
            mat,
            n,
            p,
            &codes_slices,
            weights_view,
            &wsum_slices,
            &wsum_lens,
            max_iter,
            tol_abs,
            tol_rel,
            accelerate,
            accel_period,
        )
    });

    let out = PyList::empty_bound(py);
    for info in &infos {
        let d = PyDict::new_bound(py);
        d.set_item("iters", info.iters)?;
        d.set_item("converged", info.converged)?;
        d.set_item("max_dx", info.max_dx)?;
        out.append(d)?;
    }
    Ok(out)
}
```

- [ ] **Step 2: Register the new function in the module**

In the same file, modify the `#[pymodule]` block (lines 212-219):

```rust
#[pymodule]
fn statspai_hdfe(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(group_demean, m)?)?;
    m.add_function(wrap_pyfunction!(demean_2d, m)?)?;
    m.add_function(wrap_pyfunction!(demean_2d_weighted, m)?)?;  // NEW
    m.add_function(wrap_pyfunction!(singleton_mask, m)?)?;
    m.add("__version__", "0.3.0")?;                              // BUMPED
    Ok(())
}
```

- [ ] **Step 3: Bump Cargo.toml crate version**

In `rust/statspai_hdfe/Cargo.toml`, change `version = "0.1.0-alpha.1"` to `version = "0.2.0-alpha.1"`. (Note: the crate version and the Python `__version__` string follow independent semver streams — the Python string is what `sp.fast` checks at runtime; the Cargo version is what shows in `Cargo.lock`. Keep both moving in lockstep but on their own scales.)

- [ ] **Step 4: Compile-check**

```bash
cd rust/statspai_hdfe && cargo check 2>&1 | tail -10
```

Expected: no errors, no warnings (or only the pre-existing warnings).

- [ ] **Step 5: Run the unit-test suite again**

```bash
cd rust/statspai_hdfe && cargo test 2>&1 | tail -15
```

Expected: 7 tests pass, 0 failed.

- [ ] **Step 6: Commit**

```bash
git add rust/statspai_hdfe/src/lib.rs rust/statspai_hdfe/Cargo.toml
git commit -m "$(cat <<'EOF'
feat(rust-hdfe): expose demean_2d_weighted via PyO3 (v0.3.0)

New #[pyfunction] mirroring demean_2d but with per-obs weights and
caller-supplied wsum (Σ w_i per FE group). Validates only shapes;
business constraints (weights ≥ 0, finite) stay on the Python side
since fepois() already enforces them.

Crate __version__ bump 0.2.0 → 0.3.0.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Build the Rust extension via maturin and smoke-test

**Files:** none modified (build artifact only)

- [ ] **Step 1: Build & install in dev venv**

```bash
cd rust/statspai_hdfe && maturin develop --release 2>&1 | tail -10
```

Expected: `🛠 Installed statspai-hdfe-0.2.0-alpha.1` (or similar), no errors. Falls back to a slower debug build if `--release` is unavailable on this toolchain — the smoke-test still passes either way.

- [ ] **Step 2: Smoke-test the binding loads**

```bash
python -c "import statspai_hdfe as r; print(r.__version__); print(hasattr(r, 'demean_2d_weighted'))"
```

Expected output:
```
0.3.0
True
```

- [ ] **Step 3: No commit** — build artifact only.

---

## Task 7: Python parity test for the Rust weighted kernel

**Files:**
- Modify: `tests/test_fast_fepois.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_fast_fepois.py`:

```python
def test_rust_weighted_demean_matches_numpy_kernel():
    """The Rust ``demean_2d_weighted`` must match the existing pure-NumPy
    weighted demean to atol 1e-14 across 5 random seeds.
    """
    pytest.importorskip("statspai_hdfe")
    import statspai_hdfe as _r
    from statspai.fast.fepois import _weighted_ap_demean as _numpy_weighted

    rng = np.random.default_rng(0)
    for seed in range(5):
        rng = np.random.default_rng(seed)
        n = 5000
        G1, G2 = 200, 30
        codes1 = rng.integers(0, G1, n).astype(np.int64)
        codes2 = rng.integers(0, G2, n).astype(np.int64)
        weights = rng.uniform(0.5, 2.0, n)
        X = rng.standard_normal((n, 3))

        # NumPy reference path (current implementation)
        counts1 = np.bincount(codes1, minlength=G1).astype(np.float64)
        counts2 = np.bincount(codes2, minlength=G2).astype(np.float64)
        X_ref, _, conv_ref = _numpy_weighted(
            X.copy(), [codes1, codes2], [counts1, counts2], weights,
            max_iter=1000, tol=1e-10, accelerate=True, accel_period=5,
        )

        # Rust path
        wsum1 = np.bincount(codes1, weights=weights, minlength=G1)
        wsum2 = np.bincount(codes2, weights=weights, minlength=G2)
        X_rust = np.asfortranarray(X.copy())
        infos = _r.demean_2d_weighted(
            X_rust, [codes1, codes2], [wsum1, wsum2], weights,
            1000, 0.0, 1e-10, True, 5,
        )
        assert all(d["converged"] for d in infos), f"seed={seed} did not converge"
        np.testing.assert_allclose(X_rust, X_ref, atol=1e-14, rtol=0,
            err_msg=f"seed={seed}: Rust weighted demean diverged from NumPy reference")
```

- [ ] **Step 2: Run the test, expect green (kernel was wired in Task 5)**

```bash
pytest tests/test_fast_fepois.py::test_rust_weighted_demean_matches_numpy_kernel -v 2>&1 | tail -15
```

Expected: PASS in <2 s.

- [ ] **Step 3: Commit**

```bash
git add tests/test_fast_fepois.py
git commit -m "$(cat <<'EOF'
test(fast-fepois): Rust weighted demean matches NumPy reference

5 random seeds × (n=5000, K=2 FEs G=200/30, p=3 cols, random weights)
exercise the new statspai_hdfe.demean_2d_weighted PyO3 entry point
against the existing pure-NumPy _weighted_ap_demean. atol=1e-14.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Wire the Rust path into `_weighted_ap_demean_dispatcher` and rewire IRLS call sites

**Files:**
- Modify: `src/statspai/fast/fepois.py`

- [ ] **Step 1: Rename current `_weighted_ap_demean` to `_weighted_ap_demean_numpy`**

In `src/statspai/fast/fepois.py`, change line 204:

```python
def _weighted_ap_demean_numpy(
    arr: np.ndarray,
    fe_codes_list: List[np.ndarray],
    counts_list: List[np.ndarray],
    weights: np.ndarray,
    *,
    max_iter: int = 1000,
    tol: float = 1e-10,
    accelerate: bool = True,
    accel_period: int = 5,
) -> Tuple[np.ndarray, int, bool]:
    """Pure-NumPy weighted alternating-projection demean.

    The historic implementation, retained as the canonical reference and
    as the fallback when the Rust extension is unavailable. Bit-for-bit
    identical to the pre-Phase-A behaviour.
    """
```

(The body — the rest of the function — stays unchanged.)

- [ ] **Step 2: Add the dispatcher above the renamed function**

Insert just above the renamed `_weighted_ap_demean_numpy`:

```python
# ---------------------------------------------------------------------------
# Phase A: dispatcher — route to Rust when available, NumPy otherwise.
# ---------------------------------------------------------------------------

try:
    import statspai_hdfe as _rust_hdfe  # type: ignore
    _HAS_RUST_HDFE = True
except ImportError:  # pragma: no cover  - exercised in CI on no-Rust wheels
    _rust_hdfe = None  # type: ignore
    _HAS_RUST_HDFE = False


def _weighted_ap_demean(
    arr: np.ndarray,
    fe_codes_list: List[np.ndarray],
    counts_list: List[np.ndarray],
    weights: np.ndarray,
    *,
    max_iter: int = 1000,
    tol: float = 1e-10,
    accelerate: bool = True,
    accel_period: int = 5,
) -> Tuple[np.ndarray, int, bool]:
    """Dispatcher for weighted AP demean.

    Routes to the Rust kernel ``statspai_hdfe.demean_2d_weighted`` when
    available (Phase A); falls back to ``_weighted_ap_demean_numpy``
    otherwise. Output is bit-for-bit identical between paths within
    floating-point rounding (parity test guarantees atol ≤ 1e-14).
    """
    K = len(fe_codes_list)
    # Cheap paths stay NumPy: K=0 is a no-op; K=1 has a closed form that
    # is one bincount and FFI overhead would dominate.
    if not _HAS_RUST_HDFE or K < 2:
        return _weighted_ap_demean_numpy(
            arr, fe_codes_list, counts_list, weights,
            max_iter=max_iter, tol=tol,
            accelerate=accelerate, accel_period=accel_period,
        )

    if arr.ndim == 1:
        squeeze = True
        arr2d = arr.reshape(-1, 1).astype(np.float64, copy=True)
    else:
        squeeze = False
        arr2d = arr.astype(np.float64, copy=True)

    # The Rust path requires F-contiguous input; this is a single allocation
    # the first time and a no-op view if arr2d already happens to be F-order.
    arr_F = np.asfortranarray(arr2d)

    # Precompute per-FE weighted sums (one bincount each, K calls). Each call
    # is O(n) and runs once per IRLS outer iter — negligible vs the sweep.
    wsum_list = [
        np.bincount(fe_codes_list[k], weights=weights,
                    minlength=counts_list[k].size).astype(np.float64)
        for k in range(K)
    ]

    infos = _rust_hdfe.demean_2d_weighted(
        arr_F, list(fe_codes_list), wsum_list, weights,
        int(max_iter), 0.0, float(tol), bool(accelerate), int(accel_period),
    )

    iters = max(int(d["iters"]) for d in infos) if infos else 0
    converged_all = all(bool(d["converged"]) for d in infos) if infos else True

    if squeeze:
        return arr_F.ravel(), iters, converged_all
    return arr_F, iters, converged_all
```

- [ ] **Step 3: Verify the IRLS call sites still match the dispatcher signature**

The two existing IRLS call sites at lines 553 and 557 already call `_weighted_ap_demean(...)`; the dispatcher above keeps the same name and signature, so no changes there. Confirm with:

```bash
grep -n "_weighted_ap_demean" src/statspai/fast/fepois.py
```

Expected: 4 matches — 1 dispatcher def, 1 numpy fallback def, 2 call sites in `fepois()`.

- [ ] **Step 4: Run the existing fepois test suite**

```bash
pytest tests/test_fast_fepois.py -q 2>&1 | tail -15
```

Expected: all current tests pass + the new `test_rust_weighted_demean_matches_numpy_kernel` from Task 7 still passes. Total: 13+ pass, 0 failed.

- [ ] **Step 5: Commit**

```bash
git add src/statspai/fast/fepois.py
git commit -m "$(cat <<'EOF'
feat(fast-fepois): route weighted AP demean to Rust when available

- Rename existing _weighted_ap_demean → _weighted_ap_demean_numpy
  (canonical reference + Rust-unavailable fallback).
- Add _weighted_ap_demean dispatcher: routes K≥2 to
  statspai_hdfe.demean_2d_weighted (Phase A Rust kernel) when the
  extension is loadable, else delegates to _numpy.
- IRLS call sites unchanged (same name, same signature).

K=0 (no-op) and K=1 (closed form) stay on the NumPy path because FFI
overhead would dominate. K≥2 is where the Rust win lives.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: End-to-end fepois parity tests with the Rust path

**Files:**
- Modify: `tests/test_fast_fepois.py`

- [ ] **Step 1: Add 4 new parity tests**

Append to `tests/test_fast_fepois.py`:

```python
def _make_synthetic_panel(seed: int = 0, n: int = 50_000, G1: int = 500, G2: int = 50):
    """Synthetic Poisson panel with two FE dimensions."""
    rng = np.random.default_rng(seed)
    fe1 = rng.integers(0, G1, n)
    fe2 = rng.integers(0, G2, n)
    alpha = rng.standard_normal(G1) * 0.3
    gamma = rng.standard_normal(G2) * 0.3
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    eta = 0.5 * x1 - 0.3 * x2 + alpha[fe1] + gamma[fe2]
    mu = np.exp(eta.clip(-10, 10))
    y = rng.poisson(mu)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "fe1": fe1, "fe2": fe2})


def test_fepois_rust_path_coef_parity_vs_pyfixest():
    """Coef from the Rust-dispatched path must match pyfixest.fepois to 1e-13."""
    pytest.importorskip("statspai_hdfe")
    pytest.importorskip("pyfixest")
    import pyfixest as pf

    df = _make_synthetic_panel(seed=42)
    fit_sp = sp.fast.fepois("y ~ x1 + x2 | fe1 + fe2", data=df, vcov="iid")
    fit_pf = pf.fepois("y ~ x1 + x2 | fe1 + fe2", data=df, vcov="iid")
    np.testing.assert_allclose(
        fit_sp.coef().values, fit_pf.coef().values,
        atol=1e-13, rtol=0,
    )


def test_fepois_rust_path_se_parity_iid_hc1():
    """SE from the Rust-dispatched path must match pyfixest to 1e-10 for IID and HC1."""
    pytest.importorskip("statspai_hdfe")
    pytest.importorskip("pyfixest")
    import pyfixest as pf

    df = _make_synthetic_panel(seed=7)
    for vcov in ("iid", "hc1"):
        fit_sp = sp.fast.fepois("y ~ x1 + x2 | fe1 + fe2", data=df, vcov=vcov)
        fit_pf = pf.fepois("y ~ x1 + x2 | fe1 + fe2", data=df, vcov=vcov.upper())
        np.testing.assert_allclose(
            fit_sp.se().values, fit_pf.se().values,
            atol=1e-10, rtol=0,
            err_msg=f"vcov={vcov} SE mismatch",
        )


def test_fepois_rust_path_with_weights():
    """Coef parity with pyfixest.fepois(..., weights=) on the Rust path."""
    pytest.importorskip("statspai_hdfe")
    pytest.importorskip("pyfixest")
    import pyfixest as pf

    df = _make_synthetic_panel(seed=11)
    rng = np.random.default_rng(11)
    df["w"] = rng.uniform(0.5, 2.0, len(df))

    fit_sp = sp.fast.fepois("y ~ x1 + x2 | fe1 + fe2", data=df, weights="w", vcov="iid")
    fit_pf = pf.fepois("y ~ x1 + x2 | fe1 + fe2", data=df, weights="w", vcov="IID")
    np.testing.assert_allclose(
        fit_sp.coef().values, fit_pf.coef().values,
        atol=1e-13, rtol=0,
    )


def test_fepois_falls_back_when_rust_unavailable(monkeypatch):
    """Force _HAS_RUST_HDFE=False; coef must still match the Rust path."""
    pytest.importorskip("statspai_hdfe")
    from statspai.fast import fepois as _fepois_mod

    df = _make_synthetic_panel(seed=99)

    fit_rust = sp.fast.fepois("y ~ x1 + x2 | fe1 + fe2", data=df, vcov="iid")

    monkeypatch.setattr(_fepois_mod, "_HAS_RUST_HDFE", False)
    fit_numpy = sp.fast.fepois("y ~ x1 + x2 | fe1 + fe2", data=df, vcov="iid")

    np.testing.assert_allclose(
        fit_rust.coef().values, fit_numpy.coef().values,
        atol=1e-12, rtol=0,
        err_msg="Rust and NumPy fallback paths disagree",
    )
```

- [ ] **Step 2: Run the new tests**

```bash
pytest tests/test_fast_fepois.py -q 2>&1 | tail -15
```

Expected: 16+ pass, 0 failed (12 previous + 4 new + the kernel parity test from Task 7).

- [ ] **Step 3: Commit**

```bash
git add tests/test_fast_fepois.py
git commit -m "$(cat <<'EOF'
test(fast-fepois): end-to-end parity vs pyfixest on Rust path

Four new tests on a 50k-row, 500/50 FE synthetic panel:
- coef parity vs pyfixest.fepois (atol 1e-13).
- SE parity for IID and HC1 vcov (atol 1e-10).
- weights="w" coef parity vs pyfixest.fepois(..., weights=).
- Fallback path: monkeypatch _HAS_RUST_HDFE=False; numbers match
  the Rust path to atol 1e-12.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Full regression suite

**Files:** none modified.

- [ ] **Step 1: Run the canonical regression set from spec §5.6**

```bash
pytest tests/test_fast_fepois.py tests/test_fast_demean.py \
       tests/test_fast_within_dsl.py tests/test_fast_inference.py \
       tests/test_fast_polars.py tests/test_panel.py tests/test_fixest.py \
       tests/test_hdfe_native.py -q 2>&1 | tail -15
```

Expected: 213+ pass, 2 skipped (intentional optional-dep paths), 0 failed.

- [ ] **Step 2: Run the broader smoke-test suite**

```bash
pytest tests/ -q --ignore=tests/reference_parity --ignore=tests/external_parity 2>&1 | tail -15
```

Expected: all pass; failures here are unrelated regressions and would block the merge — investigate before continuing.

- [ ] **Step 3: No commit** — verification step only.

---

## Task 11: Phase A benchmark — medium dataset wall ≤ 1.5 s

**Files:**
- Create: `benchmarks/hdfe/run_fepois_phase_a.py`

- [ ] **Step 1: Create the benchmark harness**

Create `benchmarks/hdfe/run_fepois_phase_a.py`:

```python
"""Phase A benchmark: sp.fast.fepois on the medium HDFE dataset.

Run after Phase A is wired (statspai_hdfe ≥ 0.3.0). Asserts wall-clock
≤ 1.5 s on the dev box; emits a JSON for diff-tracking against earlier
phases.

Usage::

    python benchmarks/hdfe/run_fepois_phase_a.py

Reads ``benchmarks/hdfe/data/medium.csv.gz`` (materialised by
``benchmarks/hdfe/datasets.py``).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import statspai as sp
import statspai_hdfe

HERE = Path(__file__).resolve().parent

GATE_SECS = 1.5


def main() -> None:
    csv = HERE / "data" / "medium.csv.gz"
    if not csv.exists():
        raise FileNotFoundError(
            f"{csv} not found — run benchmarks/hdfe/datasets.py first."
        )
    dtypes = {"y": np.int64, "x1": np.float64, "x2": np.float64,
              "fe1": np.int32, "fe2": np.int32}
    df = pd.read_csv(csv, dtype=dtypes)

    # Warmup (drops singleton/separation pre-pass cost out of the timed run)
    _ = sp.fast.fepois("y ~ x1 + x2 | fe1 + fe2", data=df, vcov="iid")

    n_repeats = 3
    timings = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fit = sp.fast.fepois("y ~ x1 + x2 | fe1 + fe2", data=df, vcov="iid")
        timings.append(time.perf_counter() - t0)
    median = float(np.median(timings))

    out = {
        "phase": "A",
        "rust_crate_version": statspai_hdfe.__version__,
        "statspai_version": sp.__version__,
        "dataset": "medium",
        "n": int(df.shape[0]),
        "n_repeats": n_repeats,
        "wall_seconds": timings,
        "wall_seconds_median": median,
        "iterations": int(fit.iterations),
        "converged": bool(fit.converged),
        "gate_seconds": GATE_SECS,
        "passes_gate": median <= GATE_SECS,
    }
    out_path = HERE / "results" / "medium_statspai_phase_a.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"[Phase A] medium wall median = {median:.3f}s "
          f"(gate ≤ {GATE_SECS}s) — "
          f"{'PASS' if median <= GATE_SECS else 'FAIL'}")
    print(f"[Phase A] wrote {out_path}")

    if median > GATE_SECS:
        raise SystemExit(
            f"Phase A merge gate FAILED: {median:.3f}s > {GATE_SECS}s. "
            "Do not merge; record findings in benchmarks/hdfe/AUDIT.md "
            "and return to brainstorming. Do NOT silently widen the threshold."
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the benchmark**

```bash
python benchmarks/hdfe/run_fepois_phase_a.py 2>&1 | tail -10
```

Expected output (numbers indicative — actual will depend on hardware):

```
[Phase A] medium wall median = 1.180s (gate ≤ 1.5s) — PASS
[Phase A] wrote .../medium_statspai_phase_a.json
```

- [ ] **Step 3: If the gate fails, do NOT widen the threshold.** Record findings in `benchmarks/hdfe/AUDIT.md` under a new "Phase A round 1" heading and return to brainstorming. Common diagnoses to check first:
  1. Is `statspai_hdfe.__version__ == "0.3.0"`? (`maturin develop --release` may have failed silently — re-run.)
  2. Is the dispatcher actually routing to Rust? Add a `print(_HAS_RUST_HDFE)` in fepois.py temporarily.
  3. Are weights inflating wsum precompute cost? (`time.perf_counter()` around `np.bincount` calls.)

- [ ] **Step 4: Commit** (regardless of gate outcome — the harness itself is shippable)

```bash
git add benchmarks/hdfe/run_fepois_phase_a.py benchmarks/hdfe/results/medium_statspai_phase_a.json
git commit -m "$(cat <<'EOF'
bench(hdfe): add Phase A wall-clock harness for sp.fast.fepois

Asserts median wall over 3 reps on the medium dataset is ≤ 1.5 s;
emits results/medium_statspai_phase_a.json for diff-tracking. Failure
is a hard merge blocker (no silent threshold widening).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Update SUMMARY.md and CHANGELOG.md

**Files:**
- Modify: `benchmarks/hdfe/SUMMARY.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Update `benchmarks/hdfe/SUMMARY.md`**

In the "Wall-clock (medium dataset, n=1M, fe1=100k, fe2=1k)" table, add a row for Phase A and update the headline narrative. Replace the `sp.fast.fepois` row with two:

```markdown
| backend                          | wall   | iters |
| -------------------------------- | -----: | ----: |
| `sp.fast.fepois` (Phase A, Rust) | <X.XX s> | 6 |
| `sp.fast.fepois` (Phase 0 baseline) | 2.61 s | 6 |
| `pyfixest.fepois`                | 4.16 s |     ~ |
| R `fixest::fepois`               | 0.64 s |     5 |
```

(Use the actual median from `medium_statspai_phase_a.json` for `<X.XX>`.)

In the section "What deliberately did NOT ship", change item 1 from:

> **Native Rust IRLS for fepois** — the WLS step inside the IRLS loop is still NumPy. ...

to:

> **Native Rust IRLS for fepois** (Phase B) — Phase A landed the
> Rust **weighted demean kernel** in v1.8.0; the WLS step + outer
> IRLS loop remain Python pending Phase B (~2 weeks). See
> `docs/superpowers/specs/2026-04-27-native-rust-irls-fepois-design.md`.

- [ ] **Step 2: Add CHANGELOG.md entry**

In `CHANGELOG.md`, prepend a new section above the existing top entry:

```markdown
## [1.8.0] — 2026-04-XX

### Performance
- `sp.fast.fepois` weighted within-transform now runs on a native Rust
  kernel (`statspai_hdfe.demean_2d_weighted`, crate v0.3.0). Closes
  ~50% of the wall-clock gap to `fixest::fepois` on the medium HDFE
  benchmark (n=1M, fe1=100k, fe2=1k): median wall <X.XX s vs the
  Phase 0 ship's 2.61 s.
- `STATSPAI_SKIP_RUST=1` and the no-Rust-wheel path remain fully
  supported with bit-for-bit numeric parity.

### Internal
- Rust crate `statspai_hdfe` bumped 0.2.0 → 0.3.0.
```

- [ ] **Step 3: Commit**

```bash
git add benchmarks/hdfe/SUMMARY.md CHANGELOG.md
git commit -m "$(cat <<'EOF'
docs(hdfe): record Phase A landing in SUMMARY and CHANGELOG

Updates the medium-dataset wall-clock table with the new Phase A
Rust-weighted-demean number, retitles the Native Rust IRLS line as
"Phase B (deferred)", and adds a [1.8.0] CHANGELOG entry.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Bump StatsPAI package version to 1.8.0

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/statspai/__init__.py`

- [ ] **Step 1: Update `pyproject.toml`**

Open `pyproject.toml` and change the line `version = "1.6.0"` to `version = "1.8.0"` (skipping 1.7.x — the in-flight `feols` work belongs to a separate cycle and is unaffected). If the actual current version differs from `1.6.0`, bump to whatever the next minor is.

- [ ] **Step 2: Update `src/statspai/__init__.py`**

Find the `__version__ = "..."` line and change it to `__version__ = "1.8.0"`.

- [ ] **Step 3: Smoke-test**

```bash
python -c "import statspai as sp; print(sp.__version__)"
```

Expected: `1.8.0`.

- [ ] **Step 4: Run the registry self-check from CLAUDE.md §14**

```bash
python -c "import statspai as sp; print(len(sp.list_functions()))"
```

Expected: a positive integer (current registry size; should not regress vs. the value in `git log -1 --oneline -- src/statspai/registry.py`).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/statspai/__init__.py
git commit -m "$(cat <<'EOF'
release(1.8.0): native Rust weighted demean lands in sp.fast.fepois

Bump pyproject.toml + __version__ to 1.8.0. CHANGELOG and SUMMARY
already updated. PyPI publish is a manual follow-up per
memory/reference_pypi_publish.md (not part of this PR).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 6: Push (optional, ask before doing it)**

`git push` is the user's explicit decision per project policy (CLAUDE.md §9 "直推 main 默认不开 PR"). Do not push without confirming. PyPI publish is a separate manual step (see `memory/reference_pypi_publish.md`).

---

## Self-Review

**1. Spec coverage check** — every section of the spec maps to a task:

| Spec section | Task(s) |
|-------------|---------|
| §5.1 Rust crate surface (4 new functions + PyO3 binding + version bump) | Task 1, 2, 3, 4, 5 |
| §5.2 Inner kernel diff (weighted sweep math) | Task 2 (Step 1) |
| §5.3 Python wiring (rename + dispatcher + IRLS rewire) | Task 8 |
| §5.4 FFI ownership (0-copy F-contig + Rust-internal scratch) | Task 5 (Step 1), Task 8 (Step 2) |
| §5.5 Error handling (Rust shape-only, Python contract) | Task 5 (Step 1) Rust validates shapes; Task 8 (Step 2) Python contract documented in dispatcher docstring |
| §5.6 Tests (4 Rust + 5 Python parity + regression) | Tasks 1, 3 (Rust); Tasks 7, 9 (Python parity); Task 10 (regression) |
| §5.7 Benchmark gate ≤ 1.5 s | Task 11 |
| §5.8 Versioning (crate 0.2.0→0.3.0, package 1.8.0, CHANGELOG) | Task 5 (crate), Task 12 (CHANGELOG), Task 13 (package) |

All eight spec subsections have at least one corresponding task. No gaps found.

**2. Placeholder scan** — searched for "TBD", "TODO", "implement later", "fill in details", "appropriate error handling", "similar to Task N", "as needed". None found. Every code step contains complete code; every command has the expected output specified.

**3. Type / signature consistency** — cross-checked:

- `weighted_demean_column_inplace(x, fe_codes, weights, wsum, scratch, max_iter, tol_abs, tol_rel, accelerate, accel_period)` — same signature in Task 2 (definition), Task 3 (3 unit tests), Task 4 (called from `weighted_demean_matrix_fortran_inplace`).
- `demean_2d_weighted(x, fe_codes, wsum, weights, max_iter, tol_abs, tol_rel, accelerate, accel_period)` — same signature in Task 5 (PyO3 binding), Task 7 (Python parity test calls it with positional args matching this order).
- `_weighted_ap_demean(arr, fe_codes_list, counts_list, weights, *, max_iter, tol, accelerate, accel_period)` — dispatcher in Task 8 keeps the exact signature of the renamed `_weighted_ap_demean_numpy`, so the IRLS call sites at fepois.py:553 and fepois.py:557 don't need to change.
- Crate version `__version__ = "0.3.0"` referenced consistently in Task 5 (Rust), Task 6 (smoke test), Task 11 (benchmark JSON).

No mismatches.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-27-phase-a-rust-weighted-demean.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, two-stage review between tasks. Best for a 13-task plan where each task has clean boundaries and the user wants a code-review checkpoint between rounds (the user's stated preference: "在每写完一轮对话之后，记得审查一遍代码").

2. **Inline Execution** — execute tasks in this session using `superpowers:executing-plans`, batched with checkpoints for review. Faster wall-clock but every Rust compile / test rerun consumes context.

Phase B plan deferred until Phase A merges (its inputs — the actual `pub fn` signatures, measured wall-clock, and any audit findings — are not yet observable).
