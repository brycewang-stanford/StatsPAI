# Phase B — Native Rust IRLS for `sp.fast.fepois` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the wall-clock gap between `sp.fast.fepois` and R `fixest::fepois` from the post-Phase-A 3.83× to ≤ 1.5× (target: medium wall ≤ 0.95 s vs fixest's 0.64 s) via two algorithmic + architectural changes — **sort observations by primary FE code** for sequential L1-cache-friendly sweeps, and **port the IRLS outer loop into Rust** to eliminate 12 FFI round-trips per `fepois` call.

**Architecture:** Two-stage delivery designed to mitigate the failure mode that broke Phase A — committing weeks of work to an unvalidated speedup assumption. Stage **B0** (spike, ~1-3 days) validates sort-by-primary-FE in isolation by adding a sort-aware variant of `weighted_demean_matrix_fortran_inplace` and re-running the Phase A benchmark; if wall drops to ≤ ~1.4 s, Stage **B1** (full Rust IRLS, ~1-2 weeks) proceeds. If B0 misses target, the plan returns to brainstorming before any further investment. The B0 / B1 split is **the structural counter-measure** to Phase A's "assumption broke 6 lines into the post-mortem" failure.

**Tech Stack:** Rust 1.76+ · PyO3 0.21 (abi3-py39) · Rayon 1.10 · numpy crate 0.21 · ndarray 0.15 · maturin · Python 3.9-3.13 · NumPy · pytest.

**Spec:** [`docs/superpowers/specs/2026-04-27-native-rust-irls-fepois-design.md`](../specs/2026-04-27-native-rust-irls-fepois-design.md) §6 sets the original Phase B surface. **This plan supersedes §6.1's "no new dependency / hand-coded Cholesky" assumption with the AUDIT.md finding** that sort-by-primary-FE is load-bearing — without it, even a perfect Rust IRLS port plateaus near Phase A's 2.45 s. The audit at [`benchmarks/hdfe/AUDIT.md`](../../../benchmarks/hdfe/AUDIT.md) "Phase A round 1" section is the design's ground truth.

---

## File Structure

| File | Stage | Action | Responsibility |
|------|:-----:|--------|----------------|
| `rust/statspai_hdfe/src/sort_perm.rs`            | B0 | create  | counting-sort `primary_fe_sort_perm`; pure Rust, no PyO3 |
| `rust/statspai_hdfe/src/demean.rs`               | B0 | modify  | + `weighted_group_sweep_sorted` (sequential O(n)); + `weighted_demean_matrix_fortran_inplace_sorted` driver |
| `rust/statspai_hdfe/src/lib.rs`                  | B0 | modify  | + `#[pyfunction] demean_2d_weighted_sorted`; reuse helpers |
| `rust/statspai_hdfe/Cargo.toml`                  | B0 | modify  | bump `0.2.0-alpha.1 → 0.3.0-alpha.1`; `__version__` `"0.3.0" → "0.4.0"` |
| `src/statspai/fast/fepois.py`                    | B0 | modify  | dispatcher routes K≥2 to the sorted path |
| `tests/test_fast_fepois.py`                      | B0 | modify  | + sort-aware kernel parity test + sort-aware fepois parity test |
| `benchmarks/hdfe/run_fepois_phase_b0.py`         | B0 | create  | wall measurement; **gate ≤ 1.5 s** (matches Phase A spec gate; if hit, B0 succeeds) |
| `rust/statspai_hdfe/src/irls.rs`                 | B1 | create  | `FePoisIRLSConfig`, `FePoisIRLSResult`, `FePoisIRLSWorkspace`, `pub fn fepois_loop` |
| `rust/statspai_hdfe/src/cholesky.rs`             | B1 | create  | hand-coded k×k SPD factorization + back-solve (no new dep) |
| `rust/statspai_hdfe/src/lib.rs`                  | B1 | modify  | + `#[pyfunction] fepois_irls`; bump `__version__` to `"0.5.0"` |
| `src/statspai/fast/fepois.py`                    | B1 | modify  | replace IRLS body with `_rust_hdfe.fepois_irls(...)`; vcov stays Python |
| `tests/test_fast_fepois.py`                      | B1 | modify  | + Phase B1 parity tests (coef/SE/weights/CR1/fallback) |
| `benchmarks/hdfe/run_fepois_phase_b.py`          | B1 | create  | wall measurement; **gate ≤ 0.95 s** (≤ 1.5× fixest) |
| `benchmarks/hdfe/SUMMARY.md`                     | B1 | modify  | update headline numbers + Phase row |
| `CHANGELOG.md`                                   | B1 | modify  | combined v1.8.0 release entry (Phase A primitives + Phase B IRLS) |
| `pyproject.toml` + `src/statspai/__init__.py`    | B1 | modify  | bump to `1.8.0` |

---

## Stage B0 — Sort-by-primary-FE spike

**B0 success criterion:** medium wall ≤ 1.5 s (the original Phase A gate). Failure → STOP, return to brainstorming, do NOT proceed to B1.

**B0 expected outcome (per AUDIT.md):** ~3-5× speedup on the FE1 sweep portion (currently ~80 % of Rust kernel wall). End-to-end wall projected ~1.0-1.4 s. If we land at 2.0+ s, the assumption is wrong again.

### Task B0.1: Sort-by-primary-FE primitive in Rust

**Files:**
- Create: `rust/statspai_hdfe/src/sort_perm.rs`
- Modify: `rust/statspai_hdfe/src/lib.rs` (add `mod sort_perm;`)

- [ ] **Step 1: Add the failing test**

In `rust/statspai_hdfe/src/sort_perm.rs` (new file), put:

```rust
//! Counting-sort permutation by FE code, for sort-aware sequential sweeps.
//!
//! Given codes ∈ [0, G) for n observations, returns a permutation π such
//! that ``codes[π[k]]`` is non-decreasing. Cost: O(n + G), one pass through
//! the data plus one over the bucket array. The result feeds into the
//! `weighted_group_sweep_sorted` kernel which exploits contiguity to do
//! sequential L1-cache-friendly accumulation instead of random scatter.

/// Counting-sort permutation by `codes` (assumed dense in [0, n_groups)).
/// Returns a Vec<usize> π of length n where ``codes[π[0..k]]`` are
/// non-decreasing.
pub fn primary_fe_sort_perm(codes: &[i64], n_groups: usize) -> Vec<usize> {
    let n = codes.len();
    // Phase 1: count occurrences per group.
    let mut counts = vec![0usize; n_groups];
    for &c in codes {
        counts[c as usize] += 1;
    }
    // Phase 2: prefix-sum to get group-start offsets.
    let mut starts = vec![0usize; n_groups];
    let mut acc = 0usize;
    for g in 0..n_groups {
        starts[g] = acc;
        acc += counts[g];
    }
    debug_assert_eq!(acc, n);
    // Phase 3: place each obs into its group's slot.
    let mut perm = vec![0usize; n];
    let mut cursor = starts.clone();
    for i in 0..n {
        let g = codes[i] as usize;
        perm[cursor[g]] = i;
        cursor[g] += 1;
    }
    perm
}

/// Returns the (start, len) per-group offsets corresponding to a permutation
/// produced by `primary_fe_sort_perm`. Caller may also reconstruct from
/// `counts` via prefix-sum; this helper is for callers that only have `perm`.
pub fn group_starts_from_codes_sorted(codes_sorted: &[i64], n_groups: usize) -> Vec<usize> {
    let n = codes_sorted.len();
    let mut starts = vec![n; n_groups + 1];
    starts[0] = 0;
    let mut prev: i64 = -1;
    for i in 0..n {
        let c = codes_sorted[i];
        if c != prev {
            for g in (prev + 1) as usize..=c as usize {
                starts[g] = i;
            }
            prev = c;
        }
    }
    starts[n_groups] = n;
    starts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perm_groups_obs_correctly() {
        let codes: Vec<i64> = vec![2, 0, 1, 0, 2, 1, 0];
        let perm = primary_fe_sort_perm(&codes, 3);
        // After applying perm, codes should be sorted: [0,0,0,1,1,2,2]
        let sorted: Vec<i64> = perm.iter().map(|&i| codes[i]).collect();
        assert_eq!(sorted, vec![0, 0, 0, 1, 1, 2, 2]);
    }

    #[test]
    fn perm_handles_empty_groups() {
        // Group 1 has no members.
        let codes: Vec<i64> = vec![0, 2, 0, 2];
        let perm = primary_fe_sort_perm(&codes, 3);
        let sorted: Vec<i64> = perm.iter().map(|&i| codes[i]).collect();
        assert_eq!(sorted, vec![0, 0, 2, 2]);
    }

    #[test]
    fn group_starts_round_trip() {
        let codes_sorted: Vec<i64> = vec![0, 0, 0, 1, 1, 2, 2];
        let starts = group_starts_from_codes_sorted(&codes_sorted, 3);
        assert_eq!(starts, vec![0, 3, 5, 7]);
    }
}
```

In `lib.rs`, add the line `mod sort_perm;` next to the existing `mod demean;` declaration (top of file, after the docstring).

- [ ] **Step 2: Run cargo test**

```bash
cargo test --manifest-path /Users/brycewang/Documents/GitHub/StatsPAI/rust/statspai_hdfe/Cargo.toml sort_perm 2>&1 | tail -10
```

Expected: 3 tests pass.

- [ ] **Step 3: Run full unit test suite (no regression)**

```bash
cargo test --manifest-path /Users/brycewang/Documents/GitHub/StatsPAI/rust/statspai_hdfe/Cargo.toml 2>&1 | tail -10
```

Expected: 12 (Phase A) + 3 (new) = 15 tests pass, 0 failed.

- [ ] **Step 4: Commit**

```bash
git -C /Users/brycewang/Documents/GitHub/StatsPAI add rust/statspai_hdfe/src/sort_perm.rs rust/statspai_hdfe/src/lib.rs && git -C /Users/brycewang/Documents/GitHub/StatsPAI commit -m "$(cat <<'EOF'
feat(rust-hdfe): add sort_perm — counting-sort permutation by FE code

Phase B0 prerequisite. Returns a permutation π such that codes[π[k]] is
non-decreasing, in O(n + G). Caller applies π once to weights and
columns before IRLS; the resulting sorted layout enables a sequential
weighted sweep that replaces the random-scatter inner loop currently
dominating the Rust demean kernel's wall on the medium benchmark.

3 unit tests cover non-trivial perm / empty groups / round-trip via
group_starts_from_codes_sorted helper.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task B0.2: Sequential weighted sweep that uses the sort

**Files:**
- Modify: `rust/statspai_hdfe/src/demean.rs`

- [ ] **Step 1: Add the sequential sweep + parity test**

Append to `rust/statspai_hdfe/src/demean.rs` immediately after the existing `weighted_group_sweep` function:

```rust
/// Sequential weighted group de-mean for input pre-sorted by `codes`.
///
/// Caller is responsible for permuting `x`, `weights` (and any other
/// per-observation arrays) by the perm from `sort_perm::primary_fe_sort_perm`
/// before calling this. `group_starts` has length `n_groups + 1`, with
/// `group_starts[g]..group_starts[g+1]` the contiguous slice of obs
/// belonging to group `g`. `wsum[g]` is the precomputed weighted group sum.
///
/// Cost: O(n) sequential. No random-access into `scratch` — the entire
/// per-group accumulate / divide / subtract chain happens within one
/// contiguous slice that fits in L1 for any reasonably-sized group.
#[inline]
pub fn weighted_group_sweep_sorted(
    x: &mut [f64],
    weights: &[f64],
    group_starts: &[usize],
    wsum: &[f64],
) {
    debug_assert_eq!(x.len(), weights.len());
    debug_assert_eq!(group_starts.len(), wsum.len() + 1);

    for g in 0..wsum.len() {
        let lo = group_starts[g];
        let hi = group_starts[g + 1];
        if lo == hi {
            continue;
        }
        let w = wsum[g];
        if w <= 0.0 {
            continue;
        }
        let mut acc = 0.0_f64;
        for i in lo..hi {
            acc += weights[i] * x[i];
        }
        let mean = acc / w;
        for i in lo..hi {
            x[i] -= mean;
        }
    }
}

#[cfg(test)]
mod sorted_sweep_tests {
    use super::*;

    #[test]
    fn sorted_matches_random_scatter() {
        // 4 obs, 2 groups, unequal weights (same as weighted_oneway_exact).
        // Sorted-input version applied directly should match the random-scatter
        // version on the same logical data.
        let x_random: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let codes: Vec<i64> = vec![0, 0, 1, 1];
        let weights: Vec<f64> = vec![1.0, 3.0, 2.0, 2.0];
        let wsum: Vec<f64> = vec![4.0, 4.0];

        // Sort-aware path: the input is already grouped; group_starts = [0, 2, 4].
        let mut x_sorted = x_random.clone();
        let group_starts = vec![0usize, 2, 4];
        weighted_group_sweep_sorted(&mut x_sorted, &weights, &group_starts, &wsum);

        // Random-scatter path on the same data.
        let mut x_rand = x_random.clone();
        let mut scratch = vec![0.0_f64; 2];
        weighted_group_sweep(&mut x_rand, &codes, &weights, &wsum, &mut scratch);

        for (a, b) in x_sorted.iter().zip(x_rand.iter()) {
            assert!((a - b).abs() < 1e-15, "sorted={a} random={b}");
        }
    }

    #[test]
    fn sorted_zero_weight_group_no_panic() {
        let mut x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let weights: Vec<f64> = vec![0.0, 0.0, 1.0, 1.0];
        let group_starts = vec![0usize, 2, 4];
        let wsum: Vec<f64> = vec![0.0, 2.0];
        weighted_group_sweep_sorted(&mut x, &weights, &group_starts, &wsum);
        // Group 0 untouched; group 1 demeaned: mean=3.5, residuals=[-0.5, 0.5].
        assert!((x[0] - 1.0).abs() < 1e-15);
        assert!((x[1] - 2.0).abs() < 1e-15);
        assert!((x[2] - (-0.5)).abs() < 1e-15);
        assert!((x[3] - 0.5).abs() < 1e-15);
    }
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test --manifest-path /Users/brycewang/Documents/GitHub/StatsPAI/rust/statspai_hdfe/Cargo.toml 2>&1 | tail -10
```

Expected: 17 tests pass (15 + 2 new), 0 failed.

- [ ] **Step 3: Commit**

```bash
git -C /Users/brycewang/Documents/GitHub/StatsPAI add rust/statspai_hdfe/src/demean.rs && git -C /Users/brycewang/Documents/GitHub/StatsPAI commit -m "$(cat <<'EOF'
feat(rust-hdfe): add weighted_group_sweep_sorted (sequential O(n))

Sort-aware variant of weighted_group_sweep that exploits the contiguous
per-group layout produced by sort_perm::primary_fe_sort_perm. Replaces
the random-scatter inner loop's L2-cache-miss pattern with a sequential
sweep that fits in L1 for any reasonable group size.

This is the algorithmic primitive Phase A's wall-clock gate was missing:
on the medium benchmark (n=1M, G1=100k), the random-scatter scratch
buffer is 800 KB — exceeds L1, lives in L2 with random access pattern.
The sorted variant per-group operates on a contiguous slice typically
< 1 KB, fully L1-resident.

2 unit tests cover (a) parity with the random-scatter version on
identical hand-computed data, (b) zero-weight group guard.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task B0.3: Sorted matrix driver + sort-aware multi-FE handling

**Files:**
- Modify: `rust/statspai_hdfe/src/demean.rs`

- [ ] **Step 1: Add the matrix-level sorted driver**

Append after `weighted_demean_matrix_fortran_inplace`:

```rust
/// Sort-aware weighted demean of a column-major (n × p) matrix in place.
///
/// Operates on PRE-PERMUTED inputs (caller applies π from
/// sort_perm::primary_fe_sort_perm to `x`, `weights`, the primary FE codes,
/// and the secondary FE codes). The primary FE sweep uses the sequential
/// kernel; secondary FE sweeps use the random-scatter kernel because their
/// cardinality is typically small enough that the bucket array fits in L1
/// (G2 = 1k → 8 KB).
///
/// Parameters
/// ----------
/// `mat`           — F-order (n, p), in-place; rows in primary-FE-sort order.
/// `n`, `p`        — shape.
/// `primary_starts` — group_starts for the primary FE (len = G1 + 1).
/// `primary_wsum`  — wsum array for the primary FE (len = G1).
/// `secondary_codes`, `secondary_wsum` — slices of slices, K-1 entries each
///     (one per non-primary FE).
/// `weights_sorted` — per-obs weights, in primary-FE-sort order.
/// `secondary_lens` — per-FE cardinalities for non-primary FEs.
/// `max_iter`, `tol_abs`, `tol_rel`, `accelerate`, `accel_period` — same
///     semantics as `weighted_demean_matrix_fortran_inplace`.
#[allow(clippy::too_many_arguments)]
pub fn weighted_demean_matrix_fortran_inplace_sorted(
    mat: &mut [f64],
    n: usize,
    p: usize,
    primary_starts: &[usize],
    primary_wsum: &[f64],
    secondary_codes: &[&[i64]],
    secondary_wsum: &[&[f64]],
    secondary_lens: &[usize],
    weights_sorted: &[f64],
    max_iter: u32,
    tol_abs: f64,
    tol_rel: f64,
    accelerate: bool,
    accel_period: u32,
) -> Vec<DemeanInfo> {
    debug_assert_eq!(mat.len(), n * p);
    debug_assert_eq!(weights_sorted.len(), n);
    debug_assert_eq!(secondary_codes.len(), secondary_wsum.len());
    debug_assert_eq!(secondary_codes.len(), secondary_lens.len());

    let cols: Vec<&mut [f64]> = mat.chunks_mut(n).collect();

    cols.into_par_iter()
        .map(|col| {
            // Per-thread scratch for non-primary FE dimensions.
            let mut sec_scratch: Vec<Vec<f64>> =
                secondary_lens.iter().map(|&g| vec![0.0_f64; g]).collect();

            // K==1 closed form: just the primary sequential sweep.
            if secondary_codes.is_empty() {
                weighted_group_sweep_sorted(col, weights_sorted, primary_starts, primary_wsum);
                return DemeanInfo {
                    iters: 1,
                    converged: true,
                    max_dx: 0.0,
                };
            }

            // K>=2: AP loop with mixed sweeps.
            let mut base_scale = 0.0_f64;
            for &v in col.iter() {
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
            let mut before: Vec<f64> = vec![0.0; col.len()];

            for it in 0..max_iter {
                before.copy_from_slice(col);
                // Primary (sequential) first.
                weighted_group_sweep_sorted(col, weights_sorted, primary_starts, primary_wsum);
                // Secondary (random scatter) — but on smaller bucket arrays.
                for k in 0..secondary_codes.len() {
                    weighted_group_sweep(
                        col,
                        secondary_codes[k],
                        weights_sorted,
                        secondary_wsum[k],
                        &mut sec_scratch[k],
                    );
                }

                let mut local_max = 0.0_f64;
                for i in 0..col.len() {
                    let d = (col[i] - before[i]).abs();
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
                    hist.push(col.to_vec());
                    if hist.len() >= 3 && (it + 1) % accel_period == 0 {
                        let n_h = hist.len();
                        let acc = aitken_step(&hist[n_h - 3], &hist[n_h - 2], &hist[n_h - 1]);
                        let mut max_abs_acc = 0.0_f64;
                        for &v in &acc {
                            let av = v.abs();
                            if av > max_abs_acc {
                                max_abs_acc = av;
                            }
                        }
                        if max_abs_acc < 10.0 * base_scale {
                            col.copy_from_slice(&acc);
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
        })
        .collect()
}
```

- [ ] **Step 2: Compile-check**

```bash
cargo check --manifest-path /Users/brycewang/Documents/GitHub/StatsPAI/rust/statspai_hdfe/Cargo.toml 2>&1 | tail -5
```

Expected: clean, no errors.

- [ ] **Step 3: Run tests**

```bash
cargo test --manifest-path /Users/brycewang/Documents/GitHub/StatsPAI/rust/statspai_hdfe/Cargo.toml 2>&1 | tail -10
```

Expected: 17 tests still pass; new function has no own test (covered via PyO3 path in B0.4).

- [ ] **Step 4: Commit**

```bash
git -C /Users/brycewang/Documents/GitHub/StatsPAI add rust/statspai_hdfe/src/demean.rs && git -C /Users/brycewang/Documents/GitHub/StatsPAI commit -m "$(cat <<'EOF'
feat(rust-hdfe): add weighted_demean_matrix_fortran_inplace_sorted

Rayon-parallel matrix driver that uses weighted_group_sweep_sorted for
the primary FE (sequential O(n)) and the existing random-scatter
weighted_group_sweep for secondary FEs (which have small cardinality
and L1-resident bucket arrays). AP loop / Aitken / safeguard layout
mirrors the random-scatter parent function exactly; only the inner
primary-FE sweep changes.

Coverage via the PyO3 + Python parity tests landing in B0.4 / B0.5.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task B0.4: PyO3 binding for sorted demean + dispatcher rewire

**Files:**
- Modify: `rust/statspai_hdfe/src/lib.rs`
- Modify: `rust/statspai_hdfe/Cargo.toml` (version bump)
- Modify: `src/statspai/fast/fepois.py` (dispatcher routes to sorted path)

- [ ] **Step 1: Add `#[pyfunction] demean_2d_weighted_sorted`**

In `lib.rs`, after `demean_2d_weighted`, insert a new pyfunction that takes pre-permuted `x` (F-order, n × p), `primary_starts`, `primary_wsum`, `secondary_codes` (list, K-1 entries), `secondary_wsum` (list), `weights_sorted` (1-D), and the same convergence params. The body validates shapes (similar to `demean_2d_weighted` but adapted for the sorted call surface), then calls `demean::weighted_demean_matrix_fortran_inplace_sorted` inside `py.allow_threads(...)`. Returns the same `List[Dict]` shape as `demean_2d_weighted`.

(Inline the full code in the implementer prompt — too long to paste here in full; it mirrors `demean_2d_weighted`'s validation pattern but with the sort-specific parameters.)

Add `wrap_pyfunction!(demean_2d_weighted_sorted, m)?` to the `#[pymodule]` block. Bump `__version__` to `"0.4.0"`.

In `Cargo.toml`, bump `version = "0.2.0-alpha.1" → "0.3.0-alpha.1"`.

- [ ] **Step 2: Update Python dispatcher to apply sort once + use sorted path**

In `src/statspai/fast/fepois.py`, modify `_weighted_ap_demean` (the dispatcher added in Phase A):

- Detect: `K == 2` and `_HAS_RUST_HDFE`.
- Identify the primary FE as the one with the largest cardinality (`np.argmax([c.size for c in counts_list])`).
- Apply `np.argsort(fe_codes_list[primary])` once to compute the permutation.
- Apply the perm to: `arr_F` (rows), `weights`, the secondary FE codes.
- Compute `primary_starts` from the sorted primary codes.
- Compute `wsum_list` against the permuted weights (or pre-perm — bincount is order-agnostic).
- Call `_rust_hdfe.demean_2d_weighted_sorted(...)`.
- Apply inverse permutation to the residualized output before returning to caller.

K == 1 stays on the closed-form NumPy path (unchanged).

Tests: confirm the dispatcher's behavior is bit-for-bit equivalent to the Phase A path on the existing parity test (`test_rust_weighted_demean_matches_numpy_kernel` should still pass at atol 1e-14).

- [ ] **Step 3: Compile + smoke test**

```bash
cargo check --manifest-path /Users/brycewang/Documents/GitHub/StatsPAI/rust/statspai_hdfe/Cargo.toml 2>&1 | tail -5
(cd /Users/brycewang/Documents/GitHub/StatsPAI/rust/statspai_hdfe && python3 -m maturin build --release --interpreter /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13)
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m pip install --force-reinstall --no-deps /Users/brycewang/Documents/GitHub/StatsPAI/rust/statspai_hdfe/target/wheels/statspai_hdfe-*.whl
python3 -c "import statspai_hdfe as r; print(r.__version__); print(hasattr(r, 'demean_2d_weighted_sorted'))"
```

Expected: `0.4.0` and `True`.

- [ ] **Step 4: Run pytest**

```bash
pytest /Users/brycewang/Documents/GitHub/StatsPAI/tests/test_fast_fepois.py -q 2>&1 | tail -5
```

Expected: 27 passed, 0 failed (Phase A test count); the existing kernel parity test now exercises the sorted path internally (via the dispatcher).

- [ ] **Step 5: Commit**

```bash
git -C /Users/brycewang/Documents/GitHub/StatsPAI add rust/statspai_hdfe/src/lib.rs rust/statspai_hdfe/Cargo.toml src/statspai/fast/fepois.py && git -C /Users/brycewang/Documents/GitHub/StatsPAI commit -m "$(cat <<'EOF'
feat(fast-fepois,rust-hdfe): wire sort-aware demean into IRLS dispatcher (v0.4.0)

Phase B0 spike — validates whether sort-by-primary-FE closes Phase A's
wall-clock gap to ≤ 1.5 s on the medium benchmark, before committing
to the full Phase B1 IRLS port.

- Rust crate v0.3.0 → v0.4.0; Cargo crate 0.2.0-alpha.1 → 0.3.0-alpha.1.
- New PyO3 entry point demean_2d_weighted_sorted accepts pre-permuted
  inputs and uses the sequential sweep for the primary FE.
- Python dispatcher in fepois.py: when K=2 and Rust is available,
  picks the higher-cardinality FE as primary, applies the perm once,
  routes through the sorted Rust path, applies inverse perm to the
  result. K=1 stays on the unchanged NumPy closed-form.

27 fast-fepois tests still pass at atol 1e-14 (Rust ↔ NumPy kernel
parity) and atol 1e-13 (fepois coef vs pyfixest).

Wall-clock impact verified in B0.5.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task B0.5: Phase B0 benchmark gate

**Files:**
- Create: `benchmarks/hdfe/run_fepois_phase_b0.py`

- [ ] **Step 1: Create the harness**

Mirrors `run_fepois_phase_a.py` (Phase A) but writes to `medium_statspai_phase_b0.json` and reports:
- Median wall over 3 reps.
- Per-call dispatcher overhead (sort + perm apply + inverse perm).
- Compare vs Phase A's 2.45 s and Phase 0's 2.61 s.
- **Gate: median ≤ 1.5 s.** Fail → STOP, return to brainstorming, do NOT proceed to B1.

(Implementer fills in the full harness code; structure mirrors `run_fepois_phase_a.py` with the JSON output adapted.)

- [ ] **Step 2: Run + record**

```bash
python3 /Users/brycewang/Documents/GitHub/StatsPAI/benchmarks/hdfe/run_fepois_phase_b0.py 2>&1 | tail -10
```

Expected: PASS at ≤ 1.5 s. If FAIL, the assumption (sort-by-FE delivers ≥ 2× on the FE1 sweep portion) is wrong; record findings in AUDIT.md "Phase B0 round 1" section and surface to user before B1.

- [ ] **Step 3: Commit (gate-pass case)**

```bash
git -C /Users/brycewang/Documents/GitHub/StatsPAI add benchmarks/hdfe/run_fepois_phase_b0.py benchmarks/hdfe/results/medium_statspai_phase_b0.json && git -C /Users/brycewang/Documents/GitHub/StatsPAI commit -m "$(cat <<'EOF'
bench(hdfe): Phase B0 wall-clock gate PASSED — sort-by-FE validates

Median wall on the medium benchmark (n=1M, G1=100k, G2=1k):
- Phase 0 (Python np.bincount): 2.61 s
- Phase A (Rust scatter):       2.45 s
- Phase B0 (Rust sequential):   <X.XX s>  ← actual measured
- Gate target:                  ≤ 1.50 s

Sort-by-primary-FE delivered the projected speedup. Phase B1 (full
native Rust IRLS, eliminate FFI roundtrips, IRLS workspace reuse)
is now justified to invest the additional ~1-2 weeks for the closure
to ≤ 0.95 s = ≤ 1.5× fixest.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

If the gate FAILS, STOP — do not commit a "PASSED" claim, write the gate-fail audit instead and ping the user. The decision to proceed or not is the user's call, not the implementer's.

---

## Stage B1 — Native Rust IRLS

**Pre-requisite:** B0 gate ≥ pass at ≤ 1.5 s. If B0 failed, do NOT start B1; brainstorm first.

**B1 success criterion:** medium wall ≤ 0.95 s (≤ 1.5× fixest::fepois).

### Task B1.1: IRLS workspace + config + result structs

**Files:**
- Create: `rust/statspai_hdfe/src/irls.rs`

- [ ] **Step 1: Create the file with structs**

`irls.rs` defines:
- `pub struct FePoisIRLSConfig { maxiter, tol, fe_tol, fe_maxiter, eta_clip, accel_period, max_halvings }`
- `pub struct FePoisIRLSResult { beta, x_tilde, w, eta, mu, deviance, log_likelihood, iters, converged, n_halvings, max_inner_dx }`
- `pub struct FePoisIRLSWorkspace { sort_perm, primary_starts, sec_scratch, hist_buf, before_buf, ... }` — single allocation at construction; all subsequent IRLS iters reuse these buffers.

Unit test: construct a workspace for a synthetic small problem (n=50, K=2, G1=10, G2=5), confirm dimensions agree.

- [ ] **Step 2: Run cargo test, commit.**

### Task B1.2: Hand-coded SPD Cholesky

**Files:**
- Create: `rust/statspai_hdfe/src/cholesky.rs`

- [ ] **Step 1: Implement and test.**

`pub fn cholesky_factor(a: &mut [f64], k: usize)` does in-place lower-triangular factorization of a k×k SPD matrix stored row-major. `pub fn cholesky_solve(l: &[f64], k: usize, b: &mut [f64])` does in-place forward-substitution + back-substitution to solve `L L^T x = b`. Numerical: stable enough for k ≤ ~30; adds a 1e-12 ridge if min-diagonal would underflow (matches `lapack_lite` policy).

Unit tests: a 3×3 hand-computed example + a fuzz test (random k×k SPD = `A^T A` with random A).

- [ ] **Step 2: Run, commit.**

### Task B1.3: `pub fn fepois_loop` — full IRLS body

**Files:**
- Modify: `rust/statspai_hdfe/src/irls.rs`

- [ ] **Step 1: Implement the loop.**

```rust
pub fn fepois_loop(
    y: &[f64],
    x_in: &[f64],          // F-order (n, p)
    n: usize, p: usize,
    fe_codes: &[&[i64]],
    counts: &[&[f64]],
    obs_weights: &[f64],
    config: &FePoisIRLSConfig,
    ws: &mut FePoisIRLSWorkspace,
) -> FePoisIRLSResult { ... }
```

The body:
1. Initialize `mu = max(y, 1) + 0.1`, `eta = ln(mu)`.
2. For `it in 0..config.maxiter`:
   a. Compute `z = eta + (y - mu) / mu` and `w = mu * obs_weights`.
   b. Compute `wsum_list` for each FE dim (from `w`).
   c. Apply pre-stored sort perm to `z` and to all p X columns; build a `(n, p+1)` workspace matrix that holds `[z, X]` together so a single `weighted_demean_matrix_fortran_inplace_sorted` call demeans both.
   d. Call the sorted demean — uses `ws`-managed scratch / hist / before buffers (no per-iter allocation).
   e. Apply inverse perm to the demeaned `z_tilde` and `X_tilde`.
   f. WLS: `XtWX = X_tilde.T @ (X_tilde * w[:, None])`; `XtWz = X_tilde.T @ (w * z_tilde)`; solve via Cholesky.
   g. Compute `eta_new = z - (z_tilde - X_tilde @ beta)`; clip; `mu_new = exp(eta_new)`; deviance.
   h. Step-halving up to `config.max_halvings`.
   i. Convergence check on `|new_dev - dev| / max(1, |new_dev|) < config.tol`.
3. Return `FePoisIRLSResult` with final `beta`, `X_tilde`, `w`, `eta`, `mu`, deviance, etc.

- [ ] **Step 2: Unit test on a small synthetic Poisson problem (n=200, p=2, K=2). Compare beta to a Python reference fit.**

- [ ] **Step 3: Commit.**

### Task B1.4: PyO3 binding `fepois_irls`

**Files:**
- Modify: `rust/statspai_hdfe/src/lib.rs`
- Modify: `rust/statspai_hdfe/Cargo.toml` (`__version__` `"0.4.0" → "0.5.0"`; Cargo `0.3.0-alpha.1 → 0.4.0-alpha.1`)

- [ ] **Step 1: Add the binding.**

`fepois_irls(y, x, fe_codes, counts, obs_weights, config_dict)` -> dict containing all `FePoisIRLSResult` fields.

- [ ] **Step 2: Compile + maturin smoke test.**

- [ ] **Step 3: Commit.**

### Task B1.5: Replace Python IRLS body with Rust call

**Files:**
- Modify: `src/statspai/fast/fepois.py`

- [ ] **Step 1: Replace the IRLS for-loop in `fepois()` with a single `_rust_hdfe.fepois_irls(...)` call when Rust is available.**

The Python `fepois()` becomes: parse formula, drop singletons + separation (Python pre-passes), build `(y, X, fe_codes, counts, obs_weights, config)`, call Rust IRLS, then compute vcov from the returned `X_tilde` + `w` (Python — unchanged). When Rust is unavailable, fall back to the Phase A Python IRLS (which itself uses the Phase A Rust weighted demean if available, else NumPy).

- [ ] **Step 2: Run all 27 fast-fepois tests; expect 27 pass.**

- [ ] **Step 3: Commit.**

### Task B1.6: Phase B1 parity tests

**Files:**
- Modify: `tests/test_fast_fepois.py`

- [ ] **Step 1: Add 5 new tests:**
- coef parity vs pyfixest atol ≤ 1e-13 (mirrors Phase A's `test_fepois_rust_path_coef_parity_vs_pyfixest`).
- SE parity vs Phase A NumPy fallback atol ≤ 1e-10 (verifies Phase B1 doesn't drift from Phase A on SE — and any drift vs pyfixest is the same baseline ~1e-5 we already documented).
- weights coef parity atol ≤ 1e-13.
- CR1 cluster-robust coef + SE parity (uses the user's recovered CR1 path) atol ≤ 1e-7 (same atol as `test_fepois_cluster_se_close_to_r_fixest`).
- Fallback: monkeypatch `_HAS_RUST_HDFE = False`; coef + SE atol ≤ 1e-10 vs Rust path.

- [ ] **Step 2: Run; expect 32 pass.**

- [ ] **Step 3: Commit.**

### Task B1.7: Full regression suite

- [ ] **Run** the same canonical regression set as Phase A's Task 10 (133+ tests). 0 failed required.

### Task B1.8: Phase B1 benchmark gate

**Files:**
- Create: `benchmarks/hdfe/run_fepois_phase_b.py`

- [ ] **Step 1: Create the harness — same shape as B0's harness but JSON output `medium_statspai_phase_b.json`. Gate ≤ 0.95 s.**

- [ ] **Step 2: Run.** PASS → commit. FAIL → AUDIT.md "Phase B1 round 1" + surface to user.

### Task B1.9: SUMMARY.md + CHANGELOG.md

**Files:**
- Modify: `benchmarks/hdfe/SUMMARY.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Update SUMMARY headline** with Phase B1 final wall + iters; update "What deliberately did NOT ship" section.

- [ ] **Step 2: Add a `## [1.8.0]` section to CHANGELOG.md** that combines Phase A primitives + Phase B IRLS wins. Be honest about the wall numbers (e.g., "from 4.08× of fixest down to 1.4× — driven primarily by sort-by-primary-FE sequential sweep + native Rust IRLS state machine").

- [ ] **Step 3: Commit.**

### Task B1.10: Version bump to 1.8.0

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/statspai/__init__.py`

- [ ] **Step 1: Bump version. Run `python -c "import statspai as sp; print(sp.__version__)"`. Confirm `1.8.0`.**

- [ ] **Step 2: Run registry self-check from CLAUDE.md §14: `python -c "import statspai as sp; print(len(sp.list_functions()))"`.**

- [ ] **Step 3: Commit.**

---

## Self-Review

**1. Spec coverage check** — every section of the spec maps to a task:

| Spec / AUDIT section | Task(s) |
|---------------------|---------|
| §6.1 IRLS structs (Config / Result / Workspace) | B1.1 |
| §6.1 SPD Cholesky (no new dep — hand-coded) | B1.2 |
| §6.1 `fepois_loop` core | B1.3 |
| §6.1 PyO3 binding | B1.4 |
| §6.2 Python wiring (replace IRLS body) | B1.5 |
| §6.3 Parity tests | B1.6 |
| §6.4 Benchmark gate ≤ 0.95 s | B1.8 |
| §6.5 Versioning + CHANGELOG | B1.9, B1.10 |
| AUDIT "sort-by-primary-FE is the key" | B0 (entirely) |
| AUDIT "spike-first to validate the assumption" | B0 stage exists |

The AUDIT-driven B0 spike has no §6 mapping — it is the structural addition this plan makes on top of the spec, in direct response to Phase A's failure mode.

**2. Placeholder scan** — searched for "TBD", "TODO", "implement later", "fill in details", "appropriate error handling". Two locations use the phrase "Implementer fills in the full harness code": Task B0.5 (the Phase B0 benchmark) and B0.4 (the PyO3 binding body). For the benchmark this is acceptable — the structure mirrors `run_fepois_phase_a.py` which already exists and is reproducible. For the PyO3 binding it is technical debt that the implementer should resolve by following the `demean_2d_weighted` template at `rust/statspai_hdfe/src/lib.rs:212+`. **Acceptable but flagged.**

**3. Type / signature consistency** — cross-checked:

- `primary_fe_sort_perm(codes, n_groups) -> Vec<usize>` defined in Task B0.1, called in B0.4 (Python dispatcher invokes it via PyO3 — would need a `#[pyfunction]` wrapper, which is implicit in B0.4 step 1 but should be explicit in the implementer prompt).
- `weighted_group_sweep_sorted(x, weights, group_starts, wsum)` in Task B0.2; called from `weighted_demean_matrix_fortran_inplace_sorted` in B0.3.
- `weighted_demean_matrix_fortran_inplace_sorted(...)` signature defined in B0.3, called from `demean_2d_weighted_sorted` PyO3 in B0.4.
- `fepois_loop(y, x_in, n, p, fe_codes, counts, obs_weights, config, ws) -> FePoisIRLSResult` defined in B1.3; called from `fepois_irls` PyO3 binding in B1.4.

No mismatches.

**4. Scope** — is this focused enough for a single implementation plan? Yes — Phase B0 and Phase B1 are sequenced (B1 depends on B0 success), but they share a single design context (close the wall-clock gap to fixest). They're not independent subsystems requiring separate plans.

**5. Risk-management discipline** — does the plan avoid Phase A's failure mode? Yes:

- Phase A failure: "we projected 50 % wall reduction without isolating the assumption; the assumption (Rust kernel beats NumPy bincount by 3-5×) was wrong by 2× when measured."
- Phase B counter-measure: B0 isolates ONE assumption (sort-by-primary-FE delivers the projected speedup); we measure end-to-end wall under the simplest possible change before committing to the heavier B1 work. If B0 fails, we lose ~1-3 days, not ~2 weeks.
- B0 does NOT require the IRLS workspace, the SPD Cholesky, or any of the B1 plumbing. It can ship as a standalone Phase A.5 if B1 ever needs to defer further.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-27-phase-b-rust-irls.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, two-stage review between tasks. **Strongly recommended for B1** (~10 tasks across Rust + Python, multiple files per task, integration risk if anything drifts). For B0 (5 small tasks) inline could also work but subagent stays consistent.

**2. Inline Execution** — execute tasks in the current session.

After Phase B0 lands (PASS or FAIL), surface the wall number to the user before deciding whether to start Phase B1. **B0 → B1 is a deliberate human-in-loop gate**, not an automatic continuation; the lesson from Phase A is that "auto-proceed past a missed gate" wastes weeks.
