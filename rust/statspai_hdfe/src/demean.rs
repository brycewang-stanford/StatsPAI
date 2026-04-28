//! K-way alternating-projection HDFE demean kernel with Irons–Tuck (vector
//! Aitken) acceleration.
//!
//! Mirrors the algorithm used by the existing Python/Numba path in
//! ``src/statspai/panel/hdfe.py``, but pushes the inner sweep + extrapolation
//! into Rust for ~3-10× speedup on large panels.
//!
//! ## Convergence criterion
//!
//! After each full pass over all K FE dimensions, we measure
//! ``max_dx = max_i |x_i - x_i^prev|`` and stop when
//!
//! ```text
//! max_dx <= tol_abs + tol_rel * base_scale
//! ```
//!
//! where ``base_scale = max_i |x_i^initial| + 1e-30`` (computed once before
//! the loop). The double-threshold form lets users dial absolute tolerance
//! when ``X`` has been pre-standardised, and falls back to a pure relative
//! threshold (``tol_abs=0``) which mirrors the historical behaviour.
//!
//! ## Acceleration
//!
//! Every ``accel_period`` sweeps (default 5), once we have at least three
//! buffered iterates, we extrapolate via Irons–Tuck:
//!
//! ```text
//! d1     = x1 - x0
//! d2     = x2 - 2 x1 + x0
//! alpha  = <d1, d2> / <d2, d2>     (scalar, vector inner product)
//! x_acc  = x0 - alpha * d1
//! ```
//!
//! and accept the jump only if ``max|x_acc| < 10 * base_scale`` (avoiding
//! divergent extrapolation on near-degenerate problems). The buffer is then
//! cleared, matching the SQUAREM / Varadhan-Roland (2008 §3) layout used
//! upstream.

use rayon::prelude::*;

/// Outcome of a single column's AP loop.
#[derive(Clone, Copy, Debug, Default)]
pub struct DemeanInfo {
    pub iters: u32,
    pub converged: bool,
    pub max_dx: f64,
}

/// In-place de-mean by group ``codes`` using ``counts`` (group sizes).
///
/// ``scratch`` must have length ``counts.len()`` and is zero-filled on
/// entry. Caller owns its allocation; this function never reallocates on
/// the hot path.
#[inline]
fn group_sweep(x: &mut [f64], codes: &[i64], counts: &[f64], scratch: &mut [f64]) {
    debug_assert_eq!(scratch.len(), counts.len());
    debug_assert_eq!(codes.len(), x.len());

    for s in scratch.iter_mut() {
        *s = 0.0;
    }
    for i in 0..x.len() {
        let g = codes[i] as usize;
        scratch[g] += x[i];
    }
    for g in 0..scratch.len() {
        let c = counts[g];
        if c > 0.0 {
            scratch[g] /= c;
        }
    }
    for i in 0..x.len() {
        let g = codes[i] as usize;
        x[i] -= scratch[g];
    }
}

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

/// Sequential weighted group de-mean for input pre-sorted by `codes`.
///
/// Caller is responsible for permuting `x`, `weights` (and any other
/// per-observation arrays) by the perm from
/// ``sort_perm::primary_fe_sort_perm`` before calling this. ``group_starts``
/// has length ``n_groups + 1``, with ``group_starts[g]..group_starts[g+1]``
/// the contiguous slice of obs belonging to group ``g``. ``wsum[g]`` is the
/// precomputed weighted group sum (``Σ_{i ∈ g} weights[i]``).
///
/// Cost: O(n) sequential. No random-access into a per-group scratch buffer
/// — the entire per-group accumulate / divide / subtract chain happens
/// within one contiguous slice that fits in L1 for any reasonably-sized
/// group, replacing the L2-cache-miss pattern of the random-scatter
/// `weighted_group_sweep` on high-cardinality primary FEs.
#[inline]
pub fn weighted_group_sweep_sorted(
    x: &mut [f64],
    weights: &[f64],
    group_starts: &[usize],
    wsum: &[f64],
) {
    debug_assert_eq!(x.len(), weights.len());
    debug_assert_eq!(group_starts.len(), wsum.len() + 1);
    debug_assert!(
        group_starts.last().copied().unwrap_or(0) <= x.len(),
        "group_starts terminal offset overflows x.len()"
    );

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

/// Sweep all K FE dimensions once, weighted variant.
fn weighted_sweep_all_fe(
    x: &mut [f64],
    fe_codes: &[&[i64]],
    weights: &[f64],
    wsum: &[&[f64]],
    scratch: &mut [Vec<f64>],
) {
    debug_assert_eq!(fe_codes.len(), wsum.len());
    debug_assert_eq!(fe_codes.len(), scratch.len());
    for k in 0..fe_codes.len() {
        weighted_group_sweep(x, fe_codes[k], weights, wsum[k], &mut scratch[k]);
    }
}

/// Sweep through all K FEs once.
fn sweep_all_fe(
    x: &mut [f64],
    fe_codes: &[&[i64]],
    counts: &[&[f64]],
    scratch: &mut [Vec<f64>],
) {
    for k in 0..fe_codes.len() {
        group_sweep(x, fe_codes[k], counts[k], &mut scratch[k]);
    }
}

/// Vector Irons–Tuck extrapolation. Returns the accelerated vector or the
/// last iterate if the denominator is degenerate.
fn aitken_step(x0: &[f64], x1: &[f64], x2: &[f64]) -> Vec<f64> {
    let n = x0.len();
    debug_assert_eq!(x1.len(), n);
    debug_assert_eq!(x2.len(), n);

    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for i in 0..n {
        let d1 = x1[i] - x0[i];
        let d2 = x2[i] - 2.0 * x1[i] + x0[i];
        num += d1 * d2;
        den += d2 * d2;
    }
    if den < 1e-30 {
        return x2.to_vec();
    }
    let alpha = num / den;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(x0[i] - alpha * (x1[i] - x0[i]));
    }
    out
}

/// Demean a single column in place. ``scratch`` is a per-FE workspace owned
/// by the caller; pre-allocating it lets us reuse buffers across columns.
#[allow(clippy::too_many_arguments)]
pub fn demean_column_inplace(
    x: &mut [f64],
    fe_codes: &[&[i64]],
    counts: &[&[f64]],
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
        sweep_all_fe(x, fe_codes, counts, scratch);
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
        sweep_all_fe(x, fe_codes, counts, scratch);

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

/// Demean a contiguous (n × p) column-major matrix in place, parallel over
/// columns via Rayon. Returns one `DemeanInfo` per column.
///
/// ``mat`` is a flat slice of length n*p, where column ``j`` occupies the
/// range ``[j*n, (j+1)*n)``. This matches NumPy ``.copy(order='F')`` /
/// `np.asfortranarray` layout.
#[allow(clippy::too_many_arguments)]
pub fn demean_matrix_fortran_inplace(
    mat: &mut [f64],
    n: usize,
    p: usize,
    fe_codes: &[&[i64]],
    counts: &[&[f64]],
    counts_lens: &[usize],
    max_iter: u32,
    tol_abs: f64,
    tol_rel: f64,
    accelerate: bool,
    accel_period: u32,
) -> Vec<DemeanInfo> {
    debug_assert_eq!(mat.len(), n * p);

    // Ownership trick: split mat into p column-slices, then process them in
    // parallel. Each thread allocates its own per-FE scratch buffer (cheap;
    // sized by FE cardinality, not n).
    let cols: Vec<&mut [f64]> = mat.chunks_mut(n).collect();

    cols.into_par_iter()
        .map(|col| {
            let mut scratch: Vec<Vec<f64>> =
                counts_lens.iter().map(|&g| vec![0.0_f64; g]).collect();
            demean_column_inplace(
                col,
                fe_codes,
                counts,
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

/// Weighted demean of a column-major (n × p) matrix in place, parallel
/// over columns. Mirrors `demean_matrix_fortran_inplace`; the only
/// difference is the inner per-column kernel.
///
/// `mat` is a flat slice of length `n*p` where column `j` occupies the
/// range `[j*n, (j+1)*n)` (Fortran/column-major order, matching
/// `np.asfortranarray` layout).
///
/// `wsum_lens[k]` must equal `wsum[k].len()` (the cardinality of FE
/// dimension `k`). Passing it as a separate slice avoids repeated
/// `.len()` calls inside the Rayon closure when sizing the per-thread
/// scratch buffers.
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

/// Sort-aware weighted demean of a column-major (n × p) matrix in place.
///
/// Operates on PRE-PERMUTED inputs: the caller applies π from
/// ``sort_perm::primary_fe_sort_perm`` to ``mat`` (rows), ``weights``,
/// the primary FE codes (now contiguously grouped), and the secondary
/// FE codes (re-indexed under π). The primary FE sweep uses the
/// sequential kernel (``weighted_group_sweep_sorted``); secondary FE
/// sweeps use the random-scatter kernel because their cardinality is
/// typically small enough that the bucket array fits in L1 (G2 = 1k →
/// 8 KB).
///
/// `primary_starts` has length `G1 + 1`; `primary_wsum` has length `G1`.
/// `secondary_codes` and `secondary_wsum` are slices of slices, K-1
/// entries each (one per non-primary FE), all under π. `secondary_lens`
/// gives per-FE cardinalities for non-primary FEs (used to size per-thread
/// scratch). `weights_sorted` is the per-obs weight in π order.
///
/// Note: floating-point summation order within each primary group differs
/// from the random-scatter ``weighted_demean_matrix_fortran_inplace``
/// because observations are accumulated in π order rather than original
/// order. Per-group drift is therefore ≈ 1e-15 vs the unsorted path; the
/// IRLS outer loop's relative-deviance tolerance comfortably absorbs
/// this. Bit-equality is only guaranteed when input is already in π
/// order (i.e., the perm is the identity).
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
    debug_assert_eq!(
        primary_starts.len(),
        primary_wsum.len() + 1,
        "primary_starts must have length G1+1, primary_wsum length G1"
    );
    debug_assert!(
        !primary_wsum.is_empty(),
        "sorted dispatch requires at least one primary FE group"
    );
    debug_assert_eq!(secondary_codes.len(), secondary_wsum.len());
    debug_assert_eq!(secondary_codes.len(), secondary_lens.len());

    let cols: Vec<&mut [f64]> = mat.chunks_mut(n).collect();

    cols.into_par_iter()
        .map(|col| {
            // Per-thread scratch for non-primary FE dimensions only.
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
                // Secondary (random scatter) — small bucket arrays, L1-resident.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// One-way demean: `x - mean(x | g)` should be exact in one sweep.
    #[test]
    fn oneway_exact() {
        let mut x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let codes: Vec<i64> = vec![0, 0, 1, 1, 2, 2];
        let counts: Vec<f64> = vec![2.0, 2.0, 2.0];
        let mut scratch = vec![vec![0.0_f64; 3]];
        let info = demean_column_inplace(
            &mut x,
            &[&codes],
            &[&counts],
            &mut scratch,
            100,
            0.0,
            1e-10,
            true,
            5,
        );
        assert!(info.converged);
        // group means: 1.5, 3.5, 5.5
        let expected = vec![-0.5, 0.5, -0.5, 0.5, -0.5, 0.5];
        for (a, b) in x.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
    }

    /// Two-way: alternating projection should drive `dx` to zero.
    #[test]
    fn twoway_converges() {
        // 4 obs, 2x2 FE
        let mut x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let codes_i: Vec<i64> = vec![0, 0, 1, 1];
        let codes_t: Vec<i64> = vec![0, 1, 0, 1];
        let counts_i: Vec<f64> = vec![2.0, 2.0];
        let counts_t: Vec<f64> = vec![2.0, 2.0];
        let mut scratch = vec![vec![0.0_f64; 2], vec![0.0_f64; 2]];
        let info = demean_column_inplace(
            &mut x,
            &[&codes_i, &codes_t],
            &[&counts_i, &counts_t],
            &mut scratch,
            500,
            0.0,
            1e-12,
            true,
            5,
        );
        assert!(info.converged, "AP should converge on a balanced 2x2");
        // For a balanced 2-way model with no interaction, demeaning out i and t
        // leaves the residual = x_ij - x_i. - x_.j + x_.. — which for this
        // x = [1,2;3,4] is identically zero (linear in i and t).
        for &v in x.iter() {
            assert!(v.abs() < 1e-9, "residual = {v}");
        }
    }

    #[test]
    fn aitken_handles_degenerate() {
        // Same x repeatedly => d2 = 0 => fallback to x2.
        let x0 = vec![1.0, 2.0];
        let x1 = vec![1.0, 2.0];
        let x2 = vec![1.0, 2.0];
        let r = aitken_step(&x0, &x1, &x2);
        assert_eq!(r, x2);
    }

    #[test]
    fn k_zero_is_noop() {
        let mut x: Vec<f64> = vec![1.0, 2.0, 3.0];
        let mut scratch: Vec<Vec<f64>> = vec![];
        let info = demean_column_inplace(
            &mut x,
            &[],
            &[],
            &mut scratch,
            10,
            0.0,
            1e-10,
            true,
            5,
        );
        assert!(info.converged);
        assert_eq!(x, vec![1.0, 2.0, 3.0]);
    }

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

        assert!(info_w.converged, "weighted path did not converge");
        assert!(info_u.converged, "unweighted path did not converge");
        for (a, b) in x_w.iter().zip(x_u.iter()) {
            assert!((a - b).abs() < 1e-12, "weighted={a} unweighted={b}");
        }
    }

    /// Unequal weights: convergence and finite-residual stability on a 2×2 panel.
    /// (This is a stability sanity check, not a closed-form numeric verification —
    /// the K=1 weighted means on this same data are checked element-wise in
    /// `weighted_oneway_exact`. Promoting this to a closed-form K=2 check is
    /// a tracked future hardening.)
    #[test]
    fn weighted_unequal_weights_2x2() {
        // Two units (i = 0,1), two periods (t = 0,1); 4 obs.
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

    /// Sort-aware sweep on already-grouped input must match the random-scatter
    /// version on the same logical data.
    #[test]
    fn sorted_matches_random_scatter() {
        // 4 obs, 2 groups, unequal weights — same setup as weighted_oneway_exact.
        let codes: Vec<i64> = vec![0, 0, 1, 1];
        let weights: Vec<f64> = vec![1.0, 3.0, 2.0, 2.0];
        let wsum: Vec<f64> = vec![4.0, 4.0];

        // Sorted-input path: codes are already grouped; group_starts = [0, 2, 4].
        let mut x_sorted: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let group_starts = vec![0usize, 2, 4];
        weighted_group_sweep_sorted(&mut x_sorted, &weights, &group_starts, &wsum);

        // Random-scatter path on the same data.
        let mut x_rand: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let mut scratch = vec![0.0_f64; 2];
        weighted_group_sweep(&mut x_rand, &codes, &weights, &wsum, &mut scratch);

        for (a, b) in x_sorted.iter().zip(x_rand.iter()) {
            assert!((a - b).abs() < 1e-15, "sorted={a} random={b}");
        }
    }

    /// Empty interior group (group_starts has adjacent equal entries): the
    /// `lo == hi` skip branch must fire and the function must produce
    /// correct residuals for the populated group.
    #[test]
    fn sorted_empty_interior_group_no_panic() {
        // group_starts = [0, 0, 2]: group 0 has no obs, group 1 has [0, 1].
        let mut x: Vec<f64> = vec![3.0, 4.0];
        let weights: Vec<f64> = vec![1.0, 1.0];
        let group_starts = vec![0usize, 0, 2];
        let wsum: Vec<f64> = vec![0.0, 2.0];
        weighted_group_sweep_sorted(&mut x, &weights, &group_starts, &wsum);
        // Group 1 mean = (3 + 4) / 2 = 3.5; residuals [-0.5, +0.5].
        assert!((x[0] - (-0.5)).abs() < 1e-15);
        assert!((x[1] - 0.5).abs() < 1e-15);
    }

    /// Zero-weight group: must not panic, must not produce NaN, and must
    /// leave the zero-weight group unchanged.
    #[test]
    fn sorted_zero_weight_group_no_panic() {
        // codes = [0, 0, 1, 1] with weights = [0, 0, 1, 1]: group 0 is
        // entirely zero-weight and must be left untouched; group 1 has
        // weighted mean (3+4)/2 = 3.5 and residuals [-0.5, +0.5].
        let mut x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let weights: Vec<f64> = vec![0.0, 0.0, 1.0, 1.0];
        let group_starts = vec![0usize, 2, 4];
        let wsum: Vec<f64> = vec![0.0, 2.0];
        weighted_group_sweep_sorted(&mut x, &weights, &group_starts, &wsum);
        assert!((x[0] - 1.0).abs() < 1e-15);
        assert!((x[1] - 2.0).abs() < 1e-15);
        assert!((x[2] - (-0.5)).abs() < 1e-15);
        assert!((x[3] - 0.5).abs() < 1e-15);
        for &v in x.iter() {
            assert!(v.is_finite(), "NaN/Inf leaked: {v}");
        }
    }
}
