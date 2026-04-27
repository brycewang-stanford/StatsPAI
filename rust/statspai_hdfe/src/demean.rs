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
}
