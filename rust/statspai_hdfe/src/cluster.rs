//! Cluster-robust sandwich "meat" matrix.
//!
//! Given a regression with score contributions ``s_i = x_i * r_i`` and
//! cluster assignments, the cluster-robust meat (Liang–Zeger 1986) is
//!
//!     M = Σ_g (Σ_{i ∈ g} s_i) (Σ_{i ∈ g} s_i)ᵀ
//!
//! Clusters are independent → embarrassingly parallel via Rayon. Fixed
//! blocks of clusters are accumulated in parallel and the block sums are
//! added in order, so the result does not depend on the thread count. Within each cluster only the upper triangle is updated; the
//! caller-visible side is symmetrised at the end.
//!
//! The Python wrapper at ``statspai.core._numba_kernels.cluster_meat``
//! pre-sorts ``X`` and ``residuals`` by cluster so this kernel only
//! needs ``cluster_starts`` / ``cluster_ends`` index arrays — no string
//! / object-id comparisons cross the FFI boundary.

use rayon::prelude::*;

/// Clusters per parallel block. Part of the result's definition: the
/// partial sums are grouped by these blocks.
const CLUSTERS_PER_BLOCK: usize = 64;

/// Compute the cluster-robust meat matrix.
///
/// Parameters
/// ----------
/// x : row-major slice of length ``n * k``
///     Pre-sorted design matrix; each row contributes to one cluster.
/// n, k : usize
///     Matrix dimensions.
/// residuals : slice of length ``n``
///     Pre-sorted residuals (same row order as ``x``).
/// cluster_starts, cluster_ends : slices of length ``G``
///     ``cluster_starts[g] .. cluster_ends[g]`` is the half-open row
///     range of cluster ``g``.
///
/// Returns
/// -------
/// ``Vec<f64>`` of length ``k * k`` in row-major order, fully
/// symmetrised.
pub fn cluster_meat_sorted(
    x: &[f64],
    n: usize,
    k: usize,
    residuals: &[f64],
    cluster_starts: &[usize],
    cluster_ends: &[usize],
) -> Vec<f64> {
    debug_assert_eq!(x.len(), n.checked_mul(k).expect("n*k overflow"));
    debug_assert_eq!(residuals.len(), n);
    debug_assert_eq!(cluster_starts.len(), cluster_ends.len());

    let g = cluster_starts.len();
    if g == 0 || k == 0 {
        return vec![0f64; k * k];
    }

    // Fixed blocks of clusters, summed in parallel and then added in
    // block order. The grouping of the partial sums depends only on the
    // input, not on the number of threads or on work stealing, so the
    // result is reproducible bit for bit.
    let block_meats: Vec<Vec<f64>> = (0..g)
        .collect::<Vec<usize>>()
        .par_chunks(CLUSTERS_PER_BLOCK)
        .map(|block| {
            let mut acc = vec![0f64; k * k];
            let mut score = vec![0f64; k];
            for &gi in block {
                for s in score.iter_mut() {
                    *s = 0.0;
                }
                let s_start = cluster_starts[gi];
                let s_end = cluster_ends[gi];
                debug_assert!(s_end >= s_start && s_end <= n);

                // score = X_g.T @ r_g (k-vector).
                for i in s_start..s_end {
                    let r = residuals[i];
                    let row = &x[i * k..(i + 1) * k];
                    for j in 0..k {
                        score[j] += row[j] * r;
                    }
                }

                // Upper-triangle outer product accumulation.
                for a in 0..k {
                    let sa = score[a];
                    let row_off = a * k;
                    for b in a..k {
                        acc[row_off + b] += sa * score[b];
                    }
                }
            }
            acc
        })
        .collect();
    let mut meat = vec![0f64; k * k];
    for bm in &block_meats {
        for i in 0..(k * k) {
            meat[i] += bm[i];
        }
    }

    // Mirror upper triangle to lower so callers can treat the result as
    // a fully symmetric matrix.
    for a in 0..k {
        for b in (a + 1)..k {
            meat[b * k + a] = meat[a * k + b];
        }
    }
    meat
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference implementation used by the parity test below.
    fn reference(
        x: &[f64],
        n: usize,
        k: usize,
        residuals: &[f64],
        starts: &[usize],
        ends: &[usize],
    ) -> Vec<f64> {
        let mut meat = vec![0f64; k * k];
        for g in 0..starts.len() {
            let mut score = vec![0f64; k];
            for i in starts[g]..ends[g] {
                let r = residuals[i];
                for j in 0..k {
                    score[j] += x[i * k + j] * r;
                }
            }
            for a in 0..k {
                for b in 0..k {
                    meat[a * k + b] += score[a] * score[b];
                }
            }
        }
        let _ = n; // silence unused-var lint when n only matters for asserts
        meat
    }

    #[test]
    fn matches_reference_small() {
        // 8 rows, 3 cols, 3 clusters of sizes (3, 2, 3).
        let x: Vec<f64> = (0..24).map(|v| v as f64 - 5.0).collect();
        let r: Vec<f64> = (0..8).map(|v| 0.1 * v as f64 + 0.3).collect();
        let starts = vec![0usize, 3, 5];
        let ends = vec![3usize, 5, 8];

        let got = cluster_meat_sorted(&x, 8, 3, &r, &starts, &ends);
        let want = reference(&x, 8, 3, &r, &starts, &ends);
        for i in 0..(3 * 3) {
            assert!(
                (got[i] - want[i]).abs() < 1e-12,
                "mismatch at {i}: got {} want {}",
                got[i],
                want[i]
            );
        }
    }

    #[test]
    fn empty_inputs_return_zero_matrix() {
        let got = cluster_meat_sorted(&[], 0, 4, &[], &[], &[]);
        assert_eq!(got.len(), 16);
        assert!(got.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn k_one_works() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let r = vec![1.0, 1.0, 1.0, 1.0];
        let starts = vec![0usize, 2];
        let ends = vec![2usize, 4];
        // cluster 1 score = 1 + 2 = 3; cluster 2 score = 3 + 4 = 7.
        // meat = 9 + 49 = 58.
        let got = cluster_meat_sorted(&x, 4, 1, &r, &starts, &ends);
        assert_eq!(got, vec![58.0]);
    }

    #[test]
    fn bit_identical_across_thread_counts() {
        let (n, k, g) = (20_000usize, 4usize, 1_500usize);
        let mut state: u64 = 3;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        };
        let x: Vec<f64> = (0..n * k).map(|_| next()).collect();
        let r: Vec<f64> = (0..n).map(|_| next()).collect();
        let starts: Vec<usize> = (0..g).map(|c| c * n / g).collect();
        let ends: Vec<usize> = (0..g).map(|c| (c + 1) * n / g).collect();
        let mut results = Vec::new();
        for threads in [1usize, 3, 8] {
            crate::pool::set_num_threads(threads).unwrap();
            results.push(crate::pool::install(|| {
                cluster_meat_sorted(&x, n, k, &r, &starts, &ends)
            }));
        }
        for res in &results[1..] {
            assert!(res.iter().zip(&results[0]).all(|(a, b)| a.to_bits() == b.to_bits()));
        }
        let want = reference(&x, n, k, &r, &starts, &ends);
        for i in 0..k * k {
            assert!((results[0][i] - want[i]).abs() < 1e-9);
        }
    }
}
