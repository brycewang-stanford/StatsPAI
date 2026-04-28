//! Hand-coded k×k SPD Cholesky factorization + triangular solves.
//!
//! Used by the native Rust IRLS in ``irls.rs`` for the WLS step
//! ``(X'WX) β = X'Wz``. For the typical fepois case k ≤ 30, an
//! in-place Rust loop beats a BLAS dispatch by avoiding FFI / setup
//! overhead, and avoids adding a new wheel-build dependency.
//!
//! The matrix is stored row-major as a flat `&mut [f64]` of length
//! `k*k`. After `cholesky_factor`, the lower triangle holds L such
//! that A = L L^T; the strict-upper triangle is left untouched (the
//! caller must not read it after factorization).
//!
//! `cholesky_solve` does forward + back substitution to solve
//! `L L^T x = b` in-place on `b`.

const RIDGE: f64 = 1e-12;

/// In-place Cholesky factor of a k×k SPD matrix stored row-major.
/// On return, the lower triangle of `a` holds L (the factor); the
/// strict upper triangle is unchanged. If a diagonal pivot would
/// underflow below `RIDGE`, a small ridge is added (matches the
/// `lapack_lite` policy in the Python NumPy fallback).
///
/// Returns `Err(())` if the input is too poorly conditioned to factor
/// even with ridge regularization (currently unreachable on
/// well-formed Σ_i w_i x_i x_i' for the IRLS use case).
#[allow(clippy::needless_range_loop)]
pub fn cholesky_factor(a: &mut [f64], k: usize) -> Result<(), ()> {
    debug_assert_eq!(a.len(), k * k);

    for j in 0..k {
        // Diagonal element: a[j,j] -= Σ_{m<j} L[j,m]^2; sqrt; ridge guard.
        let mut diag = a[j * k + j];
        for m in 0..j {
            let ljm = a[j * k + m];
            diag -= ljm * ljm;
        }
        if diag <= RIDGE {
            diag = diag.max(RIDGE);
        }
        let l_jj = diag.sqrt();
        if !l_jj.is_finite() || l_jj <= 0.0 {
            return Err(());
        }
        a[j * k + j] = l_jj;
        // Sub-diagonal elements in column j: L[i,j] = (a[i,j] - Σ_{m<j} L[i,m] L[j,m]) / L[j,j]
        for i in (j + 1)..k {
            let mut s = a[i * k + j];
            for m in 0..j {
                s -= a[i * k + m] * a[j * k + m];
            }
            a[i * k + j] = s / l_jj;
        }
    }
    Ok(())
}

/// In-place solve `L L^T x = b` given the lower-triangular factor `L`
/// produced by `cholesky_factor`. Forward + back substitution. The
/// strict upper triangle of `l` is ignored.
pub fn cholesky_solve(l: &[f64], k: usize, b: &mut [f64]) {
    debug_assert_eq!(l.len(), k * k);
    debug_assert_eq!(b.len(), k);

    // Forward substitution: solve L y = b → y = b after this loop.
    for i in 0..k {
        let mut s = b[i];
        for m in 0..i {
            s -= l[i * k + m] * b[m];
        }
        b[i] = s / l[i * k + i];
    }
    // Back substitution: solve L^T x = y → x = b after this loop.
    for i in (0..k).rev() {
        let mut s = b[i];
        for m in (i + 1)..k {
            s -= l[m * k + i] * b[m];
        }
        b[i] = s / l[i * k + i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 3×3 hand example: A = L L^T where L = [[2,0,0],[3,4,0],[5,6,7]].
    /// A[i,j] = Σ_m L[i,m]*L[j,m]:
    ///   A = [[4, 6, 10],
    ///        [6, 25, 39],
    ///        [10, 39, 110]]
    /// Solve A x = b where b = A [1,1,1]^T = [20, 70, 159] → x = [1,1,1].
    #[test]
    fn cholesky_3x3_hand() {
        let mut a: Vec<f64> = vec![
            4.0,  6.0,  10.0,
            6.0, 25.0,  39.0,
            10.0, 39.0, 110.0,
        ];
        cholesky_factor(&mut a, 3).expect("factor");
        // L is in lower triangle: diagonals 2,4,7; a[3]=3, a[6]=5, a[7]=6.
        assert!((a[0] - 2.0).abs() < 1e-12, "L[0,0] expected 2, got {}", a[0]);
        assert!((a[3] - 3.0).abs() < 1e-12, "L[1,0] expected 3, got {}", a[3]);
        assert!((a[4] - 4.0).abs() < 1e-12, "L[1,1] expected 4, got {}", a[4]);
        assert!((a[6] - 5.0).abs() < 1e-12, "L[2,0] expected 5, got {}", a[6]);
        assert!((a[7] - 6.0).abs() < 1e-12, "L[2,1] expected 6, got {}", a[7]);
        assert!((a[8] - 7.0).abs() < 1e-12, "L[2,2] expected 7, got {}", a[8]);

        // b = A [1,1,1]^T = [4+6+10, 6+25+39, 10+39+110] = [20, 70, 159]
        let mut b: Vec<f64> = vec![20.0, 70.0, 159.0];
        cholesky_solve(&a, 3, &mut b);
        for &v in b.iter() {
            assert!((v - 1.0).abs() < 1e-10, "expected 1.0, got {v}");
        }
    }

    /// Random k×k SPD matrices via A = M^T M + I; verify L L^T x ≈ b.
    #[test]
    fn cholesky_random_spd_fuzz() {
        // Deterministic pseudo-random for reproducibility.
        let seeds = [0u64, 1, 2, 3, 4];
        for &seed in &seeds {
            for &k in &[2usize, 3, 5, 10, 20] {
                // Build M k×k of "random" floats.
                let mut m = vec![0.0_f64; k * k];
                let mut s = seed.wrapping_mul(1103515245).wrapping_add(12345);
                for v in m.iter_mut() {
                    s = s.wrapping_mul(1103515245).wrapping_add(12345);
                    *v = (s as f64 / u64::MAX as f64) - 0.5;
                }
                // A = M^T M + k * I (SPD by construction).
                let mut a = vec![0.0_f64; k * k];
                for i in 0..k {
                    for j in 0..k {
                        let mut sum = 0.0_f64;
                        for r in 0..k {
                            sum += m[r * k + i] * m[r * k + j];
                        }
                        a[i * k + j] = sum;
                    }
                    a[i * k + i] += k as f64;
                }

                // True x and b = A x.
                let x_true: Vec<f64> = (0..k).map(|i| (i as f64 + 1.0) * 0.7).collect();
                let mut b = vec![0.0_f64; k];
                for i in 0..k {
                    let mut sum = 0.0_f64;
                    for j in 0..k {
                        sum += a[i * k + j] * x_true[j];
                    }
                    b[i] = sum;
                }

                // Factor + solve.
                let a_for_factor = a.clone();
                let mut a_mut = a_for_factor;
                cholesky_factor(&mut a_mut, k).unwrap_or_else(|_| {
                    panic!("factor failed at seed={seed} k={k}")
                });
                cholesky_solve(&a_mut, k, &mut b);

                // Compare.
                for (got, want) in b.iter().zip(x_true.iter()) {
                    assert!((got - want).abs() < 1e-8,
                        "seed={seed} k={k}: expected {want}, got {got}");
                }
            }
        }
    }
}
