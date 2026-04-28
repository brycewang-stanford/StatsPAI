//! Iterative Poisson separation detection.
//!
//! Drops rows in FE groups whose total y-sum is zero (Poisson regression
//! cannot identify FE coefficients for such groups; mu_g > 0 is required
//! by the model but Σ y_g = 0 yields no interior solution).
//!
//! The Python equivalent is ``_drop_separation`` in
//! ``src/statspai/fast/fepois.py``. This Rust port avoids the
//! ``np.unique`` + ``np.isin`` O(n log n) passes that dominate the
//! Python wall on large panels; the inner loop is pure O(n) per pass
//! over each FE dimension.

/// Iterative separation mask. Returns ``keep[i] = false`` iff
/// observation ``i`` lives in an FE group whose total y-sum is zero
/// (after iterating to fixed point).
///
/// Cost: O(n × n_iter × K) with n_iter ≤ ~3 typically. Allocates K
/// group-sum buffers (sized by g_per_fe) once and reuses them across
/// iterations.
pub fn separation_mask(
    y: &[f64],
    fe_codes: &[&[i64]],
    g_per_fe: &[usize],
) -> Vec<bool> {
    let n = y.len();
    let k = fe_codes.len();
    debug_assert_eq!(k, g_per_fe.len());
    for codes in fe_codes {
        debug_assert_eq!(codes.len(), n);
    }

    let mut keep = vec![true; n];
    if n == 0 || k == 0 {
        return keep;
    }

    // Pre-allocate per-FE group-sum scratch.
    let mut group_sums: Vec<Vec<f64>> =
        g_per_fe.iter().map(|&g| vec![0.0_f64; g]).collect();

    loop {
        let mut dropped_this_pass = false;
        for kdim in 0..k {
            // Reset scratch for this FE dim.
            for s in group_sums[kdim].iter_mut() {
                *s = 0.0;
            }
            // Accumulate y over the surviving rows.
            let codes = fe_codes[kdim];
            for i in 0..n {
                if keep[i] {
                    group_sums[kdim][codes[i] as usize] += y[i];
                }
            }
            // Find zero-sum groups (only consider groups that have at
            // least one surviving observation; we'd otherwise erroneously
            // mark already-empty groups as candidates).
            //
            // We don't track "group has ≥1 row" separately — instead we
            // check on the row-sweep below: a row in a zero-sum group
            // gets dropped, and a row in an empty group can't exist
            // (since the row IS in some group by construction).
            for i in 0..n {
                if !keep[i] {
                    continue;
                }
                if group_sums[kdim][codes[i] as usize] == 0.0 {
                    keep[i] = false;
                    dropped_this_pass = true;
                }
            }
        }
        if !dropped_this_pass {
            break;
        }
    }

    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All-positive y: no rows should be dropped.
    #[test]
    fn separation_no_drop() {
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let codes_a: Vec<i64> = vec![0, 0, 1, 1];
        let codes_b: Vec<i64> = vec![0, 1, 0, 1];
        let mask = separation_mask(&y, &[&codes_a, &codes_b], &[2, 2]);
        assert_eq!(mask, vec![true, true, true, true]);
    }

    /// Group 0 of FE-A has y = [0, 0]: those two rows must drop in
    /// one pass. Group sums on FE-B then become [1, 1] (only rows
    /// 2 and 3 survive); both groups stay populated.
    #[test]
    fn separation_one_pass() {
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let codes_a: Vec<i64> = vec![0, 0, 1, 1];
        let codes_b: Vec<i64> = vec![0, 1, 0, 1];
        let mask = separation_mask(&y, &[&codes_a, &codes_b], &[2, 2]);
        assert_eq!(mask, vec![false, false, true, true]);
    }

    /// Cascading drop: after dropping FE-A's zero-only group, FE-B
    /// then has a zero-only group that gets dropped on the second
    /// iteration.
    ///
    /// Setup: 6 rows, 2 FEs.
    ///   FE-A codes:  [0, 0, 1, 1, 2, 2]
    ///   FE-B codes:  [0, 1, 0, 1, 0, 0]
    ///   y:           [0, 0, 1, 0, 1, 1]
    ///
    /// Pass 1: FE-A group 0 has y-sum = 0 → drop rows 0, 1.
    /// After pass 1, surviving rows are 2, 3, 4, 5 with FE-B = [0, 1, 0, 0].
    /// Recompute FE-B group sums on survivors: g0 = 1+1+1 = 3, g1 = 0.
    /// Pass 2: FE-B group 1 has y-sum = 0 → drop row 3.
    /// After pass 2, surviving rows are 2, 4, 5.
    /// Pass 3: no more drops.
    ///
    /// Expected mask: [false, false, true, false, true, true].
    #[test]
    fn separation_iterative_cascade() {
        let y = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0];
        let codes_a: Vec<i64> = vec![0, 0, 1, 1, 2, 2];
        let codes_b: Vec<i64> = vec![0, 1, 0, 1, 0, 0];
        let mask = separation_mask(&y, &[&codes_a, &codes_b], &[3, 2]);
        assert_eq!(mask, vec![false, false, true, false, true, true]);
    }
}
