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
    // ``cursor`` tracks the next-free slot per group; we clone ``starts`` so
    // the prefix-sum offsets remain available to callers that want both
    // ``perm`` and the group-starts layout.
    let mut cursor = starts.clone();
    for i in 0..n {
        let g = codes[i] as usize;
        perm[cursor[g]] = i;
        cursor[g] += 1;
    }
    perm
}

/// Returns a length-(n_groups + 1) start-index array such that group g
/// occupies the half-open range ``codes_sorted[starts[g]..starts[g+1]]``.
/// Empty groups (no observations with that code) collapse to
/// ``starts[g] == starts[g+1]`` and yield an empty slice.
///
/// Caller may also reconstruct this from `counts` via prefix-sum; this
/// helper is for callers that only have already-permuted `codes_sorted`.
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

    /// Group 1 has no observations: its half-open range
    /// ``codes_sorted[starts[1]..starts[2]]`` collapses to empty.
    #[test]
    fn group_starts_handles_empty_interior_group() {
        let codes_sorted: Vec<i64> = vec![0, 0, 2, 2];
        let starts = group_starts_from_codes_sorted(&codes_sorted, 3);
        assert_eq!(starts, vec![0, 2, 2, 4]);
    }
}
