//! Iterative singleton-row detection for K-way HDFE.
//!
//! A "singleton" observation is one whose code in some FE dimension appears
//! only once (after any prior drops). Such rows contribute zero
//! within-variation and bias the cluster-robust degrees-of-freedom; the
//! standard remedy (Correia 2015) is to prune them iteratively until no
//! singletons remain. In practice 1-3 passes suffice even on nested panels.
//!
//! Returns a boolean keep mask aligned with the original ``n`` rows.

/// Build a keep-mask for K-way FE. ``fe_codes`` is a slice of K row arrays,
/// each of length ``n``. Codes do not need to be densely packed — we only
/// rely on ``max_code + 1`` as the cardinality upper bound (we compute it).
pub fn detect_singletons(fe_codes: &[&[i64]]) -> Vec<bool> {
    if fe_codes.is_empty() {
        return Vec::new();
    }
    let n = fe_codes[0].len();
    for codes in fe_codes.iter() {
        assert_eq!(codes.len(), n, "all FE columns must have length n");
    }

    let mut keep = vec![true; n];
    let max_codes: Vec<usize> = fe_codes
        .iter()
        .map(|c| c.iter().copied().max().unwrap_or(-1).max(-1) as usize + 1)
        .collect();

    loop {
        let mut dropped = false;
        for (k, codes) in fe_codes.iter().enumerate() {
            let g = max_codes[k];
            if g == 0 {
                continue;
            }
            let mut counts = vec![0_i64; g];
            for i in 0..n {
                if keep[i] {
                    counts[codes[i] as usize] += 1;
                }
            }
            for i in 0..n {
                if keep[i] && counts[codes[i] as usize] == 1 {
                    keep[i] = false;
                    dropped = true;
                }
            }
        }
        if !dropped {
            break;
        }
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_singletons() {
        let codes: Vec<i64> = vec![0, 0, 1, 1, 2, 2];
        let keep = detect_singletons(&[&codes]);
        assert!(keep.iter().all(|&b| b));
    }

    #[test]
    fn one_singleton_dropped() {
        // group 2 only appears once.
        let codes: Vec<i64> = vec![0, 0, 1, 1, 2];
        let keep = detect_singletons(&[&codes]);
        assert_eq!(keep, vec![true, true, true, true, false]);
    }

    #[test]
    fn cascading_singleton_drop() {
        // After dropping the row with i=2 (singleton in i), j=99 becomes a
        // singleton in j and must also drop in the next pass.
        let i: Vec<i64> = vec![0, 0, 1, 1, 2];
        let j: Vec<i64> = vec![0, 1, 0, 1, 99];
        let keep = detect_singletons(&[&i, &j]);
        assert_eq!(keep, vec![true, true, true, true, false]);
    }

    #[test]
    fn empty() {
        let keep: Vec<bool> = detect_singletons(&[]);
        assert!(keep.is_empty());
    }
}
