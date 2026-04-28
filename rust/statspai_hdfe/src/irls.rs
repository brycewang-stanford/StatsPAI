//! Native Rust IRLS state machine for ``sp.fast.fepois``.
//!
//! Phase B1: this module hosts ``FePoisIRLSConfig`` (knobs),
//! ``FePoisIRLSResult`` (outputs), ``FePoisIRLSWorkspace`` (per-fepois
//! persistent buffers), and ``fepois_loop`` (the main body, lands in
//! B1.3). The Python ``fepois()`` parses the formula + runs the
//! singleton/separation pre-passes, then calls the PyO3 binding
//! ``fepois_irls`` (B1.4) which delegates here.
//!
//! Design intent: eliminate the 12 PyO3 round-trips per fepois call
//! (one per IRLS iter × 2 — z and X) by keeping the entire IRLS state
//! machine in Rust. The workspace pre-allocates all per-iter buffers
//! once at construction so the inner loop is allocation-free in the
//! happy path.

/// Knobs for the IRLS outer loop.
#[derive(Clone, Debug)]
pub struct FePoisIRLSConfig {
    /// Cap on outer IRLS iterations.
    pub maxiter: u32,
    /// Relative-deviance convergence tolerance for the outer loop.
    pub tol: f64,
    /// Relative tolerance for the inner AP demean.
    pub fe_tol: f64,
    /// Cap on inner AP iterations per call.
    pub fe_maxiter: u32,
    /// Symmetric clip applied to ``eta`` to keep ``mu = exp(eta)`` in
    /// double-precision range; matches the Python path's hard-coded 30.
    pub eta_clip: f64,
    /// Aitken extrapolation period inside the inner AP demean.
    pub accel_period: u32,
    /// Cap on step-halving attempts when deviance fails to decrease.
    pub max_halvings: u32,
}

impl Default for FePoisIRLSConfig {
    fn default() -> Self {
        Self {
            maxiter: 50,
            tol: 1e-8,
            fe_tol: 1e-10,
            fe_maxiter: 1000,
            eta_clip: 30.0,
            accel_period: 5,
            max_halvings: 10,
        }
    }
}

/// Result of one ``fepois_loop`` call. ``x_tilde`` and ``w`` are the
/// final-iteration demeaned X and IRLS working weight; the Python
/// caller uses them to compute the requested vcov (IID / HC1 / CR1)
/// without re-running the within transform.
#[derive(Debug)]
pub struct FePoisIRLSResult {
    pub beta: Vec<f64>,           // length p
    pub x_tilde: Vec<f64>,        // F-order (n, p)
    pub w: Vec<f64>,              // final IRLS working weights, length n
    pub eta: Vec<f64>,            // final eta, length n
    pub mu: Vec<f64>,             // final mu, length n
    pub deviance: f64,
    pub log_likelihood: f64,
    pub iters: u32,
    pub converged: bool,
    pub n_halvings: u32,
    pub max_inner_dx: f64,
}

/// Persistent per-fepois scratch. Allocated once at construction and
/// reused across all IRLS iters; the inner loop does no further heap
/// allocation in the happy path.
///
/// ``primary_starts`` and ``sec_codes_p`` are populated by the
/// constructor (sort permutation applied once). Per-iter buffers
/// (``weights_p``, ``primary_wsum``, ``sec_wsums``, ``z_buf``,
/// ``x_buf``, ``before``, ``aitken_hist``) are sized once and
/// rewritten in place each iter.
#[derive(Debug)]
pub struct FePoisIRLSWorkspace {
    pub n: usize,
    pub p: usize,
    pub k_fe: usize,
    pub primary_idx: usize,
    pub primary_g: usize,
    pub sec_g: Vec<usize>,
    /// Permutation π such that ``primary_codes[π[k]]`` is non-decreasing.
    pub sort_perm: Vec<usize>,
    /// Inverse permutation: ``inv_perm[π[k]] = k``.
    pub inv_perm: Vec<usize>,
    /// Group-start offsets for the primary FE under π. Length G1 + 1.
    pub primary_starts: Vec<usize>,
    /// Secondary FE codes under π. Length K-1, each inner Vec length n.
    pub sec_codes_p: Vec<Vec<i64>>,
    // Per-iter buffers (allocated once):
    /// IRLS working weight in π order.
    pub weights_p: Vec<f64>,
    /// Working response z = eta + (y - mu)/mu, in π order.
    pub z_p: Vec<f64>,
    /// Combined demean matrix (z stacked with X), F-order, n × (1+p), in π order.
    pub demean_buf: Vec<f64>,
    /// Primary FE wsum, recomputed each iter.
    pub primary_wsum: Vec<f64>,
    /// Secondary FE wsums, recomputed each iter.
    pub sec_wsums: Vec<Vec<f64>>,
    /// AP-loop "before" snapshot (per column reuse).
    pub before: Vec<f64>,
    /// Aitken history slots (3 × n_col).
    pub aitken_hist: Vec<Vec<f64>>,
    /// Secondary FE scratch (per-FE, sized by sec_g).
    pub sec_scratch: Vec<Vec<f64>>,
    /// IRLS state vectors.
    pub eta: Vec<f64>,
    pub mu: Vec<f64>,
}

impl FePoisIRLSWorkspace {
    /// Construct a workspace by computing the primary-FE sort
    /// permutation, the inverse permutation, and the per-FE
    /// secondary code arrays under π. All per-iter buffers are
    /// pre-allocated to their final sizes.
    ///
    /// Parameters
    /// ----------
    /// `n`              — observations (after singleton/separation drops).
    /// `p`              — number of columns of X.
    /// `fe_codes`       — K dense int64 code arrays, each length n.
    /// `g_per_fe`       — cardinality of each FE.
    pub fn new(
        n: usize,
        p: usize,
        fe_codes: &[&[i64]],
        g_per_fe: &[usize],
    ) -> Self {
        let k_fe = fe_codes.len();
        debug_assert_eq!(k_fe, g_per_fe.len());
        debug_assert!(k_fe >= 1, "fepois_loop requires at least one FE");

        // Pick highest-cardinality FE as primary.
        let mut primary_idx = 0usize;
        for k in 1..k_fe {
            if g_per_fe[k] > g_per_fe[primary_idx] {
                primary_idx = k;
            }
        }
        let primary_g = g_per_fe[primary_idx];

        // Counting-sort permutation by primary FE codes.
        let primary_codes = fe_codes[primary_idx];
        let sort_perm = crate::sort_perm::primary_fe_sort_perm(primary_codes, primary_g);
        let mut inv_perm = vec![0usize; n];
        for (k, &pi) in sort_perm.iter().enumerate() {
            inv_perm[pi] = k;
        }

        // Apply π to primary codes to derive group_starts.
        let primary_codes_p: Vec<i64> =
            sort_perm.iter().map(|&i| primary_codes[i]).collect();
        let primary_starts = crate::sort_perm::group_starts_from_codes_sorted(
            &primary_codes_p, primary_g,
        );

        // Apply π to secondary FE codes.
        let mut sec_codes_p: Vec<Vec<i64>> = Vec::with_capacity(k_fe - 1);
        let mut sec_g: Vec<usize> = Vec::with_capacity(k_fe - 1);
        for k in 0..k_fe {
            if k == primary_idx {
                continue;
            }
            let codes_k = fe_codes[k];
            sec_codes_p.push(sort_perm.iter().map(|&i| codes_k[i]).collect());
            sec_g.push(g_per_fe[k]);
        }

        // Pre-allocate per-iter buffers.
        let n_col = 1 + p; // z + p X columns
        Self {
            n,
            p,
            k_fe,
            primary_idx,
            primary_g,
            sec_g: sec_g.clone(),
            sort_perm,
            inv_perm,
            primary_starts,
            sec_codes_p,
            weights_p: vec![0.0; n],
            z_p: vec![0.0; n],
            demean_buf: vec![0.0; n * n_col],
            primary_wsum: vec![0.0; primary_g],
            sec_wsums: sec_g.iter().map(|&g| vec![0.0; g]).collect(),
            before: vec![0.0; n],
            aitken_hist: (0..3).map(|_| vec![0.0; n]).collect(),
            sec_scratch: sec_g.iter().map(|&g| vec![0.0; g]).collect(),
            eta: vec![0.0; n],
            mu: vec![0.0; n],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Basic shape sanity: workspace allocates correct per-buffer sizes.
    #[test]
    fn workspace_shapes() {
        let n = 50usize;
        let p = 2usize;
        let g0 = 10usize;
        let g1 = 5usize;
        let codes0: Vec<i64> = (0..n as i64).map(|i| i % g0 as i64).collect();
        let codes1: Vec<i64> = (0..n as i64).map(|i| i % g1 as i64).collect();
        let ws = FePoisIRLSWorkspace::new(n, p, &[&codes0, &codes1], &[g0, g1]);

        // Primary should be FE 0 (higher cardinality).
        assert_eq!(ws.primary_idx, 0);
        assert_eq!(ws.primary_g, g0);
        assert_eq!(ws.k_fe, 2);

        // Per-buffer sizes.
        assert_eq!(ws.sort_perm.len(), n);
        assert_eq!(ws.inv_perm.len(), n);
        assert_eq!(ws.primary_starts.len(), g0 + 1);
        assert_eq!(ws.sec_codes_p.len(), 1);
        assert_eq!(ws.sec_codes_p[0].len(), n);
        assert_eq!(ws.weights_p.len(), n);
        assert_eq!(ws.z_p.len(), n);
        assert_eq!(ws.demean_buf.len(), n * (1 + p));
        assert_eq!(ws.primary_wsum.len(), g0);
        assert_eq!(ws.sec_wsums.len(), 1);
        assert_eq!(ws.sec_wsums[0].len(), g1);
        assert_eq!(ws.aitken_hist.len(), 3);

        // inv_perm round-trip: inv_perm[sort_perm[k]] == k.
        for k in 0..n {
            assert_eq!(ws.inv_perm[ws.sort_perm[k]], k);
        }
    }

    #[test]
    fn config_defaults_match_python() {
        let cfg = FePoisIRLSConfig::default();
        assert_eq!(cfg.maxiter, 50);
        assert!((cfg.tol - 1e-8).abs() < 1e-20);
        assert!((cfg.fe_tol - 1e-10).abs() < 1e-20);
        assert_eq!(cfg.eta_clip, 30.0);
        assert_eq!(cfg.max_halvings, 10);
    }
}
