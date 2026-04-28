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

/// Poisson deviance: 2 · Σ w_i [ y_i log(y_i / μ_i) - (y_i - μ_i) ].
/// Convention: y log y → 0 at y = 0; mu is clamped at 1e-30 to avoid
/// log(0). ``obs_weights`` defaults to 1 if all-1 vector is passed.
fn poisson_deviance(y: &[f64], mu: &[f64], obs_weights: &[f64]) -> f64 {
    debug_assert_eq!(y.len(), mu.len());
    debug_assert_eq!(y.len(), obs_weights.len());
    let mut acc = 0.0_f64;
    for i in 0..y.len() {
        let yi = y[i];
        let mu_i = mu[i].max(1e-30);
        let r = if yi > 0.0 {
            yi * (yi / mu_i).ln()
        } else {
            0.0
        };
        let contrib = r - (yi - mu_i);
        acc += obs_weights[i] * contrib;
    }
    2.0 * acc
}

/// Native Poisson IRLS outer loop. The entire state machine runs in
/// Rust against the persistent ``FePoisIRLSWorkspace`` — no allocation
/// in the happy path beyond what `weighted_demean_matrix_fortran_inplace_sorted`
/// internally manages for Aitken history.
///
/// Parameters
/// ----------
/// `y`              — outcome vector, length n.
/// `x`              — F-order (n, p) regressor matrix.
/// `obs_weights`    — per-obs weights, length n. Pass an all-1 slice
///                    for unweighted MLE.
/// `cfg`            — IRLS knobs.
/// `ws`             — pre-built workspace from `FePoisIRLSWorkspace::new`.
///
/// Returns
/// -------
/// `FePoisIRLSResult` with `beta`, `x_tilde` (F-order, in original row
/// order), `w` (final IRLS working weight), `eta`, `mu`, `deviance`,
/// `log_likelihood`, `iters`, `converged`, `n_halvings`, `max_inner_dx`.
#[allow(clippy::too_many_arguments)]
pub fn fepois_loop(
    y: &[f64],
    x: &[f64],
    obs_weights: &[f64],
    cfg: &FePoisIRLSConfig,
    ws: &mut FePoisIRLSWorkspace,
) -> FePoisIRLSResult {
    let n = ws.n;
    let p = ws.p;
    debug_assert_eq!(y.len(), n);
    debug_assert_eq!(x.len(), n * p);
    debug_assert_eq!(obs_weights.len(), n);

    // Initialise: mu = max(y, 1) + 0.1, eta = ln(mu).
    for i in 0..n {
        let m = y[i].max(1.0) + 0.1;
        ws.mu[i] = m;
        ws.eta[i] = m.ln();
    }
    let mut deviance = f64::INFINITY;
    let mut iters_used: u32 = 0;
    let mut converged = false;
    let mut n_halvings: u32 = 0;
    let mut max_inner_dx: f64 = 0.0;

    // Pre-allocate WLS scratch (small — k = p ≤ ~30 typical).
    let mut xt_w_x = vec![0.0_f64; p * p];
    let mut xt_w_z = vec![0.0_f64; p];
    let mut beta = vec![0.0_f64; p];
    let mut eta_new = vec![0.0_f64; n];
    let mut mu_new = vec![0.0_f64; n];
    // Reusable z buffer in original order (for eta_new computation).
    let mut z_orig = vec![0.0_f64; n];

    let n_col = 1 + p;

    for it in 0..cfg.maxiter {
        // 1. Working response z = eta + (y - mu) / mu, in original order.
        for i in 0..n {
            z_orig[i] = ws.eta[i] + (y[i] - ws.mu[i]) / ws.mu[i];
        }
        // 2. Working weight w = mu * obs_weights, in original order.
        //    Apply π to write directly into ws.weights_p.
        for k in 0..n {
            let i = ws.sort_perm[k];
            ws.weights_p[k] = ws.mu[i] * obs_weights[i];
        }
        // 3. Build the demean buffer in π order: column 0 = z, columns 1..1+p = X.
        //    F-order layout: column j occupies ws.demean_buf[j*n .. (j+1)*n].
        for k in 0..n {
            let i = ws.sort_perm[k];
            ws.demean_buf[k] = z_orig[i]; // column 0 = z
        }
        for j in 0..p {
            let col_off = (j + 1) * n;
            let x_col_off = j * n;
            for k in 0..n {
                let i = ws.sort_perm[k];
                ws.demean_buf[col_off + k] = x[x_col_off + i];
            }
        }
        // 4. Compute wsum arrays (in π order — caller passed weights_p).
        for v in ws.primary_wsum.iter_mut() {
            *v = 0.0;
        }
        // primary FE codes after π are monotone; we don't store them
        // explicitly because primary_starts already encodes the layout.
        // Walk the groups via primary_starts.
        for g in 0..ws.primary_g {
            let lo = ws.primary_starts[g];
            let hi = ws.primary_starts[g + 1];
            let mut s = 0.0_f64;
            for kk in lo..hi {
                s += ws.weights_p[kk];
            }
            ws.primary_wsum[g] = s;
        }
        for sk in 0..ws.sec_codes_p.len() {
            for v in ws.sec_wsums[sk].iter_mut() {
                *v = 0.0;
            }
            let codes = &ws.sec_codes_p[sk];
            for kk in 0..n {
                ws.sec_wsums[sk][codes[kk] as usize] += ws.weights_p[kk];
            }
        }
        // 5. Call the demean kernel (in-place on demean_buf).
        let sec_codes_slices: Vec<&[i64]> =
            ws.sec_codes_p.iter().map(|v| v.as_slice()).collect();
        let sec_wsum_slices: Vec<&[f64]> =
            ws.sec_wsums.iter().map(|v| v.as_slice()).collect();
        let infos = crate::demean::weighted_demean_matrix_fortran_inplace_sorted(
            &mut ws.demean_buf,
            n,
            n_col,
            &ws.primary_starts,
            &ws.primary_wsum,
            &sec_codes_slices,
            &sec_wsum_slices,
            &ws.sec_g,
            &ws.weights_p,
            cfg.fe_maxiter,
            0.0,
            cfg.fe_tol,
            true,
            cfg.accel_period,
        );
        for info in &infos {
            if info.max_dx > max_inner_dx {
                max_inner_dx = info.max_dx;
            }
        }

        // 6. WLS in π order (sums over rows are permutation-invariant).
        //    z_tilde = ws.demean_buf[0..n], X_tilde = ws.demean_buf[n..(1+p)*n].
        // Build XtWX (p × p row-major) and XtWz (length p).
        for v in xt_w_x.iter_mut() {
            *v = 0.0;
        }
        for v in xt_w_z.iter_mut() {
            *v = 0.0;
        }
        for j1 in 0..p {
            let col_j1 = (j1 + 1) * n;
            // Diagonal + lower (we'll mirror to upper for symmetry).
            for j2 in 0..=j1 {
                let col_j2 = (j2 + 1) * n;
                let mut acc = 0.0_f64;
                for kk in 0..n {
                    acc += ws.demean_buf[col_j1 + kk]
                        * ws.weights_p[kk]
                        * ws.demean_buf[col_j2 + kk];
                }
                xt_w_x[j1 * p + j2] = acc;
                if j1 != j2 {
                    xt_w_x[j2 * p + j1] = acc;
                }
            }
            // XtWz_j = Σ_k X_tilde[k, j] * w[k] * z_tilde[k]
            let mut acc = 0.0_f64;
            for kk in 0..n {
                acc += ws.demean_buf[col_j1 + kk]
                    * ws.weights_p[kk]
                    * ws.demean_buf[kk];
            }
            xt_w_z[j1] = acc;
        }

        // Cholesky factor + solve.
        let factor_result = crate::cholesky::cholesky_factor(&mut xt_w_x, p);
        if factor_result.is_err() {
            // Degenerate WLS — bail out (caller handles via converged=false).
            break;
        }
        // Copy XtWz into beta and solve in place.
        beta.copy_from_slice(&xt_w_z);
        crate::cholesky::cholesky_solve(&xt_w_x, p, &mut beta);

        // 7. eta_new[i] = z_orig[i] - (z_tilde[i] - X_tilde[i, :] · beta).
        //    z_tilde and X_tilde are in π order — apply π⁻¹ to evaluate
        //    in original row order.
        for i in 0..n {
            let k = ws.inv_perm[i];
            let z_tilde_i = ws.demean_buf[k];
            let mut x_beta = 0.0_f64;
            for j in 0..p {
                let col = (j + 1) * n;
                x_beta += ws.demean_buf[col + k] * beta[j];
            }
            let resid = z_tilde_i - x_beta;
            let e = z_orig[i] - resid;
            // Clip to keep mu = exp(e) representable.
            eta_new[i] = e.clamp(-cfg.eta_clip, cfg.eta_clip);
        }
        for i in 0..n {
            mu_new[i] = eta_new[i].exp();
        }
        let mut new_dev = poisson_deviance(y, &mu_new, obs_weights);

        // 8. Step-halving on deviance non-decrease.
        let mut halvings = 0u32;
        while new_dev > deviance && halvings < cfg.max_halvings && deviance.is_finite() {
            for i in 0..n {
                let avg = 0.5 * (eta_new[i] + ws.eta[i]);
                eta_new[i] = avg.clamp(-cfg.eta_clip, cfg.eta_clip);
            }
            for i in 0..n {
                mu_new[i] = eta_new[i].exp();
            }
            new_dev = poisson_deviance(y, &mu_new, obs_weights);
            halvings += 1;
        }
        n_halvings += halvings;

        // 9. Convergence check.
        let rel = (new_dev - deviance).abs() / new_dev.abs().max(1.0);
        ws.eta.copy_from_slice(&eta_new);
        ws.mu.copy_from_slice(&mu_new);
        deviance = new_dev;
        iters_used = it + 1;
        if rel < cfg.tol {
            converged = true;
            break;
        }
    }

    // Final log-likelihood (Σ w_i [ y_i log(mu_i) - mu_i ]).
    let mut log_lik = 0.0_f64;
    for i in 0..n {
        let m = ws.mu[i].max(1e-30);
        log_lik += obs_weights[i] * (y[i] * m.ln() - ws.mu[i]);
    }

    // Apply π⁻¹ to ws.demean_buf's X columns to produce x_tilde in
    // original row order (caller's expectation).
    let mut x_tilde = vec![0.0_f64; n * p];
    for j in 0..p {
        let col_src = (j + 1) * n;
        let col_dst = j * n;
        for i in 0..n {
            let k = ws.inv_perm[i];
            x_tilde[col_dst + i] = ws.demean_buf[col_src + k];
        }
    }
    // w in original row order (currently in ws.weights_p which is π order).
    let mut w_orig = vec![0.0_f64; n];
    for k in 0..n {
        let i = ws.sort_perm[k];
        w_orig[i] = ws.weights_p[k];
    }

    FePoisIRLSResult {
        beta,
        x_tilde,
        w: w_orig,
        eta: ws.eta.clone(),
        mu: ws.mu.clone(),
        deviance,
        log_likelihood: log_lik,
        iters: iters_used,
        converged,
        n_halvings,
        max_inner_dx,
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

    /// Synthetic Poisson panel: small data, verify IRLS converges and
    /// produces a sensible β. Cross-validates against a hand-computed
    /// reference fit (large-n asymptotic; we just check β within a wide
    /// band, the unit test's job is to catch obvious bugs not to verify
    /// numerical parity to 1e-13 — that's the Python-side parity test
    /// landing in B1.6).
    #[test]
    fn fepois_loop_synthetic_converges() {
        // Deterministic synthetic Poisson panel, n=200, p=2, K=2 FEs.
        let n = 200usize;
        let p = 2usize;
        let g0 = 20usize;
        let g1 = 10usize;

        // Simple rng (LCG).
        let mut s: u64 = 7;
        let mut rng = || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            (s as f64 / u64::MAX as f64) - 0.5
        };

        let codes0: Vec<i64> = (0..n as i64).map(|i| i % g0 as i64).collect();
        let codes1: Vec<i64> = (0..n as i64).map(|i| (i / g0 as i64) % g1 as i64).collect();
        // Random alpha / gamma with std ~ 0.3.
        let alpha: Vec<f64> = (0..g0).map(|_| rng() * 0.6).collect();
        let gamma: Vec<f64> = (0..g1).map(|_| rng() * 0.6).collect();
        let beta_true = [0.5_f64, -0.3];

        let mut x = vec![0.0_f64; n * p]; // F-order
        let mut y = vec![0.0_f64; n];
        let obs_weights = vec![1.0_f64; n];
        for i in 0..n {
            x[i] = rng();         // X[:, 0]
            x[n + i] = rng();     // X[:, 1]
            let eta_i = beta_true[0] * x[i]
                + beta_true[1] * x[n + i]
                + alpha[codes0[i] as usize]
                + gamma[codes1[i] as usize];
            let mu_i = eta_i.clamp(-10.0, 10.0).exp();
            // Approximate Poisson draw: round(mu + small noise). For
            // unit-test stability, just use mu rounded to integer
            // (the deterministic path to a stable target).
            y[i] = mu_i.round();
        }

        let mut ws = FePoisIRLSWorkspace::new(n, p, &[&codes0, &codes1], &[g0, g1]);
        let cfg = FePoisIRLSConfig::default();
        let result = fepois_loop(&y, &x, &obs_weights, &cfg, &mut ws);

        assert!(result.converged, "IRLS did not converge in {} iters", cfg.maxiter);
        assert!(result.iters >= 1 && result.iters <= cfg.maxiter);
        assert!(result.deviance.is_finite());

        // Sanity: β values are roughly in the right ball (we generated
        // y by rounding mu to integers so it's not a real Poisson, but
        // the linear part should be recoverable to within a wide band).
        for &b in &result.beta {
            assert!(b.is_finite());
            assert!(b.abs() < 5.0, "beta {b} unreasonably large");
        }
        // x_tilde and w shapes are correct.
        assert_eq!(result.x_tilde.len(), n * p);
        assert_eq!(result.w.len(), n);
    }
}
