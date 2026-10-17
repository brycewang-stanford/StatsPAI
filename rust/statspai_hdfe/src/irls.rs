//! Native Rust IRLS for ``sp.fast.fepois``.
//!
//! The Python ``fepois()`` parses the formula and runs the singleton and
//! separation pre-passes, then calls the PyO3 binding ``fepois_irls``,
//! which runs the whole IRLS loop here: no FFI round trip per iteration.
//!
//! Each iteration residualises the working response and every regressor
//! on the FEs with the working weights ``mu * obs_weights``, using the
//! coefficient-space solver in ``coef.rs``. The FE structure is built
//! once; each iteration only re-weights it, and every column restarts
//! from its FE coefficients of the previous iteration (a warm start, as
//! in fixest and ppmlhdfe), so late iterations need few sweeps. Columns
//! are residualised in parallel.

use rayon::prelude::*;

use crate::coef::{demean_column, Problem, Workspace};

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
    pub beta: Vec<f64>,    // length p
    pub x_tilde: Vec<f64>, // F-order (n, p)
    pub w: Vec<f64>,       // final IRLS working weights, length n
    pub eta: Vec<f64>,     // final eta, length n
    pub mu: Vec<f64>,      // final mu, length n
    pub deviance: f64,
    pub log_likelihood: f64,
    pub iters: u32,
    pub converged: bool,
    pub n_halvings: u32,
    /// Largest final change reported by any inner demean, over all
    /// iterations.
    pub max_inner_dx: f64,
    /// Whether every inner demean of the last iteration converged.
    pub fe_converged: bool,
    /// The weighted normal equations were numerically singular (a
    /// regressor is collinear with the FEs or the other regressors); the
    /// returned ``beta`` is the last one that could be solved.
    pub singular: bool,
}

/// Poisson deviance: 2 · Σ w_i [ y_i log(y_i / μ_i) - (y_i - μ_i) ].
/// Convention: y log y → 0 at y = 0; mu is clamped at 1e-30 to avoid
/// log(0).
fn poisson_deviance(y: &[f64], mu: &[f64], obs_weights: &[f64]) -> f64 {
    debug_assert_eq!(y.len(), mu.len());
    debug_assert_eq!(y.len(), obs_weights.len());
    let mut acc = 0.0_f64;
    for i in 0..y.len() {
        let yi = y[i];
        let mu_i = mu[i].max(1e-30);
        let r = if yi > 0.0 { yi * (yi / mu_i).ln() } else { 0.0 };
        let contrib = r - (yi - mu_i);
        acc += obs_weights[i] * contrib;
    }
    2.0 * acc
}

/// Native Poisson IRLS outer loop.
///
/// Parameters
/// ----------
/// `y`           — outcome vector, length n.
/// `x`           — F-order (n, p) regressor matrix.
/// `obs_weights` — per-obs weights, length n (all 1 for the unweighted MLE).
/// `fe_codes`    — K ≥ 1 dense, validated code arrays, each length n.
/// `g_per_fe`    — number of levels of each FE.
/// `cfg`         — IRLS knobs.
pub fn fepois_loop(
    y: &[f64],
    x: &[f64],
    obs_weights: &[f64],
    fe_codes: &[&[i64]],
    g_per_fe: &[usize],
    cfg: &FePoisIRLSConfig,
) -> FePoisIRLSResult {
    let n = y.len();
    let p = if n == 0 { 0 } else { x.len() / n };
    debug_assert_eq!(x.len(), n * p);
    debug_assert_eq!(obs_weights.len(), n);

    let mut mu: Vec<f64> = y.iter().map(|&v| v.max(1.0) + 0.1).collect();
    let mut eta: Vec<f64> = mu.iter().map(|m| m.ln()).collect();
    let mut w: Vec<f64> = (0..n).map(|i| mu[i] * obs_weights[i]).collect();

    let mut problem = Problem::new(fe_codes.to_vec(), g_per_fe.to_vec(), Some(w.clone()));
    // Column 0 is the working response, columns 1..=p are the regressors.
    let mut workspaces: Vec<Workspace> = (0..=p).map(|_| Workspace::new(&problem, n)).collect();
    let mut buf = vec![0.0_f64; n * (p + 1)];

    let mut deviance = f64::INFINITY;
    let mut iters_used: u32 = 0;
    let mut converged = false;
    let mut n_halvings: u32 = 0;
    let mut max_inner_dx: f64 = 0.0;
    let mut fe_converged = true;
    let mut singular = false;

    let mut xt_w_x = vec![0.0_f64; p * p];
    let mut beta = vec![0.0_f64; p];
    let mut beta_try = vec![0.0_f64; p];
    let mut eta_new = vec![0.0_f64; n];
    let mut mu_new = vec![0.0_f64; n];
    let mut z = vec![0.0_f64; n];

    for it in 0..cfg.maxiter {
        // 1. Working response and weight.
        for i in 0..n {
            z[i] = eta[i] + (y[i] - mu[i]) / mu[i];
            w[i] = mu[i] * obs_weights[i];
        }
        if it > 0 {
            problem.reweight(&w);
        }
        // 2. Residualise z and X on the FEs (warm-started per column).
        buf[..n].copy_from_slice(&z);
        buf[n..].copy_from_slice(x);
        let pr = &problem;
        let infos: Vec<_> = buf
            .par_chunks_mut(n.max(1))
            .zip(workspaces.par_iter_mut())
            .map(|(col, ws)| demean_column(pr, ws, col, cfg.fe_maxiter, 0.0, cfg.fe_tol))
            .collect();
        fe_converged = infos.iter().all(|d| d.converged);
        for d in &infos {
            if d.max_dx > max_inner_dx {
                max_inner_dx = d.max_dx;
            }
        }

        // 3. Weighted normal equations on the residualised columns.
        let (zt, xt) = buf.split_at(n);
        for j1 in 0..p {
            let c1 = &xt[j1 * n..(j1 + 1) * n];
            for j2 in 0..=j1 {
                let c2 = &xt[j2 * n..(j2 + 1) * n];
                let mut acc = 0.0_f64;
                for i in 0..n {
                    acc += c1[i] * w[i] * c2[i];
                }
                xt_w_x[j1 * p + j2] = acc;
                xt_w_x[j2 * p + j1] = acc;
            }
            let mut acc = 0.0_f64;
            for i in 0..n {
                acc += c1[i] * w[i] * zt[i];
            }
            beta_try[j1] = acc;
        }
        // A regressor the FEs absorb leaves only rounding noise: compare
        // its residualised weighted norm with the raw one.
        let absorbed_tol = (10.0 * cfg.fe_tol).powi(2).max(1e-18);
        let mut absorbed = false;
        for j in 0..p {
            let raw = &x[j * n..(j + 1) * n];
            let mut ss = 0.0_f64;
            for i in 0..n {
                ss += w[i] * raw[i] * raw[i];
            }
            if !(xt_w_x[j * p + j] > absorbed_tol * ss) {
                absorbed = true;
            }
        }
        if absorbed || crate::cholesky::cholesky_factor(&mut xt_w_x, p).is_err() {
            singular = true;
            break;
        }
        crate::cholesky::cholesky_solve(&xt_w_x, p, &mut beta_try);
        beta.copy_from_slice(&beta_try);

        // 4. eta_new = z - (z_tilde - X_tilde beta), clipped.
        for i in 0..n {
            let mut xb = 0.0_f64;
            for j in 0..p {
                xb += xt[j * n + i] * beta[j];
            }
            let e = z[i] - (zt[i] - xb);
            eta_new[i] = e.clamp(-cfg.eta_clip, cfg.eta_clip);
        }
        for i in 0..n {
            mu_new[i] = eta_new[i].exp();
        }
        let mut new_dev = poisson_deviance(y, &mu_new, obs_weights);

        // 5. Step-halving on deviance non-decrease.
        let mut halvings = 0u32;
        while new_dev > deviance && halvings < cfg.max_halvings && deviance.is_finite() {
            for i in 0..n {
                let avg = 0.5 * (eta_new[i] + eta[i]);
                eta_new[i] = avg.clamp(-cfg.eta_clip, cfg.eta_clip);
            }
            for i in 0..n {
                mu_new[i] = eta_new[i].exp();
            }
            new_dev = poisson_deviance(y, &mu_new, obs_weights);
            halvings += 1;
        }
        n_halvings += halvings;

        // 6. Convergence check on the relative deviance change.
        let rel = (new_dev - deviance).abs() / new_dev.abs().max(1.0);
        eta.copy_from_slice(&eta_new);
        mu.copy_from_slice(&mu_new);
        deviance = new_dev;
        iters_used = it + 1;
        if rel < cfg.tol {
            converged = true;
            break;
        }
    }

    let mut log_lik = 0.0_f64;
    for i in 0..n {
        let m = mu[i].max(1e-30);
        log_lik += obs_weights[i] * (y[i] * m.ln() - mu[i]);
    }
    let x_tilde = buf.split_off(n);

    FePoisIRLSResult {
        beta,
        x_tilde,
        w,
        eta,
        mu,
        deviance,
        log_likelihood: log_lik,
        iters: iters_used,
        converged: converged && !singular,
        n_halvings,
        max_inner_dx,
        fe_converged,
        singular,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_match_python() {
        let cfg = FePoisIRLSConfig::default();
        assert_eq!(cfg.maxiter, 50);
        assert!((cfg.tol - 1e-8).abs() < 1e-20);
        assert!((cfg.fe_tol - 1e-10).abs() < 1e-20);
        assert_eq!(cfg.fe_maxiter, 1000);
        assert_eq!(cfg.eta_clip, 30.0);
        assert_eq!(cfg.accel_period, 5);
        assert_eq!(cfg.max_halvings, 10);
    }

    /// Poisson panel with known parameters and y = mu exactly (no noise):
    /// the MLE then reproduces the true coefficients, so beta must be
    /// recovered to the convergence tolerance.
    #[test]
    fn noiseless_panel_recovers_beta() {
        let n = 400usize;
        let p = 2usize;
        let (g0, g1) = (20usize, 10usize);
        let mut s: u64 = 7;
        let mut rng = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        let codes0: Vec<i64> = (0..n as i64).map(|i| i % g0 as i64).collect();
        let codes1: Vec<i64> = (0..n as i64).map(|i| (i / 7) % g1 as i64).collect();
        let alpha: Vec<f64> = (0..g0).map(|_| rng() * 0.6).collect();
        let gamma: Vec<f64> = (0..g1).map(|_| rng() * 0.6).collect();
        let beta_true = [0.5_f64, -0.3];
        let mut x = vec![0.0_f64; n * p];
        let mut y = vec![0.0_f64; n];
        for i in 0..n {
            x[i] = rng();
            x[n + i] = rng();
            let e = beta_true[0] * x[i]
                + beta_true[1] * x[n + i]
                + alpha[codes0[i] as usize]
                + gamma[codes1[i] as usize];
            y[i] = (1.0 + e).exp();
        }
        let ones = vec![1.0_f64; n];
        let mut cfg = FePoisIRLSConfig::default();
        cfg.tol = 1e-14;
        cfg.fe_tol = 1e-14;
        let res = fepois_loop(&y, &x, &ones, &[&codes0, &codes1], &[g0, g1], &cfg);
        assert!(res.converged, "{res:?}");
        assert!(!res.singular);
        assert!(res.fe_converged);
        for j in 0..p {
            assert!((res.beta[j] - beta_true[j]).abs() < 1e-7, "beta {:?}", res.beta);
        }
        assert_eq!(res.x_tilde.len(), n * p);
    }

    #[test]
    fn regressor_absorbed_by_fe_is_reported_singular() {
        let n = 60usize;
        let codes0: Vec<i64> = (0..n as i64).map(|i| i % 6).collect();
        // x is constant within FE 0: residualised it is zero.
        let x: Vec<f64> = codes0.iter().map(|&c| c as f64 * 0.5).collect();
        let y: Vec<f64> = (0..n).map(|i| ((i % 5) + 1) as f64).collect();
        let ones = vec![1.0_f64; n];
        let res = fepois_loop(&y, &x, &ones, &[&codes0], &[6], &FePoisIRLSConfig::default());
        assert!(res.singular);
        assert!(!res.converged);
    }
}
