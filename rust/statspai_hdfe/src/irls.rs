//! Native Rust IRLS for ``sp.fast.fepois`` and the binomial ``sp.feglm``.
//!
//! The Python side parses the formula and runs the singleton and
//! separation pre-passes, then calls the PyO3 binding (``fepois_irls`` /
//! ``feglm_irls``), which runs the whole IRLS loop here: no FFI round trip
//! per iteration.
//!
//! Each iteration residualises the working response and every regressor
//! on the FEs with the working weights, using the coefficient-space solver
//! in ``coef.rs``. The FE structure is built once; each iteration only
//! re-weights it, and every column restarts from its FE coefficients of
//! the previous iteration (a warm start, as in fixest and ppmlhdfe), so
//! late iterations need few sweeps. Columns are residualised in parallel
//! and every pass over the rows is split into fixed chunks (``par.rs``), so
//! the result does not depend on the number of threads.

use rayon::prelude::*;

use crate::coef::{demean_column, Problem, Workspace};
use crate::par;

/// The GLM family of the IRLS loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// Poisson, log link.
    Poisson,
    /// Binomial, logit link.
    Logit,
    /// Binomial, probit link.
    Probit,
}

impl Family {
    pub fn parse(name: &str) -> Option<Family> {
        match name {
            "poisson" => Some(Family::Poisson),
            "logit" => Some(Family::Logit),
            "probit" => Some(Family::Probit),
            _ => None,
        }
    }

    /// ``(mu, dmu/deta)`` at ``eta``.
    #[inline]
    fn mean(self, eta: f64) -> (f64, f64) {
        match self {
            Family::Poisson => {
                let m = eta.exp();
                (m, m)
            }
            Family::Logit => {
                let m = 1.0 / (1.0 + (-eta).exp());
                (m, m * (1.0 - m))
            }
            Family::Probit => {
                let m = 0.5 * libm::erfc(-eta * std::f64::consts::FRAC_1_SQRT_2);
                let d = (-0.5 * eta * eta).exp() * 0.398_942_280_401_432_7;
                (m, d)
            }
        }
    }

    /// Variance function ``V(mu)``.
    #[inline]
    fn variance(self, mu: f64) -> f64 {
        match self {
            Family::Poisson => mu,
            Family::Logit | Family::Probit => (mu * (1.0 - mu)).max(1e-300),
        }
    }

    /// Starting ``eta`` from the outcome (glm's ``mustart``).
    #[inline]
    fn start(self, y: f64, w: f64) -> f64 {
        match self {
            Family::Poisson => (y.max(1.0) + 0.1).ln(),
            Family::Logit | Family::Probit => {
                let m = (w * y + 0.5) / (w + 1.0);
                let l = (m / (1.0 - m)).ln();
                // A probit index is about 1/1.6 of a logit one; only the
                // starting point, not the solution, depends on it.
                if self == Family::Probit {
                    l / 1.6
                } else {
                    l
                }
            }
        }
    }

    /// Unit deviance contribution ``d(y, mu)``.
    #[inline]
    fn unit_deviance(self, y: f64, mu: f64) -> f64 {
        match self {
            Family::Poisson => {
                let m = mu.max(1e-30);
                let r = if y > 0.0 { y * (y / m).ln() } else { 0.0 };
                2.0 * (r - (y - m))
            }
            Family::Logit | Family::Probit => {
                let m = mu.clamp(1e-300, 1.0 - 1e-16);
                let a = if y > 0.0 { y * (y / m).ln() } else { 0.0 };
                let b = if y < 1.0 {
                    (1.0 - y) * ((1.0 - y) / (1.0 - m)).ln()
                } else {
                    0.0
                };
                2.0 * (a + b)
            }
        }
    }

    /// Log-likelihood contribution (up to a term in ``y`` only).
    #[inline]
    fn loglik(self, y: f64, mu: f64) -> f64 {
        match self {
            Family::Poisson => y * mu.max(1e-30).ln() - mu,
            Family::Logit | Family::Probit => {
                let m = mu.clamp(1e-300, 1.0 - 1e-16);
                let a = if y > 0.0 { y * m.ln() } else { 0.0 };
                let b = if y < 1.0 {
                    (1.0 - y) * (1.0 - m).ln()
                } else {
                    0.0
                };
                a + b
            }
        }
    }

    /// Denominator of the relative deviance change: fixest's
    /// ``0.1 + |dev|`` for the binomial families; Poisson keeps the
    /// ``max(1, |dev|)`` of crate API 2.
    #[inline]
    fn rel_change(self, new: f64, old: f64) -> f64 {
        match self {
            Family::Poisson => (new - old).abs() / new.abs().max(1.0),
            Family::Logit | Family::Probit => (new - old).abs() / (0.1 + new.abs()),
        }
    }
}

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
    /// Symmetric clip applied to ``eta``.
    pub eta_clip: f64,
    /// Accepted for compatibility (the coefficient solver accelerates
    /// every step).
    pub accel_period: u32,
    /// Cap on step-halvings per outer iteration.
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

/// Outcome of the IRLS loop.
#[derive(Debug)]
pub struct FePoisIRLSResult {
    pub beta: Vec<f64>,    // length p
    pub x_tilde: Vec<f64>, // F-order (n, p), residualised with ``w``
    pub w: Vec<f64>,       // working weights of the last iteration, length n
    pub eta: Vec<f64>,     // final eta, length n
    pub mu: Vec<f64>,      // final mu, length n
    pub deviance: f64,
    pub log_likelihood: f64,
    pub iters: u32,
    pub converged: bool,
    pub n_halvings: u32,
    pub max_inner_dx: f64,
    pub fe_converged: bool,
    pub singular: bool,
}

fn deviance(fam: Family, y: &[f64], mu: &[f64], obs_weights: &[f64]) -> f64 {
    par::sum(y.len(), |r| {
        let mut acc = 0.0_f64;
        for i in r {
            acc += obs_weights[i] * fam.unit_deviance(y[i], mu[i]);
        }
        acc
    })
}

/// ``eta[i] = clamp(z[i] - (zt[i] - sum_j xt[j, i] beta[j]))``, row-parallel.
fn update_eta(z: &[f64], zt: &[f64], xt: &[f64], beta: &[f64], clip: f64, eta: &mut [f64]) {
    let n = z.len();
    let p = beta.len();
    par::for_each_mut(eta, |i, e| {
        let mut xb = 0.0_f64;
        for j in 0..p {
            xb += xt[j * n + i] * beta[j];
        }
        *e = (z[i] - (zt[i] - xb)).clamp(-clip, clip);
    });
}

/// ``X~' W X~`` (row-major ``p x p``, mirrored) and ``X~' W z~``, plus the
/// weighted sums of squares of the raw columns, over fixed row chunks.
fn cross_products(
    xt: &[f64],
    zt: &[f64],
    x: &[f64],
    w: &[f64],
    p: usize,
    xtwx: &mut [f64],
    xtwz: &mut [f64],
    ss_raw: &mut [f64],
) {
    let n = zt.len();
    let m = p * (p + 1) / 2 + 2 * p;
    let parts = par::sum_vec(n, m, |r| {
        let mut acc = vec![0.0_f64; m];
        let mut k = 0;
        for j1 in 0..p {
            let c1 = &xt[j1 * n..(j1 + 1) * n];
            for j2 in 0..=j1 {
                let c2 = &xt[j2 * n..(j2 + 1) * n];
                let mut a = 0.0_f64;
                for i in r.clone() {
                    a += c1[i] * w[i] * c2[i];
                }
                acc[k] = a;
                k += 1;
            }
        }
        for j in 0..p {
            let c = &xt[j * n..(j + 1) * n];
            let raw = &x[j * n..(j + 1) * n];
            let (mut a, mut b) = (0.0_f64, 0.0_f64);
            for i in r.clone() {
                a += c[i] * w[i] * zt[i];
                b += w[i] * raw[i] * raw[i];
            }
            acc[k] = a;
            acc[k + p] = b;
            k += 1;
        }
        acc
    });
    let mut k = 0;
    for j1 in 0..p {
        for j2 in 0..=j1 {
            xtwx[j1 * p + j2] = parts[k];
            xtwx[j2 * p + j1] = parts[k];
            k += 1;
        }
    }
    xtwz.copy_from_slice(&parts[k..k + p]);
    ss_raw.copy_from_slice(&parts[k + p..k + 2 * p]);
}

/// Poisson IRLS (the loop of crate API 2).
#[cfg_attr(not(test), allow(dead_code))]
pub fn fepois_loop(
    y: &[f64],
    x: &[f64],
    obs_weights: &[f64],
    fe_codes: &[&[i64]],
    g_per_fe: &[usize],
    cfg: &FePoisIRLSConfig,
) -> FePoisIRLSResult {
    glm_loop(Family::Poisson, y, x, obs_weights, fe_codes, g_per_fe, cfg)
}

/// IRLS (Fisher scoring) for a GLM with absorbed fixed effects.
pub fn glm_loop(
    fam: Family,
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

    let mut eta = vec![0.0_f64; n];
    par::for_each_mut(&mut eta, |i, e| *e = fam.start(y[i], obs_weights[i]));
    let mut mu = vec![0.0_f64; n];
    let mut dmu = vec![0.0_f64; n];
    let mut w = vec![0.0_f64; n];
    let mut z = vec![0.0_f64; n];

    // Mean, its derivative, working weight and working response at ``eta``.
    let working = |eta: &[f64], mu: &mut [f64], dmu: &mut [f64], w: &mut [f64], z: &mut [f64]| {
        let md: Vec<(f64, f64)> = if n < par::PAR_MIN_ROWS {
            eta.iter().map(|&e| fam.mean(e)).collect()
        } else {
            eta.par_iter()
                .with_min_len(par::CHUNK_ROWS)
                .map(|&e| fam.mean(e))
                .collect()
        };
        par::for_each_mut(mu, |i, v| *v = md[i].0);
        par::for_each_mut(dmu, |i, v| *v = md[i].1);
        par::for_each_mut(w, |i, v| {
            let (m, d) = md[i];
            *v = obs_weights[i] * d * d / fam.variance(m);
        });
        par::for_each_mut(z, |i, v| {
            let (m, d) = md[i];
            *v = eta[i] + (y[i] - m) / if d > 0.0 { d } else { 1e-300 };
        });
    };
    working(&eta, &mut mu, &mut dmu, &mut w, &mut z);

    let mut problem = Problem::new(fe_codes.to_vec(), g_per_fe.to_vec(), Some(w.clone()));
    let mut workspaces: Vec<Workspace> = (0..=p).map(|_| Workspace::new(&problem, n)).collect();
    let mut buf = vec![0.0_f64; n * (p + 1)];

    let mut dev = f64::INFINITY;
    let mut iters_used: u32 = 0;
    let mut converged = false;
    let mut n_halvings: u32 = 0;
    let mut max_inner_dx: f64 = 0.0;
    let mut fe_converged = true;
    let mut singular = false;

    let mut xt_w_x = vec![0.0_f64; p * p];
    let mut beta = vec![0.0_f64; p];
    let mut beta_try = vec![0.0_f64; p];
    let mut ss_raw = vec![0.0_f64; p];
    let mut eta_new = vec![0.0_f64; n];
    let mut mu_new = vec![0.0_f64; n];

    for it in 0..cfg.maxiter {
        // 1. Working response and weight (at the current eta).
        if it > 0 {
            working(&eta, &mut mu, &mut dmu, &mut w, &mut z);
            problem.reweight(&w);
        }
        // 2. Residualise z and X on the FEs (warm-started per column).
        {
            let zr: &[f64] = &z;
            let (head, tail) = buf.split_at_mut(n);
            par::for_each_mut(head, |i, v| *v = zr[i]);
            par::for_each_mut(tail, |i, v| *v = x[i]);
        }
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

        // 3. Weighted normal equations on the residualised columns. A
        // regressor the FEs absorb leaves only rounding noise: compare its
        // residualised weighted norm with the raw one.
        let (zt, xt) = buf.split_at(n);
        cross_products(xt, zt, x, &w, p, &mut xt_w_x, &mut beta_try, &mut ss_raw);
        let absorbed_tol = (10.0 * cfg.fe_tol).powi(2).max(1e-18);
        let absorbed = (0..p).any(|j| !(xt_w_x[j * p + j] > absorbed_tol * ss_raw[j]));
        if absorbed || crate::cholesky::cholesky_factor(&mut xt_w_x, p).is_err() {
            singular = true;
            break;
        }
        crate::cholesky::cholesky_solve(&xt_w_x, p, &mut beta_try);
        beta.copy_from_slice(&beta_try);

        // 4. eta_new = z - (z_tilde - X_tilde beta), clipped.
        update_eta(&z, zt, xt, &beta, cfg.eta_clip, &mut eta_new);
        {
            let en: &[f64] = &eta_new;
            par::for_each_mut(&mut mu_new, |i, m| *m = fam.mean(en[i]).0);
        }
        let mut new_dev = deviance(fam, y, &mu_new, obs_weights);

        // 5. Step-halving on deviance non-decrease.
        let mut halvings = 0u32;
        while new_dev > dev && halvings < cfg.max_halvings && dev.is_finite() {
            {
                let er: &[f64] = &eta;
                par::for_each_mut(&mut eta_new, |i, e| {
                    *e = (0.5 * (*e + er[i])).clamp(-cfg.eta_clip, cfg.eta_clip)
                });
            }
            let en: &[f64] = &eta_new;
            par::for_each_mut(&mut mu_new, |i, m| *m = fam.mean(en[i]).0);
            new_dev = deviance(fam, y, &mu_new, obs_weights);
            halvings += 1;
        }
        n_halvings += halvings;

        // 6. Convergence check on the relative deviance change.
        let rel = fam.rel_change(new_dev, dev);
        std::mem::swap(&mut eta, &mut eta_new);
        std::mem::swap(&mut mu, &mut mu_new);
        dev = new_dev;
        iters_used = it + 1;
        if rel < cfg.tol {
            converged = true;
            break;
        }
    }

    let log_lik = par::sum(n, |r| {
        let mut acc = 0.0_f64;
        for i in r {
            acc += obs_weights[i] * fam.loglik(y[i], mu[i]);
        }
        acc
    });
    let x_tilde = buf.split_off(n);

    FePoisIRLSResult {
        beta,
        x_tilde,
        w,
        eta,
        mu,
        deviance: dev,
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
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
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
            assert!(
                (res.beta[j] - beta_true[j]).abs() < 1e-7,
                "beta {:?}",
                res.beta
            );
        }
        assert_eq!(res.x_tilde.len(), n * p);
    }

    /// Logit and probit on a panel large enough for the row-parallel
    /// paths: at the solution the score equations hold (the residualised
    /// regressors are orthogonal to the working residual), and the answer
    /// is bit-identical with one thread and with eight.
    #[test]
    fn binomial_scores_vanish_and_threads_do_not_matter() {
        let n = 80_000usize;
        let (g0, g1) = (400usize, 15usize);
        let mut s: u64 = 11;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let codes0: Vec<i64> = (0..n).map(|_| (rng() * g0 as f64) as i64).collect();
        let codes1: Vec<i64> = (0..n).map(|_| (rng() * g1 as f64) as i64).collect();
        let x: Vec<f64> = (0..2 * n).map(|_| rng() - 0.5).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let e = 0.8 * x[i] - 0.5 * x[n + i] + 0.002 * codes0[i] as f64;
                f64::from(rng() < 1.0 / (1.0 + (-e).exp()))
            })
            .collect();
        let ones = vec![1.0_f64; n];
        let mut cfg = FePoisIRLSConfig::default();
        cfg.tol = 1e-12;
        cfg.fe_tol = 1e-12;
        for fam in [Family::Logit, Family::Probit] {
            cfg.eta_clip = if fam == Family::Logit { 30.0 } else { 8.0 };
            let run = |threads: usize| {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                pool.install(|| glm_loop(fam, &y, &x, &ones, &[&codes0, &codes1], &[g0, g1], &cfg))
            };
            let one = run(1);
            let eight = run(8);
            assert!(one.converged && !one.singular, "{fam:?}: {one:?}");
            assert_eq!(
                one.beta.iter().map(|b| b.to_bits()).collect::<Vec<_>>(),
                eight.beta.iter().map(|b| b.to_bits()).collect::<Vec<_>>()
            );
            // Score: sum_i x~_ij * w_i (y_i - mu_i) dmu_i / V(mu_i) = 0.
            for j in 0..2 {
                let mut score = 0.0_f64;
                let mut scale = 0.0_f64;
                for i in 0..n {
                    let (m, d) = fam.mean(one.eta[i]);
                    let r = (y[i] - m) * d / fam.variance(m);
                    score += one.x_tilde[j * n + i] * r;
                    scale += (one.x_tilde[j * n + i] * r).abs();
                }
                assert!(
                    score.abs() < 1e-7 * scale,
                    "{fam:?} score {j}: {score} / {scale}"
                );
            }
            if fam == Family::Logit {
                // The data are logit with these slopes: the MLE is near them.
                assert!((one.beta[0] - 0.8).abs() < 0.15 && (one.beta[1] + 0.5).abs() < 0.15);
            }
        }
    }

    #[test]
    fn regressor_absorbed_by_fe_is_reported_singular() {
        let n = 60usize;
        let codes0: Vec<i64> = (0..n as i64).map(|i| i % 6).collect();
        // x is constant within FE 0: residualised it is zero.
        let x: Vec<f64> = codes0.iter().map(|&c| c as f64 * 0.5).collect();
        let y: Vec<f64> = (0..n).map(|i| ((i % 5) + 1) as f64).collect();
        let ones = vec![1.0_f64; n];
        let res = fepois_loop(
            &y,
            &x,
            &ones,
            &[&codes0],
            &[6],
            &FePoisIRLSConfig::default(),
        );
        assert!(res.singular);
        assert!(!res.converged);
    }
}
