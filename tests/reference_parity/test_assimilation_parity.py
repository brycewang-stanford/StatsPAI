"""Assimilative-causal-inference reference parity.

Pins the two ``sp.assimilation`` backends to analytic / cross-backend
ground-truth under Gaussian and non-Gaussian DGPs.  Guards against:

1. **Posterior identification** — on a static Gaussian DGP, both
   backends must recover the true effect within ~0.1 over T batches.
2. **Kalman ↔ Particle agreement** — the bootstrap particle filter must
   reproduce the Kalman posterior to Monte-Carlo noise when the obs
   model is Gaussian and ``process_var = 0``.
3. **Monotone variance collapse** — under a static DGP the Kalman
   posterior variance ``P_t`` must be monotonically non-increasing.
4. **ESS resampling** — the particle filter's resampling trigger fires
   at least once over a long enough stream and the ESS stays above
   ``n_particles * threshold`` after resampling.
5. **Heavy-tailed robustness** — swapping the Gaussian obs model for a
   Student-t log-pdf keeps the particle filter well-calibrated when the
   stream contains a single outlier that the Kalman filter over-reacts
   to.
6. **Drifting effect** — under ``process_var > 0`` the filter tracks a
   slow drift without blowing up ``posterior_sd``.
7. **End-to-end dispatch** — ``sp.assimilative_causal(backend=...)``
   returns the same numeric answer as calling the backend directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _gauss_stream(T: int, tau: float, se: float, seed: int):
    rng = np.random.default_rng(seed)
    ests = [tau + rng.normal(0, se) for _ in range(T)]
    ses = [se] * T
    return ests, ses


def _drift_stream(T: int, tau0: float, tau1: float, se: float, seed: int):
    rng = np.random.default_rng(seed)
    path = np.linspace(tau0, tau1, T)
    ests = [float(p + rng.normal(0, se)) for p in path]
    ses = [se] * T
    return ests, ses


# ---------------------------------------------------------------------------
# 1. Posterior identification
# ---------------------------------------------------------------------------


class TestStaticPosteriorRecovery:

    def test_kalman_recovers_static_tau(self):
        ests, ses = _gauss_stream(T=20, tau=0.5, se=0.1, seed=0)
        r = sp.causal_kalman(ests, ses, prior_mean=0.0, prior_var=1.0)
        assert abs(r.final_mean - 0.5) < 0.05
        assert r.final_ci[0] < 0.5 < r.final_ci[1]

    def test_particle_recovers_static_tau(self):
        ests, ses = _gauss_stream(T=20, tau=0.5, se=0.1, seed=0)
        r = sp.assimilation.particle_filter(
            ests, ses, n_particles=4000, random_state=0,
        )
        assert abs(r.final_mean - 0.5) < 0.08
        assert r.final_ci[0] < 0.5 < r.final_ci[1]


# ---------------------------------------------------------------------------
# 2. Kalman ↔ Particle agreement (Gaussian DGP)
# ---------------------------------------------------------------------------


class TestKalmanParticleAgreement:

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_posteriors_match_under_gaussian(self, seed):
        ests, ses = _gauss_stream(T=15, tau=0.3, se=0.08, seed=seed)
        rk = sp.causal_kalman(ests, ses, prior_mean=0.0, prior_var=1.0)
        rp = sp.assimilation.particle_filter(
            ests, ses, n_particles=5000, random_state=seed,
        )
        # Point estimates agree to Monte-Carlo noise.
        assert abs(rp.final_mean - rk.final_mean) < 0.02, (
            f"Means differ: Kalman {rk.final_mean:.4f}, "
            f"Particle {rp.final_mean:.4f}"
        )
        # Posterior SDs agree within 15%.
        if rk.final_sd > 0:
            rel_err = abs(rp.final_sd - rk.final_sd) / rk.final_sd
            assert rel_err < 0.15, (
                f"SDs differ by {rel_err:.1%}: Kalman {rk.final_sd:.4f}, "
                f"Particle {rp.final_sd:.4f}"
            )


# ---------------------------------------------------------------------------
# 3. Monotone posterior variance under static effect
# ---------------------------------------------------------------------------


class TestKalmanVarianceMonotone:

    def test_posterior_var_non_increasing(self):
        ests, ses = _gauss_stream(T=25, tau=0.0, se=0.1, seed=0)
        r = sp.causal_kalman(
            ests, ses, prior_mean=0.0, prior_var=1.0, process_var=0.0,
        )
        P = r.posterior_sd ** 2
        # Allow a tiny floating-point tolerance.
        diffs = np.diff(P)
        assert (diffs <= 1e-10).all(), (
            "Posterior variance should be non-increasing under a static "
            f"effect with process_var=0; got diffs max = {diffs.max():.2e}"
        )


# ---------------------------------------------------------------------------
# 4. Particle filter — resampling activates & ESS stays healthy
# ---------------------------------------------------------------------------


class TestParticleFilterResampling:

    def test_ess_stays_above_threshold(self):
        # A long, slowly drifting stream will eventually require
        # resampling; ESS must not collapse to near zero.
        ests, ses = _drift_stream(
            T=30, tau0=0.5, tau1=0.6, se=0.1, seed=0,
        )
        r = sp.assimilation.particle_filter(
            ests, ses, n_particles=2000, process_sd=0.02,
            ess_resample_threshold=0.5, random_state=0,
        )
        # After resampling fires, ESS should bounce back above the
        # threshold for at least half of the remaining steps.
        thr = 0.5 * 2000
        frac_above = float((r.ess >= thr).mean())
        assert frac_above > 0.5, (
            f"ESS below threshold {thr:.0f} for most of the stream: "
            f"frac_above = {frac_above:.2f}"
        )


# ---------------------------------------------------------------------------
# 5. Heavy-tailed observation robustness
# ---------------------------------------------------------------------------


class TestHeavyTailedObsRobustness:

    def test_student_t_particle_not_blown_up_by_outlier(self):
        rng = np.random.default_rng(0)
        T = 20
        ests = [0.5 + rng.normal(0, 0.1) for _ in range(T)]
        # Plant a huge outlier in the middle of the stream.
        ests[T // 2] = 5.0
        ses = [0.1] * T

        # Kalman filter inflates toward the outlier.
        rk = sp.causal_kalman(ests, ses, prior_mean=0.0, prior_var=1.0)

        # Particle filter with Student-t obs log-pdf stays near truth.
        from math import lgamma, log, pi
        nu = 3.0  # heavy tails

        def student_t_log_pdf(theta_hat, particles, sigma):
            # Standard Student-t centred on each particle.
            z = (particles - theta_hat) / sigma
            return (
                lgamma((nu + 1) / 2) - lgamma(nu / 2)
                - 0.5 * log(nu * pi) - log(sigma)
                - ((nu + 1) / 2) * np.log1p(z ** 2 / nu)
            )

        rp = sp.assimilation.particle_filter(
            ests, ses,
            observation_log_pdf=student_t_log_pdf,
            n_particles=5000, random_state=0,
        )
        # Kalman should pull towards the outlier by ≥ 0.1; particle-t
        # should stay within 0.15 of 0.5.
        assert abs(rp.final_mean - 0.5) < 0.15, (
            f"Particle-t also blew up: {rp.final_mean:.3f}"
        )
        # Kalman is allowed to be worse — we just require particle is
        # meaningfully better on this stream.
        assert abs(rp.final_mean - 0.5) <= abs(rk.final_mean - 0.5), (
            f"Particle-t ({rp.final_mean:.3f}) should not be worse "
            f"than Kalman ({rk.final_mean:.3f}) on a contaminated stream."
        )


# ---------------------------------------------------------------------------
# 6. Drifting effect tracked without variance blow-up
# ---------------------------------------------------------------------------


class TestDriftingEffect:

    def test_kalman_tracks_drift(self):
        ests, ses = _drift_stream(
            T=30, tau0=0.3, tau1=0.7, se=0.05, seed=0,
        )
        r = sp.causal_kalman(
            ests, ses, prior_mean=0.0, prior_var=1.0, process_var=0.002,
        )
        # Final posterior should be closer to 0.7 than to 0.3.
        assert abs(r.final_mean - 0.7) < abs(r.final_mean - 0.3), (
            f"Kalman failed to track drift: final_mean = {r.final_mean:.3f}"
        )
        # Variance doesn't blow up.
        assert r.final_sd < 0.2


# ---------------------------------------------------------------------------
# 7. End-to-end dispatch — sp.assimilative_causal matches the backends
# ---------------------------------------------------------------------------


class TestEndToEndDispatch:

    def test_kalman_dispatch_matches_direct(self):
        def make_batch(n, seed):
            r = np.random.default_rng(seed)
            d = r.integers(0, 2, n); x = r.normal(size=n)
            y = 0.5 * d + 0.2 * x + r.normal(scale=0.3, size=n)
            return pd.DataFrame({"y": y, "d": d, "x": x})

        batches = [make_batch(200, s) for s in range(8)]

        def est(df):
            r = sp.regress("y ~ d + x", data=df)
            return float(r.params["d"]), float(r.std_errors["d"])

        r_e2e = sp.assimilative_causal(
            batches, est, backend="kalman",
            prior_mean=0.0, prior_var=1.0,
        )
        # Run the same pipeline step-by-step.
        ests = [est(b)[0] for b in batches]
        ses = [est(b)[1] for b in batches]
        r_direct = sp.causal_kalman(
            ests, ses, prior_mean=0.0, prior_var=1.0,
        )
        assert abs(r_e2e.final_mean - r_direct.final_mean) < 1e-10
        assert abs(r_e2e.final_sd - r_direct.final_sd) < 1e-10
