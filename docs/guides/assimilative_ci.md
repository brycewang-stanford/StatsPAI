# Assimilative Causal Inference

> *Nature Communications* 2026: bridging Bayesian data assimilation
> (weather forecasting, oceanography) with streaming causal inference.

## 1. What problem does this solve?

Most causal-inference pipelines assume the data arrive in one shot.
But in A/B testing, pharmacovigilance, and policy evaluation you
typically get **a batch at a time**, and the treatment effect itself
may drift with seasons, user segments, or policy regimes.

Assimilative causal inference treats the causal effect as a
time-varying latent state and updates its posterior every time a new
batch arrives — exactly the same mathematics that a weather model
uses to fuse satellite observations into a global atmospheric state.

## 2. State-space formulation

$$
\begin{aligned}
\theta_t &= \theta_{t-1} + w_t,
   \qquad w_t \sim \mathcal{N}(0, Q_t) \quad \text{(dynamics)}\\[2pt]
\widehat{\theta}_t &= \theta_t + e_t,
   \qquad e_t \sim \mathcal{N}(0, \sigma_t^2) \quad \text{(observation)}
\end{aligned}
$$

Under Gaussianity the Kalman filter gives closed-form posteriors
`θ_t | y_{1:t} ~ N(m_t, P_t)`.  For non-Gaussian / nonlinear settings
StatsPAI ships a bootstrap-SIR particle filter with systematic
resampling.

## 3. Two-level API

### Low-level: fuse a pre-computed stream

If you have already produced a stream of `(θ̂_t, σ_t)` from, say,
running `sp.did` or `sp.dml` on each batch:

```python
import statspai as sp

res = sp.causal_kalman(
    estimates=[θ̂_1, θ̂_2, ...],
    standard_errors=[σ_1, σ_2, ...],
    prior_mean=0.0,
    prior_var=1.0,
    process_var=0.0,    # 0 = static effect, >0 = random-walk drift
    alpha=0.05,
)
print(res.summary())
```

For heavy-tailed observation noise or non-Gaussian priors swap in the
particle filter:

```python
res = sp.assimilation.particle_filter(
    estimates, standard_errors,
    n_particles=3000,
    process_sd=0.05,     # random-walk SD
    random_state=0,
)
```

### High-level: fuse directly from raw batches

Pass the batches and a `(df → (θ̂, σ))` estimator callback and the
pipeline handles both per-batch estimation and assimilation:

```python
def est(df):
    r = sp.regress('y ~ d + x', data=df)
    return float(r.params['d']), float(r.std_errors['d'])

res = sp.assimilative_causal(
    batches=[batch_jan, batch_feb, ...],
    estimator=est,
    prior_mean=0.0,
    prior_var=1.0,
    process_var=0.01,       # allow small month-to-month drift
    backend='kalman',       # or 'particle' for non-Gaussian
)
```

## 4. Outputs

Every backend returns the same `AssimilationResult` dataclass:

| Field              | Shape  | Meaning                             |
|--------------------|--------|-------------------------------------|
| `posterior_mean`   | (T,)   | Running posterior mean `m_t`        |
| `posterior_sd`     | (T,)   | Running posterior SD `sqrt(P_t)`    |
| `posterior_ci`     | (T, 2) | Per-step CI at level `alpha`        |
| `innovations`      | (T,)   | Observation surprise `θ̂_t − m_{t|t-1}` |
| `ess`              | (T,)   | Effective sample size               |
| `final_mean`, `final_sd`, `final_ci` | scalars | End-of-stream summary |
| `trajectory()`     | method | Tidy DataFrame of the above         |

## 5. End-to-end example

```python
import numpy as np, pandas as pd, statspai as sp

# Generate 12 monthly A/B-test batches with a slowly drifting effect.
rng = np.random.default_rng(0)
def make_batch(n, tau, seed):
    r = np.random.default_rng(seed)
    d = r.integers(0, 2, n); x = r.normal(size=n)
    y = tau * d + 0.2 * x + r.normal(scale=0.3, size=n)
    return pd.DataFrame({'y': y, 'd': d, 'x': x})

tau_path = 0.5 + np.linspace(0, 0.1, 12)   # drifts from 0.5 to 0.6
batches = [make_batch(300, tau_path[t], seed=t) for t in range(12)]

def est(df):
    r = sp.regress('y ~ d + x', data=df)
    return float(r.params['d']), float(r.std_errors['d'])

res = sp.assimilative_causal(
    batches, est, prior_mean=0.0, prior_var=1.0,
    process_var=0.001,    # mild drift allowance
)
print(res.summary())
print(res.trajectory().tail())
```

## 6. Choosing between the backends

| Symptom                                   | Use                                   |
|-------------------------------------------|----------------------------------------|
| Per-batch estimates look roughly Gaussian | `backend='kalman'` (default, fast)     |
| Heavy tails / outliers per batch          | `backend='particle'` with Student-t obs model |
| Non-Gaussian prior (log-normal, bounded)  | `backend='particle'` + `prior_sampler=...` |
| You need CI bands in real time            | Either — both return `posterior_ci` per step |

## 7. Gotchas

- **Garbage-in, garbage-out.**  The filter trusts that your per-batch
  estimator is well-calibrated (nominal CIs cover at their stated
  rate).  Run `sp.smart.assumption_audit` on the estimator first.
- **`process_var = 0` means a static effect.**  If the real effect
  drifts, you'll get overconfident CIs because the filter has no room
  for state innovation.  Start with `process_var ≈ σ_t² / 10` and
  tune upwards if the innovations look persistently one-sided.
- **Particle-filter degeneracy.**  If you see
  `ESS / N < 0.2` persistently, increase `n_particles` or loosen the
  observation model (heavier tails).

## 8. References

- *Assimilative Causal Inference* — Nature Communications (2026).
- Gordon, Salmond & Smith (1993).  "Novel approach to nonlinear/non-
  Gaussian Bayesian state estimation."  *IEE Proc. F.*
- Douc & Cappé (2005).  "Comparison of resampling schemes for
  particle filtering."  *ISPA*.
