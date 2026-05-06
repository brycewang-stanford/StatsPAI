# GPU acceleration in StatsPAI

> **TL;DR.** As of v1.14, three workloads in StatsPAI route to CUDA / TPU
> when an accelerator is available: (1) the neural causal estimators
> (`sp.deepiv`, `sp.tarnet`, `sp.cfrnet`, `sp.dragonnet`, `sp.cevae`)
> via PyTorch; (2) end-to-end OLS with HDFE via `sp.fast.feols_jax`;
> and (3) **vmap'd bootstrap** via `sp.fast.feols_jax_bootstrap` —
> the largest GPU win per line of user code.
>
> Everything else in StatsPAI is CPU-only and intentionally so: most
> econometric estimators (DiD, IV, RD, synthetic control, fixest-style
> HDFE OLS, GMM) are dominated by combinatorial / memory-bound work
> where GPUs offer no speedup over a tuned Rust + Numba kernel.

---

## What is GPU-accelerated today?

| Workload | Function | Backend | Activation |
| --- | --- | --- | --- |
| Neural causal: representation networks (TARNet / CFRNet / DragonNet / CEVAE) | `sp.tarnet` / `sp.cfrnet` / `sp.dragonnet` / `sp.cevae` | PyTorch | `STATSPAI_TORCH_DEVICE={cuda,mps,auto}` env var |
| Neural IV: Deep IV (Hartford et al. 2017) | `sp.deepiv` | PyTorch | same env var |
| HDFE demean (alternating projection) | `sp.fast.demean(backend="jax")` | JAX / XLA | install `jax[cuda]` |
| OLS / WLS with HDFE | `sp.fast.feols_jax` | JAX / XLA | install `jax[cuda]` |
| **Bootstrap (pairs / cluster / wild / wild_cluster)** | `sp.fast.feols_jax_bootstrap` | JAX / XLA `vmap` | install `jax[cuda]` |

The CPU paths (`sp.fast.demean`, `sp.fast.feols`, `sp.fast.fepois`,
`sp.fast.boottest`, `sp.iv`, `sp.did`, `sp.rd`, `sp.synth`, …) all
remain the production defaults and ship without any accelerator
dependency.

---

## Why bootstrap is the headline GPU win

Single-shot OLS / WLS is **dominated by host↔device transfer overhead**
on small-to-medium datasets — the actual QR factorisation is too cheap
for GPU speedup to recover the wire cost.

Bootstrap inverts this: the *same* JIT-compiled WLS program is lifted
to a `jax.vmap` batch primitive and runs B times in lock-step on the
device. On CUDA / TPU this approaches `B / utilisation × per-iteration
time`; on CPU JAX it's still ~equal to a numpy sequential bootstrap
(JIT overhead amortises around B ≈ 100). The speedup curve crosses
favourably very quickly.

**Pairs bootstrap**: each draw resamples *rows* with replacement;
multinomial counts become per-row WLS weights. Asymptotic target:
HC1 standard errors.

**Cluster bootstrap** (Cameron, Gelbach & Miller 2008 §III.A): each
draw resamples *clusters* with replacement; observations in a cluster
sampled k times get weight k. Asymptotic target: CR1 standard errors.

**Wild bootstrap**: each draw assigns independent Rademacher signs
``η_i ∈ {-1, +1}`` per row and uses the *score formulation*
``β* = β̂ + (X'WX)⁻¹ X'W (η ⊙ û)``, mathematically identical to
refitting on ``y* = X β̂ + η ⊙ û`` but with one mat-vec per
iteration instead of a full QR.

**Wild cluster bootstrap** (Cameron, Gelbach & Miller 2008 §III.B):
same score formulation as wild, but the Rademacher signs are drawn
*per cluster*. The standard tool for few-cluster inference (G < 30,
especially G < 10) where cluster bootstrap can over-reject.

```python
import statspai as sp

boot = sp.fast.feols_jax_bootstrap(
    "log_wage ~ schooling + experience | firm + year",
    data=df,
    n_boot=2_000,
    bootstrap="cluster",  # or "pairs"
    cluster="firm",
    ci_alpha=0.05,
)
print(boot.summary())
print(boot.se_boot)
print(boot.boot_betas)        # full B × p draws for custom CI methods
```

---

## Quickstart on Google Colab

The fastest way to verify GPU acceleration without buying hardware is
[Google Colab](https://colab.research.google.com/) Pro (≈ USD 10/month
for T4 / V100, USD 50/month for A100). The free tier is also enough
for proof-of-concept benchmarks.

```python
# In a Colab notebook with a GPU runtime selected
!pip install -q statspai jax[cuda12] jaxlib

import statspai as sp
print(sp.fast.jax_device_info())
# Expect: jax: <version>, default device: cuda

# Build a benchmark dataset
import numpy as np, pandas as pd
rng = np.random.default_rng(0)
n, n_firm = 1_000_000, 5_000
firm = rng.integers(0, n_firm, size=n)
fe = rng.normal(size=n_firm)[firm]
df = pd.DataFrame({
    "y": 0.5 * rng.normal(size=n) + fe,
    "x1": rng.normal(size=n),
    "x2": rng.normal(size=n),
    "firm": firm,
})

# Time CPU vs GPU
import time

t0 = time.perf_counter()
for _ in range(2_000):
    _ = sp.fast.feols("y ~ x1 + x2 | firm", df, vcov="hc1")
print(f"CPU sequential bootstrap (B=2000): {time.perf_counter() - t0:.1f}s")

t0 = time.perf_counter()
boot = sp.fast.feols_jax_bootstrap(
    "y ~ x1 + x2 | firm", df, n_boot=2_000, bootstrap="pairs",
    vmap_chunk_size=500,  # tune up for big-HBM GPUs
)
print(f"GPU vmap'd bootstrap (B=2000):     {time.perf_counter() - t0:.1f}s")
```

**Expected result on a T4 / V100 / A100:** the JAX path beats the
sequential CPU loop by 10–100x once `n` × `B` is large enough to
saturate the device.

---

## PyTorch GPU for neural causal

Setting the `STATSPAI_TORCH_DEVICE` environment variable (or having
`torch.cuda.is_available()` true with `auto`) routes neural backends
through CUDA / MPS:

```bash
export STATSPAI_TORCH_DEVICE=cuda    # or 'auto', 'mps', 'cpu'
```

```python
import statspai as sp
print(sp.fast.torch_device_info())
# Expect: torch <version> | cuda available (1 device(s)) | resolved=cuda

# All of these will train on GPU when the env var resolves to cuda/mps
sp.tarnet(df, y="y", treat="d", covariates=["x1", "x2"])
sp.cfrnet(df, y="y", treat="d", covariates=["x1", "x2"])
sp.dragonnet(df, y="y", treat="d", covariates=["x1", "x2"])
sp.cevae(df, y="y", treat="d", covariates=["x1", "x2"])
sp.deepiv(df, y="y", treat="d", instruments=["z"], covariates=["x1"])
```

The default is `cpu` to preserve bit-for-bit numerics on existing
pinned tests; `auto` on Apple Silicon falls through to MPS (Metal
Performance Shaders) when CUDA is unavailable.

---

## What is *not* GPU-accelerated, and why

| Family | Status | Why no GPU |
| --- | --- | --- |
| HDFE alternating-projection demean (CPU default) | Rust + Rayon | Bincount-style memory pattern is bandwidth-bound; tuned Rust matches GPU at typical FE counts. |
| Cluster-robust sandwich `crve()` | Rust + Rayon (Phase 2) | Same — the per-cluster reduce is bandwidth-bound. |
| Synthetic control (Abadie 2003 family, GSC, Augmented SC) | NumPy + scipy | Optimisation is small-K convex programs; no batch dimension to vmap over. |
| DiD estimators (Callaway-Sant'Anna, Sun-Abraham, BJS) | NumPy + pandas | Group-by-time accumulation is sequential; per-cohort fits are tiny. |
| Regression discontinuity | NumPy + scipy | Local-poly bandwidth choice is sequential. |
| GMM / IV / 2SLS | NumPy + scipy | Single-shot dense linalg; same constant-cost story as `feols_jax`. |
| Bayesian causal (PyMC) | NumPyro / JAX backend optional | Routing to GPU works *via PyMC*; we don't reimplement. |

Future GPU candidates (open issues welcome):

- **Permutation tests / placebo studies** — `vmap` over permutations is
  the obvious follow-up to bootstrap.
- **DML cross-fitting** — k-fold parallel nuisance fits.
- **Synthetic control matrix completion** — large-K SVD on GPU.
- **Causal forest training** — wire `xgboost` / `cuml` for tree fits.

---

## Reproducibility

JAX uses an explicit PRNG. `seed=` is honoured; same seed → bit-
identical bootstrap draws on the same device:

```python
b1 = sp.fast.feols_jax_bootstrap("y ~ x1 | firm", df, n_boot=500, seed=42)
b2 = sp.fast.feols_jax_bootstrap("y ~ x1 | firm", df, n_boot=500, seed=42)
assert (b1.boot_betas.values == b2.boot_betas.values).all()
```

Numerics across devices (CPU JAX vs CUDA vs TPU) can differ by ~1–2 ulp
because XLA reduction order is not guaranteed identical across
hardware. For coefficient-level reporting this is well below
econometric noise; for SE-level reporting see the convergence-rate
notes in the docstrings.

---

## Honesty check

The GPU story in v1.14 is **opt-in and selective**. We deliberately
don't claim "StatsPAI is GPU-accelerated" — most of the package is
CPU-only and that's the right design for the workloads we cover. The
GPU path matters for two specific cases:

1. **Neural causal training** — already a torch-native workload; the
   only thing we contributed was the unified device routing.
2. **Bootstrap-heavy inference** — where the speedup is real and
   measurable, especially at B ≥ 1000 on n ≥ 100k.

If your workflow is "fit one DiD / IV / RD on a 10k-row sample," a
laptop CPU is probably already as fast as a cloud GPU once you account
for package import + JIT compile time. **Buy a GPU if you're either
training neural causal models in volume, or doing high-B cluster
bootstrap on large panels.**
