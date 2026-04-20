# statspai_hdfe

PyO3 + Rayon HDFE group-demean kernel for [StatsPAI](https://github.com/brycewang-stanford/StatsPAI).

Phase 1 scaffold. See
[`docs/superpowers/specs/2026-04-20-v095-rust-hdfe-spike.md`](../../docs/superpowers/specs/2026-04-20-v095-rust-hdfe-spike.md)
on the parent repo for the full rollout plan.

## Build

```bash
pip install maturin
maturin develop --release     # dev-time install into current venv
# or
maturin build --release       # produce a wheel for redistribution
```

## Phases

- **Phase 1 (current):** single-threaded `group_demean` reference.
- **Phase 2:** wire into `statspai.panel.hdfe` with numba fallback.
- **Phase 3:** Rayon-parallelised `group_demean_block` for multi-column
  demeaning.
- **Phase 4:** `cibuildwheel` matrix (macOS arm64/x86_64,
  manylinux_2_17 x86_64/aarch64, musllinux_1_2 x86_64, Windows x86_64).

## Contract

Every release on this branch must:

1. Bit-identically reproduce the NumPy reference kernel on 16+ DGPs
   (`atol=1e-10`).
2. Pass `statspai.fast.hdfe_bench` correctness gate when built into
   the current venv (see `--atol 1e-10`).
3. Never require end users to install Rust at `pip install` time —
   missing wheels must fall back to the Python path silently.
