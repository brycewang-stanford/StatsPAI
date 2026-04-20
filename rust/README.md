# StatsPAI Rust kernels

This directory is **only populated on `feat/rust-hdfe`**. The
`main` branch stays Rust-free so `pip install statspai` never
requires a Rust toolchain.

## Crates

- `statspai_hdfe/` — PyO3 + Rayon HDFE group-demean kernel.

## Design docs

- `docs/superpowers/specs/2026-04-20-v095-rust-hdfe-spike.md` — phased rollout plan.
- `docs/superpowers/specs/2026-04-20-v096-bayes-iv-fuzzyrd-perlearner.md` — v0.9.6 scope (spawned this branch).

## Build (from repo root, on this branch only)

```bash
pip install maturin
maturin develop --release --manifest-path rust/statspai_hdfe/Cargo.toml
```

## Contract

Whatever this branch builds must:

1. Pass the `statspai.fast.hdfe_bench` correctness gate at `atol=1e-10`.
2. Be gracefully skipped when the compiled extension is missing —
   users on platforms without a Rust wheel keep working via the
   Numba fallback.
3. Never reach `main` until the cibuildwheel matrix (macOS arm64 +
   x86_64, manylinux_2_17 x86_64 + aarch64, musllinux_1_2 x86_64,
   Windows x86_64) is green.
