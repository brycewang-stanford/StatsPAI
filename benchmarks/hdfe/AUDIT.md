# Post-Ship Audit — Findings & Fixes

After the initial Phase 0–8 ship, I re-read every module end-to-end and
found 13 issues. This file is the public record of what was found, what
was fixed, and what was left as a tracked follow-up. Every fix in here
landed on top of the original ship without breaking any existing test;
the cumulative regression count went from 208 → 208 (still 2 skips,
both intentional optional-dep paths).

## Severity legend

- **P0** — correctness / clarity / dead code that misleads users.
- **P1** — performance hot path; user-facing wall-clock impact.
- **P2** — code quality, type hints, lint.
- **P3** — missing functionality, larger design discussion required.

## Findings

### P0 — Correctness / clarity

| # | File:line | Issue | Status |
|---|-----------|-------|:------:|
| 1 | `fast/fepois.py:389-390` (pre-fix) | Dead `if False else` branch — leftover from refactoring. The `keep_sep` it computed was overwritten in one branch and only used in the other. | **fixed** |
| 2 | `fast/fepois.py:386-402` (pre-fix) | Two-branch separation handling could be one unified loop running on the current `keep` mask. | **fixed** |
| 3 | `fast/fepois.py` `n_dropped_singletons` | Lumped singletons + separation drops together; `n_dropped_separation` was hard-coded to 0. The summary line lied. | **fixed** — now properly split (medium dataset reports 45 singletons + 485 separation = 530) |

### P1 — Performance

| # | File | Issue | Status |
|---|------|-------|:------:|
| 4 | `fast/inference.py boottest` | `np.linalg.solve(XtWX, ...)` re-factorised the same matrix B times instead of using the precomputed `bread`. | **fixed** |
| 5 | `fast/inference.py boottest` | Bootstrap inner loop recomputed the (k×k) sandwich + per-cluster scores on every iter; for the single-coef null we only need `bread_row @ meat @ bread_row`. | **fixed** |
| 6 | `fast/demean.py rust path` | Two unnecessary copies: `np.asfortranarray(X)` followed by `np.ascontiguousarray(X_f)`. Skip when X is already F-contig; return F-contig view. | **fixed** (folded into `_demean_core`) |
| 7 | `fast/within.py transform` | Re-ran the full `_fast_demean` pipeline (factorise, singleton-detect, allocate counts) on every transform — the entire cache benefit was lost. | **fixed** — now calls `_demean_core` with the cached densified codes directly |
| 8 | `fast/fepois.py _weighted_ap_demean` | No Aitken acceleration in the IRLS-internal weighted demean (Phase 1 unweighted path had it). | **fixed** |

### P2 — Code quality

| # | File | Issue | Status |
|---|------|-------|:------:|
| 9 | `fast/inference.py` | `_rademacher_weights` / `_webb6_weights` had no docstring or named constant for the ±1 array. | **fixed** |
| 10 | `fast/dsl.py i()` | `prefix: str = None` should be `Optional[str]` for type-checkers. | **fixed** |
| 11 | `fast/etable.py` | Significance stars use Normal-z thresholds, not t-distribution. Cosmetic for large-N. | **left** — flagged in PHASE8_VERIFY.md |

### P3 — Missing features

| # | File | Issue | Status |
|---|------|-------|:------:|
| 12 | `fast/fepois.py` | No `weights=` (observation weights). pyfixest/fixest support this for survey designs. | **fixed** (Round B) — coef parity with `pyfixest.fepois(..., weights=)` to <1e-8 |
| 13 | `fast/polars_io.py` | `.to_numpy()` not using `allow_copy=False` where Polars 1.x supports it. | **fixed** (Round A) — `_polars_to_numpy_zero_copy` tries zero-copy first, falls back |

Two more items originally listed as P3 (covered in phase-level
deferrals) that landed in this round:

- ⏸→✅ **CR2 (Bell-McCaffrey) cluster SE** — Round D. `crve(type="cr2")`
  with leverage-adjusted scores via eigendecomposition of `(I - H_gg)`.
  4 new tests (positive-definite, large-G ≈ CR1, few-G inflates, type
  validation).
- ⏸→✅ **Multi-coefficient joint Wald wild cluster bootstrap** —
  Round E. New `boottest_wald(X, y, cluster, R, r=None, ...)` returning
  a `BootWaldResult`. 6 new tests including the consistency check
  `Wald = t²` for the single-coef case.
- ⏸→✅ **`df_residual` on `FePoisResult` + Student-t stars in
  `etable`** — Round C. The previous Normal-z thresholds drift on
  small-N fits; we now use t critical values when the fit exposes
  `df_residual` and fall back to z otherwise. 2 new tests.

Items that **explicitly remain unaddressed** because they need
multi-week effort or hardware we don't have:

- **Anderson(m) acceleration** in the demean inner loop — Aitken /
  Irons-Tuck stays as the default; Anderson is a v1.7.1 tracked add-on.
- **IM (Imbens-Kolesar) Satterthwaite DOF correction** — needs the
  cluster-leverage trace algebra layered on top of CR2; we ship CR2
  but not IM.
- **Cluster-robust SE inside `sp.fast.fepois`** — Phase 4 has the
  primitives (`crve`, `boottest`, `boottest_wald`); wiring them into
  the `vcov="cr1"|"cr2"|"cr3"` branch of `fepois` is straightforward
  but didn't fit this audit round and lands in v1.8.1.
- **Native Rust IRLS for `fepois`** — multi-week port; the Phase 1
  Rust demean kernel is shipped and ready to be called from a Rust
  IRLS, but the IRLS itself is still Python.
- **GPU bench (CUDA cuSPARSE / JAX-on-GPU)** — no GPU on the dev
  machine. The JAX backend is structurally GPU-ready and tested on
  CPU; lighting up real hardware is a follow-up PR with a self-hosted
  runner.

## What changed in code

```
src/statspai/fast/
├── demean.py     ← extracted ``_demean_core`` for shared kernel dispatch;
│                   skip redundant F-contig roundtrip; let WithinTransformer
│                   bypass the prep pipeline.
├── fepois.py     ← cleaner separation handling; Aitken in weighted demean;
│                   guard against all-rows-dropped after pre-passes.
├── inference.py  ← boottest reuses ``bread``; computes only the variance
│                   element we need; named constants for wild-weight draws.
├── within.py     ← transform() goes straight to _demean_core.
└── dsl.py        ← Optional[str] type for ``prefix``.
```

No public API change. No test broken.

## Empirical confirmation

```
$ pytest tests/test_fast_*.py tests/test_panel.py tests/test_fixest.py \
         tests/test_did.py tests/test_iv.py tests/test_synth.py \
         tests/test_dml.py tests/test_hdfe_native.py -q --no-cov
208 passed, 2 skipped in 18.80s
```

Numerical parity (medium = 1M rows, fe1=100k, fe2=1k):

| metric                            | pre-fix     | post-fix   |
|-----------------------------------|------------:|-----------:|
| coef diff vs pyfixest (β_x1)      | 1.6e-15     | **5.55e-16** |
| SE diff vs pyfixest (β_x1)        | 4.34e-11    | **4.34e-11** |
| `singletons` reported             | 530 (lump)  | **45**     |
| `separation` reported             | 0 (lie)     | **485**    |

Wall-clock gains (n=1500, G=30, k=2, B=9999, Rademacher):

| metric                            | pre-fix     | post-fix   |
|-----------------------------------|------------:|-----------:|
| `boottest` end-to-end             | ~3 s¹       | **~0.48 s** |

¹ Estimated from the original implementation's per-iter cost; not
measured directly because the audit fix is in the same patch series.

## What this audit explicitly did NOT do

- **Native Rust IRLS** — still the biggest open item; closing the
  ~3× wall-clock gap to fixest needs it.
- **CR2 / IM** small-cluster SE — no leverage matrix work yet.
- **Multi-coef joint wild bootstrap** — single-coef null is the most
  common case; joint Wald comes later.
- **CUDA / GPU benchmarks** — no hardware available on the dev box.

These remain in [`SUMMARY.md`](SUMMARY.md)'s "what deliberately did NOT
ship" list with the same priorities.

---

## Phase A round 1 — wall-clock gate FAILED (2026-04-27)

The Phase A spec (`docs/superpowers/specs/2026-04-27-native-rust-irls-fepois-design.md` §5.7) set a **merge blocker** of medium wall ≤ 1.5 s. The actual measurement was **2.449 s** (median of 3 reps). Per spec we MUST NOT silently widen the gate; instead we record findings here and surface to the user for a brainstorm-level decision.

### Numbers

| stage                                   | medium wall | iters | vs fixest 0.64 s |
| --------------------------------------- | ----------: | ----: | ---------------: |
| Phase 0 baseline (Python `np.bincount`) |      2.61 s |     6 |            4.08× |
| Phase A (Rust `demean_2d_weighted`)     | **2.45 s**  |     6 |       **3.83×**  |
| Phase A gate target                     |   ≤ 1.50 s  |     — |          ≤ 2.34× |
| R `fixest::fepois`                      |      0.64 s |     5 |            1.00× |

Phase A delivers a **6.1 % wall reduction** (2.61 → 2.45 s), not the projected ~50 % (which would have landed at ~1.4 s).

### What broke — the assumption

The plan assumed the IRLS-internal weighted demean was ~80 % of wall and that a Rust port would deliver ~3-5× over Python. With those numbers, end-to-end wall was projected at ~1.4 s.

What the profile actually shows (cProfile on a single timed `fepois` call, 2.789 s total):

```
ncalls   tottime  cumtime  filename:lineno(function)
     1   0.216    2.789    fepois.py:404(fepois)
    12   0.088    2.293    fepois.py:300(_weighted_ap_demean)        ← dispatcher
    12   2.190    2.190    {built-in: statspai_hdfe.demean_2d_weighted}  ← Rust kernel
     6   0.113    0.115    fepois.py:751(_poisson_deviance)
```

Two findings:

1. **The Rust kernel is 78.5 % of wall (2.19 s of 2.79 s).** The dispatcher overhead (`np.asfortranarray`, `np.bincount` for `wsum`, list packing) is only 0.088 s — i.e., the projected "Python overhead" we'd save by going to Rust was already small. The dominant cost is the Rust AP loop itself, not the Python glue.

2. **Per-Rust-call wall is ~155-318 ms** with AP iters in [14, 16]. For 1M rows × K=2 FEs × p ∈ {1,2}, that is **~5-10× slower than the fixest C++ kernel** (which does similar work in ~50 ms on similar hardware). The bottleneck is the random-scatter inner loop (`scratch[codes[i]] += weights[i] * x[i]`) hitting cache-miss territory on the G1=100k FE bucket array (800 KB scratch, exceeds L1, lives in L2 with mostly-random access pattern).

The honest reframe of the gap:

```
fixest C++   ----  hand-tuned scatter-gather + likely SIMD + sort-by-FE-code
np.bincount  ----  hand-tuned C scatter-gather (Python NumPy)
Phase A Rust ----  straight Rust scatter-gather, Rayon column-parallel
```

NumPy's `np.bincount` with float weights is already a hand-tuned C loop that does roughly what our Rust kernel does. Phase A's Rust port wins by ~2.35× over the NumPy path on the kernel itself (verified via the `_HAS_RUST_HDFE = False` diagnostic), but that is FAR less than the projected 3-5×, because we are not beating an already-optimized C loop by a large margin — we are matching it.

### What value Phase A still ships

Even with the gate failing, the Phase A work delivers:

- **Reusable crate-internal Rust primitive** (`pub fn weighted_demean_matrix_fortran_inplace`) that Phase B's native Rust IRLS will call **without** going through PyO3 — eliminating the 12 FFI round-trips per `fepois` call and the GIL release/reacquire per call.
- **PyO3 surface** (`statspai_hdfe.demean_2d_weighted`) for any future Python caller that wants weighted within-transform.
- **Crate v0.3.0** with tested binding, abi3-py39 wheel building reproducibly via `maturin build --interpreter`.
- **Python dispatcher with NumPy fallback** (`_HAS_RUST_HDFE` flag, graceful degradation on no-Rust wheels).
- **3 end-to-end parity tests** vs `pyfixest.fepois` at coef atol ≤ 1e-13 (IID + weighted + fallback) — confirmed Phase A introduces no numerical regression.
- **CR1 cluster-robust SE recovery** (commit `39c94d0`) — the user's HTZ track WIP that was destroyed by an earlier `git reset --hard` was recovered from a Claude Code auto-checkpoint dangling commit and restored. With this restore, the cluster-SE tests already committed in `8432c0d` pass, and `vcov="cr1"` is wired in.
- **Honest measurement methodology**: `run_fepois_phase_a.py` documents the exact procedure (warmup + 3 reps, JSON output, hard gate); reproducible.

### What Phase B has to do to actually close the gap

The 78.5%-Rust-kernel finding is the dominant input to Phase B's design:

- **Eliminate the 12 FFI round-trips per `fepois`** — the Rust IRLS keeps `mu`, `eta`, `z`, `w`, `X_tilde` in Rust, sweeps once per outer iter without crossing the Python boundary. Saves ~88 ms of dispatcher overhead, but more importantly enables the next two:
- **Reuse scratch + Aitken history across IRLS iters** — currently each Rust call allocates fresh `Vec<Vec<f64>>` scratch and `Vec<Vec<f64>>` Aitken history. Phase B's persistent `IRLSWorkspace` allocates these once, saves ~12 × 24 MB allocation pressure.
- **Sort observations by primary FE code once before IRLS** — converts the random-scatter inner loop into a sequential one, removing the L2 cache-miss bottleneck. Expected ~3-5× speedup on the kernel itself. This is the algorithmic change the projected ~50 % reduction was actually depending on.

With Phase B, the path to ≤ 0.95 s (≤ 1.5× fixest) is plausible. The path to ≤ 1.5 s via Phase A alone is not, given that the kernel already beats NumPy's `np.bincount` by 2.35×.

### Decision surface

The honest options are (per spec §5.7):

- **A. Do not merge Phase A; preserve work-in-place; pivot to Phase B.** Phase A's commits stay on `main` (already pushed) but `1.8.0` is not released until Phase B lands. The user's CR1 cluster-SE recovery (`39c94d0`) is independent and should land regardless.
- **B. Merge Phase A with the failing gate; document the gap honestly in CHANGELOG.** Ship the correctness validation, the reusable Rust primitive, and the dispatcher; absorb the 6 % wall improvement; flag Phase B as the gate-closer in the CHANGELOG.
- **C. Invest in Rust-kernel optimization (sort-by-FE-code, SIMD intrinsics) before Phase B.** Multi-day effort with diminishing returns; might or might not close to 1.5 s. Defers Phase B.

The spec explicitly forbids silently widening the gate. The user's call.
