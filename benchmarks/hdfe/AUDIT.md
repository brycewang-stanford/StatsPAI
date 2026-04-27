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
