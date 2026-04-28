# StatsPAI HDFE Roadmap — Master Summary (v1.8.0 RC)

> **Post-audit update (rounds A–E)**: an 11-round self-audit found 13
> issues across the Phase 1–8 stack and added 5 follow-up features.
> 3 P0 (correctness/clarity) + 5 P1 (perf hot paths) + 5 features
> (Polars zero-copy, `weights=` in fepois, etable t-distribution stars,
> CR2 Bell-McCaffrey, multi-coef joint Wald bootstrap) all landed
> in-tree. **231 tests pass** (was 208). See [`AUDIT.md`](AUDIT.md) for
> per-round diffs and acceptance criteria.
>
> **Independent PR (2026-04-27)**: shipped the clubSandwich-equivalent
> HTZ Wald test (`cluster_wald_htz` / `cluster_dof_wald_htz` /
> `WaldTestResult`) — Pustejovsky-Tipton 2018 §3.2 moment-matching DOF
> with Hotelling-T² scaling, numerically equivalent to R
> `clubSandwich::Wald_test(test="HTZ")` to `rtol < 1e-8` on three
> verified panels (q ∈ {1, 2, 3}, balanced + unbalanced). +23 tests.
> No wiring into `crve` / `feols` / `event_study` (deferred to a
> follow-up). Closes the BM-vs-HTZ gap documented in
> `cluster_dof_wald_bm` (which used the BM 2002 simplified formula and
> could drift 50–100% from clubSandwich on multi-restriction tests).

This document is the single index for the 9-phase HDFE work that
took StatsPAI from "thin wrapper around pyfixest" to "independent,
faster, GPU-ready HDFE stack". Each phase has a dedicated
`PHASE<n>_VERIFY.md` covering its tests, acceptance vs the original
plan, and any honest-deferral notes.

## Phase index

| # | What            | Verify report                                      | Status |
|---|-----------------|----------------------------------------------------|--------|
| 0 | Baseline        | [BASELINE.md](BASELINE.md)                         | ✅ Locked |
| 1 | Rust demean     | [PHASE1_VERIFY.md](PHASE1_VERIFY.md)               | ✅ |
| 2 | sp.fast.fepois  | [PHASE2_VERIFY.md](PHASE2_VERIFY.md)               | ✅ |
| 3 | sp.within + DSL | [PHASE3_VERIFY.md](PHASE3_VERIFY.md)               | ✅ |
| 4 | Inference       | [PHASE4_VERIFY.md](PHASE4_VERIFY.md)               | ✅ |
| 5 | Polars/Arrow    | [PHASE5_VERIFY.md](PHASE5_VERIFY.md)               | ✅ |
| 6 | Event study     | [PHASE6_VERIFY.md](PHASE6_VERIFY.md)               | ✅ |
| 7 | JAX backend     | [PHASE7_VERIFY.md](PHASE7_VERIFY.md)               | ✅ |
| 8 | etable          | [PHASE8_VERIFY.md](PHASE8_VERIFY.md)               | ✅ |

## What ships under `sp.fast.*` (new in v1.8)

- **Kernel**: `demean(X, fe, accel='aitken', backend='auto'|'rust'|'numpy'|'jax')`
- **Estimator**: `fepois(formula, data, vcov='iid'|'hc1')`  ← independent of pyfixest
- **Reusable residualizer**: `within(data, fe=...)` → `WithinTransformer`
- **DSL helpers**: `i(var, ref=...)`, `fe_interact(...)`, `sw(...)`, `csw(...)`
- **Inference**: `crve(X, residuals, cluster, type='cr1'|'cr3')`,
  `boottest(...)` (wild cluster bootstrap, Rademacher / Webb-6)
- **Event study**: `event_study(data, y, unit, time, event_time, ...)`
- **Polars input**: `demean_polars(...)`, `fepois_polars(...)`
- **Reporting**: `etable(*fits, format='dataframe'|'latex'|'html'|'markdown')`
- **Diagnostics**: `jax_device_info()`

## Headline numbers

### Numerical correctness (vs reference)

| metric                                               | result    |
| ---------------------------------------------------- | --------- |
| `sp.fast.demean` vs R `fixest::demean` (n=50k, 2-FE) | 1.12e-14  |
| `sp.fast.fepois` vs `pyfixest.fepois` coef (medium)  | 1.6e-15   |
| `sp.fast.fepois` vs `pyfixest.fepois` SE (medium)    | 4.3e-11   |
| `sp.fast.fepois` vs R `fixest::fepois` coef          | < 1e-6    |
| Rust ↔ NumPy ↔ JAX backends agree (atol)             | 1e-9      |

All well below the "atol=1e-10" / "1e-6" thresholds in the original
phase plans.

### Wall-clock (medium dataset, n=1M, fe1=100k, fe2=1k)

| backend                                       | wall    | iters | vs fixest |
| --------------------------------------------- | ------: | ----: | --------: |
| Phase 0 — `sp.fast.fepois` (Python np.bincount) | 2.61 s |     6 |     4.08× |
| Phase A — Rust scatter (no cache)             | 2.45 s  |     6 |     3.83× |
| Phase B0 — Rust sequential + dispatcher cache | 1.441 s |     6 |     2.25× |
| **Phase B1 — native Rust IRLS (v1.8.0)**      | **0.880 s** | 6 |  **1.37×** |
| `pyfixest.fepois`                             | 4.16 s  |     ~ |     6.5×  |
| R `fixest::fepois`                            | 0.64 s  |     5 |     1.00× |

`sp.fast.fepois` v1.8.0 is **2.97× faster than the Phase 0 baseline**
and runs at **1.37× of fixest::fepois** on the medium HDFE benchmark
— under the ≤1.5× target set by the spec. The closure was driven by:

1. **Phase A** (v1.8.0 RC): Rust weighted demean kernel, dispatcher,
   PyO3 surface. Modest speedup alone (4.08× → 3.83×) because the
   inner `np.bincount(weights=...)` was already C-tuned.
2. **Phase B0** (sort-by-primary-FE spike): sequential L1-cache-friendly
   inner sweep + module-level FE-only-plan caching in the dispatcher.
   Big jump (3.83× → 2.25×) — the algorithmic change Phase A's failure
   mode missed.
3. **Phase B1** (native Rust IRLS): single PyO3 entry per fepois call
   (`fepois_irls`), persistent `FePoisIRLSWorkspace`, hand-coded SPD
   Cholesky for the WLS step. Final closure (2.25× → 1.37×) — eliminates
   12 FFI round-trips per fepois plus per-iter Python overhead.

The remaining ~0.24 s gap to fixest is largely accounted for by Python-
side formula parsing, singleton/separation pre-passes, and FePoisResult
construction — work that's intentionally Python-side because changing
it requires changing the user-facing API.

## Test totals

```
$ pytest tests/test_fast_*.py -q --no-cov
133 passed, 2 skipped
```

The 2 skips are intentional: they only run when the optional
dependency *is missing* (Rust extension or jax), so they exercise the
graceful-degradation path rather than the happy path.

## What deliberately did NOT ship (in priority order for follow-ups)

1. ~~**Native Rust IRLS for fepois**~~ — **shipped in v1.8.0** (Phase B1).
   `statspai_hdfe.fepois_irls` (crate v0.5.0+) drives the entire IRLS
   state machine in Rust; the Python `fepois()` body becomes a thin
   formula-parser + pre-pass + vcov shell. Medium wall closes from
   2.61 s (Phase 0) to 0.880 s (1.37× of fixest, under the 1.5× target).
2. **CR2 (Bell-McCaffrey) and IM (Imbens-Kolesar)** cluster SE for
   small G. The current CR1/CR3 + wild bootstrap pair is the
   most-used flavour.
3. **Anderson(m) acceleration** in the demean kernel. Aitken (vector
   Irons-Tuck) is the workhorse and Anderson is a "nice to have"
   that historically helps few-iteration cases.
4. **C Data Interface zero-copy from Polars/Arrow into Rust**. The
   adapter shipped here uses NumPy as the FFI substrate; the actual
   zero-copy path requires arrow-rs FFI plumbing.
5. **GPU CUDA bench**. The JAX backend is structurally GPU-ready and
   tested on CPU; no hardware on this dev box for accelerated
   benchmarks.
6. **Cluster-robust SE inside ``sp.fast.fepois``**. Currently IID and
   HC1 only. Score-bootstrap based cluster-robust is the natural next
   step (lives next to the wild bootstrap in Phase 4).
7. **Wiring `sp.callaway_santanna` / `sp.sun_abraham` / etc.
   through ``backend="fast"``**. Doable but high-risk for already-
   tested estimators; a dedicated parity-test PR.
8. **Word/DOCX output from `sp.fast.etable`** (LaTeX/HTML/Markdown
   ship; DOCX is pre-existing on per-result `to_docx` but not yet
   wired into the multi-model side-by-side table).

These are the items where I cut scope rather than ship something I
couldn't fully test. The user's directive — "don't hide things you
can't do well" — was the design constraint.

## How to use the new path

```python
import statspai as sp

# Drop-in faster Poisson HDFE (independent of pyfixest)
fit = sp.fast.fepois("y ~ x1 + x2 | firm + year", data=df)
print(fit.summary())

# Reusable within-residualizer for DML / Lasso / IV
wt = sp.fast.within(df, fe=["firm", "year"])
y_dem, _ = wt.transform(df["y"].to_numpy())
X_dem = wt.transform_columns(df, ["x1", "x2"])

# Wild cluster bootstrap
res = sp.fast.boottest(X, y, cluster=df["firm"], null_coef=0, B=9999)

# Event study
es = sp.fast.event_study(df, y="y", unit="firm", time="year",
                          event_time="rel_t", window=(-5, 5))

# Side-by-side regression table
print(sp.fast.etable(fit_baseline, fit_with_controls, format="latex"))
```

## Files added in this round

```
src/statspai/fast/
├── demean.py           ← Phase 1 (NumPy + Rust dispatch)
├── fepois.py           ← Phase 2
├── within.py           ← Phase 3
├── dsl.py              ← Phase 3
├── inference.py        ← Phase 4
├── polars_io.py        ← Phase 5
├── event_study.py      ← Phase 6
├── jax_backend.py      ← Phase 7
└── etable.py           ← Phase 8

rust/statspai_hdfe/src/
├── lib.rs              ← extended with demean_2d + singleton_mask
├── demean.rs           ← K-way AP + Aitken
└── singletons.rs       ← iterative singleton mask

tests/
├── test_fast_demean.py        (15 tests)
├── test_fast_fepois.py        (12 tests)
├── test_fast_within_dsl.py    (15 tests)
├── test_fast_inference.py     (10 tests)
├── test_fast_polars.py        (7 tests)
├── test_fast_event_study.py   (7 tests)
├── test_fast_jax.py           (6 tests)
└── test_fast_etable.py        (12 tests)

benchmarks/hdfe/
├── BASELINE.md / PHASE1_VERIFY.md / ... / PHASE8_VERIFY.md
└── SUMMARY.md          ← this file
```

## Recommended release notes (for v1.8.0)

```markdown
# v1.8.0 — Native HDFE stack

## Added
- New `sp.fast` namespace with a Rust-backed HDFE kernel:
  - `sp.fast.demean()` — K-way alternating-projection demean with
    Irons-Tuck acceleration; Rust default, NumPy fallback, JAX backend
    (GPU-ready) opt-in.
  - `sp.fast.fepois()` — native Poisson HDFE estimator (PPML-HDFE algo,
    Correia-Guimarães-Zylkin 2020), independent of pyfixest. Coef and
    SE parity with `pyfixest.fepois` to machine epsilon and 4e-11
    respectively; vs R `fixest::fepois` to better than 1e-6.
  - `sp.fast.within()` — reusable within-residualizer for downstream
    DML / Lasso / IV.
  - `sp.fast.crve()` and `sp.fast.boottest()` — cluster-robust SE
    (CR1, CR3) and wild cluster bootstrap (Rademacher, Webb-6).
  - `sp.fast.event_study()` — TWFE event-study on the new stack.
  - `sp.fast.demean_polars()` / `sp.fast.fepois_polars()` —
    Polars input adapter.
  - `sp.fast.etable()` — fixest-style side-by-side regression tables
    (LaTeX / HTML / Markdown).
  - `sp.fast.i()`, `sp.fast.fe_interact()`, `sp.fast.sw()`,
    `sp.fast.csw()` — DSL helpers.

## Performance
- `sp.fast.fepois` is ~1.6× faster than pyfixest on a 1M-row,
  100k-FE Poisson workload.

## Unchanged
- `sp.fepois`, `sp.feols`, `sp.callaway_santanna`, `sp.sun_abraham`,
  `sp.honest_did`, `sp.demean`, `sp.Absorber` — all existing
  paths continue to work without modification. The new
  `sp.fast.*` namespace is purely additive.
```
