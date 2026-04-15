# SP-01: Spatial Econometrics Full-Stack — Design Spec

**Date**: 2026-04-15
**Parent**: [ecosystem gap analysis](2026-04-15-statspai-ecosystem-gap-analysis.md)
**Scope**: Domain #16 of the gap matrix — close the largest current gap in StatsPAI.
**Sprint**: S1 (sub-phases S1.1 / S1.2 / S1.3 — see §7)
**Status**: Design approved — ready for `writing-plans`.

## 1. Why this sub-project

The existing `statspai.spatial` module ships only three ML estimators
(SAR / SEM / SDM, dense matrices, 438 LOC). The R side ships
`spatialreg + spdep + splm + sphet + GWmodel` (~60 functions). The Python
side ships PySAL (`libpysal + esda + spreg + mgwr + splot`, ~100 public
functions). The gap is the largest in StatsPAI, and the existing dense
implementation does not scale beyond N ≈ 2000.

Closing this gap makes StatsPAI the first unified Python package where a
user can go weights → ESDA → regression → impacts → GWR → spatial panel
without touching `libpysal`, `spreg`, or `mgwr` directly, while still
interoperating with them.

## 2. Success criteria

A user can do each of the following with `import statspai as sp`:

1. Build spatial weights from a `GeoDataFrame` or coordinate array
   (queen, rook, KNN, distance-band, kernel, block) and row-normalise,
   binary-transform, or variance-stabilise them.
2. Run Moran's I (global and local), Geary's C, Getis-Ord G, and join
   counts, with permutation-based inference and publication-ready plots.
3. Estimate SAR / SEM / SDM / SLX / SAC models by ML (small-to-moderate N)
   or Kelejian-Prucha GMM (large N, heteroscedasticity-robust), with
   LM / Robust-LM diagnostics for model selection.
4. Recover direct / indirect / total impacts with simulated standard
   errors (LeSage-Pace 2009).
5. Fit Geographically Weighted Regression (GWR) and Multiscale GWR
   (MGWR) with cross-validation or AICc bandwidth selection.
6. Fit spatial panel models (FE × {SAR, SEM, SDM}) with Baltagi-style
   inference.

Numerical tolerance vs PySAL / R `spatialreg` on the Columbus and Boston
housing datasets: **< 1e-4 relative error on all estimated parameters**.

## 3. Architecture

```
src/statspai/spatial/
├─ __init__.py            # re-exports flat API (queen_weights, moran, sar, ...)
├─ weights/
│   ├─ core.py            # W class (sparse CSR backing)
│   ├─ contiguity.py      # queen, rook (needs geopandas)
│   ├─ distance.py        # knn, distance_band, kernel
│   ├─ block.py           # block / regime weights
│   └─ transform.py       # row / binary / variance-stabilise transforms
├─ esda/
│   ├─ moran.py           # global + local
│   ├─ geary.py
│   ├─ getis_ord.py       # G and G* (global + local)
│   ├─ join_counts.py
│   └─ plots.py           # moran_plot, lisa_cluster_map
├─ models/
│   ├─ base.py            # shared estimation machinery
│   ├─ ml.py              # SAR/SEM/SDM/SLX/SAC via ML (concentrated LL)
│   ├─ gmm.py             # Kelejian-Prucha GMM
│   ├─ diagnostics.py     # LM-err, LM-lag, Robust-LM, Moran-residual
│   ├─ impacts.py         # LeSage-Pace direct / indirect / total
│   └─ logdet.py          # exact eigen + Barry-Pace / Chebyshev approx
├─ gwr/
│   ├─ gwr.py             # GWR (Fotheringham et al.)
│   ├─ mgwr.py            # Multiscale GWR (Fotheringham-Yang-Kang 2017)
│   └─ bandwidth.py       # golden-section, CV, AICc selection
├─ panel/
│   ├─ spatial_panel.py   # FE × {SAR, SEM, SDM}
│   └─ tests.py           # Baltagi LM, CD test
└─ io/
    └─ geopandas_adapter.py   # lazy-import shim
```

## 4. Public API (final surface)

### 4.1  Weights (L1)

```python
sp.spatial.queen_weights(gdf)
sp.spatial.rook_weights(gdf)
sp.spatial.knn_weights(coords, k=5)
sp.spatial.distance_band(coords, threshold, binary=True)
sp.spatial.kernel_weights(coords, bandwidth, kernel='gaussian', fixed=True)
sp.spatial.block_weights(regimes)
sp.spatial.W(neighbors, weights=None)            # explicit constructor
# W methods: .transform('R'|'B'|'V'|'D'), .sparse, .neighbors, .islands,
#            .asymmetries, .full(), .to_libpysal()
```

### 4.2  ESDA (L2)

```python
sp.spatial.moran(y, W, permutations=999)
sp.spatial.moran_local(y, W, permutations=999)
sp.spatial.geary(y, W, permutations=999)
sp.spatial.getis_ord_g(y, W, permutations=999)
sp.spatial.getis_ord_local(y, W, permutations=999, star=True)
sp.spatial.join_counts(y, W, permutations=999)    # binary y
sp.spatial.moran_plot(y, W, ax=None)
sp.spatial.lisa_cluster_map(y, W, gdf, ax=None)   # needs geopandas
```

All return a `SpatialStatistic` result with `.I` / `.G` / etc., `.p_value`,
`.z_score`, `.expectation`, `.variance`, `.simulations`, `.summary()`.

### 4.3  Spatial regression (L3)

```python
# ML estimators (upgraded from existing dense to sparse)
sp.sar(W, data=df, formula='y ~ x1 + x2')
sp.sem(W, data=df, formula='y ~ x1 + x2')
sp.sdm(W, data=df, formula='y ~ x1 + x2')
sp.slx(W, data=df, formula='y ~ x1 + x2')         # new
sp.sac(W, data=df, formula='y ~ x1 + x2')         # new (SAR + SEM)

# GMM estimators (new)
sp.sar_gmm(W, data=df, formula=..., robust='het')
sp.sem_gmm(W, data=df, formula=..., robust='het')
sp.sarar_gmm(W, data=df, formula=...)

# Diagnostics
sp.spatial.lm_tests(formula, data, W)             # {LM-err, LM-lag, RLM-err, RLM-lag, SARMA}
sp.spatial.moran_residuals(residuals, W)

# Impacts (LeSage-Pace 2009)
sp.spatial.impacts(result, n_sim=1000, parallel=False)
# -> DataFrame(direct, indirect, total, se_direct, se_indirect, se_total)
```

### 4.4  GWR / MGWR (L4)

```python
sp.gwr(coords, y, X, bw='cv', kernel='bisquare', fixed=False)
sp.mgwr(coords, y, X, bw_init='gwr', max_iter=200)
sp.spatial.gwr_bandwidth(coords, y, X, criterion='aicc')
# result: .params (n×k local coefs), .localR2, .residuals, .summary(), .map(gdf)
```

### 4.5  Spatial panel (L5)

```python
sp.spatial_panel(
    data, formula, entity, time, W,
    model='sar',         # 'sar' | 'sem' | 'sdm'
    effects='fe',        # 'fe' | 're' | 'twoways'
)
sp.spatial.panel_lm_tests(data, formula, entity, time, W)   # Baltagi LM
```

### 4.6  Backward compatibility

Existing `sp.sar / sp.sem / sp.sdm` keep their current signature
(`(W: ndarray, data, formula, row_normalize=True, alpha=0.05)`). Internally
they now accept either an `ndarray` or a `sp.spatial.W` object and route to
the sparse solver automatically. Existing tests must continue to pass.

## 5. Key design decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | **Sparse-first.** `W._sparse: scipy.sparse.csr_matrix` is the canonical storage. Dense conversions only when strictly needed. | Current dense impl caps at N ≈ 2000. Sparse unlocks N ≥ 10⁵. |
| D2 | **Dual log-det path.** Exact eigendecomposition for N < 5000; Barry-Pace Monte-Carlo / Chebyshev approximation for N ≥ 5000 (configurable via `logdet='auto'\|'exact'\|'approx'`). | PySAL `spreg` and R `spatialreg` both use this split. |
| D3 | **Dependencies.** `numpy/scipy/pandas` hard. `geopandas`, `libpysal`, `shapely`, `joblib` **soft** — lazy-imported, raise actionable `ImportError` only when a code path needs them. | Does not bloat base install. |
| D4 | **Interop, not wrap.** We re-implement the core (weights, Moran, ML, GMM). `W.to_libpysal()` and `W.from_libpysal(w)` provide bridges. | A thin wrapper is not a real package — user request explicitly asked "front-of-house capability". |
| D5 | **Unified result object.** All L3/L4/L5 estimators return `EconometricResults` augmented with spatial extras (ρ / λ, impacts, log-lik, `.moran_residuals`). | Matches StatsPAI-wide convention; enables `sp.outreg2([sar_result, sem_result])`. |
| D6 | **LeSage-Pace impacts are default.** `result.summary()` always shows impacts for SAR / SDM / SAC models, not just ρ. | Reporting ρ without impacts is known to mislead; current StatsPAI impl does not report impacts. |
| D7 | **Cross-validated numerics.** Every estimator's test suite asserts < 1e-4 relative error vs PySAL reference on the Columbus dataset. | Gold-standard validation, consistent with DID / RD testing. |
| D8 | **Weights construction is pure-Python + numpy.** No compiled extensions in S1.1. | Keeps installation simple. We can add Cython/Numba later if profiling demands. |

## 6. YAGNI (explicitly out of scope)

- Spatial point-process models (L and K functions, `pointpats` territory).
- Bayesian spatial models (CAR, ICAR, SPDE) — future sub-spec via PyMC.
- Space-time kriging and geostatistics — out of econometric scope.
- Network-based spatial models — overlaps with SP-05 (causal discovery).
- Real-time streaming / GPU-accelerated solvers.

## 7. Phased delivery

| Phase | Scope | Estimated duration | Exit criterion |
|---|---|---|---|
| **S1.1** | L1 weights + L2 ESDA + L3 ML refactor to sparse + L3 GMM + impacts + LM tests | 3–4 wk | Columbus cross-val passes; existing tests green |
| **S1.2** | L4 GWR + MGWR + bandwidth | 2–3 wk | Georgia GWR dataset cross-val vs `mgwr` < 1e-4 |
| **S1.3** | L5 Spatial panel + Baltagi diagnostics | 2 wk | `splm::spml` cross-val on Produc dataset |

Each phase is independently mergeable to `main`.

## 8. Testing strategy

1. **Numerical cross-validation** against PySAL (Columbus, Boston, Baltimore,
   NCOVR, Georgia) and R `spatialreg` (saved reference outputs in
   `tests/fixtures/spatial/`). Tolerance: 1e-4 relative.
2. **Unit tests**: ≥3 per public function (happy path, edge case, error).
3. **Property tests**: `W.transform('R')` idempotent on already-row-standardised
   `W`; Moran's I = 0 on iid data at the 0.05 level > 94 % of runs.
4. **Regression tests**: existing `sp.sar/sem/sdm` outputs unchanged within
   1e-8 (pure refactor should not move numbers).
5. **Doc tests**: every docstring example runs in CI.

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Sparse eigensolver instability for N ≈ 5000 boundary | Auto-heuristic chooses path; warn and suggest explicit `logdet=` when condition number high. |
| `geopandas` install pain on Windows | Contiguity weights accept either `gdf` or a precomputed adjacency list; docs cover both paths. |
| LeSage-Pace impacts simulation slow for large N | Default `n_sim=1000`, document `parallel=True`; provide closed-form mean impacts without SE as fallback. |
| GWR memory for large N (n×n bandwidth evaluation) | Ship chunked implementation; warn above N=50_000. |
| MGWR convergence tuning vs `mgwr` package | Adopt `mgwr`'s back-fitting loop verbatim; cross-validate on Georgia dataset. |

## 10. Out of this spec (anchors for future specs)

- Performance profiling and potential Numba/Cython inner loops → perf sub-spec.
- Bayesian spatial models (CAR / ICAR) → future SP-Bayes.
- Spatial causal inference (spatial DID, spillover identification) →
  overlaps with domain #21 (experimental design) and existing `interference`
  module; deferred.

## 11. Post-approval flow

1. `writing-plans` skill generates a step-level implementation plan for
   **S1.1 only** (phase-gated — S1.2 / S1.3 get their own plans when their
   phase begins).
2. `executing-plans` drives the build, with user review checkpoints at
   end of L1, L2, L3.
3. Each checkpoint ends with direct push to `main`.
