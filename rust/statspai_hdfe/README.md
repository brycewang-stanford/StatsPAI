# statspai-hdfe

Rust kernels for [StatsPAI](https://github.com/brycewang-stanford/StatsPAI):
high-dimensional fixed-effect (HDFE) demeaning, the native Poisson IRLS
behind `sp.fast.fepois`, and the cluster-robust sandwich meat.

```bash
pip install statspai-hdfe        # or, with StatsPAI releases after 1.39.6: pip install "statspai[rust]"
```

StatsPAI never requires this package. Without it the same functions run
on NumPy / Numba and agree with the Rust path to within the
residualisation tolerance. Check which one is in use:

```python
import statspai as sp
sp.fast.backend_info()      # {'available': True, 'version': '0.8.0', 'threads': 8, ...}
sp.fast.set_num_threads(4)  # or set STATSPAI_NUM_THREADS before starting Python
```

`STATSPAI_SKIP_RUST=1` makes StatsPAI ignore an installed extension.

## What uses it

| StatsPAI entry point | Kernel |
| --- | --- |
| `sp.fast.feols`, `sp.fast.demean`, `sp.fast.within`, `sp.fast.event_study` | `demean_2d` |
| `sp.fast.feols(weights=)`, `sp.ppmlhdfe` (two or more FEs) | `demean_2d_weighted` |
| `sp.fast.fepois` | `fepois_irls`, `separation_mask` |
| `sp.iv(absorb=)`, `sp.didregress`, `sp.iv.iv_diag`, weak-IV diagnostics | `demean_2d` |
| `sp.regress(cluster=)` (one-way) | `cluster_meat` |
| singleton pruning in `sp.fast` | `singleton_mask` |

`sp.hdfe_ols` / `sp.Absorber`, `sp.panel` and the pyfixest-backed
`sp.feols` do not use it.

## Algorithm

The fixed-effect projection is solved in the space of FE coefficients,
as `fixest` does (Bergé 2018; `paper.bib` key `berge2018efficient`):

* **Two FEs.** One Gauss-Seidel sweep needs only the group sums of the
  column and the sparse cross-tabulation of the two FEs, so it costs
  `O(nnz)` instead of `O(n)`; on panels where a unit meets few levels of
  the other FE (worker x firm, firm x year) that is a small fraction of
  the rows. The iteration runs on the coefficients of the smaller FE and
  is accelerated with Irons-Tuck at every step.
* **Three or more FEs.** The two FEs with the most levels form a block
  solved by the two-FE routine (warm-started) inside an outer
  Gauss-Seidel over the others, accelerated with Irons-Tuck. The inner
  tolerance starts loose and follows the outer change down.
* **Stopping rule.** Stop when the bound on the change of the residual
  between two sweeps is at most `tol_abs + tol_rel * max|x|`, the rule of
  the NumPy reference in `statspai.fast.demean`.
* **IRLS.** The FE structure is built once per `fepois` call; each
  iteration re-weights it and restarts every column from its FE
  coefficients of the previous iteration.

Columns are processed in parallel. The cluster meat sums fixed blocks of
clusters in parallel and adds the block sums in order, so its result is
the same at any thread count; the demeaning is per column and therefore
also independent of the thread count.

## Safety

Every entry point validates lengths, contiguity and code ranges before
running and raises `ValueError` on bad input. The crate is built with
`panic = "unwind"`, so an unexpected panic surfaces as a Python exception
instead of aborting the interpreter. The kernels run on their own thread
pool (never Rayon's global one), which is rebuilt in a forked child.

## Build from source

```bash
pip install maturin
cd rust/statspai_hdfe
maturin develop --release       # into the active environment
maturin build --release         # a wheel in target/wheels
PYO3_PYTHON=python cargo test --release
```

Wheels are abi3: one per platform serves every CPython >= 3.10. The
`rust-hdfe` GitHub workflow tests the crate on Linux, macOS and Windows,
and builds wheels on demand; releases are uploaded from those wheels with
twine.

## Kernel API

`statspai_hdfe.API_VERSION` names the interface level; StatsPAI ignores
an extension older than the level it was written against
(`statspai._rust_backend.MIN_API_VERSION`) and warns once.

| Level | Crate | Change |
| --- | --- | --- |
| 1 | 0.1 – 0.7 | data-space alternating projections; inputs not validated |
| 2 | 0.8 | coefficient-space solver; validated inputs; `fepois_irls` reports `singular` / `fe_converged`; thread-pool control; reproducible `cluster_meat` |

## License

MIT.
