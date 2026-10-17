# Rust kernels

StatsPAI's fixed-effect machinery can run on optional Rust kernels:
high-dimensional fixed-effect (HDFE) demeaning, the whole Poisson IRLS of
`sp.fast.fepois`, and the cluster-robust sandwich meat. They ship as a
separate package so that StatsPAI itself stays pure Python:

```bash
pip install "statspai[rust]"      # pulls in statspai-hdfe
```

Nothing requires them. Without the package every function below runs on
NumPy / Numba and gives the same estimates to within the residualisation
tolerance.

```python
import statspai as sp

sp.fast.backend_info()
# {'available': True, 'version': '0.8.0', 'api_version': 2, 'threads': 8, ...}

sp.fast.set_num_threads(4)   # or STATSPAI_NUM_THREADS=4 before starting Python
fit = sp.fast.feols("y ~ x1 + x2 | firm + year", data=df)
fit.fe_kernel                # 'rust' or 'numpy'
```

Set `STATSPAI_SKIP_RUST=1` to make StatsPAI ignore an installed extension
(for example to reproduce a result computed without it).

## What uses it

| Entry point | Rust kernel used for |
| --- | --- |
| `sp.fast.feols`, `sp.fast.demean`, `sp.fast.within`, `sp.fast.event_study` | demeaning (weighted with `weights=`) |
| `sp.fast.fepois` | the whole IRLS loop, separation pre-pass |
| `sp.hdfe_ols`, `sp.absorb_ols`, `sp.demean`, `sp.Absorber` | demeaning, for plain fixed effects with `solver="map"` (varying slopes and the LSMR / LSQR solvers stay on Numba / SciPy) |
| `sp.ppmlhdfe` (two or more fixed effects) | weighted demeaning inside IRLS |
| `sp.iv(..., absorb=)`, `sp.didregress`, `sp.iv.iv_diag`, weak-IV diagnostics | demeaning |
| `sp.regress(..., cluster=)` (one-way) | cluster-robust meat |

`sp.feols` / `sp.fepois` (backed by pyfixest) and `sp.panel(method="fe")`
(backed by linearmodels) do not use it.

## How it works

The fixed-effect projection is solved in the space of the FE
coefficients, the approach of fixest (Bergé 2018, `berge2018efficient`):

- **Two fixed effects.** One Gauss-Seidel sweep needs only the group sums
  of the column and the sparse cross-tabulation of the two FEs, so it costs
  as much as the number of distinct (FE 1, FE 2) pairs, not the number of
  rows. On a firm-year panel with a million rows and 2,000 firms observed
  over 20 years, that is 40,000 cells. The iteration is accelerated with
  Irons-Tuck at every step.
- **Three or more.** The two FEs with the most levels (typically worker and
  firm) form a block solved by the two-FE routine inside an outer loop over
  the others; the inner tolerance starts loose and tightens as the outer
  loop converges.
- **Stopping rule.** As in the NumPy path: stop when the change of the
  residual between two sweeps is at most `tol_abs + tol * max|x|`.

Columns run in parallel. Results do not depend on the number of threads:
each column is processed by one thread, and the cluster meat adds fixed
blocks of clusters in a fixed order.

## Speed

Measured on 2026-10-18 on an Apple-silicon laptop:

Machine: macOS-26.6.2-arm64-arm-64bit, 8 logical CPUs, Python 3.10.20, statspai-hdfe 0.8.0. Median of 5 runs, seconds; n = 1,000,000.

| model | NumPy | Rust, 1 thread | Rust, 8 threads | fixest, 1 thread | fixest, 8 threads |
|---|---:|---:|---:|---:|---:|
| easy/feols | 0.144 | 0.053 | 0.048 | 0.078 | 0.049 |
| easy/fepois | 0.367 | 0.190 | 0.169 | 0.397 | 0.243 |
| hard/feols | 6.245 | 0.632 | 0.308 | 0.743 | 0.338 |
| hard/fepois | 23.864 | 3.203 | 1.450 | 4.645 | 2.316 |

Largest coefficient difference from the NumPy path: 2.7e-09.

The `easy` panel is the Track C one (random firm and year effects); the
`hard` one has worker, firm and year effects with 5% of observations away
from the worker's home firm, the low-mobility case where alternating
projections crawl. Reproduce with
`python benchmarks/hdfe/rust_kernels_bench.py --fixest` on an idle machine.

## Numerical agreement

The Rust and NumPy paths iterate differently, so their answers differ in
the last digits, by at most a small multiple of the stopping threshold. The
test suite checks both against the exact projection from dense least
squares (`tests/test_rust_backend.py`). The R / Stata parity results of the
JSS paper are computed without the extension.

Two consequences worth knowing:

- An estimate reproduced with and without the extension agrees to the
  tolerance, not bit for bit. Record `sp.fast.backend_info()` (the
  replication pack does) when exact reproduction matters, or set
  `STATSPAI_SKIP_RUST=1`.
- A regressor the fixed effects absorb comes back from the residualisation
  as rounding noise, not as exact zeros. `sp.fast.feols`, `sp.fast.fepois`
  and `sp.fast.event_study` detect this and raise `NumericalInstability`
  naming the regressor, with or without the extension.

## Building from source

The crate is in `rust/statspai_hdfe`:

```bash
pip install maturin
cd rust/statspai_hdfe
maturin develop --release          # into the active environment
PYO3_PYTHON=python cargo test --release
```

Wheels are abi3, so one wheel per platform covers every CPython from 3.10
on.
