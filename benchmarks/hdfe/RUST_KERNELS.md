# Rust kernels vs NumPy vs fixest

Produced by `python benchmarks/hdfe/rust_kernels_bench.py --reps 5 --fixest` on 2026-10-18 (fixest 0.14.0). Model definitions are in the script's docstring; every other thread pool pinned to one thread.

Machine: macOS-26.6.2-arm64-arm-64bit, 8 logical CPUs, Python 3.10.20, statspai-hdfe 0.8.0. Median of 5 runs, seconds; n = 1,000,000.

| model | NumPy | Rust, 1 thread | Rust, 8 threads | fixest, 1 thread | fixest, 8 threads |
|---|---:|---:|---:|---:|---:|
| easy/feols | 0.144 | 0.053 | 0.048 | 0.078 | 0.049 |
| easy/fepois | 0.367 | 0.190 | 0.169 | 0.397 | 0.243 |
| hard/feols | 6.245 | 0.632 | 0.308 | 0.743 | 0.338 |
| hard/fepois | 23.864 | 3.203 | 1.450 | 4.645 | 2.316 |

Largest coefficient difference from the NumPy path: 2.7e-09.
