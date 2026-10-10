# Efficiency pass: where the package was slow and what was done

*2026-10-11. Worktree `wt/efficiency`.*

## How the survey was made

Nothing was assumed about where time goes. Three measurements were taken
on the tree at `b9055879` (1.39.3):

1. **Cold import.** `python -X importtime -c "import statspai"`.
2. **Every docstring example.** The 1,536 runnable examples of the
   registered functions were executed and timed (337 s in total).
3. **Scaling.** 124 estimators were run at three sizes: 2,000 / 20,000 /
   200,000 rows for cross-sections, 200 / 2,000 / 20,000 units by 10
   periods for panels, 20 / 50 / 100 donors for synthetic control. The
   exponent between the two largest sizes that ran shows whether a method
   is linear. This sweep is now `benchmarks/bench_scaling.py`.

Everything that took seconds at the smallest size or grew like `n^1.5`
was profiled with `cProfile` before it was touched.

## What the survey found

**Import** takes about 1.0 s. Pandas and `scipy.stats` account for 0.55 s
and are needed by nearly every function. StatsPAI's own 685 modules
account for 0.4 s, spread evenly (0.6 ms each); no single module is
expensive. No heavy optional dependency is imported (numba, sklearn,
statsmodels, matplotlib all stay out, as the import-budget gate requires).

**Examples.** 1,327 of 1,536 run in under 0.1 s. Five take more than
10 s, and each of those is a default that refits a machine-learning
learner or a placebo test hundreds of times.

**Scaling** is where the problems were. Four patterns account for almost
all of them.

| Pattern | Where |
| --- | --- |
| A risk set or neighbourhood rebuilt from the data for each observation, `O(n^2)` | `cox` (113 s at n = 20,000), `kaplan_meier`, `logrank_test`, `match` / `psmatch2`, Harrell's C |
| A Python or pandas loop over units, clusters or groups | `event_study`, `lp_did`, `did_multiplegt_dyn`, `mixed` (a dense Cholesky per group per likelihood evaluation) |
| An iterative solver written as a Python loop | Frank-Wolfe in `sdid` (2.7 million `argmin` calls for one fit with placebo inference), coordinate descent in sparse synth |
| A dense `n x n` matrix built for a diagonal or a projection | `fracreg`, `ivqreg`, `rdit`, `rdmc`, `rd_distributional_design`, `qte(conditional_qr)`, `robreg(init='lad')` |

Two more were of a different kind: `qreg` handed a linear programme to a
general solver whose cost grows like `n^1.6`, and `nbreg` on data without
overdispersion spent its whole iteration budget and then warned that it
had not converged.

## The plan, and what was done

The rule for the pass: speed comes from doing the same arithmetic with
less overhead, and each rewrite is compared with the code it replaces
before anything else. A method whose slowness is its definition (a
bootstrap of a boosted-tree fit) is left alone and listed below.

| Step | Scope | Evidence that numbers did not move |
| --- | --- | --- |
| 1 | Cox, Kaplan-Meier, log-rank on sorted risk sets | 216 designs against the old code, 1e-12; brute-force definitions in `tests/test_cox_risk_set_kernel.py` |
| 2 | `qreg`: interior point finished at the exact vertex | 118 designs against HiGHS, 5e-13; the subgradient condition is checked on every fit |
| 3 | Nearest-neighbour matching | 2,720 designs, bit-identical |
| 4 | `event_study`, `lp_did`, `did_multiplegt_dyn` | 1,146 designs, bit-identical |
| 5 | SDID and sparse-synth kernels (numba, lazy) | 577 designs and 36,000 kernel problems, bit-identical on this machine |
| 6 | `mixed` from per-group summaries | likelihood at 6,480 parameter draws; where old and new differ the new one is right against 40-digit arithmetic |
| 7 | `n x n` allocations | bit-identical; memory tests that fail on the old code |
| 8 | Small items: ROC area, Gaussian kernel, truncated-normal quantile, `nbreg` stopping rule | bit-identical except `nbreg` at the boundary |
| 9 | `benchmarks/bench_scaling.py` | the survey, kept |

Steps 3 to 6 were delegated to four sub-agents working on disjoint files
in the same worktree, each with the instruction to keep a copy of the
original code and compare against it.

Before and after, same machine, under load:

| Call | Size | Before | After |
| --- | --- | --- | --- |
| `sp.cox` | n = 20,000 | 113 s | 0.07 s |
| `sp.kaplan_meier` | n = 200,000 | 22 s | 0.06 s |
| `sp.qreg` | n = 200,000 | 30 s | 0.7 s |
| `sp.match` | n = 200,000 | 51 s | 0.5 s |
| `sp.mixed` | n = 200,000 | 112 s | 0.12 s |
| `sp.event_study` | 20,000 units | 14 s | 0.12 s |
| `sp.lp_did` | 20,000 units | 74 s | 1.3 s |
| `sp.did_multiplegt_dyn` | 200 units | 26 s | 0.4 s |
| `sp.sdid` | 20 donors | 17 s | 0.5 s |
| `sp.synth(method='sparse')` | 20 donors | 54 s | 0.4 s |
| `sp.fracreg`, `sp.ivqreg` | n = 200,000 | out of memory | 0.35 s, 14 s |

## Things worth knowing later

- **`X.T @ np.diag(w) @ X` and `(X.T * w) @ X` are not bit-identical.**
  The second leaves `X'W` column-major and BLAS then rounds the next
  product differently. `np.ascontiguousarray(X.T * w)` restores the
  layout and the bits.
- **A numba kernel reaches BLAS through SciPy, numpy code through
  NumPy.** On this machine both link Accelerate and the SDID kernels are
  bit-identical to the Python loops. Where the two link different builds
  the last bit can differ; the kernel tests use `atol=1e-12` for that
  reason. If the SDID parity rows move on Linux CI, this is the first
  place to look.
- **A strided `ddot` rounds differently from a contiguous one.** The
  sparse-synth kernel calls `ddot` on the column in place through
  `scipy.linalg.cython_blas`, because copying the column first changed
  3,770 of 6,000 results in the last bit.
- **pre-commit and concurrent editors do not mix.** The bandit hook takes
  40 s; if another process edits any file in the worktree meanwhile, the
  commit fails with "files were modified by this hook". Commit when the
  tree is quiet.
- **`nbreg` at the boundary.** With no overdispersion the dispersion is
  not identified and the reported standard error of its logarithm depends
  on where the search stops. The coefficients do not. A proper treatment
  would report the boundary and a one-sided test, as Stata does.

## Left alone, with the reason

| Item | Measured | Why it was not changed |
| --- | --- | --- |
| Cold import, 1.0 s | 0.4 s is StatsPAI's own modules | Making `__init__` lazy would save at most 0.4 s and touches the most contended file in the repository, the stub generator and every static analyser. Several functions share a name with their subpackage (`sp.did`, `sp.synth`), which a lazy loader has to guard explicitly. Worth doing only as its own project. |
| `sp.cbps`, 39 s at n = 2,000 | 501 fits, five BFGS runs each | The default standard error is a bootstrap of a non-convex GMM fit. The remedy is an analytic sandwich variance, which is a new feature. |
| `sp.dose_response`, `sp.tmle`, `sp.dml`, `sp.metalearner`, `sp.bcf_*` | time is inside scikit-learn | The learners are the cost. |
| `sp.ipw`, `sp.overlap_weights`, `sp.g_computation`, `sp.qte` | linear, 500 bootstrap refits | By definition of the default. `sp.ipw(se_method='sandwich')` is 400 times faster and already exists. |
| `sp.mc_panel` | 200 bootstrap fits, 80,000 SVDs | Warm-starting each bootstrap fit from the full-sample fit cuts iterations by 22 to 33% but moves fitted values by 1e-8 to 3e-7. Not done here. |
| `sp.rdrobust`, 2.1 s at n = 200,000 | nearest-neighbour residual loop | On the path of a dozen Track A modules; comparable to R at this size. |
| `sp.cuminc`, 26 s at n = 200,000 | Gray's test is a recursion over event times | Linear, and the recursion carries state from one time to the next. |
| `sp.policy_tree`, depth 2 | exact search | Quadratic by design, as in `policytree`. |
| Mahalanobis matching | the distance matrix | A tree search would not reproduce the tie sets. |
| `sp.regress("y ~ x + C(g)")` with thousands of levels | dense dummies | Use `sp.hdfe_ols` / `sp.feols`. A hint when a factor has many levels would help. |
| `sp.honest_did` relative magnitudes | vertex enumeration with a rank check per candidate | Next in line if this function matters more; needs care with the parity rows. |
| Three-level `sp.mixed` | no Newton polish | Adding it would remove the 1e-4 noise in variance components and would change reported numbers. |

## Rerunning

```bash
python benchmarks/bench_scaling.py --json before.json     # tens of minutes
python benchmarks/bench_scaling.py --only cox              # one family
```

The comparison scripts and the copies of the original code used by each
step were kept in the session scratch directory and are not in the
repository; the tests listed above are what remains.
