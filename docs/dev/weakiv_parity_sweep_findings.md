# Weak-IV / diagnostics sweep — findings

Third family in the campaign (`cross_language_coverage_campaign.md`), after
RD and spatial. Reference: R `ivmodel` 1.9.1, `car` 3.1.5 and `metafor`
5.0.1, all already installed.

Both entries below are closed in 1.27.0.

Reference: R `ivmodel` 1.9.1 and `car` 3.1.5, both already installed.
Fixture: 500 obs, two instruments, one exogenous control, endogeneity
induced through a shared error.

## Verified exact (certifiable as-is)

| function | reference | agreement |
| --- | --- | ---: |
| `sp.anderson_rubin_test` — statistic | `ivmodel::AR.test` Fstat | 1.1e-15 |
| `sp.anderson_rubin_test` — p-value | `ivmodel::AR.test` p | 1.2e-13 |
| `sp.anderson_rubin_test` — `ar_ci` | `ivmodel::AR.test` ci | 1.2e-14 |
| `sp.anderson_rubin_test` — df | `(2, 496)` | exact |

## I1 — `sp.vif` rounds its returned values to two decimals (⚠️ correctness)

`sp.vif` returns `VIF` rounded to 2 decimals and `1/VIF` to 4. Against
`car::vif`: 1.16 against 1.1634266096594732, i.e. a relative error of
2.9e-03 that is entirely self-inflicted.

Identical in kind to the `sp.sqreg` defect fixed earlier in this session:
rounding applied to the *returned value* rather than to a display. A VIF
of 10 is the usual rule-of-thumb threshold, and two decimals is enough to
put a value on the wrong side of it in the fourth significant digit.

## I2 — `sp.anderson_rubin_ci` is a grid where an exact interval exists

`sp.anderson_rubin_ci` evaluates the AR statistic on a fixed beta grid and
returns the first and last grid points inside the acceptance region:
[0.3268191729922164, 0.8914962415755244] against `ivmodel`'s
[0.3241785257542151, 0.8959153200836188] — relative errors of 8.1e-03 and
4.9e-03, i.e. grid spacing.

The AR acceptance region is the solution set of a quadratic in beta and
has a closed form. **`sp.anderson_rubin_test` in the same package already
computes it that way** and returns `ar_ci` matching `ivmodel` to 1.2e-14.
So the package contains both the exact interval and a coarse approximation
of the same object, under two names, and the approximation is the one whose
name says "confidence interval".

Not a wrong answer — a grid *is* what the docstring describes — but it is a
gratuitous three orders of magnitude, and two functions disagreeing about
the same quantity is the kind of thing a user finds by accident.

## Resolution

I1 fixed by removing the `round()` calls; `sp.vif` now matches `car::vif`
to 2.0e-16.

I2 fixed by bisecting the acceptance boundary between the last in-grid
point and the first out-grid point, in `_build_set`. The grid still decides
the *shape* of the set (empty / disconnected / unbounded); only the two
endpoints move. `sp.anderson_rubin_ci` goes from 8.1e-3 to 5e-15 against
`ivmodel`. `sp.conditional_lr_ci` improves to 1.7e-4 at n_sim=200,000 and
its residual is now the simulated conditional critical value rather than
the grid, which is a T3 quantity and is graded as one.

**A note on my own error while fixing I2.** The first version passed the
two arguments to the bisection helper in the wrong order for the *upper*
endpoint — `(excess, hi, grid[idx+1])` where the signature is
`(excess, outside, inside)`. The bracket check then rejected the interval
and returned the argument named `inside`, which was the out-of-set grid
point, so the upper endpoint came back *worse* than before while the lower
endpoint was exact. This is the same shape as several defects found this
session, and it was caught in one run by comparing against `ivmodel`
rather than by reasoning about the code.

## Still to check in this family

`sp.conditional_lr_ci` (`ivmodel::CLR`), `sp.effective_f_test`
(Olea-Pflueger; Stata `weakivtest`), `sp.hausman_test` (`plm::phtest`),
`sp.meta_analysis` (`metafor::rma`), `sp.cluster_robust_se` /
`sp.wild_cluster_bootstrap` (`sandwich`, `fwildclusterboot`).
