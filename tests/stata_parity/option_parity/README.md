# Option-level Stata fixtures — deliberately outside Track A

These are **not** Track A modules and must not live one directory up.

`tests/test_parity_harness_contract.py` globs
`tests/stata_parity/results/*_Stata.json` to enumerate the Track A Stata
leg, and holds every artifact it finds to the Track A contract: a
registered module inventory, a joinable headline row, a registered
tolerance budget, and a strictness tier. That contract is right for the
81 Track A modules, which each pin one estimator's headline number
against a reference implementation.

The artifacts here answer a different question. They pin *option
switches* within an estimator — does `notyet_cutoff='cohort'` reproduce
`csdid`'s default convention, does `same_switchers` reproduce Stata's
sample restriction — so a single file carries several fits of the same
command under different options, with no single headline row. Forcing
them into the joinable row schema would either flatten away the option
axis or require registering four pseudo-modules with budgets that mean
nothing.

The glob is non-recursive, so this subdirectory is invisible to it. That
is the intended effect, not an accident.

| file | pins | consumed by |
| --- | --- | --- |
| `82_csdid_conventions_Stata.json` | `csdid` `asinr` vs default, `method(stdipw)` vs `method(ipw)` | `tests/reference_parity/test_csdid_conventions_stata_parity.py` |
| `83_sunab_control_cohort_Stata.json` | `eventstudyinteract control_cohort()` under two reference groups | `tests/reference_parity/test_sunab_control_cohort_parity.py` |
| `84_bjs_fe_covariates_Stata.json` | `did_imputation` `fe()` / `unitcontrols()` / `timecontrols()` | `tests/reference_parity/test_bjs_fe_covariates_parity.py` |
| `85_multiplegt_dyn_options_Stata.json` | `did_multiplegt_dyn` `switchers()` / `same_switchers` | `tests/reference_parity/test_multiplegt_dyn_options_parity.py` |

Regenerate any of them by running the matching `.do` from
`tests/stata_parity/` with Stata 18 MP and the packages named in its
header.
