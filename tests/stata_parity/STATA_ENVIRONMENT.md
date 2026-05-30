# Stata reference environment for the Track A parity harness

> JSS reproducibility manifest (paper Section 8). Pins the exact Stata engine and community `ado` package versions under which the committed `tests/stata_parity/results/*_Stata.json` golden values were produced and **verified to reproduce bit-for-bit** by `tests/stata_parity/verify_reproduce_stata.py` (`REPRODUCIBILITY_REPORT_STATA.md`). Every `_Stata.json` also carries an inline `provenance` block with the engine fields below.

Unlike R (`packageVersion()`), Stata exposes no per-command version primitive, so the `ado` versions here are the **verbatim first `*!` banner line** of each command's `.ado` file. Commands whose banner carries no version string are recorded as `(no *! version line)` and uninstalled commands as `NOT INSTALLED` — never a guessed version.

## How to regenerate and verify

```bash
# Re-capture the engine + ado inventory:
stata-mp -q -b do tests/stata_parity/_capture_stata_env.do
python tests/stata_parity/_gen_stata_env.py

# Re-run every Stata reference and diff each statistic against the
# committed golden JSON at a 1e-9 reproducibility tolerance:
python tests/stata_parity/verify_reproduce_stata.py
```

## Stata engine

| Field | Value |
|---|---|
| Version | Stata 18 |
| Edition | MP |
| OS | Unix |
| Machine | Mac (Apple Silicon) |
| Executable date | 07 Jun 2023 |

## Community `ado` packages (verbatim `*!` banner)

| Command | `*!` banner line | Path |
|---|---|---|
| `csdid` | *! v1.81  by pedro Sant'Anna. Compatibility checks | `~/Library/Application Support/Stata/ado/plus/c/csdid.ado` |
| `reghdfe` | *! version 6.12.3 08aug2023 | `~/Library/Application Support/Stata/ado/plus/r/reghdfe.ado` |
| `rdrobust` | *!version 10.0.0  2025-06-30 | `~/Library/Application Support/Stata/ado/plus/r/rdrobust.ado` |
| `rddensity` | *!version 2.3 2021-02-28 | `~/Library/Application Support/Stata/ado/plus/r/rddensity.ado` |
| `sdid` | *! sdid: Synthetic Difference-in-Differences | `~/Library/Application Support/Stata/ado/plus/s/sdid.ado` |
| `honestdid` | *! version 1.3.0 25Jan2024 Mauricio Caceres Bravo, mauricio.caceres.bravo@gmail.com | `~/Library/Application Support/Stata/ado/plus/h/honestdid.ado` |
| `did_imputation` | *! did_imputation: Treatment effect estimation and pre-trend testing in staggered adoption diff-in-diff designs with an imputation approach of Borusyak, Jaravel, and Spiess (2023) | `~/Library/Application Support/Stata/ado/plus/d/did_imputation.ado` |
| `jwdid` | *!v2.2 Fixing issue with RCS | `~/Library/Application Support/Stata/ado/plus/j/jwdid.ado` |
| `bacondecomp` | *! 1.0.5 16sep2022 Andrew Goodman-Bacon, Thomas Goldring, Austin Nichols | `~/Library/Application Support/Stata/ado/plus/b/bacondecomp.ado` |
| `eventstudyinteract` | *! version 0.1  24jan2022  Liyang Sun, lsun20@mit.edu | `~/Library/Application Support/Stata/ado/plus/e/eventstudyinteract.ado` |
| `sensemakr` | (no *! version line) | `~/Library/Application Support/Stata/ado/plus/s/sensemakr.ado` |
| `drdid` | (no *! version line) | `~/Library/Application Support/Stata/ado/plus/d/drdid.ado` |
| `oaxaca` | *! version 4.1.1  24apr2023  Ben Jann | `~/Library/Application Support/Stata/ado/plus/o/oaxaca.ado` |
| `ivreg2` | *! ivreg2 4.1.12  14aug2024 | `~/Library/Application Support/Stata/ado/plus/i/ivreg2.ado` |
| `ftools` | *! version 2.49.1 08aug2023 | `~/Library/Application Support/Stata/ado/plus/f/ftools.ado` |
| `gtools` | NOT INSTALLED | `—` |
| `estout` | *! version 3.33  23mar2026  Ben Jann | `~/Library/Application Support/Stata/ado/plus/e/estout.ado` |

> Built-in Stata commands used by the harness (`regress`, `ivregress`, `xtreg`, `mixed`, `melogit`, `frontier`, `xtfrontier`, `stcox`, `tobit`, `mlogit`, `ologit`, `oprobit`, `probit`, `clogit`, `nbreg`, `qreg`, `var`, `newey`, `arima`, `xtabond`, `teffects psmatch`) ship with Stata 18 MP and are versioned by the engine row above.

---

*Captured via `tests/stata_parity/_capture_stata_env.do`. Refresh whenever the Stata environment changes; the per-`_Stata.json` `provenance` block is the authoritative per-result engine record.*
