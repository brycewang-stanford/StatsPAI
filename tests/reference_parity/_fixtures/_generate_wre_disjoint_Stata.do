* Stata 18 MP / boottest 4.5.3 reference for the disjoint WRE confidence
* set of sp.iv_wild_bootstrap (tests/reference_parity/
* test_regcmd_stata_r_parity.py, added 2026-11-06). Same data and fit as
* section G of _generate_regcmd_Stata.do: two endogenous regressors, 12
* clusters, all 2^12 Rademacher sign vectors enumerated.
*
* boottest's default grid ends at 1.6644 for d1, where its own p-value is
* still 0.063 > 0.05, so the second piece it reports by default is cut at
* the end of its grid. With gridmin(-5) gridmax(10) gridpoints(301) (step
* 0.05) the second piece closes at a grid point. p_probe evaluates
* boottest's p-value at fixed points: StatsPAI's edges of the two pieces
* (printed by sp.iv_wild_bootstrap, 2026-11-06) plus and minus 1e-6, so
* the test can check that they are jumps of boottest's own p-value
* function.
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_wre_disjoint_Stata.do
* (run from the repository root; boottest from the adopath)
version 18
clear all
set more off
set type double
local out "tests/reference_parity/_fixtures/wre_disjoint_Stata.json"
tempname fh
file open `fh' using "`out'", write replace text
file write `fh' "{" _n
quietly which boottest
file write `fh' `"  "_meta": {"stata": "`c(stata_version)' `c(edition_real)' (born `c(born_date)')", "boottest": "4.5.3"},"' _n
import delimited using "tests/reference_parity/_fixtures/regcmd_wild.csv", clear asdouble
ivregress 2sls y w (d1 d2 = z1 z2 z3), vce(cluster cl) small
local BT "reps(9999) weighttype(rademacher) ptolerance(1e-12) nograph"
foreach spec in "default|" "wide|gridmin(-5) gridmax(10) gridpoints(301)" {
    gettoken key grid : spec, parse("|")
    local grid = substr("`grid'", 2, .)
    boottest d1, `BT' `grid'
    tempname CI
    matrix `CI' = r(CI)
    file write `fh' `"  "d1_`key'": {"p": "' %24.17e (r(p)) `", "ci": ["'
    forvalues i = 1/`=rowsof(`CI')' {
        if `i' > 1 file write `fh' ", "
        file write `fh' "[" %24.17e (`CI'[`i', 1]) ", " %24.17e (`CI'[`i', 2]) "]"
    }
    file write `fh' "]}," _n
}
file write `fh' `"  "p_probe": ["'
local first 1
foreach e in -0.04459353919738075 0.7028089815536716 1.5975795577401435 2.050117547846048 {
    foreach d in -1e-6 1e-6 {
        local b = `e' + `d'
        quietly boottest d1 = `b', `BT'
        if !`first' file write `fh' ", "
        local first 0
        file write `fh' "[" %24.17e (`b') ", " %24.17e (r(p)) "]"
    }
}
file write `fh' "]" _n "}" _n
file close `fh'
