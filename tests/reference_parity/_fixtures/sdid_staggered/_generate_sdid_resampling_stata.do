* Stochastic-screen numbers for tests/reference_parity/test_sdid_resampling_screen.py
*
* Stata sdid 2.0.2 (Pailanir & Clarke, SSC) placebo and bootstrap standard
* errors at five seeds, 200 replications each, on the two datasets of
* _generate_sdid_stata.do (author-distributed with Clarke, Pailanir, Athey &
* Imbens 2024):
*   prop99_example.dta -- block design, one treated unit: vce(placebo) for
*                         method(sdid / did / sc); the bootstrap is refused
*   quota_example.dta  -- staggered adoption: vce(placebo), vce(bootstrap)
* These are Monte Carlo draws from Stata's stream. sp.sdid(treat=...) cannot
* reproduce them; the test only screens (S) its seed-averaged SE against
* Stata's. sdid.ado stores e(se) through strofreal() (about seven
* significant digits), far below the Monte Carlo error.
* Run from this directory: stata-mp -q -b do _generate_sdid_resampling_stata.do
* Numbers are written %24.17e; do not hand-edit sdid_resampling_stata.json.
version 18
clear all
set more off
set type double
tempname fh
file open `fh' using "sdid_resampling_stata.json", write replace
file write `fh' "{" _n

use "prop99_example.dta", clear
foreach m in sdid did sc {
    forvalues s = 1/5 {
        qui sdid packspercapita state year treated, vce(placebo) reps(200) seed(`s') method(`m')
        file write `fh' `"  "prop99_`m'_placebo_`s'": "' %24.17e (e(se)) "," _n
    }
}

use "quota_example.dta", clear
foreach v in placebo bootstrap {
    forvalues s = 1/5 {
        qui sdid womparl country year quota, vce(`v') reps(200) seed(`s')
        file write `fh' `"  "quota_sdid_`v'_`s'": "' %24.17e (e(se)) "," _n
    }
}

file write `fh' `"  "_reference": "Stata 18 MP, sdid 2.0.2, reps(200), seeds 1-5""' _n
file write `fh' "}" _n
file close `fh'
set type float
