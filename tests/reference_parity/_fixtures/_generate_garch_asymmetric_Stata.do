* Stata 18 reference for sp.garch(model="gjr" | "egarch").
* Run from tests/reference_parity/_fixtures; writes garch_asymmetric_Stata.csv.
* Coefficients are posted in Stata's order and by Stata's names (arch,
* tarch, garch; earch, earch_a, egarch); the test maps them.
clear all
import delimited "garch_asymmetric.csv", clear
tsset t
tempname h
postfile `h' str12 model str12 name double value using garch_asymmetric_Stata, replace
local opts "nolog tolerance(1e-10) ltolerance(1e-12) nrtolerance(1e-9)"
local s_gjr    "arch(1) tarch(1) garch(1)"
local s_gjrt   "arch(1) tarch(1) garch(1) distribution(t)"
local s_gjrar  "ar(1) arch(1) tarch(1) garch(1)"
local s_eg     "earch(1) egarch(1)"
local s_egt    "earch(1) egarch(1) distribution(t)"
local s_eg2    "earch(1/2) egarch(1)"
foreach m in gjr gjrt gjrar eg egt eg2 {
    arch r, `s_`m'' `opts'
    post `h' ("`m'") ("ll") (e(ll))
    matrix b = e(b)
    matrix V = e(V)
    local k = colsof(b)
    forvalues j = 1/`k' {
        post `h' ("`m'") ("b`j'") (b[1,`j'])
        post `h' ("`m'") ("se`j'") (sqrt(V[`j',`j']))
    }
}
postclose `h'
use garch_asymmetric_Stata, clear
format value %24.16e
export delimited using "garch_asymmetric_Stata.csv", replace
erase garch_asymmetric_Stata.dta
