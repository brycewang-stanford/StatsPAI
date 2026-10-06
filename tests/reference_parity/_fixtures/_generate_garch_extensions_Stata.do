* Stata 18 reference for sp.garch(ar=, dist=) and higher-order GARCH.
* Run from tests/reference_parity/_fixtures; writes garch_extensions_Stata.csv.
clear all
import delimited "garch_extensions.csv", clear
tsset t
tempname h
postfile `h' str20 model str20 name double value using garch_extensions_Stata, replace
local opts "nolog tolerance(1e-12) ltolerance(1e-14) nrtolerance(1e-12)"
foreach m in g21 g21oim ar1 ar1t {
    if "`m'" == "g21"  arch r, arch(1) garch(1/2) `opts'
    if "`m'" == "g21oim"  arch r, arch(1) garch(1/2) vce(oim) `opts'
    if "`m'" == "ar1"  arch r, ar(1) arch(1) garch(1) `opts'
    if "`m'" == "ar1t" arch r, ar(1) arch(1) garch(1/2) distribution(t) `opts'
    post `h' ("`m'") ("ll") (e(ll))
    matrix b = e(b)
    matrix V = e(V)
    local k = colsof(b)
    forvalues j = 1/`k' {
        post `h' ("`m'") ("b`j'") (b[1,`j'])
        post `h' ("`m'") ("se`j'") (sqrt(V[`j',`j']))
    }
    if "`m'" == "ar1t" post `h' ("`m'") ("df") (e(tdf))
}
postclose `h'
use garch_extensions_Stata, clear
format value %24.16e
export delimited using "garch_extensions_Stata.csv", replace
erase garch_extensions_Stata.dta
