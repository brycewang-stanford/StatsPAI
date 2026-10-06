* Reference values for tests/reference_parity/test_spectral_parity.py
* Stata 18 (official commands only: pergram, wntestb).
* Run from this folder:
*     do _generate_spectral_Stata.do
* Reads spectral.csv; writes spectral_Stata.csv (the values pergram saves,
* one row per observation) and spectral_Stata_scalars.csv (wntestb).
clear all
set more off
import delimited using "spectral.csv", clear case(preserve) asdouble
tsset t
gen double x127 = x if t <= 127
gen double x101 = z if t <= 101
pergram x, generate(pg_x) nograph
pergram z, generate(pg_z) nograph
pergram x127, generate(pg_x127) nograph
tempname H
postfile `H' str16 series str8 stat double value using "_spectral_stata_tmp.dta", replace
foreach v in x z x127 x101 {
    wntestb `v', table
    post `H' ("`v'") ("stat") (r(stat))
    post `H' ("`v'") ("p") (r(p))
}
postclose `H'
keep t pg_x pg_z pg_x127
format pg_* %24.17g
export delimited using "spectral_Stata.csv", replace
use "_spectral_stata_tmp.dta", clear
format value %24.17g
export delimited using "spectral_Stata_scalars.csv", replace
erase "_spectral_stata_tmp.dta"
