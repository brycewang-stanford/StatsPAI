* ---------------------------------------------------------------------------
* Stata reference for tests/reference_parity/test_psm_ties_se_parity.py:
* the Abadie-Imbens (2006) standard error of psmatch2 , ai(1) with and
* without `ties`, on MatchIt::lalonde with the propensity score sp.psm fits.
*
* psm_ties_lalonde.csv holds re78, treat and that score (%.17g). From this
* directory:
*   stata-mp -b do _generate_psm_ties_stata.do
* Writes psm_ties_Stata.json (%23.16e).
*
* psmatch2 (Leuven & Sianesi) goes into a PRIVATE ado tree next to this
* file (_ado_psm_ties/, git-ignored by the _ado_*/ rule) -- never PLUS.
*
* The data are fitted in their own order and in three random orders. With
* `ties` the estimate and the standard error are the same in every order;
* without it the matches and the same-arm neighbours are taken by row
* order. The per-unit same-arm mean _self_re78 under `ties` is written in
* the file's row order.
* ---------------------------------------------------------------------------
version 18
set more off
set type double
local A "`c(pwd)'/_ado_psm_ties"
capture mkdir "`A'"
sysdir set PLUS "`A'"
adopath ++ "`A'"
capture which psmatch2
if _rc ssc install psmatch2

import delimited using "psm_ties_lalonde.csv", clear asdouble
capture rename v3 ps
capture rename _pscore ps
gen long rowid = _n
tempfile base
save `base'

file open fh using "psm_ties_Stata.json", write replace text
file write fh "{" _n
foreach ord in 0 1 2 3 {
    use `base', clear
    if `ord' > 0 {
        set seed `ord'
        gen double r = runiform()
        sort r
    }
    quietly psmatch2 treat, outcome(re78) pscore(ps) neighbor(1) ai(1) ties
    file write fh `"  "ties_order`ord'": {"est": "' %23.16e (r(att)) `", "se": "' %23.16e (r(seatt)) `"},"' _n
    quietly psmatch2 treat, outcome(re78) pscore(ps) neighbor(1) ai(1)
    file write fh `"  "noties_order`ord'": {"est": "' %23.16e (r(att)) `", "se": "' %23.16e (r(seatt)) `"},"' _n
}
use `base', clear
quietly psmatch2 treat, outcome(re78) pscore(ps) neighbor(1) ai(1) ties
sort rowid
file write fh `"  "ties_self_re78": ["'
forvalues i = 1/`=_N' {
    local sep = cond(`i' < _N, ", ", "")
    file write fh %23.16e (_self_re78[`i']) "`sep'"
}
file write fh "]" _n "}" _n
file close fh
