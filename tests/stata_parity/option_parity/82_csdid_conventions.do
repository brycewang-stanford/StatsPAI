*! 82_csdid_conventions.do
*! Golden numbers for tests/reference_parity/test_csdid_conventions_stata_parity.py
*!
*! Pins the three csdid option axes that StatsPAI's callaway_santanna()
*! mirrors, and that are easy to get silently wrong because the naming
*! does NOT line up across the two ecosystems:
*!
*!   asinr            -> notyet_cutoff='period'   (StatsPAI default, = R did)
*!   (csdid default)  -> notyet_cutoff='cohort'
*!   method(stdipw)   -> estimator='ipw' / 'stdipw'  (StatsPAI default naming)
*!   method(ipw)      -> estimator='ipw_abadie'      (Abadie 2005)
*!
*! long2 is used throughout because it is csdid's universal-base-period
*! scheme, which is StatsPAI's base_period='universal' default.
*!
*! Requires: csdid (v1.81), drdid. Stata 18 MP.

version 17
clear all
set more off

local here "`c(pwd)'"
local data "../../orig_parity/data/02_mpdta_original.csv"
import delimited "`data'", clear

tempname fh
file open `fh' using "results/82_csdid_conventions_Stata.json", write replace
file write `fh' "{" _n

* ------------------------------------------------------------------
* Helper: dump the first 12 columns of e(b) (the ATT(g,t) cells; the
* remaining w* columns are aggregation weights, not estimates).
* ------------------------------------------------------------------
program define _dump_atts
    args fh label
    matrix b = e(b)
    local names : colnames b
    file write `fh' `"  ""' "`label'" `"": {"' _n
    forvalues j = 1/12 {
        local nm : word `j' of `names'
        local v = b[1,`j']
        local comma ","
        if `j' == 12 local comma ""
        file write `fh' `"    ""' "`nm'_`j'" `"": "' %20.15f (`v') "`comma'" _n
    }
    file write `fh' "  }," _n
end

* --- 1. asinr: R / StatsPAI-default convention -----------------------
qui csdid lemp, ivar(countyreal) time(year) gvar(first_treat) ///
    notyet long2 asinr method(reg)
_dump_atts `fh' "notyet_asinr_reg"

* --- 2. csdid's own default convention -------------------------------
qui csdid lemp, ivar(countyreal) time(year) gvar(first_treat) ///
    notyet long2 method(reg)
_dump_atts `fh' "notyet_csdid_default_reg"

* --- 3. stabilized IPW (= StatsPAI 'ipw'/'stdipw') -------------------
qui csdid lemp lpop, ivar(countyreal) time(year) gvar(first_treat) ///
    long2 method(stdipw)
_dump_atts `fh' "stdipw_lpop"

* --- 4. Abadie (2005) IPW (= StatsPAI 'ipw_abadie') ------------------
qui csdid lemp lpop, ivar(countyreal) time(year) gvar(first_treat) ///
    long2 method(ipw)
_dump_atts `fh' "ipw_abadie_lpop"

file write `fh' `"  "_meta": {"csdid_version": "1.81", "stata": "18 MP"}"' _n
file write `fh' "}" _n
file close `fh'

di as txt "wrote results/82_csdid_conventions_Stata.json"
