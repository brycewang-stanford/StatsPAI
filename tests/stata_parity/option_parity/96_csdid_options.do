*! 96_csdid_options.do
*! Golden numbers for tests/reference_parity/test_csdid_options_stata_parity.py
*!
*! Standard errors, as well as ATT(g,t), for the csdid option values that
*! fixture 82 pins on point estimates only, and csdid's pairwise rule on an
*! unbalanced panel, which has no R did reference (did drops incomplete
*! units):
*!
*!   notyet long2 asinr method(reg)  -> notyet_cutoff='asinr'
*!   notyet long2 method(reg)        -> notyet_cutoff='cohort' (csdid default)
*!   long2 method(ipw)               -> estimator='ipw_abadie' (Abadie 2005)
*!   unbalanced panel, long2         -> balance='pair' (StatsPAI default),
*!                                      method(dripw) and notyet method(reg)
*!
*! Each block writes every column of e(b) with sqrt of the diagonal of e(V).
*! Requires: csdid (v1.81), drdid. Stata 18 MP.

version 17
clear all
set more off

* Double precision throughout (see README.md in this folder).
set type double

tempname fh
file open `fh' using "results/96_csdid_options_Stata.json", write replace
file write `fh' "{" _n

program define _dump_bse
    args fh label
    matrix b = e(b)
    matrix V = e(V)
    local names : colfullnames b
    local k = colsof(b)
    file write `fh' `"  ""' "`label'" `"": {"' _n
    forvalues j = 1/`k' {
        local nm : word `j' of `names'
        local v = b[1,`j']
        local s = sqrt(V[`j',`j'])
        local comma ","
        if `j' == `k' local comma ""
        file write `fh' `"    ""' "`nm'" `"": ["' %25.17e (`v') ", " %25.17e (`s') "]`comma'" _n
    }
    file write `fh' "  }," _n
end

import delimited "../../orig_parity/data/02_mpdta_original.csv", clear asdouble

qui csdid lemp, ivar(countyreal) time(year) gvar(first_treat) ///
    notyet long2 asinr method(reg)
_dump_bse `fh' "notyet_asinr_reg"

qui csdid lemp, ivar(countyreal) time(year) gvar(first_treat) ///
    notyet long2 method(reg)
_dump_bse `fh' "notyet_cohort_reg"

qui csdid lemp lpop, ivar(countyreal) time(year) gvar(first_treat) ///
    long2 method(ipw)
_dump_bse `fh' "ipw_abadie_lpop"

import delimited "data_96_csdid_unbalanced.csv", clear asdouble

qui csdid lemp lpop, ivar(countyreal) time(year) gvar(first_treat) ///
    long2 method(dripw)
_dump_bse `fh' "unbalanced_dripw_lpop"

qui csdid lemp, ivar(countyreal) time(year) gvar(first_treat) ///
    notyet long2 method(reg)
_dump_bse `fh' "unbalanced_notyet_reg"

file write `fh' `"  "_meta": {"csdid_version": "1.81", "stata": "18 MP", "precision": "double"}"' _n
file write `fh' "}" _n
file close `fh'

set type float
di as txt "wrote results/96_csdid_options_Stata.json"
