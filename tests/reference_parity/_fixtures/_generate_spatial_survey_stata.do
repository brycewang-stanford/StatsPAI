* Frozen Stata reference for sp.spatial_panel (spatial_survey parity family).
*
* xsmle (SSC; Belotti, Hughes & Piano Mortari) fixed-effects ML on the
* Produc / usaww CSVs written by _prepare_spatial_survey_data.R, with the
* maximiser tolerances tightened (nrtolerance / tolerance / ltolerance
* 1e-14) so the comparison is not limited by -ml-'s default stopping rule.
* Standard errors are xsmle's default vce(oim).
*
* xsmle is installed into a PRIVATE ado tree next to this file
* (_ado_spatial_survey/plus, not committed) -- never the user's PLUS.
*
* Run from the repository root:
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_spatial_survey_stata.do
* Writes tests/reference_parity/_fixtures/spatial_survey_stata.json
version 18
clear all
set type double
local FX "tests/reference_parity/_fixtures"
sysdir set PLUS "`c(pwd)'/`FX'/_ado_spatial_survey/plus"
capture which xsmle
if _rc {
    ssc install xsmle, replace
}
* the Mata library lxsmle.mlib lives in the private PLUS: re-index
mata: mata mlib index

import delimited using "`FX'/spatial_survey_usaww.csv", clear varnames(1)
drop v1
mkmat _all, matrix(W)
import delimited using "`FX'/spatial_survey_produc.csv", clear varnames(1)
* encode assigns codes in alphabetical order, the row order of usaww
encode state, gen(id)
xtset id year

local tol "nrtolerance(1e-14) tolerance(1e-14) ltolerance(1e-14)"
tempname fh
file open `fh' using "`FX'/spatial_survey_stata.json", write replace
file write `fh' "{"
local first 1
foreach ty in ind both {
    foreach m in sar sem sdm {
        if "`m'" == "sem" local wopt "emat(W)"
        else local wopt "wmat(W)"
        xsmle lgsp lpcap lpc lemp unemp, `wopt' model(`m') fe type(`ty') nolog `tol'
        matrix b = e(b)
        matrix V = e(V)
        local names : colfullnames b
        local k = colsof(b)
        if !`first' file write `fh' ","
        local first 0
        file write `fh' `""`m'_`ty'": {"converged": "' (e(converged)) `", "ll": "' %25.17e (e(ll)) `", "coef": {"'
        forvalues j = 1/`k' {
            local nm : word `j' of `names'
            if `j' > 1 file write `fh' ","
            file write `fh' `"""' "`nm'" `"": "' %25.17e (b[1, `j'])
        }
        file write `fh' `"}, "se": {"'
        forvalues j = 1/`k' {
            local nm : word `j' of `names'
            if `j' > 1 file write `fh' ","
            file write `fh' `"""' "`nm'" `"": "' %25.17e (sqrt(V[`j', `j']))
        }
        file write `fh' "}}"
    }
}
file write `fh' `", "versions": {"Stata": ""' "`c(stata_version)'" `"", "xsmle": ""'
* xsmle's own version line, first *! line of the ado
tempname ah
file open `ah' using "`c(sysdir_plus)'x/xsmle.ado", read text
file read `ah' line
file close `ah'
local line : subinstr local line `"""' "", all
local line : subinstr local line "*! " "", all
local line : subinstr local line "* " "", all
file write `fh' `"`line'"' `""}}"'
file close `fh'
