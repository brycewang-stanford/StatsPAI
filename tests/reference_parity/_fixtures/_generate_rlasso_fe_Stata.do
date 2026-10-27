* Reference numbers for tests/reference_parity/test_rlasso_fe_stata_parity.py.
* Needs lassopack, pdslasso and ivreg2 (ssc install lassopack / pdslasso /
* ivreg2). Run from tests/reference_parity/_fixtures/ ; writes
* rlasso_fe_Stata.json.
version 18
clear all
set type double
import delimited "rlasso_fe_panel.csv", clear asdouble
xtset id year

tempname fh
file open `fh' using "rlasso_fe_Stata.json", write replace
file write `fh' "{" _n
file write `fh' `" "_provenance": "Stata/MP `c(stata_version)', lassopack rlasso + pdslasso (ivlasso), run `c(current_date)' by _generate_rlasso_fe_Stata.do on rlasso_fe_panel.csv","' _n

* JSON helpers: numbers with a leading zero, string lists, matrix rows.
capture program drop _jfmt
program define _jfmt, rclass
    args val
    local s = strtrim(string(`val', "%21.17g"))
    if substr("`s'", 1, 1) == "." local s "0`s'"
    if substr("`s'", 1, 2) == "-." local s = "-0" + substr("`s'", 2, .)
    return local s "`s'"
end
capture program drop _jnum
program define _jnum
    args fh key val
    _jfmt `val'
    file write `fh' `"  "`key'": `r(s)',"' _n
end
capture program drop _jlist
program define _jlist
    args fh key list
    local out
    local sep
    foreach w of local list {
        local out `"`out'`sep'"`w'""'
        local sep ", "
    }
    file write `fh' `"  "`key'": [`out'],"' _n
end
capture program drop _jrow
program define _jrow
    args fh key mat
    local k = colsof(`mat')
    local out
    local sep
    forvalues j = 1/`k' {
        _jfmt `mat'[1,`j']
        local out "`out'`sep'`r(s)'"
        local sep ", "
    }
    file write `fh' `"  "`key'": [`out'],"' _n
end
capture program drop _jel
program define _jel
    * first element of an e() matrix
    args fh key ematrix
    tempname m
    mat `m' = e(`ematrix')
    _jnum `fh' `key' `m'[1,1]
end

* (1) rlasso with fixed effects: heteroskedastic and cluster loadings
foreach v in d y {
    rlasso `v' x1-x30, fe robust
    _jnum `fh' rlasso_`v'_robust_lambda0 e(lambda0)
    _jnum `fh' rlasso_`v'_robust_npsiiter e(npsiiter)
    _jlist `fh' rlasso_`v'_robust_selected "`e(selected)'"
    tempname b
    mat `b' = e(betaOLS)
    _jrow `fh' rlasso_`v'_robust_betaOLS `b'
    rlasso `v' x1-x30, fe cluster(id)
    _jnum `fh' rlasso_`v'_cluster_lambda0 e(lambda0)
    _jlist `fh' rlasso_`v'_cluster_selected "`e(selected)'"
    mat `b' = e(betaOLS)
    _jrow `fh' rlasso_`v'_cluster_betaOLS `b'
}

* (2) pdslasso with fixed effects
foreach opt in robust cluster loptions {
    if "`opt'" == "robust"   local o "robust"
    if "`opt'" == "cluster"  local o "cluster(id)"
    * heteroskedastic penalty, homoskedastic final variance
    if "`opt'" == "loptions" local o "loptions(robust)"
    pdslasso y d (x1-x30), fe `o'
    _jel `fh' pds_`opt'_beta beta_pds
    _jel `fh' pds_`opt'_V V_pds
    _jel `fh' plasso_`opt'_beta beta_plasso
    _jel `fh' plasso_`opt'_V V_plasso
    _jlist `fh' pds_`opt'_xselected "`e(xselected)'"
}

* (3) two-way: unit FE plus year dummies partialled out
foreach opt in robust cluster {
    if "`opt'" == "robust"  local o "robust"
    if "`opt'" == "cluster" local o "cluster(id)"
    pdslasso y d (i.year x1-x30), fe partial(i.year) `o'
    _jel `fh' tw_pds_`opt'_beta beta_pds
    _jel `fh' tw_pds_`opt'_V V_pds
    _jel `fh' tw_plasso_`opt'_beta beta_plasso
    _jel `fh' tw_plasso_`opt'_V V_plasso
    _jlist `fh' tw_pds_`opt'_xselected "`e(xselected)'"
}
rlasso d i.year x1-x30, fe partial(i.year) robust
_jlist `fh' tw_rlasso_d_robust_selected "`e(selected)'"
tempname b
mat `b' = e(betaOLS)
_jrow `fh' tw_rlasso_d_robust_betaOLS `b'

file write `fh' `"  "N": `=_N'"' _n "}" _n
file close `fh'
