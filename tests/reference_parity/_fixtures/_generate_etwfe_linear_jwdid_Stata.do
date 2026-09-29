* Reference for the linear sp.etwfe(hettype=...) -- Stata jwdid without
* method(), i.e. reghdfe absorbing ivar and tvar -- on
* _fixtures/etwfe_linear_jwdid_hettype.csv (360 units x 8 years, cohorts
* 2004/2006 + never treated, 80 missing outcomes, a categorical xcat and a
* time-varying continuous xc).
*
* For every specification the fixture records estat simple, estat event
* and, with the categorical covariate, estat simple, over(xcat).
*
* Generated with jwdid v2.201 (jwdid_estat v2.2, F. Rios-Avila) and the
* reghdfe 6.12.3 (08aug2023). Run from the repository root with Stata 18 and those
* commands on the adopath:
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_etwfe_linear_jwdid_Stata.do
version 18
set type double
clear all

import delimited "tests/reference_parity/_fixtures/etwfe_linear_jwdid_hettype.csv", clear asdouble

tempname fh
file open `fh' using "tests/reference_parity/_fixtures/etwfe_linear_jwdid_Stata.json", ///
    write replace text
file write `fh' "{" _n

capture program drop _wvec
program define _wvec
    * write "name": {"labels": [...], "b": [...], "se": [...]} from r(b) r(V)
    args fh name last
    tempname b V
    matrix `b' = r(b)
    matrix `V' = r(V)
    local k = colsof(`b')
    local labs : colnames `b'
    file write `fh' `"    "`name'": {"labels": ["'
    local j = 0
    foreach l of local labs {
        local ++j
        file write `fh' `""`l'""'
        if `j' < `k' file write `fh' ", "
    }
    file write `fh' "], " _n `"      "b": ["'
    forvalues j = 1/`k' {
        file write `fh' %24.16e (`b'[1, `j'])
        if `j' < `k' file write `fh' ", "
    }
    file write `fh' "], " _n `"      "se": ["'
    forvalues j = 1/`k' {
        file write `fh' %24.16e (sqrt(`V'[`j', `j']))
        if `j' < `k' file write `fh' ", "
    }
    file write `fh' "]}"
    if "`last'" == "" file write `fh' ","
    file write `fh' _n
end

capture program drop _one
program define _one
    * one jwdid specification -> one JSON object
    args fh name vars opts
    quietly jwdid `vars', ivar(id) tvar(year) gvar(g) `opts'
    file write `fh' `"  "`name'": {"' _n
    file write `fh' `"    "vars": "`vars'", "opts": "`opts'","' _n
    file write `fh' `"    "N": "' %12.0f (e(N))
    file write `fh' `", "N_clust": "' %12.0f (e(N_clust))
    file write `fh' `", "df_m": "' %12.0f (e(df_m)) "," _n
    quietly estat simple
    _wvec `fh' simple
    if strpos("`vars'", "i.xcat") > 0 {
        quietly estat event
        _wvec `fh' event
        quietly estat simple, over(xcat)
        _wvec `fh' over last
    }
    else {
        quietly estat event
        _wvec `fh' event last
    }
    file write `fh' "  }," _n
end

_one `fh' default      "y"             ""
_one `fh' event        "y"             "hettype(event)"
_one `fh' cohort       "y"             "hettype(cohort)"
_one `fh' time         "y"             "hettype(time)"
_one `fh' twfe         "y"             "hettype(twfe)"
_one `fh' never        "y"             "never"
_one `fh' never_cohort "y"             "never hettype(cohort)"
_one `fh' never_event  "y"             "never hettype(event)"
_one `fh' xcat         "y i.xcat"      ""
_one `fh' xc           "y c.xc"        ""
_one `fh' xcat_event   "y i.xcat"      "hettype(event)"
_one `fh' xcat_never   "y i.xcat"      "never"
_one `fh' xcat_xc      "y i.xcat c.xc" ""

local ver : di "`c(stata_version)'"
file write `fh' `"  "versions": "Stata `ver'; jwdid v2.201, jwdid_estat v2.2; reghdfe 6.12.3""' _n
file write `fh' "}" _n
file close `fh'
