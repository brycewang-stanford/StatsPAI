* Reference for sp.etwfe(family='poisson', fe='unit') -- hettype= and xvar=
* -- against Stata jwdid ..., method(ppmlhdfe) on
* _fixtures/etwfe_poisson_jwdid_hettype.csv (2,880 rows, 360 units, 8
* years, cohorts 2004/2006 + never treated, ~13% all-zero units, a
* time-invariant categorical covariate xcat and a time-varying continuous
* covariate xc).
*
* For every specification the fixture records estat simple on both scales
* (predict(xb) and the default response scale), estat event, predict(xb),
* and, with the categorical covariate, estat simple, predict(xb) over(xcat).
*
* Generated with jwdid v2.201 (jwdid_estat v2.2, F. Rios-Avila) and ppmlhdfe
* 2.3.0 (25feb2021). Run from the repository root with Stata 18 and those
* commands on the adopath:
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_etwfe_poisson_jwdid_Stata.do
version 18
set type double
clear all

import delimited "tests/reference_parity/_fixtures/etwfe_poisson_jwdid_hettype.csv", clear asdouble

tempname fh
file open `fh' using "tests/reference_parity/_fixtures/etwfe_poisson_jwdid_Stata.json", ///
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
    quietly jwdid `vars', ivar(id) tvar(year) gvar(g) method(ppmlhdfe) `opts'
    file write `fh' `"  "`name'": {"' _n
    file write `fh' `"    "vars": "`vars'", "opts": "`opts'","' _n
    file write `fh' `"    "N": "' %12.0f (e(N)) "," _n
    quietly estat simple, predict(xb)
    _wvec `fh' simple_link
    quietly estat simple
    _wvec `fh' simple_response
    if strpos("`vars'", "i.xcat") > 0 {
        quietly estat event, predict(xb)
        _wvec `fh' event_link
        quietly estat simple, predict(xb) over(xcat)
        _wvec `fh' over_link last
    }
    else {
        quietly estat event, predict(xb)
        _wvec `fh' event_link last
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
file write `fh' `"  "versions": "Stata `ver'; jwdid v2.201, jwdid_estat v2.2; ppmlhdfe 2.3.0""' _n
file write `fh' "}" _n
file close `fh'
