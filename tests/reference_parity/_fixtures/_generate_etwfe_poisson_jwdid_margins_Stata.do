* Reference for sp.etwfe(family='poisson', fe='unit', response_se='margins')
* -- the response-scale (count) aggregates of Stata jwdid ..., method(ppmlhdfe)
* + estat, whose standard errors come from margins after ppmlhdfe -- on
* _fixtures/etwfe_poisson_jwdid_hettype.csv (the panel of
* _generate_etwfe_poisson_jwdid_Stata.do).
*
* Three fits: the default design, the categorical covariate i.xcat (where
* margins' gradient depends on jwdid's parametrisation of the covariate
* terms), and cluster(cl) with cl = mod(id + year, 7): clusters that nest
* neither the units nor the periods, so the covariance of ppmlhdfe's _cons
* with the slopes is not zero (y - mu sums to zero within every unit and,
* the periods being regressors, within every period).  Each records
* estat simple / event / group / calendar on the response scale, the link
* scale simple aggregate, and for i.xcat estat simple, over(xcat).
*
* Generated with jwdid v2.201 (jwdid_estat v2.2, F. Rios-Avila) and ppmlhdfe
* 2.3.0 (25feb2021). Run from the repository root with Stata 18:
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_etwfe_poisson_jwdid_margins_Stata.do
version 18
set type double
clear all

import delimited "tests/reference_parity/_fixtures/etwfe_poisson_jwdid_hettype.csv", clear asdouble
generate double cl = mod(id + year, 7)

tempname fh
file open `fh' using "tests/reference_parity/_fixtures/etwfe_poisson_jwdid_margins_Stata.json", ///
    write replace text
file write `fh' "{" _n

capture program drop _wvec
program define _wvec
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
    args fh name vars opts
    quietly jwdid `vars', ivar(id) tvar(year) gvar(g) method(ppmlhdfe) `opts'
    file write `fh' `"  "`name'": {"' _n
    file write `fh' `"    "vars": "`vars'", "opts": "`opts'","' _n
    file write `fh' `"    "N": "' %12.0f (e(N)) "," _n
    quietly estat simple, predict(xb)
    _wvec `fh' simple_link
    quietly estat simple
    _wvec `fh' simple_response
    quietly estat event
    _wvec `fh' event_response
    quietly estat group
    _wvec `fh' group_response
    quietly estat calendar
    if strpos("`vars'", "i.xcat") > 0 {
        _wvec `fh' calendar_response
        quietly estat simple, over(xcat)
        _wvec `fh' over_response last
    }
    else {
        _wvec `fh' calendar_response last
    }
    file write `fh' "  }," _n
end

_one `fh' default "y" ""
_one `fh' xcat "y i.xcat" ""
_one `fh' cluster_mixed "y" "cluster(cl)"

file write `fh' `"  "versions": {"stata": "`c(stata_version)'", "' ///
    `""jwdid": "2.201", "jwdid_estat": "2.2", "ppmlhdfe": "2.3.0"}"' _n
file write `fh' "}" _n
file close `fh'
