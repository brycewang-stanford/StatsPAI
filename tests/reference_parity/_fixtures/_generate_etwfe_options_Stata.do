* Stata jwdid reference for sp.etwfe options that R etwfe does not offer, on
* _fixtures/etwfe_options_panel.csv (the panel of _generate_etwfe_options_R.R).
*
* No ivar() anywhere: jwdid then fits cohort and period effects, the design
* of sp.etwfe's default (fe='cohort', R etwfe's ivar = NULL).
*
*   pois_* / logit_*   jwdid yc|yb, method(poisson|logit) cluster(id)
*                      [never]; every estat aggregation twice, with margins'
*                      default (conditional) delta-method SE and with
*                      vce(unconditional) -- the reference for
*                      sp.etwfe(family=..., response_se='unconditional')
*   ctrl_*             jwdid y, exovar(x1) cluster(id) [never]: the additive
*                      control of sp.etwfe(controls=[...])
*
* Generated with jwdid v2.2 (jwdid_estat v2.3, F. Rios-Avila) and reghdfe
* 6.13.1. Run from the repository root with Stata 18 and those commands on the
* adopath:
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_etwfe_options_Stata.do
version 18
set type double
clear all

import delimited "tests/reference_parity/_fixtures/etwfe_options_panel.csv", clear asdouble

tempname fh
file open `fh' using "tests/reference_parity/_fixtures/etwfe_options_Stata.json", ///
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
        file write `fh' %24.17e (`b'[1, `j'])
        if `j' < `k' file write `fh' ", "
    }
    file write `fh' "], " _n `"      "se": ["'
    forvalues j = 1/`k' {
        file write `fh' %24.17e (sqrt(`V'[`j', `j']))
        if `j' < `k' file write `fh' ", "
    }
    file write `fh' "]}"
    if "`last'" == "" file write `fh' ","
    file write `fh' _n
end

capture program drop _one
program define _one
    * one jwdid specification -> one JSON object
    args fh name vars opts estopts
    quietly jwdid `vars', tvar(year) gvar(g) cluster(id) `opts'
    file write `fh' `"  "`name'": {"' _n
    file write `fh' `"    "vars": "`vars'", "opts": "`opts'", "estat": "`estopts'","' _n
    file write `fh' `"    "cmd": "`e(cmd)'", "N": "' %12.0f (e(N))
    file write `fh' `", "N_clust": "' %12.0f (e(N_clust))
    file write `fh' `", "rank": "' %12.0f (e(rank)) "," _n
    foreach agg in simple group calendar {
        quietly estat `agg', `estopts'
        _wvec `fh' `agg'
    }
    quietly estat event, `estopts'
    _wvec `fh' event last
    file write `fh' "  }," _n
end

_one `fh' pois_notyet         "yc" "method(poisson)"       ""
_one `fh' pois_notyet_uncond  "yc" "method(poisson)"       "vce(unconditional)"
_one `fh' pois_never_uncond   "yc" "method(poisson) never" "vce(unconditional)"
_one `fh' logit_notyet        "yb" "method(logit)"         ""
_one `fh' logit_notyet_uncond "yb" "method(logit)"         "vce(unconditional)"
_one `fh' logit_never_uncond  "yb" "method(logit) never"   "vce(unconditional)"
_one `fh' ctrl_notyet         "y"  "exovar(x1)"            ""
_one `fh' ctrl_never          "y"  "exovar(x1) never"      ""

local ver : di "`c(stata_version)'"
file write `fh' `"  "versions": "Stata `ver'; jwdid v2.2, jwdid_estat v2.3; reghdfe 6.13.1""' _n
file write `fh' "}" _n
file close `fh'
