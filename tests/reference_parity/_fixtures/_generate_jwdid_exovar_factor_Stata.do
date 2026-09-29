* Reference for sp.jwdid(..., exovar=<Stata factor terms>) -- the factor-
* variable expansion of statspai.did._factor_terms -- against Stata
* jwdid ..., method(ppmlhdfe) exovar(...) on the bytes of
* _fixtures/etwfe_poisson_jwdid_hettype.csv (see
* _generate_etwfe_poisson_jwdid_Stata.do for the panel).
*
* Records N and estat simple on both scales for three exovar() terms:
* a factor-by-factor interaction, the same under never, and a
* continuous-by-factor interaction.
*
* Generated with jwdid v2.2 and ppmlhdfe 2.3.3 (02nov2025), Stata 18 MP.
* Run from the repository root with Stata 18:
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_jwdid_exovar_factor_Stata.do
version 18
set type double
clear all

import delimited "tests/reference_parity/_fixtures/etwfe_poisson_jwdid_hettype.csv", clear asdouble

tempname fh
file open `fh' using "tests/reference_parity/_fixtures/jwdid_exovar_factor_Stata.json", ///
    write replace text
file write `fh' "{" _n

capture program drop _one
program define _one
    args fh name opts last
    quietly jwdid y, ivar(id) tvar(year) gvar(g) method(ppmlhdfe) `opts'
    local N = e(N)
    quietly estat simple, predict(xb)
    tempname b V
    matrix `b' = r(b)
    matrix `V' = r(V)
    local bl = `b'[1, 1]
    local sl = sqrt(`V'[1, 1])
    quietly estat simple
    matrix `b' = r(b)
    local br = `b'[1, 1]
    file write `fh' `"  "`name'": {"opts": "`opts'", "N": "' %12.0f (`N') ", " _n
    file write `fh' `"    "simple_link": {"b": "' %24.16e (`bl') `", "se": "' %24.16e (`sl') "}," _n
    file write `fh' `"    "simple_response": {"b": "' %24.16e (`br') "}}"
    if "`last'" == "" file write `fh' ","
    file write `fh' _n
end

_one `fh' year_xcat       "exovar(i.year#i.xcat)"
_one `fh' year_xcat_never "never exovar(i.year#i.xcat)"
_one `fh' xc_xcat         "exovar(c.xc#i.xcat)" last
file write `fh' "}" _n
file close `fh'
