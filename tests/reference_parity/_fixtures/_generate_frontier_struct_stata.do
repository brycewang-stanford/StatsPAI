* ---------------------------------------------------------------------------
* Stata reference for tests/reference_parity/test_frontier_struct_R_parity.py
* (markup section). Requires Stata 18 and SSC markupest + prodest installed
* into the PRIVATE ado dir _ado_spatial_survey/ (never PLUS); the lines below
* do that.
*
* Run from this directory:
*   stata-mp -b do _generate_frontier_struct_stata.do
* Reads prodest_panel.csv (the prodest parity panel) and writes
* frontier_struct_stata.json (numbers as %23.16e).
*
* Conventions pinned here
* -----------------------
* * markupest method(dlw) pmethod(lp) valueadded, free(l) state(k) proxy(m),
*   prodest poly(3): the markup of the FREE input l,
*       mu = _b[l] / ( exp(l) / exp(y) * exp(fsres) )     (corrected)
*       mu = _b[l] / ( exp(l) / exp(y) )                  (uncorrected)
*   where fsres = y - (stage-1 fitted value). y is treated as log revenue and
*   l as log expenditure on the input -- the columns are passed identically
*   to sp.markup(revenue="y", input_cost="l", flexible_input="l").
* * _b[l] is prodest's stage-1 OLS coefficient (deterministic); the stage-2
*   state coefficient (optimiser-dependent) does not enter the markup of l.
* * set type double: prodest builds polynomial terms with generate.
* ---------------------------------------------------------------------------
version 18
set more off
set type double
local ado "`c(pwd)'/_ado_spatial_survey/"
sysdir set PLUS "`ado'"
adopath ++ "`ado'"
capture which markupest
if _rc ssc install markupest, replace
capture which prodest
if _rc ssc install prodest, replace
set seed 20260918

tempname fh
file open `fh' using "frontier_struct_stata.json", write replace text
file write `fh' "{" _n

foreach corr in corrected uncorrected {
    import delimited using "prodest_panel.csv", clear asdouble
    xtset id year
    local copt = cond("`corr'" == "corrected", "corrected", "")
    markupest mkup, method(dlw) pmethod(lp) output(y) free(l) state(k) ///
        proxy(m) valueadded inputvar(l) `copt' savebeta ///
        prodestoptions("poly(3) reps(3)") id(id) t(year)
    quietly count if !missing(mkup)
    local nm = r(N)
    file write `fh' `"  "`corr'": {"b_l": "' %23.16e (_bl[1]) `", "n": "' %12.0f (`nm') `", "rows": ["'
    local first = 1
    forvalues i = 1/`=_N' {
        if !missing(mkup[`i']) {
            if !`first' file write `fh' ", "
            file write `fh' "[" %12.0f (id[`i']) ", " %12.0f (year[`i']) ", " %23.16e (mkup[`i']) "]"
            local first = 0
        }
    }
    file write `fh' "]}," _n
}
file write `fh' `"  "versions": {"stata": "`c(stata_version)'", "markupest": "1.0.1 10May2020"}"' _n
file write `fh' "}" _n
file close `fh'
