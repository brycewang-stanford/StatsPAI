* Stata leebounds (Tauchmann, SSC, v1.5) reference for the missing-value
* sample rules of sp.lee_bounds(trimming='leebounds').
*   python tests/reference_parity/_fixtures/_generate_leebounds_missing_data.py
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_leebounds_missing_stata.do
* asdouble: the outcome is read as stored (the float32 artefact is covered by
* _generate_leebounds_stata.do).
version 18
clear all
import delimited using "tests/reference_parity/_fixtures/leebounds_missing.csv", clear asdouble
tempname fh
file open `fh' using "tests/reference_parity/_fixtures/leebounds_missing_stata.json", write replace
file write `fh' "{" _n
local first = 1
forvalues k = 1/3 {
    foreach case in ymiss dmiss smiss xdrop nosel cie90 tight90 {
        preserve
        keep if sample == `k'
        if "`case'" == "ymiss" qui leebounds y2 d, select(s)
        if "`case'" == "dmiss" qui leebounds y d2, select(s)
        if "`case'" == "smiss" qui leebounds y d, select(s3)
        if "`case'" == "nosel" qui leebounds y d, cieffect
        if "`case'" == "cie90" qui leebounds y d, select(s) cieffect level(90)
        if "`case'" == "tight90" qui leebounds y d, select(s) tight(x) cieffect level(90)
        if "`case'" == "xdrop" {
            keep if !missing(xm)
            qui leebounds y d, select(s) tight(xm)
        }
        matrix b = e(b)
        matrix V = e(V)
        if `first' == 0 file write `fh' "," _n
        local first = 0
        file write `fh' `"  "`k'_`case'": {"N": "' %12.0f (e(N)) `", "lower": "' %25.17f (b[1,1]) `", "upper": "' %25.17f (b[1,2])
        file write `fh' `", "se_lower": "' %25.17f (sqrt(V[1,1])) `", "se_upper": "' %25.17f (sqrt(V[2,2]))
        if inlist("`case'", "nosel", "cie90", "tight90") {
            file write `fh' `", "cilower": "' %25.17f (e(cilower)) `", "ciupper": "' %25.17f (e(ciupper))
        }
        file write `fh' "}"
        restore
    }
}
file write `fh' _n "}" _n
file close `fh'
