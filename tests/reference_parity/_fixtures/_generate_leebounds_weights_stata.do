* Stata leebounds (Tauchmann, SSC v1.5) with probability weights, reference
* for sp.lee_bounds(weights=, trimming='leebounds', se_method='analytic').
*   python tests/reference_parity/_fixtures/_generate_leebounds_weights_data.py
*   cd tests/reference_parity/_fixtures && stata-mp -b do _generate_leebounds_weights_stata.do
* leebounds accepts pweight, fweight and iweight (not aweight).
version 18
clear all
import delimited using "leebounds_weights.csv", clear asdouble
tempname fh
file open `fh' using "leebounds_weights_stata.json", write replace
file write `fh' "{" _n
local first = 1
forvalues k = 1/4 {
    foreach case in pw cie {
        preserve
        keep if sample == `k'
        if "`case'" == "pw" qui leebounds y d [pw=w], select(s)
        if "`case'" == "cie" qui leebounds y d [pw=w], select(s) cieffect
        matrix b = e(b)
        matrix V = e(V)
        if `first' == 0 file write `fh' "," _n
        local first = 0
        file write `fh' `"  "`k'_`case'": {"N": "' %12.0f (e(N)) `", "lower": "' %25.17f (b[1,1]) `", "upper": "' %25.17f (b[1,2])
        file write `fh' `", "se_lower": "' %25.17f (sqrt(V[1,1])) `", "se_upper": "' %25.17f (sqrt(V[2,2])) `", "trim": "' %25.17f (e(trim))
        if "`case'" == "cie" file write `fh' `", "cilower": "' %25.17f (e(cilower)) `", "ciupper": "' %25.17f (e(ciupper))
        file write `fh' "}"
        restore
    }
}
file write `fh' _n "}" _n
file close `fh'
