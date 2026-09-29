* Stata leebounds (Tauchmann, SSC, v1.5) reference for
* sp.lee_bounds(trimming='leebounds').  Needs: ssc install leebounds
*   python tests/reference_parity/_fixtures/_generate_leebounds_stata_data.py
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_leebounds_stata.do
version 18
clear all
import delimited using "tests/reference_parity/_fixtures/leebounds_samples.csv", clear
tempname fh
file open `fh' using "tests/reference_parity/_fixtures/leebounds_stata.json", write replace
file write `fh' "{" _n
forvalues k = 1/8 {
    preserve
    keep if sample == `k'
    qui leebounds y d, select(s)
    matrix b = e(b)
    matrix V = e(V)
    if `k' > 1 file write `fh' "," _n
    file write `fh' `"  "`k'": {"lower": "' %25.17f (b[1,1]) `", "upper": "' %25.17f (b[1,2])
    file write `fh' `", "se_lower": "' %25.17f (sqrt(V[1,1])) `", "se_upper": "' %25.17f (sqrt(V[2,2]))
    file write `fh' `", "trim": "' %25.17f (e(trim)) "}"
    restore
}
file write `fh' _n "}" _n
file close `fh'
