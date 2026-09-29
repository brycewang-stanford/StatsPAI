* ivreghdfe (ivreg2 4.1.x + reghdfe 6.12.3) weak-IV reference with nested
* fixed effects: year inside region x year, unit inside the unit clusters.
*   python tests/reference_parity/_fixtures/_generate_weakiv_nested_fe_data.py
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_weakiv_nested_fe_Stata.do
version 18
clear all
import delimited using "tests/reference_parity/_fixtures/weakiv_nested_fe.csv", clear asdouble case(preserve)
ivreghdfe y (d = z) w1, absorb(year unit rXy) cluster(unit) ffirst
tempname fh
file open `fh' using "tests/reference_parity/_fixtures/weakiv_nested_fe_Stata.json", write replace
file write `fh' "{" _n
file write `fh' `"  "b": "' %25.17f (_b[d]) `", "se": "' %25.17f (_se[d]) "," _n
file write `fh' `"  "kp_f": "' %25.17f (e(widstat)) `", "ar_f": "' %25.17f (e(arf)) "," _n
file write `fh' `"  "ar_p": "' %25.17f (e(arfp)) `", "ar_chi2": "' %25.17f (e(archi2)) "," _n
file write `fh' `"  "df_a": "' (e(df_a)) `", "N": "' (e(N)) `", "N_clust": "' (e(N_clust)) _n
file write `fh' "}" _n
file close `fh'
