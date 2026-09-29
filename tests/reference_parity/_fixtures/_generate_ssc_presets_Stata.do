* Standard errors of x under the small-sample conventions of Stata's
* commands, for the sp.ssc presets.
*   python tests/reference_parity/_fixtures/_generate_ssc_presets_data.py
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_ssc_presets_Stata.do
version 18
clear all
import delimited using "tests/reference_parity/_fixtures/ssc_presets.csv", clear asdouble
xtset firm year
tempname fh
file open `fh' using "tests/reference_parity/_fixtures/ssc_presets_Stata.json", write replace
file write `fh' "{" _n
qui regress y x w, vce(cluster ind)
file write `fh' `"  "regress_cl": "' %25.17f (_se[x]) "," _n
qui regress y x w, vce(robust)
file write `fh' `"  "regress_hc1": "' %25.17f (_se[x]) "," _n
qui areg y x w, absorb(firm) vce(cluster ind)
file write `fh' `"  "areg_cl": "' %25.17f (_se[x]) "," _n
qui reghdfe y x w, absorb(firm) vce(cluster ind)
file write `fh' `"  "reghdfe_cl": "' %25.17f (_se[x]) "," _n
qui xtreg y x w, fe vce(cluster ind)
file write `fh' `"  "xtreg_fe_cl": "' %25.17f (_se[x]) "," _n
qui ivregress 2sls y w (x = z), vce(cluster ind)
file write `fh' `"  "ivregress_cl": "' %25.17f (_se[x]) "," _n
qui ivregress 2sls y w (x = z), vce(cluster ind) small
file write `fh' `"  "ivregress_small_cl": "' %25.17f (_se[x]) "," _n
qui ivregress 2sls y w (x = z), vce(robust)
file write `fh' `"  "ivregress_hc": "' %25.17f (_se[x]) _n
file write `fh' "}" _n
file close `fh'
