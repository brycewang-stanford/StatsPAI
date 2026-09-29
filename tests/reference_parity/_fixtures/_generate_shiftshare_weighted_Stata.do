* ivregress 2sls with analytic weights and HC1 (vce(robust) small): the
* reference for sp.bartik(weights=).
*   python tests/reference_parity/_fixtures/_generate_shiftshare_weighted_data.py
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_shiftshare_weighted_Stata.do
version 18
clear all
import delimited using "tests/reference_parity/_fixtures/shiftshare_weighted_loc.csv", clear asdouble
qui ivregress 2sls y c1 c2 (x = z) [aw = w], vce(robust) small
tempname fh
file open `fh' using "tests/reference_parity/_fixtures/shiftshare_weighted_Stata.json", write replace
file write `fh' "{" `""b": "' %25.17f (_b[x]) `", "se": "' %25.17f (_se[x])
file write `fh' `", "b_c1": "' %25.17f (_b[c1]) `", "se_c1": "' %25.17f (_se[c1]) "}" _n
file close `fh'
