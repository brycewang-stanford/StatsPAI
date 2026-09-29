* Stata boottest (Roodman et al. 2019) WRE bootstrap after ivregress 2sls
* with clustered SEs; 10 clusters, so reps(2000) enumerates all 2^10
* Rademacher sign vectors and the p-value is exact.
*   python tests/reference_parity/_fixtures/_generate_iv_wild_data.py
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_iv_wild_Stata.do
version 18
clear all
import delimited using "tests/reference_parity/_fixtures/iv_wild_data.csv", clear asdouble
qui ivregress 2sls y w (x = z), vce(cluster cl)
qui boottest x, reps(2000) weight(rademacher) nograph
matrix C = r(CI)
tempname fh
file open `fh' using "tests/reference_parity/_fixtures/iv_wild_Stata.json", write replace
file write `fh' "{" `""p": "' %25.17f (r(p)) `", "lo": "' %25.17f (C[1,1])
file write `fh' `", "hi": "' %25.17f (C[1,2]) `", "reps": "' (r(reps)) "}" _n
file close `fh'
