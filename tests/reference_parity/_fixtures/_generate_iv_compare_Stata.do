* Reference for sp.iv_compare on rd_iv_ivw.csv (y on d, instruments z1 z2 z3,
* controls x1 x2): ivregress 2sls / liml and ivreg2 fuller(1), each with the
* default and the robust VCE and the small-sample correction, and the
* first-stage F of the excluded instruments (estat firststage).
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_iv_compare_Stata.do
version 18
clear all
set type double
import delimited using "tests/reference_parity/_fixtures/rd_iv_ivw.csv", clear asdouble
tempname fh
file open `fh' using "tests/reference_parity/_fixtures/iv_compare_Stata.json", write replace
file write `fh' "{" _n
foreach v in "unadjusted" "robust" {
    foreach est in 2sls liml {
        qui ivregress `est' y x1 x2 (d = z1 z2 z3), vce(`v') small
        file write `fh' `"  "`est'_`v'": {"b": "' %24.16e (_b[d]) `", "se": "' %24.16e (_se[d]) "}," _n
    }
    local opt = cond("`v'" == "robust", "robust", "")
    qui ivreg2 y x1 x2 (d = z1 z2 z3), fuller(1) small `opt'
    file write `fh' `"  "fuller_`v'": {"b": "' %24.16e (_b[d]) `", "se": "' %24.16e (_se[d]) "}," _n
    qui ivregress 2sls y x1 x2 (d = z1 z2 z3), vce(`v') small
    qui estat firststage
    matrix S = r(singleresults)
    file write `fh' `"  "first_stage_F_`v'": "' %24.16e (S[1, 4]) "," _n
}
qui which ivreg2
file write `fh' `"  "versions": {"stata": "`c(stata_version)'", "ivreg2": "4.1.12"}"' _n
file write `fh' "}" _n
file close `fh'
