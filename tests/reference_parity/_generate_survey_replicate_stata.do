* Frozen Stata 18 reference for replicate-weight designs: svy regress /
* svy logit / svy mean with brrweight(), fay(), jkrweight(..., multiplier())
* and bsrweight() on _fixtures/survey_replicate_data.csv (the replicate
* weights exported from R by _generate_survey_replicate_R.R).
* Writes _fixtures/survey_replicate_stata.json. Run from the repository
* root:  stata-mp -b do tests/reference_parity/_generate_survey_replicate_stata.do
* svy keeps the replicate estimates in float unless saving(, double) is
* given, which moves e(V) by ~1e-7 relative; every call saves in double.
* Numbers are written in %24.17e (Stata's %g drops the leading zero, which
* is not JSON). df -1 = Stata reports no df (svy bootstrap: normal).
version 18
set type double
import delimited tests/reference_parity/_fixtures/survey_replicate_data.csv, clear asdouble
tempname fh
tempfile reps
file open `fh' using tests/reference_parity/_fixtures/survey_replicate_stata.json, write replace
file write `fh' "{" _n
local first 1
foreach spec in brr fay jk1 jk1mse boot {
    if "`spec'" == "brr"    svyset [pw=w], brrweight(brr_*) vce(brr)
    if "`spec'" == "fay"    svyset [pw=w], brrweight(fay_*) fay(0.3) vce(brr)
    if "`spec'" == "jk1"    svyset [pw=w], jkrweight(jk1_*, multiplier(0.95)) vce(jackknife)
    if "`spec'" == "jk1mse" svyset [pw=w], jkrweight(jk1_*, multiplier(0.95)) vce(jackknife) mse
    if "`spec'" == "boot"   svyset [pw=w], bsrweight(boot_*) vce(bootstrap)
    quietly svy, saving(`reps', replace double): regress y x1 x2
    matrix V = e(V)
    local df = cond(missing(e(df_r)), -1, e(df_r))
    local g "[`=strtrim(string(sqrt(V[3,3]), "%24.17e"))', `=strtrim(string(sqrt(V[1,1]), "%24.17e"))', `=strtrim(string(sqrt(V[2,2]), "%24.17e"))']"
    quietly svy, saving(`reps', replace double): logit yb x1 x2
    matrix V = e(V)
    local l "[`=strtrim(string(sqrt(V[3,3]), "%24.17e"))', `=strtrim(string(sqrt(V[1,1]), "%24.17e"))', `=strtrim(string(sqrt(V[2,2]), "%24.17e"))']"
    quietly svy, saving(`reps', replace double): mean y
    matrix V = e(V)
    local m = strtrim(string(sqrt(V[1,1]), "%24.17e"))
    if !`first' file write `fh' "," _n
    local first 0
    file write `fh' `"  "`spec'": {"df": `df', "glm_se": `g', "logit_se": `l', "mean_se": `m'}"'
}
file write `fh' _n "}" _n
file close `fh'
