* Reference for sp.sharpened_qvalues: Michael Anderson's do-file
* fdr_sharpened_qvalues.do ("Last modified: M. Anderson, 11/20/07"), run
* unchanged on every p-vector in sharpened_qvalues_input.csv.
*
* The do-file is not committed (no licence is stated). It was distributed at
* https://are.berkeley.edu/~mlanderson/downloads/fdr_sharpened_qvalues.do.zip
* (404 since 2026-09); the Wayback copy used here is
*   https://web.archive.org/web/20250216223818id_/https://are.berkeley.edu/~mlanderson/downloads/fdr_sharpened_qvalues.do.zip
* sha256 of the unzipped fdr_sharpened_qvalues.do:
*   43e4b411465015e9dfc6ea51ec9467445fbe708a2d81618d52b5213542ff8b44
* (The 2021 Wayback copy, sha256 ed48e595..., differs only in whitespace and
* `~=` versus `!=`.)
*
* The do-file waits at `pause` for the user to paste p-values into the float
* variable pval. Here `pause` is replaced by a program that appends the
* p-values of the current case, stored as float, which is what pasting into
* a float variable gives. Nothing else in the do-file is touched.
*
* The do-file runs under `set type float`, Stata's default, so the
* variables it generates are float. (Under `set type double` they are double
* and a few q-values on a threshold move by 0.001.)
*
* Output: sharpened_qvalues_Stata.csv with case, the position of the
* p-value in its case, and pval / bky06_qval in %21x (exact).
*
* Run from the repository root with Stata 18:
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_sharpened_qvalues_Stata.do <path to fdr_sharpened_qvalues.do>
args anderson_do
if `"`anderson_do'"' == "" {
    display as error "pass the path of fdr_sharpened_qvalues.do"
    exit 198
}
version 18
set type double
clear all
local fix "tests/reference_parity/_fixtures"

import delimited "`fix'/sharpened_qvalues_input.csv", clear asdouble ///
    stringcols(1) varnames(1)
generate long order = _n
bysort case (order): generate long pos = _n
tempfile all
save `all'
levelsof case, local(cases)

capture program drop pause
program define pause
    if `"`0'"' == "on" | `"`0'"' == "off" exit
    if _N == 0 {
        append using "$BKY_CASE_FILE"
        exit
    }
    display as error "unexpected pause with data in memory"
    exit 459
end

tempname fh
file open `fh' using "`fix'/sharpened_qvalues_Stata.csv", write replace text
file write `fh' "case,pos,pval_hex,bky06_qval_hex" _n
foreach c of local cases {
    use `all', clear
    keep if case == "`c'"
    sort pos
    keep pval
    recast float pval, force
    tempfile one
    save `one'
    global BKY_CASE_FILE "`one'"
    clear
    * Stata's default storage type, under which the do-file's gen creates
    * float variables (bky06_qval, fdr_temp1, fdr_temp2).
    set type float
    do `"`anderson_do'"'
    set type double
    version 18
    forvalues i = 1/`=_N' {
        file write `fh' "`c',`i'," %21x (pval[`i']) "," %21x (bky06_qval[`i']) _n
    }
}
file close `fh'
