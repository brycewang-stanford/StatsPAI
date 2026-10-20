* Stata wyoung (Reif, SSC 2025-01-10) with a cmd(regress ...) that has no SE
* option, i.e. classical OLS t-statistics, for sp.westfall_young(robust=False).
* Same data and settings as _generate_wyoung_Stata.do (20,000 random
* re-randomizations; sp.westfall_young enumerates all of them).
*   sysdir set PLUS "<scratch>/ado/plus"
*   ssc install wyoung
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_wyoung_classical_Stata.do
version 18
clear all
set linesize 255
tempname fh M
file open `fh' using "tests/reference_parity/_fixtures/wyoung_classical_Stata.json", write replace
file write `fh' "{" _n
import delimited using "tests/reference_parity/_fixtures/wyoung_strata.csv", clear asdouble
wyoung y1 y2 y3, cmd(regress OUTCOMEVAR treat x i.stratum) familyp(treat) permute(treat) strata(stratum) reps(20000) seed(12345)
matrix `M' = r(table)
file write `fh' `"  "strata": ["'
forvalues i = 1/3 {
    if `i' > 1 file write `fh' ", "
    file write `fh' "[" %25.16e (`M'[`i',1]) ", " %25.16e (`M'[`i',2]) ", " %25.16e (`M'[`i',3]) ", " %25.16e (`M'[`i',4]) "]"
}
file write `fh' "]," _n
wyoung y1 y2 y3, cmd(regress OUTCOMEVAR treat x) familyp(treat) permute(treat) reps(20000) seed(12345)
matrix `M' = r(table)
file write `fh' `"  "complete": ["'
forvalues i = 1/3 {
    if `i' > 1 file write `fh' ", "
    file write `fh' "[" %25.16e (`M'[`i',1]) ", " %25.16e (`M'[`i',2]) ", " %25.16e (`M'[`i',3]) ", " %25.16e (`M'[`i',4]) "]"
}
file write `fh' "]," _n
file write `fh' `"  "columns": ["coef", "stderr", "p", "pwyoung"], "reps": 20000"' _n "}" _n
file close `fh'
