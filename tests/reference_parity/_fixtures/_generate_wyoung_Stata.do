* Stata wyoung (Reif, SSC 2025-01-10) permutation Westfall-Young p-values on
* the two designs of _generate_wyoung_data.py, for sp.westfall_young.
* wyoung draws 20,000 random re-randomizations (Monte Carlo SE of a p-value
* below 0.0036); sp.westfall_young enumerates all of them.
* Install wyoung into a private ado tree first, e.g.
*   sysdir set PLUS "<scratch>/ado/plus"
*   ssc install wyoung
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_wyoung_Stata.do
version 18
clear all
set linesize 255
tempname fh M
file open `fh' using "tests/reference_parity/_fixtures/wyoung_Stata.json", write replace
file write `fh' "{" _n
import delimited using "tests/reference_parity/_fixtures/wyoung_strata.csv", clear asdouble
wyoung y1 y2 y3, cmd(regress OUTCOMEVAR treat x i.stratum, robust) familyp(treat) permute(treat) strata(stratum) reps(20000) seed(12345)
matrix `M' = r(table)
file write `fh' `"  "strata": ["'
forvalues i = 1/3 {
    if `i' > 1 file write `fh' ", "
    file write `fh' "[" %25.16e (`M'[`i',1]) ", " %25.16e (`M'[`i',2]) ", " %25.16e (`M'[`i',3]) ", " %25.16e (`M'[`i',4]) "]"
}
file write `fh' "]," _n
import delimited using "tests/reference_parity/_fixtures/wyoung_cluster.csv", clear asdouble
wyoung y1 y2 y3, cmd(regress OUTCOMEVAR treat x, cluster(cl)) familyp(treat) permute(treat) cluster(cl) reps(20000) seed(12345)
matrix `M' = r(table)
file write `fh' `"  "cluster": ["'
forvalues i = 1/3 {
    if `i' > 1 file write `fh' ", "
    file write `fh' "[" %25.16e (`M'[`i',1]) ", " %25.16e (`M'[`i',2]) ", " %25.16e (`M'[`i',3]) ", " %25.16e (`M'[`i',4]) "]"
}
file write `fh' "]," _n
file write `fh' `"  "columns": ["coef", "stderr", "p", "pwyoung"], "reps": 20000"' _n "}" _n
file close `fh'
