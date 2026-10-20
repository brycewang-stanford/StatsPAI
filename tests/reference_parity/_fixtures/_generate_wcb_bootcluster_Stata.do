* Stata 18 + SSC boottest references for the crossing-bootcluster case of
* tests/reference_parity/test_wcb_bootcluster_parity.py (boottest is used
* as installed; its version line is recorded).
*   python _generate_wcb_bootcluster_data.py && stata-mp -b do _generate_wcb_bootcluster_Stata.do
version 18
clear all
set type double
import delimited using "wcb_bootcluster.csv", clear asdouble
capture file close fh
file open fh using "wcb_bootcluster_Stata.json", write replace text
file write fh "{" _n
qui reg y x w, vce(cluster a)
local first 1
foreach h in 0 0.2 0.4 0.6 {
    foreach bc in "b" "b a" {
        qui boottest x=`h', reps(9999) seed(1) bootcluster(`bc') weighttype(rademacher) ptolerance(1e-14) noci nograph
        local key = "h`h'_" + subinstr("`bc'", " ", "", .)
        if !`first' file write fh "," _n
        local first 0
        file write fh `"  "`key'": {"p": "' %25.17e (r(p)) `", "t": "' %25.17e (r(t)) `", "reps": "' (r(reps)) "}"
    }
}
qui reg y x w, vce(cluster b)
qui boottest x=0.2, reps(9999) seed(1) bootcluster(a) weighttype(rademacher) ptolerance(1e-14) noci nograph
file write fh "," _n `"  "clusterb_bc_a": {"p": "' %25.17e (r(p)) `", "t": "' %25.17e (r(t)) `", "reps": "' (r(reps)) "}"
file write fh _n "}" _n
file close fh
