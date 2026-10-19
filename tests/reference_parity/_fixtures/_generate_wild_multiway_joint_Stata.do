* Stata boottest (Roodman, MacKinnon, Nielsen and Webb 2019) references for
* sp.wild_cluster_boot with multi-way clustering, joint restrictions and
* absorbed fixed effects that are not nested in the clusters.
* Least squares of y on x1 x2 (fixest_postestimation_data.csv); clusters a = cl
* (12) and b = f3 (5). Rademacher weights with reps above 2^G enumerate every
* sign vector of the bootstrap clusters, so the p-values are exact.
*   python tests/reference_parity/_fixtures/_generate_fixest_postestimation_data.py
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_wild_multiway_joint_Stata.do
version 18
clear all
import delimited using "tests/reference_parity/_fixtures/fixest_postestimation_data.csv", clear asdouble
gen a = cl
gen b = f3
tempname fh
file open `fh' using "tests/reference_parity/_fixtures/wild_multiway_joint_Stata.json", write replace
file write `fh' "{" _n
qui reg y x1 x2, cluster(a)
* one coefficient, one way
qui boottest x1=0.75, reps(9999) weight(rademacher) nograph
file write `fh' `"  "oneway": {"p": "' %25.17f (r(p)) `", "t": "' %25.17f (r(t)) `", "reps": "' (r(reps)) "}," _n
* one coefficient, two ways, bootstrap clusters a and b
foreach bc in a b {
    qui boottest x1=0.75, cluster(a b) bootcluster(`bc') reps(9999) weight(rademacher) nograph
    file write `fh' `"  "twoway_`bc'": {"p": "' %25.17f (r(p)) `", "t": "' %25.17f (r(t)) `", "reps": "' (r(reps)) "}," _n
}
* two restrictions jointly, one way and two ways
qui boottest (x1=0.75) (x2=-0.25), reps(9999) weight(rademacher) nograph
file write `fh' `"  "joint_oneway": {"p": "' %25.17f (r(p)) `", "F": "' %25.17f (r(F)) `", "reps": "' (r(reps)) "}," _n
foreach bc in a b {
    qui boottest (x1=0.75) (x2=-0.25), cluster(a b) bootcluster(`bc') reps(9999) weight(rademacher) nograph
    file write `fh' `"  "joint_twoway_`bc'": {"p": "' %25.17f (r(p)) `", "F": "' %25.17f (r(F)) `", "reps": "' (r(reps)) "}"
    if "`bc'" == "a" file write `fh' ","
    file write `fh' _n
}
* absorbed effects that are not nested in the clusters (reghdfe): boottest
* bootstraps the model with the effects as dummies
gen yy = y + 0.33*x2
local i 0
foreach fe in f3 f1 {
    local ++i
    qui reghdfe yy x1 x2, absorb(`fe') cluster(cl)
    qui boottest x2, reps(9999) weight(rademacher) nograph
    file write `fh' `"  ,"absorbed_`i'": {"p": "' %25.17f (r(p)) `", "reps": "' (r(reps)) "}" _n
}
* two effects: boottest takes one absorbed effect, so f3 enters as dummies
qui reghdfe yy x1 x2 i.f3, absorb(f1) cluster(cl)
qui boottest x2, reps(9999) weight(rademacher) nograph
file write `fh' `"  ,"absorbed_3": {"p": "' %25.17f (r(p)) `", "reps": "' (r(reps)) "}" _n
qui reghdfe yy x1 x2, absorb(f1) cluster(cl)
qui boottest (x1=0.45) (x2=0.1), reps(9999) weight(rademacher) nograph
file write `fh' `"  ,"absorbed_joint": {"p": "' %25.17f (r(p)) `", "F": "' %25.17f (r(F)) `", "reps": "' (r(reps)) "}" _n
file write `fh' "}" _n
file close `fh'
