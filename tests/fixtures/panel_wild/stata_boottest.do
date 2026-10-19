* Stata 18 MP + boottest reference for tests/test_panel_wild_boottest.py.
* 40 entities x 6 periods; the cluster c = (id + 2 t) mod 12 cuts across
* entities, so the entity effects are not nested in the clusters.
* From this directory: stata-mp -b do stata_boottest.do
set more off
set type double
import delimited using "panel_wild.csv", clear asdouble
areg y x1 x2, absorb(id) vce(cluster c)
foreach v in x1 x2 {
    boottest `v', reps(10000) weighttype(rademacher) nograph
    matrix ci = r(CI)
    display "`v' " %20.15g r(p) " " %20.15g ci[1,1] " " %20.15g ci[1,2]
}
