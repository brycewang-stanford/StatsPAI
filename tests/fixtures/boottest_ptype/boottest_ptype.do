* Stata 18 + boottest (SSC). Produces stata_dist.csv and the four r(p)
* values pinned in tests/test_option_scan_r7_openA.py.
*   symmetric p = .595595595595595606
*   equaltail p = .580580580580580596
*   lower     p = .290290290290290298
*   upper     p = .709709709709709702
*   t         = -.7452775697490591211
* data.csv: numpy default_rng(1), G = 14 clusters of 20, written with %.17g.
clear
set type double
import delimited "data.csv", clear asdouble
quietly regress y x, cluster(g)
foreach p in symmetric equaltail lower upper {
    quietly boottest x, ptype(`p') reps(999) seed(42) noci svmat(t)
    di "`p' p=" %20.0g r(p) " t=" %21.0g r(t)
    matrix D_`p' = r(dist)
}
matrix D = D_symmetric
svmat double D, names(dd)
keep dd1
drop if missing(dd1)
format dd1 %21.0g
export delimited using "stata_dist.csv", replace
