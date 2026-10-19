* Stata 18 MP reference for tests/test_margins_fe_refusal.py (30 groups x 8).
* From this directory: stata-mp -b do stata_margins_fe.do
set more off
set type double
import delimited using "fep.csv", clear asdouble
poisson y x i.g
margins, dydx(x)
poisson y x i.g, vce(cluster g)
margins, dydx(x)
ppmlhdfe y x, a(g) d vce(cluster g)
margins, dydx(x)
