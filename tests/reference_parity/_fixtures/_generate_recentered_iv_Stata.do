* Reference values for tests/reference_parity/test_recentered_iv_stata.py.
* Stata 18 MP; ivreghdfe 1.1.4 (29nov2025), ivreg2 4.1.12, reghdfe 6.13.1,
* ritest 1.1.7 (feb2020). Data from _generate_recentered_iv_data.py.
* Run from tests/reference_parity/_fixtures:
*   stata-mp -b do _generate_recentered_iv_Stata.do
set type double
set more off

* ---------------------------------------------------------------------
* (d) 2SLS with the expected instrument given: recentred and as control
* ---------------------------------------------------------------------
import delimited "recentered_iv_hdfe.csv", clear asdouble
gen double zr = z - mu
ivreghdfe y x1 (d = zr), absorb(f) cluster(c)
di "RECENTER_CLUSTER " %24.17g _b[d] " " %24.17g _se[d] " " e(N)
ivreghdfe y x1 mu (d = z), absorb(f) cluster(c)
di "CONTROL_CLUSTER " %24.17g _b[d] " " %24.17g _se[d]
ivreghdfe y x1 (d = z), absorb(f) cluster(c)
di "NAIVE_CLUSTER " %24.17g _b[d] " " %24.17g _se[d]
ivreghdfe y x1 (d = zr), absorb(f) robust
di "RECENTER_ROBUST " %24.17g _b[d] " " %24.17g _se[d]

* ---------------------------------------------------------------------
* (c) Randomisation inference on 200 fixed within-stratum permutations.
* The statistic is T = zs' M (y - b0 d), M = residual maker of x and the
* stratum FE; it is what sp.recentered_iv ranks.
* ---------------------------------------------------------------------
* The permutations are stored gzip-compressed (the plain CSV is above the
* repository's 500 KB file limit); decompress to a temporary file first.
shell gunzip -c "recentered_iv_ri_perms.csv.gz" > "_recentered_iv_ri_perms_tmp.csv"
import delimited "_recentered_iv_ri_perms_tmp.csv", clear asdouble
erase "_recentered_iv_ri_perms_tmp.csv"
save "recentered_iv_ri_perms.dta", replace

capture program drop ristat
program define ristat, eclass
  tempvar prod
  quietly gen double `prod' = zs * ey
  quietly summarize `prod'
  tempname b
  matrix `b' = (r(sum))
  matrix colnames `b' = T
  ereturn post `b'
  ereturn local cmd "ristat"
end

foreach b0 in 0 1.2 {
  import delimited "recentered_iv_ri.csv", clear asdouble
  gen double yb = y - `b0' * d
  quietly reghdfe yb x, absorb(g) resid(ey) tol(1e-14)
  ristat
  di "TOBS_`b0' " %24.17g _b[T]
  ritest zs _b[T], samplingsourcefile("recentered_iv_ri_perms.dta") ///
      samplingmatchvar(id) reps(200) nodots: ristat
  matrix c = r(c)
  matrix n = r(n)
  di "RI_`b0' c=" c[1,1] " n=" n[1,1]
}
erase "recentered_iv_ri_perms.dta"
