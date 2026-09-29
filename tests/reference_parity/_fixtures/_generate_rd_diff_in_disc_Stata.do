* Difference-in-discontinuities reference: rdrobust (conventional) in each
* period with a common h, and the pooled triangular-kernel regression fully
* interacted in side and period, clustered by site and robust.
*   python tests/reference_parity/_fixtures/_generate_rd_diff_in_disc_data.py
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_rd_diff_in_disc_Stata.do
version 18
clear all
import delimited using "tests/reference_parity/_fixtures/rd_diff_in_disc.csv", clear asdouble
local h = 0.6
qui rdrobust y x if post == 0, h(`h') kernel(triangular) p(1)
local pre = e(tau_cl)
qui rdrobust y x if post == 1, h(`h') kernel(triangular) p(1)
local postrd = e(tau_cl)
gen double w = max(0, 1 - abs(x) / `h')
gen double T = x >= 0
gen double Tx = T * x
gen double P = post
gen double PT = post * T
gen double Px = post * x
gen double PTx = post * T * x
qui reg y T x Tx P PT Px PTx [aw = w] if w > 0, vce(cluster site)
local b = _b[PT]
local se_cl = _se[PT]
local N = e(N)
qui reg y T x Tx P PT Px PTx [aw = w] if w > 0, vce(robust)
local se_hc1 = _se[PT]
tempname fh
file open `fh' using "tests/reference_parity/_fixtures/rd_diff_in_disc_Stata.json", write replace
file write `fh' "{" `""h": "' %25.17f (`h') `", "rd_pre": "' %25.17f (`pre') `", "rd_post": "' %25.17f (`postrd')
file write `fh' `", "b": "' %25.17f (`b') `", "se_cluster": "' %25.17f (`se_cl') `", "se_hc1": "' %25.17f (`se_hc1') `", "N": "' (`N') "}" _n
file close `fh'
