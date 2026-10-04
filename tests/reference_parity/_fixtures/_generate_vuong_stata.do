* ---------------------------------------------------------------------------
* Stata reference for tests/reference_parity/test_vuong_pscl_parity.py
*
* Requires: Stata 18 (zip and zinb are official).
* Run:      stata -b do _generate_vuong_stata.do   (from this directory,
*           after Rscript _generate_vuong_pscl.R has written vuong_data.csv)
*
* A second, independent reference for the Vuong statistic of a
* zero-inflated model against its plain counterpart. Stata 18 refuses the
* `vuong` option ("Vuong test is not appropriate for testing zero
* inflation") and computes it only under `forcevuong`; the return code of
* the refusal is recorded as well.
* ---------------------------------------------------------------------------
version 18
clear all
import delimited "vuong_data.csv", clear asdouble

capture zip y x1 x2, inflate(x1 x2) vuong
local rc_refused = _rc
quietly zip y x1 x2, inflate(x1 x2) forcevuong
local v_zip = e(vuong)
quietly zinb y x1 x2, inflate(x1 x2) forcevuong
local v_zinb = e(vuong)

tempname fh
file open `fh' using "vuong_stata.json", write replace text
file write `fh' "{" _n
file write `fh' `"  "zip_poisson": "' %21.16e (`v_zip') "," _n
file write `fh' `"  "zinb_nb2": "' %21.16e (`v_zinb') "," _n
file write `fh' `"  "rc_vuong_without_force": `rc_refused',"' _n
file write `fh' `"  "_meta": {"stata_version": "`c(stata_version)'", "generated": "`c(current_date)'"}"' _n
file write `fh' "}" _n
file close `fh'
