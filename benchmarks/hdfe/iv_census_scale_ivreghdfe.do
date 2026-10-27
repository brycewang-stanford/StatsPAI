* Stata side of iv_census_scale_bench.py.
*   stata-mp -b do iv_census_scale_ivreghdfe.do <csv> <out.json> <threads>
* Loads the CSV (set type double), runs a 20k-row warm-up fit, then times
* the paper's command on the full data. Writes coefficients, SEs, N and
* timings as JSON.
version 18
clear all
set more off
set type double
args csv out threads
set processors `=min(`threads', c(processors_lic))'

timer clear
timer on 1
import delimited using "`csv'", clear asdouble
timer off 1

preserve
keep in 1/20000
quietly ivreghdfe married x (tight = z), absorb(grp#tract#sex) cluster(county)
restore

timer on 2
ivreghdfe married x (tight = z), absorb(grp#tract#sex) cluster(county)
timer off 2
timer list
local load_s = r(t1)
local fit_s = r(t2)

tempname fh
file open `fh' using "`out'", write replace
file write `fh' "{" _n
file write `fh' `"  "engine": "stata","' _n
file write `fh' `"  "n_rows": "' (_N) "," _n
file write `fh' `"  "load_s": "' %20.6f (`load_s') "," _n
file write `fh' `"  "fit_s": "' %20.6f (`fit_s') "," _n
file write `fh' `"  "n_obs": "' (e(N)) "," _n
file write `fh' `"  "n_clusters": "' (e(N_clust)) "," _n
file write `fh' `"  "first_stage_F": "' %23.16e (e(widstat)) "," _n
file write `fh' `"  "coef": {"tight": "' %23.16e (_b[tight]) `", "x": "' %23.16e (_b[x]) "}," _n
file write `fh' `"  "se": {"tight": "' %23.16e (_se[tight]) `", "x": "' %23.16e (_se[x]) "}," _n
file write `fh' `"  "stata_version": "' "`c(stata_version)'" `", "processors": "' (c(processors)) _n
file write `fh' "}" _n
file close `fh'
