* Reference values for tests/reference_parity/test_zivot_andrews_parity.py
* Stata 18 with the user-written command zandrews 1.0.5 (C. F. Baum, SSC).
* Run from this folder:
*     do _generate_zivot_andrews_Stata.do
* Reads zivot_andrews.csv; writes
*   zivot_andrews_Stata.csv       one row per case: minimum t, its
*                                 observation, the lag order chosen
*   zivot_andrews_Stata_path.csv  the t statistic at every candidate date
* zandrews is installed into _ado_zandrews/ next to this file (gitignored),
* not into the user's PLUS directory.
clear all
set more off
set type double
capture mkdir "_ado_zandrews"
net set ado "_ado_zandrews"
capture which zandrews
if _rc ssc install zandrews
adopath ++ "_ado_zandrews"

import delimited using "zivot_andrews.csv", clear case(preserve) asdouble
gen long obs = _n
tsset obs

tempname cases path
postfile `cases' int id str8 series str12 model str8 lagmethod ///
    int maxlags double trim double level double tmin int tminobs ///
    int bestlag int nobs using "_za_cases", replace
postfile `path' int id int obs double t using "_za_path", replace

* each spec: lagmethod maxlags trim level. zandrews uses maxlags() only
* with lagmethod(input); otherwise it searches up to int(T^0.25) whatever
* maxlags() says (the AIC / BIC / TTest rows pass 6 or 8 and get 3).
local id 0
foreach v in rw brk {
    foreach b in intercept trend both {
        foreach spec in "input 1 0.15 0.10" "input 3 0.15 0.10" ///
            "input 2 0.10 0.10" "AIC 6 0.15 0.10" "BIC 6 0.15 0.10" ///
            "TTest 6 0.15 0.10" "TTest 8 0.07 0.02" "AIC 3 0.20 0.10" {
            tokenize `spec'
            local ++id
            capture drop zt
            quietly zandrews `v', break(`b') lagmethod(`1') maxlags(`2') ///
                trim(`3') level(`4') generate(zt)
            post `cases' (`id') ("`v'") ("`b'") ("`1'") (`2') (`3') (`4') ///
                (r(tmin)) (r(tminobs)) (`r(bestlag)') (`r(nobs)')
            forvalues i = 1/`=_N' {
                if zt[`i'] < . post `path' (`id') (`i') (zt[`i'])
            }
        }
    }
}
postclose `cases'
postclose `path'
use "_za_cases", clear
format trim level tmin %24.17g
export delimited using "zivot_andrews_Stata.csv", replace
use "_za_path", clear
format t %24.17g
export delimited using "zivot_andrews_Stata_path.csv", replace
erase "_za_cases.dta"
erase "_za_path.dta"
