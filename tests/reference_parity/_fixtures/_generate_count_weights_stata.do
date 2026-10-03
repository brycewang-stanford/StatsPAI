*! _generate_count_weights_stata.do
*! Weighted poisson / ppmlhdfe / logit / probit under each Stata
*! weight type and variance, for tests/reference_parity/
*! test_count_weights_stata_parity.py. Stata 18 MP; ppmlhdfe from SSC.
*! Run from this directory: writes count_weights_stata.json.
*!
*! Double precision throughout (set type double + asdouble): without it
*! import delimited stores decimals as float; see
*! tests/stata_parity/option_parity/README.md.
clear all
set more off
set type double
import delimited "../../stata_translation_holdout/holdout_cross.csv", clear asdouble case(preserve)
tempname fh
file open `fh' using "count_weights_stata.json", write replace
file write `fh' "{" _n
program define _w
    args fh key
    local ll = cond(missing(e(ll)), -1e300, e(ll))
    local ll0 = cond(missing(e(ll_0)), -1e300, e(ll_0))
    file write `fh' `"  ""' "`key'" `"": {"N": "' (e(N)) `", "ll": "' %24.16e (`ll') `", "ll_0": "' %24.16e (`ll0')
    foreach v in x1 x2 {
        file write `fh' `", "b_`v'": "' %24.16e (_b[`v']) `", "se_`v'": "' %24.16e (_se[`v'])
    }
    file write `fh' "}," _n
end
* nbreg is left out: on this outcome its alpha goes to the boundary and
* Stata reports "convergence not achieved". The test checks sp.nbreg's
* weights by the frequency-expansion identity instead.
qui poisson cnt x1 x2 [iw=w]
_w `fh' "poisson_iw"
qui poisson cnt x1 x2 [fw=fw]
_w `fh' "poisson_fw"
qui poisson cnt x1 x2 [pw=w]
_w `fh' "poisson_pw"
qui poisson cnt x1 x2 [pw=w], vce(cluster g)
_w `fh' "poisson_pw_cluster"
foreach cmd in logit probit {
    qui `cmd' yb x1 x2 [iw=w]
    _w `fh' "`cmd'_iw"
    qui `cmd' yb x1 x2 [fw=fw]
    _w `fh' "`cmd'_fw"
    qui `cmd' yb x1 x2 [pw=w]
    _w `fh' "`cmd'_pw"
    qui `cmd' yb x1 x2 [pw=w], vce(cluster g)
    _w `fh' "`cmd'_pw_cluster"
}
qui ppmlhdfe cnt x1 x2 [pw=w], absorb(k) vce(robust)
_w `fh' "ppmlhdfe_pw"
qui ppmlhdfe cnt x1 x2 [pw=w], absorb(k) vce(cluster g)
_w `fh' "ppmlhdfe_pw_cluster"
qui ppmlhdfe cnt x1 x2, absorb(k) vce(cluster g)
_w `fh' "ppmlhdfe_cluster"
file write `fh' `"  "_meta": {"stata": "' (c(stata_version)) `", "data": "tests/stata_translation_holdout/holdout_cross.csv"}"' _n
file write `fh' "}" _n
file close `fh'
