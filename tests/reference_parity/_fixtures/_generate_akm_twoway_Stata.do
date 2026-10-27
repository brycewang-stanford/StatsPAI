* Saved fixed effects of reghdfe and felsdvreg on the AKM fixture.
*   python tests/reference_parity/_fixtures/_generate_akm_twoway_data.py
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_akm_twoway_Stata.do
* reghdfe 6.13 (SSC). felsdvreg is Stata Journal st0143 (sj8-2), installed
* into a private adopath under c(tmpdir), never into PLUS.
* Writes akm_twoway_Stata.csv (one row per observation, the saved effects)
* and akm_twoway_Stata.json (coefficients, standard errors, constants).
version 18
clear all
set type double
local ado "`c(tmpdir)'/statspai_ado_felsdvreg"
capture mkdir "`ado'"
capture which felsdvreg
if _rc {
    net set ado "`ado'"
    adopath ++ "`ado'"
    net install st0143, from("http://www.stata-journal.com/software/sj8-2") replace
}
else adopath ++ "`ado'"

import delimited using "tests/reference_parity/_fixtures/akm_twoway.csv", clear asdouble
gen long obs = _n

tempname fh
file open `fh' using "tests/reference_parity/_fixtures/akm_twoway_Stata.json", write replace
file write `fh' "{" _n

* (1) two absorbed effects, saved under names (tight tolerance: the saved
*     effects inherit the convergence error of the demeaning)
reghdfe lnw x, absorb(wfe2=worker ffe2=firm) tolerance(1e-14) iterate(100000)
gen byte s2 = e(sample)
file write `fh' `"  "reghdfe2": {"b": "' %24.16e (_b[x]) `", "se": "' %24.16e (_se[x]) `", "cons": "' %24.16e (_b[_cons]) `", "N": "' (e(N)) "}," _n

* (2) three absorbed effects, clustered by firm
reghdfe lnw x, absorb(wfe3=worker ffe3=firm yfe3=year) cluster(firm) tolerance(1e-14) iterate(100000)
gen byte s3 = e(sample)
file write `fh' `"  "reghdfe3": {"b": "' %24.16e (_b[x]) `", "se": "' %24.16e (_se[x]) `", "cons": "' %24.16e (_b[_cons]) `", "N": "' (e(N)) "}," _n

* (2b) analytic weights: the saved effects average zero with the weights
gen double wt = 1 + mod(obs, 3)
reghdfe lnw x [aw=wt], absorb(wfe4=worker ffe4=firm) tolerance(1e-14) iterate(100000)
gen byte s4 = e(sample)
file write `fh' `"  "reghdfe_aw": {"b": "' %24.16e (_b[x]) `", "se": "' %24.16e (_se[x]) `", "cons": "' %24.16e (_b[_cons]) `", "N": "' (e(N)) "}," _n

* (3) savefe: __hdfe1__, __hdfe2__
reghdfe lnw x, absorb(worker firm, savefe) tolerance(1e-14) iterate(100000)
rename __hdfe1__ sv1
rename __hdfe2__ sv2
file write `fh' `"  "savefe": {"b": "' %24.16e (_b[x]) `", "cons": "' %24.16e (_b[_cons]) "}," _n

* (4) felsdvreg: person effect absorbed, firm effects as dummies
felsdvreg lnw x, ivar(worker) jvar(firm) feff(fel_f) peff(fel_p) xb(fel_xb) ///
    res(fel_r) mover(fel_m) mnum(fel_mn) pobs(fel_po) group(fel_g)
file write `fh' `"  "felsdvreg": {"b": "' %24.16e (_b[x]) `", "se": "' %24.16e (_se[x]) `", "N": "' (e(N)) "}," _n

* (5) felsdvreg, normalize
felsdvreg lnw x, ivar(worker) jvar(firm) feff(fln_f) peff(fln_p) xb(fln_xb) ///
    res(fln_r) mover(fln_m) mnum(fln_mn) pobs(fln_po) group(fln_g) normalize
file write `fh' `"  "felsdvreg_norm": {"b": "' %24.16e (_b[x]) `", "se": "' %24.16e (_se[x]) `", "N": "' (e(N)) "}" _n
file write `fh' "}" _n
file close `fh'

sort obs
keep obs worker firm year s2 s3 s4 wt wfe2 ffe2 wfe3 ffe3 yfe3 wfe4 ffe4 sv1 sv2 ///
    fel_f fel_p fel_xb fel_r fel_m fel_mn fel_po fel_g fln_f fln_p fln_g
format wfe2 ffe2 wfe3 ffe3 yfe3 wfe4 ffe4 sv1 sv2 fel_f fel_p fel_xb fel_r fln_f fln_p %24.16e
export delimited using "tests/reference_parity/_fixtures/akm_twoway_Stata.csv", replace datafmt
