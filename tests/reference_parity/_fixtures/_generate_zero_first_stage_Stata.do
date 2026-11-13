* Reference for sp.zero_first_stage on zero_first_stage.csv.
* The estimator is a combination of four regressions and one IV fit, so
* the reference runs those in Stata and combines them here:
*   pi_zfs, gamma  : first stage and reduced form where zfs == 1
*   pi_main, rho   : first stage and reduced form where zfs == 0
*   beta_iv        : 2SLS where zfs == 0
*   corrected      : (rho - gamma) / pi_main
* Three designs: robust; clustered on g; clustered on g with f absorbed
* (reghdfe / ivreghdfe).
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_zero_first_stage_Stata.do
version 18
clear all
set type double
import delimited using "tests/reference_parity/_fixtures/zero_first_stage.csv", clear asdouble
tempname fh
file open `fh' using "tests/reference_parity/_fixtures/zero_first_stage_Stata.json", write replace
file write `fh' "{" _n
foreach design in robust cluster absorb {
    if "`design'" == "robust" {
        local ols "regress"
        local iv "ivregress 2sls"
        local opt "vce(robust)"
        local ivopt "vce(robust) small"
    }
    if "`design'" == "cluster" {
        local ols "regress"
        local iv "ivregress 2sls"
        local opt "vce(cluster g)"
        local ivopt "vce(cluster g) small"
    }
    if "`design'" == "absorb" {
        local ols "reghdfe"
        local iv "ivreghdfe"
        local opt "absorb(f) vce(cluster g)"
        local ivopt "absorb(f) cluster(g)"
    }
    qui `ols' d z x if zfs == 1, `opt'
    local pi_zfs = _b[z]
    local pi_zfs_se = _se[z]
    qui `ols' y z x if zfs == 1, `opt'
    local gamma = _b[z]
    local gamma_se = _se[z]
    qui `ols' d z x if zfs == 0, `opt'
    local pi_main = _b[z]
    local pi_main_se = _se[z]
    qui `ols' y z x if zfs == 0, `opt'
    local rho = _b[z]
    qui `iv' y x (d = z) if zfs == 0, `ivopt'
    local beta_iv = _b[d]
    local beta_iv_se = _se[d]
    local corrected = (`rho' - `gamma') / `pi_main'
    file write `fh' `"  "`design'": {"pi_zfs": "' %24.16e (`pi_zfs') `", "pi_zfs_se": "' %24.16e (`pi_zfs_se') ///
        `", "gamma": "' %24.16e (`gamma') `", "gamma_se": "' %24.16e (`gamma_se') ///
        `", "pi_main": "' %24.16e (`pi_main') `", "pi_main_se": "' %24.16e (`pi_main_se') ///
        `", "rho": "' %24.16e (`rho') `", "beta_iv": "' %24.16e (`beta_iv') ///
        `", "beta_iv_se": "' %24.16e (`beta_iv_se') `", "corrected": "' %24.16e (`corrected') "}," _n
}
file write `fh' `"  "versions": {"stata": "18 MP", "reghdfe": "6.13.1", "ivreghdfe": "1.1.4", "ivreg2": "4.1.12"}"' _n
file write `fh' "}" _n
file close `fh'
