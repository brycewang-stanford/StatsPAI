* t / F / p conventions of the estimators an external audit compared with
* StatsPAI 1.40.1 (2026-10-23). For each command: b and se of x1 and x2,
* the joint test of x1 = x2 = 0 (F with both df under -small- style
* commands, chi2 otherwise), its p-value, and e(N). The ML fits are run to
* nrtolerance(1e-14) with ltolerance(0): at the defaults Stata stops up to
* 5e-8 short of the optimum (probit stops on the log-likelihood change).
* Data: inference_conventions.csv (_generate_inference_conventions_data.py).
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_inference_conventions_Stata.do
version 18
clear all
set type double
import delimited using "tests/reference_parity/_fixtures/inference_conventions.csv", clear asdouble
xtset g t
tempname fh
file open `fh' using "tests/reference_parity/_fixtures/inference_conventions_Stata.json", write replace
file write `fh' "{" _n
local first 1

local iv "y x1 x2 (xd = z1 z2)"
local cmds `""gmm_unadj_sandwich_small|ivregress gmm `iv', wmatrix(unadjusted) vce(robust) small" "gmm_unadj_sandwich|ivregress gmm `iv', wmatrix(unadjusted) vce(robust)" "gmm_unadj_small|ivregress gmm `iv', wmatrix(unadjusted) small" "gmm_unadj|ivregress gmm `iv', wmatrix(unadjusted)" "gmm_robust_small|ivregress gmm `iv', small" "gmm_robust|ivregress gmm `iv'" "gmm_cluster_small|ivregress gmm `iv', wmatrix(cluster g) small" "gmm_cluster|ivregress gmm `iv', wmatrix(cluster g)" "gmm_robust_vce_cluster_small|ivregress gmm `iv', wmatrix(robust) vce(cluster g) small" "gmm_robust_vce_cluster|ivregress gmm `iv', wmatrix(robust) vce(cluster g)" "gmm_unadj_vce_cluster_small|ivregress gmm `iv', wmatrix(unadjusted) vce(cluster g) small" "gmm_unadj_vce_cluster|ivregress gmm `iv', wmatrix(unadjusted) vce(cluster g)" "tsls_small|ivregress 2sls `iv', small" "tsls|ivregress 2sls `iv'" "liml_robust_small|ivregress liml `iv', vce(robust) small" "liml_robust|ivregress liml `iv', vce(robust)" "logit_robust|logit yb x1 x2, vce(robust) nrtolerance(1e-14) tolerance(1e-14) ltolerance(0)" "probit_robust|probit yb x1 x2, vce(robust) nrtolerance(1e-14) tolerance(1e-14) ltolerance(0)" "poisson_robust|poisson yc x1 x2, vce(robust) nrtolerance(1e-14) tolerance(1e-14) ltolerance(0)" "xtfe_cluster|xtreg yp x1 x2, fe vce(cluster g)" "xtre|xtreg yp x1 x2, re" "xtbe|xtreg yp x1 x2, be""'

foreach spec of local cmds {
    gettoken key cmd : spec, parse("|")
    local cmd = substr(`"`cmd'"', 2, .)
    qui `cmd'
    local N = e(N)
    local b1 = _b[x1]
    local s1 = _se[x1]
    local b2 = _b[x2]
    local s2 = _se[x2]
    qui test x1 x2
    local stat = cond(r(F) < ., r(F), r(chi2))
    local df_r = cond(r(df_r) < ., r(df_r), -1)
    if !`first' file write `fh' "," _n
    local first 0
    file write `fh' `"  "`key'": {"b_x1": "' %24.17e (`b1') `", "se_x1": "' %24.17e (`s1') `", "b_x2": "' %24.17e (`b2') `", "se_x2": "' %24.17e (`s2') `", "joint": "' %24.17e (`stat') `", "joint_p": "' %24.17e (r(p)) `", "df": "' %4.0f (r(df)) `", "df_r": "' %6.0f (`df_r') `", "N": "' %6.0f (`N') "}"
}

* dup = 2 * x1: the identified effect of x1 is x1 + 2 * dup whichever
* column Stata omits
qui regress y x1 x2 dup
qui lincom x1 + 2 * dup
local eff = r(estimate)
local se = r(se)
qui test (x1 + 2 * dup = 0) (x2 = 0)
file write `fh' "," _n `"  "collinear": {"effect": "' %24.17e (`eff') `", "se": "' %24.17e (`se') `", "joint": "' %24.17e (r(F)) `", "joint_p": "' %24.17e (r(p)) `", "df": "' %4.0f (r(df)) `", "df_r": "' %6.0f (r(df_r)) "}"
file write `fh' _n "}" _n
file close `fh'
