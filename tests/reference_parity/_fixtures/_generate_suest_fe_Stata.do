* sp.suest with factor regressors, absorbed fixed effects and two-way
* clustering, on suest_fe_data.csv.
*  A. Stata suest after regress with i. dummies: vce(robust), vce(cluster r),
*     vce(cluster m).
*  B. The stacked fully interacted regression (one row per equation and
*     observation, every regressor and fixed effect interacted with the
*     equation) in reghdfe 6.13.1, vce(cluster r m), vce(cluster r) and
*     vce(cluster id) (id = the original row: suest's robust variance), and
*     in regress, vce(cluster r m) (Stata 18, per-term G/(G-1)). The
*     standard deviations reghdfe standardizes by are written as sd_<case>.
*   python tests/reference_parity/_fixtures/_generate_suest_fe_data.py
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_suest_fe_Stata.do
* reghdfe / ftools / require come from SSC; set PLUS to a private ado
* directory holding them before running.
version 18
clear all
set type double
import delimited using "tests/reference_parity/_fixtures/suest_fe_data.csv", clear asdouble
tempname fh
file open `fh' using "tests/reference_parity/_fixtures/suest_fe_Stata.json", write replace
file write `fh' "{" _n

capture program drop wcase
program define wcase
    args fh key names
    matrix b = e(b)
    matrix V = e(V)
    file write `fh' `"  "`key'": {"b": ["'
    local first 1
    foreach r of local names {
        if !`first' file write `fh' ", "
        local first 0
        file write `fh' %24.16e (b[1, "`r'"])
    }
    file write `fh' "], " _n `"    "V": ["'
    local first 1
    foreach r of local names {
        foreach c of local names {
            if !`first' file write `fh' ", "
            local first 0
            file write `fh' %24.16e (V["`r'", "`c'"])
        }
    }
    file write `fh' "]"
    foreach s in N df_m df_a N_clust N_clust1 N_clust2 rank df_r {
        if "`e(`s')'" != "" file write `fh' `", "`s'": "' %24.16e (e(`s'))
    }
    file write `fh' "}," _n
end

capture program drop wsd
program define wsd
    args fh key names
    file write `fh' `"  "`key'": ["'
    local first 1
    foreach v of local names {
        qui sum `v'
        if !`first' file write `fh' ", "
        local first 0
        file write `fh' %24.16e (r(sd))
    }
    file write `fh' "]," _n
end

* ---- A. suest after regress ---------------------------------------------
qui reg yr x w i.f
est store e1
qui reg yf x i.f i.g
est store e2
local A e1_mean:x e1_mean:w e1_mean:2.f e1_mean:3.f e1_mean:4.f e1_mean:_cons
local A `A' e2_mean:x e2_mean:2.f e2_mean:3.f e2_mean:4.f e2_mean:20.g
local A `A' e2_mean:30.g e2_mean:40.g e2_mean:50.g e2_mean:_cons
qui suest e1 e2, vce(robust)
wcase `fh' A_robust "`A'"
qui suest e1 e2, vce(cluster r)
wcase `fh' A_cluster_r "`A'"
qui suest e1 e2, vce(cluster m)
wcase `fh' A_cluster_m "`A'"
qui test [e1_mean]x = [e2_mean]x
file write `fh' `"  "A_cluster_m_test_x": {"chi2": "' %24.16e (r(chi2)) "}," _n

* ---- B. stacked regression ----------------------------------------------
gen long id = _n
expand 2
bysort id: gen byte eq = _n
gen double yy = cond(eq == 1, yr, yf)
drop if missing(yy)
foreach v in x w {
    gen double `v'_1 = `v' * (eq == 1)
}
gen double x_2 = x * (eq == 2)
forvalues k = 2/4 {
    gen double f`k'_1 = (f == `k') * (eq == 1)
    gen double f`k'_2 = (f == `k') * (eq == 2)
}
foreach k in 20 30 40 50 {
    gen double g`k'_2 = (g == `k') * (eq == 2)
}
* m absorbed in the second equation only; constant on the first
gen double me = cond(eq == 2, m, -1)

* B0: yr ~ x and yf ~ x, two-way: positive semi-definite, no fix-up
qui reghdfe yy x_1 x_2, absorb(eq) vce(cluster r m) tol(1e-14)
wcase `fh' B0_reghdfe_rm "x_1 x_2"

* B1: no fixed effects (equation intercepts absorbed), slopes only. The
* inclusion-exclusion matrix has negative eigenvalues here; reghdfe sets
* them to zero in its standardized-variable basis, regress in the original
* one.
local B1 x_1 w_1 f2_1 f3_1 f4_1 x_2 f2_2 f3_2 f4_2 g20_2 g30_2 g40_2 g50_2
qui reghdfe yy `B1', absorb(eq) vce(cluster r m) tol(1e-14)
wcase `fh' B1_reghdfe_rm "`B1'"
wsd `fh' sd_B1 "`B1'"
qui regress yy `B1' ibn.eq, noconstant vce(cluster r m)
wcase `fh' B1_regress_rm "`B1'"

* B2: yr ~ x + w | f  and  yf ~ x + i.g | f + m
local B2 x_1 w_1 x_2 g20_2 g30_2 g40_2 g50_2
qui reghdfe yy `B2', absorb(eq#f me) vce(cluster r m) tol(1e-14)
wcase `fh' B2_reghdfe_rm "`B2'"
wsd `fh' sd_B2 "`B2'"
qui reghdfe yy `B2', absorb(eq#f me) vce(cluster r) tol(1e-14)
wcase `fh' B2_reghdfe_r "`B2'"
qui reghdfe yy `B2', absorb(eq#f me) vce(cluster id) tol(1e-14)
wcase `fh' B2_reghdfe_id "`B2'"
file write `fh' `"  "end": 0"' _n "}" _n
file close `fh'
