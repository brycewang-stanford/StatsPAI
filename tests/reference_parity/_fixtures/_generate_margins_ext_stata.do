* ---------------------------------------------------------------------------
* Stata reference for tests/reference_parity/test_margins_ext_parity.py
*
* Requires: Stata 18 (official commands only; no ado dir needed).
* Run:      stata-mp -b do _generate_margins_ext_stata.do   (from this directory)
*
* What this fixture pins (sp.margins, 1.32 extensions):
*  - margins, dydx() on factor variables (discrete change vs the base level)
*    after regress / logit / probit
*  - dy/dx through c.x##c.x (StatsPAI: x + I(x**2)), AME and atmeans
*  - factor x continuous interaction (probit i.g##c.z)
*  - analytic weights [aw=w]: margins averages with the fit's weights
*  - count models with exposure(): predict() = number of events
* ---------------------------------------------------------------------------
version 18
clear all
set type double
import delimited "margins_ext_data.csv", clear asdouble

capture file close fh
file open fh using "margins_ext_stata.json", write replace text
file write fh "{" _n
capture program drop _num
program define _num
    args key value
    file write fh `"  "`key'": "' %21.16e (`value') `","' _n
end
capture program drop _mat
program define _mat
    args key mname
    tempname M
    matrix `M' = `mname'
    local nr = rowsof(`M')
    local nc = colsof(`M')
    local cn : colfullnames `M'
    local rn : rownames `M'
    file write fh `"  "`key'": {"cols": ["'
    local i 0
    foreach c of local cn {
        local ++i
        file write fh `"""' `"`c'"' `"""'
        if `i' < `nc' file write fh ", "
    }
    file write fh `"], "rows": ["'
    local i 0
    foreach r of local rn {
        local ++i
        file write fh `"""' `"`r'"' `"""'
        if `i' < `nr' file write fh ", "
    }
    file write fh `"], "v": ["'
    forvalues r = 1/`nr' {
        file write fh "["
        forvalues c = 1/`nc' {
            if missing(`M'[`r', `c']) file write fh "null"
            else file write fh %21.16e (`M'[`r', `c'])
            if `c' < `nc' file write fh ", "
        }
        file write fh "]"
        if `r' < `nr' file write fh ", "
    }
    file write fh "]}," _n
end

* 1. regress, quadratic + factor
quietly regress yl c.x##c.x i.g z
_mat ols_quad__b e(b)
_num ols_quad__df_r e(df_r)
quietly margins, dydx(x g z)
_mat ols_quad__dydx r(table)
quietly margins, dydx(x g z) atmeans
_mat ols_quad__dydx_atmeans r(table)
quietly margins, dydx(x) at(x=(-1 0 1))
_mat ols_quad__dydx_x_at_x r(table)

* 2. regress, factor x continuous, analytic weights
quietly regress yl i.g##c.x z [aw=w]
_mat ols_aw__b e(b)
quietly margins, dydx(x g z)
_mat ols_aw__dydx r(table)
quietly margins, at(x=(0 1))
_mat ols_aw__at_x r(table)
quietly margins r.g
_mat ols_aw__r_g r(table)

* 3. logit, quadratic + factor
quietly logit yb c.x##c.x i.g z, nrtolerance(1e-14) tolerance(1e-14) ltolerance(1e-14)
_mat logit_quad__b e(b)
quietly margins, dydx(x g z)
_mat logit_quad__dydx r(table)
quietly margins, dydx(x g z) atmeans
_mat logit_quad__dydx_atmeans r(table)

* 4. probit, factor x continuous interaction
quietly probit yb x i.g##c.z, nrtolerance(1e-14) tolerance(1e-14) ltolerance(1e-14)
_mat probit_gz__b e(b)
quietly margins, dydx(x g z)
_mat probit_gz__dydx r(table)

* 5. count models with exposure
quietly poisson yc x z i.g, exposure(expo) nrtolerance(1e-14) tolerance(1e-14) ltolerance(1e-14)
_mat poisson_expo__b e(b)
quietly margins, dydx(x z g)
_mat poisson_expo__dydx r(table)
quietly margins, at(x=(0 1))
_mat poisson_expo__at_x r(table)
quietly glm yc x z, family(poisson) link(log) exposure(expo)
_mat glm_expo__b e(b)
quietly margins, dydx(x z)
_mat glm_expo__dydx r(table)

file write fh `"  "_meta": {"' _n
file write fh `"    "stata_version": "' `"""' "`c(stata_version)'" `"""' `","' _n
file write fh `"    "born_date": "' `"""' "`c(born_date)'" `"""' `","' _n
file write fh `"    "generated": "' `"""' "`c(current_date)'" `"""' `","' _n
file write fh `"    "data": "margins_ext_data.csv""' _n
file write fh `"  }"' _n
file write fh "}" _n
file close fh
display "wrote margins_ext_stata.json"
