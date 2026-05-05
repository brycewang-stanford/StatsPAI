* tests/stata_parity/14_ols_cluster.do
*
* Module 14: OLS + cluster-robust (CR1).
*   StatsPAI:  sp.regress(robust="cluster", cluster_var="countyreal")
*   R:         lm + sandwich::vcovCL(type="HC1")
*   Stata:     regress, vce(cluster countyreal)
*
* Tolerance: rel < 1e-3 (CR1 small-sample factor matches by design).

version 18
clear all

do _common.do
stata_parity_init, module(14_ols_cluster)
stata_parity_open, module(14_ols_cluster)

import delimited "${STATA_PARITY_DATA}/14_ols_cluster.csv", clear case(preserve)

regress lemp treat year, vce(cluster countyreal)

local n = e(N)
matrix B = e(b)
matrix V = e(V)
local vars : colnames B
foreach v of local vars {
    local bv = B[1, "`v'"]
    local sv = sqrt(V["`v'", "`v'"])
    local lo = `bv' - ${STATA_PARITY_Z95} * `sv'
    local hi = `bv' + ${STATA_PARITY_Z95} * `sv'
    if "`v'" == "_cons" local stat "beta_(Intercept)"
    else                local stat "beta_`v'"
    stata_parity_row, stat("`stat'") est(`bv') std(`sv') cilo(`lo') cihi(`hi') nob(`n')
}

stata_parity_extra, key(formula) val("lemp ~ treat + year")
stata_parity_extra, key(vcov) val("cluster (CR1)")
stata_parity_extra, key(cluster_var) val(countyreal)
stata_parity_extra, key(stata_command) val("regress lemp treat year, vce(cluster countyreal)")

stata_parity_close, module(14_ols_cluster)
