* tests/stata_parity/15_hdfe_cluster.do
*
* Module 15: HDFE 2-way FE with cluster-robust SE.
*   StatsPAI:  sp.fast.feols(... vcov="cluster", cluster="firm")
*   R:         fixest::feols(... | firm + year, cluster = ~firm)
*   Stata:     reghdfe ..., absorb(firm year) vce(cluster firm)
*
* Tolerance: rel < 1e-3 on point estimate, < 5e-2 on SE (ssc convention).

version 18
clear all

do _common.do
stata_parity_init, module(15_hdfe_cluster)
stata_parity_open, module(15_hdfe_cluster)

import delimited "${STATA_PARITY_DATA}/15_hdfe_cluster.csv", clear case(preserve)

reghdfe y x1 x2, absorb(firm year) vce(cluster firm)

local n = e(N)
matrix B = e(b)
matrix V = e(V)
foreach v in x1 x2 {
    local bv = B[1, "`v'"]
    local sv = sqrt(V["`v'", "`v'"])
    local lo = `bv' - ${STATA_PARITY_Z95} * `sv'
    local hi = `bv' + ${STATA_PARITY_Z95} * `sv'
    stata_parity_row, stat(beta_`v') est(`bv') std(`sv') cilo(`lo') cihi(`hi') nob(`n')
}

stata_parity_extra, key(formula) val("y ~ x1 + x2 | firm + year")
stata_parity_extra, key(vcov) val(cluster)
stata_parity_extra, key(cluster_var) val(firm)
stata_parity_extra, key(stata_command) val("reghdfe y x1 x2, absorb(firm year) vce(cluster firm)")

stata_parity_close, module(15_hdfe_cluster)
