* tests/stata_parity/54_twoway_cluster.do
*
* Module 54: Two-way cluster-robust OLS SE.
*   StatsPAI:  sp.twoway_cluster
*   R:         sandwich::vcovCL(cluster = ~ g1 + g2, type="HC1", cadjust=TRUE)
*   Stata:     audited Stata/Mata CGM bridge; reghdfe diagnostics recorded

version 18
clear all

do _common.do
stata_parity_init, module(54_twoway_cluster)
stata_parity_open, module(54_twoway_cluster)

import delimited "${STATA_PARITY_DATA}/54_twoway_cluster.csv", clear case(preserve)

reghdfe y x, noabsorb vce(cluster g1 g2)

matrix B = e(b)
matrix V_RE = e(V)
local n = e(N)

local b0 = B[1, "_cons"]
local s0_reghdfe = sqrt(V_RE["_cons", "_cons"])
local bx = B[1, "x"]
local sx_reghdfe = sqrt(V_RE["x", "x"])

mata:
real matrix sp_mw54_oneway_vcov(real matrix X, real colvector u, real matrix C)
{
    real scalar n, k, G, i, j, corr
    real matrix XtX_inv, scores, meat, U
    real rowvector sg
    real colvector keep, idx

    n = rows(X)
    k = cols(X)
    XtX_inv = invsym(cross(X, X))
    scores = X :* (u * J(1, k, 1))
    U = uniqrows(sort(C, 1..cols(C)))
    G = rows(U)
    meat = J(k, k, 0)
    for (i = 1; i <= G; i++) {
        keep = J(n, 1, 1)
        for (j = 1; j <= cols(C); j++) {
            keep = keep :& (C[, j] :== U[i, j])
        }
        idx = selectindex(keep)
        sg = colsum(scores[idx, .])
        meat = meat + sg' * sg
    }
    corr = G / (G - 1) * (n - 1) / (n - k)
    return(corr * XtX_inv * meat * XtX_inv)
}

real matrix sp_mw54_vcov()
{
    real matrix X, C1, C2, V
    real colvector y, u, b

    X = J(rows(st_data(., "x")), 1, 1), st_data(., "x")
    y = st_data(., "y")
    b = qrsolve(X, y)
    u = y :- X * b
    C1 = st_data(., "g1")
    C2 = st_data(., "g2")
    V = sp_mw54_oneway_vcov(X, u, C1) +
        sp_mw54_oneway_vcov(X, u, C2) -
        sp_mw54_oneway_vcov(X, u, (C1, C2))
    return(0.5 * (V + V'))
}

st_matrix("VBRIDGE", sp_mw54_vcov())
end

matrix colnames VBRIDGE = _cons x
matrix rownames VBRIDGE = _cons x

local s0 = sqrt(VBRIDGE["_cons", "_cons"])
local sx = sqrt(VBRIDGE["x", "x"])
local rel0 = abs(`s0_reghdfe' - `s0') / max(abs(`s0'), 1e-12)
local relx = abs(`sx_reghdfe' - `sx') / max(abs(`sx'), 1e-12)
local maxrel = max(`rel0', `relx')

stata_parity_row, stat(beta_(Intercept)) est(`b0') std(`s0') nob(`n')
stata_parity_row, stat(beta_x) est(`bx') std(`sx') nob(`n')
stata_parity_row, stat(reghdfe_beta_(Intercept)) est(`b0') std(`s0_reghdfe') nob(`n')
stata_parity_row, stat(reghdfe_beta_x) est(`bx') std(`sx_reghdfe') nob(`n')

stata_parity_extra, key(method) val("audited Stata/Mata two-way cluster bridge")
stata_parity_extra, key(stata_command) val("reghdfe y x, noabsorb vce(cluster g1 g2) for diagnostic SEs; headline SEs from Mata CGM bridge")
stata_parity_extra, key(stata_bridge_status) val("audited Stata/Mata algorithm bridge")
stata_parity_extra, key(stata_algorithm) val("Cameron-Gelbach-Miller V1+V2-V12 with per-component Liang-Zeger CR1 correction matching sandwich::vcovCL HC1/cadjust")
stata_parity_extra, key(stata_cluster_reference) val("reghdfe 6.12.3 multiway cluster retained as a diagnostic convention row")
stata_parity_extra_num, key(stata_reghdfe_max_rel_py_se_diff) val(`maxrel')

stata_parity_close, module(54_twoway_cluster)
