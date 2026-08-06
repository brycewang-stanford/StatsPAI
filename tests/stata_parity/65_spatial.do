* tests/stata_parity/65_spatial.do
*
* Module 65: spatial ML -- SAR, SEM, and the Spatial Durbin Model.
*   StatsPAI:  sp.sar / sp.sem / sp.sdm
*   R:         spatialreg::lagsarlm / errorsarlm / lagsarlm(Durbin=TRUE)
*   Stata:     spregress, ml dvarlag() / errorlag() / ivarlag()   <-- this file
*
* This module's committed skip reason used to say that "Stata's spregress is
* the natural analog but follows a distinct ML/estimand convention". That was
* inherited, not measured. It is wrong: on the same 144-cell lattice
* spregress reproduces every spatialreg coefficient to 5e-8 or better --
* ordinary ML optimiser convergence, not a convention difference.
*
*   parameter   StatsPAI / spatialreg     Stata spregress        rel
*   sar_x1      0.8017316893681659        0.8017316950947706     7.1e-09
*   sar_rho     0.5486991846577806        0.5486991609294254     4.3e-08
*   sem_lambda  0.5983424828615727        0.5983424699246154     2.2e-08
*   sdm_W_x1   -0.1948291288077453       -0.1948291305414327     8.9e-09
*
* The weights matrix is the reason this works at all. `spmatrix create
* contiguity` needs a linked shapefile, which this fixture does not carry, so
* the CSV's (grid_row, grid_col) columns are used to rebuild the *identical*
* rook-contiguity, row-standardised W in Mata and load it with
* `spmatrix spfrommata`. Building W from the same lattice definition the R
* side uses is what makes this a parity check rather than a comparison of two
* different neighbourhoods.
*
* Point estimates only. spregress and spatialreg report different asymptotic
* variance estimators for the same fit -- measured 2026-08-06, the SEs sit
* 4.6e-4 (sar_const) to 2.7e-2 (sem_lambda) apart, against this module's
* registered rel_se budget of 1e-6:
*
*   sar_x1      py 0.0989968339   Stata 0.0982459312   rel 7.6e-03
*   sar_rho     py 0.0730732023   Stata 0.0731733450   rel 1.4e-03
*   sem_lambda  py 0.0799400480   Stata 0.0821124234   rel 2.7e-02
*
* so `se = .` is emitted on every row and no SE column is joined. Recovering
* spregress's information-matrix convention would be a variance-estimator
* study, not an estimator check.
*
* Tolerance: rel < 1e-6 on the point estimates.

version 18
clear all

do _common.do
stata_parity_init, module(65_spatial)
stata_parity_open, module(65_spatial)

import delimited "${STATA_PARITY_DATA}/65_spatial.csv", clear case(preserve)
count
local n = r(N)

gen long id = _n
spset id

* ------------------------------------------------------------------
* Rebuild the R side's W: rook contiguity on the (grid_row, grid_col)
* lattice -- neighbours are cells at Manhattan distance 1 -- then divide
* each row by its own neighbour count (spdep style="W").
* ------------------------------------------------------------------
mata:
gr = st_data(., "grid_row")
gc = st_data(., "grid_col")
n  = rows(gr)
Wmat = J(n, n, 0)
for (i = 1; i <= n; i++) {
    for (j = 1; j <= n; j++) {
        if (i != j) {
            if (abs(gr[i] - gr[j]) + abs(gc[i] - gc[j]) == 1) {
                Wmat[i, j] = 1
            }
        }
    }
}
for (i = 1; i <= n; i++) {
    s = sum(Wmat[i, .])
    if (s > 0) {
        Wmat[i, .] = Wmat[i, .] :/ s
    }
}
ids = st_data(., "id")
st_numscalar("wrowdev", max(abs(rowsum(Wmat) :- 1)))
end

* Guard the rebuilt lattice: every row of a row-standardised W must sum to 1,
* so a lattice built from the wrong columns fails loudly here rather than
* silently grading a different neighbourhood. The check is written in Stata
* rather than Mata because a Mata `if {}` block written inline in a do-file
* swallows the following `end`.
if scalar(wrowdev) > 1e-12 {
    display as error "rebuilt W is not row-standardised; max |rowsum-1| = " scalar(wrowdev)
    exit 459
}

spmatrix spfrommata Wsp = Wmat ids, replace

* ------------------------------------------------------------------
* SAR: y = rho*W*y + X*beta + e
* ------------------------------------------------------------------
qui spregress y x1 x2, ml dvarlag(Wsp)
matrix B = e(b)
stata_parity_row, stat(sar_const) est(`=B[1,"y:_cons"]') nob(`n')
stata_parity_row, stat(sar_x1)    est(`=B[1,"y:x1"]')    nob(`n')
stata_parity_row, stat(sar_x2)    est(`=B[1,"y:x2"]')    nob(`n')
stata_parity_row, stat(sar_rho)   est(`=B[1,"Wsp:y"]')   nob(`n')

* ------------------------------------------------------------------
* SEM: y = X*beta + u, u = lambda*W*u + e
* ------------------------------------------------------------------
qui spregress y x1 x2, ml errorlag(Wsp)
matrix B2 = e(b)
stata_parity_row, stat(sem_const)  est(`=B2[1,"y:_cons"]')   nob(`n')
stata_parity_row, stat(sem_x1)     est(`=B2[1,"y:x1"]')      nob(`n')
stata_parity_row, stat(sem_x2)     est(`=B2[1,"y:x2"]')      nob(`n')
stata_parity_row, stat(sem_lambda) est(`=B2[1,"Wsp:e.y"]')   nob(`n')

* ------------------------------------------------------------------
* SDM: SAR plus spatially lagged covariates (Durbin=TRUE in spatialreg)
* ------------------------------------------------------------------
qui spregress y x1 x2, ml dvarlag(Wsp) ivarlag(Wsp: x1 x2)
matrix B3 = e(b)
stata_parity_row, stat(sdm_const) est(`=B3[1,"y:_cons"]')  nob(`n')
stata_parity_row, stat(sdm_x1)    est(`=B3[1,"y:x1"]')     nob(`n')
stata_parity_row, stat(sdm_x2)    est(`=B3[1,"y:x2"]')     nob(`n')
stata_parity_row, stat(sdm_W_x1)  est(`=B3[1,"Wsp:x1"]')   nob(`n')
stata_parity_row, stat(sdm_W_x2)  est(`=B3[1,"Wsp:x2"]')   nob(`n')
stata_parity_row, stat(sdm_rho)   est(`=B3[1,"Wsp:y"]')    nob(`n')

stata_parity_extra, key(stata_command_sar) val("spregress y x1 x2, ml dvarlag(W)")
stata_parity_extra, key(stata_command_sem) val("spregress y x1 x2, ml errorlag(W)")
stata_parity_extra, key(stata_command_sdm) val("spregress y x1 x2, ml dvarlag(W) ivarlag(W: x1 x2)")
stata_parity_extra, key(weights) val("rook contiguity on (grid_row, grid_col), row-standardised; rebuilt in Mata and loaded with spmatrix spfrommata because spmatrix create contiguity requires a linked shapefile")
stata_parity_extra, key(se_convention) val("point estimates only: spregress and spatialreg report different asymptotic variance estimators for the same fit (4.6e-4 to 2.7e-2 apart), against a registered rel_se budget of 1e-6")
stata_parity_extra, key(prior_skip_reason_corrected) val("the pre-2026-08-06 skip reason claimed spregress follows a distinct ML/estimand convention; measured, it reproduces every spatialreg coefficient to 5e-8")
stata_parity_extra, key(stata_bridge_status) val("materialized 2026-08-06 with licensed Stata 18")

stata_parity_close, module(65_spatial)
