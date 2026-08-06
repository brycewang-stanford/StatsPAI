* tests/stata_parity/66_spatial_gmm.do
*
* Module 66: spatial GMM / 2SLS.
*   StatsPAI:  sp.sar_gmm(w_lags=1), sp.sem_gmm
*   R:         spatialreg::stsls(W2X=FALSE), spatialreg::GMerrorsar
*   Stata:     audited Stata/Mata GS2SLS bridge                   <-- this file
*
* Why a hand-built bridge rather than `spregress, gs2sls`. The two are not
* the same estimator on this fixture, and the difference is the instrument
* set, which is the whole content of the `W2X` argument:
*
*   spatialreg::stsls(W2X=FALSE)  instruments Wy with [X, WX]
*   spregress, gs2sls             also uses the second-order lags [X, WX, W2X]
*
* Measured 2026-08-06, `spregress y x1 x2, gs2sls dvarlag(W)` lands 1.5e-2
* from the R reference on the intercept and 1.8e-3 on x1 -- a real estimator
* difference, not noise. Rather than record that as an incompatibility, this
* script builds the documented estimator directly: form Wy and WX from the
* same lattice, then run 2SLS with exactly the first-order instruments. That
* reproduces `stsls(W2X=FALSE)` to machine precision *including the standard
* errors*, because `ivregress 2sls, small` uses the same n-k divisor stsls
* does:
*
*   sar_gmm_const   py 0.974364447786435    Stata 0.9743644477864348   ~1e-16
*   sar_gmm_x1      py 0.7344387161194063   Stata 0.7344387161194058   ~1e-16
*   sar_gmm_rho     py 0.07581347788845566  Stata 0.0758134778884556   ~1e-16
*   se(sar_gmm_x1)  py 0.08631191358840778  Stata 0.0863119135884078   ~1e-16
*
* This follows the same audited-bridge precedent as modules 08 (DML PLR),
* 31 (DFL), 32 (RIF), 53 (CR2) and 54/56 (multiway cluster): where the
* canonical Stata command implements a different convention, the do-file
* implements the *documented algorithm* and says so, rather than passing off
* a near-miss as parity.
*
* SEM-GMM is not bridged. `sem_gmm_*` pins spatialreg::GMerrorsar, the
* Kelejian-Prucha nonlinear GM estimator for lambda followed by FGLS.
* `spregress y x1 x2, gs2sls errorlag(W)` is close but not like-for-like --
* measured 4.5e-5 on the intercept and 5.5e-5 on x1, against this module's
* registered 1e-6 budget -- so those four statistics stay py<->R rows and the
* spregress numbers are recorded in the extras for the record rather than
* joined. Implementing the KP moment system in Mata is the promotion path.
*
* Tolerance: rel < 1e-6 on the SAR-GMM coefficients and standard errors.

version 18
clear all

do _common.do
stata_parity_init, module(66_spatial_gmm)
stata_parity_open, module(66_spatial_gmm)

import delimited "${STATA_PARITY_DATA}/66_spatial_gmm.csv", clear case(preserve)
count
local n = r(N)

* ------------------------------------------------------------------
* Rebuild the R side's W (rook contiguity on the (grid_row, grid_col)
* lattice, row-standardised) and form the spatial lags it implies.
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
st_numscalar("wrowdev", max(abs(rowsum(Wmat) :- 1)))
st_store(., st_addvar("double", "Wy"),  Wmat * st_data(., "y"))
st_store(., st_addvar("double", "Wx1"), Wmat * st_data(., "x1"))
st_store(., st_addvar("double", "Wx2"), Wmat * st_data(., "x2"))
end

if scalar(wrowdev) > 1e-12 {
    display as error "rebuilt W is not row-standardised; max |rowsum-1| = " scalar(wrowdev)
    exit 459
}

* ------------------------------------------------------------------
* SAR-2SLS: y = rho*Wy + X*beta + e, instrumenting Wy with WX.
* `small` gives the n-k divisor that stsls reports.
* ------------------------------------------------------------------
ivregress 2sls y x1 x2 (Wy = Wx1 Wx2), small

foreach term in _cons x1 x2 Wy {
    local bv = _b[`term']
    local sv = _se[`term']
    local lo = `bv' - ${STATA_PARITY_Z95} * `sv'
    local hi = `bv' + ${STATA_PARITY_Z95} * `sv'
    if "`term'" == "_cons" {
        local stat "sar_gmm_const"
    }
    else if "`term'" == "Wy" {
        local stat "sar_gmm_rho"
    }
    else {
        local stat "sar_gmm_`term'"
    }
    stata_parity_row, stat("`stat'") est(`bv') std(`sv') cilo(`lo') cihi(`hi') nob(`n')
}

* ------------------------------------------------------------------
* SEM-GMM: recorded, not joined. See the header.
* ------------------------------------------------------------------
gen long id = _n
spset id
mata:
ids = st_data(., "id")
end
spmatrix spfrommata Wsp = Wmat ids, replace
capture qui spregress y x1 x2, gs2sls errorlag(Wsp)
if _rc == 0 {
    matrix B2 = e(b)
    stata_parity_extra_num, key(spregress_gs2sls_errorlag_const) val(`=B2[1,"y:_cons"]')
    stata_parity_extra_num, key(spregress_gs2sls_errorlag_x1)    val(`=B2[1,"y:x1"]')
    stata_parity_extra_num, key(spregress_gs2sls_errorlag_x2)    val(`=B2[1,"y:x2"]')
    stata_parity_extra_num, key(spregress_gs2sls_errorlag_lambda) val(`=B2[1,"Wsp:e.y"]')
}

stata_parity_extra, key(stata_command) val("ivregress 2sls y x1 x2 (Wy = Wx1 Wx2), small  -- Wy/Wx built from the same row-standardised rook W")
stata_parity_extra, key(stata_bridge_status) val("audited Stata/Mata GS2SLS algorithm bridge, materialized 2026-08-06 with licensed Stata 18")
stata_parity_extra, key(instrument_set) val("first-order lags only [X, WX], matching spatialreg::stsls(W2X=FALSE); spregress gs2sls adds W2X and lands 1.5e-2 away on the intercept")
stata_parity_extra, key(sem_gmm_not_joined) val("sem_gmm_* pins spatialreg::GMerrorsar (Kelejian-Prucha nonlinear GM then FGLS); spregress gs2sls errorlag is 4.5e-5 / 5.5e-5 away against a 1e-6 budget, so its numbers are recorded above as extras rather than joined. Implementing the KP moment system in Mata is the promotion path.")

stata_parity_close, module(66_spatial_gmm)
