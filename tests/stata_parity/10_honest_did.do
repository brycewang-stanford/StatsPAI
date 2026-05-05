* tests/stata_parity/10_honest_did.do
*
* Module 10: Honest DiD smoothness sensitivity bounds.
*   StatsPAI:  sp.honest_did(method="FLCI")
*   R:         HonestDiD::createSensitivityResults(method="FLCI")
*   Stata:     honestdid (Pedro Sant'Anna's port)
*
* Mirrors the hand-crafted event study in 10_honest_did.{py,R}:
*   pre  rel-time {-3,-2,-1} : att = (0.01, -0.02, 0.0), SE = 0.05
*   post rel-time { 0, 1, 2} : att = (0.5,   0.4,   0.3), SE = 0.10
*
* Tolerance: abs < 0.05 on CI bounds.

version 18
clear all

do _common.do
stata_parity_init, module(10_honest_did)
stata_parity_open, module(10_honest_did)

* Build hand-crafted beta and Sigma matrices.
matrix b = (0.01, -0.02, 0.0, 0.5, 0.4, 0.3)
matrix V = J(6, 6, 0)
matrix V[1,1] = 0.05^2
matrix V[2,2] = 0.05^2
matrix V[3,3] = 0.05^2
matrix V[4,4] = 0.10^2
matrix V[5,5] = 0.10^2
matrix V[6,6] = 0.10^2
matrix colnames b = pre3 pre2 pre1 post0 post1 post2
matrix colnames V = pre3 pre2 pre1 post0 post1 post2
matrix rownames V = pre3 pre2 pre1 post0 post1 post2

* Pretend we have a model "stored" so honestdid can pull b/V. Easiest
* path: use honestdid's bmat/vcovmat options to feed matrices directly.
* Per honestdid help: honestdid, b(matname) vcov(matname) numpre(#)
*    mvec(numlist) [delta(sd) coefplot ...]

local mvec "0 0.05 0.1 0.2 0.5"

honestdid, b(b) vcov(V) numpre(3) mvec(`mvec') delta(sd)

* honestdid stashes its CI matrix in the Mata struct named in
* s(HonestEventStudy). The CI matrix is K x 3: column 1 = M (with .
* for the "Original" row), columns 2/3 = lb/ub.
local hes "`s(HonestEventStudy)'"
mata: st_matrix("CIMAT", `hes'.CI)

local n_rows = rowsof(CIMAT)
local n_pseudo = 1000
foreach m of local mvec {
    forvalues i = 1/`n_rows' {
        local mrow = CIMAT[`i', 1]
        if `mrow' >= . continue
        if abs(`mrow' - `m') < 1e-9 {
            local lb = CIMAT[`i', 2]
            local ub = CIMAT[`i', 3]
            stata_parity_row, stat(ci_lower_M_`m') est(`lb') nob(`n_pseudo')
            stata_parity_row, stat(ci_upper_M_`m') est(`ub') nob(`n_pseudo')
            continue, break
        }
    }
}

stata_parity_extra, key(method) val(FLCI)
stata_parity_extra, key(alpha)  val(0.05)
stata_parity_extra, key(stata_command) val("honestdid, b(b) vcov(V) numpre(3) mvec(...) delta(sd)")

stata_parity_close, module(10_honest_did)
