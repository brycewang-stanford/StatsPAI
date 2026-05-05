* tests/stata_parity/21_honest_relmags.do
*
* Module 21: Honest-DiD relative-magnitudes bounds.
*   StatsPAI:  sp.honest_did(method="C-LF", restriction="relative_magnitudes")
*   R:         HonestDiD::createSensitivityResults_relativeMagnitudes
*   Stata:     honestdid (delta(rm))
*
* Tolerance: abs < 0.10 (cone-solver vs analytic).

version 18
clear all

do _common.do
stata_parity_init, module(21_honest_relmags)
stata_parity_open, module(21_honest_relmags)

matrix b = (0.01, -0.02, 0.0, 0.5, 0.4, 0.3)
matrix V = J(6, 6, 0)
forvalues i=1/3 {
    matrix V[`i',`i'] = 0.05^2
}
forvalues i=4/6 {
    matrix V[`i',`i'] = 0.10^2
}
matrix colnames b = pre3 pre2 pre1 post0 post1 post2
matrix colnames V = pre3 pre2 pre1 post0 post1 post2
matrix rownames V = pre3 pre2 pre1 post0 post1 post2

local mbarvec "0 0.5 1 1.5 2"
honestdid, b(b) vcov(V) numpre(3) mvec(`mbarvec') delta(rm)

local hes "`s(HonestEventStudy)'"
mata: st_matrix("CIMAT", `hes'.CI)

local n_rows = rowsof(CIMAT)
local n_pseudo = 1000
foreach m of local mbarvec {
    forvalues i = 1/`n_rows' {
        local mrow = CIMAT[`i', 1]
        if `mrow' >= . continue
        if abs(`mrow' - `m') < 1e-9 {
            local lb = CIMAT[`i', 2]
            local ub = CIMAT[`i', 3]
            stata_parity_row, stat(ci_lower_Mbar_`m') est(`lb') nob(`n_pseudo')
            stata_parity_row, stat(ci_upper_Mbar_`m') est(`ub') nob(`n_pseudo')
            continue, break
        }
    }
}

stata_parity_extra, key(method) val(relative_magnitudes)
stata_parity_extra, key(alpha) val(0.05)
stata_parity_extra, key(stata_command) val("honestdid, b(b) vcov(V) numpre(3) mvec(...) delta(rm)")

stata_parity_close, module(21_honest_relmags)
