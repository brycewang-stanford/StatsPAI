* tests/stata_parity/89_rdms.do
*
* Module 89: multi-score / geographic RD at boundary points.
*   StatsPAI:  sp.rdms
*   R:         rdmulti::rdms
*   Stata:     rdms (rdpackages, GitHub-only)
*
* Both references are Cattaneo-group code, so neither side is a bridge.
*
* rdmulti is NOT on SSC -- `ssc describe rdmulti` returns r(601), tested
* rather than assumed (CLAUDE.md §5.1). It is installed from the
* rdpackages GitHub mirror into a local, gitignored ado path so the
* user's PLUS directory is left alone, the same arrangement 86_fect uses.
*
* Tolerance: rel < 1e-6.

version 18
clear all

do _common.do
stata_parity_init, module(89_rdms)
stata_parity_open, module(89_rdms)

* ---- local ado path for the GitHub-only reference ------------------------
local ado_rdmulti "`c(pwd)'/_ado_rdmulti"
capture mkdir "`ado_rdmulti'"
adopath + "`ado_rdmulti'"
capture which rdms
if _rc != 0 {
    local plus_saved : sysdir PLUS
    sysdir set PLUS "`ado_rdmulti'"
    net install rdmulti, ///
        from("https://raw.githubusercontent.com/rdpackages/rdmulti/master/stata") replace
    sysdir set PLUS "`plus_saved'"
}
which rdms

import delimited "${STATA_PARITY_DATA}/89_rdms.csv", clear case(preserve)

local n = _N

* Stata rdms takes the outcome, the two scores and the treatment
* indicator positionally (`rdms Y X X2 Z`), and the boundary points as a
* pair of *variables* named in cvar(), not as a matrix. The three (C, C2)
* pairs therefore live in the first three rows of two helper columns;
* everything past row 3 is missing, which is how the ado infers the
* number of cutoffs.
gen double Cvar  = .
gen double C2var = .
replace Cvar  = 0     in 1
replace C2var = -0.5  in 1
replace Cvar  = 0     in 2
replace C2var = 0     in 2
replace Cvar  = 0     in 3
replace C2var = 0.5   in 3

rdms y x1 x2 z, cvar(Cvar C2var)

* Return-name trap: Stata's e(B) is the *bias bandwidth* b, while R's
* rdms$B is the *bias-corrected estimate*. Same letter, different object.
* The estimates are in e(b) / e(coefs), the standard errors in
* e(SE_rb) / e(SE_cl), and the effective sample sizes in e(sampsis)
* (R calls that one Nh).
tempname bc coefs serb secl H Nh
matrix `bc'    = e(b)
matrix `coefs' = e(coefs)
matrix `serb'  = e(SE_rb)
matrix `secl'  = e(SE_cl)
matrix `H'     = e(H)
matrix `Nh'    = e(sampsis)

local tags "bm05 bp00 bp05"
forvalues i = 1/3 {
    local tag : word `i' of `tags'
    local v_bc    = `bc'[1, `i']
    local v_serb  = `serb'[1, `i']
    local v_coefs = `coefs'[1, `i']
    local v_secl  = `secl'[1, `i']
    local v_h     = `H'[1, `i']
    local v_nl    = `Nh'[1, `i']
    local v_nr    = `Nh'[2, `i']
    stata_parity_row, stat(`tag'_biascorrected_est) est(`v_bc') std(`v_serb') nob(`n')
    stata_parity_row, stat(`tag'_conventional_est) est(`v_coefs') std(`v_secl') nob(`n')
    stata_parity_row, stat(`tag'_bandwidth_h) est(`v_h') nob(`n')
    stata_parity_row, stat(`tag'_n_eff_left) est(`v_nl') nob(`n')
    stata_parity_row, stat(`tag'_n_eff_right) est(`v_nr') nob(`n')
}

stata_parity_extra, key(stata_command) val("rdms y x1 x2 z, cvar(Cvar C2var)")
stata_parity_extra, key(install_note) ///
    val("rdmulti is not on SSC (ssc describe rdmulti: r(601)); net install from raw.githubusercontent.com/rdpackages/rdmulti/master/stata into tests/stata_parity/_ado_rdmulti")
stata_parity_extra, key(reference_role) val("canonical (Cattaneo group maintains R and Stata)")

stata_parity_close, module(89_rdms)
