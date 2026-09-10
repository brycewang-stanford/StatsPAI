* tests/stata_parity/88_rdbwselect.do
*
* Module 88: RD bandwidth selection.
*   StatsPAI:  sp.rdbwselect
*   R:         rdrobust::rdbwselect
*   Stata:     rdbwselect
*
* Both references are maintained by the Calonico-Cattaneo-Farrell-
* Titiunik group, so neither side is a bridge: this is a canonical
* dual reference in the sense of CLAUDE.md §5.1.
*
* Mirrors the sweep in ../r_parity/88_rdbwselect.{py,R}: all ten
* selectors at the p = 1 default, then polynomial order, kernel,
* covariates, clustering and derivative order one cell at a time.
*
* Tolerance: rel < 1e-6 on every bandwidth.

version 18
clear all

do _common.do
stata_parity_init, module(88_rdbwselect)
stata_parity_open, module(88_rdbwselect)

import delimited "${STATA_PARITY_DATA}/88_rdbwselect.csv", clear case(preserve)

local n = _N

* Emit the four bandwidth quantities from the last rdbwselect fit.
* e(mat_h) and e(mat_b) are 1 x 2 matrices ordered (left, right).
capture program drop emit_bws
program define emit_bws
    args prefix nobs
    tempname H B
    matrix `H' = e(mat_h)
    matrix `B' = e(mat_b)
    stata_parity_row, stat(`prefix'_h_left)  est(`=`H'[1,1]') nob(`nobs')
    stata_parity_row, stat(`prefix'_h_right) est(`=`H'[1,2]') nob(`nobs')
    stata_parity_row, stat(`prefix'_b_left)  est(`=`B'[1,1]') nob(`nobs')
    stata_parity_row, stat(`prefix'_b_right) est(`=`B'[1,2]') nob(`nobs')
end

* --- Block A: all ten selectors at the p = 1 default. -----------------
*
* certwo is emitted by the Python and R sides but NOT here, and the
* omission is the reference implementation's, not ours: Stata
* rdbwselect 10.0.0 (2025-06-30) exits with a conformability error
* (r(3200)) on bwselect(certwo). It is not this fixture -- the same
* call fails identically on rdrobust_senate.dta, the dataset the
* rdrobust authors ship with the package, while the structurally
* identical msetwo succeeds on both. R's rdbwselect computes certwo
* without complaint and StatsPAI reproduces it to 1e-12, so the cell
* keeps a canonical reference; it just has one language in it instead
* of two. CLAUDE.md §5.1 case 3: do not copy a reference bug, and do
* not drop the cell to make the row uniform.
foreach m in mserd msetwo msesum msecomb1 msecomb2 cerrd cersum cercomb1 cercomb2 {
    rdbwselect y x, c(0) bwselect(`m')
    emit_bws `m' `n'
}

* Guard: if a future rdrobust release fixes certwo, this assertion fires
* and tells us to restore the cell rather than leaving a stale exclusion
* in place forever.
capture rdbwselect y x, c(0) bwselect(certwo)
if _rc == 0 {
    display as error "certwo now works in Stata rdbwselect -- restore it to Block A"
    exit 459
}
if _rc != 3200 {
    display as error "certwo failed with r(" _rc ") not the documented r(3200)"
    exit 459
}

* --- Block B: polynomial order. ---------------------------------------
foreach p in 2 3 {
    rdbwselect y x, c(0) p(`p')
    emit_bws mserd_p`p' `n'
}

* --- Block C: kernel. --------------------------------------------------
foreach k in uniform epanechnikov {
    rdbwselect y x, c(0) kernel(`k')
    emit_bws mserd_`k' `n'
}

* --- Block D: covariate-adjusted selection. ---------------------------
rdbwselect y x, c(0) covs(z1)
emit_bws mserd_covs `n'

* --- Block E: clustered selection. ------------------------------------
* Stata spells clustering through vce(), and the two spellings are not
* interchangeable: vce(nncluster cl) keeps nearest-neighbour residuals
* and lands 0.65% away from R, while vce(cluster cl) switches to the
* hc-family residuals R's own cluster= path uses and agrees with it to
* 5e-9. The matching option is the one selected here; the other is a
* different estimator, not a looser one.
rdbwselect y x, c(0) vce(cluster cl)
emit_bws mserd_cluster `n'

* --- Block F: derivative order (regression kink). ---------------------
rdbwselect y x, c(0) deriv(1) p(2)
emit_bws mserd_deriv1 `n'

stata_parity_extra, key(stata_command) val("rdbwselect y x, c(0) bwselect(mserd)")
stata_parity_extra, key(kernel) val(triangular)
stata_parity_extra, key(reference_role) val("canonical (Cattaneo group maintains R and Stata)")

stata_parity_close, module(88_rdbwselect)
