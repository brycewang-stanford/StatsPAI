* tests/stata_parity/06_rd.do
*
* Module 06: RD CCT bias-corrected.
*   StatsPAI:  sp.rdrobust
*   R:         rdrobust::rdrobust
*   Stata:     rdrobust (same algorithm, both R/Stata distributions
*              are written by Calonico-Cattaneo-Titiunik authors).
*
* Tolerance: rel < 1e-3 at default-h, < 1e-13 at forced common h.

version 18
clear all

do _common.do
stata_parity_init, module(06_rd)
stata_parity_open, module(06_rd)

import delimited "${STATA_PARITY_DATA}/06_rd.csv", clear case(preserve)

* Default mserd bandwidth selector.
rdrobust voteshare_next margin, c(0)

local n = e(N)
* e(tau_cl) = conventional point estimate.
* e(tau_bc) = bias-corrected point estimate.
* e(se_tau_cl) / e(se_tau_rb) = conventional / robust SE.
local conv_est = e(tau_cl)
local conv_se  = e(se_tau_cl)
local bc_est   = e(tau_bc)
local rb_se    = e(se_tau_rb)
local h_l      = e(h_l)
local b_l      = e(b_l)

local lo = `conv_est' - ${STATA_PARITY_Z95} * `conv_se'
local hi = `conv_est' + ${STATA_PARITY_Z95} * `conv_se'
stata_parity_row, stat(default_conventional_est) est(`conv_est') std(`conv_se') cilo(`lo') cihi(`hi') nob(`n')

local lo = `bc_est' - ${STATA_PARITY_Z95} * `rb_se'
local hi = `bc_est' + ${STATA_PARITY_Z95} * `rb_se'
stata_parity_row, stat(default_robust_est) est(`bc_est') std(`rb_se') cilo(`lo') cihi(`hi') nob(`n')

stata_parity_row, stat(default_bandwidth_h) est(`h_l') nob(`n')
stata_parity_row, stat(default_bandwidth_b) est(`b_l') nob(`n')

* Forced bandwidth replicate at h = b = 0.042287 (matches r_parity).
local H_FORCED = 0.042287
rdrobust voteshare_next margin, c(0) h(`H_FORCED') b(`H_FORCED')

local n = e(N)
local conv_est = e(tau_cl)
local conv_se  = e(se_tau_cl)
local bc_est   = e(tau_bc)
local rb_se    = e(se_tau_rb)

local lo = `conv_est' - ${STATA_PARITY_Z95} * `conv_se'
local hi = `conv_est' + ${STATA_PARITY_Z95} * `conv_se'
stata_parity_row, stat(forced_h0.042287_conventional_est) est(`conv_est') std(`conv_se') cilo(`lo') cihi(`hi') nob(`n')

local lo = `bc_est' - ${STATA_PARITY_Z95} * `rb_se'
local hi = `bc_est' + ${STATA_PARITY_Z95} * `rb_se'
stata_parity_row, stat(forced_h0.042287_robust_est) est(`bc_est') std(`rb_se') cilo(`lo') cihi(`hi') nob(`n')

stata_parity_extra, key(kernel) val(triangular)
stata_parity_extra, key(p) val(1)
stata_parity_extra, key(q) val(2)
stata_parity_extra, key(bwselect) val(mserd)
stata_parity_extra, key(stata_command) val("rdrobust Y X, c(0)")

stata_parity_close, module(06_rd)
