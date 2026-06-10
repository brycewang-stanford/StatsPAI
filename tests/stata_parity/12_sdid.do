* tests/stata_parity/12_sdid.do
*
* Module 12: Synthetic DiD on California Prop99 replica.
*   StatsPAI:  sp.synth(method="sdid")
*   R:         synthdid::synthdid_estimate + synthdid::synthdid_se(method="placebo")
*   Stata:     sdid (Daniel Pailanir et al., SSC)
*
* The headline ATT row is point-only; Stata's native placebo SE is
* emitted as an explicitly named diagnostic row.

version 18
clear all

do _common.do
stata_parity_init, module(12_sdid)
stata_parity_open, module(12_sdid)

import delimited "${STATA_PARITY_DATA}/12_sdid.csv", clear case(preserve) ///
    stringcols(1)

* sdid syntax: sdid Yvar unitvar timevar treatvar, vce(method)
sdid cigsale state year treated, vce(placebo) seed(42)

local n = e(N)
local att = e(ATT)
local se  = e(se)
stata_parity_row, stat(att_sdid) est(`att') nob(`n')
stata_parity_row, stat(se_stata_sdid_placebo) est(`se') nob(`n')

stata_parity_extra, key(method) val(synthdid_estimate)
stata_parity_extra, key(se_method) val(placebo)
stata_parity_extra, key(stata_command) val("sdid cigsale state year treated, vce(placebo)")
stata_parity_extra, key(se_reference) val("Stata records sdid placebo SEs as a diagnostic row; att_sdid is point-only.")

stata_parity_close, module(12_sdid)
