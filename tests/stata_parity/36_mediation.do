* tests/stata_parity/36_mediation.do
*
* Module 36: Causal mediation analysis.
*   StatsPAI:  sp.mediation
*   R:         mediation::mediate (Imai-Keele-Tingley quasi-Bayesian MC)
*   Stata:     paramed (Emsley & Liu; closed-form natural effects)
*
* Tolerance: rel < 1e-6 on ACME/ADE/total point estimates; SE rows
* are diagnostics because mediation packages use different simulation /
* delta-method conventions.
*
* Cross-package mapping:
*   paramed `nie` = natural indirect effect ~ R's ACME (= a * b in
*       linear-linear model with no/small AxM interaction)
*   paramed `nde` = natural direct effect    ~ R's ADE
*   paramed `mte` = marginal total effect    ~ R's total
*   Proportion mediated = nie / mte
*
* paramed produces delta-method SEs; the R side bootstraps with 200
* simulations. The point estimates should match closely (closed-form
* OLS coefficients); the SEs may differ at the 5-10% level due to
* delta vs bootstrap noise.

version 18
clear all

do _common.do
stata_parity_init, module(36_mediation)
stata_parity_open, module(36_mediation)

import delimited "${STATA_PARITY_DATA}/36_mediation.csv", clear case(preserve)

local n = _N

* paramed with `nointer`: no AxM interaction term, matching the R
* side which fits lm(y ~ treat + m). Without interaction, NDE = CDE
* and paramed outputs only {cde, nie, te}.
paramed y, avar(treat) mvar(m) a0(0) a1(1) m(0) yreg(linear) mreg(linear) nointer

* Effects matrix with nointer: rows are cde, nie, te; columns are
* estimate, se, p, ci_lo, ci_hi. CDE = ADE (no interaction).
matrix E = e(effects)
local nde_b   = E[1, 1]
local nde_se  = E[1, 2]
local nie_b   = E[2, 1]
local nie_se  = E[2, 2]
local mte_b   = E[3, 1]
local mte_se  = E[3, 2]
local prop    = `nie_b' / `mte_b'

stata_parity_row, stat(acme)          est(`nie_b') std(`nie_se') nob(`n')
stata_parity_row, stat(ade)           est(`nde_b') std(`nde_se') nob(`n')
stata_parity_row, stat(total_effect)  est(`mte_b') std(`mte_se') nob(`n')
stata_parity_row, stat(prop_mediated) est(`prop')                nob(`n')

stata_parity_extra, key(method) val("paramed (Emsley-Liu)")
stata_parity_extra, key(se_method) val("delta")
stata_parity_extra, key(stata_command) val("paramed y, avar(treat) mvar(m) a0(0) a1(1) m(0) yreg(linear) mreg(linear) nointer")
stata_parity_extra, key(note) val("nointer matches the R side's lm(y ~ treat + m) specification; with no AxM interaction, paramed's NIE = R's ACME, CDE = NDE = ADE. SEs differ because paramed uses the delta method while mediation::mediate bootstraps (sims=200).")

stata_parity_close, module(36_mediation)
