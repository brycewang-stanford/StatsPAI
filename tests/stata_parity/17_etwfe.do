* tests/stata_parity/17_etwfe.do
*
* Module 17: Wooldridge ETWFE.
*   StatsPAI:  sp.etwfe / sp.etwfe_emfx / sp.wooldridge_did
*   R:         etwfe::etwfe + etwfe::emfx
*   Stata:     jwdid (Wooldridge port by Fernando Rios-Avila)
*
* Four aggregations are pinned, not one. Through StatsPAI 1.26.0 the
* pooled ATT matched all three languages to 1e-13 while the per-cohort
* ATTs beneath it were off by up to 37%, because they were read off a
* separate, unsaturated cohort x post regression. A headline row cannot
* police the aggregation beneath it.
*
*   att_etwfe              jwdid + estat simple  (default notyet)
*   att_group_notyet_<g>   jwdid + estat group   (default notyet)
*   att_etwfe_never        jwdid, never + estat simple
*   att_group_never_<g>    jwdid, never + estat group
*
* Tolerance: rel_est < 1e-6, rel_se < 1e-3.

version 18
clear all

do _common.do
stata_parity_init, module(17_etwfe)
stata_parity_open, module(17_etwfe)

import delimited "${STATA_PARITY_DATA}/17_etwfe.csv", clear case(preserve)

* jwdid: Wooldridge ETWFE estimator.
* Syntax: jwdid depvar [if] [in], ivar(panelvar) tvar(timevar) gvar(cohortvar)
jwdid lemp, ivar(countyreal) tvar(year) gvar(first_treat) cluster(countyreal)
local n = e(N)

* Pooled ATT under the default not-yet-treated comparison group.
estat simple
matrix B = r(b)
matrix V = r(V)
local bv = B[1, 1]
local sv = sqrt(V[1, 1])
local lo = `bv' - ${STATA_PARITY_Z95} * `sv'
local hi = `bv' + ${STATA_PARITY_Z95} * `sv'
stata_parity_row, stat(att_etwfe) est(`bv') std(`sv') cilo(`lo') cihi(`hi') nob(`n')

* Per-cohort ATTs from the same fit.
estat group
matrix BG = r(b)
matrix VG = r(V)
local cohorts 2004 2006 2007
local j = 0
foreach g of local cohorts {
    local ++j
    local bg = BG[1, `j']
    local sg = sqrt(VG[`j', `j'])
    stata_parity_row, stat(att_group_notyet_`g') est(`bg') std(`sg') nob(`n')
}

* Never-treated comparison group. The StatsPAI counterpart is
* sp.wooldridge_did, which absorbs unit fixed effects exactly as
* jwdid's ivar() does.
jwdid lemp, ivar(countyreal) tvar(year) gvar(first_treat) cluster(countyreal) never
local nn = e(N)

estat simple
matrix BN = r(b)
matrix VN = r(V)
local bnv = BN[1, 1]
local snv = sqrt(VN[1, 1])
local nlo = `bnv' - ${STATA_PARITY_Z95} * `snv'
local nhi = `bnv' + ${STATA_PARITY_Z95} * `snv'
stata_parity_row, stat(att_etwfe_never) est(`bnv') std(`snv') cilo(`nlo') cihi(`nhi') nob(`nn')

estat group
matrix BNG = r(b)
matrix VNG = r(V)
local j = 0
foreach g of local cohorts {
    local ++j
    local bg = BNG[1, `j']
    local sg = sqrt(VNG[`j', `j'])
    stata_parity_row, stat(att_group_never_`g') est(`bg') std(`sg') nob(`nn')
}

stata_parity_extra, key(method) val(jwdid)
stata_parity_extra, key(stata_command) val("jwdid lemp, ivar(countyreal) tvar(year) gvar(first_treat) [never] | estat simple | estat group")

stata_parity_close, module(17_etwfe)
