* ---------------------------------------------------------------------------
* Stata reference for sp.match(method='radius', caliper_scale='sd') in
* tests/reference_parity/test_psm_options_parity.py.
*
* psmatch2 has no standard-deviation caliper: caliper() is a raw width on
* the score scale (help psmatch2: "value for maximum distance of controls").
* sp.match reads caliper_scale='sd' as caliper * sd(score), the sd taken
* over the full estimation sample (ddof 1), as it does for nearest-neighbour
* matching. This file computes that width from psmatch2's own _pscore and
* runs psmatch2 radius matching with it, so the fixture pins the radius
* estimate at the SD-scaled width. It also runs kernel matching with and
* without caliper() to record whether psmatch2 kernel matching reads it.
*
* psmatch2 is installed into a private ado directory, never PLUS.
* Run from this directory:  stata-mp -b do _generate_psm_radius_sd_stata.do
* Writes psm_radius_sd_Stata.json (numbers as %23.16e).
* ---------------------------------------------------------------------------
version 18
set more off
set type double
local A "`c(pwd)'/_ado_psm_options"
capture mkdir "`A'"
sysdir set PLUS "`A'"
adopath ++ "`A'"
capture which psmatch2
if _rc ssc install psmatch2

import delimited using "psm_options_data.csv", clear asdouble
local X "x1 x2 x3 x4"
sort id

file open fh using "psm_radius_sd_Stata.json", write replace text
file write fh "{" _n

quietly psmatch2 d `X', outcome(y) logit neighbor(1)
quietly summarize _pscore
local sd = r(sd)
file write fh `"  "pscore_sd": "' %23.16e (`sd') "," _n

foreach c in 0.1 0.25 {
    local tag = subinstr("`c'", ".", "p", .)
    local w = `c' * `sd'
    quietly psmatch2 d `X', outcome(y) logit radius caliper(`w')
    file write fh `"  "radius_sd`tag'": {"caliper_sd": `c', "width": "' %23.16e (`w') `", "est": "' %23.16e (r(att)) `", "se": "' %23.16e (r(seatt)) "}," _n
}

quietly psmatch2 d `X', outcome(y) logit kernel kerneltype(epan) bwidth(0.06)
local k0 = r(att)
quietly psmatch2 d `X', outcome(y) logit kernel kerneltype(epan) bwidth(0.06) caliper(0.001)
local k1 = r(att)
file write fh `"  "kernel_epan_bw0p06": {"est": "' %23.16e (`k0') `", "est_with_caliper0p001": "' %23.16e (`k1') "}" _n
file write fh "}" _n
file close fh
