* ---------------------------------------------------------------------------
* Black-box reference numbers for the clean-room rewrite of kernel, radius
* and local linear regression matching (docs/dev/cleanroom/psmatch2.md), for
* tests/reference_parity/test_cleanroom_psmatch2.py.
*
* Requires Stata 18 + psmatch2 4.0.12 (ssc install psmatch2). psmatch2 is
* used as a black box: only r(att), r(seatt) and the variables it generates
* are read. The score is passed with pscore(), so no logit fit enters.
*
* Run from this directory:  stata -b do _generate_psmatch2_cleanroom.do
* Output: psmatch2_cleanroom_stata.json
* ---------------------------------------------------------------------------
clear all
set type double

tempname fh
file open `fh' using "psmatch2_cleanroom_stata.json", write replace
file write `fh' "{" _n
file write `fh' `"  "_source": "Stata 18 MP + psmatch2 4.0.12, _generate_psmatch2_cleanroom.do","' _n

capture program drop loadkd
program define loadkd
    import delimited using "psmatch2_kernel_data.csv", clear asdouble
    keep x1 x2 d y _pscore
    rename _pscore ps
end

capture program drop put
program define put
    args fh key
    file write `fh' `"  "`key'": {"att": "' %21.16e (r(att)) `", "seatt": "'
    if missing(r(seatt)) file write `fh' "null"
    else file write `fh' %21.16e (r(seatt))
    file write `fh' "}," _n
end

* 1. kernel matching, each kernel, bwidth(0.1); and the SE rebuilt from
*    the generated variables (unweighted variances, controls with _weight)
foreach kt in epan normal biweight uniform tricube {
    loadkd
    quietly psmatch2 d, pscore(ps) outcome(y) kernel kerneltype(`kt') bwidth(0.1)
    put `fh' kernel_`kt'
}

* 2. local linear regression matching, bwidth(0.3)
foreach kt in tricube normal uniform biweight {
    loadkd
    quietly psmatch2 d, pscore(ps) outcome(y) llr kerneltype(`kt') bwidth(0.3)
    put `fh' llr_`kt'
}

* 3. llr with the Epanechnikov kernel (smoothed outcome + nearest neighbour)
loadkd
quietly psmatch2 d, pscore(ps) outcome(y) llr kerneltype(epan) bwidth(0.3)
put `fh' llr_epan

* 4. a control exactly one bandwidth away (|u| = 1)
capture program drop loadbd
program define loadbd
    clear
    set obs 6
    generate double y = .
    generate byte d = .
    generate double ps = .
    local yv 1 2 10 20 30 40
    local dv 1 1 0 0 0 0
    local pv .5 .25 .25 .5 .75 .375
    forvalues i = 1/6 {
        quietly replace y = `: word `i' of `yv'' in `i'
        quietly replace d = `: word `i' of `dv'' in `i'
        quietly replace ps = `: word `i' of `pv'' in `i'
    }
end
foreach kt in epan uniform tricube biweight normal {
    loadbd
    quietly psmatch2 d, pscore(ps) outcome(y) kernel kerneltype(`kt') bwidth(0.25)
    put `fh' boundary_kernel_`kt'
}
loadbd
quietly psmatch2 d, pscore(ps) outcome(y) radius caliper(0.25)
put `fh' boundary_radius
* weight of the control at ps = .75 (exactly h from the treated unit at .5)
loadbd
quietly psmatch2 d, pscore(ps) outcome(y) kernel kerneltype(uniform) bwidth(0.25)
quietly summarize _weight if d == 0 & ps == .75
file write `fh' `"  "boundary_uniform_weight_at_h": "' %21.16e (r(mean)) "," _n
loadbd
quietly psmatch2 d, pscore(ps) outcome(y) kernel kerneltype(epan) bwidth(0.25)
quietly count if d == 0 & ps == .75 & missing(_weight)
file write `fh' `"  "boundary_epan_weight_at_h_missing": "'
if r(N) == 1 file write `fh' "true" _n
else file write `fh' "false" _n
file write `fh' "}" _n
file close `fh'
