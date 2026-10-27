* Reference for sp.summary_index on summary_index_input.csv.
*
* Part 1, method='anderson': Stata swindex (SSC, Schwab, Janzen, Magnan and
* Thompson, distribution date 2021-02-24) with normby() set to the control
* indicator and flip(y5), under its rescaling options. Run twice: under
* `set type double` (the parity rows) and under `set type float`, Stata's
* default, where swindex stores its standardized components and weights as
* float (reported, not a parity row).
*
* Part 2, method='klk' and the options swindex does not have (weights,
* imputation, listwise): computed by hand from Stata primitives, summarize
* (with [aweight=]), egen rowmean / rownonmiss, bysort egen mean, correlate
* [aweight=], covariance and Mata invsym.
*
* swindex sorts the data, so every row is written with its id.
* Output: summary_index_Stata.csv (config, id, index in %21x) and
* summary_index_Stata_weights.csv (config, outcome, mean, sd, weight).
*
* Run from the repository root with Stata 18 and swindex on the adopath:
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_summary_index_Stata.do
version 18
set type double
clear all
local fix "tests/reference_parity/_fixtures"

import delimited "`fix'/summary_index_input.csv", clear asdouble varnames(1)
generate double control = (treat == 0) if !missing(treat)
tempfile base
save `base'

tempname fh fw
file open `fh' using "`fix'/summary_index_Stata.csv", write replace text
file write `fh' "config,id,index_hex" _n
file open `fw' using "`fix'/summary_index_Stata_weights.csv", write replace text
file write `fw' "config,outcome,mean_hex,sd_hex,weight_hex" _n

capture program drop _dump
program define _dump
    args fh config var
    sort id
    forvalues i = 1/`=_N' {
        file write `fh' "`config'," (id[`i']) "," %21x (`var'[`i']) _n
    }
end

* ---------------------------------------------------------------- swindex
foreach type in double float {
    local sfx = cond("`type'" == "double", "", "_float")
    set type `type'
    foreach spec in ///
        "A_ref|y1 y2 y3 y4 y5|normby(control)" ///
        "A_norescale|y1 y2 y3 y4 y5|normby(control) norescale" ///
        "A_full|y1 y2 y3 y4 y5|normby(control) fullrescale" ///
        "A_nonormby|y1 y2 y3 y4 y5|" ///
        "A_miss_ref|m1 m2 m3 m4 m5|normby(control)" ///
        "A_miss_norescale|m1 m2 m3 m4 m5|normby(control) norescale" {
        gettoken cfg rest : spec, parse("|")
        gettoken bar rest : rest, parse("|")
        gettoken vars rest : rest, parse("|")
        gettoken bar opts : rest, parse("|")
        local flipv = cond(substr("`vars'", 1, 1) == "y", "y5", "m5")
        use `base', clear
        swindex `vars', generate(ix) flip(`flipv') `opts'
        matrix W = r(wt)
        local k = 0
        foreach v of local vars {
            local ++k
            file write `fw' "`cfg'`sfx',`v',,," %21x (W[`k', 1]) _n
        }
        _dump `fh' "`cfg'`sfx'" ix
    }
}
set type double

* ------------------------------------------------------------ by hand
* Standardize the components named in `vars' by the reference rows `refif'
* with optional [aweight=w]; z_* are the z-scores, y5 / m5 sign-flipped.
capture program drop _zscores
program define _zscores
    args fw config vars refif wexp
    local k = 0
    foreach v of local vars {
        local ++k
        local s = cond(inlist("`v'", "y5", "m5"), -1, 1)
        generate double s_`k' = `s' * `v'
        quietly summarize s_`k' `wexp' if `refif'
        generate double z_`k' = (s_`k' - r(mean)) / r(sd)
        file write `fw' "`config',`v'," %21x (r(mean)) "," %21x (r(sd)) "," _n
    }
end

* Index from z_1 ... z_K with weights in the Stata matrix WV, available rule
capture program drop _windex
program define _windex
    args K
    generate double num = 0
    generate double den = 0
    forvalues k = 1/`K' {
        quietly replace num = num + z_`k' * WV[`k', 1] if !missing(z_`k')
        quietly replace den = den + WV[`k', 1] if !missing(z_`k')
    }
    generate double ix = num / den
end

local Y "y1 y2 y3 y4 y5"
local M "m1 m2 m3 m4 m5"

* K_ref_none: KLK, control reference, no rescaling (complete data)
use `base', clear
_zscores `fw' K_none "`Y'" "control == 1" ""
egen double ix = rowmean(z_1 z_2 z_3 z_4 z_5)
_dump `fh' K_none ix

* K_ref: KLK rescaled by the control mean and sd of the index
quietly summarize ix if control == 1
replace ix = (ix - r(mean)) / r(sd)
_dump `fh' K_ref ix

* K_fullsample: KLK standardized on every row (treat=None), no rescaling
use `base', clear
_zscores `fw' K_fullsample "`Y'" "1" ""
egen double ix = rowmean(z_1 z_2 z_3 z_4 z_5)
_dump `fh' K_fullsample ix

* K_miss_avail: KLK with missing components, available rule
use `base', clear
_zscores `fw' K_miss_avail "`M'" "control == 1" ""
egen double ix = rowmean(z_1 z_2 z_3 z_4 z_5)
_dump `fh' K_miss_avail ix

* K_miss_listwise
egen nobs = rownonmiss(z_1 z_2 z_3 z_4 z_5)
replace ix = . if nobs < 5
_dump `fh' K_miss_listwise ix

* K_miss_impute: missing z at the treat-group mean (missing treat is its
* own group) for rows with at least one component
drop ix
forvalues k = 1/5 {
    bysort treat: egen double zbar_`k' = mean(z_`k')
    replace z_`k' = zbar_`k' if missing(z_`k') & nobs > 0
}
egen double ix = rowmean(z_1 z_2 z_3 z_4 z_5)
_dump `fh' K_miss_impute ix

* K_w_miss_ref: KLK, aweights, missing components, rescaled by control
use `base', clear
_zscores `fw' K_w_miss_ref "`M'" "control == 1" "[aweight=w]"
egen double ix = rowmean(z_1 z_2 z_3 z_4 z_5)
quietly summarize ix [aweight=w] if control == 1
replace ix = (ix - r(mean)) / r(sd)
_dump `fh' K_w_miss_ref ix

* A_w_miss_ref: Anderson by hand, aweights, missing components, rescaled
* by control. Covariance over complete cases with the same aweights.
use `base', clear
_zscores `fw' A_w_miss_ref "`M'" "control == 1" "[aweight=w]"
quietly correlate z_1 z_2 z_3 z_4 z_5 [aweight=w], covariance
matrix C = r(C)
mata: wv = rowsum(invsym(st_matrix("C")))
mata: st_matrix("WV", wv)
forvalues k = 1/5 {
    local v : word `k' of `M'
    file write `fw' "A_w_miss_ref_w,`v',,," %21x (WV[`k', 1]) _n
}
_windex 5
quietly summarize ix [aweight=w] if control == 1
replace ix = (ix - r(mean)) / r(sd)
_dump `fh' A_w_miss_ref ix

* A_miss_impute / A_miss_listwise: Anderson (unweighted) by hand, weights
* from the complete-case covariance, then imputation or listwise; not
* rescaled.
use `base', clear
_zscores `fw' A_miss_hand "`M'" "control == 1" ""
quietly correlate z_1 z_2 z_3 z_4 z_5, covariance
matrix C = r(C)
mata: wv = rowsum(invsym(st_matrix("C")))
mata: st_matrix("WV", wv)
egen nobs = rownonmiss(z_1 z_2 z_3 z_4 z_5)
_windex 5
replace ix = . if nobs < 5
_dump `fh' A_miss_listwise ix
drop ix num den
forvalues k = 1/5 {
    bysort treat: egen double zbar_`k' = mean(z_`k')
    replace z_`k' = zbar_`k' if missing(z_`k') & nobs > 0
}
_windex 5
_dump `fh' A_miss_impute ix

file close `fh'
file close `fw'
