* Goldsmith-Pinkham, Sorkin & Swift's Rotemberg-weight summary table, as their
* make_rotemberg_summary_ADH.do computes it (bartik_weight, per-industry first
* stages, ch_weak AR intervals), on a synthetic two-period panel, written
* unrounded for sp.rotemberg_summary. bartik_weight.ado and ch_weak.ado (MIT,
* github.com/paulgp/bartik-weight/code) are fetched into the gitignored
* _ado_bartik/.
*   python tests/reference_parity/_fixtures/_generate_rotemberg_summary_data.py
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_rotemberg_summary_Stata.do
version 18
clear all
set more off
set matsize 2000
* doubles throughout (GPSS's do-file leaves generated variables float)
set type double
local ado "tests/reference_parity/_fixtures/_ado_bartik"
adopath + "`ado'"
capture which bartik_weight
if _rc {
    display as error "copy bartik_weight.ado and ch_weak.ado from github.com/paulgp/bartik-weight/code into `ado'"
    exit 601
}

import delimited using "tests/reference_parity/_fixtures/rotemberg_shocks.csv", clear asdouble
tempfile shocks
save `shocks'
import delimited using "tests/reference_parity/_fixtures/rotemberg_panel.csv", clear asdouble
reshape long sh, i(unit year) j(ind)
merge m:1 year ind using `shocks', nogen
rename sh share_
rename g g_
reshape wide share_ g_, i(unit year) j(ind)

local controls c1 c2 t2
local weight w
local y y
local x x
levelsof year, local(years)

foreach t of local years {
    foreach var of varlist share_* {
        gen t`t'_`var' = (year == `t') * `var'
    }
    foreach var of varlist g_* {
        gen t`t'_`var'b = `var' if year == `t'
        egen t`t'_`var' = max(t`t'_`var'b), by(unit)
        drop t`t'_`var'b
    }
}

* Per-industry first stage F (clustered Wald), as the GPSS do-file.
foreach var of varlist share_* {
    local ind = substr("`var'", 7, .)
    tempvar temp
    qui gen `temp' = `var' * g_`ind'
    qui regress `x' `temp' `controls' [aweight=`weight'], cluster(unit)
    qui test `temp'
    local F_`ind' = r(F)
    drop `temp'
}

* Industry share mean / sd by year.
preserve
keep share_* unit year `weight'
reshape long share_, i(unit year) j(ind)
gen share_pop = share_ * `weight'
collapse (sd) share_sd = share_ (rawsum) share_pop `weight' [aweight = `weight'], by(ind year)
tempfile tmp
save `tmp'
restore

bartik_weight, z(t*_share_*) weightstub(t*_g_*) x(`x') y(`y') controls(`controls') weight_var(`weight')
mat beta = r(beta)
mat alpha = r(alpha)
mat G = r(G)
qui desc t*_share_*, varlist
local varlist = r(varlist)

* keep the panel for the AR intervals
tempfile panel
save `panel'

clear
svmat double beta
svmat double alpha
svmat double G
gen ind = ""
gen year = ""
local t = 1
foreach var in `varlist' {
    if regexm("`var'", "t(.*)_share_(.*)") {
        qui replace year = regexs(1) if _n == `t'
        qui replace ind = regexs(2) if _n == `t'
    }
    local t = `t' + 1
}
destring ind year, replace

tempname fh
file open `fh' using "tests/reference_parity/_fixtures/rotemberg_summary_Stata.json", write replace
file write `fh' "{" _n
* Panel C (before collapsing): sum / mean of alpha by year
foreach t of local years {
    qui sum alpha1 if year == `t'
    file write `fh' `"  "C_`t'": ["' %24.16e (r(sum)) ", " %24.16e (r(mean)) "]," _n
}
* industry x period table
file write `fh' `"  "cells": ["'
local N = _N
forvalues i = 1/`N' {
    if `i' > 1 file write `fh' ","
    file write `fh' "[" (ind[`i']) "," (year[`i']) "," %24.16e (alpha1[`i']) "," %24.16e (beta1[`i']) "," %24.16e (G1[`i']) "]"
}
file write `fh' "]," _n

merge 1:1 ind year using `tmp', nogen
gen beta2 = alpha1 * beta1
gen indshare2 = alpha1 * (share_pop/`weight')
gen indshare_sd2 = alpha1 * share_sd
gen G2 = alpha1 * G1
collapse (sum) alpha1 beta2 indshare2 indshare_sd2 G2 (mean) G1, by(ind)
gen agg_beta = beta2 / alpha1
gen agg_indshare = indshare2 / alpha1
gen agg_indshare_sd = indshare_sd2 / alpha1
gen agg_g = G2 / alpha1
gen F = .
levelsof ind, local(industries)
foreach ind in `industries' {
    qui replace F = `F_`ind'' if ind == `ind'
}
gsort -alpha1

* industry table
file write `fh' `"  "industries": ["'
local N = _N
forvalues i = 1/`N' {
    if `i' > 1 file write `fh' ","
    file write `fh' "[" (ind[`i']) "," %24.16e (alpha1[`i']) "," %24.16e (agg_g[`i']) "," %24.16e (agg_beta[`i']) "," %24.16e (F[`i']) "," %24.16e (agg_indshare[`i']) "," %24.16e (agg_indshare_sd[`i']) "]"
}
file write `fh' "]," _n

* Panel A
qui total alpha1 if alpha1 > 0
mat b = e(b)
local sp = b[1,1]
qui total alpha1 if alpha1 < 0
mat b = e(b)
local sn = b[1,1]
qui sum alpha1 if alpha1 > 0
local mp = r(mean)
qui sum alpha1 if alpha1 < 0
local mn = r(mean)
file write `fh' `"  "A_pos": ["' %24.16e (`sp') ", " %24.16e (`mp') ", " %24.16e (abs(`sp')/(abs(`sp')+abs(`sn'))) "]," _n
file write `fh' `"  "A_neg": ["' %24.16e (`sn') ", " %24.16e (`mn') ", " %24.16e (abs(`sn')/(abs(`sp')+abs(`sn'))) "]," _n

* Panel B
qui corr alpha1 agg_g agg_beta F agg_indshare_sd
mat C = r(C)
file write `fh' `"  "B": ["'
forvalues i = 1/5 {
    if `i' > 1 file write `fh' ","
    file write `fh' "["
    forvalues j = 1/5 {
        if `j' > 1 file write `fh' ","
        file write `fh' %24.16e (C[`i',`j'])
    }
    file write `fh' "]"
}
file write `fh' "]," _n

* top 5 industries for the AR intervals
forvalues i = 1/5 {
    local top`i' = ind[`i']
}

* Panel E
gen positive_weight = alpha1 > 0
gen agg_beta_weight = agg_beta * alpha1
preserve
collapse (sum) agg_beta_weight alpha1 (mean) agg_beta, by(positive_weight)
egen total_agg_beta = total(agg_beta_weight)
gen share = agg_beta_weight / total_agg_beta
gsort -positive_weight
file write `fh' `"  "E_pos": ["' %24.16e (agg_beta_weight[1]) ", " %24.16e (share[1]) ", " %24.16e (agg_beta[1]) "]," _n
file write `fh' `"  "E_neg": ["' %24.16e (agg_beta_weight[2]) ", " %24.16e (share[2]) ", " %24.16e (agg_beta[2]) "]," _n
restore

* Panel D intervals: ch_weak on the grid -10(.1)10
use `panel', clear
file write `fh' `"  "D_ci": {"'
forvalues i = 1/5 {
    local ind = `top`i''
    tempvar temp
    qui gen `temp' = share_`ind' * g_`ind'
    qui ch_weak, p(.05) beta_range(-10(.1)10) y(`y') x(`x') z(`temp') weight(`weight') controls(`controls') cluster(unit)
    if `i' > 1 file write `fh' ", "
    file write `fh' `""`ind'": ["' %24.16e (r(beta_min)) ", " %24.16e (r(beta_max)) "]"
    drop `temp'
}
file write `fh' "}" _n "}" _n
file close `fh'
