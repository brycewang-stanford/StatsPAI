* sp.did_multiplegt on individual-level data vs Stata did_multiplegt_old
* (Stata 18 MP). Data: dcdh_disaggregated_panel.csv (see the .py generator).
* Writes dcdh_disaggregated_Stata.csv: tag,name,value. Point estimates only
* (breps(0)); the bootstrap SEs are not compared.
version 18
clear all
set type double
local here "`c(pwd)'"
tempname fh
file open `fh' using "`here'/dcdh_disaggregated_Stata.csv", write replace
file write `fh' "tag,name,value" _n

capture program drop put
program define put
    args fh tag name value
    file write `fh' "`tag',`name'," %21.17g (`value') _n
end

capture program drop runall
program define runall
    args fh tag yv dv extra
    did_multiplegt_old `yv' g t `dv', breps(0) `extra'
    put `fh' `tag' effect_0 e(effect_0)
    put `fh' `tag' n_switchers_0 e(N_switchers_effect_0)
    did_multiplegt_old `yv' g t `dv', breps(0) placebo(1) `extra'
    put `fh' `tag' placebo_1 e(placebo_1)
    did_multiplegt_old `yv' g t `dv', breps(0) placebo(1) dynamic(1) robust_dynamic `extra'
    put `fh' `tag' dyn_placebo_1 e(placebo_1)
    put `fh' `tag' dyn_effect_0 e(effect_0)
    put `fh' `tag' dyn_effect_1 e(effect_1)
end

import delimited using "`here'/dcdh_disaggregated_panel.csv", clear asdouble
runall `fh' indiv y d ""
runall `fh' indiv_count y2 d2 ""
runall `fh' indiv_ctrl y d "controls(x)"
runall `fh' indiv_abs ya da ""

* the same data collapsed to (g, t) cells, weighted by cell size
import delimited using "`here'/dcdh_disaggregated_panel.csv", clear asdouble
gen n = 1
collapse (mean) y y2 ya d d2 da x (sum) n, by(g t)
runall `fh' cell_w y d "weight(n)"
runall `fh' cell_unw y d ""
runall `fh' cell_w_ctrl y d "weight(n) controls(x)"
runall `fh' cell_abs_unw ya da ""
runall `fh' cell_unw_ctrl y d "controls(x)"
file close `fh'
