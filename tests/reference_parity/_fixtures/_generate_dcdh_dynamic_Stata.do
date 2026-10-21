* sp.did_multiplegt dynamic / placebo / controls conventions vs Stata 18
* did_multiplegt_old, with and without robust_dynamic.
* Data: dcdh_dynamic_panel.csv (see _generate_dcdh_dynamic_data.py).
* Writes dcdh_dynamic_Stata.csv: tag,name,value. Point estimates only
* (breps(0)). Tags: <layout>_<treatment>_<controls>_<balance>_<mode>, with
* layout indiv (rows) or cell (collapsed, unweighted), treatment db / dc,
* controls none / x, balance bal / unb (keep == 1), mode nr (no
* robust_dynamic, placebo(2)) or rd (robust_dynamic dynamic(3) placebo(3)).
version 18
clear all
set type double
local here "`c(pwd)'"
tempname fh
file open `fh' using "`here'/dcdh_dynamic_Stata.csv", write replace
file write `fh' "tag,name,value" _n

capture program drop put
program define put
    args fh tag name value
    file write `fh' "`tag',`name'," %21.17g (`value') _n
end

capture program drop runone
program define runone
    args fh tag yv dv ctrl
    local copt ""
    if "`ctrl'" == "x" local copt "controls(x)"
    did_multiplegt_old `yv' g t `dv', breps(0) placebo(2) `copt'
    put `fh' `tag'_nr effect_0 e(effect_0)
    put `fh' `tag'_nr placebo_1 e(placebo_1)
    put `fh' `tag'_nr placebo_2 e(placebo_2)
    did_multiplegt_old `yv' g t `dv', breps(0) robust_dynamic dynamic(3) placebo(3) `copt'
    forvalues l = 0/3 {
        put `fh' `tag'_rd effect_`l' e(effect_`l')
        put `fh' `tag'_rd N_switchers_effect_`l' e(N_switchers_effect_`l')
    }
    forvalues l = 1/3 {
        put `fh' `tag'_rd placebo_`l' e(placebo_`l')
    }
end

foreach layout in indiv cell {
    foreach bal in bal unb {
        import delimited using "`here'/dcdh_dynamic_panel.csv", clear asdouble
        if "`bal'" == "unb" keep if keep == 1
        if "`layout'" == "cell" {
            collapse (mean) y yb dc db x, by(g t)
        }
        foreach ctrl in none x {
            runone `fh' `layout'_db_`ctrl'_`bal' yb db `ctrl'
            runone `fh' `layout'_dc_`ctrl'_`bal' y dc `ctrl'
        }
    }
}
file close `fh'
