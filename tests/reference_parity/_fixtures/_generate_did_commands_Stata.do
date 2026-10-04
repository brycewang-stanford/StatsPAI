* Stata references for test_stata_did_commands_parity.py.
* drdid 1.91, csdid 1.81, jwdid and did2s (SSC), Stata 18. Data: did_commands_data.csv
* (_generate_did_commands_data.py). Each block is one command line the test
* hands to sp.stata, followed by the e(b) / e(V) Stata left behind.
* Run from this folder: stata-mp -q -b do _generate_did_commands_Stata.do
version 18
clear all
set linesize 255
import delimited using "did_commands_data.csv", clear asdouble

tempname fh
file open `fh' using "did_commands_Stata.json", write replace
file write `fh' "{" _n
global first 1

cap program drop emit
program emit
    args fh tag
    tempname b V
    matrix `b' = e(b)
    matrix `V' = e(V)
    local names : colfullnames `b'
    local k = colsof(`b')
    if $first == 0 file write `fh' "," _n
    global first 0
    file write `fh' `""`tag'": {"names": ["'
    forvalues j = 1/`k' {
        local nm : word `j' of `names'
        if `j' > 1 file write `fh' ", "
        file write `fh' `""`nm'""'
    }
    file write `fh' `"], "b": ["'
    forvalues j = 1/`k' {
        if `j' > 1 file write `fh' ", "
        file write `fh' %24.17e (`b'[1,`j'])
    }
    file write `fh' `"], "se": ["'
    forvalues j = 1/`k' {
        if `j' > 1 file write `fh' ", "
        file write `fh' %24.17e (sqrt(`V'[`j',`j']))
    }
    file write `fh' "]}"
end

* the unweighted mean of the posted rows and its SE through their joint e(V)
cap program drop emitmean
program emitmean
    args fh tag
    tempname m s
    mata: st_numscalar("`m'", mean(st_matrix("e(b)")'))
    mata: st_numscalar("`s'", sqrt(sum(st_matrix("e(V)")) / cols(st_matrix("e(V)"))^2))
    file write `fh' "," _n `""`tag'": {"names": ["mean"], "b": ["'
    file write `fh' %24.17e (`m') `"], "se": ["' %24.17e (`s') "]}"
end

* --- drdid: cohort 2004 against the never treated, 2003 -> 2005 -------------
local two "if (g == 2004 | g == 0) & (year == 2003 | year == 2005)"
foreach m in drimp dripw reg stdipw ipw {
    qui drdid y x1 x2 `two', ivar(id) time(year) treatment(d04) `m'
    emit `fh' drdid_panel_`m'
    qui drdid y x1 x2 `two', time(year) treatment(d04) `m'
    emit `fh' drdid_rc_`m'
}
qui drdid y x1 x2 `two', ivar(id) time(year) treatment(d04)
emit `fh' drdid_panel_default
qui drdid y x1 x2 `two', time(year) treatment(d04) dripw rc1
emit `fh' drdid_rc_dripw_rc1

* --- csdid: the improved doubly robust cells --------------------------------
qui csdid y x1 x2, ivar(id) time(year) gvar(g) method(drimp)
emit `fh' csdid_drimp
qui csdid y x1 x2, ivar(id) time(year) gvar(g) method(drimp) notyet long2
emit `fh' csdid_drimp_notyet_long2
* a bare estimator name is swallowed by csdid's `*`: this is method(dripw)
qui csdid y x1 x2, ivar(id) time(year) gvar(g) ipw
emit `fh' csdid_bare_ipw
qui csdid y x1 x2, ivar(id) time(year) gvar(g) method(dripw)
emit `fh' csdid_dripw

* --- a covariate that varies over time ---------------------------------------
qui csdid y xt, ivar(id) time(year) gvar(g) method(dripw)
emit `fh' csdid_tv_dripw
qui csdid y xt x2, ivar(id) time(year) gvar(g) method(reg) notyet long2
emit `fh' csdid_tv_reg_notyet_long2
qui csdid y xt, ivar(id) time(year) gvar(g) method(drimp) long2
emit `fh' csdid_tv_drimp_long2

* --- csdid_estat / estat aggregations ---------------------------------------
foreach a in simple group calendar event {
    qui csdid y x1 x2, ivar(id) time(year) gvar(g) method(dripw)
    qui csdid_estat `a', post
    emit `fh' csdid_dripw_`a'
    qui csdid y x1 x2, ivar(id) time(year) gvar(g) method(drimp) notyet
    qui estat `a', post
    emit `fh' csdid_drimp_notyet_`a'
}
qui csdid y x1 x2, ivar(id) time(year) gvar(g) method(dripw)
qui csdid_estat event, window(-2 1) post
emit `fh' csdid_dripw_event_window

* --- jwdid and its estat ----------------------------------------------------
foreach a in simple group calendar event {
    qui jwdid y, ivar(id) tvar(year) gvar(g)
    qui estat `a', post
    emit `fh' jwdid_`a'
    if "`a'" != "simple" emitmean `fh' jwdid_`a'_mean
}
qui jwdid y, ivar(id) tvar(year) gvar(g) never
qui estat event, post
emit `fh' jwdid_never_event
qui jwdid y, ivar(id) tvar(year) gvar(g) never
qui estat simple, post
emit `fh' jwdid_never_simple

* --- drdid, all: the five estimators side by side ---------------------------
preserve
keep `two'
qui drdid y x1 x2, ivar(id) time(year) treatment(d04) all
emit `fh' drdid_panel_all
qui drdid y x1 x2, time(year) treatment(d04) all
emit `fh' drdid_rc_all
restore

* --- csdid's default propensity trimming is none -----------------------------
* four never-treated units are given a covariate value deep in the treated
* range, so their propensity score is above 0.995
preserve
gen xsep = x1 + 2*(g > 0)
replace xsep = 9 if g == 0 & inlist(id, 1, 5, 9, 13)
qui csdid y xsep, ivar(id) time(year) gvar(g) method(dripw)
emit `fh' csdid_trim_default
qui csdid y xsep, ivar(id) time(year) gvar(g) method(dripw) pscoretrim(0.995)
emit `fh' csdid_trim_995
restore

* --- reghdfe with factor variables ------------------------------------------
qui reghdfe y ib2003.g xt, absorb(year) vce(cluster id)
emit `fh' reghdfe_ib
qui reghdfe y i.g xt, absorb(year) vce(cluster id) noconstant
emit `fh' reghdfe_i

* --- teffects ra / ipwra on the 2005 cross-section, x2 as the treatment ------
* (This block comes before did2s: with unit(), did2s 0.5 demeans the outcome
* in memory and leaves it demeaned.)
preserve
keep if year == 2005
foreach e in ate atet {
    qui teffects ra (y x1 xt) (x2), `e'
    emit `fh' teffects_ra_`e'
    qui teffects ipwra (y x1 xt) (x2 x1 xt), `e'
    emit `fh' teffects_ipwra_`e'
    qui teffects ipwra (y x1 xt) (x2 x1), `e'
    emit `fh' teffects_ipwra_ps1_`e'
}
restore

* --- did2s ------------------------------------------------------------------
gen d = g > 0 & year >= g
gen relshift = cond(g > 0, year - g + 10, 0)
gen wt = 1 + x2
gen post0 = relshift == 10
gen post1 = relshift == 11
gen post2 = relshift >= 12 & g > 0
qui did2s y, first_stage(i.id i.year) second_stage(i.d) treatment(d) cluster(id)
emit `fh' did2s_static
qui did2s y, first_stage(i.year) second_stage(i.d) treatment(d) cluster(id) unit(id)
emit `fh' did2s_static_unit
qui did2s y, first_stage(i.id i.year) second_stage(ib0.relshift) treatment(d) cluster(id)
emit `fh' did2s_event
qui did2s y, first_stage(i.x2#i.year) second_stage(i.d) treatment(d) cluster(id) unit(id)
emit `fh' did2s_cell_fe
qui did2s y, first_stage(i.id i.year xt) second_stage(i.d) treatment(d) cluster(id)
emit `fh' did2s_control
qui did2s y [aw=wt], first_stage(i.id i.year) second_stage(i.d) treatment(d) cluster(id)
emit `fh' did2s_weighted
qui did2s y, first_stage(i.id i.year) second_stage(post0 post1 post2) treatment(d) cluster(id)
emit `fh' did2s_dummies
qui did2s y, first_stage(i.id i.year) second_stage(d) treatment(d) cluster(x2)
emit `fh' did2s_cluster_x2

file write `fh' _n "}" _n
file close `fh'
