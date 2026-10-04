* Stata references for test_stata_did_commands_parity.py.
* drdid 1.91, csdid 1.81, jwdid (SSC), Stata 18. Data: did_commands_data.csv
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

file write `fh' _n "}" _n
file close `fh'
