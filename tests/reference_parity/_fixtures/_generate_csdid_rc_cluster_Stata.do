* csdid (v1) / csdid2 references for tests/reference_parity/test_cs_rc_cluster_csdid_parity.py
* Repeated cross-section use (no ivar), as csdid / csdid2 run without ivar().
version 18
clear all
import delimited using "csdid2_rc_cluster.csv", clear asdouble
tempname fh
file open `fh' using "csdid_rc_cluster_Stata.json", write replace
file write `fh' "{" _n
local first 1
foreach cl in none cy {
    local copt
    if "`cl'" != "none" local copt cluster(`cl')
    qui csdid y, time(year) gvar(g) method(dripw) long2 `copt'
    matrix b = e(b)
    mata: st_matrix("s", sqrt(diagonal(st_matrix("e(V)")))')
    local bt ""
    local st ""
    forvalues j = 1/18 {
        local bt `bt' `: display %24.17e b[1,`j']'
        local st `st' `: display %24.17e s[1,`j']'
    }
    qui csdid y, time(year) gvar(g) method(dripw) long2 `copt' agg(group)
    matrix bg = e(b)
    mata: st_matrix("sg", sqrt(diagonal(st_matrix("e(V)")))')
    qui csdid y, time(year) gvar(g) method(dripw) long2 `copt' agg(simple)
    matrix bs = e(b)
    mata: st_matrix("ss", sqrt(diagonal(st_matrix("e(V)")))')
    if !`first' file write `fh' "," _n
    local first 0
    file write `fh' `""`cl'": {"cell_att": ["' 
    forvalues j = 1/18 {
        if `j' > 1 file write `fh' ", "
        file write `fh' %24.17e (b[1,`j'])
    }
    file write `fh' `"], "cell_se": ["'
    forvalues j = 1/18 {
        if `j' > 1 file write `fh' ", "
        file write `fh' %24.17e (s[1,`j'])
    }
    file write `fh' `"], "group_att": ["'
    forvalues j = 1/4 {
        if `j' > 1 file write `fh' ", "
        file write `fh' %24.17e (bg[1,`j'])
    }
    file write `fh' `"], "group_se": ["'
    forvalues j = 1/4 {
        if `j' > 1 file write `fh' ", "
        file write `fh' %24.17e (sg[1,`j'])
    }
    file write `fh' `"], "simple_att": "' %24.17e (bs[1,1]) `", "simple_se": "' %24.17e (ss[1,1])
    qui csdid2 y, time(year) gvar(g) method(dripw) long2 `copt' agg(group)
    matrix b2 = e(b)
    mata: st_matrix("s2", sqrt(diagonal(st_matrix("e(V)")))')
    file write `fh' `", "csdid2_group_att": ["'
    forvalues j = 1/4 {
        if `j' > 1 file write `fh' ", "
        file write `fh' %24.17e (b2[1,`j'])
    }
    file write `fh' `"], "csdid2_group_se": ["'
    forvalues j = 1/4 {
        if `j' > 1 file write `fh' ", "
        file write `fh' %24.17e (s2[1,`j'])
    }
    file write `fh' "]}"
}
file write `fh' _n "}" _n
file close `fh'
