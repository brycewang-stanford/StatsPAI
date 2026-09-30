* csdid (v1) panel reference, time-invariant cluster, for test_cs_rc_cluster_csdid_parity.py
* mpdta_cluster.csv = sp.datasets.mpdta() with cl = mod(countyreal, 23) + 1.
version 18
clear all
import delimited using "mpdta_cluster.csv", clear asdouble
tempname fh
file open `fh' using "csdid_panel_cluster_Stata.json", write replace
qui csdid lemp, ivar(countyreal) time(year) gvar(first_treat) method(dripw) long2 cluster(cl)
mata: st_matrix("s", sqrt(diagonal(st_matrix("e(V)")))')
matrix b = e(b)
local k = colsof(s)
file write `fh' `"{"cell_att": ["'
forvalues j = 1/`k' {
    if `j' > 1 file write `fh' ", "
    file write `fh' %24.17e (b[1,`j'])
}
file write `fh' `"], "cell_se": ["'
forvalues j = 1/`k' {
    if `j' > 1 file write `fh' ", "
    file write `fh' %24.17e (s[1,`j'])
}
qui csdid lemp, ivar(countyreal) time(year) gvar(first_treat) method(dripw) long2 cluster(cl) agg(group)
matrix bg = e(b)
mata: st_matrix("sg", sqrt(diagonal(st_matrix("e(V)")))')
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
qui csdid lemp, ivar(countyreal) time(year) gvar(first_treat) method(dripw) long2 cluster(cl) agg(simple)
matrix bs = e(b)
mata: st_matrix("ss", sqrt(diagonal(st_matrix("e(V)")))')
file write `fh' `"], "simple_att": "' %24.17e (bs[1,1]) `", "simple_se": "' %24.17e (ss[1,1]) "}" _n
file close `fh'
