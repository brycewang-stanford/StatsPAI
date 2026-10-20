* Stata xteventplot, smpath(line) (xtevent SSC v3.1.0; Freyaldenhoven,
* Hansen, Perez Perez, Shapiro) for sp.xtevent_smpath: the "least wiggly"
* polynomial path through the Wald region of the event-time coefficients.
* xtevent is a black box here: only its help file and its outputs are used.
* Stata stores no result for the path, so it is read back from the graph's
* serset (first variable: the path, second: event time), and the least Wald
* statistic it prints for each polynomial order is read back from a log.
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_xtevent_smpath_Stata.do
version 18
clear all
local ado "tests/reference_parity/_fixtures/_ado_xtevent"
capture mkdir "`ado'"
adopath + "`ado'"
capture which xtevent
if _rc {
    net set ado "`ado'"
    ssc install xtevent, replace
    net set ado PLUS
}
capture program drop _plot
program define _plot
    * xteventplot with its display logged to smpath_tmp.log
    args opts
    capture log close smlog
    log using "smpath_tmp.log", text replace name(smlog)
    xteventplot, `opts' nosupt
    log close smlog
end
capture program drop _path
program define _path
    args fh key
    tempname lf
    local walds ""
    local crit ""
    local found 1
    file open `lf' using "smpath_tmp.log", read text
    file read `lf' line
    while r(eof) == 0 {
        local w3 = word(`"`macval(line)'"', 3)
        local w4 = word(`"`macval(line)'"', 4)
        if strpos(`"`macval(line)'"', "Wald Critical Value") local crit `w4'
        else if strpos(`"`macval(line)'"', "Wald value") local walds `walds' `w3'
        if strpos(`"`macval(line)'"', "Could not find a polynomial") local found 0
        file read `lf' line
    }
    file close `lf'
    erase "smpath_tmp.log"
    file write `fh' `"  "`key'": {"smpath": "'
    if `found' {
        preserve
        serset use, clear
        unab vars : _all
        rename `: word 1 of `vars'' smp
        rename `: word 2 of `vars'' k
        keep if !missing(k) & !missing(smp)
        file write `fh' "{"
        forvalues i = 1/`=_N' {
            if `i' > 1 file write `fh' ", "
            file write `fh' `"""' %4.0f (k[`i']) `"": "' %24.16e (smp[`i'])
        }
        file write `fh' "}"
        restore
    }
    else file write `fh' "null"
    file write `fh' `", "critical": "' %21.10e (real("`crit'")) `", "wald_by_order": ["'
    local j = 0
    foreach w of local walds {
        if `j' > 0 file write `fh' ","
        file write `fh' %21.10e (real("`w'"))
        local ++j
    }
    file write `fh' "]}"
end
tempname fh
file open `fh' using "tests/reference_parity/_fixtures/xtevent_smpath_Stata.json", write replace
file write `fh' "{" _n
import delimited using "tests/reference_parity/_fixtures/xtevent_continuous.csv", clear asdouble
xtset id t
qui xtevent y x, policyvar(z) window(2)
_plot "smpath(line)"
_path `fh' cont_w2
file write `fh' "," _n
qui xtevent y x, policyvar(z) window(-3 4) vce(cluster cl)
_plot "smpath(line)"
_path `fh' cont_w34_cluster
file write `fh' "," _n
qui xtevent y x, policyvar(z) window(3) norm(-2)
_plot "smpath(line)"
_path `fh' cont_w3_norm2
file write `fh' "," _n
qui xtevent y x, policyvar(z) window(3)
_plot "smpath(line, postwindow(2))"
_path `fh' cont_w3_postwindow2
file write `fh' "," _n
import delimited using "tests/reference_parity/_fixtures/xtevent_binary.csv", clear asdouble
xtset id t
qui xtevent y, policyvar(z) window(4)
_plot "smpath(line)"
_path `fh' bin_w4
file write `fh' "," _n
qui xtevent y, policyvar(z) window(4) vce(cluster cl)
_plot "smpath(line, maxorder(2))"
_path `fh' bin_w4_cluster_maxorder2
file write `fh' _n "}" _n
file close `fh'
