* Reference for sp.private_info_bound: the lower-bound estimator of Hendren
* (2013) as coded in the author's replication package (Econometrica
* supplement 10931, "Lower Bound Estimates (Stata)/makem.do"; the elicitation
* basis pnc* as in "Data Construction (Stata)/createhrs_all.do", LTC
* version: T1, T2, focal 0 and focal 1, times agep_0 = 1 and agep_1 =
* (age - 80) / 15, separately by rejection class). The loop body below is
* makem.do with its globals filled in ($cov = x1 x2, $subj = pnc*,
* $finelist = reject_list, $aglist = agegrp, $std = vce(cluster hhid));
* the only change is tight ML tolerances on the probits (at the default the
* probit stops about 1e-7 short of the maximum). Both of the paper's
* specifications run: probit ($probit = 1) and linear ($probit = 0).
* Run from the repository root:
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_private_info_bound_Stata.do
version 18
set type double
clear all
local tight "tolerance(1e-12) ltolerance(1e-14) nrtolerance(1e-14)"
import delimited "tests/reference_parity/_fixtures/private_info_bound_data.csv", clear asdouble

* --- basis, as createhrs_all.do (reject coded 0/1 there) ---
gen double reject = reject_list - 1
gen double p_use = z
gen double agep_0 = 1
gen double agep_1 = agep
local k = 0
foreach r in 0 1 {
foreach q in 0 1 {
foreach j in 1 2 4 5 {
local k = `k' + 1
if `j' == 1 {
gen double pnc`k' = 0
replace pnc`k' = agep_`q' * p_use^`j' if reject == `r'
}
if `j' == 2 {
gen double pnc`k' = 0
replace pnc`k' = agep_`q' *(2*p_use^`j'-1)  if reject == `r'
}
if `j' == 4 {
gen double pnc`k' = 0
replace pnc`k' = agep_`q' if z == 0 & reject == `r'
}
if `j' == 5 {
gen double pnc`k' = 0
replace pnc`k' = agep_`q' if z == 1 & reject == `r'
}
}
}
}

tempname fh
file open `fh' using "tests/reference_parity/_fixtures/private_info_bound_Stata.txt", write replace
file write `fh' "stata_version " "`c(stata_version)'" _n

foreach probit in 1 0 {
    local tag = cond(`probit', "probit", "linear")
    gen double m_`tag' = .
    forval j = 1/2 {
        _rmcoll x1 x2 if reject_list == `j', forcedrop
        local cov_use `r(varlist)'
        _rmcoll pnc* if reject_list == `j', forcedrop
        local subj_use `r(varlist)'
        if `probit' == 0 {
            reg loss `cov_use' `subj_use' if reject_list == `j', vce(cluster hhid)
            test `subj_use'
            file write `fh' "`tag'_stat_`j' " %21.17g (r(F)) _n
            file write `fh' "`tag'_p_`j' " %21.17g (r(p)) _n
            predict double temp_e_x_p if reject_list == `j'
            reg loss `cov_use' if reject_list == `j', vce(cluster hhid)
            predict double temp_e_x if temp_e_x_p ~= . & reject_list == `j'
        }
        if `probit' == 1 {
            probit loss `cov_use' `subj_use' if reject_list == `j', vce(cluster hhid) iter(50) `tight'
            test `subj_use'
            file write `fh' "`tag'_stat_`j' " %21.17g (r(chi2)) _n
            file write `fh' "`tag'_p_`j' " %21.17g (r(p)) _n
            predict double temp_e_x_p if reject_list == `j'
            probit loss `cov_use' if reject_list == `j', vce(cluster hhid) iter(50) `tight'
            predict double temp_e_x if temp_e_x_p ~= . & reject_list == `j'
        }
        gen double temp_i_x_p = temp_e_x_p - temp_e_x if reject_list == `j'
        gsort agegrp - temp_i_x_p
        gen double runner = 1 if _n == 1
        replace runner = (runner[_n-1]+1)*(agegrp[_n-1]==agegrp[_n]) + (agegrp[_n-1]~=agegrp[_n]) if _n > 1
        by agegrp : gen double temp_m = sum(temp_i_x_p) / runner - temp_i_x_p if reject_list == `j'
        summarize temp_m if reject_list == `j', meanonly
        file write `fh' "`tag'_M_`j' " %21.17g (r(mean)) _n
        replace m_`tag' = temp_m if reject_list == `j'
        drop temp_e_x_p temp_e_x temp_i_x_p runner temp_m
    }
}
file close `fh'
sort id
format m_probit m_linear %21.17g
export delimited id m_probit m_linear using "tests/reference_parity/_fixtures/private_info_bound_Stata_rows.csv", replace datafmt
