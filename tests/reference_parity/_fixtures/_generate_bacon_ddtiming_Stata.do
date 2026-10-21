* Reference for sp.bacon_decomposition(balance='observed_rows'): Stata
* ddtiming (Goldring, version 0.2, 13may2019) on the three unbalanced
* panels of bacon_ddtiming_data.csv (_generate_bacon_ddtiming_data.py).
*
* For each panel: ddtiming's e() scalars (e(dd), and by comparison type
* e = earlier vs later, l = later vs earlier, u = vs never treated,
* a = vs already treated: total weight e(wt_sum_*) and average 2x2
* e(dd_avg_*)), the TWFE coefficient from reghdfe 6.13.1, and the 2x2 rows
* ddtiming's savedata() writes (8 decimals).
*
* Run from the repository root with Stata 18 and ddtiming / reghdfe on the
* adopath:
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_bacon_ddtiming_Stata.do
version 18
set type double
clear all

local fix "tests/reference_parity/_fixtures"
tempname fh
file open `fh' using "`fix'/bacon_ddtiming_Stata.json", write replace text
file write `fh' "{" _n
forvalues p = 0/2 {
    import delimited "`fix'/bacon_ddtiming_data.csv", clear asdouble
    keep if panel == `p'
    xtset unit year
    local save "`c(tmpdir)'/ddtiming_rows_`p'"
    quietly ddtiming y d, i(unit) t(year) savedata("`save'") replace nodraw
    file write `fh' `"  "`p'": {"scalars": {"'
    local first 1
    foreach k in dd dd_avg_e dd_avg_l dd_avg_u dd_avg_a wt_sum_e wt_sum_l wt_sum_u wt_sum_a {
        if missing(e(`k')) continue
        if !`first' file write `fh' ", "
        file write `fh' `""`k'": "' %24.16e (e(`k'))
        local first 0
    }
    quietly reghdfe y d, absorb(unit year)
    file write `fh' `"}, "twfe": "' %24.16e (_b[d]) `", "rows": ["'
    import delimited "`save'.csv", clear asdouble
    forvalues i = 1/`=_N' {
        if `i' > 1 file write `fh' ", "
        file write `fh' "[" %10.0f (dd_type[`i']) ", " %10.0f (time_lower[`i']) ", " %10.0f (time_upper[`i']) ", " %24.16e (dd_est[`i']) ", " %24.16e (weight[`i']) "]"
    }
    file write `fh' "]}"
    if `p' < 2 file write `fh' ","
    file write `fh' _n
}
file write `fh' "}" _n
file close `fh'
