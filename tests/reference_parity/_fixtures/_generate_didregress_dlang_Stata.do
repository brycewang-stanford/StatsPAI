* Reference numbers for sp.didregress(aggregate=...) from Stata 18.
*
*   python tests/reference_parity/_fixtures/_generate_didregress_dlang_data.py
*   cd tests/reference_parity/_fixtures
*   stata-mp -b do _generate_didregress_dlang_Stata.do
*
* Writes didregress_dlang_Stata.csv: one row per case with the ATET, its
* standard error, e(df_r), e(N) and the confidence bounds Stata reports.
set more off
set type double
tempname H
postfile `H' str24 case double(b se df_r N lb ub) using _dlang_results, replace

program define _post
    args H case
    tempname T
    matrix `T' = r(table)
    post `H' ("`case'") (_b[r1vs0.d]) (_se[r1vs0.d]) (e(df_r)) (e(N)) ///
        (`T'[5, 1]) (`T'[6, 1])
end

import delimited using didregress_dlang.csv, clear
didregress (y x w) (d), group(g) time(t) aggregate(dlang)
_post `H' rcs_dlang_xw
didregress (y x w) (d), group(g) time(t) aggregate(dlang, varying)
_post `H' rcs_varying_xw
didregress (y x w z) (d), group(g) time(t) aggregate(dlang)
_post `H' rcs_dlang_xwz
didregress (y x w z) (d), group(g) time(t) aggregate(dlang, varying)
_post `H' rcs_varying_xwz
didregress (y) (d), group(g) time(t) aggregate(dlang)
_post `H' rcs_dlang_none
didregress (y x w) (d), group(g) time(t) aggregate(standard)
_post `H' rcs_standard_xw
didregress (y x w z) (d), group(g) time(t) aggregate(standard)
_post `H' rcs_standard_xwz

import delimited using didregress_dlang_panel.csv, clear
xtset id t
xtdidregress (y x) (d), group(g) time(t) aggregate(dlang)
_post `H' panel_dlang_x
xtdidregress (y x) (d), group(g) time(t) aggregate(dlang, varying)
_post `H' panel_varying_x
xtdidregress (y x) (d), group(g) time(t) aggregate(standard)
_post `H' panel_standard_x

postclose `H'
use _dlang_results, clear
format b se lb ub %21.17g
export delimited using didregress_dlang_Stata.csv, replace datafmt
erase _dlang_results.dta
