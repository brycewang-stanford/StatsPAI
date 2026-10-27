* Stata fuzzydid (de Chaisemartin, D'Haultfoeuille & Guyonvarch, SSC,
* Distribution-Date 20190507) on the fixtures written by
* _generate_fuzzydid_data.py. Reference for tests/reference_parity/
* test_fuzzy_did_stata_parity.py.
*
*   python tests/reference_parity/_fixtures/_generate_fuzzydid_data.py
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_fuzzydid_Stata.do
*
* fuzzydid is installed into a private ado directory (never the user's
* PLUS): set FUZZYDID_ADO to it, e.g.
*   net set ado "<dir>"
*   ssc install fuzzydid
* Point estimates are deterministic. The bootstrap rows (breps(500)) are a
* stochastic screen: the Python side draws its own resamples.
version 18
clear all
set type double
local ado : env FUZZYDID_ADO
if "`ado'" != "" adopath ++ "`ado'"
local fx "tests/reference_parity/_fixtures"

tempname fh
file open `fh' using "`fx'/fuzzydid_Stata.json", write replace
file write `fh' "{" _n

capture program drop _wmat
program define _wmat
    * _wmat handle key matrix last(0/1)
    args fh key mat last
    tempname M
    matrix `M' = `mat'
    local r = rowsof(`M')
    local c = colsof(`M')
    local names : rownames `M'
    file write `fh' `"  "`key'": {"'
    forvalues i = 1/`r' {
        local nm : word `i' of `names'
        if `i' > 1 file write `fh' ", "
        if `c' == 1 {
            file write `fh' `""`nm'": "' %24.16e (`M'[`i', 1])
        }
        else {
            file write `fh' `""`nm'": ["' %24.16e (`M'[`i', 1]) ", " %24.16e (`M'[`i', 2]) "]"
        }
    }
    file write `fh' "}"
    if `last' == 0 file write `fh' ","
    file write `fh' _n
end

* ---------------------------------------------------------------- 2x2
import delimited using "`fx'/fuzzydid_2x2.csv", clear asdouble
file write `fh' `"  "N_2x2": "' (_N) "," _n
qui fuzzydid y g t d, did tc cic nose
_wmat `fh' "2x2_base" e(b_LATE) 0
qui fuzzydid y g t d, did tc cic nose numerator
_wmat `fh' "2x2_numerator" e(b_LATE) 0
qui fuzzydid y g t d, did tc nose continuous(x1)
_wmat `fh' "2x2_x1" e(b_LATE) 0
qui fuzzydid y g t d, did tc nose continuous(x1) qualitative(x2)
_wmat `fh' "2x2_x1_x2" e(b_LATE) 0
qui fuzzydid y g t d, did tc nose continuous(x1) modelx(ols logit)
_wmat `fh' "2x2_x1_ols_logit" e(b_LATE) 0
qui fuzzydid y g t d, did tc nose continuous(x1) modelx(ols probit)
_wmat `fh' "2x2_x1_ols_probit" e(b_LATE) 0
qui fuzzydid yb g t d, did tc nose continuous(x1) modelx(logit logit)
_wmat `fh' "2x2_yb_x1_logit_logit" e(b_LATE) 0
* where the 1e-8 gaps of the logit / probit rows come from: logit's
* default stopping rule against a tight one, on one cell's E(D|X) model
qui logit d x1 if g == 1 & t == 0
matrix _bdef = e(b)'
_wmat `fh' "logit_cell10_default" _bdef 0
qui logit d x1 if g == 1 & t == 0, tolerance(1e-14) ltolerance(1e-14) nrtolerance(1e-14)
matrix _btight = e(b)'
_wmat `fh' "logit_cell10_tight" _btight 0
set seed 20261028
qui fuzzydid y g t d, did tc cic breps(500)
_wmat `fh' "2x2_boot_se" e(se_LATE) 0
_wmat `fh' "2x2_boot_ci" e(ci_LATE) 0
set seed 20261028
qui fuzzydid y g t d, did tc cic breps(500) cluster(cl)
_wmat `fh' "2x2_boot_cluster_se" e(se_LATE) 0
set seed 20261028
qui fuzzydid y g t d, did tc cic breps(500) eqtest
_wmat `fh' "2x2_eqtest_b" e(b_LATE_eqtest) 0
_wmat `fh' "2x2_eqtest_se" e(se_LATE_eqtest) 0

* ---------------------------------------------------------------- ordered
import delimited using "`fx'/fuzzydid_ordered.csv", clear asdouble
file write `fh' `"  "N_ordered": "' (_N) "," _n
qui fuzzydid y g t d, did tc cic nose
_wmat `fh' "ordered_base" e(b_LATE) 0
qui fuzzydid y g t d, did tc cic nose numerator
_wmat `fh' "ordered_numerator" e(b_LATE) 0
qui fuzzydid y g t d, did tc cic nose newcateg(1 3)
_wmat `fh' "ordered_newcateg_1_3" e(b_LATE) 0
qui fuzzydid y g t d, did tc nose continuous(x1)
_wmat `fh' "ordered_x1" e(b_LATE) 0
qui fuzzydid y g t d, did tc nose continuous(x1) modelx(ols ols logit)
_wmat `fh' "ordered_x1_ols_ols_logit" e(b_LATE) 0
qui fuzzydid y g t d, did tc nose continuous(x1) modelx(ols ols probit)
_wmat `fh' "ordered_x1_ols_ols_probit" e(b_LATE) 0

* ---------------------------------------------------------------- multi
import delimited using "`fx'/fuzzydid_multi.csv", clear asdouble
file write `fh' `"  "N_multi": "' (_N) "," _n
qui fuzzydid y gt gf t d, did tc cic nose
_wmat `fh' "multi_base" e(b_LATE) 0
qui fuzzydid y gt gf t d, did tc cic nose newcateg(0 2)
_wmat `fh' "multi_newcateg_0_2" e(b_LATE) 0
qui fuzzydid y gt gf t d, did tc nose continuous(x1)
_wmat `fh' "multi_x1" e(b_LATE) 0
set seed 20261028
qui fuzzydid y gt gf t d, did tc cic breps(500)
_wmat `fh' "multi_boot_se" e(se_LATE) 1

file write `fh' "}" _n
file close `fh'
