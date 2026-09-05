* tests/stata_parity/86_fect.do
*
* Module 86: fect counterfactual estimators (Liu, Wang and Xu 2024).
*   StatsPAI:  sp.fect(..., method = "fe" / "ife" / "mc")
*   R:         fect::fect(Y ~ D + X1 + X2, method = ..., force = "two-way",
*                         se = FALSE, CV = FALSE, tol = 1e-8)
*   Stata:     fect Y, treat(D) unit(id) time(time) cov(X1 X2)
*                   method("fe"|"ife"|"mc") force("two-way") tol(1e-12) maxiterations(20000)
*
* The Stata port (fect_stata, by the same authors) is not on SSC
* (`ssc describe fect` returns r(601)); it is installed from GitHub into
* a local ado directory next to this do-file (tests/stata_parity/_ado_fect,
* git-ignored) the first time the module runs, together with its SSC
* dependency _gwtmean. reghdfe is required from the regular PLUS path.
*
* Rows mirror the R/Python sides: <m>_att_avg, <m>_beta_x1, <m>_beta_x2,
* <m>_cons_reghdfe (fect_stata's reghdfe constant, a diagnostic that is
* not fect's mu and is deliberately not joined), and <m>_att_on_<k>
* (n = cell count) in
* fect's relative-period coding (0 = last untreated period, 1 = first
* treated period). fect_stata does not report att.avg.unit or the
* pre-treatment RMSE, so those rows are R/Python-only.

version 18
clear all

do _common.do
stata_parity_init, module(86_fect)
stata_parity_open, module(86_fect)

* ---- local ado path for the GitHub-only bridge ---------------------------
local ado_fect "`c(pwd)'/_ado_fect"
capture mkdir "`ado_fect'"
adopath + "`ado_fect'"
capture which fect
if _rc != 0 {
    local plus_saved : sysdir PLUS
    sysdir set PLUS "`ado_fect'"
    net install fect, from("https://raw.githubusercontent.com/xuyiqing/fect_stata/master/") replace
    capture which _gwtmean
    if _rc != 0 {
        ssc install _gwtmean, replace
    }
    sysdir set PLUS "`plus_saved'"
}
capture which _gwtmean
if _rc != 0 {
    local plus_saved : sysdir PLUS
    sysdir set PLUS "`ado_fect'"
    ssc install _gwtmean, replace
    sysdir set PLUS "`plus_saved'"
}
which fect
which reghdfe

import delimited "${STATA_PARITY_DATA}/86_fect.csv", clear case(preserve) asdouble
local n = _N

foreach m in fe ife mc {
    if "`m'" == "fe" {
        fect Y, treat(D) unit(id) time(time) cov(X1 X2) method("fe") force("two-way") tol(1e-12) maxiterations(20000)
    }
    else if "`m'" == "ife" {
        fect Y, treat(D) unit(id) time(time) cov(X1 X2) method("ife") r(2) force("two-way") tol(1e-12) maxiterations(20000)
    }
    else {
        fect Y, treat(D) unit(id) time(time) cov(X1 X2) method("mc") lambda(0.002) force("two-way") tol(1e-12) maxiterations(20000)
    }
    matrix ATT = e(ATT)
    matrix C = e(coefs)
    matrix ATTs = e(ATTs)
    local att = ATT[1, 1]
    local n_tr = ATT[1, 2]
    local cons = C[1, "cons"]
    local b1 = C[1, "X1"]
    local b2 = C[1, "X2"]
    stata_parity_row, statname(`m'_att_avg) estimate(`att') nobs(`n')
    stata_parity_row, statname(`m'_beta_x1) estimate(`b1') nobs(`n')
    stata_parity_row, statname(`m'_beta_x2) estimate(`b2') nobs(`n')
    * fect_stata's e(coefs) "cons" is the reghdfe constant of the untreated
    * two-way regression, not fect's grand mean mu, so it is recorded as an
    * un-joined diagnostic row rather than compared with the R/Python mu.
    stata_parity_row, statname(`m'_cons_reghdfe) estimate(`cons') nobs(`n')
    local K = rowsof(ATTs)
    forvalues j = 1/`K' {
        local s = ATTs[`j', 1]
        local cnt = ATTs[`j', 2]
        local a = ATTs[`j', 3]
        local s_int = round(`s')
        stata_parity_row, statname(`m'_att_on_`s_int') estimate(`a') nobs(`cnt')
    }
}

stata_parity_extra, key(method) val("fect_stata (GitHub xuyiqing/fect_stata), force two-way, tol 1e-8")
stata_parity_extra, key(r_ife) val("2")
stata_parity_extra, key(lambda_mc) val("0.002")
stata_parity_extra, key(install_note) val("fect is not on SSC (ssc describe fect: r(601)); installed with net install from raw.githubusercontent.com/xuyiqing/fect_stata/master into tests/stata_parity/_ado_fect plus ssc install _gwtmean")
stata_parity_extra, key(stata_command) val("fect Y, treat(D) unit(id) time(time) cov(X1 X2) method(fe|ife|mc) force(two-way) tol(1e-12) maxiterations(20000) [r(2) | lambda(0.002)]")
stata_parity_close, module(86_fect)
