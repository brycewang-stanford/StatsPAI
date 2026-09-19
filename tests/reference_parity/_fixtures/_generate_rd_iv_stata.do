* ---------------------------------------------------------------------------
* Stata reference for tests/reference_parity/test_rd_iv_R_parity.py (IV
* block of the rd_iv parity family, Stata side).
*
* Requires Stata 18 and, in a PRIVATE ado directory (never PLUS):
*   ssc install ivreg2 ranktest avar weakivtest weakiv
*   ssc install rdrobust
*   net install st0108, from(http://www.stata-journal.com/software/sj6-3)
*     (-jive- is not on SSC: -ssc describe jive- returns r(601); it ships
*      with Stata Journal 6(3) as package st0108)
* The block below installs them into _ado_rd_iv/ next to this file on
* first run.
*
* Run _generate_rd_iv_R.R first (writes rd_iv_ivw.csv / rd_iv_rueda.csv),
* then, from this directory:
*   stata-mp -b do _generate_rd_iv_stata.do
* Writes rd_iv_Stata.json (numbers as %23.16e).
*
* Conventions pinned here
* -----------------------
* * weakivtest (Montiel Olea & Pflueger) is run after ivreg2 with -robust-
*   or -cluster()-; r(F_eff) is the effective F. Its denominator is
*   tr(Sigma_hat Q_ZZ) with Z orthogonalised on the included exogenous
*   regressors.
* * weakiv ..., md small: minimum-distance (Wald) version with small-
*   sample adjustment, i.i.d. errors. CLR / K / AR are evaluated at
*   H0: beta = 0. The confidence-set endpoints are only available as the
*   display strings e(clr_cset) / e(k_cset) (6-7 significant digits).
* * rdrobust ..., deriv(1): the kink estimator (rdrobust is maintained by
*   the same authors in R and Stata); e(tau_cl) / e(tau_bc) / e(se_tau_cl)
*   / e(se_tau_rb) / e(h_l) / e(b_l). Reads rd_iv_kink.csv (written by
*   _generate_rd_iv_rd_R.R).
* * jive (st0108): ujive1 / ujive2 are the Angrist-Imbens-Krueger JIVE1 /
*   JIVE2 instruments in an IV second stage; default V is
*   s^2 (Xj'X)^-1 Xj'Xj (Xj'X)^-1' with s^2 from the DEMEANED residuals
*   (r(Var) * (N-1) / (N-k)) -- equal to e'e/(N-k) because the constant is
*   in the instrument set; -robust- is HC0. e(rmse) = sqrt(s^2) is stored
*   so the test can check that.
* ---------------------------------------------------------------------------
version 18
set more off
set type double

local here "`c(pwd)'"
local ado "`here'/_ado_rd_iv"
capture mkdir "`ado'"
net set ado "`ado'"
adopath ++ "`ado'"
foreach p in ivreg2 ranktest avar weakivtest weakiv rdrobust {
    capture which `p'
    if _rc ssc install `p', replace
}
capture which jive
if _rc net install st0108, from(http://www.stata-journal.com/software/sj6-3) replace

tempname fh
file open `fh' using "rd_iv_Stata.json", write replace text
file write `fh' "{" _n
file write `fh' `"  "meta": {"stata_version": "`c(stata_version)'", "'
foreach p in ivreg2 weakivtest weakiv jive rdrobust {
    * first "*!" line, or weakivtest's "* this version:" line
    quietly findfile `p'.ado
    file open _v using "`r(fn)'", read text
    local ver ""
    local i 0
    file read _v line
    while r(eof) == 0 & `"`ver'"' == "" & `i' < 40 {
        if strpos(`"`line'"', "*!") == 1 | strpos(`"`line'"', "this version:") {
            local ver = trim(subinstr(subinstr(`"`line'"', char(9), " ", .), `"""', "", .))
        }
        local ++i
        file read _v line
    }
    file close _v
    file write `fh' `""`p'": "`ver'", "'
}
file write `fh' `""note": "weakivtest after ivreg2; weakiv md small; jive st0108"},"' _n

* ---- weakivtest: effective F ----------------------------------------------
file write `fh' `"  "weakivtest": {"' _n
import delimited using "rd_iv_ivw.csv", clear asdouble
quietly ivreg2 y x1 x2 (d = z1), robust
quietly weakivtest
file write `fh' `"    "ivw1_hc": "' %23.16e (r(F_eff)) "," _n
quietly ivreg2 y x1 x2 (d = z1), cluster(cl)
quietly weakivtest
file write `fh' `"    "ivw1_cl": "' %23.16e (r(F_eff)) "," _n
quietly ivreg2 y x1 x2 (d = z1 z2 z3), robust
quietly weakivtest
file write `fh' `"    "ivw3_hc": "' %23.16e (r(F_eff)) "," _n
quietly ivreg2 y x1 x2 (d = z1 z2 z3), cluster(cl)
quietly weakivtest
file write `fh' `"    "ivw3_cl": "' %23.16e (r(F_eff)) "," _n
import delimited using "rd_iv_rueda.csv", clear asdouble
quietly ivreg2 e_vote_buying lpopulation lpotencial (lm_pob_mesa = lz_pob_mesa_f), robust
quietly weakivtest
file write `fh' `"    "rueda_hc": "' %23.16e (r(F_eff)) "," _n
quietly ivreg2 e_vote_buying lpopulation lpotencial (lm_pob_mesa = lz_pob_mesa_f), cluster(muni_code)
quietly weakivtest
file write `fh' `"    "rueda_cl": "' %23.16e (r(F_eff)) _n
file write `fh' "  }," _n

* ---- weakiv md small (i.i.d.), k = 3, H0: beta = 0 -------------------------
import delimited using "rd_iv_ivw.csv", clear asdouble
quietly weakiv ivreg2 y x1 x2 (d = z1 z2 z3), md small
file write `fh' `"  "weakiv_md_small_ivw3": {"clr_stat": "' %23.16e (e(clr_stat)) ", "
file write `fh' `""clr_p": "' %23.16e (e(clr_p)) ", "
file write `fh' `""k_chi2": "' %23.16e (e(k_chi2)) ", "
file write `fh' `""k_p": "' %23.16e (e(k_p)) ", "
file write `fh' `""ar_chi2": "' %23.16e (e(ar_chi2)) ", "
file write `fh' `""clr_cset": ""' "`e(clr_cset)'" `"", "'
file write `fh' `""k_cset": ""' "`e(k_cset)'" `""},"' _n

* ---- jive (st0108) ----------------------------------------------------------
file write `fh' `"  "jive": {"' _n
local first 1
foreach v in ujive1 ujive2 {
    foreach r in "" robust {
        quietly jive y x1 x2 (d = z1 z2 z3), `v' `r'
        matrix b = e(b)
        matrix V = e(V)
        local key = cond("`r'" == "", "`v'", "`v'_robust")
        if !`first' file write `fh' "," _n
        local first 0
        * e(b) order: d x1 x2 _cons
        file write `fh' `"    "`key'": {"b": ["' %23.16e (b[1,1]) ", " %23.16e (b[1,2]) ", " %23.16e (b[1,3]) ", " %23.16e (b[1,4]) "], "
        file write `fh' `""se": ["' %23.16e (sqrt(V[1,1])) ", " %23.16e (sqrt(V[2,2])) ", " %23.16e (sqrt(V[3,3])) ", " %23.16e (sqrt(V[4,4])) "], "
        file write `fh' `""rmse": "' %23.16e (e(rmse)) `", "N": "' %9.0f (e(N)) "}"
    }
}
file write `fh' _n "  }," _n

* ---- rdrobust, deriv(1) (regression kink) ------------------------------------
import delimited using "rd_iv_kink.csv", clear asdouble
file write `fh' `"  "kink": {"' _n
local specs `" "sharp_p1_hc1|y x, deriv(1) p(1) vce(hc1)" "sharp_pdef_nn|y x, deriv(1)" "sharp_p1_hc1_h04|y x, deriv(1) p(1) vce(hc1) h(0.4)" "fuzzy_p1_hc1|y x, fuzzy(t) deriv(1) p(1) vce(hc1)" "fuzzy_pdef_nn|y x, fuzzy(t) deriv(1)" "fuzzy_p1_hc1_h04|y x, fuzzy(t) deriv(1) p(1) vce(hc1) h(0.4)" "'
local first 1
foreach sp of local specs {
    gettoken key cmd : sp, parse("|")
    local cmd : subinstr local cmd "|" ""
    quietly rdrobust `cmd'
    if !`first' file write `fh' "," _n
    local first 0
    file write `fh' `"    "`key'": {"coef": ["' %23.16e (e(tau_cl)) ", " %23.16e (e(tau_bc)) "], "
    file write `fh' `""se": ["' %23.16e (e(se_tau_cl)) ", " %23.16e (e(se_tau_rb)) "], "
    file write `fh' `""h": "' %23.16e (e(h_l)) `", "b": "' %23.16e (e(b_l)) `", "p": "' %3.0f (e(p)) "}"
}
file write `fh' _n "  }" _n
file write `fh' "}" _n
file close `fh'
