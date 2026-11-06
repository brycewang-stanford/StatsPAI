* Stata 18 MP reference numbers for
* tests/reference_parity/test_regcmd_stata_r_parity.py:
*   A. intreg      (regcmd_cs.csv): oim, robust, cluster, [pw], [pw] + cluster
*   B. heckprobit  (regcmd_cs.csv): oim, robust, cluster, [pw], [pw] + cluster
*   C. gologit2    (regcmd_cs.csv): all / no / some parallel lines, logit and
*                  probit links, robust and cluster
*   D. prais       (textbook_ts.csv): prais / corc x six rhotypes x iterated /
*                  twostep, iterated to tolerance(1e-14); robust, hc2, hc3
*   E. nl          (textbook_cs.csv): vce gnr / robust / cluster / hc2 / hc3
*   F. mprobit     (textbook_cs.csv): base outcome 1 and 3, no constants
*   G. boottest    (regcmd_wild.csv): WRE after ivregress 2sls, small,
*                  Rademacher weights enumerated over all 2^12 sign vectors;
*                  ptolerance(1e-12) so that the confidence-set search is not
*                  stopped at its default precision (1e-3)
*   H. regress / ivreg2 on x-only z-scores (ivreport_fx.csv), for
*      sp.standardize_coefs(scale_y=False)
* All ML fits run to tolerance(1e-13) ltolerance(1e-15) nrtolerance(1e-12),
* so the comparison is not limited by where Stata's iterations stop.
*
* gologit2 3.2.8 (SSC) is installed into a private ado tree that is
* gitignored (tests/reference_parity/_fixtures/_ado_regcmd/), never into the
* user's PLUS; boottest and ivreg2 are taken from the adopath.
*   python tests/reference_parity/_fixtures/_generate_regcmd_data.py
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_regcmd_Stata.do
* (run from the repository root)
version 18
clear all
set more off
set type double
local ado "`c(pwd)'/tests/reference_parity/_fixtures/_ado_regcmd"
capture mkdir "`ado'"
adopath ++ "`ado'"
capture findfile gologit2.ado, path("`ado'/g")
if _rc {
    local plus "`c(sysdir_plus)'"
    sysdir set PLUS "`ado'"
    ssc install gologit2, replace
    sysdir set PLUS "`plus'"
}
local MLTOL "tolerance(1e-13) ltolerance(1e-15) nrtolerance(1e-12) iterate(500)"

capture program drop jvec
program define jvec
    * "key": {"names": [...], "b": [...], "se": [...], "N": #, "ll": #}
    args fh key
    tempname B V
    matrix `B' = e(b)
    matrix `V' = e(V)
    local names : colfullnames `B'
    local k = colsof(`B')
    file write `fh' `"  "`key'": {"names": ["'
    forvalues j = 1/`k' {
        local nm : word `j' of `names'
        if `j' > 1 file write `fh' ", "
        file write `fh' `""`nm'""'
    }
    file write `fh' "]," _n `"    "b": ["'
    forvalues j = 1/`k' {
        if `j' > 1 file write `fh' ", "
        file write `fh' %24.17e (`B'[1, `j'])
    }
    file write `fh' "]," _n `"    "se": ["'
    forvalues j = 1/`k' {
        if `j' > 1 file write `fh' ", "
        if missing(sqrt(`V'[`j', `j'])) file write `fh' "null"
        else file write `fh' %24.17e (sqrt(`V'[`j', `j']))
    }
    file write `fh' "]," _n `"    "N": "' (e(N))
    if !missing(e(ll)) file write `fh' `", "ll": "' %24.17e (e(ll))
    if !missing(e(converged)) file write `fh' `", "converged": "' (e(converged))
    if !missing(e(N_clust)) file write `fh' `", "N_clust": "' (e(N_clust))
    file write `fh' "}," _n
end

capture program drop jnum
program define jnum
    * "key": #
    args fh key value
    if missing(`value') file write `fh' `"  "`key'": null,"' _n
    else file write `fh' `"  "`key'": "' %24.17e (`value') "," _n
end

tempname fh
file open `fh' using "tests/reference_parity/_fixtures/regcmd_Stata.json", write replace
file write `fh' "{" _n
file write `fh' `"  "stata_version": "`c(stata_version)'", "stata_born": "`c(born_date)'", "edition": "MP","' _n
quietly which gologit2
file write `fh' `"  "gologit2": "3.2.8","' _n

* =============================================================== A. intreg
import delimited using "tests/reference_parity/_fixtures/regcmd_cs.csv", clear asdouble
intreg ilo ihi x1 x2 x3, `MLTOL'
jvec `fh' "intreg_oim"
jnum `fh' "intreg_oim_N_unc" e(N_unc)
jnum `fh' "intreg_oim_N_lc" e(N_lc)
jnum `fh' "intreg_oim_N_rc" e(N_rc)
jnum `fh' "intreg_oim_N_int" e(N_int)
intreg ilo ihi x1 x2 x3, vce(robust) `MLTOL'
jvec `fh' "intreg_robust"
intreg ilo ihi x1 x2 x3, vce(cluster g) `MLTOL'
jvec `fh' "intreg_cluster"
intreg ilo ihi x1 x2 x3 [pw=pw], `MLTOL'
jvec `fh' "intreg_pw"
intreg ilo ihi x1 x2 x3 [pw=pw], vce(cluster g) `MLTOL'
jvec `fh' "intreg_pw_cluster"

* =========================================================== B. heckprobit
heckprobit ybh x1 x2 x3, select(s = z x1) `MLTOL'
jvec `fh' "heckprobit_oim"
jnum `fh' "heckprobit_oim_rho" e(rho)
jnum `fh' "heckprobit_oim_N_selected" e(N_selected)
heckprobit ybh x1 x2 x3, select(s = z x1) vce(robust) `MLTOL'
jvec `fh' "heckprobit_robust"
heckprobit ybh x1 x2 x3, select(s = z x1) vce(cluster g) `MLTOL'
jvec `fh' "heckprobit_cluster"
heckprobit ybh x1 x2 x3 [pw=pw], select(s = z x1) `MLTOL'
jvec `fh' "heckprobit_pw"
heckprobit ybh x1 x2 x3 [pw=pw], select(s = z x1) vce(cluster g) `MLTOL'
jvec `fh' "heckprobit_pw_cluster"

* ============================================================= C. gologit2
gologit2 yo x1 x2 x3, `MLTOL'
jvec `fh' "gologit_npl"
gologit2 yo x1 x2 x3, pl `MLTOL'
jvec `fh' "gologit_pl"
gologit2 yo x1 x2 x3, npl(x1) `MLTOL'
jvec `fh' "gologit_npl_x1"
gologit2 yo x1 x2 x3, pl(x2 x3) `MLTOL'
jvec `fh' "gologit_pl_x2x3"
gologit2 yo x1 x2 x3, link(probit) `MLTOL'
jvec `fh' "gologit_probit_npl"
gologit2 yo x1 x2 x3, link(probit) npl(x1) `MLTOL'
jvec `fh' "gologit_probit_npl_x1"
gologit2 yo x1 x2 x3, npl(x1) vce(robust) `MLTOL'
jvec `fh' "gologit_npl_x1_robust"
gologit2 yo x1 x2 x3, npl(x1) vce(cluster g) `MLTOL'
jvec `fh' "gologit_npl_x1_cluster"
gologit2 yo x1 x2 x3, link(probit) pl vce(cluster g) `MLTOL'
jvec `fh' "gologit_probit_pl_cluster"

* ================================================================ D. prais
import delimited using "tests/reference_parity/_fixtures/textbook_ts.csv", clear asdouble
tsset t
foreach m in prais corc {
    local opt = cond("`m'" == "corc", "corc", "")
    foreach rt in regress freg tscorr dw theil nagar {
        foreach step in iter two {
            local two = cond("`step'" == "two", "twostep", "")
            quietly prais y x1 x2 d, rhotype(`rt') `opt' `two' tolerance(1e-14) iterate(5000)
            jvec `fh' "prais_`m'_`rt'_`step'"
            jnum `fh' "prais_`m'_`rt'_`step'_rho" e(rho)
            jnum `fh' "prais_`m'_`rt'_`step'_r2" e(r2)
            jnum `fh' "prais_`m'_`rt'_`step'_F" e(F)
            jnum `fh' "prais_`m'_`rt'_`step'_rmse" e(rmse)
            jnum `fh' "prais_`m'_`rt'_`step'_dw" e(dw)
            jnum `fh' "prais_`m'_`rt'_`step'_dw0" e(dw_0)
        }
    }
    foreach v in robust hc2 hc3 {
        quietly prais y x1 x2 d, `opt' vce(`v') tolerance(1e-14) iterate(5000)
        jvec `fh' "prais_`m'_`v'"
        jnum `fh' "prais_`m'_`v'_rho" e(rho)
        jnum `fh' "prais_`m'_`v'_F" e(F)
    }
}

* =================================================================== E. nl
import delimited using "tests/reference_parity/_fixtures/textbook_cs.csv", clear asdouble
generate double xp = exp(x1 / 2)
generate double ynl = 2 + 3 * xp^0.5 + z2 / 4
foreach v in gnr robust "cluster g" hc2 hc3 {
    local tag = word("`v'", 1)
    quietly nl (ynl = {a} + {b} * xp^{c}), initial(a 1 b 1 c 1) vce(`v') eps(1e-14) iterate(2000)
    jvec `fh' "nl_`tag'"
    jnum `fh' "nl_`tag'_rss" e(rss)
}

* ============================================================== F. mprobit
generate byte mode = 1 + (x1 + z1 > 0) + (x1 - x2 + z2 > 0.5)
mprobit mode x1 x2, baseoutcome(1) intpoints(120) `MLTOL'
jvec `fh' "mprobit_base1"
mprobit mode x1 x2, baseoutcome(3) intpoints(120) `MLTOL'
jvec `fh' "mprobit_base3"
mprobit mode x1 x2, baseoutcome(1) noconstant intpoints(120) `MLTOL'
jvec `fh' "mprobit_base1_nocons"

* ============================================================= G. boottest
import delimited using "tests/reference_parity/_fixtures/regcmd_wild.csv", clear asdouble
capture program drop jboot
program define jboot
    * "key": {"p": #, "t": #, "reps": #, "ci": [[lo, hi], ...],
    *         "p_at_ci": [[p(lo), p(hi)], ...]}
    * p_at_ci re-runs the test at each reported bound, so the p-value
    * function can be compared there exactly.
    args fh key var opts
    tempname CI
    matrix `CI' = r(CI)
    local p = r(p)
    local t = r(t)
    local reps = r(reps)
    local rows = rowsof(`CI')
    file write `fh' `"  "`key'": {"p": "' %24.17e (`p') `", "t": "' %24.17e (`t')
    file write `fh' `", "reps": "' (`reps') `", "ci": ["'
    forvalues i = 1/`rows' {
        if `i' > 1 file write `fh' ", "
        file write `fh' "[" %24.17e (`CI'[`i', 1]) ", " %24.17e (`CI'[`i', 2]) "]"
    }
    file write `fh' `"], "p_at_ci": ["'
    forvalues i = 1/`rows' {
        if `i' > 1 file write `fh' ", "
        file write `fh' "["
        forvalues j = 1/2 {
            local bound = `CI'[`i', `j']
            quietly boottest `var' = `bound', `opts'
            if `j' > 1 file write `fh' ", "
            file write `fh' %24.17e (r(p))
        }
        file write `fh' "]"
    }
    file write `fh' "]}," _n
end
local BT "reps(9999) weighttype(rademacher) ptolerance(1e-12) nograph"
ivregress 2sls y w (d1 = z1 z2), vce(cluster cl) small
jvec `fh' "iv_over"
boottest d1, `BT'
jboot `fh' "wre_over_b0" d1 "`BT'"
boottest d1 = 0.5, `BT'
jboot `fh' "wre_over_b05" d1 "`BT'"
boottest d1, `BT' level(90)
jboot `fh' "wre_over_l90" d1 "`BT' level(90)"
* alpha * 2^12 = 256 is an integer: the interpolated bootstrap quantile is an
* order statistic and the confidence set is exactly {p > alpha}
boottest d1, `BT' level(93.75)
jboot `fh' "wre_over_l9375" d1 "`BT' level(93.75)"
ivregress 2sls y w (d1 = z1), vce(cluster cl) small
jvec `fh' "iv_just"
boottest d1, `BT'
jboot `fh' "wre_just_b0" d1 "`BT'"
ivregress 2sls y (d1 = z1 z2 z3), vce(cluster cl) small
jvec `fh' "iv_nocontrol"
boottest d1, `BT'
jboot `fh' "wre_nocontrol_b0" d1 "`BT'"
ivregress 2sls y w (d1 d2 = z1 z2 z3), vce(cluster cl) small
jvec `fh' "iv_two"
boottest d1, `BT'
jboot `fh' "wre_two_d1" d1 "`BT'"
boottest d2, `BT'
jboot `fh' "wre_two_d2" d2 "`BT'"
boottest d2, `BT' level(93.75)
jboot `fh' "wre_two_d2_l9375" d2 "`BT' level(93.75)"
boottest d1, `BT' level(93.75)
jboot `fh' "wre_two_d1_l9375" d1 "`BT' level(93.75)"

* ================================================ H. x-only standardisation
import delimited using "tests/reference_parity/_fixtures/ivreport_fx.csv", clear asdouble
foreach v in x z w1 w2 d {
    quietly summarize `v'
    generate double s_`v' = (`v' - r(mean)) / r(sd)
}
quietly regress y s_x s_w1 s_w2 s_d, vce(robust)
jvec `fh' "stdx_regress_robust"
quietly regress y s_x w1 w2 d, vce(cluster region)
jvec `fh' "stdx_only_x_regress_cluster"
quietly ivreg2 y s_w1 s_w2 s_d (s_x = s_z), cluster(region) small
jvec `fh' "stdx_ivreg2_cluster_small"

file write `fh' `"  "end": null"' _n "}" _n
file close `fh'
