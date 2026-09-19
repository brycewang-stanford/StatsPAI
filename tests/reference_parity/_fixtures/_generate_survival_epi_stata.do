* ---------------------------------------------------------------------------
* Stata references for tests/reference_parity/test_survival_epi_R_parity.py
* (Stata-side block). Requires Stata 18 and the SSC packages stcompet
* (Coviello & Boggess) and diagt (Seed), installed into a PRIVATE ado
* directory next to this file -- never into PLUS:
*
*   net set ado "<this dir>/_ado_survival_epi"
*   ssc install stcompet ; ssc install diagt
*
* Run _generate_survival_epi_data.py first, then from this directory:
*   stata-mp -b do _generate_survival_epi_stata.do
* Writes survival_epi_stata.json (numbers as %23.16e: %g drops the leading
* zero, which JSON forbids).
*
* Conventions pinned here
* -----------------------
* * stcompet: Marubini-Valsecchi delta-method SE; log(-log) bounds
*   CIF^exp(+-z se / (CIF log CIF)).
* * stcrreg: Fine-Gray; vce(robust) is its only variance, which is the
*   Fine-Gray sandwich times N/(N-1).
* * stcox, shared(): gamma frailty (variance theta) by penalised partial
*   likelihood; e(ll) is the integrated log likelihood, e(V) the inverse of
*   the full penalised information (beta block), conditional on theta.
* * dstdize: binomial variance sum w^2 r (1 - r) / n, normal CI with the
*   lower bound truncated at 0. istdize: exact Poisson (cii, poisson) CI.
* * cc ..., by() bd tarone: Breslow-Day with the M-H common OR.
* * roctab: default SE is DeLong's; hanley gives Hanley-McNeil.
* * diagti a b c d = (true+/test+, true+/test-, true-/test+, true-/test-);
*   exact binomial CIs; results in percent.
* * kdensity: kernels on Stata's scales (epan2, biweight, rectangle,
*   triangle have support [-1, 1]); default width
*   0.9 min(sd, IQR/1.349) N^(-1/5) with summarize's percentiles.
* * lpoly se(): (X'WX)^-1 X'W^2X (X'WX)^-1 sigma2(x0), sigma2(x0) from a
*   degree p+2 fit with pilot width pwidth().
* * power twoproportions: Pearson chi-squared test (pooled null), group 1
*   = controls (p0), group 2 = cases (p1).
* ---------------------------------------------------------------------------
version 18
set more off
set type double
local here "`c(pwd)'"
adopath ++ "`here'/_ado_survival_epi"

tempname fh
file open `fh' using "survival_epi_stata.json", write replace text
file write `fh' "{" _n

capture program drop jarr
program define jarr
    * jarr handle varname: write the non-missing values of varname as [..]
    args fh v
    file write `fh' "["
    local first 1
    forvalues i = 1/`=_N' {
        if !missing(`v'[`i']) {
            if !`first' file write `fh' ", "
            file write `fh' %23.16e (`v'[`i'])
            local first 0
        }
    }
    file write `fh' "]"
end

* ---- stcompet ---------------------------------------------------------------
import delimited using "survival_epi_cr.csv", clear asdouble
stset time, failure(status==1)
stcompet ci=ci se=se hi=hi lo=lo, compet1(2)
preserve
keep if status==1
bysort time: keep if _n==1
sort time
file write `fh' `"  "stcompet_cause1": {"time": "'
jarr `fh' time
file write `fh' `", "ci": "'
jarr `fh' ci
file write `fh' `", "se": "'
jarr `fh' se
file write `fh' `", "lo": "'
jarr `fh' lo
file write `fh' `", "hi": "'
jarr `fh' hi
file write `fh' "}," _n
restore
preserve
keep if status==2
bysort time: keep if _n==1
sort time
file write `fh' `"  "stcompet_cause2": {"time": "'
jarr `fh' time
file write `fh' `", "ci": "'
jarr `fh' ci
file write `fh' `", "se": "'
jarr `fh' se
file write `fh' "}," _n
restore

* ---- stcrreg ----------------------------------------------------------------
foreach c in 1 2 {
    local o = 3 - `c'
    quietly stset time, failure(status==`c')
    quietly stcrreg x1 x2 grp, compete(status==`o')
    matrix b = e(b)
    matrix V = e(V)
    file write `fh' `"  "stcrreg_cause`c'": {"b": ["' %23.16e (b[1,1]) ", " %23.16e (b[1,2]) ", " %23.16e (b[1,3]) "], "
    file write `fh' `""se": ["' %23.16e (sqrt(V[1,1])) ", " %23.16e (sqrt(V[2,2])) ", " %23.16e (sqrt(V[3,3])) "], "
    file write `fh' `""ll": "' %23.16e (e(ll)) ", " `""N": "' %23.16e (e(N)) "}," _n
}

* ---- stcox, Breslow (default) and Efron ties, no competing events -----------------
preserve
drop if status == 2
quietly stset time, failure(status==1)
foreach t in breslow efron {
    quietly stcox x1 x2, nohr `t'
    matrix b = e(b)
    matrix V = e(V)
    file write `fh' `"  "stcox_`t'": {"b": ["' %23.16e (b[1,1]) ", " %23.16e (b[1,2]) "], "
    file write `fh' `""se": ["' %23.16e (sqrt(V[1,1])) ", " %23.16e (sqrt(V[2,2])) "], "
    file write `fh' `""ll": "' %23.16e (e(ll)) ", " `""ll_0": "' %23.16e (e(ll_0)) "}," _n
}
restore

* ---- stcox, shared() ---------------------------------------------------------
import delimited using "survival_epi_frailty.csv", clear asdouble
stset time, failure(event)
quietly stcox x1 x2, shared(cid) nohr
matrix b = e(b)
matrix V = e(V)
file write `fh' `"  "stcox_shared": {"b": ["' %23.16e (b[1,1]) ", " %23.16e (b[1,2]) "], "
file write `fh' `""se": ["' %23.16e (sqrt(V[1,1])) ", " %23.16e (sqrt(V[2,2])) "], "
file write `fh' `""theta": "' %23.16e (e(theta)) ", " `""ll": "' %23.16e (e(ll)) ", "
file write `fh' `""ll_c": "' %23.16e (e(ll_c)) ", " `""chi2_c": "' %23.16e (e(chi2_c)) "}," _n

* ---- dstdize / istdize --------------------------------------------------------
import delimited using "survival_epi_std.csv", clear asdouble
preserve
keep age std_pop
rename std_pop pop
tempfile stdpop
save `stdpop'
restore
preserve
gen byte grp = 1
quietly dstdize events pop age, by(grp) using(`stdpop')
matrix A = r(adj)
matrix L = r(lb)
matrix U = r(ub)
matrix S = r(se)
matrix C = r(crude)
file write `fh' `"  "dstdize": {"adj": "' %23.16e (A[1,1]) ", " `""lb": "' %23.16e (L[1,1]) ", "
file write `fh' `""ub": "' %23.16e (U[1,1]) ", " `""se": "' %23.16e (S[1,1]) ", " `""crude": "' %23.16e (C[1,1]) "}," _n
restore
* istdize: the data in memory are the study population, whose case
* variable holds the TOTAL observed count on every stratum row; the using
* file is the standard (reference) population with stratum-specific cases.
preserve
keep age ref_events ref_pop
tempfile refpop
save `refpop'
restore
egen double obs_total = total(events)
quietly istdize obs_total pop age using `refpop', popvars(ref_events ref_pop)
matrix M = r(smr)
matrix L = r(lb_smr)
matrix U = r(ub_smr)
matrix E = r(cases_exp)
file write `fh' `"  "istdize": {"smr": "' %23.16e (M[1,1]) ", " `""lb": "' %23.16e (L[1,1]) ", "
file write `fh' `""ub": "' %23.16e (U[1,1]) ", " `""cases_exp": "' %23.16e (E[1,1]) "}," _n

* ---- Breslow-Day (cc, by() bd tarone) ------------------------------------------
import delimited using "survival_epi_bd.csv", clear asdouble
expand 4
bysort stratum: gen byte cell = _n
gen byte exposed = inlist(cell, 1, 2)
gen byte case = inlist(cell, 1, 3)
gen double w = cond(cell==1, a, cond(cell==2, b, cond(cell==3, c, d)))
quietly cc case exposed [fw=w], by(stratum) bd tarone
file write `fh' `"  "cc_bd": {"or_mh": "' %23.16e (r(or)) ", " `""bd": "' %23.16e (r(chi2_bd)) ", "
file write `fh' `""p_bd": "' %23.16e (chi2tail(r(df_bd), r(chi2_bd))) ", " `""tarone": "' %23.16e (r(chi2_t)) ", "
file write `fh' `""p_tarone": "' %23.16e (chi2tail(r(df_t), r(chi2_t))) ", " `""df_bd": "' %23.16e (r(df_bd)) "}," _n

* ---- roctab --------------------------------------------------------------------
import delimited using "survival_epi_roc.csv", clear asdouble
foreach s in score score_tied {
    quietly roctab y `s'
    file write `fh' `"  "roctab_`s'": {"area": "' %23.16e (r(area)) ", " `""se": "' %23.16e (r(se)) ", "
    file write `fh' `""lb": "' %23.16e (r(lb)) ", " `""ub": "' %23.16e (r(ub)) ", "
    quietly roctab y `s', hanley
    file write `fh' `""se_hanley": "' %23.16e (r(se)) ", " `""lb_hanley": "' %23.16e (r(lb)) ", "
    file write `fh' `""ub_hanley": "' %23.16e (r(ub)) "}," _n
}

* ---- diagti ---------------------------------------------------------------------
import delimited using "survival_epi_diag.csv", clear asdouble
forvalues k = 1/`=_N' {
    local a = tp[`k']
    local b = fn[`k']
    local c = fp[`k']
    local d = tn[`k']
    quietly diagti `a' `b' `c' `d'
    file write `fh' `"  "diagti_`k'": {"sens": ["' %23.16e (r(sens)) ", " %23.16e (r(sens_lb)) ", " %23.16e (r(sens_ub)) "], "
    file write `fh' `""spec": ["' %23.16e (r(spec)) ", " %23.16e (r(spec_lb)) ", " %23.16e (r(spec_ub)) "], "
    file write `fh' `""ppv": "' %23.16e (r(ppv)) ", " `""npv": "' %23.16e (r(npv)) ", "
    file write `fh' `""lrp": "' %23.16e (r(lrp)) ", " `""lrn": "' %23.16e (r(lrn)) ", " `""prev": "' %23.16e (r(prev)) "}," _n
}

* ---- kdensity / lpoly ----------------------------------------------------------
import delimited using "survival_epi_at.csv", clear asdouble
tempfile at
save `at'
import delimited using "survival_epi_kd.csv", clear asdouble
merge 1:1 _n using `at', nogenerate
quietly summarize x, detail
file write `fh' `"  "summarize_x": {"p25": "' %23.16e (r(p25)) ", " `""p75": "' %23.16e (r(p75)) ", " `""sd": "' %23.16e (r(sd)) "}," _n
quietly kdensity x, nograph n(10) generate(gx gd)
file write `fh' `"  "kdensity_default": {"bwidth": "' %23.16e (r(bwidth)) `", "x": "'
jarr `fh' gx
file write `fh' `", "density": "'
jarr `fh' gd
file write `fh' "}," _n
file write `fh' `"  "kdensity_at": {"at": "'
jarr `fh' at
foreach k in epan2 biweight gaussian rectangle triangle {
    quietly kdensity x, at(at) bwidth(0.7) kernel(`k') generate(d_`k') nograph
    file write `fh' `", "`k'": "'
    jarr `fh' d_`k'
}
file write `fh' "}," _n
file write `fh' `"  "lpoly_at": {"at": "'
jarr `fh' at
local j 0
foreach spec in "1 epan2" "0 epan2" "2 gaussian" "1 biweight" {
    local j = `j' + 1
    tokenize `spec'
    quietly lpoly y x, at(at) bwidth(0.8) degree(`1') kernel(`2') ///
        generate(f`j') se(s`j') pwidth(1.2) nograph
    file write `fh' `", "fit_`1'_`2'": "'
    jarr `fh' f`j'
    file write `fh' `", "se_`1'_`2'": "'
    jarr `fh' s`j'
}
file write `fh' "}," _n

* ---- power twoproportions (case-control) --------------------------------------
local p0 = 0.3
local p1 = 2*`p0'/(1+`p0')
file write `fh' `"  "power_cc": {"'
quietly power twoproportions `p0' `p1', n1(400) n2(200)
file write `fh' `""n200_r2": "' %23.16e (r(power)) ", "
quietly power twoproportions `p0' `p1', n1(200) n2(200)
file write `fh' `""n200_r1": "' %23.16e (r(power)) ", "
quietly power twoproportions `p0' `p1', n1(200) n2(200) onesided
file write `fh' `""n200_r1_one": "' %23.16e (r(power)) ", "
quietly power twoproportions `p0' `p1', n1(60) n2(30)
file write `fh' `""n30_r2": "' %23.16e (r(power)) ", "
quietly power twoproportions `p0' `p1', n1(150) n2(50)
file write `fh' `""n50_r3": "' %23.16e (r(power)) "}," _n

file write `fh' `"  "provenance": {"stata_version": "`c(stata_version)'", "'
file write `fh' `""stata_born_date": "`c(born_date)'", "'
file write `fh' `""stcompet": "1.0.7 (06nov2012)", "diagt": "2.032 (diagti 2.053)", "'
file write `fh' `""generated_by": "tests/reference_parity/_fixtures/_generate_survival_epi_stata.do"}"' _n
file write `fh' "}" _n
file close `fh'
