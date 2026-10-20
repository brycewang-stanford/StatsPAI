* ---------------------------------------------------------------------------
* Stata reference for tests/reference_parity/test_misc_sens_stata_parity.py
* (round-2 "misc_sens" family). Stata 18, `set type double`.
*
* User-written commands go into a PRIVATE ado directory next to this file,
* never PLUS:  _ado_misc_sens/  ietoolkit 7.5 (iebaltab, SSC) and leebounds
* 1.5 (SSC, Tauchmann). The do-file installs them when missing.
*
* Run _generate_misc_sens_data.py and _generate_misc_sens_R.R first (the R
* script writes the imputed datasets misc_sens_mi_imputed.csv), then from
* this directory:
*   stata-mp -b do _generate_misc_sens_stata.do
* Writes misc_sens_stata.json (numbers as %23.16e).
*
* Conventions pinned here
* -----------------------
* * mi estimate (flong import of R's five mice imputations): Barnard-Rubin
*   small-sample df (the default for regress; nu_c = e(df_r) = 196), and
*   p-values and CIs from t(df_mi). e(b_mi) orders x1 x2 x3 _cons. The
*   small-sample FMI e(fmi_mi) is Barnard & Rubin's (1999, p. 953)
*   1 - lambda(nu_small)/lambda(nu_c) U/T with lambda(u) = (u+1)/(u+3),
*   not mice's (riv + 2/(df+3))/(riv+1) (see [MI] mi estimate, Methods and
*   formulas).
* * regress, vce(robust) is HC1 with t(N - k) inference; vce(cluster) uses
*   G/(G-1) (N-1)/(N-k) and t(G - 1).
* * testparm after regress reports F(q, N - k) (or F(q, G - 1) clustered);
*   StatsPAI's subgroup test reports the Wald chi2 = q F.
* * tabulate, chi2 is Pearson's chi2 without Yates' continuity correction.
* * iebaltab (default vce): per-variable pair test is the OLS t-test of the
*   variable on a group dummy (pooled variance); pair "0_1" is group 0 minus
*   group 1; nrmd = diff / sqrt((var_0 + var_1)/2); ftest regresses the
*   group dummy on all balance variables and tests them jointly (F).
* * leebounds: default vce(analytic); thresholds via _pctile. As shipped it
*   stores each threshold in a local macro (~15 digits), so which side of the
*   rounded threshold the quantile observation falls decides whether it is
*   kept; here the lower bound drops it (keeps floor((1-q) n)) and the upper
*   bound takes the fractional tie branch. The patched copy (block 4b) holds
*   the thresholds exactly and computes fractional trimming on both sides.
* ---------------------------------------------------------------------------
version 18
clear all
set more off
set type double

local here "`c(pwd)'"
local ado "`here'/_ado_misc_sens"
capture mkdir "`ado'"
net set ado "`ado'"
adopath ++ "`ado'"
capture which iebaltab
if _rc ssc install ietoolkit, replace
capture which leebounds
if _rc ssc install leebounds, replace

tempname fh
file open `fh' using "misc_sens_stata.json", write replace text
file write `fh' "{" _n

capture program drop wrow
program define wrow
    * wrow handle "name" matrix  -> "name": [a, b, ...],
    args fh name M
    local k = colsof(`M')
    file write `fh' `"  "`name'": ["'
    forvalues j = 1/`k' {
        local sep = cond(`j' < `k', ", ", "")
        file write `fh' %23.16e (`M'[1, `j']) "`sep'"
    }
    file write `fh' "]," _n
end

capture program drop wscal
program define wscal
    args fh name val
    file write `fh' `"  "`name'": "' %23.16e (`val') "," _n
end

* ---- 1. mi estimate on the five R imputations ------------------------------
import delimited using "misc_sens_mi.csv", clear asdouble
gen mm = 0
gen mid = _n
tempfile orig
save `orig'
import delimited using "misc_sens_mi_imputed.csv", clear asdouble
rename imp mm
rename id mid
append using `orig'
mi import flong, m(mm) id(mid) imputed(y x2) clear
mi estimate: regress y x1 x2 x3
matrix b = e(b_mi)
matrix V = e(V_mi)
matrix se = vecdiag(V)
forvalues j = 1/`=colsof(se)' {
    matrix se[1, `j'] = sqrt(se[1, `j'])
}
wrow `fh' "mi_b" b
wrow `fh' "mi_se" se
matrix df = e(df_mi)
matrix fmi = e(fmi_mi)
wrow `fh' "mi_df" df
wrow `fh' "mi_fmi" fmi
matrix T = r(table)
matrix p = T["pvalue", 1...]
matrix lo = T["ll", 1...]
matrix hi = T["ul", 1...]
wrow `fh' "mi_p" p
wrow `fh' "mi_ci_lo" lo
wrow `fh' "mi_ci_hi" hi

* ---- 2. subgroup_analysis: y ~ d + x1 + x2 within x3 and g3 ---------------
import delimited using "misc_sens_ovb.csv", clear asdouble
foreach vce in robust ols {
    local opt = cond("`vce'" == "robust", "vce(robust)", "")
    quietly regress y d x1 x2, `opt'
    matrix r = (_b[d], _se[d], e(df_r))
    wrow `fh' "sub_`vce'_overall" r
    foreach g in x3 g3 {
        quietly levelsof `g', local(levs)
        foreach v of local levs {
            quietly regress y d x1 x2 if `g' == `v', `opt'
            matrix T = r(table)
            matrix r = (_b[d], _se[d], T["pvalue", "d"], T["ll", "d"], T["ul", "d"], e(N))
            wrow `fh' "sub_`vce'_`g'_`v'" r
        }
        quietly regress y d x1 x2 i.`g' i.`g'#c.d, `opt'
        quietly testparm i.`g'#c.d
        matrix r = (r(F), r(df), r(df_r), r(p))
        wrow `fh' "sub_`vce'_het_`g'" r
    }
}

* ---- 3. ancova / negd -------------------------------------------------------
encode region, gen(reg)
quietly regress post grp pre x2 i.reg, vce(robust)
matrix T = r(table)
matrix r = (_b[grp], _se[grp], T["pvalue", "grp"], T["ll", "grp"], T["ul", "grp"], e(N))
wrow `fh' "ancova_hc1" r
quietly regress post grp pre x2 i.reg, vce(cluster cl)
matrix T = r(table)
matrix r = (_b[grp], _se[grp], T["pvalue", "grp"], T["ll", "grp"], T["ul", "grp"], e(N_clust))
wrow `fh' "ancova_cluster" r
quietly regress post grp pre, vce(robust)
matrix T = r(table)
matrix r = (_b[grp], _se[grp], T["pvalue", "grp"], T["ll", "grp"], T["ul", "grp"])
wrow `fh' "negd_ancova" r
gen chg = post - pre
quietly regress chg grp x2, vce(robust)
matrix T = r(table)
matrix r = (_b[grp], _se[grp], T["pvalue", "grp"], T["ll", "grp"], T["ul", "grp"])
wrow `fh' "negd_change" r

* ---- 4. attrition -----------------------------------------------------------
import delimited using "misc_sens_attr.csv", clear asdouble
quietly tabulate treat obs, chi2
matrix r = (r(chi2), r(p))
wrow `fh' "attr_chi2" r
gen attrit = 1 - obs
foreach v in age inc {
    quietly regress attrit `v'
    matrix T = r(table)
    matrix r = (_b[`v'], _se[`v'], T["pvalue", "`v'"])
    wrow `fh' "attr_reg_`v'" r
}
quietly leebounds y treat, select(obs)
matrix b = e(b)
matrix V = e(V)
matrix r = (b[1,1], b[1,2], sqrt(V[1,1]), sqrt(V[2,2]), e(trim))
wrow `fh' "attr_lee" r
* the same with the treated arm relabelled, so the control arm is trimmed
gen treat0 = 1 - treat
quietly leebounds y treat0, select(obs)
matrix b = e(b)
matrix r = (b[1,1], b[1,2])
wrow `fh' "attr_lee_flip" r
* leebounds with the trimming thresholds held exactly (the round-1 teffects
* patch): a patched copy that holds them as exact doubles (%21x) instead of
* rounded decimal text. Owner decision pending (2026-10-21 cleanup): this
* patch depends on the ado's text.
local adop "`here'/_ado_misc_sens_patched"
capture mkdir "`adop'"
capture mkdir "`adop'/l"
tempfile p1
filefilter "`ado'/l/leebounds.ado" "`p1'", from("local uth = r(r1)") to("local uth : display %21x r(r1)") replace
filefilter "`p1'" "`adop'/l/leebounds.ado", from("local lth = r(r1)") to("local lth : display %21x r(r1)") replace
adopath - "`ado'"
adopath ++ "`adop'"
discard
quietly leebounds y treat, select(obs)
matrix b = e(b)
matrix V = e(V)
matrix r = (b[1,1], b[1,2], sqrt(V[1,1]), sqrt(V[2,2]))
wrow `fh' "attr_lee_exact" r
adopath - "`adop'"
adopath ++ "`ado'"
discard

* ---- 5. balance: iebaltab ----------------------------------------------------
quietly iebaltab b1 b2 b3, grpvar(treat) ftest
matrix R = r(iebtab_rmat)
matrix F = r(iebtab_fmat)
foreach v in 1 2 3 {
    local rr = `v'
    matrix r = (R[`rr', "mean_1"], R[`rr', "mean_0"], R[`rr', "var_1"], R[`rr', "var_0"], ///
                R[`rr', "diff_0_1"], R[`rr', "t_0_1"], R[`rr', "p_0_1"], R[`rr', "nrmd_0_1"])
    wrow `fh' "bal_b`v'" r
}
matrix r = (F[1, "ff_0_1"], F[1, "fp_0_1"], F[1, "fn_0_1"])
wrow `fh' "bal_ftest" r

file write `fh' `"  "versions": {"stata": "`c(stata_version)'", "iebaltab": "7.5", "leebounds": "1.5"}"' _n
file write `fh' "}" _n
file close `fh'
