* ---------------------------------------------------------------------------
* Stata reference for tests/reference_parity/test_ldv_design_stata_parity.py
*
* Requires: Stata 18 (official commands only; nothing to install).
* Run:      stata -b do _generate_ldv_design_stata.do   (from this directory)
*
* What this fixture pins
* ----------------------
* tobit with vce(robust), vce(cluster g), sampling weights [pw=w] and both,
* for left censoring at 0 and for two-sided censoring; and ML heckman
* (Stata's default) under the same variance / weight options; and qreg at
* three quantiles under vce(iid) and vce(robust) (this Stata 18 build's qreg
* refuses vce(cluster)).  Before 1.32
* sp.tobit offered only the observed-information variance and no weights,
* and sp.heckman only the two-step estimator.
*
* Stata runs with tolerance(1e-12) ltolerance(1e-14) nrtolerance(1e-12): at
* its defaults ml stops ~1e-6 (relative) short of the optimum, which
* sp.tobit reaches by Newton polish (log-likelihood higher by ~3e-11).
*
* Data is generated and exported HERE so both sides read the same bytes.
* ---------------------------------------------------------------------------
version 18
clear all
set type double
set seed 20260926
set obs 1200

gen long   id = _n
gen int    g  = mod(_n, 48) + 1
gen double cu = rnormal()
bysort g (id): replace cu = cu[1]
sort id
gen double x1 = rnormal() + 0.4*cu
gen double x2 = runiform()
gen double w  = exp(0.5*x2 + 0.3*rnormal())
gen double ys = 0.3 + (0.8 + 0.4*x2)*x1 - 0.6*x2 + 0.7*cu + rnormal()
gen double y  = max(ys, 0)
gen double y2 = min(max(ys, 0), 1.5)
drop ys
* Heckman selection data, drawn after the Tobit data so those draws are
* unchanged: z3 is the exclusion restriction, corr(u, v) = 0.6.
gen double z3 = rnormal()
gen double hu = rnormal()
gen double hv = 0.6*hu + sqrt(1 - 0.36)*rnormal()
gen byte   s  = (0.2 + 0.5*x1 - 0.4*x2 + 0.8*z3 + 0.3*cu + hu) > 0
gen double yh = 1 + 0.9*x1 + 0.5*x2 + 0.5*cu + 1.3*hv if s
drop hu hv
* Quantile-regression outcome, heteroskedastic in x2 so iid and robust
* standard errors differ; drawn last so the earlier draws are unchanged.
gen double yq = 1 + 0.7*x1 - 0.4*x2 + 0.6*cu + (0.5 + x2)*rnormal()

format x1 x2 w y y2 z3 yh yq %21.16e
export delimited id g x1 x2 w y y2 z3 s yh yq using "ldv_design_data.csv", replace datafmt

file open fh using "ldv_design_stata.json", write replace text
file write fh "{" _n
foreach spec in "left_oim|y|ll(0)||" "left_robust|y|ll(0)||vce(robust)" ///
                "left_cluster|y|ll(0)||vce(cluster g)" "left_pw|y|ll(0)|[pw=w]|" ///
                "left_pw_cluster|y|ll(0)|[pw=w]|vce(cluster g)" ///
                "two_oim|y2|ll(0) ul(1.5)||" "two_pw_cluster|y2|ll(0) ul(1.5)|[pw=w]|vce(cluster g)" {
    tokenize "`spec'", parse("|")
    * tokens: 1 tag, 3 depvar, 5 limits, 7 weight (or |), then options
    local tag `1'
    local dv `3'
    local lim `5'
    if "`7'" == "|" {
        local wexp ""
        local opt "`8'"
    }
    else {
        local wexp "`7'"
        local opt "`9'"
    }
    quietly tobit `dv' x1 x2 `wexp', `lim' `opt' tolerance(1e-12) ltolerance(1e-14) nrtolerance(1e-12)
    file write fh `"  "`tag'": {"' _n
    foreach t in x1 x2 _cons {
        file write fh `"    "b_`t'": "' %23.16e (_b[`dv':`t']) `","' _n
        file write fh `"    "se_`t'": "' %23.16e (_se[`dv':`t']) `","' _n
    }
    file write fh `"    "sigma": "' %23.16e (sqrt(_b[/var(e.`dv')])) `","' _n
    file write fh `"    "ll": "' %23.16e (e(ll)) `","' _n
    file write fh `"    "N": "' %12.0f (e(N)) _n
    file write fh `"  },"' _n
}
* ---- heckman (ML, Stata's default) -------------------------------------
foreach spec in "heckman_oim||" "heckman_robust||vce(robust)" ///
                "heckman_cluster||vce(cluster g)" "heckman_pw|[pw=w]|" ///
                "heckman_pw_cluster|[pw=w]|vce(cluster g)" {
    tokenize "`spec'", parse("|")
    local tag `1'
    if "`3'" == "|" {
        local wexp ""
        local opt "`4'"
    }
    else {
        local wexp "`3'"
        local opt "`5'"
    }
    quietly heckman yh x1 x2 `wexp', select(s = x1 x2 z3) `opt' ///
        tolerance(1e-12) ltolerance(1e-14) nrtolerance(1e-12)
    file write fh `"  "`tag'": {"' _n
    foreach t in x1 x2 _cons {
        file write fh `"    "b_`t'": "' %23.16e (_b[yh:`t']) `","' _n
        file write fh `"    "se_`t'": "' %23.16e (_se[yh:`t']) `","' _n
    }
    foreach t in x1 x2 z3 _cons {
        file write fh `"    "bs_`t'": "' %23.16e (_b[s:`t']) `","' _n
        file write fh `"    "ses_`t'": "' %23.16e (_se[s:`t']) `","' _n
    }
    file write fh `"    "athrho": "' %23.16e (_b[/athrho]) `", "se_athrho": "' %23.16e (_se[/athrho]) `","' _n
    file write fh `"    "lnsigma": "' %23.16e (_b[/lnsigma]) `", "se_lnsigma": "' %23.16e (_se[/lnsigma]) `","' _n
    file write fh `"    "rho": "' %23.16e (e(rho)) `", "sigma": "' %23.16e (e(sigma)) `","' _n
    file write fh `"    "lambda": "' %23.16e (e(lambda)) `", "se_lambda": "' %23.16e (e(selambda)) `","' _n
    file write fh `"    "ll": "' %23.16e (e(ll)) `", "N": "' %12.0f (e(N)) `", "N_selected": "' %12.0f (e(N) - e(N_cens)) _n
    file write fh `"  },"' _n
}
* ---- qreg (fitted sparsity, Hall-Sheather bandwidth: Stata's defaults) --
foreach q in 25 50 75 {
    foreach v in iid robust {
        quietly qreg yq x1 x2, quantile(0.`q') vce(`v')
        * replay to post r(table): p-values and intervals use t(e(df_r))
        quietly qreg
        matrix T = r(table)
        file write fh `"  "qreg_`q'_`v'": {"' _n
        local i = 0
        foreach t in x1 x2 _cons {
            local ++i
            file write fh `"    "b_`t'": "' %23.16e (_b[`t']) `","' _n
            file write fh `"    "se_`t'": "' %23.16e (_se[`t']) `","' _n
            file write fh `"    "p_`t'": "' %23.16e (T[4, `i']) `","' _n
            file write fh `"    "ll_`t'": "' %23.16e (T[5, `i']) `","' _n
            file write fh `"    "ul_`t'": "' %23.16e (T[6, `i']) `","' _n
        }
        file write fh `"    "df_r": "' %12.0f (e(df_r)) `","' _n
        file write fh `"    "bwidth": "' %23.16e (e(bwidth)) `", "N": "' %12.0f (e(N)) _n
        file write fh `"  },"' _n
    }
}
file write fh `"  "_meta": {"' _n
file write fh `"    "stata_version": "' `"""' "`c(stata_version)'" `"""' `","' _n
file write fh `"    "flavor": "' `"""' "`c(flavor)' `c(edition_real)'" `"""' `","' _n
file write fh `"    "seed": 20260926,"' _n
file write fh `"    "generated": "' `"""' "`c(current_date)'" `"""' _n
file write fh `"  }"' _n
file write fh "}" _n
file close fh
display "wrote ldv_design_stata.json and ldv_design_data.csv"
