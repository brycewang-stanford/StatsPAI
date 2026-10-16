* ---------------------------------------------------------------------------
* Stata reference for tests/reference_parity/test_ldv_gaps_stata_parity.py
*
* Fills validation-scope cells the ldv_design fixture leaves empty, on the
* same CSV bytes (ldv_design_data.csv, written by _generate_ldv_design_stata.do):
*   tobit two-sided [0, 1.5]: vce(robust), vce(cluster g), [pw=w]
*   tobit right-censored at 1.5: vce(oim), vce(robust), vce(cluster g)
*   qreg [pw=w] at three quantiles (Stata's robust variance for pweights)
* Requires Stata 18. Run from this directory:  stata -b do _generate_ldv_gaps_stata.do
* ---------------------------------------------------------------------------
version 18
clear all
set type double
import delimited using "ldv_design_data.csv", clear asdouble

file open fh using "ldv_gaps_stata.json", write replace text
file write fh "{" _n
foreach spec in "two_robust|ll(0) ul(1.5)||vce(robust)" ///
                "two_cluster|ll(0) ul(1.5)||vce(cluster g)" ///
                "two_pw|ll(0) ul(1.5)|[pw=w]|" ///
                "right_oim|ul(1.5)||" "right_robust|ul(1.5)||vce(robust)" ///
                "right_cluster|ul(1.5)||vce(cluster g)" {
    tokenize "`spec'", parse("|")
    local tag `1'
    local lim `3'
    if "`5'" == "|" {
        local wexp ""
        local opt "`6'"
    }
    else {
        local wexp "`5'"
        local opt "`7'"
    }
    quietly tobit y2 x1 x2 `wexp', `lim' `opt' tolerance(1e-12) ltolerance(1e-14) nrtolerance(1e-12)
    file write fh `"  "`tag'": {"' _n
    foreach t in x1 x2 _cons {
        file write fh `"    "b_`t'": "' %23.16e (_b[y2:`t']) `","' _n
        file write fh `"    "se_`t'": "' %23.16e (_se[y2:`t']) `","' _n
    }
    file write fh `"    "N": "' %12.0f (e(N)) _n
    file write fh `"  },"' _n
}
foreach q in 25 50 75 {
    quietly qreg yq x1 x2 [pw=w], quantile(0.`q')
    file write fh `"  "qreg_pw_q`q'": {"' _n
    foreach t in x1 x2 _cons {
        file write fh `"    "b_`t'": "' %23.16e (_b[`t']) `","' _n
        file write fh `"    "se_`t'": "' %23.16e (_se[`t']) `","' _n
    }
    file write fh `"    "vce": ""' "`e(vce)'" `"", "N": "' %12.0f (e(N)) _n
    if `q' == 75 file write fh `"  }"' _n
    else file write fh `"  },"' _n
}
file write fh "}" _n
file close fh
