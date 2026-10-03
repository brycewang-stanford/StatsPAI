*! _generate_joint_test_stata.do
*! Joint Wald tests after regress / ivregress / logit / poisson / xtreg, fe
*! under each variance option, for tests/reference_parity/
*! test_joint_wald_stata_parity.py. Stata 18 MP, official commands only.
*! Run from this directory: writes joint_test_stata.json.
*!
*! Double precision throughout (set type double + asdouble): without it
*! import delimited stores decimals as float; see
*! tests/stata_parity/option_parity/README.md.
clear all
set more off
set type double
import delimited "../../stata_translation_holdout/holdout_cross.csv", clear asdouble case(preserve)
tempname fh
file open `fh' using "joint_test_stata.json", write replace
file write `fh' "{" _n
program define _w
    args fh key
    local stat = cond(missing(r(F)), r(chi2), r(F))
    local kind = cond(missing(r(F)), "chi2", "F")
    local dfr = cond(missing(r(df_r)), -1, r(df_r))
    file write `fh' `"  ""' "`key'" `"": {"kind": ""' "`kind'" `"", "stat": "' %24.16e (`stat') `", "df": "' (r(df)) `", "df_r": "' (`dfr') `", "p": "' %24.16e (r(p)) "}," _n
end
foreach v in ols robust hc2 hc3 cluster {
    local opt ""
    if "`v'" == "robust" local opt "vce(robust)"
    if "`v'" == "hc2" local opt "vce(hc2)"
    if "`v'" == "hc3" local opt "vce(hc3)"
    if "`v'" == "cluster" local opt "vce(cluster g)"
    qui regress y x1 x2 i.k, `opt'
    qui testparm i.k
    _w `fh' "regress_`v'_factor"
    qui test x1 = x2
    _w `fh' "regress_`v'_equal"
    qui test x1 x2
    _w `fh' "regress_`v'_both"
}
foreach v in ols robust cluster {
    local opt ""
    if "`v'" == "robust" local opt "vce(robust)"
    if "`v'" == "cluster" local opt "vce(cluster g)"
    qui ivregress 2sls y x1 (x2 = z z2), `opt'
    qui test x1 x2
    _w `fh' "iv_`v'_both"
    qui test x1 = x2
    _w `fh' "iv_`v'_equal"
    * `small`: degrees-of-freedom adjusted variance and F / t statistics,
    * which is the convention sp.ivreg reports.
    qui ivregress 2sls y x1 (x2 = z z2), `opt' small
    qui test x1 x2
    _w `fh' "iv_`v'_small_both"
    qui test x1 = x2
    _w `fh' "iv_`v'_small_equal"
}
foreach v in ols robust {
    local opt ""
    if "`v'" == "robust" local opt "vce(robust)"
    qui logit yb x1 x2 i.k, `opt'
    qui testparm i.k
    _w `fh' "logit_`v'_factor"
    qui poisson cnt x1 x2 i.k, `opt'
    qui test x1 x2
    _w `fh' "poisson_`v'_both"
}
import delimited "../../stata_translation_holdout/holdout_panel.csv", clear asdouble case(preserve)
xtset id t
foreach v in ols cluster {
    local opt ""
    if "`v'" == "cluster" local opt "vce(cluster id)"
    qui xtreg y x i.t, fe `opt'
    qui testparm i.t
    _w `fh' "xtfe_`v'_time"
}
file write `fh' `"  "_meta": {"stata": "18 MP", "precision": "double", "data": "tests/stata_translation_holdout/holdout_{cross,panel}.csv"}"' _n
file write `fh' "}" _n
file close `fh'
set type float
