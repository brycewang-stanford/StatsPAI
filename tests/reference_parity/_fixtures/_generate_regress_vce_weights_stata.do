*! _generate_regress_vce_weights_stata.do
*! regress under analytic weights and under the cluster-type variances
*! that had no reference: CR2, two-way clustering, delete-one-cluster
*! jackknife. For tests/reference_parity/test_regress_vce_weights_stata_parity.py.
*! Stata 18 MP, official commands only. Run from this directory: writes
*! regress_vce_weights_stata.json.
*!
*! Double precision throughout (set type double + asdouble): without it
*! import delimited stores decimals as float; see
*! tests/stata_parity/option_parity/README.md.
clear all
set more off
set type double
import delimited "../../stata_translation_holdout/holdout_cross.csv", clear asdouble case(preserve)
tempname fh
file open `fh' using "regress_vce_weights_stata.json", write replace
file write `fh' "{" _n
program define _w
    args fh key
    file write `fh' `"  ""' "`key'" `"": {"N": "' (e(N)) `", "df_r": "' (e(df_r))
    foreach v in x1 x2 _cons {
        file write `fh' `", "b_`v'": "' %24.16e (_b[`v']) `", "se_`v'": "' %24.16e (_se[`v'])
    }
    file write `fh' "}," _n
end
* analytic weights x the variances regress offers
qui regress y x1 x2 [aw=w]
_w `fh' "aw_classical"
qui regress y x1 x2 [aw=w], vce(robust)
_w `fh' "aw_hc1"
qui regress y x1 x2 [aw=w], vce(hc2)
_w `fh' "aw_hc2"
qui regress y x1 x2 [aw=w], vce(hc3)
_w `fh' "aw_hc3"
qui regress y x1 x2 [aw=w], vce(cluster g)
_w `fh' "aw_cr1"
qui regress y x1 x2 [aw=w], vce(hc2 g)
_w `fh' "aw_cr2"
qui regress y x1 x2 [aw=w], vce(cluster g k)
_w `fh' "aw_twoway"
* unweighted cluster-type variances
qui regress y x1 x2, vce(hc2 g)
_w `fh' "cr2"
qui regress y x1 x2, vce(cluster g k)
_w `fh' "twoway"
qui regress y x1 x2, vce(jackknife, cluster(g))
_w `fh' "jackknife_cluster"
qui regress y x1 x2, vce(jackknife, cluster(g) mse)
_w `fh' "jackknife_cluster_mse"
file write `fh' `"  "_meta": {"stata": "' (c(stata_version)) `", "data": "tests/stata_translation_holdout/holdout_cross.csv"}"' _n
file write `fh' "}" _n
file close `fh'
