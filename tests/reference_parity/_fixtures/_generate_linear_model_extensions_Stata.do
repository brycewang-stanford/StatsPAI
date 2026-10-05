* Reference values for tests/reference_parity/test_linear_model_extensions_parity.py
* Stata 18. Run from this folder:
*     do _generate_linear_model_extensions_Stata.do
* Reads linear_model_extensions.csv, writes linear_model_extensions_Stata.csv
* with one row per (model, quantity).
clear all
set more off
import delimited using "linear_model_extensions.csv", clear case(preserve)
gen double ly = log(y)
xtset id t

tempname H
postfile `H' str40 model str24 term double value using "_lme_stata_tmp.dta", replace

foreach fam in gaussian binomial poisson {
    local dep = cond("`fam'" == "gaussian", "ly", cond("`fam'" == "binomial", "d", "c"))
    foreach cs in independent exchangeable ar1 {
        foreach opt in "" "nmp" {
            local tag = cond("`opt'" == "", "n", "nmp")
            local corr = cond("`cs'" == "ar1", "ar 1", "`cs'")
            xtgee `dep' treat x1 x2 x4, family(`fam') corr(`corr') `opt' tolerance(1e-12) iterate(500)
            foreach v in _cons treat x1 x2 x4 {
                post `H' ("`fam'_`cs'_`tag'") ("b_`v'") (_b[`v'])
                post `H' ("`fam'_`cs'_`tag'") ("se_model_`v'") (_se[`v'])
            }
            post `H' ("`fam'_`cs'_`tag'") ("scale") (e(phi))
            matrix Rw = e(R)
            post `H' ("`fam'_`cs'_`tag'") ("alpha") (Rw[1, 2])
            xtgee `dep' treat x1 x2 x4, family(`fam') corr(`corr') `opt' vce(robust) tolerance(1e-12) iterate(500)
            foreach v in _cons treat x1 x2 x4 {
                post `H' ("`fam'_`cs'_`tag'") ("se_robust_`v'") (_se[`v'])
            }
        }
    }
}

boxcox y x1 x2 x3 x4 x5 x6 treat, model(lhsonly) iterate(200)
post `H' ("boxcox") ("lambda") (_b[/theta])
post `H' ("boxcox") ("ll") (e(ll))
post `H' ("boxcox") ("ll_m1") (e(ll_tm1))
post `H' ("boxcox") ("ll_0") (e(ll_t0))
post `H' ("boxcox") ("ll_1") (e(ll_t1))
post `H' ("boxcox") ("chi2_m1") (e(chi2_tm1))
post `H' ("boxcox") ("chi2_0") (e(chi2_t0))
post `H' ("boxcox") ("chi2_1") (e(chi2_t1))

stset time, failure(event)
stcox treat x1 x2 x4, efron vce(robust) nohr
foreach v in treat x1 x2 x4 {
    post `H' ("cox_efron_robust") ("b_`v'") (_b[`v'])
    post `H' ("cox_efron_robust") ("se_`v'") (_se[`v'])
}
stcox treat x1 x2 x4, efron vce(cluster id) nohr
foreach v in treat x1 x2 x4 {
    post `H' ("cox_efron_cluster") ("se_`v'") (_se[`v'])
}

* Kaplan-Meier with sts's default (log-log) interval
sts generate km_s = s
sts generate km_lb = lb(s)
sts generate km_ub = ub(s)
foreach tt in 5 15 30 60 {
    quietly summarize time if time <= `tt' & event == 1
    local tmax = r(max)
    foreach q in s lb ub {
        quietly summarize km_`q' if time == `tmax'
        post `H' ("km") ("`q'_`tt'") (r(mean))
    }
}

postclose `H'
use "_lme_stata_tmp.dta", clear
format value %21.15e
export delimited using "linear_model_extensions_Stata.csv", replace
erase "_lme_stata_tmp.dta"
