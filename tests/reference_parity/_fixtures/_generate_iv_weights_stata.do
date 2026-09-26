* ---------------------------------------------------------------------------
* Stata reference for tests/reference_parity/test_iv_weights_stata_parity.py
*
* Requires: Stata 18 (official ivregress only; nothing to install).
* Run:      stata -b do _generate_iv_weights_stata.do   (from this directory)
*
* What this fixture pins
* ----------------------
* ivregress 2sls / liml with analytic weights [aw=w] under the default,
* vce(robust) and vce(cluster g) standard errors (all with `small`, the
* t / F convention sp.iv reports).  Before 1.32, sp.iv / sp.ivreg accepted
* weights= and dropped it, returning the unweighted estimate.
*
* The effect of d varies with w, so the weighted and unweighted estimands
* differ by far more than any tolerance: an ignored weight cannot pass.
* Twenty outcomes are missing so the weight column must stay aligned with
* the rows ivregress (and sp.iv) keep.
*
* Data is generated and exported HERE so both sides read the same bytes.
* ---------------------------------------------------------------------------
version 18
clear all
set type double
set seed 20260926
set obs 900

gen long   id = _n
gen double x1 = rnormal()
gen double z1 = rnormal()
gen double z2 = rnormal()
gen double w  = 0.2 + 4.8*runiform()
gen int    g  = mod(_n, 45) + 1
gen double v  = rnormal()
gen double d  = 0.8*z1 + 0.4*z2 + 0.3*x1 + v
gen double y  = 1 + (1 + 0.5*w)*d + 0.5*x1 + v + rnormal()
replace y = . if mod(_n, 45) == 7
drop v

format x1 z1 z2 w d y %21.16e
export delimited id x1 z1 z2 w g d y using "iv_weights_data.csv", replace datafmt

file open fh using "iv_weights_stata.json", write replace text
file write fh "{" _n
foreach m in 2sls liml {
    foreach v in unadjusted robust cluster {
        local vopt = cond("`v'" == "cluster", "vce(cluster g)", "vce(`v')")
        quietly ivregress `m' y x1 (d = z1 z2) [aw=w], small `vopt'
        file write fh `"  "`m'_`v'": {"' _n
        foreach t in d x1 _cons {
            file write fh `"    "b_`t'": "' %21.16e (_b[`t']) `","' _n
            file write fh `"    "se_`t'": "' %21.16e (_se[`t']) `","' _n
        }
        file write fh `"    "r2": "' %21.16e (e(r2)) `","' _n
        file write fh `"    "N": "' %21.16e (e(N)) _n
        file write fh `"  },"' _n
    }
}
* Unweighted 2SLS on the same sample: the guard that weights moved the answer.
quietly ivregress 2sls y x1 (d = z1 z2), small
file write fh `"  "2sls_unweighted": {"' _n
file write fh `"    "b_d": "' %21.16e (_b[d]) _n
file write fh `"  },"' _n
file write fh `"  "_meta": {"' _n
file write fh `"    "stata_version": "' `"""' "`c(stata_version)'" `"""' `","' _n
file write fh `"    "flavor": "' `"""' "`c(flavor)' `c(edition_real)'" `"""' `","' _n
file write fh `"    "seed": 20260926,"' _n
file write fh `"    "generated": "' `"""' "`c(current_date)'" `"""' _n
file write fh `"  }"' _n
file write fh "}" _n
file close fh
display "wrote iv_weights_stata.json and iv_weights_data.csv"
