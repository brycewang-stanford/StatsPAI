* ---------------------------------------------------------------------------
* Stata reference for tests/reference_parity/test_teffects_design_stata_parity.py
*
* Requires: Stata 18 (official teffects only; nothing to install).
* Run:      stata -b do _generate_teffects_design_stata.do   (from this directory)
*
* What this fixture pins
* ----------------------
* teffects aipw (logit propensity, per-arm linear outcome) and teffects ipw
* with sampling
* weights (see the note on [iw=] below), with vce(cluster g), and with both -- the ATE and the two
* potential-outcome means with their standard errors.  sp.aipw gained
* weights= / cluster= in 1.32; before that neither was accepted.
*
* The effect varies with w and w is related to the covariates, so an
* ignored weight moves the ATE far outside any tolerance.  The propensity
* stays inside (0.01, 0.99) so sp.aipw's clipping never binds.
*
* Data is generated and exported HERE so both sides read the same bytes.
* ---------------------------------------------------------------------------
version 18
clear all
set type double
set seed 20260926
set obs 1500

gen long   id = _n
gen int    g  = mod(_n, 60) + 1
gen double cu = rnormal()
bysort g (id): replace cu = cu[1]
sort id
gen double x1 = rnormal() + 0.5*cu
gen double x2 = rnormal()
gen double x3 = runiform()
gen double w  = exp(0.4*x2 + 0.3*runiform())
gen byte   d  = runiform() < invlogit(-0.2 + 0.6*x1 - 0.4*x2 + 0.5*x3)
gen double y  = 1 + (1.5 + 0.8*x2)*d + 0.7*x1 + 0.3*x2 - 0.5*x3 + cu + rnormal()

format x1 x2 x3 w y %21.16e
export delimited id g x1 x2 x3 w d y using "teffects_design_data.csv", replace datafmt

file open fh using "teffects_design_stata.json", write replace text
file write fh "{" _n
* case k: weight expression, options, JSON key
*
* teffects aipw refuses [pw=] (rc 101) and accepts [iw=]. With
* vce(cluster c) the iweighted sandwich is the sampling-weight one (sum over
* clusters of the weighted scores), so "weights" is pinned as
* [iw=w], vce(cluster id): one observation per cluster, which is the pweight
* robust sandwich because teffects applies no G/(G-1) factor. Under
* vce(robust) Stata instead reads iweights as frequencies (effective N =
* sum of w); that case is kept as "iw_robust" to pin the difference.
local w1 "[iw=w]"
local o1 "vce(cluster id)"
local n1 "weights"
local w2 ""
local o2 "vce(cluster g)"
local n2 "cluster"
local w3 "[iw=w]"
local o3 "vce(cluster g)"
local n3 "weights_cluster"
local w4 ""
local o4 ""
local n4 "plain"
local w5 "[iw=w]"
local o5 "vce(robust)"
local n5 "iw_robust"
forvalues k = 1/5 {
    quietly teffects aipw (y x1 x2 x3) (d x1 x2 x3) `w`k'', ate `o`k''
    local ate = _b[ATE:r1vs0.d]
    local ate_se = _se[ATE:r1vs0.d]
    quietly teffects aipw (y x1 x2 x3) (d x1 x2 x3) `w`k'', pomeans `o`k''
    file write fh `"  "`n`k''": {"ate": "' %23.16e (`ate') `", "ate_se": "' %23.16e (`ate_se') ", "
    file write fh `""po0": "' %23.16e (_b[POmeans:0.d]) `", "po0_se": "' %23.16e (_se[POmeans:0.d]) ", "
    file write fh `""po1": "' %23.16e (_b[POmeans:1.d]) `", "po1_se": "' %23.16e (_se[POmeans:1.d]) ", "
    file write fh `""N": "' %12.0f (e(N)) `"},"' _n
}
* ---- teffects ipw (logit, normalised weights): pweights are allowed -------
* sp.ipw(se_method="sandwich") is pinned against the robust / cluster SE.
local k = 0
foreach stat in ate atet {
    foreach spec in "plain||" "pw|[pw=w]|" "cluster||vce(cluster g)" "pw_cluster|[pw=w]|vce(cluster g)" {
        gettoken tag rest : spec, parse("|")
        gettoken bar rest : rest, parse("|")
        gettoken wexp rest : rest, parse("|")
        if "`wexp'" == "|" {
            local wexp ""
        }
        else {
            gettoken bar rest : rest, parse("|")
        }
        local opt "`rest'"
        quietly teffects ipw (y) (d x1 x2 x3) `wexp', `stat' `opt'
        local b = _b[`=upper("`stat'")':r1vs0.d]
        local s = _se[`=upper("`stat'")':r1vs0.d]
        file write fh `"  "ipw_`stat'_`tag'": {"est": "' %23.16e (`b') `", "se": "' %23.16e (`s') `"},"' _n
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
display "wrote teffects_design_stata.json and teffects_design_data.csv"
