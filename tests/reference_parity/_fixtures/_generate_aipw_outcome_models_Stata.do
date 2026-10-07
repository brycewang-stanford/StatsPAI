* Reference values for tests/reference_parity/test_aipw_outcome_models_stata_parity.py
* Stata 18: teffects aipw with a logit / probit / poisson outcome model and a logit
* treatment model, robust, weighted (iweights; teffects aipw refuses pweights) and clustered. Run from this directory.
* Writes aipw_outcome_models_stata.txt (one "key value" pair per line).
clear all
import delimited using "aipw_outcome_models_data.csv", clear
* teffects aipw refuses [pw=]. With [iw=] and vce(robust) it reads the
* weights as frequencies; with [iw=] and one cluster per row the sandwich is
* the sampling-weight one (see _generate_teffects_design_stata.do).
generate long id = _n
tempname fh
file open `fh' using "aipw_outcome_models_stata.txt", write replace text
foreach om in logit probit poisson {
    local yv = cond("`om'" == "poisson", "yc", "yb")
    foreach cs in plain weights cluster {
        local wt = cond("`cs'" == "weights", "[iw = w]", "")
        local vc = cond("`cs'" == "cluster", "vce(cluster g)", "")
        if "`cs'" == "weights" local vc "vce(cluster id)"
        teffects aipw (`yv' x1 x2 x3, `om') (d x1 x2 x3) `wt', `vc'
        matrix b = e(b)
        matrix V = e(V)
        file write `fh' "`om'_`cs'_ate " %21.15e (b[1,1]) _n
        file write `fh' "`om'_`cs'_ate_se " %21.15e (sqrt(V[1,1])) _n
        file write `fh' "`om'_`cs'_po0 " %21.15e (b[1,2]) _n
        file write `fh' "`om'_`cs'_po0_se " %21.15e (sqrt(V[2,2])) _n
        lincom _b[ATE:r1vs0.d] + _b[POmean:0.d]
        file write `fh' "`om'_`cs'_po1 " %21.15e (r(estimate)) _n
        file write `fh' "`om'_`cs'_po1_se " %21.15e (r(se)) _n
    }
}
file close `fh'
type "aipw_outcome_models_stata.txt"
