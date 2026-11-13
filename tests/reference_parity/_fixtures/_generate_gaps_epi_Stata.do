* Stata reference for the survival / epidemiology items of the 2026-11 gap
* list.
*
*   cd tests/reference_parity/_fixtures
*   stata-mp -b do _generate_gaps_epi_Stata.do
*
* Reads gaps_surv.csv, gaps_obs.csv and gaps_long.csv
* (_generate_gaps_epi_data.py); writes gaps_epi_Stata.csv (long: case, name,
* value). Stata 18 MP, official commands only.
clear all
set type double
tempname fh
file open `fh' using "gaps_epi_Stata.csv", write replace
file write `fh' "case,name,value" _n

program define _put
    args fh case name value
    file write `fh' "`case',`name'," %24.16e (`value') _n
end

program define _coefs
    args fh case
    matrix b = e(b)
    matrix V = e(V)
    local names : colfullnames b
    local j = 0
    foreach nm of local names {
        local ++j
        local clean = subinstr("`nm'", ":", "__", .)
        local clean = subinstr("`clean'", "/", "", .)
        _put `fh' `case' b_`clean' b[1,`j']
        _put `fh' `case' se_`clean' sqrt(V[`j',`j'])
    }
end

* --- 1. Cox, exact partial likelihood ---------------------------------------
import delimited "gaps_surv.csv", clear
stset time, failure(event)
stcox age ecog arm, exactp nohr
_coefs `fh' cox_exact_plain
_put `fh' cox_exact_plain ll e(ll)
_put `fh' cox_exact_plain ll_0 e(ll_0)
_put `fh' cox_exact_plain chi2 e(chi2)
stcox age ecog arm, exactp nohr strata(site)
_coefs `fh' cox_exact_strata
_put `fh' cox_exact_strata ll e(ll)
stset time, failure(event) enter(entry)
stcox age ecog arm, exactp nohr
_coefs `fh' cox_exact_entry
_put `fh' cox_exact_entry ll e(ll)
_put `fh' cox_exact_entry ll_0 e(ll_0)
stcox age ecog arm, exactp nohr strata(site)
_coefs `fh' cox_exact_entry_strata
_put `fh' cox_exact_entry_strata ll e(ll)

* --- 2. AFT, robust and clustered variance ----------------------------------
stset time, failure(event)
local tight "iterate(200) tolerance(1e-13) ltolerance(1e-15) nrtolerance(1e-13)"
foreach d in weibull exponential lognormal loglogistic {
    streg age ecog arm, distribution(`d') time vce(robust) `tight'
    _coefs `fh' aft_`d'_robust
    streg age ecog arm, distribution(`d') time vce(cluster clinic) `tight'
    _coefs `fh' aft_`d'_cluster
    _put `fh' aft_`d'_cluster N_clust e(N_clust)
}

* --- 3. Regression adjustment with a logistic / Poisson outcome model -------
import delimited "gaps_obs.csv", clear
foreach m in logit poisson {
    local yv = cond("`m'" == "logit", "ybin", "ycnt")
    teffects ra (`yv' x1 x2 x3, `m') (a), ate
    _put `fh' ra_`m'_ate b _b[ATE:r1vs0.a]
    _put `fh' ra_`m'_ate se _se[ATE:r1vs0.a]
    _put `fh' ra_`m'_ate pom0 _b[POmean:0.a]
    _put `fh' ra_`m'_ate pom0_se _se[POmean:0.a]
    teffects ra (`yv' x1 x2 x3, `m') (a), atet
    _put `fh' ra_`m'_att b _b[ATET:r1vs0.a]
    _put `fh' ra_`m'_att se _se[ATET:r1vs0.a]
    _put `fh' ra_`m'_att pom0 _b[POmean:0.a]
    teffects ra (`yv' x1 x2 x3, `m') (a), pomeans
    _put `fh' ra_`m'_pom pom0 _b[POmeans:0.a]
    _put `fh' ra_`m'_pom pom1 _b[POmeans:1.a]
    _put `fh' ra_`m'_pom pom0_se _se[POmeans:0.a]
    _put `fh' ra_`m'_pom pom1_se _se[POmeans:1.a]
    nlcom (ratio: _b[POmeans:1.a] / _b[POmeans:0.a]) ///
          (lnratio: ln(_b[POmeans:1.a] / _b[POmeans:0.a]))
    matrix nb = r(b)
    matrix nV = r(V)
    _put `fh' ra_`m'_pom ratio nb[1,1]
    _put `fh' ra_`m'_pom ratio_se sqrt(nV[1,1])
    _put `fh' ra_`m'_pom lnratio nb[1,2]
    _put `fh' ra_`m'_pom lnratio_se sqrt(nV[2,2])
    if "`m'" == "logit" {
        nlcom (lnor: logit(_b[POmeans:1.a]) - logit(_b[POmeans:0.a]))
        matrix nb = r(b)
        matrix nV = r(V)
        _put `fh' ra_`m'_pom lnor nb[1,1]
        _put `fh' ra_`m'_pom lnor_se sqrt(nV[1,1])
    }
}
* linear outcome model, ratio of the potential-outcome means
teffects ra (y x1 x2 x3) (a), pomeans
nlcom (ratio: _b[POmeans:1.a] / _b[POmeans:0.a]) ///
      (lnratio: ln(_b[POmeans:1.a] / _b[POmeans:0.a]))
matrix nb = r(b)
matrix nV = r(V)
_put `fh' ra_linear_pom ratio nb[1,1]
_put `fh' ra_linear_pom ratio_se sqrt(nV[1,1])
_put `fh' ra_linear_pom lnratio_se sqrt(nV[2,2])

* --- 7. Standard errors of the variance components --------------------------
* mixed stores log standard deviations and the arc-hyperbolic tangent of the
* correlation; _diparm turns them into variances, the covariance and the
* correlation with delta-method standard errors, the numbers of its table.
import delimited "gaps_long.csv", clear
program define _vc
    args fh case name
    _put `fh' `case' `name' r(est)
    _put `fh' `case' `name'_se r(se)
end
foreach meth in reml mle {
    mixed y trt visit age || pid:, `meth'
    _diparm lns1_1_1, f(exp(2*@)) d(2*exp(2*@))
    _vc `fh' mixed_ri_`meth' var_cons
    _diparm lnsig_e, f(exp(2*@)) d(2*exp(2*@))
    _vc `fh' mixed_ri_`meth' var_resid
    _put `fh' mixed_ri_`meth' ll e(ll)
}
mixed y trt visit age || pid: visit, mle covariance(unstructured)
_diparm lns1_1_1, f(exp(2*@)) d(2*exp(2*@))
_vc `fh' mixed_rs_mle var_visit
_diparm lns1_1_2, f(exp(2*@)) d(2*exp(2*@))
_vc `fh' mixed_rs_mle var_cons
_diparm atr1_1_1_2 lns1_1_1 lns1_1_2, f(tanh(@1)*exp(@2+@3)) ///
    d((1-tanh(@1)^2)*exp(@2+@3) tanh(@1)*exp(@2+@3) tanh(@1)*exp(@2+@3))
_vc `fh' mixed_rs_mle cov
_diparm atr1_1_1_2, tanh
_vc `fh' mixed_rs_mle corr
_diparm lnsig_e, f(exp(2*@)) d(2*exp(2*@))
_vc `fh' mixed_rs_mle var_resid
_put `fh' mixed_rs_mle ll e(ll)

local tight "iterate(300) tolerance(1e-12) ltolerance(1e-14) nrtolerance(1e-12)"
melogit ybin trt visit age || pid:, intmethod(mcaghermite) intpoints(7) `tight'
_put `fh' melogit var_cons _b[/var(_cons[pid])]
_put `fh' melogit var_cons_se _se[/var(_cons[pid])]
_put `fh' melogit ll e(ll)
mepoisson ycnt trt visit age || pid:, intmethod(mcaghermite) intpoints(7) `tight'
_put `fh' mepoisson var_cons _b[/var(_cons[pid])]
_put `fh' mepoisson var_cons_se _se[/var(_cons[pid])]
_put `fh' mepoisson ll e(ll)
* The adaptive-quadrature fits above stop a little short of the maximum of
* their own objective (the log likelihood StatsPAI reaches is higher); the
* Laplace fits converge and are the tight reference for the standard errors.
melogit ybin trt visit age || pid:, intmethod(laplace) `tight'
_put `fh' melogit_laplace var_cons _b[/var(_cons[pid])]
_put `fh' melogit_laplace var_cons_se _se[/var(_cons[pid])]
_put `fh' melogit_laplace ll e(ll)
* (mepoisson, intmethod(laplace) does not converge on these data: r(430)
* at the default settings, a variance without a standard error otherwise.)

file close `fh'
