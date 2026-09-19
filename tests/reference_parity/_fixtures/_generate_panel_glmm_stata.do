* ---------------------------------------------------------------------------
* Stata reference for tests/reference_parity/test_panel_glmm_parity.py.
*
* Reads panel_glmm_data.csv, panel_count_data.csv, panel_ife_data.csv and
* gmm_nl_data.csv (written by _generate_panel_glmm_data.py) and
* general_gmm_data.csv (written by _generate_general_gmm_R.R),
* absorb_ols_data.csv and mixlogit_data.csv, and
* writes panel_glmm_Stata.json: e(b), sqrt(diag(e(V))) and e(ll) for every
* specification below, numbers as %23.16e (JSON has no leading-dot floats).
*
* Run (from this directory, Stata 18):
*   stata-mp -b do _generate_panel_glmm_stata.do
* SSC packages: ftools, reghdfe, require, regife, mixlogit, installed into
* the private ado directory _ado_panel_glmm/ next to this file (gitignored):
*   sysdir set PLUS "_ado_panel_glmm" ; ssc install <pkg>
* It is put on the adopath before the first SSC command below.
*
* Conventions pinned here
* -----------------------
* * Integration method is always explicit.  intmethod(laplace) is the
*   Laplace approximation with the observed curvature of the log integrand
*   at the conditional mode; intmethod(mcaghermite) intpoints(k) is
*   mode-curvature adaptive Gauss-Hermite quadrature.  The commands'
*   default -- mvaghermite, 7 points, nodes re-centred at the posterior
*   mean -- is recorded once (spec pois_mvagh7_default) for documentation.
* * Convergence is tightened (tolerance 1e-12, ltolerance 1e-14,
*   nrtolerance 1e-12).  At Stata's defaults the ME commands stop ~1e-5
*   (relative) from the optimum on these likelihoods -- gamma _cons moves
*   by 1.6e-5 when the criteria are tightened -- which would swamp any
*   implementation difference.
* * vce(oim) (the default).  Variance components are reported by Stata on
*   the variance scale (var(_cons[gid])) with e(b) holding them in that
*   metric; /lnalpha (NB-2) and /logs (gamma, log of the coefficient of
*   variation, phi = exp(2*logs)) are the dispersion parameters.
* * estat icc after mixed / melogit / meologit: the logit-scale latent
*   residual variance is pi^2/3 for the two logit models.
* * lrtest: chi2 = 2 (ll_full - ll_restricted), df = difference in e(k),
*   p from the chi2(df) upper tail with no boundary (chibar2) adjustment.
* * e(k) is recorded for every fit (number of parameters, for AIC / BIC).
* ---------------------------------------------------------------------------
version 18
clear all
set more off
set type double

global TIGHT "tolerance(1e-12) ltolerance(1e-14) nrtolerance(1e-12)"
sysdir set PLUS "`c(pwd)'/_ado_panel_glmm"
adopath ++ "`c(pwd)'/_ado_panel_glmm"

import delimited using "panel_glmm_data.csv", clear asdouble

tempname fh
file open `fh' using "panel_glmm_Stata.json", write replace text
file write `fh' "{" _n
global PG_FH `fh'
global PG_FIRST 1

capture program drop pg_dump
program define pg_dump
    syntax , spec(string)
    tempname b V
    matrix `b' = e(b)
    matrix `V' = e(V)
    local names : colfullnames `b'
    local k = colsof(`b')
    if $PG_FIRST == 0 {
        file write $PG_FH "," _n
    }
    global PG_FIRST 0
    file write $PG_FH `"  "`spec'": {"b": {"'
    forvalues i = 1/`k' {
        local nm : word `i' of `names'
        local sep = cond(`i' < `k', ", ", "")
        file write $PG_FH `""`nm'": "' %23.16e (`b'[1, `i']) "`sep'"
    }
    file write $PG_FH `"}, "se": {"'
    forvalues i = 1/`k' {
        local nm : word `i' of `names'
        local sep = cond(`i' < `k', ", ", "")
        file write $PG_FH `""`nm'": "' %23.16e (sqrt(`V'[`i', `i'])) "`sep'"
    }
    * Missing e() scalars (regife posts no e(ll) / e(k)) are written as null.
    local ll = cond(missing(e(ll)), "null", strofreal(e(ll), "%23.16e"))
    local kk = cond(missing(e(k)), "null", strofreal(e(k), "%6.0f"))
    local cv "`e(converged)'"
    if "`cv'" == "" local cv = cond(missing(e(converged)), "null", strofreal(e(converged), "%2.0f"))
    local jj = cond(missing(e(J)), "null", strofreal(e(J), "%23.16e"))
    file write $PG_FH `"}, "ll": `ll', "N": "' %12.0f (e(N)) `", "k": `kk', "J": `jj'"'
    file write $PG_FH `", "converged": `cv', "cmdline": ""' `"`e(cmdline)'"' `"""'
    file write $PG_FH "}"
end

capture program drop pg_lr
program define pg_lr
    syntax anything(name=models), spec(string)
    lrtest `models'
    if $PG_FIRST == 0 {
        file write $PG_FH "," _n
    }
    global PG_FIRST 0
    file write $PG_FH `"  "`spec'": {"chi2": "' %23.16e (r(chi2)) `", "df": "' %6.0f (r(df)) `", "p": "' %23.16e (r(p)) "}"
end

capture program drop pg_icc
program define pg_icc
    syntax , spec(string)
    estat icc
    if $PG_FIRST == 0 {
        file write $PG_FH "," _n
    }
    global PG_FIRST 0
    file write $PG_FH `"  "`spec'": {"icc": "' %23.16e (r(icc2)) `", "se": "' %23.16e (r(se2))
    file write $PG_FH `", "lb": "' %23.16e (r(ci2)[1, 1]) `", "ub": "' %23.16e (r(ci2)[1, 2]) "}"
end

* ---- Poisson (offset) ------------------------------------------------------
mepoisson y_pois x1 x2, offset(lexpo) || gid:, intmethod(laplace) $TIGHT
pg_dump, spec(pois_laplace)
mepoisson y_pois x1 x2, offset(lexpo) || gid:, intmethod(mcaghermite) intpoints(7) $TIGHT
pg_dump, spec(pois_mcagh7)
mepoisson y_pois x1 x2, offset(lexpo) || gid:, $TIGHT
pg_dump, spec(pois_mvagh7_default)

* ---- binomial counts ---------------------------------------------------------
melogit y_succ x1 x2 || gid:, binomial(n_trials) intmethod(laplace) $TIGHT
pg_dump, spec(binom_laplace)

* ---- negative binomial (NB-2, mean dispersion) -----------------------------
menbreg y_nb x1 x2 || gid:, intmethod(laplace) $TIGHT
pg_dump, spec(nb_laplace)
menbreg y_nb x1 x2 || gid:, intmethod(mcaghermite) intpoints(7) $TIGHT
pg_dump, spec(nb_mcagh7)

* ---- gamma, log link --------------------------------------------------------
* At the tight criteria the gamma fits end "backed up" (r(430)): the
* log likelihood cannot be improved in the 14th digit.  The estimates are
* kept and e(converged) = 0 is recorded alongside them; iterate(60) only
* stops Stata from spending another 240 backed-up iterations there.
capture noisily meglm y_gam x1 x2 || gid:, family(gamma) link(log) intmethod(laplace) $TIGHT iterate(60)
pg_dump, spec(gamma_laplace)
capture noisily meglm y_gam x1 x2 || gid:, family(gamma) link(log) intmethod(mcaghermite) intpoints(7) $TIGHT iterate(60)
pg_dump, spec(gamma_mcagh7)

* ---- ordered logit ------------------------------------------------------------
meologit y_ord x1 x2 || gid:, intmethod(laplace) $TIGHT
pg_dump, spec(ologit_laplace)
pg_icc, spec(ologit_laplace_icc)
meologit y_ord x1 x2 || gid:, intmethod(mcaghermite) intpoints(7) $TIGHT
pg_dump, spec(ologit_mcagh7)

* ---- Gaussian meglm (exact under any integration rule) -----------------------
meglm y_gau x1 x2 || gid:, intmethod(laplace) $TIGHT
pg_dump, spec(gauss_meglm)

* ---- binary logit + ICC --------------------------------------------------------
melogit y_bin x1 x2 || gid:, intmethod(laplace) $TIGHT
pg_dump, spec(bin_laplace)
pg_icc, spec(bin_laplace_icc)

* ---- mixed + ICC (ML and REML) -------------------------------------------------
mixed y_gau x1 x2 || gid:, ml $TIGHT
pg_dump, spec(mixed_ml)
pg_icc, spec(mixed_ml_icc)
estimates store m_full
mixed y_gau x1 x2 || gid:, reml $TIGHT
pg_dump, spec(mixed_reml)
pg_icc, spec(mixed_reml_icc)

* ---- likelihood-ratio tests (Stata lrtest: naive chi2(df), df = e(k) gap) --
* Fixed-effect restriction (interior null).
mixed y_gau x1 || gid:, ml $TIGHT
pg_dump, spec(mixed_ml_x1only)
estimates store m_x1
pg_lr m_full m_x1, spec(lr_fixed)
* Random slope on x1 (outcome y_rs, which has one), unstructured
* covariance: adds a variance and a covariance, df 2.  lrtest prints a
* boundary note but reports the naive chi2(2) tail.
mixed y_rs x1 x2 || gid:, ml $TIGHT
pg_dump, spec(rs_intercept_only)
estimates store r_int
mixed y_rs x1 x2 || gid: x1, covariance(unstructured) ml $TIGHT
pg_dump, spec(rs_slope_un)
estimates store r_slope_un
pg_lr r_slope_un r_int, spec(lr_slope_un)
* Random slope, independent covariance (adds one variance: df 1).
mixed y_rs x1 x2 || gid: x1, covariance(independent) ml $TIGHT
pg_dump, spec(rs_slope_ind)
estimates store r_slope_ind
pg_lr r_slope_ind r_int, spec(lr_slope_ind)
* Fixed-effect restriction in a logit GLMM.
melogit y_bin x1 x2 || gid:, intmethod(laplace) $TIGHT
estimates store b_full
melogit y_bin x1 || gid:, intmethod(laplace) $TIGHT
pg_dump, spec(bin_laplace_x1only)
estimates store b_x1
pg_lr b_full b_x1, spec(lr_melogit_fixed)

* ---- xtnbreg (Hausman-Hall-Griliches), panel_count_data.csv -----------------
* fe: conditional likelihood; drops panels with all-zero outcomes (2 here).
* re: 1/(1+delta_i) ~ Beta(r, s), parameters /ln_r and /ln_s.
import delimited using "panel_count_data.csv", clear asdouble
xtset pid year
xtnbreg y z1 z2, fe $TIGHT
pg_dump, spec(xtnbreg_fe)
xtnbreg y z1 z2 i.year, fe $TIGHT
pg_dump, spec(xtnbreg_fe_year)
xtnbreg y z1 z2, re $TIGHT
pg_dump, spec(xtnbreg_re)

* ---- interactive fixed effects: regife (SSC, Gomez), panel_ife_data.csv ------
* regife needs reghdfe, ftools and require; they live in a private ado
* directory next to this file (never the user's PLUS):
*   _ado_panel_glmm/  <- ssc install ftools / reghdfe / require / regife
* noconstant: the model has no intercept (regife adds one by default).
* regife's SEs come from reghdfe with the factors and loadings absorbed as
* i.period#c.loading and i.unit#c.factor; its df counts r(N+T) absorbed
* parameters.  Without the `require` package reghdfe errors with r(9) and
* regife silently posts the *pooled-OLS starting values* as its estimate
* (the IFE estimate is then only in e(bend)).
import delimited using "panel_ife_data.csv", clear asdouble
regife y xa xb, ife(unit period, 2) noconstant tolerance(1e-12)
pg_dump, spec(ife_regife)
regife y xa xb, ife(unit period, 2) noconstant tolerance(1e-12) vce(cluster unit)
pg_dump, spec(ife_regife_cluster)

* ---- gmm: linear IV (general_gmm_data.csv) and exponential-mean IV -----------
* Stata defaults: winitial(unadjusted) = (Z'Z/N)^-1, wmatrix(robust),
* vce(robust), uncentred S.  The robust VCE of the two-step / iterated
* estimators uses the weight the last round was minimised with (built from
* the penultimate estimate) and S at the final estimate.  igmm criteria
* are tightened from their 1e-6 defaults.
global GTIGHT "conv_ptol(1e-13) conv_vtol(1e-15)"
import delimited using "general_gmm_data.csv", clear asdouble
gmm (y - {xb: x1 _cons}), instruments(z1 z2 z3) twostep $GTIGHT
pg_dump, spec(gmm_lin_twostep)
gmm (y - {xb: x1 _cons}), instruments(z1 z2 z3) igmm igmmeps(1e-13) igmmweps(1e-13) $GTIGHT
pg_dump, spec(gmm_lin_igmm)
gmm (y - {xb: x1 _cons}), instruments(z1 z2 z3) onestep $GTIGHT
pg_dump, spec(gmm_lin_onestep)
import delimited using "gmm_nl_data.csv", clear asdouble
gmm (y*exp(-{xb: x1 _cons}) - 1), instruments(z1 z2 z3) twostep $GTIGHT
pg_dump, spec(gmm_nl_twostep)
gmm (y*exp(-{xb: x1 _cons}) - 1), instruments(z1 z2 z3) igmm igmmeps(1e-13) igmmweps(1e-13) $GTIGHT
pg_dump, spec(gmm_nl_igmm)
gmm (y*exp(-{xb: x1 _cons}) - 1), instruments(z1 z2 z3) onestep $GTIGHT
pg_dump, spec(gmm_nl_onestep)

* ---- reghdfe (absorb_ols_data.csv): singletons, aweights, 2-way clusters -----
* reghdfe drops the 6 singleton firms; [aw=w] weights; the multi-way cluster
* VCE scales every inclusion-exclusion term by G_min/(G_min - 1).  The
* unweighted two-way-cluster VCE on this sample is not PSD and reghdfe
* returns missing SEs after its eigenvalue fix, so only the weighted
* two-way fit is recorded.
import delimited using "absorb_ols_data.csv", clear asdouble
reghdfe y x1 x2, absorb(firm year) tol(1e-14)
pg_dump, spec(reghdfe_iid)
reghdfe y x1 x2, absorb(firm year) tol(1e-14) vce(cluster firm)
pg_dump, spec(reghdfe_cl_firm)
reghdfe y x1 x2 [aw=w], absorb(firm year) tol(1e-14)
pg_dump, spec(reghdfe_aw_iid)
reghdfe y x1 x2 [aw=w], absorb(firm year) tol(1e-14) vce(cluster firm)
pg_dump, spec(reghdfe_aw_cl_firm)
reghdfe y x1 x2 [aw=w], absorb(firm year) tol(1e-14) vce(cluster firm year)
pg_dump, spec(reghdfe_aw_cl_firm_year)

* ---- mixlogit (SSC, Hole), mixlogit_data.csv ------------------------------------
* Default draws: nrep(50) burn(15) -- individual n uses
* invnormal(halton(50, krnd, 1 + 15 + 50*(n-1))), which sp.mixlogit
* reproduces with n_draws=50, halton_burn=15, halton_shift=False.  Default
* VCE is oim; robust is the individual-level sandwich times N/(N-1).
* ln(1): the last rand() variable (comfort) is lognormal, b = exp(m + s z).
* corr: Cholesky parameterisation /l11 /l21 /l22.
import delimited using "mixlogit_data.csv", clear asdouble
global MTIGHT "nrep(50) burn(15) tolerance(1e-10) ltolerance(1e-12)"
mixlogit chosen price, group(cs) id(pid) rand(quality comfort) $MTIGHT
pg_dump, spec(mixlogit_oim)
mixlogit chosen price, group(cs) id(pid) rand(quality comfort) $MTIGHT robust
pg_dump, spec(mixlogit_robust)
mixlogit chosen price, group(cs) id(pid) rand(quality comfort) ln(1) $MTIGHT
pg_dump, spec(mixlogit_lognormal)
mixlogit chosen price, group(cs) id(pid) rand(quality comfort) corr $MTIGHT
pg_dump, spec(mixlogit_corr)

file write `fh' _n "}" _n
file close `fh'
