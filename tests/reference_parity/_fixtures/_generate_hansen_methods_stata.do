* ---------------------------------------------------------------------------
* Stata reference for tests/reference_parity/test_hansen_methods_stata_parity.py
*
* Reads textbook_cs.csv, textbook_panel.csv and textbook_ts.csv (written by
* _generate_textbook_methods_data.py) and writes hansen_methods_Stata.csv:
* one "key,value" row per number, %23.16e.
*
* Run (from this directory, Stata 18):
*   stata-mp -b do _generate_hansen_methods_stata.do
*
* Iterative estimators are run to tight tolerances so that the reference is
* the optimum and not where Stata's default convergence rule stops.
* ---------------------------------------------------------------------------
clear all
set more off

capture program drop emit
program emit
    args key value
    file write fh "`key'," %23.16e (`value') _n
end
capture file close fh
file open fh using "hansen_methods_Stata.csv", write replace
file write fh "key,value" _n

* ============================================================ cross-section
import delimited using "textbook_cs.csv", clear asdouble

* --- constrained least squares
constraint define 1 x1 + x2 = 1
constraint define 2 d = 0.5
foreach v in "" "vce(robust)" "vce(cluster g)" {
    local tag = cond("`v'" == "", "ols", cond("`v'" == "vce(robust)", "rob", "clu"))
    cnsreg y x1 x2 d, constraints(1 2) `v'
    emit cns.`tag'.b_x1 _b[x1]
    emit cns.`tag'.b_x2 _b[x2]
    emit cns.`tag'.b_cons _b[_cons]
    emit cns.`tag'.se_x1 _se[x1]
    emit cns.`tag'.se_cons _se[_cons]
    emit cns.`tag'.F e(F)
    emit cns.`tag'.rmse e(rmse)
}

* --- nonlinear least squares
generate double xp = exp(x1 / 2)
generate double ynl = 2 + 3 * xp^0.5 + z2 / 4
foreach v in "" "vce(robust)" "vce(cluster g)" {
    local tag = cond("`v'" == "", "ols", cond("`v'" == "vce(robust)", "rob", "clu"))
    nl (ynl = {a} + {b} * xp^{c}), initial(a 1 b 1 c 1) `v' eps(1e-14) iterate(2000)
    emit nl.`tag'.a _b[/a]
    emit nl.`tag'.b _b[/b]
    emit nl.`tag'.c _b[/c]
    emit nl.`tag'.se_a _se[/a]
    emit nl.`tag'.se_b _se[/b]
    emit nl.`tag'.se_c _se[/c]
    emit nl.`tag'.rss e(rss)
    emit nl.`tag'.r2 e(r2)
    emit nl.`tag'.rmse e(rmse)
}

* --- principal components and factor analysis
pca y endog x1 x2 z1 z2
matrix E = e(Ev)
matrix L = e(L)
forvalues j = 1/6 {
    emit pca.ev`j' E[1,`j']
    emit pca.l`j'1 L[`j',1]
    emit pca.l`j'2 L[`j',2]
}
factor y endog x1 x2 z1 z2
matrix E = e(Ev)
matrix L = e(L)
matrix U = e(Psi)
emit pf.k e(f)
forvalues j = 1/6 {
    emit pf.ev`j' E[1,`j']
    emit pf.l`j'1 L[`j',1]
    emit pf.u`j' U[1,`j']
}
factor y endog x1 x2 z1 z2, pcf factors(2)
matrix L = e(L)
matrix U = e(Psi)
forvalues j = 1/6 {
    emit pcf.l`j'1 L[`j',1]
    emit pcf.l`j'2 L[`j',2]
    emit pcf.u`j' U[1,`j']
}
factor y endog x1 x2 z1 z2, ml factors(1) ltolerance(1e-14) tolerance(1e-12) nrtolerance(1e-12)
matrix L = e(L)
matrix U = e(Psi)
emit ml.ll e(ll)
emit ml.chi2_1 e(chi2_1)
emit ml.df_1 e(df_1)
emit ml.chi2_i e(chi2_i)
emit ml.aic e(aic)
emit ml.bic e(bic)
forvalues j = 1/6 {
    emit ml.l`j'1 L[`j',1]
    emit ml.u`j' U[1,`j']
}

* --- overidentification after 2SLS and LIML
ivregress 2sls y x1 x2 (endog = z1 z2 b)
estat overid
emit oid.sargan r(sargan)
emit oid.p_sargan r(p_sargan)
emit oid.basmann r(basmann)
emit oid.p_basmann r(p_basmann)
ivregress 2sls y x1 x2 (endog = z1 z2 b), vce(robust)
estat overid
emit oid.score r(score)
emit oid.p_score r(p_score)
ivregress liml y x1 x2 (endog = z1 z2 b)
estat overid
emit oid.ar r(ar)
emit oid.p_ar r(p_ar)
emit oid.basmann_f r(basmann_f)
emit oid.p_basmann_f r(p_basmann_f)

* --- collinear instruments are dropped
generate double z3 = z1 + z2
generate double z4 = 2 * x1 - 1
ivregress 2sls y x1 x2 (endog = z1 z2 z3 z4)
emit ivc.b_endog _b[endog]
emit ivc.se_endog _se[endog]

* --- jackknife
keep if g <= 3
regress y x1 x2, vce(jackknife)
emit jk.n e(N)
emit jk.se_x1 _se[x1]
emit jk.se_x2 _se[x2]
emit jk.se_cons _se[_cons]
jackknife ratio = (_b[x1] / _b[x2]): regress y x1 x2
emit jk.ratio _b[ratio]
emit jk.ratio_se _se[ratio]
jackknife ratio = (_b[x1] / _b[x2]), mse: regress y x1 x2
emit jk.ratio_se_mse _se[ratio]
jackknife s2 = (e(rss) / e(N)), cluster(g): regress y x1 x2
emit jk.s2 _b[s2]
emit jk.s2_se _se[s2]

* ==================================================================== panel
import delimited using "textbook_panel.csv", clear asdouble
xtset id year
generate double ti = mod(id, 3)
generate double hi = id > 20
xtreg y x1 x2 ti hi, re
emit re.b_x1 _b[x1]
emit re.b_ti _b[ti]
emit re.b_hi _b[hi]
emit re.b_cons _b[_cons]
emit re.se_x1 _se[x1]
emit re.se_ti _se[ti]
emit re.sigma_u e(sigma_u)
emit re.sigma_e e(sigma_e)
xtreg y x1 x2 ti hi, re vce(robust)
emit re.rob_se_x1 _se[x1]
emit re.rob_se_ti _se[ti]
xtreg y x1 x2, fe
emit re.fe_sigma_e e(sigma_e)

* --- Hausman-Taylor
bysort id: egen double wbar = mean(w)
generate double zi = wbar + 0.3 * ti
foreach v in "" "vce(robust)" {
    local tag = cond("`v'" == "", "conv", "rob")
    xthtaylor y x1 x2 w ti zi, endog(x2 zi) `v'
    emit ht.`tag'.b_x1 _b[x1]
    emit ht.`tag'.b_w _b[w]
    emit ht.`tag'.b_x2 _b[x2]
    emit ht.`tag'.b_ti _b[ti]
    emit ht.`tag'.b_zi _b[zi]
    emit ht.`tag'.b_cons _b[_cons]
    emit ht.`tag'.se_x1 _se[x1]
    emit ht.`tag'.se_x2 _se[x2]
    emit ht.`tag'.se_ti _se[ti]
    emit ht.`tag'.se_zi _se[zi]
    emit ht.`tag'.se_cons _se[_cons]
    emit ht.`tag'.sigma_u e(sigma_u)
    emit ht.`tag'.sigma_e e(sigma_e)
}
xthtaylor y x1 x2 w ti zi, endog(x2 zi)
emit ht.conv.chi2 e(chi2)
* period dummies: the estimates move with the year that is left out
forvalues t = 2001/2008 {
    generate double yr`t' = year == `t'
}
xthtaylor y x1 x2 ti zi yr2002 yr2003 yr2004 yr2005 yr2006 yr2007 yr2008, endog(x2 zi)
emit ht.base2001.b_zi _b[zi]
emit ht.base2001.b_x1 _b[x1]
xthtaylor y x1 x2 ti zi yr2001 yr2002 yr2003 yr2004 yr2006 yr2007 yr2008, endog(x2 zi)
emit ht.base2005.b_zi _b[zi]
emit ht.base2005.b_x1 _b[x1]

* ============================================================== time series
import delimited using "textbook_ts.csv", clear asdouble
tsset t
var z1 z2, lags(1/2) exog(x1 d)
emit var.ll e(ll)
emit var.b11 [z1]_b[L1.z1]
emit var.b1x [z1]_b[x1]
emit var.b2d [z2]_b[d]
emit var.se1x [z1]_se[x1]
emit var.aic e(aic)
irf create m1, set(_hansen_irf, replace) step(4)
irf table oirf, impulse(z1) response(z2)
preserve
use _hansen_irf.irf, clear
keep if impulse == "z1" & response == "z2"
sort step
forvalues s = 0/4 {
    local row = `s' + 1
    emit var.oirf`s' oirf[`row']
}
restore
varsoc z1 z2, maxlag(3) exog(x1 d)
matrix S = r(stats)
forvalues p = 0/3 {
    local row = `p' + 1
    emit soc.ll`p' S[`row',2]
    emit soc.aic`p' S[`row',7]
    emit soc.sbic`p' S[`row',9]
}

* --- Dickey-Fuller p-values above the fitted range of the approximation
generate double e1 = z1 in 1
replace e1 = 1.08 * L.e1 + z1 in 2/l
dfuller e1, lags(1) trend
emit df.ct_stat r(Zt)
emit df.ct_p r(p)
dfuller e1, lags(1)
emit df.c_stat r(Zt)
emit df.c_p r(p)

* ============================================================ dynamic panel
* --- system GMM with a dummy for every period but the first, on an
*     unbalanced panel: the periods lost to the lags and one more dummy are
*     omitted from the regressors (dynpanel_abdata.csv is `webuse abdata')
import delimited using "dynpanel_abdata.csv", clear asdouble
xtset id year
xtdpd L(0/2).n yr1977-yr1984, iv(yr1977-yr1984) dgmmiv(n) lgmmiv(n)
emit dpd.one.b_l1 _b[L1.n]
emit dpd.one.b_l2 _b[L2.n]
emit dpd.one.b_cons _b[_cons]
emit dpd.one.b_yr1979 _b[yr1979]
emit dpd.one.b_yr1983 _b[yr1983]
emit dpd.one.se_l1 _se[L1.n]
emit dpd.one.se_cons _se[_cons]
emit dpd.one.sargan e(sargan)
emit dpd.one.zrank e(zrank)
emit dpd.one.k e(rank)
xtdpd L(0/2).n yr1977-yr1984, iv(yr1977-yr1984) dgmmiv(n) lgmmiv(n) vce(robust)
emit dpd.rob.se_l1 _se[L1.n]
emit dpd.rob.se_l2 _se[L2.n]
emit dpd.rob.se_cons _se[_cons]
xtdpd L(0/2).n yr1977-yr1984, iv(yr1977-yr1984) dgmmiv(n) lgmmiv(n) twostep vce(robust)
emit dpd.two.b_l1 _b[L1.n]
emit dpd.two.b_l2 _b[L2.n]
emit dpd.two.b_cons _b[_cons]
emit dpd.two.se_l1 _se[L1.n]
emit dpd.two.se_cons _se[_cons]

* --- a two-step weight that has no inverse: six firms are observed before
*     1978, and the moments dated in their first years outnumber them
import delimited using "dynpanel_abdata.csv", clear asdouble
drop if year < 1978 & id > 6
xtset id year
xtdpd L(0/2).n, dgmmiv(n) lgmmiv(n) twostep
emit dpd.sing.b_l1 _b[L1.n]
emit dpd.sing.b_l2 _b[L2.n]
emit dpd.sing.b_cons _b[_cons]
emit dpd.sing.se_l1 _se[L1.n]
emit dpd.sing.se_cons _se[_cons]
emit dpd.sing.sargan e(sargan)
emit dpd.sing.zrank e(zrank)
xtdpd L(0/2).n, dgmmiv(n) lgmmiv(n) twostep vce(robust)
emit dpd.sing.wc_l1 _se[L1.n]
emit dpd.sing.wc_l2 _se[L2.n]
emit dpd.sing.wc_cons _se[_cons]

* --- an endogenous regressor with its own GMM instruments and an exogenous
*     one among the standard instruments
import delimited using "dynpanel_abdata.csv", clear asdouble
xtset id year
xi: xtdpd L(0/2).n L(0/1).w k i.year, iv(k i.year) dgmmiv(n w, lag(2 4)) lgmmiv(n w) twostep vce(robust)
emit dpd.endo.b_l1 _b[L1.n]
emit dpd.endo.b_l2 _b[L2.n]
emit dpd.endo.b_w _b[w]
emit dpd.endo.b_w_l1 _b[L1.w]
emit dpd.endo.b_k _b[k]
emit dpd.endo.b_cons _b[_cons]
emit dpd.endo.se_l1 _se[L1.n]
emit dpd.endo.se_w _se[w]
emit dpd.endo.se_k _se[k]
emit dpd.endo.se_cons _se[_cons]
emit dpd.endo.zrank e(zrank)

* ============================================================ threshold
* --- a threshold in x2: the intercept and the slope of x1 change above it
import delimited using "textbook_ts.csv", clear asdouble
tsset t
generate double yt = 1 + 0.5 * x1 + (x2 > 0.3) * (1 + x1) + 0.5 * z1
threshold yt, regionvars(x1) threshvar(x2)
matrix T = e(thresholds)
matrix b = e(b)
matrix V = e(V)
emit thr.ols.gamma T[1,2]
emit thr.ols.ssr e(ssr)
emit thr.ols.b1_x1 b[1,1]
emit thr.ols.b1_cons b[1,2]
emit thr.ols.b2_x1 b[1,3]
emit thr.ols.b2_cons b[1,4]
emit thr.ols.v1_x1 V[1,1]
emit thr.ols.v2_x1 V[3,3]
emit thr.ols.v2_cons V[4,4]
threshold yt z2, regionvars(x1) threshvar(x2) trim(15) vce(robust)
matrix T = e(thresholds)
matrix b = e(b)
matrix V = e(V)
emit thr.rob.gamma T[1,2]
emit thr.rob.ssr e(ssr)
emit thr.rob.b_z2 b[1,1]
emit thr.rob.b1_x1 b[1,2]
emit thr.rob.b1_cons b[1,3]
emit thr.rob.b2_x1 b[1,4]
emit thr.rob.b2_cons b[1,5]
emit thr.rob.v_z2 V[1,1]
emit thr.rob.v1_x1 V[2,2]
emit thr.rob.v2_x1 V[4,4]
emit thr.rob.v2_cons V[5,5]
* --- a kink in x2, by nonlinear least squares run to a tight tolerance
generate double yk = 1 + 0.5 * x1 - min(x2 - 0.3, 0) + 2 * max(x2 - 0.3, 0) + 0.3 * z1
nl (yk = {b1}*(x2-{c})*(x2<{c}) + {b2}*(x2-{c})*(x2>{c}) + {b3}*x1 + {b4}), initial(b1 -1 b2 2 b3 .5 b4 1 c .25) vce(robust) eps(1e-14) iterate(2000)
emit kink.b_below _b[/b1]
emit kink.b_above _b[/b2]
emit kink.b_x1 _b[/b3]
emit kink.b_cons _b[/b4]
emit kink.gamma _b[/c]
emit kink.se_below _se[/b1]
emit kink.se_above _se[/b2]
emit kink.se_x1 _se[/b3]
emit kink.se_cons _se[/b4]
emit kink.se_gamma _se[/c]
emit kink.rss e(rss)

file close fh
capture erase _hansen_irf.irf
