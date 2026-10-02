* ---------------------------------------------------------------------------
* Stata reference for tests/reference_parity/test_textbook_methods_stata_parity.py
*
* Reads textbook_ts.csv, textbook_panel.csv, textbook_cs.csv and
* textbook_rcm.csv (written by _generate_textbook_methods_data.py) and writes
* textbook_methods_Stata.csv: one "key,value" row per number, %23.16e.
*
* Run (from this directory, Stata 18):
*   stata-mp -b do _generate_textbook_methods_stata.do
* Community commands: xtserial (Stata Journal st0039), xtoverid, ivreg2,
* ranktest and rcm (SSC), installed into the private ado directory
* _ado_textbook_methods/ next to this file (gitignored).
* ---------------------------------------------------------------------------
clear all
set more off
capture mkdir "_ado_textbook_methods"
adopath ++ "_ado_textbook_methods"
net set ado "_ado_textbook_methods"
capture which xtserial
if _rc net install st0039, from(http://www.stata-journal.com/software/sj3-2)
foreach pkg in ranktest ivreg2 xtoverid rcm {
    capture which `pkg'
    if _rc ssc install `pkg'
}

capture program drop emit
program emit
    args key value
    file write fh "`key'," %23.16e (`value') _n
end
capture file close fh
file open fh using "textbook_methods_Stata.csv", write replace
file write fh "key,value" _n

* ============================================================ cross-section
import delimited using "textbook_cs.csv", clear asdouble
* --- regress: no constant, robust and clustered model F
regress y x1 x2 d, noconstant
emit nocons.r2 e(r2)
emit nocons.r2_a e(r2_a)
emit nocons.F e(F)
emit nocons.df_m e(df_m)
emit nocons.rmse e(rmse)
regress y x1 x2 d, vce(robust)
emit robust.F e(F)
emit robust.r2 e(r2)
regress y x1 x2 d, vce(cluster g)
emit cluster.F e(F)
emit cluster.df_r e(df_r)
* --- estat after regress
regress y x1 x2 d
estat hettest
emit hettest.normal_fitted.chi2 r(chi2)
emit hettest.normal_fitted.p r(p)
estat hettest, iid
emit hettest.iid_fitted.chi2 r(chi2)
estat hettest, iid rhs
emit hettest.iid_rhs.chi2 r(chi2)
emit hettest.iid_rhs.df r(df)
estat hettest, rhs
emit hettest.normal_rhs.chi2 r(chi2)
estat hettest x1 d, iid
emit hettest.iid_x1d.chi2 r(chi2)
estat hettest, fstat rhs
emit hettest.fstat_rhs.F r(F)
emit hettest.fstat_rhs.p r(p)
estat imtest, white
emit white.chi2 r(chi2_h)
emit white.df r(df_h)
emit imtest.skew r(chi2_s)
emit imtest.df_skew r(df_s)
emit imtest.kurt r(chi2_k)
emit imtest.total r(chi2_t)
emit imtest.df_total r(df_t)
estat ovtest
emit ovtest.F r(F)
emit ovtest.p r(p)
estat ovtest, rhs
emit ovtest_rhs.F r(F)
emit ovtest_rhs.df r(df)
estat vif
emit vif.v1 r(vif_1)
local n1 = r(name_1)
local n2 = r(name_2)
local n3 = r(name_3)
file write fh "vif.order_`n1'_`n2'_`n3',1" _n
emit vif.v2 r(vif_2)
emit vif.v3 r(vif_3)
estat ic
matrix S = r(S)
emit ic.ll S[1,3]
emit ic.df S[1,4]
emit ic.aic S[1,5]
emit ic.bic S[1,6]
* --- IV: endogeneity, overidentification, Hausman
ivregress 2sls y x1 x2 d (endog = z1 z2)
estimates store iv
emit iv.b_endog _b[endog]
emit iv.se_endog _se[endog]
estat endogenous
emit endog.durbin r(durbin)
emit endog.p_durbin r(p_durbin)
emit endog.wu r(wu)
emit endog.p_wu r(p_wu)
estat overid
emit overid.sargan r(sargan)
emit overid.p_sargan r(p_sargan)
ivregress 2sls y x1 x2 d (endog = z1 z2), vce(robust)
estat endogenous
emit endog_robust.regF r(regF)
emit endog_robust.p_regF r(p_regF)
emit endog_robust.df_d r(regFdf_d)
ivregress 2sls y x1 x2 d (endog = z1 z2), vce(cluster g)
estat endogenous
emit endog_cluster.regF r(regF)
emit endog_cluster.p_regF r(p_regF)
emit endog_cluster.df_d r(regFdf_d)
regress y endog x1 x2 d
estimates store ols
hausman iv ols, sigmamore
emit hausman_iv.chi2 r(chi2)
emit hausman_iv.df r(df)
emit hausman_iv.p r(p)
hausman iv ols, constant sigmamore
emit hausman_iv_cons.chi2 r(chi2)
emit hausman_iv_cons.df r(df)
* --- logit: classification table, information criteria
logit b x1 x2 d
estat classification
emit clas.sens r(P_p1)
emit clas.spec r(P_n0)
emit clas.ppv r(P_1p)
emit clas.npv r(P_0n)
emit clas.correct r(P_corr)
estat classification, cutoff(0.3)
emit clas30.correct r(P_corr)
estat ic
matrix S = r(S)
emit logit_ic.ll S[1,3]
emit logit_ic.aic S[1,5]
emit logit_ic.bic S[1,6]

* ============================================================== time series
import delimited using "textbook_ts.csv", clear asdouble
tsset t
* --- serial correlation after regress
regress y x1 x2 d
estat bgodfrey
matrix M = r(chi2)
emit bg.l1_zero M[1,1]
estat bgodfrey, lags(3)
matrix M = r(chi2)
emit bg.l3_zero M[1,1]
estat bgodfrey, lags(2) nomiss0
matrix M = r(chi2)
emit bg.l2_drop M[1,1]
estat dwatson
emit dw.d r(dw)
* --- prais
foreach rt in regress freg tscorr dw theil nagar {
    foreach m in prais corc {
        local opt = cond("`m'" == "corc", "corc", "")
        foreach step in iter two {
            local two = cond("`step'" == "two", "twostep", "")
            quietly prais y x1 x2 d, rhotype(`rt') `opt' `two'
            emit prais.`m'.`rt'.`step'.rho e(rho)
            emit prais.`m'.`rt'.`step'.b_x1 _b[x1]
            emit prais.`m'.`rt'.`step'.se_x1 _se[x1]
            emit prais.`m'.`rt'.`step'.b_cons _b[_cons]
            emit prais.`m'.`rt'.`step'.se_cons _se[_cons]
            emit prais.`m'.`rt'.`step'.r2 e(r2)
            emit prais.`m'.`rt'.`step'.F e(F)
            emit prais.`m'.`rt'.`step'.rmse e(rmse)
            emit prais.`m'.`rt'.`step'.dw e(dw)
            emit prais.`m'.`rt'.`step'.dw0 e(dw_0)
            emit prais.`m'.`rt'.`step'.N e(N)
        }
    }
}
quietly prais y x1 x2 d, vce(robust)
emit prais.robust.se_x1 _se[x1]
emit prais.robust.F e(F)
quietly prais y x1 x2 d, corc vce(hc3)
emit prais.hc3.se_x1 _se[x1]
* --- correlogram
corrgram y, lags(8)
forvalues k = 1/8 {
    emit corrgram.ac`k' r(ac`k')
    emit corrgram.pac`k' r(pac`k')
    emit corrgram.q`k' r(q`k')
}
corrgram y, lags(4) yw
forvalues k = 1/4 {
    emit corrgram_yw.pac`k' r(pac`k')
}
wntestq y, lags(6)
emit wntestq.stat r(stat)
emit wntestq.p r(p)
* --- VAR
varsoc z1 z2, maxlag(4)
matrix S = r(stats)
forvalues i = 1/5 {
    local lag = `i' - 1
    emit varsoc.ll`lag' S[`i',2]
    emit varsoc.fpe`lag' S[`i',6]
    emit varsoc.aic`lag' S[`i',7]
    emit varsoc.hqic`lag' S[`i',8]
    emit varsoc.sbic`lag' S[`i',9]
}
emit varsoc.lr2 S[3,3]
emit varsoc.p2 S[3,5]
var z1 z2, lags(1/2)
emit var.ll e(ll)
emit var.aic e(aic)
emit var.fpe e(fpe)
emit var.detsig e(detsig_ml)
emit var.rmse_1 e(rmse_1)
emit var.r2_2 e(r2_2)
emit var.chi2_1 e(chi2_1)
emit var.b_z1_L1z2 _b[z1:L1.z2]
emit var.se_z1_L1z2 _se[z1:L1.z2]
varwle
matrix W = r(chi2)
emit varwle.z1_l1 W[1,1]
emit varwle.z1_l2 W[2,1]
emit varwle.all_l2 W[2,3]
varlmar, mlag(3)
matrix L = r(lm)
emit varlmar.l1 L[1,2]
emit varlmar.l2 L[2,2]
emit varlmar.l3 L[3,2]
varstable
matrix E = r(Modulus)
emit varstable.m1 E[1,1]
emit varstable.m2 E[1,2]
emit varstable.m4 E[1,4]
vargranger
matrix G = r(gstats)
emit vargranger.z1_z2 G[1,1]
emit vargranger.z2_z1 G[3,1]
fcast compute f_, step(4) nose
emit fcast.z1_h1 f_z1[121]
emit fcast.z2_h1 f_z2[121]
emit fcast.z1_h4 f_z1[124]
drop if t > 120
drop f_*
tsset t
* --- cointegration
vecrank c1 c2 c3, lags(2)
matrix T = e(trace)
matrix L = e(lambda)
emit vecrank.trace0 T[1,1]
emit vecrank.trace1 T[1,2]
emit vecrank.lambda1 L[1,1]
vec c1 c2 c3, lags(2) rank(1)
emit vec.ll e(ll)
emit vec.aic e(aic)
emit vec.sbic e(sbic)
emit vec.detsig e(detsig_ml)
emit vec.alpha_c1 _b[D_c1:L._ce1]
emit vec.se_alpha_c1 _se[D_c1:L._ce1]
emit vec.gamma_c2_c3 _b[D_c2:LD.c3]
emit vec.se_gamma_c2_c3 _se[D_c2:LD.c3]
emit vec.cons_c3 _b[D_c3:_cons]
emit vec.se_cons_c3 _se[D_c3:_cons]
matrix B = e(beta)
matrix VB = e(V_beta)
emit vec.beta_c2 B[1,2]
emit vec.beta_c3 B[1,3]
emit vec.beta_cons B[1,4]
emit vec.se_beta_c2 sqrt(VB[2,2])
emit vec.se_beta_c3 sqrt(VB[3,3])
emit vec.rmse_1 e(rmse_1)
emit vec.r2_1 e(r2_1)
emit vec.chi2_1 e(chi2_1)
veclmar, mlag(2)
matrix L = r(lm)
emit veclmar.l1 L[1,2]
emit veclmar.l2 L[2,2]
vecstable
matrix E = r(Modulus)
emit vecstable.m3 E[1,3]
emit vecstable.m4 E[1,4]
vec c1 c2 c3, lags(3) rank(2) trend(rconstant)
emit vec_rc.ll e(ll)
matrix B = e(beta)
matrix VB = e(V_beta)
emit vec_rc.beta1_c3 B[1,3]
emit vec_rc.beta1_cons B[1,4]
emit vec_rc.beta2_c3 B[1,7]
emit vec_rc.se_beta1_c3 sqrt(VB[3,3])
emit vec_rc.se_beta1_cons sqrt(VB[4,4])
emit vec_rc.alpha_c2_ce2 _b[D_c2:L._ce2]
emit vec_rc.se_alpha_c2_ce2 _se[D_c2:L._ce2]
emit vec_rc.rmse_1 e(rmse_1)
emit vec_rc.chi2_1 e(chi2_1)
emit vec_rc.k e(k_rank)
emit vec_rc.df_m e(df_m)
vec c1 c2 c3, lags(2) rank(1) trend(rtrend)
emit vec_rt.ll e(ll)
emit vec_rt.k e(k_rank)
matrix B = e(beta)
matrix VB = e(V_beta)
emit vec_rt.beta_c2 B[1,2]
emit vec_rt.beta_trend B[1,4]
emit vec_rt.beta_cons B[1,5]
emit vec_rt.se_beta_trend sqrt(VB[4,4])
emit vec_rt.cons_c1 _b[D_c1:_cons]
emit vec_rt.se_cons_c1 _se[D_c1:_cons]
emit vec_rt.se_alpha_c2 _se[D_c2:L._ce1]
vec c1 c2 c3, lags(2) rank(1) trend(trend)
emit vec_ct.ll e(ll)
emit vec_ct.k e(k_rank)
matrix B = e(beta)
emit vec_ct.beta_c2 B[1,2]
emit vec_ct.beta_trend B[1,4]
emit vec_ct.beta_cons B[1,5]
emit vec_ct.trend_c1 _b[D_c1:_trend]
emit vec_ct.se_trend_c1 _se[D_c1:_trend]
emit vec_ct.cons_c1 _b[D_c1:_cons]
emit vec_ct.se_cons_c1 _se[D_c1:_cons]
emit vec_ct.se_alpha_c2 _se[D_c2:L._ce1]
vec c1 c2 c3, lags(2) rank(1) trend(none)
emit vec_n.ll e(ll)
matrix B = e(beta)
matrix VB = e(V_beta)
emit vec_n.beta_c2 B[1,2]
emit vec_n.se_beta_c2 sqrt(VB[2,2])
emit vec_n.se_alpha_c1 _se[D_c1:L._ce1]

* ==================================================================== panel
import delimited using "textbook_panel.csv", clear asdouble
xtset id year
xtsum y x1
emit xtsum.x1_sd r(sd)
emit xtsum.x1_sd_b r(sd_b)
emit xtsum.x1_sd_w r(sd_w)
emit xtsum.x1_min_w r(min_w)
emit xtsum.x1_max_b r(max_b)
xtreg y x1 x2 w, fe
estimates store FE
emit fe.cons _b[_cons]
emit fe.se_cons _se[_cons]
emit fe.sigma_u e(sigma_u)
emit fe.sigma_e e(sigma_e)
emit fe.rho e(rho)
emit fe.r2_w e(r2_w)
emit fe.r2_b e(r2_b)
emit fe.r2_o e(r2_o)
emit fe.corr e(corr)
xtreg y x1 x2 w, fe vce(robust)
emit fe_robust.se_cons _se[_cons]
emit fe_robust.se_x1 _se[x1]
xtreg y x1 x2 w, re
estimates store RE
emit re.b_x1 _b[x1]
emit re.se_x1 _se[x1]
emit re.sigma_u e(sigma_u)
emit re.sigma_e e(sigma_e)
emit re.rho e(rho)
emit re.r2_w e(r2_w)
emit re.r2_b e(r2_b)
emit re.r2_o e(r2_o)
emit re.thta_min e(thta_min)
emit re.thta_max e(thta_max)
xttest0
emit xttest0.lm r(lm)
hausman FE RE
emit hausman_xt.chi2 r(chi2)
emit hausman_xt.df r(df)
hausman FE RE, sigmamore
emit hausman_xt_more.chi2 r(chi2)
hausman FE RE, constant sigmamore
emit hausman_xt_cons.chi2 r(chi2)
emit hausman_xt_cons.df r(df)
xtreg y x1 x2 w, re vce(robust)
emit re_robust.se_x1 _se[x1]
xtoverid
emit xtoverid.j r(j)
emit xtoverid.df r(jdf)
xtreg y x1 x2 w, mle
emit mle.b_x1 _b[x1]
emit mle.se_x1 _se[x1]
emit mle.se_cons _se[_cons]
emit mle.sigma_u e(sigma_u)
emit mle.sigma_e e(sigma_e)
emit mle.ll e(ll)
emit mle.chi2_c e(chi2_c)
emit mle.chi2 e(chi2)
xtreg y x1 x2 w, be
emit be.b_x1 _b[x1]
emit be.se_x1 _se[x1]
xtserial y x1 x2 w
emit xtserial.F r(F)
emit xtserial.p r(p)
emit xtserial.corr r(corr)

* ======================================================= regression control
import delimited using "textbook_rcm.csv", clear asdouble
xtset unit t
rcm y, trunit(1) trperiod(31) nofigure
emit rcm.best.K e(K_preds_sel)
emit rcm.best.aicc e(aicc)
emit rcm.best.aic e(aic)
emit rcm.best.bic e(bic)
emit rcm.best.mbic e(mbic)
emit rcm.best.r2 e(r2)
emit rcm.best.att e(att)
emit rcm.best.mse e(mse)
emit rcm.best.rmse e(rmse)
rcm y, trunit(1) trperiod(31) method(forward) criterion(bic) nofigure
emit rcm.forward_bic.K e(K_preds_sel)
emit rcm.forward_bic.att e(att)
rcm y, trunit(1) trperiod(31) method(backward) criterion(mbic) nofigure
emit rcm.backward_mbic.K e(K_preds_sel)
emit rcm.backward_mbic.att e(att)
rcm y, trunit(1) trperiod(31) nofigure placebo(unit cut(2))
matrix P = e(pval)
matrix M = e(mspe)
emit rcm.placebo.p_two_31 P[1,2]
emit rcm.placebo.p_two_36 P[6,2]
emit rcm.placebo.p_right_36 P[6,3]
emit rcm.placebo.pre_mspe_treated M[1,1]
emit rcm.placebo.ratio_treated M[1,3]
emit rcm.placebo.ratio_row2 M[2,3]
emit rcm.placebo.relative_row2 M[2,4]

file close fh
