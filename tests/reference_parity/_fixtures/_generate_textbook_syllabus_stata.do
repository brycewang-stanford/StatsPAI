* ---------------------------------------------------------------------------
* Stata reference for tests/reference_parity/test_textbook_syllabus_stata_parity.py
*
* Reads textbook_cs.csv and textbook_ts.csv (written by
* _generate_textbook_methods_data.py) and writes textbook_syllabus_Stata.csv:
* one "key,value" row per number, %23.16e.
*
* Run (from this directory, Stata 18):
*   stata-mp -b do _generate_textbook_syllabus_stata.do
* Community command: kpss (SSC), installed into the private ado directory
* _ado_textbook_methods/ next to this file (gitignored).
*
* Likelihoods are iterated to tight tolerances so that the reference is the
* optimum and not where Stata's default convergence rule stops.
* ---------------------------------------------------------------------------
clear all
set more off
capture mkdir "_ado_textbook_methods"
adopath ++ "_ado_textbook_methods"
net set ado "_ado_textbook_methods"
capture which kpss
if _rc ssc install kpss

capture program drop emit
program emit
    args key value
    file write fh "`key'," %23.16e (`value') _n
end
capture file close fh
file open fh using "textbook_syllabus_Stata.csv", write replace
file write fh "key,value" _n
local tight "nrtolerance(1e-12) tolerance(1e-13) ltolerance(1e-13) iterate(500)"

* ============================================================ cross-section
import delimited using "textbook_cs.csv", clear asdouble

* --- tests on a variance, z tests
sdtest y == 2
emit sd1.chi2 r(chi2)
emit sd1.df r(df)
emit sd1.p_l r(p_l)
emit sd1.p r(p)
emit sd1.p_u r(p_u)
sdtest y, by(d)
emit sd2.F r(F)
emit sd2.df_1 r(df_1)
emit sd2.df_2 r(df_2)
emit sd2.p_l r(p_l)
emit sd2.p r(p)
emit sd2.p_u r(p_u)
ztest y == 1, sd(2)
emit z1.z r(z)
emit z1.se r(se)
emit z1.p r(p)
emit z1.p_l r(p_l)
ztest y, by(d) sd1(2) sd2(3)
emit z2.z r(z)
emit z2.se r(se)
emit z2.p r(p)
sdtesti 10 . 1.14 2
emit sdi.chi2 r(chi2)
emit sdi.p_l r(p_l)
ztesti 10 88 0.7071 85
emit zi.z r(z)

* --- proportions, normality, confidence intervals, tests of association
prtest d == 0.4
emit pr1.z r(z)
emit pr1.se r(se)
emit pr1.lb r(lb)
emit pr1.ub r(ub)
emit pr1.p r(p)
prtest d, by(b)
emit pr2.z r(z)
emit pr2.se r(se_diff)
emit pr2.se0 r(se_diff0)
emit pr2.lb r(lb_diff)
emit pr2.ub r(ub_diff)
emit pr2.p r(p)
prtesti 50 0.52 0.4
emit pri.z r(z)
sktest y x1
matrix sk = r(table)
emit sk.y.p_skew sk[1,2]
emit sk.y.p_kurt sk[1,3]
emit sk.y.chi2 sk[1,4]
emit sk.y.p_chi2 sk[1,5]
emit sk.x1.p_skew sk[2,2]
emit sk.x1.chi2 sk[2,4]
emit sk.x1.p_chi2 sk[2,5]
sktest y, noadjust
emit skna.y.chi2 r(chi2)
emit skna.y.p_chi2 r(p_chi2)
swilk y
emit sw.W r(W)
emit sw.V r(V)
emit sw.z r(z)
emit sw.p r(p)
ci means y
emit cim.mean r(mean)
emit cim.se r(se)
emit cim.lb r(lb)
emit cim.ub r(ub)
ci variances y
emit civ.var r(Var)
emit civ.lb r(lb)
emit civ.ub r(ub)
ci variances y, sd
emit cis.lb r(lb)
emit cis.ub r(ub)
foreach m in exact wald wilson agresti jeffreys {
    ci proportions d, `m'
    emit cip.`m'.lb r(lb)
    emit cip.`m'.ub r(ub)
}
ci means y, level(90)
emit cim90.lb r(lb)
tabulate b d, chi2 exact lrchi2 V
emit tab2.chi2 r(chi2)
emit tab2.p r(p)
emit tab2.chi2_lr r(chi2_lr)
emit tab2.V r(CramersV)
emit tab2.p_exact r(p_exact)
emit tab2.p1_exact r(p1_exact)
tabulate g d, chi2 lrchi2 V
emit tabk.chi2 r(chi2)
emit tabk.p r(p)
emit tabk.chi2_lr r(chi2_lr)
emit tabk.p_lr r(p_lr)
emit tabk.V r(CramersV)

* --- ivregress gmm: robust weight matrix by default
ivregress gmm y x1 x2 (endog = z1 z2)
emit gmm.b_endog _b[endog]
emit gmm.se_endog _se[endog]
emit gmm.b_x1 _b[x1]
emit gmm.se_x1 _se[x1]
estat overid
emit gmm.J r(HansenJ)
emit gmm.J_p r(p_HansenJ)
ivregress gmm y x1 x2 (endog = z1 z2), wmatrix(unadjusted)
emit gmmu.b_endog _b[endog]
emit gmmu.se_endog _se[endog]

* --- heckman: observed outcome only when b == 1
generate double ysel = y if b == 1
heckman ysel x1 x2, select(x1 x2 z1) twostep
emit heck2.b_x1 _b[x1]
emit heck2.se_x1 _se[x1]
emit heck2.b_cons _b[_cons]
emit heck2.se_cons _se[_cons]
emit heck2.lambda e(lambda)
emit heck2.se_lambda e(selambda)
emit heck2.rho e(rho)
emit heck2.sigma e(sigma)
heckman ysel x1 x2, select(x1 x2 z1) `tight'
emit heckml.b_x1 _b[x1]
emit heckml.se_x1 _se[x1]
emit heckml.b_cons _b[_cons]
emit heckml.athrho _b[/athrho]
emit heckml.se_athrho _se[/athrho]
emit heckml.ll e(ll)
heckman ysel x1 x2, select(x1 x2 z1) vce(robust) `tight'
emit heckrb.se_x1 _se[x1]

* --- truncated regression
truncreg y x1 x2 if y > 0, ll(0) `tight'
emit trunc.b_x1 _b[x1]
emit trunc.se_x1 _se[x1]
emit trunc.sigma _b[/sigma]
emit trunc.ll e(ll)

* --- factor-variable names in test / lincom, testparm, nlcom
regress y x1 x2 i.g
testparm i.g
emit fv.testparm_F r(F)
emit fv.testparm_df r(df)
test 2.g = 3.g
emit fv.eq_F r(F)
regress y c.x1##i.b x2
test 1.b 1.b#c.x1
emit fv.joint_F r(F)
lincom x1 + 1.b#c.x1
emit fv.lincom_b r(estimate)
emit fv.lincom_se r(se)
regress y x1 x2 d
nlcom _b[x1] / _b[x2]
matrix nb = r(b)
matrix nV = r(V)
emit nl.ratio_b nb[1,1]
emit nl.ratio_se sqrt(nV[1,1])
nlcom exp(_b[x1]) - 1
matrix nb = r(b)
matrix nV = r(V)
emit nl.exp_b nb[1,1]
emit nl.exp_se sqrt(nV[1,1])
nlcom _b[x1] / (1 - _b[d]) + _b[_cons]^2
matrix nb = r(b)
matrix nV = r(V)
emit nl.mix_b nb[1,1]
emit nl.mix_se sqrt(nV[1,1])

* ============================================================== time series
import delimited using "textbook_ts.csv", clear asdouble
tsset t

* --- serial correlation and ARCH tests
regress y x1 x2
estat durbinalt
matrix m = r(chi2)
emit dalt1.chi2 m[1,1]
estat durbinalt, lags(3)
matrix m = r(chi2)
emit dalt3.chi2 m[1,1]
estat durbinalt, lags(2) nomiss0
matrix m = r(chi2)
emit dalt2d.chi2 m[1,1]
estat archlm
matrix m = r(arch)
emit arch1.chi2 m[1,1]
estat archlm, lags(3)
matrix m = r(arch)
emit arch3.chi2 m[1,1]

* --- Phillips-Perron and KPSS
pperron c1
emit pp1.Zt r(Zt)
emit pp1.Zrho r(Zrho)
emit pp1.p r(pval)
emit pp1.lags r(lags)
pperron c2, trend lags(3)
emit pp2.Zt r(Zt)
emit pp2.Zrho r(Zrho)
emit pp2.p r(pval)
pperron c3, noconstant lags(2)
emit pp3.Zt r(Zt)
emit pp3.Zrho r(Zrho)
kpss c1, maxlag(3)
emit kpss_ct.l0 r(kpss0)
emit kpss_ct.l3 r(kpss3)
kpss c1, maxlag(3) notrend
emit kpss_c.l0 r(kpss0)
emit kpss_c.l3 r(kpss3)

* --- arima: a constant always, OPG standard errors
arima y, ar(1) `tight'
emit ar1.b_cons _b[y:_cons]
emit ar1.se_cons _se[y:_cons]
emit ar1.b_ar _b[ARMA:L.ar]
emit ar1.se_ar _se[ARMA:L.ar]
emit ar1.sigma _b[sigma:_cons]
emit ar1.ll e(ll)
arima c1, arima(1,1,0) `tight'
emit ari.b_cons _b[c1:_cons]
emit ari.b_ar _b[ARMA:L.ar]
emit ari.ll e(ll)

* --- svar: short-run (exactly and over-identified) and long-run
matrix A = (1,0,0 \ .,1,0 \ .,.,1)
matrix B = (.,0,0 \ 0,.,0 \ 0,0,.)
svar c1 c2 c3, lags(1/2) aeq(A) beq(B) `tight'
emit svar.A21 _b[/A:2_1]
emit svar.se_A21 _se[/A:2_1]
emit svar.A32 _b[/A:3_2]
emit svar.se_A32 _se[/A:3_2]
emit svar.B11 _b[/B:1_1]
emit svar.se_B11 _se[/B:1_1]
emit svar.B33 _b[/B:3_3]
emit svar.ll e(ll)
irf create s, set(_textbook_syllabus_irf, replace) step(4)
preserve
use _textbook_syllabus_irf.irf, clear
summarize sirf if impulse == "c2" & response == "c3" & step == 2, meanonly
local sirf = r(mean)
summarize fevd if impulse == "c2" & response == "c3" & step == 3, meanonly
local fevd = r(mean)
summarize sfevd if impulse == "c1" & response == "c2" & step == 4, meanonly
local sfevd = r(mean)
restore
emit svar.sirf_c2_c3_2 `sirf'
emit var.fevd_c2_c3_3 `fevd'
emit svar.sfevd_c1_c2_4 `sfevd'
matrix A = (1,0,0 \ 0,1,0 \ .,.,1)
svar c1 c2 c3, lags(1/2) aeq(A) beq(B) `tight'
emit svaro.A32 _b[/A:3_2]
emit svaro.se_A32 _se[/A:3_2]
emit svaro.ll e(ll)
emit svaro.chi2 e(chi2_oid)
matrix C = (.,0,0 \ .,.,0 \ .,.,.)
svar c1 c2 c3, lags(1/2) lreq(C) `tight'
emit svarl.C11 _b[/C:1_1]
emit svarl.se_C11 _se[/C:1_1]
emit svarl.C21 _b[/C:2_1]
emit svarl.se_C21 _se[/C:2_1]
emit svarl.C32 _b[/C:3_2]
emit svarl.se_C32 _se[/C:3_2]
emit svarl.C33 _b[/C:3_3]

file close fh
capture erase _textbook_syllabus_irf.irf
