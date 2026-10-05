* Reference numbers for tests/reference_parity/test_kohler_kreuter_stata_parity.py
* Run in Stata 18 from this folder:  do kk_syllabus_reference.do
* It writes kk_syllabus_stata.txt, one "name value" pair per line at full
* precision (%21.0g). The test reads that file; do not edit it by hand.
clear all
set more off
capture program drop emit
program emit
    args name value
    file write out "`name' " %21.0g (`value') _n
end
import delimited kk_syllabus.csv, clear case(preserve) asdouble
file open out using kk_syllabus_stata.txt, write text replace

* --- comparing distributions
ranksum y if g < 3, by(g)
emit ranksum_z r(z)
emit ranksum_sum1 r(sum_obs)
emit ranksum_var r(Var_a)
signrank y = z
emit signrank_z r(z)
emit signrank_pos r(sum_pos)
emit signrank_var r(Var_a)
kwallis y, by(g)
emit kwallis_chi2 r(chi2)
emit kwallis_chi2_adj r(chi2_adj)
spearman y x1
emit spearman_rho r(rho)
emit spearman_p r(p)
ktau o g
emit ktau_a r(tau_a)
emit ktau_b r(tau_b)
emit ktau_score r(score)
emit ktau_se r(se_score)
emit ktau_p r(p)
ksmirnov y, by(f)
emit ks_D1 r(D_1)
emit ks_p1 r(p_1)
emit ks_D2 r(D_2)
emit ks_D r(D)
emit ks_p r(p)
median y, by(g)
emit median_chi2 r(chi2)
emit median_p r(p)
median y, by(f)
emit median2_chi2 r(chi2)
emit median2_chi2_cc r(chi2_cc)
robvar y, by(g)
emit robvar_w0 r(w0)
emit robvar_w50 r(w50)
emit robvar_w10 r(w10)
oneway y g, bonferroni
emit oneway_F r(F)
emit oneway_mss r(mss)
emit oneway_rss r(rss)
emit oneway_bartlett r(chi2bart)
tabulate o g, chi2 lrchi2 V gamma taub
emit tab_chi2 r(chi2)
emit tab_lr r(chi2_lr)
emit tab_V r(CramersV)
emit tab_gamma r(gamma)
emit tab_gamma_ase r(ase_gam)
emit tab_taub r(taub)
emit tab_taub_ase r(ase_taub)

* --- summaries with weights
summarize y [aweight = w], detail
emit sum_aw_mean r(mean)
emit sum_aw_sd r(sd)
emit sum_aw_p25 r(p25)
emit sum_aw_p50 r(p50)
emit sum_aw_p90 r(p90)
emit sum_aw_skew r(skewness)
emit sum_aw_kurt r(kurtosis)
centile y, centile(10 50 90)
emit centile_10 r(c_1)
emit centile_50 r(c_2)
emit centile_90 r(c_3)
emit centile_lb_50 r(lb_2)
emit centile_ub_50 r(ub_2)
xtile q4 = y, nquantiles(4)
summarize q4
emit xtile_mean r(mean)
xtile q5w = y [aweight = w], nquantiles(5)
summarize q5w
emit xtile_w_mean r(mean)

* --- means, proportions, totals, ratios
mean y, over(g)
emit mean_over_b1 _b[c.y@1.g]
emit mean_over_se1 _se[c.y@1.g]
emit mean_over_se3 _se[c.y@3.g]
mean y [pweight = w], over(g)
emit mean_pw_b2 _b[c.y@2.g]
emit mean_pw_se2 _se[c.y@2.g]
mean y, vce(cluster psu)
emit mean_cl_se _se[y]
mean y [aweight = w]
emit mean_aw_se _se[y]
mean y [fweight = fw]
emit mean_fw_b _b[y]
emit mean_fw_se _se[y]
proportion o
emit prop_b2 _b[2.o]
emit prop_se2 _se[2.o]
matrix T = r(table)
emit prop_ll2 T[5,2]
emit prop_ul2 T[6,2]
proportion o [pweight = w]
emit prop_pw_se2 _se[2.o]
total y, over(f)
emit total_b0 _b[c.y@0.f]
emit total_se0 _se[c.y@0.f]
ratio (y/x1), over(f)
emit ratio_b1 _b[c._ratio_1@1.f]
emit ratio_se1 _se[c._ratio_1@1.f]

* --- survey design
svyset psu [pweight = w], strata(strata)
svy: mean y
emit svy_mean_b _b[y]
emit svy_mean_se _se[y]
emit svy_df e(df_r)
estat effects
matrix D = r(deff)
emit svy_deff D[1,1]
matrix D = r(deft)
emit svy_deft D[1,1]
svy: mean y, over(g)
emit svy_over_se2 _se[c.y@2.g]
svy: proportion o
emit svy_prop_b3 _b[3.o]
emit svy_prop_se3 _se[3.o]
svy: total y
emit svy_total_b _b[y]
emit svy_total_se _se[y]
svy: ratio (y/x1)
emit svy_ratio_b _b[_ratio_1]
emit svy_ratio_se _se[_ratio_1]
svy, subpop(if f == 1): mean y
emit svy_sub_b _b[y]
emit svy_sub_se _se[y]
emit svy_sub_df e(df_r)
svy: tabulate o f
emit svy_tab_chi2 e(cun_Pear)
emit svy_tab_F e(F_Pear)
emit svy_tab_df1 e(df1_Pear)
emit svy_tab_df2 e(df2_Pear)
svy: tabulate g f
emit svy_tab2_F e(F_Pear)
emit svy_tab2_df1 e(df1_Pear)
svy: regress y x1 x2 i.g
emit svy_reg_b_x1 _b[x1]
emit svy_reg_se_x1 _se[x1]
emit svy_reg_se_g3 _se[3.g]
emit svy_reg_r2 e(r2)
generate double strata1 = strata
replace strata1 = 99 if psu == psu[1]
svyset psu [pweight = w], strata(strata1) singleunit(certainty)
svy: mean y
emit svy_cert_se _se[y]
svyset psu [pweight = w], strata(strata1) singleunit(scaled)
svy: mean y
emit svy_scaled_se _se[y]
svyset psu [pweight = w], strata(strata1) singleunit(centered)
svy: mean y
emit svy_centered_se _se[y]

* --- regression diagnostics
regress y x1 x2 f
predict double rs, rstandard
predict double rt, rstudent
predict double ck, cooksd
predict double df, dfits
predict double wl, welsch
predict double cv, covratio
predict double sp, stdp
predict double sf, stdf
predict double sr, stdr
dfbeta
foreach v in rs rt ck df wl cv sp sf sr _dfbeta_1 _dfbeta_3 {
    summarize `v'
    emit infl_`v'_mean r(mean)
    emit infl_`v'_sd r(sd)
}
estat ovtest, rhs
emit reset_rhs_F r(F)
margins, at(x1 = (30(20)70))
matrix M = r(table)
emit margins_at_b1 M[1,1]
emit margins_at_se3 M[2,3]
regress y c.x1##i.f x2
margins f, at(x1 = (40 60))
matrix M = r(table)
emit margins_f_b4 M[1,4]
emit margins_f_se4 M[2,4]
margins, dydx(x1) at(f = (0 1))
matrix M = r(table)
emit margins_dydx_b2 M[1,2]
emit margins_dydx_se2 M[2,2]

* --- logistic regression diagnostics
logit d x1 f
emit logit_ll0 e(ll_0)
emit logit_chi2 e(chi2)
emit logit_r2p e(r2_p)
estat gof
emit gof_chi2 r(chi2)
emit gof_df r(df)
estat gof, group(8)
emit hl_chi2 r(chi2)
emit hl_df r(df)
lroc, nograph
emit roc_area r(area)
predict double lr, residuals
predict double lh, hat
predict double lrs, rstandard
predict double ldev, deviance
predict double ldx2, dx2
predict double ldd, ddeviance
predict double ldb, dbeta
foreach v in lr lh lrs ldev ldx2 ldd ldb {
    summarize `v'
    emit linfl_`v'_mean r(mean)
    emit linfl_`v'_sd r(sd)
}
predict double xbhat, xb
generate double xbhatsq = xbhat^2
logit d xbhat xbhatsq
emit linktest_hatsq _b[xbhatsq]
emit linktest_se _se[xbhatsq]
logit d x1 f
estimates store full
logit d x1 if e(sample)
lrtest full .
emit lrtest_chi2 r(chi2)

* --- tests after mean, nested blocks, epidemiological tables, outcomes
mean y, over(g)
test _b[c.y@1.g] = _b[c.y@3.g]
emit meantest_F r(F)
emit meantest_p r(p)
lincom _b[c.y@1.g] - _b[c.y@3.g]
emit lincom_est r(estimate)
emit lincom_se r(se)
emit lincom_p r(p)
nestreg: regress y (x1) (x2 f)
matrix W = r(wald)
emit nest_F1 W[1,1]
emit nest_F2 W[2,1]
emit nest_p2 W[2,4]
emit nest_r2 W[2,5]
emit nest_change W[2,6]
anova y g
emit anova_F e(F)
emit anova_r2 e(r2)
cc d f
emit cc_or r(or)
emit cc_lb r(lb_or)
emit cc_ub r(ub_or)
emit cc_chi2 r(chi2)
emit cc_afe r(afe)
emit cc_afp r(afp)
cs d f
emit cs_rd r(rd)
emit cs_lb_rd r(lb_rd)
emit cs_rr r(rr)
emit cs_lb_rr r(lb_rr)
emit cs_ub_rr r(ub_rr)
emit cs_afe r(afe)
emit cs_afp r(afp)
mlogit o x1 f
predict double pm1 pm2 pm3 pm4
summarize pm2
emit mlogit_p2_mean r(mean)
emit mlogit_p2_sd r(sd)
ologit o x1 f
predict double po1 po2 po3 po4
summarize po3
emit ologit_p3_mean r(mean)
emit ologit_p3_sd r(sd)
preserve
statsby m = r(mean) s = r(sd), by(g) clear: summarize y
emit statsby_m2 m[2]
emit statsby_s3 s[3]
restore

* --- data management
generate double a1 = recode(x1, 40, 50, 60, 100)
generate double a2 = irecode(x1, 40, 50, 60)
generate double a3 = autocode(x1, 4, 10, 90)
recode x1 (min/39 = 1) (40/59 = 2) (60/max = 3), generate(a4)
egen a5 = cut(x1), group(3)
foreach v in a1 a2 a3 a4 a5 {
    summarize `v'
    emit dm_`v'_mean r(mean)
    emit dm_`v'_sd r(sd)
}
collapse (mean) m = y (sd) s = y (count) k = y [aweight = w], by(g)
list
emit collapse_m1 m[1]
emit collapse_s2 s[2]
emit collapse_k3 k[3]
file close out
