# ---------------------------------------------------------------------------
# Answer key for tests/external_parity/test_ding_linear_model.py
#
# Peng Ding, "Linear Model and Extensions" (2024), replication files on the
# Harvard Dataverse. This script reruns the deterministic, real-data part of
# every chapter's R program (simulations are left out; where a chapter only
# simulates, one draw is stored so both sides see the same numbers) and
# writes what R reports. Neither the programs nor the data are redistributed.
#
# Requires: R with car, sandwich, lmtest, MASS, mgcv, glmnet, leaps, KernSmooth,
#           margins, nnet, mlogit, pscl, gee, quantreg, survival, timereg,
#           mlbench, Matching, mediation, foreign, jsonlite.
# Run:      STATSPAI_DING_LM_DIR=/path/to/unzipped/dataverse_files \
#               Rscript tests/external_parity/ding_linear_model_reference.R
#
# It writes data/ding_linear_model_R.json next to itself and, into
# $STATSPAI_DING_LM_DIR/_statspai/, the tables the Python side cannot build
# from the dataverse files alone (data shipped inside R packages, and the
# stored simulation draws).
# ---------------------------------------------------------------------------
suppressMessages({
  library(car); library(sandwich); library(lmtest); library(MASS)
  library(foreign); library(jsonlite); library(mlbench); library(Matching)
  library(glmnet); library(leaps); library(KernSmooth); library(margins)
  library(nnet); library(mlogit); library(pscl); library(gee)
  library(quantreg); library(survival)
})
D = paste0(normalizePath(Sys.getenv("STATSPAI_DING_LM_DIR")), "/")
Sdir = paste0(D, "_statspai"); dir.create(Sdir, showWarnings = FALSE)
save_csv = function(d, name) write.csv(d, paste0(Sdir, "/", name), row.names = FALSE)
tight = glm.control(epsilon = 1e-14, maxit = 100)
## glm.nb's theta iteration can fail under a very tight tolerance when theta
## runs off to infinity; such fits are skipped (NULL) rather than forced
nb_fit = function(...) tryCatch(suppressWarnings(glm.nb(..., control = glm.control(epsilon = 1e-12, maxit = 200))),
                                error = function(e) NULL)
ct = function(m) { s = summary(m)$coef; list(names = rownames(s), est = unname(s[, 1]), se = unname(s[, 2])) }
out = list()

## ---- chapter 1: OLS and logit ------------------------------------------
message("chapter 1: OLS and logit")
prostate = read.table(paste0(D, "prostate.txt"), header = TRUE)
out$ch1_prostate = ct(lm(lpsa ~ lcavol + lweight + age + lbph + svi + lcp + gleason + pgg45, data = prostate))
mroz = read.table(paste0(D, "mroz.txt"), header = TRUE)
mroz_fit = glm(inlf ~ kidslt6 + kidsge6 + age + educ + hushrs + husage + huseduc + huswage + unem + city,
               family = binomial(link = logit), data = mroz, control = tight)
out$ch1_mroz = c(ct(mroz_fit), list(deviance = mroz_fit$deviance, null_deviance = mroz_fit$null.deviance, aic = mroz_fit$aic))

## ---- chapter 5: intervals, joint F test ---------------------------------
message("chapter 5: intervals, joint F test")
galton = read.table(paste0(D, "GaltonFamilies.txt"), header = TRUE)
galton_fit = lm(childHeight ~ midparentHeight, data = galton)
new_data = data.frame(midparentHeight = seq(60, 80, by = 0.5))
out$ch5_galton = c(ct(galton_fit), list(
  ci = unname(predict(galton_fit, new_data, interval = "confidence")),
  pi = unname(predict(galton_fit, new_data, interval = "prediction"))))
lalonde = read.table(paste0(D, "lalonde.txt"), header = TRUE)
lalonde_fit = lm(re78 ~ ., data = lalonde)
lh = linearHypothesis(lalonde_fit, c("age=0", "educ=0", "black=0", "hisp=0", "married=0", "nodegr=0",
                                     "re74=0", "re75=0", "u74=0", "u75=0"))
lh_hc = linearHypothesis(lalonde_fit, c("age=0", "educ=0", "black=0", "hisp=0", "married=0", "nodegr=0",
                                        "re74=0", "re75=0", "u74=0", "u75=0"), white.adjust = "hc3")
nt = lalonde; nt$treat = 1; nc = lalonde; nc$treat = 0
out$ch5_lalonde = c(ct(lalonde_fit), list(
  F = lh$F[2], F_p = lh$`Pr(>F)`[2], F_df = c(lh$Df[2], lh$Res.Df[2]),
  F_hc3 = lh_hc$F[2], F_hc3_p = lh_hc$`Pr(>F)`[2],
  mean1 = mean(predict(lalonde_fit, nt)), mean0 = mean(predict(lalonde_fit, nc)),
  r2 = summary(lalonde_fit)$r.squared, fstat = unname(summary(lalonde_fit)$fstatistic)))

## ---- chapter 6: HC0 to HC4 ----------------------------------------------
message("chapter 6: HC0 to HC4")
hc_table = function(fit) {
  se = sapply(c("hc0", "hc1", "hc2", "hc3", "hc4"), function(t) sqrt(diag(hccm(fit, type = t))))
  list(names = names(coef(fit)), est = unname(coef(fit)), ols = unname(summary(fit)$coef[, 2]),
       hc0 = unname(se[, 1]), hc1 = unname(se[, 2]), hc2 = unname(se[, 3]), hc3 = unname(se[, 4]), hc4 = unname(se[, 5]))
}
out$ch6_lalonde = hc_table(lalonde_fit)
isq = read.dta(paste0(D, "isq.dta"))
isq = na.omit(isq[, c("multish", "lnpop", "lnpopsq", "lngdp", "lncolony", "lndist", "freedom", "militexp", "arms",
                      "year83", "year86", "year89", "year92")])
save_csv(isq, "isq_complete.csv")
out$ch6_isq = hc_table(lm(multish ~ ., data = isq))
out$ch6_isq_log = hc_table(lm(log(multish + 1) ~ ., data = isq))
data(BostonHousing); BH = BostonHousing; BH$chas = as.numeric(BH$chas) - 1
save_csv(BH, "BostonHousing.csv")
boston_fit = lm(medv ~ ., data = BH)
out$ch6_boston = hc_table(boston_fit)
out$ch6_boston_log = hc_table(lm(log(medv) ~ ., data = BH))

## ---- chapter 8: nested-model F test -------------------------------------
lalonde_treat = lm(re78 ~ treat, data = lalonde); lalonde1 = lm(re78 ~ 1, data = lalonde)
a2 = anova(lalonde_treat, lalonde_fit); a3 = anova(lalonde1, lalonde_treat, lalonde_fit)
out$ch8_anova = list(F2 = a2$F[2], p2 = a2$`Pr(>F)`[2], rss2 = a2$RSS, df2 = a2$Df[2],
                     F3 = a3$F[2:3], p3 = a3$`Pr(>F)`[2:3], rss3 = a3$RSS)

## ---- chapter 11: leverage and influence ---------------------------------
message("chapter 11: leverage and influence")
out$ch11_lalonde = list(hat = unname(hatvalues(lalonde_fit)), rstandard = unname(rstandard(lalonde_fit)),
                        rstudent = unname(rstudent(lalonde_fit)), cook = unname(cooks.distance(lalonde_fit)),
                        dffits = unname(dffits(lalonde_fit)))

## ---- chapter 12: leave-one-out prediction intervals ---------------------
BHo = BH[order(BH$medv), ]
fo = lm(medv ~ ., data = BHo); n = nrow(BHo); p = ncol(BHo) - 1
h = hatvalues(fo); e = resid(fo); s = summary(fo)$sigma
loo_pred = BHo$medv - e / (1 - h)
loo_sigma = sqrt(s^2 * (n - p - 1) - e^2 / (1 - h)) / sqrt(n - p - 2)
cvt = qt(0.975, df = n - p - 2)
## the book's code uses df n-p-1 with p = 13 covariates plus an intercept;
## both versions are stored, the second is the textbook's exact arithmetic
loo_sigma_book = sqrt(s^2 * (n - p) - e^2 / (1 - h)) / sqrt(n - p - 1)
cvt_book = qt(0.975, df = n - p - 1)
out$ch12_boston = list(loo_pred = unname(loo_pred),
  lower = unname(loo_pred - cvt * loo_sigma / sqrt(1 - h)), upper = unname(loo_pred + cvt * loo_sigma / sqrt(1 - h)),
  lower_book = unname(loo_pred - cvt_book * loo_sigma_book / sqrt(1 - h)),
  upper_book = unname(loo_pred + cvt_book * loo_sigma_book / sqrt(1 - h)),
  press = sum((e / (1 - h))^2))

## ---- chapter 13: best subset --------------------------------------------
message("chapter 13: best subset")
penn = read.table(paste0(D, "pennbonus.txt"))
subset_key = function(formula, data) {
  n = nrow(data); p = ncol(data) - 1
  sm = summary(regsubsets(formula, nvmax = p, data = data, method = "exhaustive"))
  aic = n * log(sm$rss / n) + 2 * (2:(p + 1)); bic = n * log(sm$rss / n) + log(n) * (2:(p + 1))
  list(rss = sm$rss, aic = aic, bic = bic, best_aic = which.min(aic), best_bic = which.min(bic),
       which_aic = colnames(sm$which)[sm$which[which.min(aic), ]][-1],
       which_bic = colnames(sm$which)[sm$which[which.min(bic), ]][-1],
       adjr2 = sm$adjr2, cp = sm$cp)
}
out$ch13_penn = subset_key(duration ~ ., penn)
out$ch13_boston = subset_key(medv ~ ., BH)

## ---- chapters 14 and 15: ridge and lasso --------------------------------
message("chapters 14 and 15: ridge and lasso")
lam = c(0, 0.5, 1, 2, 5, 10, 50)
rf = lm.ridge(medv ~ ., data = BH, lambda = lam)
rf_fine = lm.ridge(medv ~ ., data = BH, lambda = seq(0, 5, 0.01))
out$ch14_ridge = list(lambda = lam, coef = unname(coef(rf)), names = colnames(coef(rf)), gcv = unname(rf$GCV),
                      kHKB = rf$kHKB, kLW = rf$kLW,
                      gcv_min_lambda = seq(0, 5, 0.01)[which.min(rf_fine$GCV)])
xm = as.matrix(BH[, names(BH) != "medv"]); yv = BH$medv
gl = glmnet(xm, yv, lambda = c(1, 0.5, 0.1), thresh = 1e-12, maxit = 1e7)
out$ch15_lasso = list(lambda = c(1, 0.5, 0.1), coef = unname(as.matrix(coef(gl))), names = rownames(coef(gl)))

## ---- chapter 16: Box-Cox, polynomial terms ------------------------------
pennlm = lm(duration ~ ., data = penn)
bc = boxcox(pennlm, lambda = seq(0.2, 0.4, 0.05), plotit = FALSE, interp = FALSE)
bc_fine = boxcox(pennlm, lambda = seq(-1, 2, 0.001), plotit = FALSE, interp = FALSE)
lhat = bc_fine$x[which.max(bc_fine$y)]
inside = bc_fine$x[bc_fine$y > max(bc_fine$y) - qchisq(0.95, 1) / 2]
out$ch16_boxcox_penn = list(lambda = bc$x, loglik = bc$y, lambda_hat = lhat, ci = range(inside))
data(jobs, package = "mediation")
save_csv(jobs, "jobs_mediation.csv")
jobslm = lm(job_seek ~ treat + econ_hard + depress1 + sex + age + occp + marital + nonwhite + educ + income, data = jobs)
bcj = boxcox(jobslm, lambda = seq(1.5, 3, 0.1), plotit = FALSE, interp = FALSE)
out$ch16_boxcox_jobs = list(lambda = bcj$x, loglik = bcj$y)
census00 = read.dta(paste0(D, "census00.dta"))
out$ch16_census1 = ct(lm(logwk ~ educ + exper + black, data = census00))
out$ch16_census2 = ct(lm(logwk ~ educ + exper + I(exper^2) + black, data = census00))

## additive model for log wages: P-splines so that both sides use one basis
## (the book's call uses mgcv's default thin plate basis, stored for contrast)
suppressMessages(library(mgcv))
gp = gam(logwk ~ s(educ, bs = "ps", k = 10) + s(exper, bs = "ps", k = 10) + black, data = census00,
         method = "REML", control = gam.control(epsilon = 1e-12))
gt = gam(logwk ~ s(educ) + s(exper) + black, data = census00)
out$ch16_gam = list(sp = unname(gp$sp), edf = unname(summary(gp)$s.table[, 1]), par = unname(summary(gp)$p.coeff),
                    par_se = unname(summary(gp)$se[1:2]), scale = gp$sig2, fitted_head = unname(fitted(gp)[1:20]),
                    tp_edf = unname(summary(gt)$s.table[, 1]), tp_black = unname(coef(gt)["black"]),
                    tp_fitted_head = unname(fitted(gt)[1:20]))

## ---- chapter 17: interactions -------------------------------------------
message("chapter 17: interactions")
hsb = read.table(paste0(D, "hsbdemo.txt"))
out$ch17_inter = ct(lm(read ~ math * socst, data = hsb))
out$ch17_inter_c = ct(lm(read ~ math.c * socst.c, data = hsb))

## ---- chapter 19: WLS, FGLS, local linear --------------------------------
message("chapter 19: WLS, FGLS, local linear")
ols = lm(medv ~ ., data = BH); dr = BH; dr$medv = log(resid(ols)^2)
w_fgls = exp(-fitted(lm(medv ~ ., data = dr))); BHw = BH; BHw$w = w_fgls
out$ch19_fgls = ct(lm(medv ~ . - w, weights = w, data = BHw))
out$ch19_fgls_w = unname(w_fgls)
lav = read.csv(paste0(D, "lavoteall.csv"))
out$ch19_lav_ols = ct(lm(t ~ x, data = lav)); out$ch19_lav_wls = ct(lm(t ~ x, weights = n, data = lav))
out$ch19_lav_wls0 = ct(lm(t ~ 0 + x + I(1 - x), weights = n, data = lav))
out$ch19_census_wls = ct(lm(logwk ~ age + educ + exper + exper2 + black, weights = perwt, data = census00))
set.seed(19); nn = 500; xs = seq(0, 1, length.out = nn); ys = sin(8 * xs) + rnorm(nn, 0, 0.5)
save_csv(data.frame(x = xs, y = ys), "sim_locpoly.csv")
hh = dpill(xs, ys); lp = locpoly(xs, ys, bandwidth = hh)
out$ch19_locpoly = list(h = hh, x = lp$x, y = lp$y)

## ---- chapter 20: binary outcomes ----------------------------------------
message("chapter 20: binary outcomes")
flu = read.table(paste0(D, "fludata.txt"), header = TRUE); flu = within(flu, rm(receive))
flu_logit = glm(outcome ~ ., family = binomial(link = logit), data = flu, control = tight)
emp = apply(flu, 2, mean); da = data.frame(rbind(emp, emp)); da[1, 1] = 1; da[2, 1] = 0
pr = predict(flu_logit, newdata = da, type = "response", se.fit = TRUE)
ape = summary(margins(flu_logit))
out$ch20_flu = c(ct(flu_logit), list(
  lr = flu_logit$null.deviance - flu_logit$deviance, lr_df = flu_logit$df.null - flu_logit$df.residual,
  lr_p = pchisq(flu_logit$null.deviance - flu_logit$deviance, df = flu_logit$df.null - flu_logit$df.residual, lower.tail = FALSE),
  pred = unname(pr$fit), pred_se = unname(pr$se.fit),
  ame_names = ape$factor, ame = ape$AME, ame_se = ape$SE))
set.seed(20); n = 100; x = rnorm(n, 0, 3); y = rbinom(n, 1, 1 / (1 + exp(-1 + x)))
save_csv(data.frame(x = x, y = y), "sim_links.csv")
for (l in c("probit", "logit", "cloglog", "cauchit")) {
  f = glm(y ~ x, family = binomial(link = l), control = tight)
  out[[paste0("ch20_link_", l)]] = c(ct(f), list(deviance = f$deviance, correct = mean((fitted(f) > 0.5) == y)))
}
out$ch20_link_lpm = ct(lm(y ~ x))
sam = read.csv(paste0(D, "samarani.csv"))
out$ch20_samarani = ct(glm(case_comb ~ ds1 + ds2 + ds3 + ds4_a + ds4_b + ds5 + ds1_3 + center,
                           family = binomial(link = logit), data = sam, control = tight))

## ---- chapter 21: categorical outcomes -----------------------------------
message("chapter 21: categorical outcomes")
kar = read.table(paste0(D, "karolinska.txt"), header = TRUE)
kar = kar[, c("highdiag", "hightreat", "age", "rural", "male", "survival")]
kar$loneyear = as.numeric(kar$survival != "1")
out$ch21_logit_diag = ct(glm(loneyear ~ highdiag + age + rural + male, data = kar, family = binomial, control = tight))
mn = nnet::multinom(survival ~ highdiag + age + rural + male, data = kar, trace = FALSE, abstol = 1e-14, reltol = 1e-14, maxit = 2000)
smn = summary(mn)
out$ch21_multinom = list(levels = mn$lev, names = colnames(smn$coefficients), est = unname(smn$coefficients),
                         se = unname(smn$standard.errors), deviance = mn$deviance,
                         probs = unname(predict(mn, type = "probs")[1:5, ]))
po = polr(factor(survival) ~ highdiag + age + rural + male, Hess = TRUE, data = kar)
spo = summary(po)$coefficients
out$ch21_polr = list(names = rownames(spo), est = unname(spo[, 1]), se = unname(spo[, 2]), deviance = po$deviance,
                     probs = unname(predict(po, type = "probs")[1:5, ]))
data("Fishing", package = "mlogit")
save_csv(Fishing, "Fishing.csv")
Fish = dfidx(Fishing, varying = 2:9, shape = "wide", choice = "mode")
ml = function(f) { m = mlogit(f, data = Fish); s = summary(m)$CoefTable
  list(names = rownames(s), est = unname(s[, 1]), se = unname(s[, 2]), loglik = as.numeric(logLik(m))) }
out$ch21_fish_cond0 = ml(mode ~ 0 + price + catch)
out$ch21_fish_cond = ml(mode ~ price + catch)
out$ch21_fish_ind = ml(mode ~ 0 | income)
out$ch21_fish_general = ml(mode ~ price + catch | income)

## ---- chapter 22: count outcomes -----------------------------------------
message("chapter 22: count outcomes")
gym = read.dta(paste0(D, "gym_treatment_exp_weekly.dta"))
f.reg = weekly_visit ~ incentive_commit + incentive + target + member_gym_pre
out$ch22 = list()
for (wk in c(1, 5, 20, 52)) {
  gw = gym[which(gym$incentive_week == wk), ]
  ols = lm(f.reg, data = gw)
  po_ = glm(f.reg, family = poisson(link = "log"), data = gw, control = tight)
  nb = nb_fit(f.reg, data = gw)
  zp = zeroinfl(f.reg, dist = "poisson", data = gw, reltol = 1e-14)
  zn = zeroinfl(f.reg, dist = "negbin", data = gw, reltol = 1e-14)
  cz = function(z) { s = summary(z)$coef
    list(count = unname(s$count[1:5, 1]), count_se = unname(s$count[1:5, 2]),
         zero = unname(s$zero[, 1]), zero_se = unname(s$zero[, 2]), loglik = as.numeric(logLik(z)), aic = AIC(z)) }
  out$ch22[[paste0("week", wk)]] = list(
    ols = c(ct(ols), list(aic = AIC(ols))), poisson = c(ct(po_), list(aic = po_$aic, loglik = as.numeric(logLik(po_)))),
    nb = if (is.null(nb)) NULL else c(ct(nb), list(theta = nb$theta, se_theta = nb$SE.theta, aic = nb$aic, loglik = as.numeric(logLik(nb)))),
    zip = cz(zp), zinb = c(cz(zn), list(theta = zn$theta)))
}

## ---- chapter 24: sandwich for GLMs --------------------------------------
message("chapter 24: sandwich for GLMs")
out$ch24_boston = list(hc0 = unname(sqrt(diag(vcovHC(boston_fit, type = "HC0")))),
                       hc1 = unname(sqrt(diag(vcovHC(boston_fit, type = "HC1")))),
                       hc3 = unname(sqrt(diag(vcovHC(boston_fit, type = "HC3")))))
out$ch24_flu_sandwich = unname(sqrt(diag(sandwich(flu_logit))))
set.seed(24); n = 1000; x = rnorm(n); y_pois = rpois(n, exp(x / 5)); y_nb = rnegbin(n, mu = exp(x / 5), theta = 0.2)
y_wr = rpois(n, x^2)
save_csv(data.frame(x = x, y_pois = y_pois, y_nb = y_nb, y_wr = y_wr), "sim_counts.csv")
sw = function(m) list(est = unname(coef(m)), se = unname(sqrt(diag(vcov(m)))), robust = unname(sqrt(diag(sandwich(m)))))
out$ch24_pois_pois = sw(glm(y_pois ~ x, family = poisson, control = tight))
out$ch24_nb_pois = sw(glm(y_nb ~ x, family = poisson, control = tight))
nbnb = nb_fit(y_nb ~ x)
out$ch24_nb_nb = c(sw(nbnb), list(theta = nbnb$theta, se_theta = nbnb$SE.theta))
out$ch24_wr_pois = sw(glm(y_wr ~ x, family = poisson, control = tight))
wrnb = nb_fit(y_wr ~ x)
if (!is.null(wrnb)) out$ch24_wr_nb = c(sw(wrnb), list(theta = wrnb$theta))

## ---- chapter 25: generalized estimating equations -----------------------
message("chapter 25: generalized estimating equations")
gs = function(g) { s = summary(g)$coef
  list(names = rownames(s), est = unname(s[, 1]), naive = unname(s[, 2]), robust = unname(s[, 4]),
       scale = g$scale, alpha = g$working.correlation[1, min(2, ncol(g$working.correlation))]) }
quiet = function(expr) { sink(tempfile()); on.exit(sink()); suppressMessages(expr) }
Pten = read.csv(paste0(D, "PtenAnalysisData.csv"))[, -(7:9)]
Pten = Pten[order(Pten$mouseid), ]
out$ch25_pten_ind = gs(quiet(gee(somasize ~ factor(fa) * pten, id = mouseid, family = gaussian, corstr = "independence", data = Pten, tol = 1e-10, maxiter = 200)))
out$ch25_pten_exch = gs(quiet(gee(somasize ~ factor(fa) * pten, id = mouseid, family = gaussian, corstr = "exchangeable", data = Pten, tol = 1e-10, maxiter = 200)))
out$ch25_pten_exch_x = gs(quiet(gee(somasize ~ factor(fa) * pten + numctrl + numpten, id = mouseid, family = gaussian, corstr = "exchangeable", data = Pten, tol = 1e-10, maxiter = 200)))
hyg = read.csv(paste0(D, "hygaccess.csv"))
hyg = hyg[, c("r4_hyg_access", "treat_cat_1", "bl_c_hyg_access", "vid", "eligible")]
hyg = hyg[which(hyg$eligible == "Eligible" & hyg$r4_hyg_access != "Missing"), ]
hyg$y = ifelse(hyg$r4_hyg_access == "Yes", 1, 0); hyg$z = hyg$treat_cat_1; hyg$x = hyg$bl_c_hyg_access
save_csv(hyg[, c("y", "z", "x", "vid")], "hygaccess_analysis.csv")
out$ch25_hyg_ind = gs(quiet(gee(y ~ z, id = vid, family = binomial(link = logit), corstr = "independence", data = hyg, tol = 1e-10, maxiter = 200)))
out$ch25_hyg_exch = gs(quiet(gee(y ~ z, id = vid, family = binomial(link = logit), corstr = "exchangeable", data = hyg, tol = 1e-10, maxiter = 200)))
out$ch25_hyg_exch_x = gs(quiet(gee(y ~ z + x, id = vid, family = binomial(link = logit), corstr = "exchangeable", data = hyg, tol = 1e-10, maxiter = 200)))
out$ch25_gym_normal = gs(quiet(gee(f.reg, id = id, family = gaussian, corstr = "independence", data = gym, tol = 1e-10, maxiter = 200)))
out$ch25_gym_poisson = gs(quiet(gee(f.reg, id = id, family = poisson(link = log), corstr = "independence", data = gym, tol = 1e-10, maxiter = 200)))
out$ch25_gym_poisson_short = gs(quiet(gee(f.reg, id = id, subset = (incentive_week > 0 & incentive_week < 15),
                                          family = poisson(link = log), corstr = "independence", data = gym, tol = 1e-10, maxiter = 200)))

## ---- chapter 26: quantile regression ------------------------------------
message("chapter 26: quantile regression")
taus = (1:9) / 10
out$ch26_galton = list(taus = taus, coef = unname(coef(rq(childHeight ~ midparentHeight, tau = taus, data = galton))))
f.q = logwk ~ educ + exper + exper2 + black
for (yr in c("80", "90", "00")) {
  cen = read.dta(paste0(D, "census", yr, ".dta"))
  r0 = rq(f.q, data = cen, tau = taus); rw = rq(f.q, data = cen, tau = taus, weights = perwt)
  sk = summary(rw, se = "ker"); sn = summary(r0, se = "nid"); si = summary(r0, se = "iid"); sk0 = summary(r0, se = "ker")
  message("  census ", yr)
  out[[paste0("ch26_census", yr)]] = list(coef = unname(coef(r0)), coef_w = unname(coef(rw)),
    se_ker_w = unname(sapply(sk, function(s) s$coef[, 2])), se_nid_w = if (yr == "80") unname(sapply(c(0.1, 0.5, 0.9), function(t)
      summary(rq(f.q, data = cen, tau = t, weights = perwt), se = "nid")$coef[, 2])) else NULL,
    se_ker = unname(sapply(sk0, function(s) s$coef[, 2])),
    se_nid = unname(sapply(sn, function(s) s$coef[, 2])), se_iid = unname(sapply(si, function(s) s$coef[, 2])))
}
message("  star")
star = read.csv(paste0(D, "star.csv"))
star.rq = rq(pscore ~ small + regaide + black + girl + poor + tblack + texp + tmasters + factor(fe), data = star)
set.seed(26); b_iid = summary(star.rq, se = "boot", R = 200)$coef[2:9, 2]
set.seed(26); b_cl = summary(star.rq, se = "boot", cluster = star$classid, R = 200)$coef[2:9, 2]
out$ch26_star = list(names = names(coef(star.rq))[2:9], est = unname(coef(star.rq)[2:9]),
                     se_boot = unname(b_iid), se_boot_cluster = unname(b_cl),
                     se_ker = unname(summary(star.rq, se = "ker")$coef[2:9, 2]),
                     objective = sum(abs(resid(star.rq))) / 2)

## ---- chapter 27: survival -----------------------------------------------
message("chapter 27: survival")
COMBINE = read.table(paste0(D, "combine_data.txt"), header = TRUE)[, -1]
cx = function(m) { s = summary(m)$coef; r = "robust se" %in% colnames(s)
  list(names = rownames(s), est = unname(s[, "coef"]), se = unname(s[, "se(coef)"]),
       robust = if (r) unname(s[, "robust se"]) else NULL, loglik = m$loglik) }
km = survfit(Surv(futime, relapse) ~ NALTREXONE + THERAPY, data = COMBINE)
skm = summary(km, times = c(7, 28, 56, 84, 112))
out$ch27_km = list(strata = as.character(skm$strata), time = skm$time, surv = skm$surv, se = skm$std.err,
                   lower = skm$lower, upper = skm$upper)
out$ch27_combine = cx(coxph(Surv(futime, relapse) ~ NALTREXONE * THERAPY + AGE + GENDER + T0_PDA + site, robust = TRUE, data = COMBINE))
out$ch27_combine_breslow = cx(coxph(Surv(futime, relapse) ~ NALTREXONE * THERAPY + AGE + GENDER + T0_PDA + site, ties = "breslow", robust = TRUE, data = COMBINE))
out$ch27_combine_strata = cx(coxph(Surv(futime, relapse) ~ NALTREXONE * THERAPY + AGE + GENDER + T0_PDA + strata(site), robust = TRUE, data = COMBINE))
fda = read.dta(paste0(D, "fda.dta"))
out$ch27_fda = cx(coxph(Surv(acttime, censor) ~ hcomm + hfloor + scomm + sfloor + prespart + demhsmaj + demsnmaj +
  prevgenx + lethal + deathrt1 + acutediz + hosp01 + hospdisc + hhosleng + mandiz01 + femdiz01 + peddiz01 + orphdum +
  natreg + I(natreg^2) + vandavg3 + wpnoavg3 + condavg3 + orderent + stafcder, robust = TRUE, data = fda))
save_csv(gehan, "gehan.csv")
sd_ = survdiff(Surv(time, cens) ~ treat, data = gehan)
cg = coxph(Surv(time, cens) ~ treat, data = gehan); scg = summary(cg)
out$ch27_gehan = c(cx(cg), list(logrank_chisq = sd_$chisq, logrank_p = 1 - pchisq(sd_$chisq, 1),
  lr = unname(scg$logtest[1]), wald = unname(scg$waldtest[1]), score = unname(scg$sctest[1]),
  concordance = unname(scg$concordance[1])))
skg = summary(survfit(Surv(time, cens) ~ treat, data = gehan), times = c(5, 10, 15, 20))
out$ch27_gehan_km = list(strata = as.character(skg$strata), time = skg$time, surv = skg$surv, se = skg$std.err,
                         lower = skg$lower, upper = skg$upper)
if (requireNamespace("timereg", quietly = TRUE)) {
  data(diabetes, package = "timereg"); save_csv(diabetes, "diabetes_timereg.csv")
  out$ch27_diabetes_robust = cx(coxph(Surv(time, status) ~ treat + adult + agedx, robust = TRUE, data = diabetes))
  out$ch27_diabetes_cluster = cx(coxph(Surv(time, status) ~ treat + adult + agedx, robust = TRUE, cluster = id, data = diabetes))
}

target = file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "data", "ding_linear_model_R.json")
write(toJSON(out, digits = 15, auto_unbox = TRUE, null = "null"), target)
cat("wrote", target, "with", length(out), "entries\n")
