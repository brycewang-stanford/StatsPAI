# CausalAgentBench v2: reference values for the gold answers.
#
# Computes, on the frozen files in tests/agent_bench/data/, every quantity
# the tasks ask for with the canonical R implementation, and for tasks that
# leave the method open, the value under each acceptable estimator.
# build_golds.py turns reference_values.json into golds.json.
#
# Run from tests/agent_bench/:   Rscript golds/reference.R
suppressPackageStartupMessages({
  library(jsonlite); library(sandwich); library(AER); library(fixest)
  library(did); library(rdrobust); library(rddensity); library(Synth)
  library(synthdid); library(Matching); library(DoubleML); library(mlr3)
  library(mlr3learners); library(survival); library(HonestDiD)
  library(bacondecomp); library(etwfe); library(didimputation); library(EValue)
  library(lme4); library(frontier); library(sensemakr); library(vars)
  library(plm)
})
lgr::get_logger("mlr3")$set_threshold("warn")
D <- function(f) read.csv(file.path("data", f))
out <- list()
put <- function(key, value) out[[key]] <<- unname(as.numeric(value))

# ------------------------------------------------------------------ card
card <- D("card.csv")
X <- "exper + expersq + black + south + smsa"
m <- lm(as.formula(paste("lwage ~ educ +", X)), card)
put("ols.beta_educ", coef(m)["educ"])
put("ols.se_educ", sqrt(diag(vcovHC(m, "HC1")))["educ"])
iv <- ivreg(as.formula(paste("lwage ~ educ +", X, "| nearc4 +", X)), data = card)
put("iv.beta_educ", coef(iv)["educ"])
put("iv.se_educ", sqrt(diag(vcovHC(iv, "HC1")))["educ"])
iv0 <- ivreg(lwage ~ educ | nearc4, data = card)
put("iv_nocontrols.beta_educ", coef(iv0)["educ"])
fs <- lm(as.formula(paste("educ ~ nearc4 +", X)), card)
put("iv.first_stage_f", summary(fs)$coefficients["nearc4", "t value"]^2)
dml_theta <- c(); dml_se <- c()
for (s in 1:20) {
  set.seed(1000 + 100000 * s)
  dd <- DoubleMLData$new(card, y_col = "lwage", d_cols = "educ",
                         x_cols = c("exper", "expersq", "black", "south", "smsa"))
  fit <- DoubleMLPLR$new(dd, lrn("regr.lm"), lrn("regr.lm"), n_folds = 5)
  fit$fit()
  dml_theta <- c(dml_theta, fit$coef); dml_se <- c(dml_se, fit$se)
}
put("dml.theta", mean(dml_theta)); put("dml.theta_sd_over_splits", sd(dml_theta))
put("dml.se", mean(dml_se))

# ------------------------------------------------------------------ mpdta
mp <- D("mpdta.csv")
cs_att <- function(est, cg, base = "varying") {
  a <- att_gt("lemp", "year", "countyreal", "first_treat", data = mp,
              est_method = est, control_group = cg, base_period = base,
              bstrap = FALSE, cband = FALSE)
  aggte(a, type = "simple", bstrap = FALSE, cband = FALSE)
}
for (est in c("dr", "ipw", "reg")) for (cg in c("nevertreated", "notyettreated")) {
  ag <- cs_att(est, cg)
  put(sprintf("cs_%s_%s.att", est, cg), ag$overall.att)
  put(sprintf("cs_%s_%s.se", est, cg), ag$overall.se)
}
mp$cohort <- ifelse(mp$first_treat == 0, 10000, mp$first_treat)
sa <- feols(lemp ~ sunab(cohort, year) | countyreal + year, mp, cluster = ~countyreal)
sa_att <- summary(sa, agg = "att")$coeftable
put("sa.att", sa_att["ATT", "Estimate"]); put("sa.se", sa_att["ATT", "Std. Error"])
co <- coef(sa); rel <- suppressWarnings(as.numeric(sub("^year::", "", names(co))))
put("sa_eventtime.att", mean(co[!is.na(rel) & rel >= 0]))
ew <- etwfe(lemp ~ 0, tvar = year, gvar = first_treat, data = mp, vcov = ~countyreal)
em <- emfx(ew, type = "simple")
put("etwfe.att", em$estimate[1]); put("etwfe.se", em$std.error[1])
bjs <- did_imputation(mp, yname = "lemp", gname = "first_treat", tname = "year",
                      idname = "countyreal")
put("bjs.att", bjs$estimate[1]); put("bjs.se", bjs$std.error[1])
mp$D <- as.integer(mp$first_treat > 0 & mp$year >= mp$first_treat)
tw <- feols(lemp ~ D | countyreal + year, mp, cluster = ~countyreal)
put("twfe.beta", coef(tw)["D"]); put("twfe.se", se(tw)["D"])
bc <- bacon(lemp ~ D, data = mp, id_var = "countyreal", time_var = "year", quietly = TRUE)
put("bacon.beta_twfe", sum(bc$estimate * bc$weight))
put("bacon.weight_later_vs_earlier", sum(bc$weight[bc$type == "Later vs Earlier Treated"]))
honest <- function(base) {
  a <- att_gt("lemp", "year", "countyreal", "first_treat", data = mp,
              est_method = "dr", control_group = "nevertreated",
              base_period = base, bstrap = FALSE, cband = FALSE)
  es <- aggte(a, type = "dynamic", bstrap = FALSE, cband = FALSE)
  e <- es$egt; keep <- e != -1
  beta <- es$att.egt[keep]
  inf <- es$inf.function$dynamic.inf.func.e[, keep, drop = FALSE]
  sigma <- crossprod(inf) / nrow(inf)^2
  npre <- sum(e[keep] < 0); npost <- sum(e[keep] >= 0)
  r <- createSensitivityResults(beta, sigma, numPrePeriods = npre,
                                numPostPeriods = npost, Mvec = 0.05,
                                l_vec = basisVector(1, npost))
  c(r$lb[1], r$ub[1])
}
h <- honest("universal"); put("honest_universal.ci_lower", h[1]); put("honest_universal.ci_upper", h[2])
h <- honest("varying"); put("honest_varying.ci_lower", h[1]); put("honest_varying.ci_upper", h[2])

# ------------------------------------------------------------------ lee
lee <- D("lee.csv")
rd <- rdrobust(lee$voteshare_next, lee$margin)
put("rd.tau_conventional", rd$coef[1]); put("rd.tau_bias_corrected", rd$coef[2])
put("rd.se_robust", rd$se[3])
dens <- rddensity(lee$margin)
put("rddensity.p_value", dens$test$p_jk)

# ------------------------------------------------------------------ basque
bq <- D("basque.csv")
bq$id <- as.integer(factor(bq$region))
tr <- unique(bq$id[bq$treated == 1])
pre <- sort(unique(bq$year[bq$year < 1970])); post <- sort(unique(bq$year[bq$year >= 1970]))
sp_list <- lapply(pre, function(y) list("gdppc", y, "mean"))
dp <- dataprep(foo = bq, predictors = NULL, special.predictors = sp_list,
               dependent = "gdppc", unit.variable = "id", time.variable = "year",
               treatment.identifier = tr, controls.identifier = setdiff(unique(bq$id), tr),
               time.predictors.prior = pre, time.optimize.ssr = pre,
               unit.names.variable = "region", time.plot = sort(unique(bq$year)))
so <- invisible(capture.output(syn <- synth(dp, optimxmethod = "BFGS", verbose = FALSE)))
gap <- dp$Y1plot - dp$Y0plot %*% syn$solution.w
yrs <- as.numeric(rownames(dp$Y1plot))
put("synth.avg_post_gap", mean(gap[yrs >= 1970]))
bq$W <- as.integer(bq$id == tr & bq$year >= 1970)
pm <- panel.matrices(bq[, c("id", "year", "gdppc", "W")])
put("sdid_basque.avg_post_gap", synthdid_estimate(pm$Y, pm$N0, pm$T0))
put("sc_basque.avg_post_gap", sc_estimate(pm$Y, pm$N0, pm$T0))

# ------------------------------------------------------------------ california
ca <- D("california.csv")
ca$W <- as.integer(ca$treated == 1)
pmc <- panel.matrices(ca[, c("state", "year", "cigsale", "W")])
put("sdid.att", synthdid_estimate(pmc$Y, pmc$N0, pmc$T0))
put("sc.att", sc_estimate(pmc$Y, pmc$N0, pmc$T0))
put("did.att", did_estimate(pmc$Y, pmc$N0, pmc$T0))

# ------------------------------------------------------------------ nsw
nsw <- D("nsw_dw.csv")
cov <- c("age", "education", "black", "hispanic", "married", "re74", "re75")
ps <- glm(as.formula(paste("treat ~", paste(cov, collapse = "+"))), binomial, nsw)
mt <- Match(Y = nsw$re78, Tr = nsw$treat, X = fitted(ps), estimand = "ATT", M = 1,
            replace = TRUE, ties = TRUE)
put("psm.att", mt$est)
mt1 <- Match(Y = nsw$re78, Tr = nsw$treat, X = fitted(ps), estimand = "ATT", M = 1,
             replace = TRUE, ties = FALSE)
put("psm_noties.att", mt1$est)
allcov <- c(cov, "nodegree")
reg <- lm(as.formula(paste("re78 ~ treat +", paste(allcov, collapse = "+"))), nsw)
put("ols_adjust.att", coef(reg)["treat"])
sens <- sensemakr(reg, treatment = "treat")
put("sensemakr.rv_q", sens$sensitivity_stats$rv_q)
psa <- glm(as.formula(paste("treat ~", paste(allcov, collapse = "+"))), binomial, nsw)
e <- fitted(psa); t1 <- nsw$treat == 1
w0 <- e[!t1] / (1 - e[!t1])
put("ipw.att", mean(nsw$re78[t1]) - sum(w0 * nsw$re78[!t1]) / sum(w0))
mu0 <- predict(lm(as.formula(paste("re78 ~", paste(allcov, collapse = "+"))), nsw[!t1, ]), nsw)
p1 <- mean(t1)
put("aipw.att", mean(t1 * (nsw$re78 - mu0) / p1 - (!t1) * e * (nsw$re78 - mu0) / ((1 - e) * p1)))

# ------------------------------------------------------------------ survival
sv <- D("synth_survival.csv")
cx <- coxph(Surv(time, event) ~ x1 + x2 + treat, sv, ties = "efron")
put("cox.beta_x1", coef(cx)["x1"]); put("cox.beta_x2", coef(cx)["x2"])
put("cox.beta_treat", coef(cx)["treat"])
put("cox.ph_global_p_value", cox.zph(cx)$table["GLOBAL", "p"])
put("cox_breslow.beta_treat", coef(coxph(Surv(time, event) ~ x1 + x2 + treat, sv, ties = "breslow"))["treat"])

# ------------------------------------------------------------------ E-value
ev <- evalues.RR(2.5, 1.8, 3.2)
put("evalue.evalue_estimate", ev["E-values", "point"])
put("evalue.evalue_ci", ev["E-values", "lower"])

# ------------------------------------------------------------------ panel
pn <- D("panel.csv")
lm1 <- lmer(y ~ x1 + (1 | gid), pn, REML = TRUE)
vc <- as.data.frame(VarCorr(lm1))
put("lmm.beta_intercept", fixef(lm1)[1]); put("lmm.beta_x1", fixef(lm1)[2])
put("lmm.icc", vc$vcov[1] / sum(vc$vcov))
put("lmm_ml.beta_x1", fixef(lmer(y ~ x1 + (1 | gid), pn, REML = FALSE))[2])
put("pooled_ols.beta_x1", coef(lm(y ~ x1, pn))["x1"])
pdat <- pdata.frame(pn, index = c("gid"))
pdat$t <- ave(pn$gid, pn$gid, FUN = seq_along)
pdat <- pdata.frame(data.frame(pn, t = pdat$t), index = c("gid", "t"))
fe <- plm(y ~ x1, pdat, model = "within"); re <- plm(y ~ x1, pdat, model = "random")
put("fe.beta_x1", coef(fe)["x1"]); put("re.beta_x1", coef(re)["x1"])
put("hausman.p_value", phtest(fe, re)$p.value)
lp <- D("logit_panel.csv")
g1 <- glmer(y ~ x1 + (1 | gid), lp, family = binomial, nAGQ = 1)
put("glmm.beta_intercept", fixef(g1)[1]); put("glmm.beta_x1", fixef(g1)[2])
put("glmm_agq.beta_x1", fixef(glmer(y ~ x1 + (1 | gid), lp, family = binomial, nAGQ = 10))[2])

# ------------------------------------------------------------------ frontier
sf <- D("sfa.csv")
fr <- sfa(lny ~ lnk + lnl, data = sf)
put("sfa.beta_lnk", coef(fr)["lnk"]); put("sfa.beta_lnl", coef(fr)["lnl"])
put("sfa.mean_efficiency", mean(efficiencies(fr)))
put("sfa_ols.beta_lnk", coef(lm(lny ~ lnk + lnl, sf))["lnk"])

# ------------------------------------------------------------------ wage gap
wg <- D("wage_gap.csv")
A <- wg[wg$female == 0, ]; B <- wg[wg$female == 1, ]
bA <- coef(lm(log_wage ~ educ + exper, A)); bB <- coef(lm(log_wage ~ educ + exper, B))
xA <- c(1, colMeans(A[, c("educ", "exper")])); xB <- c(1, colMeans(B[, c("educ", "exper")]))
put("oaxaca.gap", mean(A$log_wage) - mean(B$log_wage))
put("oaxaca.endowments", sum((xA - xB) * bB))
put("oaxaca.coefficients", sum(xB * (bA - bB)))
put("oaxaca.interaction", sum((xA - xB) * (bA - bB)))
put("oaxaca_refA.explained", sum((xA - xB) * bA))
put("oaxaca_refB.explained", sum((xA - xB) * bB))
bP <- coef(lm(log_wage ~ educ + exper, wg))
put("oaxaca_pooled.explained", sum((xA - xB) * bP))
bPd <- coef(lm(log_wage ~ educ + exper + female, wg))[1:3]
put("oaxaca_pooled_dummy.explained", sum((xA - xB) * bPd))
dfl <- function(f) {
  pm <- glm(f, binomial, wg); p <- fitted(pm)
  w <- (1 - p) / p * mean(wg$female) / (1 - mean(wg$female))
  sum(w[wg$female == 1] * B$log_wage) / sum(w[wg$female == 1])
}
put("dfl.counterfactual_mean", dfl(female ~ educ + exper))
put("dfl_sq.counterfactual_mean", dfl(female ~ educ + exper + I(educ^2) + I(exper^2)))
put("dfl.explained", dfl(female ~ educ + exper) - mean(B$log_wage))

# ------------------------------------------------------------------ mediation
md <- D("mediation.csv")
ma <- lm(m ~ treat + x, md); mb <- lm(y ~ treat + m + x, md)
put("mediation.acme", coef(ma)["treat"] * coef(mb)["m"])
put("mediation.ade", coef(mb)["treat"])

# ------------------------------------------------------------------ trade
trd <- D("trade.csv")
ft <- feols(y ~ x1 + x2 | firm + year, trd, cluster = ~firm)
put("hdfe.beta_x1", coef(ft)["x1"]); put("hdfe.beta_x2", coef(ft)["x2"])
put("hdfe.se_x1", se(ft)["x1"])

# ------------------------------------------------------------------ macro
mc <- D("macro.csv")
v <- VAR(mc[, c("y1", "y2")], p = 2, type = "const")
put("var.y1_on_L1_y1", coef(v)$y1["y1.l1", "Estimate"])
put("var.y2_on_L1_y1", coef(v)$y2["y1.l1", "Estimate"])
P <- t(chol(summary(v)$covres))
ir <- irf(v, impulse = "y1", response = "y2", n.ahead = 1, ortho = TRUE, boot = FALSE)
put("var.irf_var_h1", ir$irf$y1[2, "y2"] / P[1, 1])
n <- nrow(mc); y1 <- mc$y1; y2 <- mc$y2
idx <- 3:(n - 1)
lpd <- data.frame(lead = y2[idx + 1], y1 = y1[idx], y1l1 = y1[idx - 1], y1l2 = y1[idx - 2],
                  y2l1 = y2[idx - 1], y2l2 = y2[idx - 2])
put("lp.irf_lp_h1", coef(lm(lead ~ y1 + y1l1 + y1l2 + y2l1 + y2l2, lpd))["y1"])

out$`_versions` <- NULL
meta <- list(r = R.version.string,
             packages = sapply(c("did", "fixest", "rdrobust", "rddensity", "Synth",
                                 "synthdid", "Matching", "DoubleML", "survival",
                                 "HonestDiD", "bacondecomp", "etwfe", "didimputation",
                                 "EValue", "lme4", "frontier", "sensemakr", "vars",
                                 "plm", "AER", "sandwich"),
                               function(p) as.character(packageVersion(p))))
writeLines(toJSON(list(values = out, meta = meta), digits = NA, auto_unbox = TRUE,
                  pretty = TRUE), "golds/reference_values.json")
cat("wrote", length(out), "values\n")
