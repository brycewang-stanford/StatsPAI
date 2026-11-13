# R reference for the survival / epidemiology items of the 2026-11 gap list.
#
#   Rscript tests/reference_parity/_fixtures/_generate_gaps_epi_R.R
#
# Reads gaps_surv.csv, gaps_obs.csv, gaps_long.csv and gaps_gee_alpha.csv
# (_generate_gaps_epi_data.py); writes gaps_epi_R.json.
#
# R 4.5.2; survival 3.8.3, marginaleffects 0.32.0, WeightIt 1.7.0,
# sandwich 3.1.1, geepack 1.3.13, PSweight 2.1.2. GPL packages are run as
# black boxes.
suppressMessages({
  library(survival); library(marginaleffects); library(WeightIt)
  library(sandwich); library(geepack); library(PSweight); library(jsonlite)
})
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(sub("--file=", "", args[grep("--file=", args)]))
s <- read.csv(file.path(here, "gaps_surv.csv"))
o <- read.csv(file.path(here, "gaps_obs.csv"))
l <- read.csv(file.path(here, "gaps_long.csv"))
ga <- read.csv(file.path(here, "gaps_gee_alpha.csv"))
out <- list()

# --- 1. Cox, exact partial likelihood --------------------------------------
# (start, stop] data are left to Stata: coxph's exact method for counting-
# process data enumerates the subsets and does not finish on these data.
cx <- function(f) list(
  coef = as.numeric(coef(f)), se = as.numeric(sqrt(diag(vcov(f)))),
  loglik = f$loglik, score = f$score, wald = f$wald.test,
  concordance = unname(summary(f)$concordance[1])
)
f1 <- coxph(Surv(time, event) ~ age + ecog + arm, data = s, ties = "exact")
f2 <- coxph(Surv(time, event) ~ age + ecog + arm + strata(site), data = s, ties = "exact")
bh <- basehaz(f1, centered = FALSE)
out$cox_exact <- list(
  plain = c(cx(f1), list(basehaz_time = bh$time, basehaz = bh$hazard)),
  strata = cx(f2)
)

# --- 2. AFT, robust and clustered variance ---------------------------------
# survreg(robust = TRUE) is the sandwich without a small-sample factor;
# cluster = sums the score contributions within clinics, again no factor.
aft <- list()
for (dist in c("weibull", "exponential", "lognormal", "loglogistic")) {
  f <- survreg(Surv(time, event) ~ age + ecog + arm, data = s, dist = dist)
  fr <- survreg(Surv(time, event) ~ age + ecog + arm, data = s, dist = dist, robust = TRUE)
  fc <- survreg(Surv(time, event) ~ age + ecog + arm, data = s, dist = dist, cluster = clinic)
  aft[[dist]] <- list(
    coef = as.numeric(coef(f)), se = as.numeric(sqrt(diag(vcov(f)))),
    se_robust = as.numeric(sqrt(diag(vcov(fr)))),
    se_cluster = as.numeric(sqrt(diag(vcov(fc)))),
    n = nrow(s), n_clusters = length(unique(s$clinic))
  )
}
out$aft <- aft

# --- 3. g-computation from one GLM with the treatment as a regressor -------
gc <- list()
tight <- glm.control(epsilon = 1e-14, maxit = 100)
for (m in c("logit", "poisson")) {
  fam <- if (m == "logit") binomial() else poisson()
  yv <- if (m == "logit") "ybin" else "ycnt"
  g <- glm(as.formula(paste(yv, "~ a + x1 + x2 + x3")), data = o, family = fam, control = tight)
  one <- list()
  for (pop in c("ATE", "ATT", "ATC")) {
    nd <- switch(pop, ATE = o, ATT = o[o$a == 1, ], ATC = o[o$a == 0, ])
    p1 <- mean(predict(g, transform(nd, a = 1), type = "response"))
    p0 <- mean(predict(g, transform(nd, a = 0), type = "response"))
    ac <- avg_comparisons(g, variables = "a", newdata = nd)
    ar <- avg_comparisons(g, variables = "a", newdata = nd, comparison = "ratioavg")
    one[[pop]] <- list(
      difference = ac$estimate, ratio = ar$estimate,
      odds_ratio = if (m == "logit") (p1 / (1 - p1)) / (p0 / (1 - p0)) else NA,
      mean1 = p1, mean0 = p0
    )
  }
  gc[[m]] <- one
}
out$gcomp_pooled <- gc

# --- 4. IPW with weights truncated at percentiles ---------------------------
tr <- list()
for (est in c("ATE", "ATT", "ATC")) {
  W <- weightit(a ~ x1 + x2 + x3, data = o, method = "glm", estimand = est)
  for (at in c(0.95, 0.99)) {
    wt <- suppressMessages(trim(W, at = at, lower = TRUE))$weights
    f <- lm(y ~ a, data = o, weights = wt)
    foc <- if (est == "ATT") o$a == 1 else if (est == "ATC") o$a == 0 else rep(FALSE, nrow(o))
    tr[[paste(est, at, "both", sep = "_")]] <- list(
      estimate = unname(coef(f)[2]),
      se_hc0 = unname(sqrt(diag(vcovHC(f, type = "HC0")))[2]),
      lower = min(wt[!foc]), upper = max(wt[!foc])
    )
  }
  wt <- suppressMessages(trim(W, at = 0.9))$weights
  f <- lm(y ~ a, data = o, weights = wt)
  tr[[paste(est, "0.9_upper", sep = "_")]] <- list(
    estimate = unname(coef(f)[2]),
    se_hc0 = unname(sqrt(diag(vcovHC(f, type = "HC0")))[2]),
    upper = max(wt)
  )
}
out$ipw_truncate <- tr

# --- 5. GEE: QIC -------------------------------------------------------------
# Independence: geepack's own fit. Exchangeable and AR(1): geepack with the
# working correlation fixed at the value sp.gee estimates (gaps_gee_alpha.csv),
# so that both programs evaluate the criterion at the same coefficients.
qic <- list()
yof <- c(binomial = "ybin", poisson = "ycnt", gaussian = "y")
l <- l[order(l$pid, l$visit), ]
for (fam in names(yof)) {
  fm <- as.formula(paste(yof[fam], "~ trt + visit + age"))
  f <- geeglm(fm, id = pid, data = l, family = fam, corstr = "independence")
  q <- QIC(f)
  qic[[paste(fam, "independence", sep = "_")]] <- list(
    QIC = unname(q["QIC"]), QICu = unname(q["QICu"]),
    quasi = unname(q["Quasi Lik"]), CIC = unname(q["CIC"]),
    coef = as.numeric(coef(f)))
  for (cs in c("exchangeable", "ar1")) {
    a <- ga$alpha[ga$family == fam & ga$corstr == cs]
    R <- if (cs == "exchangeable") matrix(a, 4, 4) + diag(1 - a, 4) else a^abs(outer(1:4, 1:4, "-"))
    zc <- fixed2Zcor(R, id = l$pid, waves = l$visit + 1)
    f <- geeglm(fm, id = pid, data = l, family = fam, corstr = "fixed", zcor = zc)
    q <- QIC(f)
    qic[[paste(fam, cs, sep = "_")]] <- list(
      QIC = unname(q["QIC"]), QICu = unname(q["QICu"]),
      quasi = unname(q["Quasi Lik"]), CIC = unname(q["CIC"]),
      coef = as.numeric(coef(f)))
  }
}
out$gee_qic <- qic

# --- 8. Overlap weights: sandwich standard error ----------------------------
ps <- list()
for (w in c("overlap", "IPW", "treated", "entropy")) {
  r <- PSweight(ps.formula = a ~ x1 + x2 + x3, yname = "y", data = o, weight = w)
  sm <- summary(r)$estimates
  ps[[w]] <- list(estimate = sm[1, "Estimate"], se = sm[1, "Std.Error"])
}
out$psweight <- ps

out[["_meta"]] <- list(
  R = R.version.string,
  survival = as.character(packageVersion("survival")),
  marginaleffects = as.character(packageVersion("marginaleffects")),
  WeightIt = as.character(packageVersion("WeightIt")),
  sandwich = as.character(packageVersion("sandwich")),
  geepack = as.character(packageVersion("geepack")),
  PSweight = as.character(packageVersion("PSweight"))
)
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE, na = "null"),
           file.path(here, "gaps_epi_R.json"))
cat("wrote gaps_epi_R.json\n")
