# Original-data parity for Hernán & Robins, *Causal Inference: What If*,
# Chapter 14 (G-estimation of a structural nested mean model).
#
#   Source data : sp.datasets.nhefs() complete case (n=1566), dumped by
#                 the Python side to data/08_nhefs_ch14_gestimation.csv so
#                 both languages read identical bytes.
#   Published   : Program 14.2 -- additive SNMM psi ~ 3.4 kg,
#                 95% CI (2.5, 4.5).
#
# This script carries TWO base-R references:
#
#   (1) snmm_psi_logistic -- the book's EXACT Program 14.2 procedure.
#       For a candidate psi form H(psi) = wt82_71 - psi*qsmk, fit the
#       logistic propensity model qsmk ~ H(psi) + L over the canonical
#       confounders, and find the psi at which the coefficient on H(psi)
#       is exactly zero (root-find).  The 95% CI is read off where that
#       coefficient's Wald z-statistic crosses +/-1.96 (book's approach).
#
#   (2) snmm_psi_linear -- the linear moment-condition g-estimator that
#       StatsPAI's sp.g_estimation implements: residualise wt82_71 and
#       qsmk on L (OLS) and set psi = cov(Yres, Ares)/var(Ares).  This is
#       the tight (machine-precision) cross-language anchor for the
#       StatsPAI point estimate.
#
# Writes results/08_nhefs_ch14_gestimation_R.json.

.script_dir <- (function() {
  args <- commandArgs(trailingOnly = FALSE)
  m <- grep("^--file=", args, value = TRUE)
  if (length(m) > 0) dirname(normalizePath(sub("^--file=", "", m[1])))
  else getwd()
})()
HERE <- .script_dir
DATA_DIR <- file.path(HERE, "data")
RESULTS_DIR <- file.path(HERE, "results")
dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages({
  library(jsonlite)
})

MODULE <- "08_nhefs_ch14_gestimation"
df <- read.csv(file.path(DATA_DIR, paste0(MODULE, ".csv")))
n <- nrow(df)

A <- df$qsmk
Y <- df$wt82_71

# Canonical confounder set (book Programs 12.x-15.x): categoricals as
# factors, continuous with quadratics -- identical to _nhefs.book_design.
conf <- paste0(
  "factor(sex)+factor(race)+factor(education)+factor(exercise)+",
  "factor(active)+age+I(age^2)+smokeintensity+I(smokeintensity^2)+",
  "smokeyrs+I(smokeyrs^2)+wt71+I(wt71^2)"
)

# ----- (1) Book Program 14.2: logistic g-estimation -----------------------
# For candidate psi, return the coefficient on H(psi) (and its Wald z) in
# the logistic model qsmk ~ H(psi) + L.
fit_H <- function(psi) {
  df$Hpsi <- Y - psi * A
  fml <- as.formula(paste("qsmk ~ Hpsi +", conf))
  m <- glm(fml, data = df, family = binomial())
  s <- summary(m)$coefficients
  c(coef = unname(s["Hpsi", "Estimate"]),
    z = unname(s["Hpsi", "Estimate"] / s["Hpsi", "Std. Error"]))
}

coef_on_H <- function(psi) fit_H(psi)["coef"]
wald_z    <- function(psi) fit_H(psi)["z"]

psi_logistic <- uniroot(coef_on_H, lower = 2.0, upper = 5.0,
                        tol = 1e-8)$root
# CI: psi values where the Wald z of the H(psi) coefficient hits +/-1.96.
ci_lo <- uniroot(function(p) wald_z(p) - 1.96,
                 lower = 0.0, upper = psi_logistic, tol = 1e-6)$root
ci_hi <- uniroot(function(p) wald_z(p) + 1.96,
                 lower = psi_logistic, upper = 7.0, tol = 1e-6)$root

# ----- (2) Linear moment-condition g-estimator (StatsPAI algorithm) -------
des <- as.formula(paste("~", conf))
X <- model.matrix(des, data = df)          # includes intercept
Yres <- residuals(lm.fit(X, Y))
Ares <- residuals(lm.fit(X, A))
psi_linear <- sum(Yres * Ares) / sum(Ares^2)

rows <- list(
  list(module = MODULE, side = "R", statistic = "snmm_psi",
       estimate = psi_logistic, se = NULL, n = n, published = 3.4,
       citation = "Hernán-Robins, What If Program 14.2 (logistic g-estimation)",
       extra = list(ci = c(ci_lo, ci_hi))),
  list(module = MODULE, side = "R", statistic = "snmm_psi_linear",
       estimate = psi_linear, se = NULL, n = n, published = 3.4,
       citation = "Linear moment-condition g-estimate (StatsPAI algorithm)",
       extra = list())
)
payload <- list(
  module = MODULE, side = "R", rows = rows,
  extra = list(engine = "base R glm root-find (Program 14.2) + lm.fit linear g-est",
               estimand = "additive SNMM psi (qsmk -> wt82_71)",
               ci_snmm = c(ci_lo, ci_hi),
               published_psi = 3.4,
               published_snmm_ci = c(2.5, 4.5))
)
writeLines(toJSON(payload, auto_unbox = TRUE, null = "null", digits = 10),
           file.path(RESULTS_DIR, paste0(MODULE, "_R.json")))
cat(sprintf(
  "[%s] logistic psi=%.4f 95%% CI (%.2f,%.2f)  linear psi=%.4f  (book 3.4 [2.5,4.5])\n",
  MODULE, psi_logistic, ci_lo, ci_hi, psi_linear))
