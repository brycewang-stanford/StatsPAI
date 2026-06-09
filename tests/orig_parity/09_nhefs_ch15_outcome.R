# Original-data parity for Hernán & Robins, *Causal Inference: What If*,
# Chapter 15 (Outcome regression and propensity scores).
#
#   Source data : sp.datasets.nhefs() complete case (n=1566), dumped by
#                 the Python side to data/09_nhefs_ch15_outcome.csv so
#                 both languages read identical bytes.
#   Published   : Program 15.1 -- qsmk main coef 2.56, interaction
#                 qsmk:smokeintensity 0.0467; qsmk effect at
#                 smokeintensity=5 is 2.79 kg, at 40 is 4.43 kg.
#                 Program 15.3 -- PS-as-covariate qsmk effect ~3.5 kg.
#                 Program 15.4 -- PS-decile stratification ~3.5 kg.
#
# This script reproduces the book's *exact* estimators in base R:
#   (a) lm() with a qsmk:smokeintensity interaction on the canonical
#       confounder set (factors + quadratics);
#   (b) a logistic propensity model glm(qsmk ~ L, binomial), then the
#       ATE both with the PS as a linear covariate and via PS-decile FE.
# Writes results/09_nhefs_ch15_outcome_R.json.

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

MODULE <- "09_nhefs_ch15_outcome"
df <- read.csv(file.path(DATA_DIR, paste0(MODULE, ".csv")))
n <- nrow(df)

# Canonical confounder fragment (matches _nhefs.R_CONF_FORMULA).
conf <- paste0(
  "factor(sex)+factor(race)+factor(education)+factor(exercise)+",
  "factor(active)+age+I(age^2)+smokeintensity+I(smokeintensity^2)+",
  "smokeyrs+I(smokeyrs^2)+wt71+I(wt71^2)"
)

# ---------------------------------------------------------------------
# (a) Outcome regression with effect modification -- Program 15.1.
# ---------------------------------------------------------------------
# qsmk + smokeintensity + qsmk:smokeintensity, plus the canonical
# confounders.  smokeintensity enters linearly here; its square is part
# of the confounder block, so drop the bare smokeintensity from `conf`
# is unnecessary -- lm() collapses duplicate terms automatically, but we
# build the RHS explicitly to mirror the Python formula exactly.
a_rhs <- paste0(
  "qsmk + smokeintensity + qsmk:smokeintensity + ",
  "factor(sex)+factor(race)+factor(education)+factor(exercise)+",
  "factor(active)+age+I(age^2)+I(smokeintensity^2)+",
  "smokeyrs+I(smokeyrs^2)+wt71+I(wt71^2)"
)
a_fit <- lm(as.formula(paste("wt82_71 ~", a_rhs)), data = df)
a_co <- summary(a_fit)$coefficients
b_qsmk <- unname(a_co["qsmk", "Estimate"])
se_qsmk <- unname(a_co["qsmk", "Std. Error"])
b_int <- unname(a_co["qsmk:smokeintensity", "Estimate"])
se_int <- unname(a_co["qsmk:smokeintensity", "Std. Error"])
eff_5 <- b_qsmk + 5 * b_int
eff_40 <- b_qsmk + 40 * b_int

# ---------------------------------------------------------------------
# (b) Propensity-score adjustment -- Programs 15.2-15.4.
# ---------------------------------------------------------------------
# PS model: logistic P(qsmk=1|L).
psmod <- glm(as.formula(paste("qsmk ~", conf)), data = df, family = binomial())
df$ps <- predict(psmod, type = "response")

# (b)(i) PS as a continuous covariate in the outcome model (Prog 15.3).
psi_fit <- lm(wt82_71 ~ qsmk + ps, data = df)
psi_co <- summary(psi_fit)$coefficients
b_psi <- unname(psi_co["qsmk", "Estimate"])
se_psi <- unname(psi_co["qsmk", "Std. Error"])
ci_psi <- b_psi + c(-1, 1) * qnorm(0.975) * se_psi

# (b)(ii) PS-decile stratification (Prog 15.4): ten PS strata as FE.
# Match pandas.qcut decile edges (type-7 quantiles, include lowest).
brks <- quantile(df$ps, probs = seq(0, 1, 0.1), type = 7)
df$psdecile <- cut(df$ps, breaks = brks, labels = FALSE,
                   include.lowest = TRUE)
psd_fit <- lm(wt82_71 ~ qsmk + factor(psdecile), data = df)
psd_co <- summary(psd_fit)$coefficients
b_psd <- unname(psd_co["qsmk", "Estimate"])
se_psd <- unname(psd_co["qsmk", "Std. Error"])
ci_psd <- b_psd + c(-1, 1) * qnorm(0.975) * se_psd

# Cross-reference: standardized marginal ATE via g-computation
# (fit outcome model with qsmk*everything, predict under qsmk=0/1, mean diff).
g_fit <- lm(as.formula(paste0("wt82_71 ~ qsmk*(smokeintensity) + ", conf)),
            data = df)
d0 <- df; d0$qsmk <- 0
d1 <- df; d1$qsmk <- 1
gcomp <- mean(predict(g_fit, d1)) - mean(predict(g_fit, d0))

rows <- list(
  list(module = MODULE, side = "R", statistic = "om_qsmk_main_coef",
       estimate = b_qsmk, se = se_qsmk, n = n, published = 2.56,
       citation = "Hernán-Robins, What If Program 15.1 (qsmk main coef)",
       extra = list(note = "conditional effect at smokeintensity=0; NOT marginal")),
  list(module = MODULE, side = "R", statistic = "om_qsmk_x_smkint",
       estimate = b_int, se = se_int, n = n, published = 0.0467,
       citation = "Hernán-Robins, What If Program 15.1 (qsmk:smokeintensity)",
       extra = list()),
  list(module = MODULE, side = "R", statistic = "om_effect_smkint5",
       estimate = eff_5, se = NULL, n = n, published = 2.79,
       citation = "Hernán-Robins, What If Program 15.1 (effect at smokeintensity=5)",
       extra = list()),
  list(module = MODULE, side = "R", statistic = "om_effect_smkint40",
       estimate = eff_40, se = NULL, n = n, published = 4.43,
       citation = "Hernán-Robins, What If Program 15.1 (effect at smokeintensity=40)",
       extra = list()),
  list(module = MODULE, side = "R", statistic = "ps_in_outcome_ate",
       estimate = b_psi, se = se_psi, n = n, published = 3.5,
       citation = "Hernán-Robins, What If Program 15.3 (PS as covariate)",
       extra = list()),
  list(module = MODULE, side = "R", statistic = "ps_decile_ate",
       estimate = b_psd, se = se_psd, n = n, published = 3.5,
       citation = "Hernán-Robins, What If Program 15.4 (PS-decile stratification)",
       extra = list()),
  list(module = MODULE, side = "R", statistic = "gcomp_std_ate",
       estimate = gcomp, se = NULL, n = n, published = 3.5,
       citation = "Hernán-Robins, What If Ch.15 (standardized marginal ATE, cross-ref)",
       extra = list())
)
payload <- list(
  module = MODULE, side = "R", rows = rows,
  extra = list(engine = "base R lm + glm(binomial) propensity score",
               ps_min = min(df$ps), ps_max = max(df$ps), ps_mean = mean(df$ps),
               ci_ps_in_outcome = ci_psi, ci_ps_decile = ci_psd)
)
writeLines(toJSON(payload, auto_unbox = TRUE, null = "null", digits = 10),
           file.path(RESULTS_DIR, paste0(MODULE, "_R.json")))
cat(sprintf(
  paste0("[%s] (a) qsmk_main=%.4f (book 2.56) int=%.4f (book 0.0467) ",
         "eff@5=%.3f (book 2.79) eff@40=%.3f (book 4.43)  ",
         "(b) ps_in_outcome=%.4f ps_decile=%.4f gcomp=%.4f (book ~3.5)\n"),
  MODULE, b_qsmk, b_int, eff_5, eff_40, b_psi, b_psd, gcomp))
