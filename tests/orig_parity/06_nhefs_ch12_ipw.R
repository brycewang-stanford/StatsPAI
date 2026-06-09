# Original-data parity for Hernán & Robins, *Causal Inference: What If*,
# Chapter 12 (IP weighting / marginal structural models).
#
#   Source data : sp.datasets.nhefs() complete case (n=1566), dumped by
#                 the Python side to data/06_nhefs_ch12_ipw.csv so both
#                 languages read identical bytes.
#   Published   : crude diff 2.54 kg (§12.2); IP-weighted ATE 3.4 kg,
#                 95% CI (2.4, 4.5) (Program 12.4).
#
# This script reproduces the book's *exact* estimator: a logistic
# propensity model for qsmk, stabilized weights sw = P(A)/P(A|L), and a
# saturated marginal structural model wt82_71 ~ qsmk fit by WLS with
# robust (HC1) standard errors.  Writes results/06_nhefs_ch12_ipw_R.json.

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
  library(sandwich)
  library(jsonlite)
})

MODULE <- "06_nhefs_ch12_ipw"
df <- read.csv(file.path(DATA_DIR, paste0(MODULE, ".csv")))
n <- nrow(df)

# Crude (unadjusted) difference.
crude <- mean(df$wt82_71[df$qsmk == 1]) - mean(df$wt82_71[df$qsmk == 0])

# Stabilized IP weights (book Program 12.3): denominator P(A=1|L) from a
# logistic model with the canonical confounder set; numerator P(A=1).
conf <- paste0(
  "factor(sex)+factor(race)+factor(education)+factor(exercise)+",
  "factor(active)+age+I(age^2)+smokeintensity+I(smokeintensity^2)+",
  "smokeyrs+I(smokeyrs^2)+wt71+I(wt71^2)"
)
den <- glm(as.formula(paste("qsmk ~", conf)), data = df, family = binomial())
num <- glm(qsmk ~ 1, data = df, family = binomial())
pd <- predict(den, type = "response")
pn <- predict(num, type = "response")
sw <- ifelse(df$qsmk == 1, pn / pd, (1 - pn) / (1 - pd))

# Saturated MSM via WLS, robust (HC1) SE -- book Program 12.4.
msm <- lm(wt82_71 ~ qsmk, data = df, weights = sw)
b <- unname(coef(msm)["qsmk"])
V <- sandwich::vcovHC(msm, type = "HC1")
se <- sqrt(V["qsmk", "qsmk"])
ci <- b + c(-1, 1) * qnorm(0.975) * se

rows <- list(
  list(module = MODULE, side = "R", statistic = "crude_diff",
       estimate = crude, se = NULL, n = n, published = 2.54,
       citation = "Hernán-Robins, What If §12.2 (crude)", extra = list()),
  list(module = MODULE, side = "R", statistic = "ipw_att",
       estimate = b, se = se, n = n, published = 3.4,
       citation = "Hernán-Robins, What If Program 12.4 (stabilized MSM)",
       extra = list())
)
payload <- list(
  module = MODULE, side = "R", rows = rows,
  extra = list(engine = "base R glm + lm(weights) + sandwich::HC1",
               sw_mean = mean(sw), sw_min = min(sw), sw_max = max(sw),
               ci_ipw = ci, published_ipw_ci = c(2.4, 4.5))
)
writeLines(toJSON(payload, auto_unbox = TRUE, null = "null", digits = 10),
           file.path(RESULTS_DIR, paste0(MODULE, "_R.json")))
cat(sprintf("[%s] crude=%.4f (book 2.54)  ipw_att=%.4f 95%% CI (%.2f,%.2f)  (book 3.4 [2.4,4.5])\n",
            MODULE, crude, b, ci[1], ci[2]))
