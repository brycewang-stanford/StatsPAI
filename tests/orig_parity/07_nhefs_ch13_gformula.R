# Original-data parity for Hernán & Robins, *Causal Inference: What If*,
# Chapter 13 (standardization / the parametric g-formula).
#
#   Source data : sp.datasets.nhefs() complete case (n=1566), dumped by
#                 the Python side to data/07_nhefs_ch13_gformula.csv so
#                 both languages read identical bytes.
#   Published   : crude diff 2.54 kg (§12.2); standardized ATE 3.5 kg,
#                 bootstrap 95% CI ≈ (2.6, 4.5) (Program 13.3, outcome
#                 model wt82_71 ~ qsmk + qsmk:smokeintensity + conf).
#
# This script reproduces the book's *exact* standardization estimator in
# base R: fit the linear outcome model with the qsmk:smokeintensity
# effect-modification term, predict every subject's outcome under qsmk=1
# and qsmk=0, average the contrast; 95% CI by nonparametric bootstrap
# (boot package, 2000 reps).  It also reports the *additive* outcome
# model (no interaction) as a tight machine-precision cross-language
# anchor for StatsPAI's sp.g_computation, which fits an additive Q.
# Writes results/07_nhefs_ch13_gformula_R.json.

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
  library(boot)
  library(jsonlite)
})

MODULE <- "07_nhefs_ch13_gformula"
df <- read.csv(file.path(DATA_DIR, paste0(MODULE, ".csv")))
n <- nrow(df)

# Canonical confounder model (book Programs 12.x-15.x): categoricals as
# factors; continuous with quadratics.
conf <- paste0(
  "factor(sex)+factor(race)+factor(education)+factor(exercise)+",
  "factor(active)+age+I(age^2)+smokeintensity+I(smokeintensity^2)+",
  "smokeyrs+I(smokeyrs^2)+wt71+I(wt71^2)"
)

# Crude (unadjusted) difference.
crude <- mean(df$wt82_71[df$qsmk == 1]) - mean(df$wt82_71[df$qsmk == 0])

# ---- Standardization point estimate for a given resample ----
# fmla is the outcome model formula; computes E[Y^{a=1}] - E[Y^{a=0}] by
# predicting the whole (resampled) sample under qsmk=1 and qsmk=0.
std_ate <- function(data, fmla) {
  fit <- lm(as.formula(fmla), data = data)
  d1 <- data; d1$qsmk <- 1
  d0 <- data; d0$qsmk <- 0
  mean(predict(fit, newdata = d1)) - mean(predict(fit, newdata = d0))
}

fmla_int <- paste("wt82_71 ~ qsmk + qsmk:smokeintensity +", conf)  # book 13.3
fmla_add <- paste("wt82_71 ~ qsmk +", conf)                        # additive Q

ate_int <- std_ate(df, fmla_int)
ate_add <- std_ate(df, fmla_add)

# ---- Nonparametric bootstrap of the book (interaction) estimator ----
set.seed(42)
boot_stat <- function(data, idx) std_ate(data[idx, , drop = FALSE], fmla_int)
bres <- boot(df, boot_stat, R = 2000)
se_int <- sd(bres$t)
ci_int <- as.numeric(quantile(bres$t, c(0.025, 0.975)))

# bootstrap of the additive estimator too (for the additive anchor's CI)
set.seed(42)
boot_stat_add <- function(data, idx) std_ate(data[idx, , drop = FALSE], fmla_add)
bres_add <- boot(df, boot_stat_add, R = 2000)
se_add <- sd(bres_add$t)
ci_add <- as.numeric(quantile(bres_add$t, c(0.025, 0.975)))

rows <- list(
  list(module = MODULE, side = "R", statistic = "crude_diff",
       estimate = crude, se = NULL, n = n, published = 2.54,
       citation = "Hernán-Robins, What If §12.2 (crude)", extra = list()),
  list(module = MODULE, side = "R", statistic = "gformula_ate",
       estimate = ate_add, se = se_add, n = n, published = 3.5,
       citation = "Hernán-Robins, What If Program 13.3 (standardization)",
       extra = list(model = "additive (no qsmk:smokeintensity)",
                    ci = ci_add)),
  list(module = MODULE, side = "R", statistic = "gformula_interaction",
       estimate = ate_int, se = se_int, n = n, published = 3.5,
       citation = "Hernán-Robins, What If Program 13.3 (qsmk:smokeintensity)",
       extra = list(model = "wt82_71 ~ qsmk + qsmk:smokeintensity + conf",
                    ci = ci_int))
)
payload <- list(
  module = MODULE, side = "R", rows = rows,
  extra = list(engine = "base R lm + standardization + boot(R=2000)",
               ci_gformula_additive = ci_add,
               ci_gformula_interaction = ci_int,
               published_gformula_ci = c(2.6, 4.5))
)
writeLines(toJSON(payload, auto_unbox = TRUE, null = "null", digits = 10),
           file.path(RESULTS_DIR, paste0(MODULE, "_R.json")))
cat(sprintf(paste0("[%s] crude=%.4f (book 2.54)  g-formula additive=%.4f ",
                   "95%% CI (%.2f,%.2f)  interaction(book)=%.4f 95%% CI ",
                   "(%.2f,%.2f)  (book 3.5 [2.6,4.5])\n"),
            MODULE, crude, ate_add, ci_add[1], ci_add[2],
            ate_int, ci_int[1], ci_int[2]))
