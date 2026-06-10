#!/usr/bin/env Rscript
# R reference values for sp.kaplan_meier / sp.logrank_test parity.
#
#   * sp.kaplan_meier -> survival::survfit (product-limit estimator):
#       per-group survival at fixed query times + median survival.
#   * sp.logrank_test -> survival::survdiff (Mantel-Haenszel log-rank):
#       chi-square statistic, df, p-value, observed/expected events.
#
# The product-limit estimator and the Mantel-Haenszel statistic are both
# unambiguous on a tie-free design, so sp is expected to reproduce R to
# numerical precision.
#
# Run from the repository root:
#   Rscript tests/reference_parity/_fixtures/_generate_survival_km.R
suppressMessages({
  library(survival)
  library(jsonlite)
})

df <- read.csv("tests/reference_parity/_fixtures/survival_km_data.csv")

# log-rank on the full sample
sd <- survdiff(Surv(time, status) ~ group, data = df)
df_lr <- length(sd$n) - 1
logrank <- list(
  chisq = unname(sd$chisq),
  df = df_lr,
  p_value = unname(1 - pchisq(sd$chisq, df_lr)),
  observed = unname(sd$obs),
  expected = unname(sd$exp)
)

# per-group KM survival at fixed query times + median
qt <- c(2, 4, 6, 8, 10, 12)
km <- list()
for (gv in c(0, 1)) {
  sub <- df[df$group == gv, ]
  f <- survfit(Surv(time, status) ~ 1, data = sub)
  s <- summary(f, times = qt, extend = TRUE)
  med <- unname(quantile(f, probs = 0.5)$quantile)
  km[[as.character(gv)]] <- list(
    times = qt,
    surv = unname(s$surv),
    median = med
  )
}

out <- list(logrank = logrank, km = km, n = nrow(df))
writeLines(
  toJSON(out, auto_unbox = TRUE, digits = 15),
  "tests/reference_parity/_fixtures/survival_km_R.json"
)
cat("wrote survival_km_R.json\n")
