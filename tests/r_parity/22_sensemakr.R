# StatsPAI sensemakr parity (R side) -- Module 22.
#
# Reads data/22_sensemakr.csv (NSW-DW replica) and runs
# sensemakr::sensemakr with re74 as the benchmark covariate.
# Tolerance: rel < 1e-3 on RV_q and partial R^2.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(sensemakr)
})

MODULE <- "22_sensemakr"

df <- read_csv_strict(MODULE)

fit <- lm(re78 ~ treat + age + education + black + hispanic + married +
                  re74 + re75, data = df)

sm <- sensemakr::sensemakr(
  model = fit,
  treatment = "treat",
  benchmark_covariates = "re74",
  q = 1, alpha = 0.05
)

# Extract scalars
sens <- sm$sensitivity_stats
bench <- sm$bounds

beta_treat <- sens$estimate
se_treat   <- sens$se
t_treat    <- sens$t_statistic
partial_r2_yd <- sens$r2yd.x
rv_q       <- sens$rv_q
rv_q_alpha <- sens$rv_qa

# Benchmark partial R^2 with re74. sensemakr returns partial R^2
# of Y vs re74 (residualised) and D vs re74 (residualised).
# These are part of `bench` for q=1, alpha=0.05.
partial_r2_Y <- bench$r2dz.x[1]
partial_r2_D <- bench$r2yz.dx[1]

rows <- list(
  parity_row(MODULE, "beta_treat",
             estimate = beta_treat, se = se_treat, n = nrow(df)),
  parity_row(MODULE, "t_treat", estimate = t_treat, n = nrow(df)),
  parity_row(MODULE, "partial_r2_yd",
             estimate = partial_r2_yd, n = nrow(df)),
  parity_row(MODULE, "rv_q", estimate = rv_q, n = nrow(df)),
  parity_row(MODULE, "rv_q_alpha", estimate = rv_q_alpha, n = nrow(df)),
  parity_row(MODULE, "benchmark_re74_partial_r2_Y",
             estimate = partial_r2_Y, n = nrow(df)),
  parity_row(MODULE, "benchmark_re74_partial_r2_D",
             estimate = partial_r2_D, n = nrow(df))
)

write_results(MODULE, rows,
              extra = list(benchmark = "re74", alpha = 0.05, q = 1))
