# Reference values for sp.lmtp vs R lmtp 1.5.4 (AGPL: used as a black box,
# outputs only); checked in tests/reference_parity/test_lmtp_parity.py.
# Run from this folder:  Rscript _generate_lmtp_R.R
# SL.glm learners, folds = 1, default lmtp_control() (trim 0.999, bound
# 1e-5); policy A + 0.5 at every time point.
suppressMessages({library(lmtp); library(jsonlite)})
set.seed(56); n <- 500
W <- round(rnorm(n), 6); A <- round(1 + 0.5*W + rnorm(n), 6)
Y <- round(1 + A + 0.5*W + rnorm(n), 6); Yb <- as.integer(Y > 2)
write.csv(data.frame(W, A, Y, Yb), "lmtp_point.csv", row.names = FALSE)
set.seed(57)
W <- round(rnorm(n), 6); A1 <- round(1 + 0.5*W + rnorm(n), 6)
L2 <- round(0.4*A1 + 0.3*W + rnorm(n), 6); A2 <- round(0.5*A1 + 0.4*L2 + rnorm(n), 6)
Y <- round(1 + 0.5*A1 + A2 + 0.5*L2 + rnorm(n), 6); Yb <- as.integer(Y > 2.5)
write.csv(data.frame(W, A1, L2, A2, Y, Yb), "lmtp_long.csv", row.names = FALSE)
sh <- function(data, trt) data[[trt]] + 0.5
p <- read.csv("lmtp_point.csv"); l <- read.csv("lmtp_long.csv")
fit <- function(d, trt, y, tv, type) {
  r <- lmtp_tmle(d, trt, y, "W", time_vary = tv, shift = sh, outcome_type = type,
                 folds = 1, learners_outcome = "SL.glm", learners_trt = "SL.glm")
  list(est = r$estimate@x, se = r$estimate@std_error, ratios = r$density_ratios)
}
out <- list(
  point_continuous = fit(p, "A", "Y", NULL, "continuous"),
  point_binary = fit(p, "A", "Yb", NULL, "binomial"),
  long_continuous = fit(l, c("A1", "A2"), "Y", list(NULL, "L2"), "continuous"),
  long_binary = fit(l, c("A1", "A2"), "Yb", list(NULL, "L2"), "binomial"),
  versions = list(R = R.version.string, lmtp = as.character(packageVersion("lmtp"))))
writeLines(toJSON(out, auto_unbox = TRUE, digits = I(17)), "lmtp_R.json")
