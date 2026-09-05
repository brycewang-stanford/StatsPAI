# StatsPAI interflex parity (R side) -- Module 87.
#
# Reads data/87_interflex.csv (written by 87_interflex.py) and runs
# interflex::interflex three times on the same bytes:
#   linear  : estimator = "linear",  vartype = "delta", vcov.type = "robust",
#             neval = 5, Z = "Z1"
#   binning : estimator = "binning", cutoffs = c(0.3, 1.7), vartype = "delta"
#   kernel  : estimator = "kernel",  bw = 1, CI = FALSE, same X.eval
# Rows mirror the Python side (linear_me_<k>, linear_ate, binning_x0_<j>,
# binning_me_<j>, lkurtosis, p_wald, p_lr, kernel_me_<k>).
# Tolerance: rel_est 1e-6, rel_se 1e-6 (closed-form (W)LS on both sides).

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(interflex)
})

MODULE <- "87_interflex"
NEVAL <- 5L  # equally spaced over range(X), shared by all three sides
CUTOFFS <- c(0.3, 1.7)  # explicit bins shared by all three sides

df <- read_csv_strict(MODULE)
n <- nrow(df)
rows <- list()
add <- function(stat, est, se = NA, nobs = n) {
  rows[[length(rows) + 1L]] <<- parity_row(
    module = MODULE, statistic = stat, estimate = est, se = se, n = nobs
  )
}

lin <- interflex(estimator = "linear", Y = "Y", D = "D", X = "X", Z = "Z1",
                 data = df, neval = NEVAL, vartype = "delta",
                 vcov.type = "robust", CI = TRUE, figure = FALSE)
est.lin <- lin$est.lin[[1]]
for (k in seq_len(nrow(est.lin))) {
  add(paste0("linear_me_", k), unname(est.lin[k, "ME"]), unname(est.lin[k, "sd"]))
}
# interflex prints Avg.estimate rounded to 3 dp, so the ATE row is rebuilt
# from the same lm + HC1 covariance interflex uses (linear.R gen.ATE):
# TE_i = b_D + b_DX X_i averaged over treated units, delta-method SE.
dl <- df
dl$DX <- dl$D * dl$X
fit.lin <- lm(Y ~ X + D + DX + Z1, data = dl)
V.lin <- sandwich::vcovHC(fit.lin, type = "HC1")
xt <- dl$X[dl$D == 1]
ate <- unname(coef(fit.lin)["D"] + coef(fit.lin)["DX"] * mean(xt))
g <- c(1, mean(xt))
ate.se <- sqrt(as.numeric(t(g) %*% V.lin[c("D", "DX"), c("D", "DX")] %*% g))
stopifnot(abs(as.numeric(lin$Avg.estimate[[1]][1, "ATE"]) - round(ate, 3)) < 1e-9)
add("linear_ate", ate, ate.se)

bin <- interflex(estimator = "binning", Y = "Y", D = "D", X = "X", Z = "Z1",
                 data = df, cutoffs = CUTOFFS, vartype = "delta", vcov.type = "robust",
                 CI = TRUE, figure = FALSE)
est.bin <- bin$est.bin[[1]]
cuts.X <- sort(unique(c(min(df$X), CUTOFFS, max(df$X))))
counts <- as.integer(table(cut(df$X, breaks = cuts.X, include.lowest = TRUE)))
for (j in seq_len(nrow(est.bin))) {
  add(paste0("binning_x0_", j), unname(est.bin[j, "x0"]), nobs = counts[j])
  add(paste0("binning_me_", j), unname(est.bin[j, "coef"]), unname(est.bin[j, "sd"]),
      nobs = counts[j])
}
add("lkurtosis", Lmoments::Lmoments(df$X, returnobject = TRUE)$ratios[4])
# interflex reports the rounded (3 dp) p-values in $tests; recompute the
# unrounded Wald and LR statistics with the same fitted models so the row
# is a full-precision comparison. This is interflex's own construction
# (binning.R: lmtest::waldtest / lrtest of the fully interacted model
# against the linear model with the HC1 covariance).
groupX <- cut(df$X, breaks = cuts.X, labels = FALSE)
groupX[which(df$X == min(df$X))] <- 1
dw <- df
dw$DX <- dw$D * dw$X
f0 <- "Y ~ X + D + DX + Z1"
f1 <- f0
for (i in 2:3) {
  dw[, paste0("G.", i)] <- as.numeric(groupX == i)
  dw[, paste0("G.", i, ".X")] <- as.numeric(groupX == i) * dw$X
  dw[, paste0("DG.", i)] <- as.numeric(groupX == i) * dw$D
  dw[, paste0("DG.", i, ".X")] <- as.numeric(groupX == i) * dw$D * dw$X
  dw[, paste0("Z.Z1.G.", i)] <- as.numeric(groupX == i) * dw$Z1
  dw[, paste0("ZX.Z1.G.", i)] <- as.numeric(groupX == i) * dw$Z1 * dw$X
  f1 <- paste0(f1, " + G.", i, " + G.", i, ".X + DG.", i, " + DG.", i, ".X",
               " + Z.Z1.G.", i, " + ZX.Z1.G.", i)
}
fit0 <- lm(as.formula(f0), data = dw)
fit1 <- lm(as.formula(f1), data = dw)
V1 <- sandwich::vcovHC(fit1, type = "HC1")
add("p_wald", lmtest::waldtest(fit1, fit0, test = "Chisq", vcov = V1)[[4]][2])
add("p_lr", lmtest::lrtest(fit1, fit0)[[5]][2])
stopifnot(abs(as.numeric(bin$tests$p.wald) - round(rows[[length(rows) - 1L]]$estimate, 3)) < 1e-9)
stopifnot(abs(as.numeric(bin$tests$p.lr) - round(rows[[length(rows)]]$estimate, 3)) < 1e-9)

ker <- interflex(estimator = "kernel", Y = "Y", D = "D", X = "X", Z = "Z1",
                 data = df, bw = 1, neval = NEVAL, CI = FALSE, figure = FALSE)
est.ker <- ker$est.kernel[[1]]
for (k in seq_len(nrow(est.ker))) {
  add(paste0("kernel_me_", k), unname(est.ker[k, "TE"]))
}

write_results(MODULE, rows, extra = list(
  neval = 5, cutoffs = CUTOFFS, bw_kernel = 1,
  interflex_version = as.character(packageVersion("interflex")),
  p_value_note = "p_wald / p_lr are the unrounded statistics of interflex's own Wald/LR construction; $tests reports them rounded to 3 dp and is asserted equal after rounding"
))
