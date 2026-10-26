# StatsPAI Newey-West HAC parity (R side) -- Module 51.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) dirname(normalizePath(sub("^--file=", "", .file_arg[1]))) else getwd()
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({ library(sandwich); library(lmtest) })

MODULE <- "51_newey"
df <- read_csv_strict(MODULE)
fit <- lm(y ~ x, data = df)
# NeweyWest with lag=4, no prewhitening. adjust = FALSE is sandwich's
# default; the adjust = TRUE rows below carry the N/(N-K) factor Stata's
# `newey` applies.
V <- NeweyWest(fit, lag = 4, prewhite = FALSE, adjust = FALSE)
co <- coef(fit); se <- sqrt(diag(V))

rows <- list(
  parity_row(MODULE, "beta_intercept", estimate = unname(co["(Intercept)"]), se = unname(se["(Intercept)"]), n = nrow(df)),
  parity_row(MODULE, "beta_x",         estimate = unname(co["x"]),           se = unname(se["x"]),           n = nrow(df))
)
V_small <- NeweyWest(fit, lag = 4, prewhite = FALSE, adjust = TRUE)
se_small <- sqrt(diag(V_small))
rows[[length(rows) + 1L]] <- parity_row(MODULE, "beta_intercept_small", estimate = unname(co["(Intercept)"]), se = unname(se_small["(Intercept)"]), n = nrow(df))
rows[[length(rows) + 1L]] <- parity_row(MODULE, "beta_x_small",         estimate = unname(co["x"]),           se = unname(se_small["x"]),           n = nrow(df))
write_results(MODULE, rows, extra = list(vcov = "NeweyWest", lag = 4, package = "sandwich::NeweyWest"))
