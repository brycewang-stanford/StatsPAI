# StatsPAI ordered logit parity (R side) -- Module 45.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) dirname(normalizePath(sub("^--file=", "", .file_arg[1]))) else getwd()
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({ library(MASS) })

MODULE <- "45_ologit"
df <- read_csv_strict(MODULE)
df$y <- factor(df$y, ordered = TRUE)

fit <- MASS::polr(
  y ~ x,
  data = df,
  method = "logistic",
  Hess = TRUE,
  control = list(maxit = 10000, reltol = 1e-14)
)
sm <- summary(fit)
co <- sm$coefficients

# MASS::polr stores betas and intercepts together: rows = c(x, 0|1, 1|2)
# Cut points: "0|1", "1|2" map to Stata's /cut1, /cut2.
beta_x  <- co["x", "Value"]
se_x    <- co["x", "Std. Error"]
cut1    <- co["0|1", "Value"]
se_cut1 <- co["0|1", "Std. Error"]
cut2    <- co["1|2", "Value"]
se_cut2 <- co["1|2", "Std. Error"]

rows <- list(
  parity_row(MODULE, "beta_x",    estimate = beta_x,  se = se_x,    n = nrow(df)),
  parity_row(MODULE, "beta_cut1", estimate = cut1,    se = se_cut1, n = nrow(df)),
  parity_row(MODULE, "beta_cut2", estimate = cut2,    se = se_cut2, n = nrow(df))
)

write_results(MODULE, rows, extra = list(package = "MASS::polr",
                                          method = "logistic",
                                          optim_reltol = 1e-14,
                                          optim_maxit = 10000))
