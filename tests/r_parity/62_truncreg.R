# StatsPAI truncated regression parity (R side) -- Module 62.
#
# truncreg estimates sigma on the natural scale, matching the Stata
# truncreg /sigma row; the Python side delta-maps exp(ln_sigma).

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) dirname(normalizePath(sub("^--file=", "", .file_arg[1]))) else getwd()
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({ library(truncreg) })

MODULE <- "62_truncreg"
df <- read_csv_strict(MODULE)

# method='NR' (maxLik Newton-Raphson) converges past the default BFGS
# stopping point to the same likelihood optimum found by sp.truncreg
# and Stata truncreg (logLik gain ~5e-8 over the BFGS default).
fit <- truncreg(y ~ x1 + x2, data = df, point = 0, direction = "left",
                method = "NR", iterlim = 500)
co <- coef(fit); se <- sqrt(diag(vcov(fit)))

rows <- list(
  parity_row(MODULE, "beta_intercept", estimate = unname(co["(Intercept)"]), se = unname(se["(Intercept)"]), n = nrow(df)),
  parity_row(MODULE, "beta_x1",        estimate = unname(co["x1"]),          se = unname(se["x1"]),          n = nrow(df)),
  parity_row(MODULE, "beta_x2",        estimate = unname(co["x2"]),          se = unname(se["x2"]),          n = nrow(df)),
  parity_row(MODULE, "sigma",          estimate = unname(co["sigma"]),       se = unname(se["sigma"]),       n = nrow(df))
)
write_results(MODULE, rows,
              extra = list(package = "truncreg::truncreg",
                           truncation = "left at 0"))
