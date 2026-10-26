# StatsPAI negative binomial parity (R side) -- Module 42.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) dirname(normalizePath(sub("^--file=", "", .file_arg[1]))) else getwd()
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({ library(MASS) })

MODULE <- "42_nbreg"
df <- read_csv_strict(MODULE)

# Tight IRLS / theta alternation so all sides share one optimum.
fit <- MASS::glm.nb(y ~ x1 + x2, data = df,
                    control = glm.control(epsilon = 1e-12, maxit = 200))
co <- coef(fit); se <- sqrt(diag(vcov(fit)))

rows <- list(
  parity_row(MODULE, "beta_intercept", estimate = unname(co["(Intercept)"]), se = unname(se["(Intercept)"]), n = nrow(df)),
  parity_row(MODULE, "beta_x1",        estimate = unname(co["x1"]),          se = unname(se["x1"]),          n = nrow(df)),
  parity_row(MODULE, "beta_x2",        estimate = unname(co["x2"]),          se = unname(se["x2"]),          n = nrow(df))
)

# glm.nb's SEs hold theta at its estimate. The beta_*_condse rows carry the
# same numbers under the name of that convention, which sp.nbreg reports as
# model_info["se_conditional_on_dispersion"]; the beta_* rows compare it
# with sp.nbreg's joint observed information (Stata nbreg's default).
for (nm in c("(Intercept)", "x1", "x2")) {
  lab <- if (nm == "(Intercept)") "intercept" else nm
  rows[[length(rows) + 1L]] <- parity_row(MODULE, paste0("beta_", lab, "_condse"),
    estimate = unname(co[nm]), se = unname(se[nm]), n = nrow(df))
}

write_results(MODULE, rows, extra = list(package = "MASS::glm.nb"))
