# StatsPAI binary probit parity (R side) -- Module 48.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) dirname(normalizePath(sub("^--file=", "", .file_arg[1]))) else getwd()
source(file.path(.script_dir, "_common.R"))

MODULE <- "48_probit"
df <- read_csv_strict(MODULE)
# IRLS run at epsilon = 1e-12 (as 57_logit) so every side sits on the same
# optimum; glm's default 1e-8 stops 3e-7 short of it on this fixture.
fit <- glm(y ~ x1 + x2, data = df, family = binomial(link = "probit"),
           control = glm.control(epsilon = 1e-12, maxit = 100))
co <- coef(fit); se <- sqrt(diag(vcov(fit)))

rows <- list(
  parity_row(MODULE, "beta_intercept", estimate = unname(co["(Intercept)"]), se = unname(se["(Intercept)"]), n = nrow(df)),
  parity_row(MODULE, "beta_x1",        estimate = unname(co["x1"]),          se = unname(se["x1"]),          n = nrow(df)),
  parity_row(MODULE, "beta_x2",        estimate = unname(co["x2"]),          se = unname(se["x2"]),          n = nrow(df))
)
# glm's SEs are the expected information. sampleSelection::probit maximises
# the same likelihood and reports the observed information (the convention
# of sp.probit and Stata probit), so it pins that convention on *_oim rows.
suppressPackageStartupMessages(library(sampleSelection))
fit_oim <- sampleSelection::probit(y ~ x1 + x2, data = df)
co_o <- coef(fit_oim); se_o <- sqrt(diag(vcov(fit_oim)))
for (nm in c("(Intercept)", "x1", "x2")) {
  lab <- if (nm == "(Intercept)") "intercept" else nm
  rows[[length(rows) + 1L]] <- parity_row(MODULE, paste0("beta_", lab, "_oim"),
    estimate = unname(co_o[nm]), se = unname(se_o[nm]), n = nrow(df))
}
write_results(MODULE, rows, extra = list(link = "probit", package = "stats::glm",
                                  package_oim = "sampleSelection::probit"))
