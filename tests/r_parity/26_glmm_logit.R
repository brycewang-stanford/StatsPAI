# StatsPAI GLMM logit parity (R side) -- Module 26.
#
# Reads data/26_glmm_logit.csv and runs lme4::glmer with family =
# binomial(link = "logit"), Laplace approximation (nAGQ = 1).
# Tolerance: rel < 1e-2.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(lme4)
})

MODULE <- "26_glmm_logit"

df <- read_csv_strict(MODULE)
df$gid <- as.factor(df$gid)

fit <- lme4::glmer(y ~ x1 + (1 | gid), data = df,
                    family = binomial(link = "logit"),
                    nAGQ = 1L)

co  <- fixef(fit)
ses <- sqrt(diag(vcov(fit)))
ll  <- as.numeric(logLik(fit))

rows <- list(
  parity_row(MODULE, "beta_intercept",
             estimate = unname(co["(Intercept)"]),
             se = unname(ses["(Intercept)"]),
             n = nrow(df)),
  parity_row(MODULE, "beta_x1",
             estimate = unname(co["x1"]),
             se = unname(ses["x1"]),
             n = nrow(df)),
  parity_row(MODULE, "logLik", estimate = ll, n = nrow(df))
)

write_results(MODULE, rows,
              extra = list(family = "binomial(logit)",
                           nAGQ = 1L,
                           n_groups = length(unique(df$gid))))
