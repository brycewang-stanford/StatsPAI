# StatsPAI LMM parity (R side) -- Module 25.
#
# Reads data/25_lmm.csv and runs lme4::lmer with REML.
# Tolerance: rel < 1e-3 on the fixed effects and ICC.

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

MODULE <- "25_lmm"

df <- read_csv_strict(MODULE)
df$gid <- as.factor(df$gid)

fit <- lme4::lmer(y ~ x1 + (1 | gid), data = df, REML = TRUE)

co  <- fixef(fit)
ses <- sqrt(diag(vcov(fit)))
ll  <- as.numeric(logLik(fit))

# ICC from variance components.
vc <- as.data.frame(VarCorr(fit))
sigma2_group <- vc$vcov[vc$grp == "gid"]
sigma2_resid <- vc$vcov[vc$grp == "Residual"]
icc_val <- sigma2_group / (sigma2_group + sigma2_resid)

rows <- list(
  parity_row(MODULE, "beta_intercept",
             estimate = unname(co["(Intercept)"]),
             se = unname(ses["(Intercept)"]),
             n = nrow(df)),
  parity_row(MODULE, "beta_x1",
             estimate = unname(co["x1"]),
             se = unname(ses["x1"]),
             n = nrow(df)),
  parity_row(MODULE, "logLik", estimate = ll, n = nrow(df)),
  parity_row(MODULE, "icc", estimate = icc_val, n = nrow(df))
)

write_results(MODULE, rows,
              extra = list(method = "REML",
                           n_groups = length(unique(df$gid))))
