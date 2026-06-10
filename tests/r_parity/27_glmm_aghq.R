# StatsPAI GLMM AGHQ parity (R side) -- Module 27.
#
# Reads data/27_glmm_aghq.csv and runs lme4::glmer with nAGQ = 8.
# Tolerance: rel < 1e-6 on fixed-effect point estimates after tight
# optimiser controls; rel < 5e-2 on SE because fixed-effect covariance
# conventions differ across implementations.

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

MODULE <- "27_glmm_aghq"

df <- read_csv_strict(MODULE)
df$gid <- as.factor(df$gid)

aghq_control <- lme4::glmerControl(
  optimizer = "bobyqa",
  optCtrl = list(maxfun = 200000, rhobeg = 2e-3, rhoend = 1e-12)
)

fit <- lme4::glmer(y ~ x1 + (1 | gid), data = df,
                    family = binomial(link = "logit"),
                    nAGQ = 8L,
                    control = aghq_control)

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
                           nAGQ = 8L,
                           n_groups = length(unique(df$gid)),
                           optimizer = "bobyqa",
                           optimizer_maxfun = 200000L,
                           optimizer_rhobeg = 2e-3,
                           optimizer_rhoend = 1e-12))
