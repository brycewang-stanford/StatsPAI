# StatsPAI Augmented SCM parity (R side) -- Module 18.
#
# Reads data/18_augsynth.csv and runs augsynth::augsynth.
# Tolerance: rel < 0.20 on the post-treatment ATT (regularisation
# convention non-uniqueness, same family as Module 07 SCM).

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(augsynth)
})

MODULE <- "18_augsynth"

df <- read_csv_strict(MODULE)
df$treated_indicator <- as.integer(
  df$region == "Basque Country" & df$year >= 1970
)

# augsynth::augsynth API: outcome ~ treatment_indicator | aux_covariates
fit <- augsynth::augsynth(
  form = gdppc ~ treated_indicator,
  unit = region,
  time = year,
  data = df,
  progfunc = "Ridge",
  scm = TRUE
)

# Aggregate to the post-treatment ATT (average of effects).
sm <- summary(fit)
att_est <- sm$average_att$Estimate

# Pre-period RMSPE: compute from the synthetic counterfactual via
# predict(). predict(fit) returns the synthetic-control trajectory
# named by year. Compare against the treated unit's actual outcome.
synth_traj <- predict(fit)
yrs <- as.integer(names(synth_traj))
treated_y <- df$gdppc[df$region == "Basque Country"]
treated_yrs <- df$year[df$region == "Basque Country"]
treated_y_aligned <- treated_y[match(yrs, treated_yrs)]
pre_idx <- yrs < 1970
pre_residuals <- treated_y_aligned[pre_idx] - synth_traj[pre_idx]
pre_rmspe <- sqrt(mean(pre_residuals^2))

rows <- list(
  parity_row(MODULE, "att_augmented",
             estimate = att_est, n = nrow(df)),
  parity_row(MODULE, "pre_rmspe", estimate = pre_rmspe, n = nrow(df))
)

write_results(MODULE, rows,
              extra = list(method = "augsynth::augsynth",
                           progfunc = "Ridge"))
