# Classical SCM unique-solution parity (R side) -- Module 52.
#
# Reads data/52_scm_unique.csv (unique convex-hull SCM DGP written by
# the Python side) and runs Synth::synth with each pre-period as its
# own predictor, so the donor-weight problem is the identified convex
# programme.  Reports the average post-treatment gap and the recovered
# donor weights.  See 52_scm_unique.py for the DGP and rationale.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(Synth)
})

MODULE <- "52_scm_unique"

df <- read_csv_strict(MODULE)
df$id <- as.integer(factor(df$region,
                           levels = c(paste0("donor", 0:4), "treated")))
units <- unique(df[, c("region", "id")])
ctrl <- units$id[units$region != "treated"]
tr   <- units$id[units$region == "treated"]

# Each pre-period (0..19) as its own special predictor -> match the
# full pre-treatment path, i.e. the identified convex programme.
sp_list <- lapply(0:19, function(p) list("y", p, "mean"))

dp <- dataprep(
  foo = df, predictors = NULL, dependent = "y",
  unit.variable = "id", time.variable = "year",
  treatment.identifier = tr, controls.identifier = ctrl,
  special.predictors = sp_list,
  time.predictors.prior = 0:19, time.optimize.ssr = 0:19,
  time.plot = 0:29, unit.names.variable = "region"
)
out <- synth(dp, verbose = FALSE,
             custom.v = rep(1, length(sp_list)),
             Sigf.ipop = 20,
             Margin.ipop = 1e-12,
             Bound.ipop = 10)

w <- as.numeric(out$solution.w)
synth_path <- dp$Y0plot %*% out$solution.w
gap_post <- mean(dp$Y1plot[21:30] - synth_path[21:30])
pre_rmse <- sqrt(mean((dp$Y1plot[1:20] - synth_path[1:20])^2))

rows <- c(
  list(parity_row(module = MODULE, statistic = "avg_post_gap",
                  estimate = gap_post, n = nrow(df))),
  list(parity_row(module = MODULE, statistic = "pre_treatment_rmse",
                  estimate = pre_rmse, n = nrow(df)))
)
donor_names <- units$region[match(ctrl, units$id)]
for (k in seq_along(ctrl)) {
  rows <- c(rows, list(parity_row(
    module = MODULE, statistic = paste0("weight_", donor_names[k]),
    estimate = w[k], n = nrow(df))))
}

write_results(MODULE, rows,
              extra = list(method = "classic (per-period predictors)",
                           true_gap = 2.0,
                           custom.v = "uniform fixed predictor weights",
                           Sigf.ipop = 20,
                           Margin.ipop = 1e-12,
                           Bound.ipop = 10,
                           optimizer_note = paste(
                             "Fixed custom.v removes the unnecessary",
                             "outer V-optimization on this exact",
                             "convex-hull fixture; tightened ipop",
                             "controls make the headline gap a",
                             "machine-level point-estimate match.",
                             "Tiny non-zero distractor weights remain",
                             "diagnostic rows.")))
