# StatsPAI Synthetic DID parity (R side) -- Module 12.
#
# Reads data/12_sdid.csv (the StatsPAI California-Prop99 replica) and
# runs synthdid::synthdid_estimate. The headline ATT row is point-only;
# synthdid's native placebo SE is emitted as an explicitly named
# diagnostic row.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(synthdid)
})

MODULE <- "12_sdid"

df <- read_csv_strict(MODULE)

# synthdid wants a panel matrix; build it via panel.matrices().
panel <- synthdid::panel.matrices(
  panel = df,
  unit = "state",
  time = "year",
  outcome = "cigsale",
  treatment = "treated"
)

set.seed(PARITY_SEED)
tau_hat <- synthdid::synthdid_estimate(
  Y = panel$Y, N0 = panel$N0, T0 = panel$T0
)

# Placebo SE: synthdid_se with method = "placebo".
se <- synthdid::synthdid_se(tau_hat, method = "placebo")

rows <- list(
  parity_row(
    module    = MODULE,
    statistic = "att_sdid",
    estimate  = as.numeric(tau_hat),
    se        = NA,
    ci_lo     = NA,
    ci_hi     = NA,
    n         = nrow(df)
  ),
  parity_row(
    module    = MODULE,
    statistic = "se_synthdid_placebo",
    estimate  = se,
    n         = nrow(df)
  )
)

write_results(MODULE, rows,
              extra = list(method = "synthdid_estimate",
                           N0 = panel$N0,
                           T0 = panel$T0,
                           se_method = "placebo",
                           se_reference = paste(
                             "R records synthdid_se placebo SEs as a",
                             "diagnostic row; att_sdid is point-only."
                           )))
