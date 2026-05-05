# StatsPAI Synthetic DID parity (R side) -- Module 12.
#
# Reads data/12_sdid.csv (the StatsPAI California-Prop99 replica) and
# runs synthdid::synthdid_estimate. Tolerance: rel < 1e-3 on the
# post-treatment ATT (placebo SE typically tracks the StatsPAI
# bootstrap SE within Monte Carlo error).

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
    se        = se,
    ci_lo     = as.numeric(tau_hat) - qnorm(0.975) * se,
    ci_hi     = as.numeric(tau_hat) + qnorm(0.975) * se,
    n         = nrow(df)
  )
)

write_results(MODULE, rows,
              extra = list(method = "synthdid_estimate",
                           N0 = panel$N0,
                           T0 = panel$T0,
                           se_method = "placebo"))
