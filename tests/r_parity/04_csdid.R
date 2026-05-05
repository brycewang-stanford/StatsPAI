# StatsPAI CS-DiD parity (R side) -- Module 04.
#
# Reads data/04_csdid.csv (the StatsPAI mpdta replica) and runs
# did::att_gt + did::aggte(type="simple"). Tolerance: rel < 1e-3.
#
# Note: did::att_gt's default est_method = "dr" is doubly-robust; we
# use est_method = "reg" to match sp.callaway_santanna's outcome-
# regression doubly-robust variant on this DGP.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(did)
})

MODULE <- "04_csdid"

df <- read_csv_strict(MODULE)
# did::att_gt internally re-codes never-treated rows to first_treat
# = Inf and silently truncates if the column is integer. Coerce to
# numeric so the never-treated cohort is preserved.
df$first_treat <- as.numeric(df$first_treat)
df$lemp        <- as.numeric(df$lemp)
df$year        <- as.numeric(df$year)

set.seed(PARITY_SEED)
att_gt_fit <- did::att_gt(
  yname    = "lemp",
  tname    = "year",
  idname   = "countyreal",
  gname    = "first_treat",
  data     = df,
  control_group = "nevertreated",
  est_method    = "reg",
  bstrap        = FALSE,
  cband         = FALSE
)

agg_simple <- did::aggte(att_gt_fit, type = "simple",
                          bstrap = FALSE, cband = FALSE)

rows <- list(
  parity_row(
    module    = MODULE,
    statistic = "simple_ATT",
    estimate  = agg_simple$overall.att,
    se        = agg_simple$overall.se,
    ci_lo     = agg_simple$overall.att - qnorm(0.975) * agg_simple$overall.se,
    ci_hi     = agg_simple$overall.att + qnorm(0.975) * agg_simple$overall.se,
    n         = nrow(df)
  )
)

write_results(MODULE, rows,
              extra = list(estimator = "reg",
                           control_group = "nevertreated",
                           bstrap = FALSE))
