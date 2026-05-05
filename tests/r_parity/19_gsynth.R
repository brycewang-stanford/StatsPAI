# StatsPAI Generalized SCM parity (R side) -- Module 19.
#
# Reads data/19_gsynth.csv (Basque replica) and runs gsynth::gsynth
# (Xu 2017). Tolerance: rel < 0.20 on the post-treatment ATT.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(gsynth)
})

MODULE <- "19_gsynth"

df <- read_csv_strict(MODULE)

set.seed(PARITY_SEED)
fit <- gsynth::gsynth(
  formula = gdppc ~ treated_indicator,
  data    = df,
  index   = c("region", "year"),
  force   = "two-way",
  CV      = TRUE,
  r       = c(0, 5),
  se      = FALSE,
  inference = "parametric",
  nboots  = 50
)

# Extract aggregated post-treatment ATT.
att_avg <- fit$att.avg

# Selected number of factors:
n_factors <- fit$r.cv

# Pre-treatment RMSE: gsynth exposes it directly as fit$rmse, the
# pre-treatment root MSPE.
pre_rmse <- fit$rmse

rows <- list(
  parity_row(MODULE, "att_gsynth", estimate = att_avg, n = nrow(df)),
  parity_row(MODULE, "n_factors", estimate = n_factors, n = nrow(df)),
  parity_row(MODULE, "pre_rmse", estimate = pre_rmse, n = nrow(df))
)

write_results(MODULE, rows,
              extra = list(method = "gsynth::gsynth (Xu 2017)",
                           force = "two-way", CV = TRUE))
