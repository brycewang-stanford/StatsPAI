# StatsPAI Changes-in-Changes parity (R side) -- Module 74.
#
# Reads data/74_cic.csv and runs qte::CiC, the R implementation of the
# Athey-Imbens (2006) estimator. Tolerance: rel < 1e-6 on the ATT and on
# each quantile treatment effect.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(qte)
})

MODULE <- "74_cic"

df <- read_csv_strict(MODULE)
probs <- seq(0.1, 0.9, 0.1)

fit <- qte::CiC(
  y ~ treat, t = 2, tmin1 = 1, tname = "t", data = df,
  panel = TRUE, idname = "id", se = FALSE, probs = probs
)

rows <- list(
  parity_row(
    module    = MODULE,
    statistic = "cic_ATT",
    estimate  = as.numeric(fit$ate),
    se        = NA_real_,
    n         = nrow(df)
  )
)
for (k in seq_along(probs)) {
  rows[[length(rows) + 1]] <- parity_row(
    module    = MODULE,
    statistic = sprintf("qte_%02d", round(probs[k] * 100)),
    estimate  = as.numeric(fit$qte[k]),
    se        = NA_real_,
    n         = nrow(df)
  )
}

write_results(MODULE, rows, extra = list(
  package = "qte",
  version = as.character(utils::packageVersion("qte"))
))
