# StatsPAI dCDH intertemporal event-study parity (R side) -- Module 78.
#
# Runs the authors' own DIDmultiplegtDYN::did_multiplegt_dyn (de Chaisemartin
# & D'Haultfoeuille 2024) on the panel dumped by 78_multiplegt_dyn.py.
#
# Only point estimates are emitted. The package reports analytical
# influence-function standard errors; sp.did_multiplegt_dyn has only a
# cluster bootstrap, so comparing SEs would compare two different variance
# estimators.
#
# Environment note: DIDmultiplegtDYN pulls in rgl (needs X11/GLU, absent on
# a headless mac) and requires the r-universe `polars` package. Both are
# handled here rather than left to fail cryptically:
#   options(rgl.useNULL = TRUE)
#   install.packages("polars", repos = "https://rpolars.r-universe.dev")
# polars 1.13.0 additionally needs rlang >= 1.2.0.
#
# Tolerance: rel < 1e-6.

options(rgl.useNULL = TRUE)

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(polars)
  library(DIDmultiplegtDYN)
})

MODULE <- "78_multiplegt_dyn"
N_EFFECTS <- 4
N_PLACEBOS <- 2

df <- as.data.frame(read_csv_strict(MODULE))

res <- did_multiplegt_dyn(
  df = df, outcome = "y", group = "id", time = "t", treatment = "d",
  effects = N_EFFECTS, placebo = N_PLACEBOS, cluster = "id", graph_off = TRUE
)

rows <- list()
# The package pads its row names ("Effect_1    "), so match on trimmed ones.
emit <- function(tbl, prefix, k) {
  nm <- sprintf("%s_%d", prefix, k)
  i <- match(nm, trimws(rownames(tbl)))
  if (is.na(i)) {
    stop(sprintf("%s not found; rows are %s", nm,
                 paste(trimws(rownames(tbl)), collapse = ", ")))
  }
  rows[[length(rows) + 1]] <<- parity_row(
    module    = MODULE,
    statistic = nm,
    estimate  = unname(tbl[i, "Estimate"]),
    se        = NA_real_,
    n         = as.integer(tbl[i, "Switchers"])
  )
}

for (k in seq_len(N_EFFECTS)) emit(res$results$Effects, "Effect", k)
for (k in seq_len(N_PLACEBOS)) emit(res$results$Placebos, "Placebo", k)

rows[[length(rows) + 1]] <- parity_row(
  module    = MODULE,
  statistic = "Av_tot_eff",
  estimate  = unname(
    res$results$ATE[match("Av_tot_eff", trimws(rownames(res$results$ATE))),
                    "Estimate"]
  ),
  se        = NA_real_,
  n         = nrow(df)
)

# --- switch-off design --------------------------------------------------
off <- as.data.frame(read_csv_strict(sprintf("%s_off", MODULE)))
res_off <- did_multiplegt_dyn(
  df = off, outcome = "y", group = "id", time = "t", treatment = "d",
  effects = 2, placebo = 1, cluster = "id", graph_off = TRUE
)
emit_off <- function(tbl, prefix, k) {
  nm <- sprintf("%s_%d", prefix, k)
  i <- match(nm, trimws(rownames(tbl)))
  if (is.na(i)) stop(sprintf("%s not found in the switch-off design", nm))
  rows[[length(rows) + 1]] <<- parity_row(
    module    = MODULE,
    statistic = sprintf("off_%s", nm),
    estimate  = unname(tbl[i, "Estimate"]),
    se        = NA_real_,
    n         = as.integer(tbl[i, "Switchers"])
  )
}
for (k in 1:2) emit_off(res_off$results$Effects, "Effect", k)
emit_off(res_off$results$Placebos, "Placebo", 1)

write_results(MODULE, rows, extra = list(
  reference = "DIDmultiplegtDYN::did_multiplegt_dyn",
  DIDmultiplegtDYN = as.character(utils::packageVersion("DIDmultiplegtDYN"))
))
