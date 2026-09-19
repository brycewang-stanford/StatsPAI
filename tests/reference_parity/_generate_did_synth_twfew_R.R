#!/usr/bin/env Rscript
# Frozen R references for the TWFE-decomposition part of
# tests/reference_parity/test_did_synth_R_parity.py.
#
#   TwoWayFEWeights 2.1.0 (de Chaisemartin & D'Haultfoeuille's own package)
#     twowayfeweights(type = "feTR"): every cell weight, the TWFE beta, the
#     counts / sums of positive and negative weights, and both summary
#     measures (sensibility, sensibility2).
#   bacondecomp 0.1.1 bacon(): the Goodman-Bacon 2x2 estimates and weights,
#     which sp.twfe_decomposition's docstring says it reports.
#
# Datasets
#   mpdta   tests/r_parity/data/16_bjs.csv (the Track A mpdta bytes); one
#           observation per county x year cell, balanced.
#   grouped _fixtures/did_synth_twfew_panel.csv: several observations per
#           group x period cell, unbalanced, sampling weights, and a treatment
#           partly rolled out inside two cells (TwoWayFEWeights then replaces D
#           by its cell mean).
#
# Regenerate (after _fixtures/_generate_did_synth_data.py):
#   Rscript tests/reference_parity/_generate_did_synth_twfew_R.R

suppressPackageStartupMessages({
  library(TwoWayFEWeights)
  library(bacondecomp)
  library(jsonlite)
})

script_dir <- tryCatch(
  dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))),
  error = function(e) "tests/reference_parity"
)
fixtures <- file.path(script_dir, "_fixtures")

# TwoWayFEWeights residualises D on the two fixed effects with fixest's
# iterative demeaning, whose default convergence tolerance (fixef.tol = 1e-6)
# leaves ~1e-8 relative error in the cell weights of an unbalanced panel.
# Tightening the tolerance changes no convention -- the same algorithm is run
# to convergence -- and lets the comparison reach machine precision. The
# default-tolerance weights are ALSO written (suffix _default_tol) so the size
# of that solver error is on record.
fixest::setFixest_estimation(fixef.tol = 1e-11, fixef.iter = 100000)
repo <- normalizePath(file.path(script_dir, "..", ".."))

pack <- function(r) {
  cells <- as.data.frame(r$dat_result)
  cells <- cells[order(cells$G, cells$T), ]
  list(
    beta = r$beta, nr_plus = r$nr_plus, nr_minus = r$nr_minus,
    nr_weights = r$nr_weights, sum_plus = r$sum_plus, sum_minus = r$sum_minus,
    sensibility = r$sensibility,
    sensibility2 = if (is.null(r$sensibility2)) NA else r$sensibility2,
    tot_cells = r$tot_cells,
    cells = list(G = cells$G, T = cells$T, weight = cells$weight)
  )
}

mp <- read.csv(file.path(repo, "tests", "r_parity", "data", "16_bjs.csv"))
mp$D <- as.integer(mp$first_treat > 0 & mp$year >= mp$first_treat)
fe_mp <- twowayfeweights(mp, "lemp", "countyreal", "year", "D", type = "feTR")

gp <- read.csv(file.path(fixtures, "did_synth_twfew_panel.csv"))
fe_gp <- twowayfeweights(gp, "y", "g", "t", "d", type = "feTR")
fe_gp_w <- twowayfeweights(gp, "y", "g", "t", "d", type = "feTR", weights = gp$w)
fixest::setFixest_estimation(reset = TRUE)
fe_gp_default <- twowayfeweights(gp, "y", "g", "t", "d", type = "feTR")

bc <- bacon(lemp ~ D, data = mp, id_var = "countyreal", time_var = "year", quietly = TRUE)
bc <- bc[order(bc$type, bc$treated, bc$untreated), ]

res <- list(
  provenance = list(
    R = R.version.string,
    TwoWayFEWeights = as.character(packageVersion("TwoWayFEWeights")),
    bacondecomp = as.character(packageVersion("bacondecomp")),
    fixest = as.character(packageVersion("fixest")),
    jsonlite = as.character(packageVersion("jsonlite"))
  ),
  twfew_mpdta = pack(fe_mp),
  twfew_grouped = pack(fe_gp),
  twfew_grouped_weighted = pack(fe_gp_w),
  twfew_grouped_default_tol = pack(fe_gp_default),
  bacon_mpdta = list(
    type = as.character(bc$type), treated = bc$treated, untreated = bc$untreated,
    estimate = bc$estimate, weight = bc$weight,
    weighted_sum = sum(bc$estimate * bc$weight)
  )
)

writeLines(toJSON(res, auto_unbox = TRUE, digits = NA, na = "null"),
           file.path(fixtures, "did_synth_twfew_R.json"))
cat("wrote", file.path(fixtures, "did_synth_twfew_R.json"), "\n")
