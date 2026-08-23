# StatsPAI BJS pre-trend parity (R side) -- Module 84.
#
# Runs didimputation::did_imputation (Butts & Borusyak, CRAN) on the same
# CSV bytes as the Python side.
#
# This side pins the POST-treatment horizons only, and that boundary is a
# finding rather than a shortcut. R and Stata implement different
# pre-trend normalisations and neither package exposes the other's:
#
#   * Stata did_imputation, pretrends(k) estimates exactly k lead
#     indicators and pools every earlier relative time into the omitted
#     category, so the leads are differenced against that pooled block.
#   * R didimputation takes `pretrends` as a flag, not a count. With
#     pretrends = TRUE it estimates a lead for every pre-period and omits
#     relative time -1, so the leads are differenced against -1.
#
# The two therefore fit different designs, and the difference is not a
# vertical shift: on this fixture the R leads and the Stata leads do not
# even have the same spacing, because changing the number of lead dummies
# changes what the omitted block contains. Passing pretrends = 3L to R
# returns an NA row and a coercion warning rather than Stata's object.
#
# sp.did_imputation(pretrend_method="bjs") implements the Stata
# convention, which is the one its docstring names, so the lead vector is
# pinned against Stata in 84_bjs_pretrends.do and the R side stays on the
# horizons where all three implementations answer the same question.
#
# Tolerance: rel_est < 1e-6. Standard errors are deliberately NOT within
# the tolerance budget -- see the note in compare.py::TOLERANCES.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(didimputation)
  library(data.table)
})

MODULE <- "84_bjs_pretrends"
HORIZONS <- c(0, 1, 2, 3)

df <- read_csv_strict(MODULE)
setDT(df)
# didimputation wants Inf for never-treated; the shared CSV encodes 0.
df[, g_inf := ifelse(g == 0, Inf, g)]

fit <- didimputation::did_imputation(
  data = df, yname = "y", gname = "g_inf", tname = "time", idname = "unit",
  horizon = HORIZONS, pretrends = FALSE, cluster_var = "unit"
)
fit <- as.data.frame(fit)
fit$rel <- suppressWarnings(as.numeric(as.character(fit$term)))

rows <- list()
for (h in HORIZONS) {
  r <- fit[!is.na(fit$rel) & fit$rel == h, ]
  if (nrow(r) != 1L) stop(sprintf("did_imputation returned no horizon %d", h))
  rows[[length(rows) + 1L]] <- parity_row(
    module    = MODULE,
    statistic = sprintf("tau%d_att", h),
    estimate  = as.numeric(r$estimate[1]),
    se        = as.numeric(r$std.error[1])
  )
}

write_results(MODULE, rows,
              extra = list(
                reference = "didimputation::did_imputation",
                cluster = "unit",
                horizons = HORIZONS,
                pretrends_pinned_here = FALSE,
                pretrend_convention_note = paste(
                  "R takes pretrends as a flag and omits relative time -1;",
                  "Stata takes a count and pools earlier periods. Different",
                  "designs, not a shift. Leads are pinned against Stata."
                )))
