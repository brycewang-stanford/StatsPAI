# StatsPAI Sun-Abraham parity (R side) -- Module 05.
#
# Reads data/05_sunab.csv (the StatsPAI mpdta replica) and runs
# fixest::feols(lemp ~ sunab(first_treat, year) | countyreal + year,
#               cluster = ~countyreal).
#
# Rows emitted (all taken from fixest's own aggregation, no by-hand
# re-weighting):
#   weighted_avg_ATT      summary(fit, agg = "att"): cohort-size weighted
#                         post-treatment ATT and its clustered SE.
#   att_rel_<e>           coeftable(fit): fixest's default per-relative-
#                         time aggregation (cohort shares treated as fixed
#                         in the variance).
#   att_rel_<e>_fixedshare identical numbers under a second name, so the
#                         Python side can pin its share_variance=False
#                         path (fixest convention) at machine level while
#                         its default share_variance=True path (Sun &
#                         Abraham 2021 Prop. 3 / Stata eventstudyinteract)
#                         is compared to the att_rel_<e> rows and differs
#                         from fixest by exactly the documented cohort-share
#                         term at multi-cohort relative times.
#
# Tolerance: rel_est 1e-6 (machine level); rel_se budget registered in
# compare.py::TOLERANCES with the share-term mechanism.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(fixest)
})

MODULE <- "05_sunab"

df <- read_csv_strict(MODULE)
df$first_treat <- as.numeric(df$first_treat)

fit <- fixest::feols(
  lemp ~ sunab(first_treat, year) | countyreal + year,
  data = df,
  cluster = ~ countyreal
)

# Aggregated post-treatment ATT (cohort-size weighted) with its SE.
agg_tab <- fixest::coeftable(summary(fit, agg = "att"))
att_row <- which(rownames(agg_tab) == "ATT")
if (length(att_row) != 1L) att_row <- 1L

rows <- list(
  parity_row(
    module    = MODULE,
    statistic = "weighted_avg_ATT",
    estimate  = unname(agg_tab[att_row, "Estimate"]),
    se        = unname(agg_tab[att_row, "Std. Error"]),
    n         = nrow(df)
  )
)

# Per-relative-time coefficients: fixest's native sunab aggregation.
es_tab <- fixest::coeftable(fit)
rel <- as.integer(sub("^year::(-?\\d+)$", "\\1", rownames(es_tab)))
keep <- !is.na(rel)
es_tab <- es_tab[keep, , drop = FALSE]
rel <- rel[keep]

for (j in order(rel)) {
  est <- unname(es_tab[j, "Estimate"])
  se  <- unname(es_tab[j, "Std. Error"])
  for (suffix in c("", "_fixedshare")) {
    rows[[length(rows) + 1L]] <- parity_row(
      module    = MODULE,
      statistic = paste0("att_rel_", rel[j], suffix),
      estimate  = est,
      se        = se,
      n         = nrow(df)
    )
  }
}

write_results(
  MODULE, rows,
  extra = list(
    cluster = "countyreal",
    ssc = "fixest default: adj=TRUE, fixef.K='nested', cluster.adj=TRUE",
    variance_convention = paste(
      "fixest::sunab treats the cohort shares as fixed when aggregating",
      "cohort-by-relative-time coefficients; Sun & Abraham (2021, Prop. 3)",
      "and Stata eventstudyinteract add the share-estimation term."
    )
  )
)
