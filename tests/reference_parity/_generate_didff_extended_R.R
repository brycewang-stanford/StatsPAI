# Reference values for the extended sp.functional_form_test surface against
# R `didFF` 0.1.0 (Roth & Sant'Anna 2023, Econometrica).
#
# Regenerate with:
#   Rscript tests/reference_parity/_generate_didff_extended_R.R
#
# Writes:
#   _fixtures/didff_extended_reference.json
#   _fixtures/didff_discrete_panel.csv    few distinct outcome values
#
# Covers what the original parity file does not: sampling weights, the
# automatic / discrete binning rule, binpoint padding, and the dynamic
# aggregation window (balance_e / min_e / max_e).

suppressMessages({
  library(didFF)
  library(jsonlite)
})

stopifnot(as.character(packageVersion("didFF")) == "0.1.0")

script_dir <- tryCatch(
  dirname(normalizePath(sys.frame(1)$ofile)),
  error = function(e) "tests/reference_parity"
)
fixtures <- file.path(script_dir, "_fixtures")

mpdta <- read.csv(file.path(fixtures, "mpdta_did_package.csv"))
gcol <- if ("first.treat" %in% names(mpdta)) "first.treat" else "first_treat"

# A deterministic, unit-constant sampling weight. Derived arithmetically from
# the county id rather than drawn, so R and Python read byte-identical weights
# and any difference in the result is the estimator's, not the fixture's.
mpdta$w <- 1 + ((mpdta$countyreal * 7919) %% 97) / 50
write.csv(mpdta, file.path(fixtures, "mpdta_did_package_weighted.csv"),
          row.names = FALSE)

out <- list()

pull <- function(res) {
  list(
    pval = res$pval,
    level = as.character(res$table$level),
    implied_density = res$table$implied_density
  )
}

# ------------------------------------------------------------- weights ----
# `w` is strictly positive and varies across counties, so it exercises the
# weighting path; the unweighted run is kept alongside to prove the weights
# actually move the answer.
out$weights <- list(
  nbins6 = pull(didFF(data = mpdta, yname = "lemp", tname = "year",
                      idname = "countyreal", gname = gcol,
                      weightsname = "w", nbins = 6, seed = 0)),
  nbins10 = pull(didFF(data = mpdta, yname = "lemp", tname = "year",
                       idname = "countyreal", gname = gcol,
                       weightsname = "w", nbins = 10, seed = 0)),
  unweighted6 = pull(didFF(data = mpdta, yname = "lemp", tname = "year",
                           idname = "countyreal", gname = gcol,
                           nbins = 6, seed = 0))
)

# -------------------------------------------------------- default bins ----
# No nbins and no binpoints: lemp has many distinct values, so the reference
# cuts into min(20, n_distinct) = 20 equal-width bins.
out$auto_default <- pull(didFF(data = mpdta, yname = "lemp", tname = "year",
                               idname = "countyreal", gname = gcol, seed = 0))

# ------------------------------------------------------------ discrete ----
# An outcome with a handful of distinct values: the reference switches to one
# bin per value and warns.
set.seed(20260811)
n_units <- 300
periods <- 1:5
cohorts <- c(3, 4, 0)  # 0 = never treated
unit_g <- sample(rep(cohorts, length.out = n_units))
discrete <- do.call(rbind, lapply(seq_len(n_units), function(u) {
  g <- unit_g[u]
  treated <- (g > 0) & (periods >= g)
  lambda <- exp(0.2 + 0.05 * periods + 0.3 * treated)
  data.frame(unit = u, time = periods, first_treat = g,
             y = rpois(length(periods), lambda))
}))
discrete$y <- pmin(discrete$y, 8)  # keep the support small and stable
write.csv(discrete, file.path(fixtures, "didff_discrete_panel.csv"),
          row.names = FALSE)

out$discrete <- list(
  auto = pull(didFF(data = discrete, yname = "y", tname = "time",
                    idname = "unit", gname = "first_treat", seed = 0)),
  # An explicit nbins always cuts, however few distinct values there are.
  forced_bins = pull(didFF(data = discrete, yname = "y", tname = "time",
                           idname = "unit", gname = "first_treat",
                           nbins = 4, seed = 0))
)
out$discrete_n_distinct <- length(unique(
  discrete$y[(discrete$time < discrete$first_treat) | (discrete$first_treat == 0)]
))

# ----------------------------------------------------------- binpoints ----
yr <- range(mpdta$lemp[(mpdta$year < mpdta[[gcol]]) | (mpdta[[gcol]] == 0)])
out$binpoints <- list(
  covering = pull(didFF(data = mpdta, yname = "lemp", tname = "year",
                        idname = "countyreal", gname = gcol,
                        binpoints = c(yr[1], 5, 7, 9, yr[2]), seed = 0)),
  # Stops short at both ends: the reference pads to the outcome range and warns.
  short = pull(didFF(data = mpdta, yname = "lemp", tname = "year",
                     idname = "countyreal", gname = gcol,
                     binpoints = c(5, 7, 9), seed = 0)),
  range = yr
)

# --------------------------------------------------------- aggregation ----
aggs <- list()
for (a in c("simple", "group", "calendar", "dynamic")) {
  aggs[[a]] <- pull(didFF(data = mpdta, yname = "lemp", tname = "year",
                          idname = "countyreal", gname = gcol,
                          nbins = 6, aggte_type = a, seed = 0))
}
out$aggregation <- aggs

# Dynamic aggregation with an event-time window.
out$dynamic_window <- list(
  min0 = pull(didFF(data = mpdta, yname = "lemp", tname = "year",
                    idname = "countyreal", gname = gcol, nbins = 6,
                    aggte_type = "dynamic", min_e = 0, seed = 0)),
  max1 = pull(didFF(data = mpdta, yname = "lemp", tname = "year",
                    idname = "countyreal", gname = gcol, nbins = 6,
                    aggte_type = "dynamic", min_e = 0, max_e = 1, seed = 0)),
  balanced1 = pull(didFF(data = mpdta, yname = "lemp", tname = "year",
                         idname = "countyreal", gname = gcol, nbins = 6,
                         aggte_type = "dynamic", balance_e = 1, seed = 0))
)

# ------------------------------------------------- rejection + weights ----
# Every mpdta p-value here is 1, so on its own this file would never exercise
# the critical value. The reject panel does, and running it weighted checks
# that weights reach the *test*, not just the point estimates.
reject <- read.csv(file.path(fixtures, "functional_form_reject_panel.csv"))
reject$w <- 1 + ((reject$id * 7919) %% 97) / 50
write.csv(reject, file.path(fixtures, "functional_form_reject_weighted.csv"),
          row.names = FALSE)

out$reject <- list(
  unweighted = pull(didFF(data = reject, yname = "y", tname = "t",
                          idname = "id", gname = "g", nbins = 6, seed = 0)),
  weighted = pull(didFF(data = reject, yname = "y", tname = "t",
                        idname = "id", gname = "g", weightsname = "w",
                        nbins = 6, seed = 0)),
  auto = pull(didFF(data = reject, yname = "y", tname = "t",
                    idname = "id", gname = "g", seed = 0))
)

# -------------------------------------------------------------- distDD ----
# distDD reports the treatment effect on P(Y in bin) instead of the implied
# counterfactual density: no sign flip, no zeroing of treated cells, bins over
# the whole panel, and no test.
pull_dist <- function(res) {
  list(
    level = as.character(res$table$level),
    estimates = res$table$test.estimates,
    se = res$table$test.se
  )
}

# NOTE: the automatic-binning case (no nbins) is deliberately absent.
# distDD bins over the whole panel, which on mpdta leaves one of the 20
# automatic bins empty; didFF 0.1.0 then builds its output table from
# `levels(droplevels(bins))` (19 entries) alongside `point_estimates` (20) and
# dies with "arguments imply differing number of rows: 20, 19". That is an
# upstream defect, not a divergence — StatsPAI reports all 20 bins with the
# empty one flagged `used = False`, which is what the parity test pins.
out$distdd <- list(
  mpdta6 = pull_dist(distDD(data = mpdta, yname = "lemp", tname = "year",
                            idname = "countyreal", gname = gcol, nbins = 6,
                            seed = 0)),
  mpdta_weighted = pull_dist(distDD(data = mpdta, yname = "lemp",
                                    tname = "year", idname = "countyreal",
                                    gname = gcol, weightsname = "w",
                                    nbins = 6, seed = 0)),
  discrete = pull_dist(distDD(data = discrete, yname = "y", tname = "time",
                              idname = "unit", gname = "first_treat",
                              seed = 0)),
  simple_agg = pull_dist(distDD(data = mpdta, yname = "lemp", tname = "year",
                                idname = "countyreal", gname = gcol,
                                nbins = 6, aggte_type = "simple", seed = 0))
)

# ------------------------------------------------------------- write -----
out$meta <- list(
  didff_version = as.character(packageVersion("didFF")),
  r_version = paste(R.version$major, R.version$minor, sep = "."),
  generated_by = "_generate_didff_extended_R.R"
)

write(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE),
      file.path(fixtures, "didff_extended_reference.json"))
cat("wrote", file.path(fixtures, "didff_extended_reference.json"), "\n")
