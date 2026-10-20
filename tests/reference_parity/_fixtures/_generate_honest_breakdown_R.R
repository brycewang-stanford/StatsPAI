# Reference sensitivity grids for sp.breakdown_m under HonestDiD's sign,
# shape and second-difference restrictions, from R HonestDiD 0.2.8
# (Rambachan & Roth 2023; MIT licence).
#
#   cd tests/reference_parity/_fixtures
#   Rscript _generate_honest_breakdown_R.R <delta> <M values, comma separated>
#   (one entry per run, merged into honest_breakdown_R.json; the runs that
#   make the fixture are RUNS in _generate_honest_breakdown_R.sh)
#
# HonestDiD has no breakdown function: the breakdown value is read off the
# sensitivity table as the M at which the robust confidence set first
# contains zero. Each run records createSensitivityResults(...) /
# createSensitivityResults_relativeMagnitudes(...) with method =
# "Conditional" (no simulation, so a correct port accepts the same grid
# points) over a grid of M that brackets the crossing. The test asserts
# that sp.breakdown_m lies between the last M at which HonestDiD's set
# excludes zero and the first at which it contains it.
#
# Input: did_synth_honest_es.csv's covariance (3 pre / 3 post, correlated),
# with a weaker post-period effect so that the crossing happens at a
# moderate M; target the on-impact effect (l_vec = c(1, 0, 0)).
suppressPackageStartupMessages({ library(HonestDiD); library(jsonlite) })

es <- read.csv("did_synth_honest_es.csv")
S6 <- unname(as.matrix(es[, paste0("s", 0:5)]))
b <- c(0.012, -0.018, 0.009, 0.075, 0.09, 0.11)
npre <- 3; npost <- 3; lv <- c(1, 0, 0)

argv <- commandArgs(trailingOnly = TRUE)
delta <- argv[[1]]
mvec <- as.numeric(strsplit(argv[[2]], ",")[[1]])

dir_args <- function(tail) {
  a <- list()
  if (tail == "PB") a$biasDirection <- "positive"
  if (tail == "NB") a$biasDirection <- "negative"
  if (tail == "I") a$monotonicityDirection <- "increasing"
  if (tail == "D") a$monotonicityDirection <- "decreasing"
  a
}

if (startsWith(delta, "SDRM") || startsWith(delta, "RM")) {
  tail <- sub("^SDRM|^RM", "", delta)
  args <- c(list(betahat = b, sigma = S6, numPrePeriods = npre,
                 numPostPeriods = npost, l_vec = lv, Mbarvec = mvec,
                 method = "Conditional", alpha = 0.05),
            dir_args(tail))
  if (startsWith(delta, "SDRM")) args$bound <- "deviation from linear trend"
  r <- suppressWarnings(do.call(createSensitivityResults_relativeMagnitudes, args))
  m_col <- r$Mbar
} else {
  tail <- sub("^SD", "", delta)
  args <- c(list(betahat = b, sigma = S6, numPrePeriods = npre,
                 numPostPeriods = npost, l_vec = lv, Mvec = mvec,
                 method = "Conditional", alpha = 0.05),
            dir_args(tail))
  r <- suppressWarnings(do.call(createSensitivityResults, args))
  m_col <- r$M
}

path <- "honest_breakdown_R.json"
out <- if (file.exists(path)) fromJSON(path, simplifyVector = TRUE) else list()
out[[delta]] <- list(delta = delta, method = "Conditional", M = m_col,
                     lb = r$lb, ub = r$ub)
out$`_inputs` <- list(betahat = b, sigma = S6, npre = npre, npost = npost,
                      l_vec = lv, event_times = c(-4, -3, -2, 0, 1, 2))
out$`_provenance` <- list(HonestDiD = as.character(packageVersion("HonestDiD")),
                          R = R.version.string)
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE, na = "null"), path)
