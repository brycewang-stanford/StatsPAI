# Original-data parity for Card (1995):
#   Source data : wooldridge::card  (the original NLSYM extract)
#   Published   : Card (1995) Table 2 col 2 (OLS) = 0.075;
#                            col 5 (IV nearc4)   = 0.132.
#
# This script:
#   1. Dumps the wooldridge::card data to data/01_card_original.csv
#      so the Python side reads identical bytes.
#   2. Runs lm() + AER::ivreg with the same specification.
#   3. Writes results/01_card_original_R.json with both R numbers
#      and the published numbers from the paper.

.script_dir <- (function() {
  args <- commandArgs(trailingOnly = FALSE)
  m <- grep("^--file=", args, value = TRUE)
  if (length(m) > 0) dirname(normalizePath(sub("^--file=", "", m[1])))
  else getwd()
})()
HERE <- .script_dir
DATA_DIR <- file.path(HERE, "data")
RESULTS_DIR <- file.path(HERE, "results")
dir.create(DATA_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages({
  library(wooldridge)
  library(AER)
  library(jsonlite)
})

data("card", package = "wooldridge")

# Drop NAs in the modelling subset to match the canonical specification.
keep_vars <- c("lwage", "educ", "exper", "expersq",
               "black", "south", "smsa", "nearc4", "nearc2")
card_sub <- card[, keep_vars]
card_sub <- card_sub[complete.cases(card_sub), ]

write.csv(card_sub, file.path(DATA_DIR, "01_card_original.csv"),
          row.names = FALSE)

# OLS with HC1 robust SE (matching the StatsPAI default + the
# Wooldridge textbook reproduction).
ols <- lm(lwage ~ educ + exper + expersq + black + south + smsa,
          data = card_sub)
ols_se <- sqrt(diag(sandwich::vcovHC(ols, type = "HC1")))

# IV with nearc4 as instrument for educ.
iv <- AER::ivreg(
  lwage ~ exper + expersq + black + south + smsa + educ |
          exper + expersq + black + south + smsa + nearc4,
  data = card_sub
)
iv_se <- sqrt(diag(sandwich::vcovHC(iv, type = "HC1")))

build_row <- function(stat, est, se, n, published, citation) {
  list(module = jsonlite::unbox("01_card_original"),
       side = jsonlite::unbox("R"),
       statistic = jsonlite::unbox(stat),
       estimate = jsonlite::unbox(est),
       se = jsonlite::unbox(se),
       n = jsonlite::unbox(n),
       published = jsonlite::unbox(published),
       citation = jsonlite::unbox(citation),
       extra = list())
}

rows <- list(
  build_row("ols_beta_educ",
            unname(coef(ols)["educ"]),
            unname(ols_se["educ"]),
            nrow(card_sub),
            0.075,
            "Card (1995) Table 2, col 2"),
  build_row("iv_beta_educ_nearc4",
            unname(coef(iv)["educ"]),
            unname(iv_se["educ"]),
            nrow(card_sub),
            0.132,
            "Card (1995) Table 2, col 5")
)

payload <- list(
  module = jsonlite::unbox("01_card_original"),
  side = jsonlite::unbox("R"),
  rows = rows,
  extra = list(data_source = jsonlite::unbox("wooldridge::card"),
               n_obs = jsonlite::unbox(nrow(card_sub)))
)

writeLines(
  jsonlite::toJSON(payload, pretty = TRUE, na = "null", null = "null",
                    digits = NA),
  file.path(RESULTS_DIR, "01_card_original_R.json")
)
message("OK -- wrote 01_card_original_R.json")
