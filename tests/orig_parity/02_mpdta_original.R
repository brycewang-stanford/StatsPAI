# Original-data parity for did::mpdta (the Callaway-Sant'Anna 2021
# vignette panel). Published value: simple ATT = -0.0454 (SE 0.0113).

.script_dir <- (function() {
  args <- commandArgs(trailingOnly = FALSE)
  m <- grep("^--file=", args, value = TRUE)
  if (length(m) > 0) dirname(normalizePath(sub("^--file=", "", m[1])))
  else getwd()
})()
DATA_DIR <- file.path(.script_dir, "data")
RESULTS_DIR <- file.path(.script_dir, "results")
dir.create(DATA_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages({
  library(did)
  library(jsonlite)
})

data("mpdta", package = "did")

# Rename to the StatsPAI convention so the Python side can read it
# without a column-renaming step.
out <- mpdta
names(out)[names(out) == "first.treat"] <- "first_treat"
write.csv(out, file.path(DATA_DIR, "02_mpdta_original.csv"),
          row.names = FALSE)

# Run R-side did::att_gt + aggte simple ATT.
fit <- did::att_gt(
  yname = "lemp", tname = "year", idname = "countyreal",
  gname = "first.treat", data = mpdta,
  control_group = "nevertreated",
  est_method = "reg", bstrap = FALSE
)
agg <- did::aggte(fit, type = "simple", bstrap = FALSE, cband = FALSE)

build_row <- function(stat, est, se, n, published, citation) {
  list(module = jsonlite::unbox("02_mpdta_original"),
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
  build_row("simple_ATT",
            agg$overall.att, agg$overall.se,
            nrow(mpdta),
            -0.0454,
            "Callaway-Sant'Anna (2021), R 'did' vignette aggte simple")
)

payload <- list(
  module = jsonlite::unbox("02_mpdta_original"),
  side = jsonlite::unbox("R"),
  rows = rows,
  extra = list(data_source = jsonlite::unbox("did::mpdta"),
               n_obs = jsonlite::unbox(nrow(mpdta)))
)
writeLines(
  jsonlite::toJSON(payload, pretty = TRUE, na = "null", null = "null",
                    digits = NA),
  file.path(RESULTS_DIR, "02_mpdta_original_R.json")
)
message("OK -- wrote 02_mpdta_original_R.json")
