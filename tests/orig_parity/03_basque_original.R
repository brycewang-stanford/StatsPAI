# Original-data parity for Synth::basque (Abadie-Gardeazabal 2003).
# Published value: average post-1970 GDPpc gap = -0.855 (10$/capita).

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
  library(Synth)
  library(jsonlite)
})

data("basque", package = "Synth")

# Drop the nation-level "Spain" row (regionno 1) and keep only the
# 17 region-level rows. The Basque country is regionno 17.
basque_sub <- basque[basque$regionno != 1, ]

# Save canonical CSV (Python-friendly column names).
out <- basque_sub
names(out)[names(out) == "regionno"] <- "region_id"
names(out)[names(out) == "regionname"] <- "region"
names(out)[names(out) == "gdpcap"] <- "gdppc"
write.csv(out, file.path(DATA_DIR, "03_basque_original.csv"),
          row.names = FALSE)

# Synth::synth + Synth::dataprep on the canonical Abadie-Gardeazabal
# specification: pre-treatment 1955-1969, post-treatment 1970-1997,
# treated unit = Basque (regionno 17).
treated_id <- 17
controls <- setdiff(unique(basque_sub$regionno), treated_id)
pre_years <- 1955:1969
post_years <- 1970:1997

special_preds <- lapply(pre_years, function(yr) list("gdpcap", yr, "mean"))

dp <- Synth::dataprep(
  foo = basque_sub,
  predictors = NULL,
  predictors.op = "mean",
  dependent = "gdpcap",
  unit.variable = "regionno",
  time.variable = "year",
  special.predictors = special_preds,
  treatment.identifier = treated_id,
  controls.identifier = controls,
  time.predictors.prior = pre_years,
  time.optimize.ssr = pre_years,
  time.plot = c(pre_years, post_years),
  unit.names.variable = "regionname"
)

set.seed(42)
sy <- Synth::synth(data.prep.obj = dp, optimxmethod = "BFGS", verbose = FALSE)
Y0_synth <- dp$Y0plot %*% sy$solution.w
Y1_treat <- dp$Y1plot
gap <- Y1_treat - Y0_synth
post_idx <- which(rownames(Y1_treat) %in% as.character(post_years))
avg_post_gap <- mean(gap[post_idx])

build_row <- function(stat, est, se, n, published, citation) {
  list(module = jsonlite::unbox("03_basque_original"),
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
  build_row("avg_post_gap",
            avg_post_gap, NA_real_, nrow(basque_sub),
            -0.855,
            "Abadie-Gardeazabal (2003) Figure 2 / Synth vignette")
)
payload <- list(
  module = jsonlite::unbox("03_basque_original"),
  side = jsonlite::unbox("R"),
  rows = rows,
  extra = list(data_source = jsonlite::unbox("Synth::basque"),
               n_obs = jsonlite::unbox(nrow(basque_sub)))
)
writeLines(
  jsonlite::toJSON(payload, pretty = TRUE, na = "null", null = "null",
                    digits = NA),
  file.path(RESULTS_DIR, "03_basque_original_R.json")
)
message("OK -- wrote 03_basque_original_R.json")
