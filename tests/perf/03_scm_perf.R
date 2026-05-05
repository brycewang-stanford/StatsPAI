# Track C performance -- Classical SCM (R side).
# Times Synth::synth at n_donors in {20, 50, 100}.

.script_dir <- (function() {
  args <- commandArgs(trailingOnly = FALSE)
  m <- grep("^--file=", args, value = TRUE)
  if (length(m) > 0) dirname(normalizePath(sub("^--file=", "", m[1])))
  else getwd()
})()
RESULTS_DIR <- file.path(.script_dir, "results")
dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages({
  library(Synth)
  library(jsonlite)
})

N_DONORS_LIST <- c(20L, 50L, 100L)
T_PERIODS <- 30L
N_REPS <- 2L  # Synth is slow at large donor counts

make_panel <- function(n_donors, t = T_PERIODS, seed = 42L) {
  set.seed(seed)
  n_units <- n_donors + 1L
  F_mat <- matrix(rnorm(t * 2L), nrow = t, ncol = 2L)
  Lambda <- matrix(rnorm(n_units * 2L), nrow = n_units, ncol = 2L)
  rows <- list()
  for (i in seq_len(n_units) - 1L) {
    for (s in seq_len(t)) {
      y_val <- sum(Lambda[i + 1L, ] * F_mat[s, ]) + rnorm(1L, 0, 0.3)
      rows[[length(rows) + 1L]] <- data.frame(
        unit_id = i, year = 1969L + s, y = y_val
      )
    }
  }
  do.call(rbind, rows)
}

time_one <- function(fn, n_reps, warmup = 1L) {
  for (i in seq_len(warmup)) suppressMessages(fn())
  t <- numeric(n_reps)
  for (i in seq_len(n_reps)) {
    gc()
    t0 <- Sys.time(); suppressMessages(fn())
    t[i] <- as.numeric(Sys.time() - t0, units = "secs")
  }
  list(median = median(t), iqr = IQR(t), min = min(t), max = max(t))
}

rows <- list()
for (n_donors in N_DONORS_LIST) {
  df <- make_panel(n_donors)
  # Synth requires positive integer unit IDs and a unit-names column.
  df$unit_id <- df$unit_id + 1L
  df$unit_name <- paste0("unit", df$unit_id)
  treated_id <- 1L
  controls <- setdiff(seq_len(n_donors + 1L), treated_id)
  pre_years <- 1970:1984
  post_years <- 1985:1999
  special_preds <- lapply(pre_years, function(yr) list("y", yr, "mean"))

  fn <- function() {
    dp <- Synth::dataprep(
      foo = df, predictors = NULL, predictors.op = "mean",
      dependent = "y", unit.variable = "unit_id",
      time.variable = "year",
      special.predictors = special_preds,
      treatment.identifier = treated_id,
      controls.identifier = controls,
      time.predictors.prior = pre_years,
      time.optimize.ssr = pre_years,
      time.plot = c(pre_years, post_years),
      unit.names.variable = "unit_name"
    )
    Synth::synth(data.prep.obj = dp, optimxmethod = "BFGS",
                  verbose = FALSE)
  }
  res <- time_one(fn, n_reps = N_REPS)
  rows[[length(rows) + 1L]] <- list(
    estimator = jsonlite::unbox("03_scm"),
    side = jsonlite::unbox("R"),
    n = jsonlite::unbox(as.integer(n_donors)),
    n_reps = jsonlite::unbox(N_REPS),
    median_time_s = jsonlite::unbox(res$median),
    iqr_time_s = jsonlite::unbox(res$iqr),
    min_time_s = jsonlite::unbox(res$min),
    max_time_s = jsonlite::unbox(res$max),
    peak_mem_mb = jsonlite::unbox(NA_real_),
    extra = list(n_donors = jsonlite::unbox(n_donors),
                 T = jsonlite::unbox(T_PERIODS))
  )
  cat(sprintf("  n_donors=%4d  T=%d  median=%.3fs\n", n_donors,
              T_PERIODS, res$median))
}

payload <- list(estimator = jsonlite::unbox("03_scm"),
                side = jsonlite::unbox("R"), rows = rows,
                hardware = list(R_version = jsonlite::unbox(R.version$version.string)),
                extra = list())
writeLines(
  jsonlite::toJSON(payload, pretty = TRUE, na = "null", null = "null", digits = NA),
  file.path(RESULTS_DIR, "03_scm_R.json")
)
message("OK -- wrote 03_scm_R.json")
