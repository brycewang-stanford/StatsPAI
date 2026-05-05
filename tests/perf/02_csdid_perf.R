# Track C performance -- CS-DiD (R side).
# Times did::att_gt + did::aggte at N_units in {1000, 5000, 25000}.

.script_dir <- (function() {
  args <- commandArgs(trailingOnly = FALSE)
  m <- grep("^--file=", args, value = TRUE)
  if (length(m) > 0) dirname(normalizePath(sub("^--file=", "", m[1])))
  else getwd()
})()
RESULTS_DIR <- file.path(.script_dir, "results")
dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages({
  library(did)
  library(jsonlite)
})

N_UNITS_LIST <- c(1000L, 5000L, 25000L)
T <- 5L
N_REPS <- 3L

make_panel <- function(n_units, t = T, seed = 42L) {
  set.seed(seed)
  units <- rep(seq_len(n_units), each = t)
  years <- rep(seq_len(t), times = n_units)
  cohort <- integer(n_units)
  treated_idx <- sample.int(n_units, size = as.integer(0.4 * n_units))
  third <- length(treated_idx) %/% 3
  cohort[treated_idx[seq_len(third)]] <- 2L
  cohort[treated_idx[(third + 1L):(2L * third)]] <- 3L
  cohort[treated_idx[(2L * third + 1L):length(treated_idx)]] <- 4L
  first_treat <- cohort[units]
  treat <- as.integer((first_treat > 0L) & (years >= first_treat))
  unit_fe <- rnorm(n_units, 0, 0.5)
  year_fe <- rnorm(t, 0, 0.3)
  eps <- rnorm(n_units * t, 0, 0.4)
  y <- unit_fe[units] + year_fe[years] - 0.05 * treat + eps
  data.frame(y = y, unit = units, year = years,
             first_treat = first_treat, treat = treat)
}

time_one <- function(fn, n_reps, warmup = 1L) {
  for (i in seq_len(warmup)) fn()
  t <- numeric(n_reps)
  for (i in seq_len(n_reps)) {
    gc()
    t0 <- Sys.time(); fn()
    t[i] <- as.numeric(Sys.time() - t0, units = "secs")
  }
  list(median = median(t), iqr = IQR(t), min = min(t), max = max(t))
}

rows <- list()
for (n_units in N_UNITS_LIST) {
  df <- make_panel(n_units)
  fn <- function() {
    fit <- did::att_gt(
      yname = "y", tname = "year", idname = "unit",
      gname = "first_treat", data = df,
      control_group = "nevertreated",
      est_method = "reg", bstrap = FALSE
    )
    did::aggte(fit, type = "simple", bstrap = FALSE, cband = FALSE)
  }
  res <- time_one(fn, n_reps = N_REPS)
  rows[[length(rows) + 1L]] <- list(
    estimator = jsonlite::unbox("02_csdid"),
    side = jsonlite::unbox("R"),
    n = jsonlite::unbox(as.integer(n_units * T)),
    n_reps = jsonlite::unbox(N_REPS),
    median_time_s = jsonlite::unbox(res$median),
    iqr_time_s = jsonlite::unbox(res$iqr),
    min_time_s = jsonlite::unbox(res$min),
    max_time_s = jsonlite::unbox(res$max),
    peak_mem_mb = jsonlite::unbox(NA_real_),
    extra = list(n_units = jsonlite::unbox(n_units), T = jsonlite::unbox(T))
  )
  cat(sprintf("  N_units=%6d  N_obs=%7d  median=%.3fs\n",
              n_units, n_units * T, res$median))
}

payload <- list(
  estimator = jsonlite::unbox("02_csdid"),
  side = jsonlite::unbox("R"),
  rows = rows,
  hardware = list(R_version = jsonlite::unbox(R.version$version.string)),
  extra = list()
)
writeLines(
  jsonlite::toJSON(payload, pretty = TRUE, na = "null", null = "null", digits = NA),
  file.path(RESULTS_DIR, "02_csdid_R.json")
)
message("OK -- wrote 02_csdid_R.json")
