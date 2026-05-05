# Track C performance benchmark -- HDFE 2-way FE (R side).
# Times fixest::feols at N in {1e4, 1e5, 1e6} on the same deterministic
# panel as 01_hdfe_perf.py (regenerated on the R side; identical
# seed produces identical bytes).

.script_dir <- (function() {
  args <- commandArgs(trailingOnly = FALSE)
  m <- grep("^--file=", args, value = TRUE)
  if (length(m) > 0) dirname(normalizePath(sub("^--file=", "", m[1])))
  else getwd()
})()
RESULTS_DIR <- file.path(.script_dir, "results")
dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages({
  library(fixest)
  library(jsonlite)
})

N_LIST <- c(1e4, 1e5, 1e6)
N_REPS <- 5L

make_panel <- function(n, n_firms, n_years = 20L, seed = 42L) {
  set.seed(seed)
  firm <- sample.int(n_firms, n, replace = TRUE)
  year <- sample.int(n_years, n, replace = TRUE)
  firm_fe <- rnorm(n_firms, 0, 1.0)
  year_fe <- rnorm(n_years, 0, 0.5)
  x1 <- rnorm(n)
  x2 <- rnorm(n)
  y <- 2.0 * x1 - 1.5 * x2 + firm_fe[firm] + year_fe[year] + rnorm(n, sd = 0.5)
  data.frame(y = y, x1 = x1, x2 = x2, firm = factor(firm), year = factor(year))
}

time_one <- function(fn, n_reps = 5L, warmup = 1L) {
  for (i in seq_len(warmup)) fn()
  t <- numeric(n_reps)
  for (i in seq_len(n_reps)) {
    gc()
    t0 <- Sys.time()
    fn()
    t[i] <- as.numeric(Sys.time() - t0, units = "secs")
  }
  list(median = median(t),
       iqr = IQR(t),
       min = min(t),
       max = max(t))
}

rows <- list()
for (n in as.integer(N_LIST)) {
  n_firms <- max(50L, as.integer(sqrt(n) * 2))
  df <- make_panel(n, n_firms)
  fn <- function() {
    fixest::feols(y ~ x1 + x2 | firm + year, data = df, vcov = "iid")
  }
  res <- time_one(fn, n_reps = N_REPS, warmup = 1L)
  rows[[length(rows) + 1L]] <- list(
    estimator = jsonlite::unbox("01_hdfe"),
    side = jsonlite::unbox("R"),
    n = jsonlite::unbox(n),
    n_reps = jsonlite::unbox(N_REPS),
    median_time_s = jsonlite::unbox(res$median),
    iqr_time_s = jsonlite::unbox(res$iqr),
    min_time_s = jsonlite::unbox(res$min),
    max_time_s = jsonlite::unbox(res$max),
    peak_mem_mb = jsonlite::unbox(NA_real_),
    extra = list(n_firms = jsonlite::unbox(n_firms),
                 package = jsonlite::unbox("fixest"))
  )
  cat(sprintf("  N=%9d  n_firms=%5d  median=%.3fs  iqr=%.3fs\n",
              n, n_firms, res$median, res$iqr))
}

payload <- list(
  estimator = jsonlite::unbox("01_hdfe"),
  side = jsonlite::unbox("R"),
  rows = rows,
  hardware = list(platform = jsonlite::unbox(R.version$os),
                  R_version = jsonlite::unbox(R.version$version.string)),
  extra = list()
)
writeLines(
  jsonlite::toJSON(payload, pretty = TRUE, na = "null", null = "null", digits = NA),
  file.path(RESULTS_DIR, "01_hdfe_R.json")
)
message("OK -- wrote 01_hdfe_R.json")
