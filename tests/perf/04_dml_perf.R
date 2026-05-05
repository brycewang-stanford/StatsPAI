# Track C performance -- DML PLR (R side).
# Times DoubleML::DoubleMLPLR with mlr3::regr.lm at N in {1k, 5k, 10k}.

.script_dir <- (function() {
  args <- commandArgs(trailingOnly = FALSE)
  m <- grep("^--file=", args, value = TRUE)
  if (length(m) > 0) dirname(normalizePath(sub("^--file=", "", m[1])))
  else getwd()
})()
RESULTS_DIR <- file.path(.script_dir, "results")
dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages({
  library(DoubleML)
  library(mlr3)
  library(mlr3learners)
  library(data.table)
  library(jsonlite)
  lgr::get_logger("mlr3")$set_threshold("warn")
})

N_LIST <- c(1000L, 5000L, 10000L)
N_REPS <- 3L

make_data <- function(n, p = 5L, seed = 42L) {
  set.seed(seed)
  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  d <- as.integer((X[, 1] + 0.5 * X[, 2] + rnorm(n)) > 0)
  y <- 0.5 * d + rowSums(X) * 0.1 + rnorm(n)
  out <- as.data.table(X)
  setnames(out, paste0("x", seq_len(p)))
  out[, d := d]
  out[, y := y]
  out
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
for (n in N_LIST) {
  dt <- make_data(n)
  fn <- function() {
    dml_data <- DoubleML::DoubleMLData$new(
      data = dt, y_col = "y", d_cols = "d",
      x_cols = paste0("x", 1:5)
    )
    dml_obj <- DoubleML::DoubleMLPLR$new(
      data = dml_data,
      ml_l = mlr3::lrn("regr.lm"),
      ml_m = mlr3::lrn("regr.lm"),
      n_folds = 5L
    )
    dml_obj$fit()
  }
  res <- time_one(fn, n_reps = N_REPS)
  rows[[length(rows) + 1L]] <- list(
    estimator = jsonlite::unbox("04_dml"),
    side = jsonlite::unbox("R"),
    n = jsonlite::unbox(as.integer(n)),
    n_reps = jsonlite::unbox(N_REPS),
    median_time_s = jsonlite::unbox(res$median),
    iqr_time_s = jsonlite::unbox(res$iqr),
    min_time_s = jsonlite::unbox(res$min),
    max_time_s = jsonlite::unbox(res$max),
    peak_mem_mb = jsonlite::unbox(NA_real_),
    extra = list(model = jsonlite::unbox("plr"),
                 ml_g = jsonlite::unbox("regr.lm"),
                 n_folds = jsonlite::unbox(5L))
  )
  cat(sprintf("  N=%6d  median=%.3fs\n", n, res$median))
}

payload <- list(estimator = jsonlite::unbox("04_dml"),
                side = jsonlite::unbox("R"), rows = rows,
                hardware = list(R_version = jsonlite::unbox(R.version$version.string)),
                extra = list())
writeLines(
  jsonlite::toJSON(payload, pretty = TRUE, na = "null", null = "null", digits = NA),
  file.path(RESULTS_DIR, "04_dml_R.json")
)
message("OK -- wrote 04_dml_R.json")
