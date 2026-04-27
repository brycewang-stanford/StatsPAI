#!/usr/bin/env Rscript
# Run R `fixest::fepois` on one HDFE Poisson baseline dataset and emit JSON
# to stdout. Stderr carries diagnostics. Designed to be invoked as a
# subprocess by `run_baseline.py`.
#
# Usage:
#   Rscript run_r.R --dataset small [--repeats 3] [--warmup 1]

suppressPackageStartupMessages({
  for (pkg in c("data.table", "fixest", "jsonlite")) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      stop(sprintf("required R package '%s' is not installed", pkg))
    }
  }
})

library(data.table)
library(fixest)
library(jsonlite)

# --- arg parsing (small, hand-rolled to avoid optparse dep) -----------------
parse_args <- function(argv) {
  ds <- NULL; repeats <- 3L; warmup <- 1L
  i <- 1
  while (i <= length(argv)) {
    a <- argv[i]
    if (a == "--dataset")      { ds      <- argv[i + 1]; i <- i + 2 }
    else if (a == "--repeats") { repeats <- as.integer(argv[i + 1]); i <- i + 2 }
    else if (a == "--warmup")  { warmup  <- as.integer(argv[i + 1]); i <- i + 2 }
    else stop(sprintf("unknown arg: %s", a))
  }
  if (is.null(ds))                                  stop("--dataset is required")
  if (!ds %in% c("small", "medium", "large"))      stop("dataset must be small|medium|large")
  list(dataset = ds, repeats = repeats, warmup = warmup)
}

args <- parse_args(commandArgs(trailingOnly = TRUE))

# Resolve script directory robustly: parse the --file= entry that Rscript
# always passes. Fall back to CWD if invoked via `R -e source(...)`.
.script_dir <- function() {
  full <- commandArgs(trailingOnly = FALSE)
  hit  <- grep("^--file=", full, value = TRUE)
  if (length(hit) >= 1) {
    return(normalizePath(dirname(sub("^--file=", "", hit[1])), mustWork = FALSE))
  }
  getwd()
}
here <- .script_dir()

csv_path <- file.path(here, "data", paste0(args$dataset, ".csv.gz"))
if (!file.exists(csv_path)) {
  stop(sprintf("dataset file not found: %s — run datasets.py first", csv_path))
}

message(sprintf("[run_r] loading %s …", csv_path))
t_load_0 <- proc.time()[["elapsed"]]
df <- data.table::fread(
  csv_path,
  colClasses = c(y = "integer", x1 = "numeric", x2 = "numeric",
                 fe1 = "integer", fe2 = "integer"),
  showProgress = FALSE
)
t_load <- proc.time()[["elapsed"]] - t_load_0
message(sprintf("[run_r] loaded n=%d rows in %.2fs", nrow(df), t_load))

# Memory-peak proxy on Unix: gc() returns max used since session start.
mem_before <- gc(reset = TRUE, full = TRUE)

run_one <- function() {
  fixest::fepois(
    fml  = y ~ x1 + x2 | fe1 + fe2,
    data = df,
    fixef.rm = "perfect",
    glm.iter = 25,
    glm.tol  = 1e-8
  )
}

# Warmup
for (i in seq_len(args$warmup)) {
  invisible(run_one())
}

runs <- numeric(args$repeats)
fit <- NULL
for (i in seq_len(args$repeats)) {
  t0 <- proc.time()[["elapsed"]]
  fit <- run_one()
  runs[i] <- proc.time()[["elapsed"]] - t0
}

mem_after <- gc()  # base R splits "max used (Mb)" into two columns; "max
                   # used" count is col 6, its Mb value is col 7. Sum
                   # across the two rows (Ncells + Vcells) for total peak.
peak_mb <- if (ncol(mem_after) >= 7) sum(mem_after[, 7]) else NA_real_

co <- coef(fit)
se <- tryCatch(se(fit), error = function(e) setNames(rep(NA_real_, length(co)), names(co)))

out <- list(
  dataset      = args$dataset,
  backend      = "fixest",
  warmup       = args$warmup,
  repeats      = args$repeats,
  n_rows       = nrow(df),
  load_seconds = round(t_load, 4),
  peak_rss_mb  = round(peak_mb, 1),
  wall = list(
    wall_runs = round(runs, 6),
    wall_min  = min(runs),
    wall_mean = mean(runs),
    wall_max  = max(runs)
  ),
  coefs = list(
    x1 = unname(co["x1"]),
    x2 = unname(co["x2"])
  ),
  se = list(
    x1 = unname(se["x1"]),
    x2 = unname(se["x2"])
  ),
  iterations = tryCatch(fit$iterations, error = function(e) NA_integer_)
)

cat(jsonlite::toJSON(out, auto_unbox = TRUE, digits = 12, na = "null"), "\n")
