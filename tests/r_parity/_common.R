# Shared helpers for the R side of the StatsPAI Track A parity harness.
#
# Each NN_<name>.R script:
#   1. Reads data/<name>.csv (written by NN_<name>.py).
#   2. Runs the canonical R reference implementation.
#   3. Writes results/<name>_R.json via write_results().
#
# Tolerance budget is documented in _common.py; R-side scripts emit
# raw numbers and let compare.py make the pass/fail decision so the
# tolerance lives in exactly one place.

PARITY_SEED <- 42L

# Locate this file's directory so HERE works whether the script is
# invoked from the repo root or from replication/parity/.
.this_file <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  m <- grep("--file=", args)
  if (length(m) > 0) {
    return(normalizePath(sub("--file=", "", args[m[1]])))
  }
  # Fallback: if sourced interactively.
  if (!is.null(sys.frame(1)$ofile)) {
    return(normalizePath(sys.frame(1)$ofile))
  }
  normalizePath("_common.R")
}

HERE <- dirname(.this_file())
DATA_DIR    <- file.path(HERE, "data")
# RESULTS_DIR is overridable via env var so a reproducibility-verification
# run can write to a staging directory and diff against the committed
# golden JSON without clobbering it. Defaults to the committed results/.
RESULTS_DIR <- Sys.getenv("STATSPAI_R_PARITY_RESULTS_DIR",
                          unset = file.path(HERE, "results"))
dir.create(DATA_DIR,    showWarnings = FALSE, recursive = TRUE)
dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)


read_csv_strict <- function(name) {
  # Use read.csv with default options and turn factors off; the CSV
  # was written by Pandas with float_format="%.16g".
  utils::read.csv(file.path(DATA_DIR, paste0(name, ".csv")),
                  stringsAsFactors = FALSE, check.names = FALSE)
}


parity_row <- function(module, statistic, estimate,
                       se = NA, ci_lo = NA, ci_hi = NA, n = NA,
                       extra = list()) {
  # JSON-safe: convert NA / Inf / NaN to NULL via .json_safe (jsonlite
  # serialises NA as null when na = "null"). Use unbox() on every
  # scalar so the JSON shape matches the Python side: scalars are
  # emitted as bare values, not as 1-element arrays.
  list(
    module    = jsonlite::unbox(module),
    side      = jsonlite::unbox("R"),
    statistic = jsonlite::unbox(statistic),
    estimate  = jsonlite::unbox(.json_safe(estimate)),
    se        = jsonlite::unbox(.json_safe(se)),
    ci_lo     = jsonlite::unbox(.json_safe(ci_lo)),
    ci_hi     = jsonlite::unbox(.json_safe(ci_hi)),
    n         = jsonlite::unbox(.json_safe(n)),
    extra     = if (length(extra) == 0) list() else extra
  )
}


.json_safe <- function(x) {
  if (length(x) == 0) return(NA)
  x <- x[1]
  if (is.na(x) || is.nan(x) || is.infinite(x)) return(NA)
  x
}


# Capture the exact R software environment that produced these numbers,
# so the committed _R.json is self-describing for JSS reproducibility
# (Section 8, "Reproducibility of this paper"). We record the R version,
# the platform triple, and the version of every attached/loaded
# non-base package -- this is the package set that the reference
# implementation in the calling NN_<name>.R script actually used.
# A reviewer can diff this block against their own sessionInfo() and
# explain any residual numerical gap by a package-version delta rather
# than guessing. The full machine-readable manifest also lives in
# tests/r_parity/R_ENVIRONMENT.md and the renv-style renv.lock.
.r_provenance <- function() {
  si <- utils::sessionInfo()
  pkgs <- list()
  add_pkgs <- function(src) {
    if (is.null(src)) return(invisible())
    for (p in names(src)) {
      v <- tryCatch(as.character(src[[p]]$Version), error = function(e) NA)
      if (!is.na(v) && is.null(pkgs[[p]])) pkgs[[p]] <<- jsonlite::unbox(v)
    }
  }
  add_pkgs(si$otherPkgs)   # library()-attached reference packages
  add_pkgs(si$loadedOnly)  # pkg::fn namespace loads (e.g. via ::)
  list(
    r_version    = jsonlite::unbox(R.version.string),
    platform     = jsonlite::unbox(R.version$platform),
    running       = jsonlite::unbox(si$running %||% NA),
    captured_via = jsonlite::unbox("tests/r_parity/_common.R::write_results"),
    packages     = if (length(pkgs) == 0) list() else pkgs
  )
}

# Null-coalescing helper (base R has no %||% before 4.4 in all builds).
`%||%` <- function(a, b) if (is.null(a) || length(a) == 0) b else a


write_results <- function(module, rows, extra = list()) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    install.packages("jsonlite", repos = "https://cloud.r-project.org")
  }
  out <- file.path(RESULTS_DIR, paste0(module, "_R.json"))
  payload <- list(
    module     = jsonlite::unbox(module),
    side       = jsonlite::unbox("R"),
    rows       = rows,
    extra      = if (length(extra) == 0) list() else extra,
    provenance = .r_provenance()
  )
  # digits = NA emits full IEEE-754 precision (15-17 significant
  # digits) so the parity comparator sees the actual numerical
  # output rather than the jsonlite default 4-digit rounding.
  txt <- jsonlite::toJSON(
    payload, pretty = TRUE, na = "null", null = "null", digits = NA
  )
  writeLines(txt, out)
  message(sprintf("OK -- wrote %s (%d rows)", out, length(rows)))
}
