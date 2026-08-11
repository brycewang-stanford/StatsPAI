# StatsPAI LP-DiD parity (R side) -- Module 83.
#
# No LP-DiD package is installed on this R side, so this is a DIRECT
# TRANSCRIPTION of the estimator sp.lp_did documents and implements,
# written independently from the Python code:
#
#   at horizon h, regress  dy = Y_{t+h} - Y_{t-1}  on  dd = d_t - d_{t-1}
#     * treated arm      : dd == 1
#     * clean control    : d == 0 across [t + min(-1, h-1), t + max(0, h)]
#                          and dd == 0
#     * calendar-time fixed effects
#     * cluster-robust SE by unit
#
# Reading the same CSV bytes as the Python side, the two must agree to
# floating-point noise: they are the same OLS on the same rows.
#
# Tolerance: rel < 1e-10.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(sandwich)
  library(lmtest)
})

MODULE <- "83_lpdid"
COMPARED_HORIZONS <- c(-2, 0, 1, 2, 3)

df <- read_csv_strict(MODULE)
df <- df[order(df$unit, df$time), ]

units <- unique(df$unit)
n_periods <- length(unique(df$time))

# Per-unit treatment path, indexed by period position.
paths <- split(df$d, df$unit)
ys <- split(df$y, df$unit)

lpdid_at_horizon <- function(h) {
  win_lo <- min(-1L, h - 1L)
  win_hi <- max(0L, h)

  recs <- list()
  for (u in units) {
    d_u <- paths[[as.character(u)]]
    y_u <- ys[[as.character(u)]]
    n <- length(d_u)
    for (k in seq_len(n)) {
      # k is the position of period t (1-based). Y_{t-1} is k-1, Y_{t+h} is k+h.
      base_idx <- k - 1L
      fut_idx <- k + h
      if (base_idx < 1L || fut_idx < 1L || fut_idx > n) next
      dy <- y_u[fut_idx] - y_u[base_idx]
      dd <- d_u[k] - d_u[base_idx]

      lo <- k + win_lo
      hi <- k + win_hi
      stable_zero <- FALSE
      if (lo >= 1L && hi <= n) {
        stable_zero <- all(d_u[lo:hi] == 0)
      }

      keep <- (dd == 1) || (stable_zero && dd == 0)
      if (!keep) next
      recs[[length(recs) + 1L]] <- data.frame(
        unit = u, time = k, dy = dy, dd = max(dd, 0), stringsAsFactors = FALSE
      )
    }
  }
  if (length(recs) == 0L) return(NULL)
  do.call(rbind, recs)
}

rows <- list()
for (h in COMPARED_HORIZONS) {
  s <- lpdid_at_horizon(h)
  if (is.null(s)) stop(sprintf("empty LP-DiD sample at horizon %d", h))

  fit <- lm(dy ~ dd + factor(time), data = s)
  # sp.lp_did applies the Stata-style cluster correction
  #     G/(G-1) * (N-1)/(N-K)
  # vcovCL(cadjust = TRUE) supplies only the first factor, so the second is
  # applied here explicitly. Without it the two sides differ by ~0.4% on this
  # fixture -- small enough to look like noise and wrong for a stated reason.
  vc <- sandwich::vcovCL(fit, cluster = s$unit, type = "HC0", cadjust = TRUE)
  n_obs <- nrow(s)
  k_par <- fit$rank
  vc <- vc * ((n_obs - 1) / (n_obs - k_par))
  ct <- lmtest::coeftest(fit, vcov. = vc)

  rows[[length(rows) + 1L]] <- parity_row(
    module    = MODULE,
    statistic = sprintf("lpdid_h%d_att", h),
    estimate  = unname(ct["dd", "Estimate"]),
    se        = unname(ct["dd", "Std. Error"]),
    n         = nrow(s)
  )
}

write_results(MODULE, rows,
              extra = list(reference_kind = "direct_transcription_no_r_package",
                           clean_controls = "not_yet_treated",
                           time_fe = TRUE,
                           cluster = "unit"))
