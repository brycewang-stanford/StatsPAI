# StatsPAI RD bandwidth selection parity (R side) -- Module 88.
#
# Reads data/88_rdbwselect.csv and runs rdrobust::rdbwselect across the
# same sweep the Python side runs: all ten selectors at p = 1, then
# polynomial order, kernel, covariates, clustering and the derivative
# order one cell at a time.
#
# Tolerance: rel < 1e-6 on every bandwidth.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(rdrobust)
})

MODULE <- "88_rdbwselect"

df <- read_csv_strict(MODULE)

SELECTORS <- c(
  "mserd", "msetwo", "msesum", "msecomb1", "msecomb2",
  "cerrd", "certwo", "cersum", "cercomb1", "cercomb2"
)

rows <- list()

# rdbwselect returns bws as a 1 x 4 matrix ordered h(left), h(right),
# b(left), b(right).
add_rows <- function(rows, prefix, fit) {
  bws <- fit$bws
  quantities <- list(
    h_left  = bws[1, 1],
    h_right = bws[1, 2],
    b_left  = bws[1, 3],
    b_right = bws[1, 4]
  )
  for (q in names(quantities)) {
    rows[[length(rows) + 1L]] <- parity_row(
      module    = MODULE,
      statistic = sprintf("%s_%s", prefix, q),
      estimate  = quantities[[q]],
      n         = nrow(df)
    )
  }
  rows
}

# --- Block A: all ten selectors at the p = 1 default. -----------------
for (m in SELECTORS) {
  fit <- rdrobust::rdbwselect(y = df$y, x = df$x, c = 0, bwselect = m)
  rows <- add_rows(rows, m, fit)
}

# --- Block B: polynomial order. ---------------------------------------
for (p in c(2, 3)) {
  fit <- rdrobust::rdbwselect(y = df$y, x = df$x, c = 0, p = p)
  rows <- add_rows(rows, sprintf("mserd_p%d", p), fit)
}

# --- Block C: kernel. --------------------------------------------------
for (kern in c("uniform", "epanechnikov")) {
  fit <- rdrobust::rdbwselect(y = df$y, x = df$x, c = 0, kernel = kern)
  rows <- add_rows(rows, sprintf("mserd_%s", kern), fit)
}

# --- Block D: covariate-adjusted selection. ---------------------------
fit <- rdrobust::rdbwselect(
  y = df$y, x = df$x, c = 0, covs = as.matrix(df[, "z1", drop = FALSE])
)
rows <- add_rows(rows, "mserd_covs", fit)

# --- Block E: clustered selection. ------------------------------------
fit <- rdrobust::rdbwselect(y = df$y, x = df$x, c = 0, cluster = df$cl)
rows <- add_rows(rows, "mserd_cluster", fit)

# --- Block F: derivative order (regression kink). ---------------------
fit <- rdrobust::rdbwselect(y = df$y, x = df$x, c = 0, deriv = 1, p = 2)
rows <- add_rows(rows, "mserd_deriv1", fit)

write_results(
  MODULE, rows,
  extra = list(
    selectors = SELECTORS,
    n_rows = length(rows),
    rdrobust_version = as.character(utils::packageVersion("rdrobust"))
  )
)
