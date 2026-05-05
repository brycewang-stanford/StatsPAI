# StatsPAI RD density parity (R side) -- Module 09.
#
# Reads data/09_rddensity.csv (the StatsPAI Lee 2008 replica) and
# runs rddensity::rddensity with package defaults. Tolerance:
# rel < 1e-3 (iterative bandwidth selection).

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(rddensity)
})

MODULE <- "09_rddensity"

df <- read_csv_strict(MODULE)

fit <- rddensity::rddensity(X = df$margin, c = 0.0)

# Density estimates on each side of the cutoff at the cutoff point;
# rddensity stores them in fit$hat$left / fit$hat$right.
density_left  <- fit$hat$left
density_right <- fit$hat$right
density_diff  <- density_right - density_left

# Test statistics: $test contains conventional and robust z and p.
pvalue <- fit$test$p_jk      # robust bias-corrected p-value
zstat  <- fit$test$t_jk

# Bandwidths: fit$h$left and fit$h$right
bw_l <- fit$h$left
bw_r <- fit$h$right

rows <- list(
  parity_row(MODULE, "density_diff",   estimate = density_diff,  n = nrow(df)),
  parity_row(MODULE, "density_left",   estimate = density_left,  n = nrow(df)),
  parity_row(MODULE, "density_right",  estimate = density_right, n = nrow(df)),
  parity_row(MODULE, "test_pvalue",    estimate = pvalue,        n = nrow(df)),
  parity_row(MODULE, "bandwidth_left", estimate = bw_l,          n = nrow(df)),
  parity_row(MODULE, "bandwidth_right",estimate = bw_r,          n = nrow(df))
)

write_results(MODULE, rows,
              extra = list(polynomial_order = fit$opt$p,
                           test_kind = "Cattaneo-Jansson-Ma (2020)"))
