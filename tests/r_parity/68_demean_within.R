# StatsPAI demean parity — Module 68 (R side).
#
# Reads data/68_demean_within.csv and computes the within-transformation
# for y and the three x columns using the textbook entity-mean subtraction
# (sum/group + broadcast), which is the same projection sp.demean(solver='map')
# produces to machine precision. Emits the same row statistics the Python
# side writes so compare.py joins on (module, statistic).

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

MODULE <- "68_demean_within"

df <- read_csv_strict(MODULE)
n <- nrow(df)

# Entity-mean projection: subtract per-id mean of the column.
mean_within <- function(v) {
  out <- v
  for (id in unique(df$id)) {
    m <- df$id == id
    out[m] <- out[m] - mean(out[m])
  }
  out
}

dem_y <- mean_within(df$y)
dem_x1 <- mean_within(df$x1)
dem_x2 <- mean_within(df$x2)
dem_x3 <- mean_within(df$x3)

rows <- list(
  parity_row(MODULE, "demean_y",
             estimate = unname(dem_y[1]), n = n)
)
# Sample the same three observations the Python side does. Python uses
# 0-based k in (0, n %/% 2, n - 1) and names the row after k; R is 1-based,
# so the middle point is n %/% 2 + 1, not n %/% 2. Before this correction R
# sampled row 80 (1-based) and named it "row79" while Python sampled 0-based
# index 80 and named it "row80": the two names never joined, so compare.py
# silently graded 7 of this module's 10 statistics. The Stata bridge added
# on 2026-08-06 is what surfaced it.
for (k in c(1, n %/% 2 + 1, n)) {
  rows <- c(rows, list(
    parity_row(MODULE, sprintf("demean_x1_row%d", k - 1),
               estimate = unname(dem_x1[k]), n = n),
    parity_row(MODULE, sprintf("demean_x2_row%d", k - 1),
               estimate = unname(dem_x2[k]), n = n),
    parity_row(MODULE, sprintf("demean_x3_row%d", k - 1),
               estimate = unname(dem_x3[k]), n = n)
  ))
}

write_results(MODULE, rows, extra = list(
  transform = "within (entity-mean projection)",
  note = "sp.demean(solver='map') vs textbook mean-within; machine tier"
))