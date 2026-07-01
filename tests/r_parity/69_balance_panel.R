# StatsPAI balance_panel parity — Module 69 (R side).
#
# Reads data/69_balance_panel.csv, applies the same "keep only entities
# observed in every period" filter as sp.balance_panel using base R
# counts, sorts by (id, year), and emits the same summary statistics the
# Python side writes so compare.py joins on (module, statistic).

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

MODULE <- "69_balance_panel"

df <- read_csv_strict(MODULE)

n_periods <- length(unique(df$year))
counts <- ave(df$year, df$id, FUN = function(x) length(unique(x)))
balanced <- df[counts == n_periods, ]
balanced <- balanced[order(balanced$id, balanced$year), ]
rownames(balanced) <- NULL

n <- nrow(balanced)
rows <- list(
  parity_row(MODULE, "n_obs_balanced",
             estimate = unname(n), n = nrow(df)),
  parity_row(MODULE, "n_units_kept",
             estimate = unname(length(unique(balanced$id))), n = nrow(df))
)
for (k in c(1, n %/% 2 + 1, n)) {
  rows <- c(rows, list(
    parity_row(MODULE, sprintf("row%d_id", k - 1),
               estimate = unname(balanced$id[k]), n = nrow(df)),
    parity_row(MODULE, sprintf("row%d_year", k - 1),
               estimate = unname(balanced$year[k]), n = nrow(df))
  ))
}

write_results(MODULE, rows, extra = list(
  filter = "keep only entities observed in every period",
  note = "sp.balance_panel vs base R counts"
))